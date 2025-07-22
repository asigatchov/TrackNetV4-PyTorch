"""
TrackNet Training Script

Usage Examples:
python train.py --data dataset/train
python train.py --data dataset/train --batch 8 --epochs 50 --lr 0.001
python train.py --data dataset/train --optimizer Lion --lr 0.0003 --batch 16
python train.py --resume best.pth --data dataset/train --lr 0.0001
python train.py --resume checkpoint.pth --data dataset/train --optimizer Lion --epochs 100
python train.py --data training_data/train --batch 3 --lr 0.0001 --resume best_model.pth --optimizer Lion


Parameters:
--data: Training dataset path (required)
--resume: Checkpoint path for resuming
--split: Train/val split ratio (default: 0.8)
--seed: Random seed (default: 26)
--batch: Batch size (default: 3)
--epochs: Training epochs (default: 30)
--workers: Data loader workers (default: 0)
--device: Device auto/cpu/cuda/mps (default: auto)
--optimizer: Adadelta/Adam/AdamW/Lion/SGD (default: AdamW)
--lr: Learning rate (default: auto per optimizer)
--wd: Weight decay (default: 0)
--scheduler: ReduceLROnPlateau/None (default: ReduceLROnPlateau)
--factor: LR reduction factor (default: 0.5)
--patience: LR reduction patience (default: 3)
--min_lr: Minimum learning rate (default: 1e-6)
--plot: Loss plot interval (default: 10)
--out: Output directory (default: outputs)
--name: Experiment name (default: exp)
"""

import argparse
import json
import signal
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from lion_pytorch import Lion
from model.loss import WeightedBinaryCrossEntropy
from model.tracknet import TrackNet
from preprocessing.tracknet_dataset import FrameHeatmapDataset


def parse_args():
    parser = argparse.ArgumentParser(description="TrackNet Training")
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--split', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=26)
    parser.add_argument('--batch', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['Adadelta', 'Adam', 'AdamW', 'Lion', 'SGD'])
    parser.add_argument('--lr', type=float)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--scheduler', type=str, default='ReduceLROnPlateau',
                        choices=['ReduceLROnPlateau', 'None'])
    parser.add_argument('--factor', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--plot', type=int, default=10)
    parser.add_argument('--out', type=str, default='outputs')
    parser.add_argument('--name', type=str, default='exp')

    args = parser.parse_args()

    if args.lr is None:
        lr_defaults = {'Adadelta': 1.0, 'Adam': 0.001, 'AdamW': 0.001, 'Lion': 0.0003, 'SGD': 0.01}
        args.lr = lr_defaults[args.optimizer]

    return args


class Trainer:
    def __init__(self, args):
        self.args = args
        self.start_epoch = 0
        self.interrupted = False
        self.best_loss = float('inf')
        self.device = self._get_device()
        self._setup_dirs()
        self._load_checkpoint()
        self.losses = {'batch': [], 'steps': [], 'lrs': [], 'train': [], 'val': []}
        self.step = 0
        signal.signal(signal.SIGINT, self._interrupt)
        signal.signal(signal.SIGTERM, self._interrupt)

    def _get_device(self):
        if self.args.device == 'auto':
            if torch.backends.mps.is_available():
                return torch.device('mps')
            elif torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        return torch.device(self.args.device)

    def _setup_dirs(self):
        print("Setting up output directories...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_resumed" if self.args.resume else ""
        self.save_dir = Path(self.args.out) / f"{self.args.name}{suffix}_{timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "checkpoints").mkdir(exist_ok=True)
        (self.save_dir / "plots").mkdir(exist_ok=True)
        with open(self.save_dir / "config.json", 'w') as f:
            json.dump(vars(self.args), f, indent=2)
        print(f"Output directory created: {self.save_dir}")

    def _load_checkpoint(self):
        if not self.args.resume: return
        print("Loading checkpoint...")
        path = Path(self.args.resume)
        if not path.exists(): raise FileNotFoundError(f"Checkpoint not found: {path}")
        self.checkpoint = torch.load(path, map_location='cpu')
        self.start_epoch = self.checkpoint['epoch'] + (0 if self.checkpoint.get('is_emergency', False) else 1)
        print(f"Checkpoint loaded, resuming from epoch {self.start_epoch + 1}")

    def _interrupt(self, signum, frame):
        print("\nInterrupt detected, saving emergency checkpoint...")
        self.interrupted = True

    def setup_data(self):
        print("Loading dataset...")
        dataset = FrameHeatmapDataset(self.args.data)
        print(f"Dataset loaded: {len(dataset)} samples")

        print("Splitting dataset...")
        torch.manual_seed(self.args.seed)
        train_size = int(self.args.split * len(dataset))
        train_ds, val_ds = random_split(dataset, [train_size, len(dataset) - train_size])

        print("Creating data loaders...")
        self.train_loader = DataLoader(train_ds, batch_size=self.args.batch, shuffle=True,
                                       num_workers=self.args.workers, pin_memory=self.device.type == 'cuda')
        self.val_loader = DataLoader(val_ds, batch_size=self.args.batch, shuffle=False,
                                     num_workers=self.args.workers, pin_memory=self.device.type == 'cuda')
        print(f"Data loaders ready - Train: {len(train_ds)} | Val: {len(val_ds)}")

    def _create_optimizer(self):
        optimizers = {
            'Adadelta': lambda: torch.optim.Adadelta(self.model.parameters(), lr=self.args.lr,
                                                     weight_decay=self.args.wd),
            'Adam': lambda: torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd),
            'AdamW': lambda: torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd),
            'Lion': lambda: Lion(self.model.parameters(), lr=self.args.lr, weight_decay=max(self.args.wd, 0.01)),
            'SGD': lambda: torch.optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd,
                                           momentum=0.9)
        }
        return optimizers[self.args.optimizer]()

    def setup_model(self):
        print("Initializing model...")
        self.model = TrackNet().to(self.device)
        self.criterion = WeightedBinaryCrossEntropy()
        self.optimizer = self._create_optimizer()

        if self.args.scheduler == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=self.args.factor,
                                               patience=self.args.patience, min_lr=self.args.min_lr)
        else:
            self.scheduler = None

        if hasattr(self, 'checkpoint'):
            print("Loading model state from checkpoint...")
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            print("Model state loaded successfully")

        print(f"Model ready - Optimizer: {self.args.optimizer} | LR: {self.args.lr} | WD: {self.args.wd}")

    def save_checkpoint(self, epoch, train_loss, val_loss, is_emergency=False):
        print("Saving checkpoint...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'is_emergency': is_emergency,
            'history': self.losses.copy(),
            'step': self.step,
            'timestamp': timestamp
        }

        prefix = "emergency_" if is_emergency else "checkpoint_"
        filename = f"{prefix}epoch_{epoch + 1}_{timestamp}.pth"
        filepath = self.save_dir / "checkpoints" / filename
        torch.save(checkpoint, filepath)

        if not is_emergency and val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(checkpoint, self.save_dir / "checkpoints" / "best_model.pth")
            print(f"Checkpoint saved: {filename} (Best model updated)")
            return filepath, True

        print(f"Checkpoint saved: {filename}")
        return filepath, False

    def plot_curves(self, epoch):
        print("Generating training plots...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        if self.losses['batch']:
            ax1.plot(self.losses['steps'], self.losses['batch'], 'b-', alpha=0.3, label='Batch Loss')
        if self.losses['train']:
            epochs = list(range(1, len(self.losses['train']) + 1))
            ax1.plot(epochs, self.losses['train'], 'bo-', label='Train')
            ax1.plot(epochs, self.losses['val'], 'ro-', label='Val')

        ax1.set_xlabel('Batch/Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        if self.losses['lrs']:
            ax2.plot(self.losses['steps'], self.losses['lrs'], 'g-')
            ax2.set_xlabel('Batch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / "plots" / f"epoch_{epoch + 1}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training plots saved for epoch {epoch + 1}")

    def validate(self):
        print("Starting validation...")
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(total=len(self.val_loader), desc="Validation", ncols=100)
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                if self.interrupted:
                    val_pbar.close()
                    break
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                val_pbar.update(1)
                val_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            val_pbar.close()

        avg_loss = total_loss / len(self.val_loader)
        print(f"Validation completed - Average loss: {avg_loss:.6f}")
        return avg_loss

    def train(self):
        print(f"Starting training on {self.device}")
        self.setup_data()
        self.setup_model()

        for epoch in range(self.start_epoch, self.args.epochs):
            if self.interrupted: break

            print(f"\nEpoch {epoch + 1}/{self.args.epochs}")
            start_time = time.time()
            self.model.train()
            total_loss = 0.0

            train_pbar = tqdm(total=len(self.train_loader), desc=f"Training", ncols=100)
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                if self.interrupted:
                    train_pbar.close()
                    print("Emergency save triggered...")
                    val_loss = self.validate()
                    self.save_checkpoint(epoch, total_loss / (batch_idx + 1), val_loss, True)
                    self.plot_curves(epoch)
                    return

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                batch_loss = loss.item()
                total_loss += batch_loss
                self.step += 1

                if self.step % self.args.plot == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.losses['batch'].append(batch_loss)
                    self.losses['steps'].append(self.step)
                    self.losses['lrs'].append(current_lr)

                train_pbar.update(1)
                train_pbar.set_postfix({'loss': f'{batch_loss:.6f}'})
            train_pbar.close()

            train_loss = total_loss / len(self.train_loader)
            val_loss = self.validate()

            self.losses['train'].append(train_loss)
            self.losses['val'].append(val_loss)

            current_lr = self.optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time

            print(f"Epoch [{epoch + 1}/{self.args.epochs}] Train: {train_loss:.6f} Val: {val_loss:.6f} "
                  f"LR: {current_lr:.6e} Time: {elapsed:.1f}s")

            if self.scheduler:
                print("Updating learning rate scheduler...")
                self.scheduler.step(val_loss)

            _, is_best = self.save_checkpoint(epoch, train_loss, val_loss)
            if is_best:
                print(f"New best model! Val Loss: {val_loss:.6f}")

            self.plot_curves(epoch)

        if not self.interrupted:
            print("\nTraining completed successfully!")
            print(f"All results saved to: {self.save_dir}")


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
