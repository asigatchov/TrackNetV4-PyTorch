"""
TrackNet Training Script - Streamlined Version

Usage Examples:
1. Basic training:
   python train.py --data dataset/Professional_reorg_train

2. Custom training:
   python train.py --data dataset/train --batch 8 --epochs 50 --lr 2.0 --device cuda

3. Advanced training:
   python train.py --data dataset/train --batch 4 --epochs 100 --lr 1.5 --split 0.9 --out outputs --name advanced_exp --plot 5 --patience 5

4. Resume training:
   python train.py --resume checkpoints/best_model.pth --data dataset/train

5. Resume with modified settings:
   python train.py --resume checkpoints/checkpoint_epoch_20.pth --data dataset/train --epochs 100 --device cuda

Parameter Functions:
- data: Path to training dataset directory (required)
- resume: Path to checkpoint file for resuming training (optional)
- split: Proportion of data for training (default: 0.8)
- seed: Random seed for data splitting (default: 26)
- batch: Number of samples per training batch (default: 3)
- epochs: Total training epochs (default: 30)
- workers: Number of data loading workers (default: 0)
- device: Computing device - auto/cpu/cuda/mps (default: auto)
- optimizer: Optimizer type - Adadelta/Adam/SGD (default: Adadelta)
- lr: Initial learning rate for optimizer (default: 1.0)
- wd: Weight decay for optimizer (default: 0)
- scheduler: Learning rate scheduler type (default: ReduceLROnPlateau)
- factor: Factor by which LR is reduced (default: 0.5)
- patience: Epochs to wait before reducing LR (default: 3)
- min_lr: Minimum learning rate (default: 1e-6)
- plot: Interval for recording batch losses (default: 10)
- out: Directory to save training outputs (default: training_outputs)
- name: Name prefix for experiment files (default: tracknet_experiment)
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

from model.loss import WeightedBinaryCrossEntropy
from model.tracknet import TrackNet
from preprocessing.tracknet_dataset import FrameHeatmapDataset


def parse_args():
    parser = argparse.ArgumentParser(description="TrackNet Training Script - Streamlined")

    parser.add_argument('--data', type=str, required=True, help='Path to training dataset directory')
    parser.add_argument('--resume', type=str, help='Path to checkpoint file for resuming training')
    parser.add_argument('--split', type=float, default=0.8, help='Ratio of training data (default: 0.8)')
    parser.add_argument('--seed', type=int, default=26, help='Random seed for data splitting (default: 26)')
    parser.add_argument('--batch', type=int, default=3, help='Batch size for training (default: 3)')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs (default: 30)')
    parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers (default: 0)')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto/cpu/cuda/mps (default: auto)')
    parser.add_argument('--optimizer', type=str, default='Adadelta', choices=['Adadelta', 'Adam', 'SGD'],
                        help='Optimizer type (default: Adadelta)')
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate (default: 1.0)')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay (default: 0)')
    parser.add_argument('--scheduler', type=str, default='ReduceLROnPlateau', choices=['ReduceLROnPlateau', 'None'],
                        help='LR scheduler (default: ReduceLROnPlateau)')
    parser.add_argument('--factor', type=float, default=0.5, help='LR reduction factor (default: 0.5)')
    parser.add_argument('--patience', type=int, default=3, help='LR reduction patience (default: 3)')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate (default: 1e-6)')
    parser.add_argument('--plot', type=int, default=10, help='Batch loss recording interval (default: 10)')
    parser.add_argument('--out', type=str, default='training_outputs',
                        help='Output directory (default: training_outputs)')
    parser.add_argument('--name', type=str, default='tracknet_experiment',
                        help='Experiment name (default: tracknet_experiment)')

    return parser.parse_args()


class Trainer:
    def __init__(self, args):
        self.args = args
        self.start_epoch = 0
        self.interrupted = False
        self.best_loss = float('inf')

        self.device = self._setup_device()
        self._setup_dirs()
        self._load_checkpoint()

        self.batch_losses = []
        self.batch_steps = []
        self.batch_lrs = []
        self.epoch_train_losses = []
        self.epoch_val_losses = []
        self.step = 0

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_device(self):
        if self.args.device == 'auto':
            if torch.backends.mps.is_available():
                return torch.device('mps')
            elif torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        return torch.device(self.args.device)

    def _setup_dirs(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.args.resume:
            name = f"{self.args.name}_resumed_{timestamp}"
        else:
            name = f"{self.args.name}_{timestamp}"

        self.save_dir = Path(self.args.out) / name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "checkpoints").mkdir(exist_ok=True)
        (self.save_dir / "plots").mkdir(exist_ok=True)

        with open(self.save_dir / "config.json", 'w') as f:
            json.dump(vars(self.args), f, indent=2)

    def _load_checkpoint(self):
        if not self.args.resume:
            return

        checkpoint_path = Path(self.args.resume)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint: {checkpoint_path}")
        self.checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if self.checkpoint.get('is_emergency', False):
            self.start_epoch = self.checkpoint['epoch']
            print(f"Resuming from incomplete epoch {self.start_epoch + 1}")
        else:
            self.start_epoch = self.checkpoint['epoch'] + 1
            print(f"Resuming from epoch {self.start_epoch + 1}")

    def _signal_handler(self, signum, frame):
        print("\nInterrupt detected, saving...")
        self.interrupted = True

    def setup_data(self):
        dataset = FrameHeatmapDataset(self.args.data)
        torch.manual_seed(self.args.seed)

        train_size = int(self.args.split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch, shuffle=True,
                                       num_workers=self.args.workers, pin_memory=self.device.type == 'cuda')
        self.val_loader = DataLoader(val_dataset, batch_size=self.args.batch, shuffle=False,
                                     num_workers=self.args.workers, pin_memory=self.device.type == 'cuda')

        print(f"Dataset: {len(dataset)} | Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    def setup_model(self):
        self.model = TrackNet().to(self.device)
        self.criterion = WeightedBinaryCrossEntropy()

        if self.args.optimizer == "Adadelta":
            self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        elif self.args.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd,
                                             momentum=0.9)

        if self.args.scheduler == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=self.args.factor,
                                               patience=self.args.patience, min_lr=self.args.min_lr)
        else:
            self.scheduler = None

        if hasattr(self, 'checkpoint'):
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            if self.scheduler and self.checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(self.checkpoint['scheduler_state_dict'])
            print("Model state loaded from checkpoint")

    def save_checkpoint(self, epoch, train_loss, val_loss, is_emergency=False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'is_emergency': is_emergency,
            'history': {
                'batch_losses': self.batch_losses,
                'batch_steps': self.batch_steps,
                'batch_lrs': self.batch_lrs,
                'epoch_train_losses': self.epoch_train_losses,
                'epoch_val_losses': self.epoch_val_losses,
                'step': self.step
            },
            'timestamp': timestamp
        }

        if is_emergency:
            filename = f"emergency_epoch_{epoch + 1}_{timestamp}.pth"
        else:
            filename = f"checkpoint_epoch_{epoch + 1}_{timestamp}.pth"

        filepath = self.save_dir / "checkpoints" / filename
        torch.save(checkpoint, filepath)

        if not is_emergency and val_loss < self.best_loss:
            self.best_loss = val_loss
            best_path = self.save_dir / "checkpoints" / "best_model.pth"
            torch.save(checkpoint, best_path)
            return filepath, True

        return filepath, False

    def plot_curves(self, epoch):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        if self.batch_losses:
            ax1.plot(self.batch_steps, self.batch_losses, 'b-', alpha=0.3, label=f'Batch Loss (every {self.args.plot})')
        if self.epoch_train_losses:
            epochs = list(range(1, len(self.epoch_train_losses) + 1))
            ax1.plot(epochs, self.epoch_train_losses, 'bo-', label='Train Loss')
            ax1.plot(epochs, self.epoch_val_losses, 'ro-', label='Val Loss')

        ax1.set_xlabel('Batch/Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        if self.batch_lrs:
            ax2.plot(self.batch_steps, self.batch_lrs, 'g-')
            ax2.set_xlabel('Batch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / "plots" / f"training_epoch_{epoch + 1}.png", dpi=150, bbox_inches='tight')
        plt.close()

    def validate(self):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                if self.interrupted:
                    break
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self):
        print(f"Training on {self.device}")
        self.setup_data()
        self.setup_model()

        for epoch in range(self.start_epoch, self.args.epochs):
            if self.interrupted:
                break

            start_time = time.time()
            self.model.train()
            total_loss = 0.0

            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{self.args.epochs}") as pbar:
                for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                    if self.interrupted:
                        val_loss = self.validate()
                        self.save_checkpoint(epoch, total_loss / (batch_idx + 1), val_loss, is_emergency=True)
                        self.plot_curves(epoch)
                        print(f"Emergency save completed")
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
                        self.batch_losses.append(batch_loss)
                        self.batch_steps.append(self.step)
                        self.batch_lrs.append(current_lr)

                    pbar.update(1)
                    pbar.set_postfix({'loss': f'{batch_loss:.6f}'})

            if self.interrupted:
                val_loss = self.validate()
                self.save_checkpoint(epoch, total_loss / len(self.train_loader), val_loss, is_emergency=True)
                self.plot_curves(epoch)
                print(f"Emergency save completed")
                return

            train_loss = total_loss / len(self.train_loader)
            val_loss = self.validate()

            self.epoch_train_losses.append(train_loss)
            self.epoch_val_losses.append(val_loss)

            current_lr = self.optimizer.param_groups[0]['lr']

            print(
                f"Epoch [{epoch + 1}/{self.args.epochs}] Train: {train_loss:.6f} Val: {val_loss:.6f} LR: {current_lr:.6e} Time: {time.time() - start_time:.1f}s")

            if self.scheduler:
                self.scheduler.step(val_loss)

            checkpoint_path, is_best = self.save_checkpoint(epoch, train_loss, val_loss)
            if is_best:
                print(f"Best model saved! Val Loss: {val_loss:.6f}")

            self.plot_curves(epoch)

        if not self.interrupted:
            print("Training completed successfully!")
            final_plot = self.save_dir / "plots" / "final_training_curves.png"
            plt.savefig(final_plot, dpi=150, bbox_inches='tight')
            print(f"Results saved to: {self.save_dir}")


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
