"""
TrackNet Training Script - Optimized with Resume Functionality

This script trains a TrackNet model for badminton tracking using PyTorch.
Supports custom configurations and checkpoint resuming for interrupted training.

Usage Examples:
1. Basic training with default settings:
   python train.py --data dataset/Professional_reorg_train

2. Custom training with specific parameters:
   python train.py --data dataset/train --batch 8 --epochs 50 --lr 2.0 --device cuda

3. Advanced training with all custom settings:
   python train.py --data dataset/train --batch 4 --epochs 100 --lr 1.5 --split 0.9 --out outputs --name advanced_exp --plot 5 --patience 5

4. Resume training from checkpoint:
   python train.py --resume checkpoints/best_model.pth --data dataset/train

5. Resume with modified settings:
   python train.py --resume checkpoints/checkpoint_epoch_20.pth --data dataset/train --epochs 100 --device cuda

6. High-performance training:
   python train.py --data dataset/train --batch 16 --workers 4 --device cuda --seed 42 --name gpu_training

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
import logging
import signal
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from tracknet_v4 import TrackNetV4 as TrackNet
from preprocessing.tracknet_dataset import FrameHeatmapDataset


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="TrackNet Training Script with Resume Functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --data dataset/train
  %(prog)s --data dataset/train --batch 8 --epochs 50
  %(prog)s --resume checkpoints/best_model.pth
  %(prog)s --resume checkpoint.pth --epochs 100 --device cuda
        """
    )

    # Main arguments
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the training dataset directory')
    parser.add_argument('--resume', type=str,
                        help='Path to checkpoint file for resuming training')

    # Dataset arguments
    parser.add_argument('--split', type=float, default=0.8,
                        help='Ratio of training data (default: 0.8)')
    parser.add_argument('--seed', type=int, default=26,
                        help='Random seed for data splitting (default: 26)')

    # Training arguments
    parser.add_argument('--batch', type=int, default=3,
                        help='Batch size for training (default: 3)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of data loading workers (default: 0)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: auto/cpu/cuda/mps (default: auto)')

    # Optimizer arguments
    parser.add_argument('--optimizer', type=str, default='Adadelta',
                        choices=['Adadelta', 'Adam', 'SGD'],
                        help='Optimizer type (default: Adadelta)')
    parser.add_argument('--lr', type=float, default=1.0,
                        help='Learning rate (default: 1.0)')
    parser.add_argument('--wd', type=float, default=0,
                        help='Weight decay for optimizer (default: 0)')

    # Learning rate scheduler arguments
    parser.add_argument('--scheduler', type=str, default='ReduceLROnPlateau',
                        choices=['ReduceLROnPlateau', 'None'],
                        help='Learning rate scheduler type (default: ReduceLROnPlateau)')
    parser.add_argument('--factor', type=float, default=0.5,
                        help='Factor by which LR is reduced (default: 0.5)')
    parser.add_argument('--patience', type=int, default=3,
                        help='Epochs to wait before reducing LR (default: 3)')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate (default: 1e-6)')

    # Logging and saving arguments
    parser.add_argument('--plot', type=int, default=10,
                        help='Interval for recording batch losses (default: 10)')
    parser.add_argument('--out', type=str, default='training_outputs',
                        help='Directory to save outputs (default: training_outputs)')
    parser.add_argument('--name', type=str, default='tracknet_experiment',
                        help='Name for this experiment (default: tracknet_experiment)')

    args = parser.parse_args()
    return args


class WeightedBinaryCrossEntropy(nn.Module):
    """
    Weighted Binary Cross Entropy Loss as defined in the paper
    WBCE = -Σ[(1-w)² * ŷ * log(y) + w² * (1-ŷ) * log(1-y)]
    where w = y (prediction value as weight)
    """

    def __init__(self, epsilon=1e-7):
        super(WeightedBinaryCrossEntropy, self).__init__()
        self.epsilon = epsilon  # Prevent log(0)

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Model predictions [B, 3, H, W], range [0,1]
            y_true: Ground truth labels [B, 3, H, W], range {0,1}
        Returns:
            loss: Scalar loss value
        """
        # Clamp predictions to valid range, avoid log(0)
        y_pred = torch.clamp(y_pred, self.epsilon, 1 - self.epsilon)

        # w = y (paper definition: weight equals prediction)
        w = y_pred

        # Calculate weighted binary cross entropy
        # WBCE = -Σ[(1-w)² * ŷ * log(y) + w² * (1-ŷ) * log(1-y)]
        term1 = (1 - w) ** 2 * y_true * torch.log(y_pred)
        term2 = w ** 2 * (1 - y_true) * torch.log(1 - y_pred)

        # Negative sign and sum
        wbce = -(term1 + term2)

        # Return batch average loss
        return wbce.mean()


class TrainingMonitor:
    """Training monitor for logging and visualization"""

    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

        # Training history
        self.batch_losses = []  # Batch losses
        self.batch_steps = []  # Batch steps
        self.batch_lrs = []  # Batch learning rates

        self.epoch_train_losses = []  # Epoch training losses
        self.epoch_val_losses = []  # Epoch validation losses
        self.epoch_steps = []  # Epoch corresponding batch steps

        self.current_batch = 0

        # Setup logging
        self.setup_logger()

    def setup_logger(self):
        """Setup logging configuration"""
        log_file = self.save_dir / "training.log"

        # Configure logging format, remove console output to reduce verbosity
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[logging.FileHandler(log_file)]
        )
        self.logger = logging.getLogger(__name__)

    def load_history(self, checkpoint):
        """Load training history from checkpoint"""
        if 'training_history' in checkpoint:
            history = checkpoint['training_history']
            self.batch_losses = history.get('batch_losses', [])
            self.batch_steps = history.get('batch_steps', [])
            self.batch_lrs = history.get('batch_lrs', [])
            self.epoch_train_losses = history.get('epoch_train_losses', [])
            self.epoch_val_losses = history.get('epoch_val_losses', [])
            self.epoch_steps = history.get('epoch_steps', [])
            self.current_batch = history.get('current_batch', 0)

    def get_history(self):
        """Get current training history for saving"""
        return {
            'batch_losses': self.batch_losses,
            'batch_steps': self.batch_steps,
            'batch_lrs': self.batch_lrs,
            'epoch_train_losses': self.epoch_train_losses,
            'epoch_val_losses': self.epoch_val_losses,
            'epoch_steps': self.epoch_steps,
            'current_batch': self.current_batch
        }

    def update_batch_loss(self, loss, lr):
        """Update batch loss for plotting"""
        self.current_batch += 1

        # Record according to configured interval
        if self.current_batch % self.args.plot == 0:
            self.batch_losses.append(loss)
            self.batch_steps.append(self.current_batch)
            self.batch_lrs.append(lr)

    def update_epoch_loss(self, train_loss, val_loss):
        """Update epoch losses"""
        self.epoch_train_losses.append(train_loss)
        self.epoch_val_losses.append(val_loss)
        self.epoch_steps.append(self.current_batch)

    def plot_training_curves(self, save_path):
        """Plot training curves with English labels"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Loss curves
        # Plot batch losses
        if self.batch_losses:
            ax1.plot(self.batch_steps, self.batch_losses, 'b-', alpha=0.3,
                     label=f'Batch Loss (every {self.args.plot} batches)')

        # Plot epoch losses
        if self.epoch_train_losses:
            ax1.plot(self.epoch_steps, self.epoch_train_losses, 'bo-',
                     markersize=8, linewidth=2, label='Epoch Train Loss')
        if self.epoch_val_losses:
            ax1.plot(self.epoch_steps, self.epoch_val_losses, 'ro-',
                     markersize=8, linewidth=2, label='Epoch Val Loss')

        ax1.set_xlabel('Batch Number')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Learning rate curve
        if self.batch_lrs:
            ax2.plot(self.batch_steps, self.batch_lrs, 'g-', linewidth=2)
            ax2.set_xlabel(f'Batch Number (every {self.args.plot} batches)')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


class ModelCheckpoint:
    """Model checkpoint manager"""

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.best_loss = float('inf')

    def save_checkpoint(self, model, optimizer, scheduler, epoch, metrics, training_history):
        """Save checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'training_history': training_history,
            'timestamp': timestamp
        }

        # Save every epoch
        filename = f"checkpoint_epoch_{epoch + 1}_{timestamp}.pth"
        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)

        # If best model, save as best_model.pth
        if metrics['val_loss'] < self.best_loss:
            self.best_loss = metrics['val_loss']
            best_path = self.save_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            return filepath, True

        return filepath, False


class Trainer:
    """Main trainer class"""

    def __init__(self, args):
        self.args = args
        self.start_epoch = 0
        self.resume_checkpoint = None

        # Setup device
        if args.device == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(args.device)

        # Load checkpoint if resuming
        if args.resume:
            self.load_checkpoint()

        # Create save directories
        self.setup_directories()

        # Initialize components
        self.monitor = TrainingMonitor(args, self.save_dir)
        self.checkpoint = ModelCheckpoint(self.save_dir / "checkpoints")

        # Load training history if resuming
        if self.resume_checkpoint:
            self.monitor.load_history(self.resume_checkpoint)
            if 'metrics' in self.resume_checkpoint and 'val_loss' in self.resume_checkpoint['metrics']:
                self.checkpoint.best_loss = self.resume_checkpoint['metrics']['val_loss']

        # Setup interrupt handling
        self.emergency_save = False
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def load_checkpoint(self):
        """Load checkpoint for resuming training"""
        checkpoint_path = Path(self.args.resume)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        print(f"Loading checkpoint from: {checkpoint_path}")
        self.resume_checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.start_epoch = self.resume_checkpoint['epoch'] + 1

        print(f"Resuming training from epoch {self.start_epoch}")

        print("Dataset path must be specified with --data for resuming training")

    def signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        print("\nInterrupt signal detected, emergency saving...")
        self.emergency_save = True

    def setup_directories(self):
        """Setup directory structure"""
        if self.args.resume:
            # For resumed training, create a new timestamped directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{self.args.name}_resumed_{timestamp}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{self.args.name}_{timestamp}"

        self.save_dir = Path(self.args.out) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.save_dir / "checkpoints").mkdir(exist_ok=True)
        (self.save_dir / "plots").mkdir(exist_ok=True)
        (self.save_dir / "configs").mkdir(exist_ok=True)

        # Save configuration
        config_path = self.save_dir / "configs" / "training_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(vars(self.args), f, indent=4, ensure_ascii=False)

    def prepare_data(self):
        """Prepare dataset and data loaders"""
        print("Loading dataset...")
        dataset = FrameHeatmapDataset(self.args.data)
        print(f"Dataset size: {len(dataset)}")

        # Set random seed
        torch.manual_seed(self.args.seed)

        # Split dataset
        train_size = int(self.args.split * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch,
            shuffle=True,
            num_workers=self.args.workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")

    def prepare_model(self):
        """Prepare model, loss function and optimizer"""
        # Create model
        self.model = TrackNet().to(self.device)
        self.criterion = WeightedBinaryCrossEntropy()

        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {total_params:,}")

        # Create optimizer
        if self.args.optimizer == "Adadelta":
            self.optimizer = torch.optim.Adadelta(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.wd
            )
        elif self.args.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.wd
            )
        elif self.args.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.wd,
                momentum=0.9
            )

        # Create learning rate scheduler
        if self.args.scheduler == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.args.factor,
                patience=self.args.patience,
                min_lr=self.args.min_lr
            )
        else:
            self.scheduler = None

        # Load checkpoint states if resuming
        if self.resume_checkpoint:
            print("Loading model state from checkpoint...")
            self.model.load_state_dict(self.resume_checkpoint['model_state_dict'])

            print("Loading optimizer state from checkpoint...")
            self.optimizer.load_state_dict(self.resume_checkpoint['optimizer_state_dict'])

            if self.scheduler and self.resume_checkpoint['scheduler_state_dict']:
                print("Loading scheduler state from checkpoint...")
                self.scheduler.load_state_dict(self.resume_checkpoint['scheduler_state_dict'])

            print(f"Successfully resumed from epoch {self.start_epoch}")

    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        batch_count = 0

        for inputs, targets in self.train_loader:
            if self.emergency_save:
                break

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Record loss
            batch_loss = loss.item()
            total_loss += batch_loss
            batch_count += 1

            # Update batch loss for plotting
            current_lr = self.optimizer.param_groups[0]['lr']
            self.monitor.update_batch_loss(batch_loss, current_lr)

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        return avg_loss

    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        batch_count = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                if self.emergency_save:
                    break

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                batch_count += 1

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        return avg_loss

    def emergency_checkpoint(self, epoch, train_loss, val_loss):
        """Emergency save checkpoint"""
        # Create emergency save directory
        emergency_dir = Path("emergency_saves") / datetime.now().strftime("%Y%m%d_%H%M%S")
        emergency_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        checkpoint_path = emergency_dir / f"emergency_checkpoint_epoch_{epoch + 1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'training_history': self.monitor.get_history()
        }, checkpoint_path)

        # Save training curves
        plot_path = emergency_dir / "training_curves.png"
        self.monitor.plot_training_curves(plot_path)

        # Save configuration
        config_path = emergency_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(vars(self.args), f, indent=4, ensure_ascii=False)

        print(f"Emergency save completed: {emergency_dir}")

    def train(self):
        """Main training loop"""
        print(f"Starting training...")
        print(f"Using device: {self.device}")
        if self.args.resume:
            print(f"Resuming from epoch {self.start_epoch}")
        print("-" * 50)

        # Prepare data and model
        self.prepare_data()
        self.prepare_model()

        # Training loop
        for epoch in range(self.start_epoch, self.args.epochs):
            if self.emergency_save:
                break

            epoch_start_time = time.time()

            # Show epoch progress
            print(f"\nEpoch [{epoch + 1}/{self.args.epochs}]")

            # Training
            with tqdm(total=len(self.train_loader), desc="Training", ncols=80) as pbar:
                self.model.train()
                total_loss = 0.0

                for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                    if self.emergency_save:
                        break

                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()

                    batch_loss = loss.item()
                    total_loss += batch_loss

                    # Update batch loss record
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.monitor.update_batch_loss(batch_loss, current_lr)

                    pbar.update(1)
                    pbar.set_postfix({'loss': f'{batch_loss:.6f}'})

                train_loss = total_loss / len(self.train_loader)

            # Validation
            with tqdm(total=len(self.val_loader), desc="Validation", ncols=80) as pbar:
                val_loss = self.validate()
                pbar.update(len(self.val_loader))
                pbar.set_postfix({'loss': f'{val_loss:.6f}'})

            # Update epoch loss record
            self.monitor.update_epoch_loss(train_loss, val_loss)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Print epoch results
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")

            # Update learning rate
            if self.scheduler:
                self.scheduler.step(val_loss)

            # Save checkpoint
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr
            }

            checkpoint_path, is_best = self.checkpoint.save_checkpoint(
                self.model, self.optimizer, self.scheduler, epoch, metrics, self.monitor.get_history()
            )

            if is_best:
                print(f"Saved best model! Validation loss: {val_loss:.6f}")

            # Save training curves
            plot_path = self.save_dir / "plots" / f"training_curves_epoch_{epoch + 1}.png"
            self.monitor.plot_training_curves(plot_path)

            # Log
            self.monitor.logger.info(
                f"Epoch {epoch + 1}/{self.args.epochs}: "
                f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                f"lr={current_lr:.6f}, time={time.time() - epoch_start_time:.2f}s"
            )

        # Handle interruption or normal completion
        if self.emergency_save:
            self.emergency_checkpoint(epoch, train_loss, val_loss)
        else:
            print("\nTraining completed!")

            # Save final training curves
            final_plot_path = self.save_dir / "plots" / "final_training_curves.png"
            self.monitor.plot_training_curves(final_plot_path)

            print(f"All results saved to: {self.save_dir}")


# ================== Main Program Entry ==================
if __name__ == "__main__":
    args = parse_arguments()
    trainer = Trainer(args)
    trainer.train()
