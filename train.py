"""
TrackNet Training Script - Fixed Resume Functionality

This script trains a TrackNet model for badminton tracking using PyTorch.
Supports custom configurations and proper checkpoint resuming for interrupted training.

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
    """Parse command line arguments with smart defaults"""
    parser = argparse.ArgumentParser(
        description="TrackNet Training Script with Fixed Resume Functionality",
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

    # Store default values for comparison
    args._defaults = {
        'split': 0.8, 'seed': 26, 'batch': 3, 'epochs': 30, 'workers': 0, 'device': 'auto',
        'optimizer': 'Adadelta', 'lr': 1.0, 'wd': 0, 'scheduler': 'ReduceLROnPlateau',
        'factor': 0.5, 'patience': 3, 'min_lr': 1e-6, 'plot': 10,
        'out': 'training_outputs', 'name': 'tracknet_experiment'
    }

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
        self.batch_steps = []  # Batch steps (global counter)
        self.batch_lrs = []  # Batch learning rates

        self.epoch_train_losses = []  # Epoch training losses
        self.epoch_val_losses = []  # Epoch validation losses
        self.epoch_steps = []  # Epoch corresponding batch steps

        self.global_batch_count = 0  # Global batch counter across all epochs

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

    def load_history(self, checkpoint, reset_history=False):
        """Load training history from checkpoint or reset for fresh start"""
        if reset_history:
            print("Resetting training history for fresh plotting...")
            # Keep all lists empty for fresh start
            self.batch_losses = []
            self.batch_steps = []
            self.batch_lrs = []
            self.epoch_train_losses = []
            self.epoch_val_losses = []
            self.epoch_steps = []
            self.global_batch_count = 0
        elif 'training_history' in checkpoint:
            history = checkpoint['training_history']
            self.batch_losses = history.get('batch_losses', [])
            self.batch_steps = history.get('batch_steps', [])
            self.batch_lrs = history.get('batch_lrs', [])
            self.epoch_train_losses = history.get('epoch_train_losses', [])
            self.epoch_val_losses = history.get('epoch_val_losses', [])
            self.epoch_steps = history.get('epoch_steps', [])
            self.global_batch_count = history.get('global_batch_count', 0)

    def reset_epoch_batch_count(self):
        """Reset batch count for a new/restarted epoch"""
        # Keep global history but reset for proper epoch restart
        pass

    def get_history(self):
        """Get current training history for saving"""
        return {
            'batch_losses': self.batch_losses,
            'batch_steps': self.batch_steps,
            'batch_lrs': self.batch_lrs,
            'epoch_train_losses': self.epoch_train_losses,
            'epoch_val_losses': self.epoch_val_losses,
            'epoch_steps': self.epoch_steps,
            'global_batch_count': self.global_batch_count
        }

    def update_batch_loss(self, loss, lr):
        """Update batch loss for plotting"""
        self.global_batch_count += 1

        # Record according to configured interval
        if self.global_batch_count % self.args.plot == 0:
            self.batch_losses.append(loss)
            self.batch_steps.append(self.global_batch_count)
            self.batch_lrs.append(lr)

    def update_epoch_loss(self, train_loss, val_loss):
        """Update epoch losses"""
        self.epoch_train_losses.append(train_loss)
        self.epoch_val_losses.append(val_loss)
        self.epoch_steps.append(self.global_batch_count)

    def plot_training_curves(self, save_path):
        """Plot training curves"""
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

    def save_checkpoint(self, model, optimizer, scheduler, epoch, metrics, training_history, is_emergency=False,
                        config=None):
        """Save checkpoint with proper completion status"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'training_history': training_history,
            'timestamp': timestamp,
            'is_emergency': is_emergency,  # Flag to indicate emergency save
            'epoch_completed': not is_emergency,  # Whether the epoch was fully completed
            'config': config  # Save current config for parameter merging
        }

        # Choose filename based on save type
        if is_emergency:
            filename = f"emergency_checkpoint_epoch_{epoch + 1}_{timestamp}.pth"
        else:
            filename = f"checkpoint_epoch_{epoch + 1}_{timestamp}.pth"

        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)

        # If best model and not emergency, save as best_model.pth
        if not is_emergency and metrics['val_loss'] < self.best_loss:
            self.best_loss = metrics['val_loss']
            best_path = self.save_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            return filepath, True

        return filepath, False


class Trainer:
    """Main trainer class with fixed resume logic"""

    def __init__(self, args):
        self.args = args
        self.start_epoch = 0
        self.resume_checkpoint = None
        self.is_resuming_from_emergency = False

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

        # Load training history if resuming (with fresh plotting option)
        if self.resume_checkpoint:
            # Reset history for fresh plotting in resumed training
            self.monitor.load_history(self.resume_checkpoint, reset_history=True)
            print("Training history reset for fresh plotting")

            # Don't load best loss - start fresh tracking
            print("Best model tracking reset for fresh evaluation")

        # Setup interrupt handling
        self.emergency_save = False
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def merge_parameters_with_checkpoint(self):
        """Merge parameters with smart priority: command_line > checkpoint > defaults"""
        if not self.resume_checkpoint:
            return

        # Get saved config from checkpoint
        checkpoint_config = self.resume_checkpoint.get('config', {})
        if not checkpoint_config:
            print("No config found in checkpoint, using command line + defaults")
            return

        print("Merging parameters with priority: command_line > checkpoint > defaults")

        # Track what we're using from where
        sources = {'command_line': [], 'checkpoint': [], 'defaults': []}

        # Check each parameter
        for param_name, default_value in self.args._defaults.items():
            current_value = getattr(self.args, param_name)
            checkpoint_value = checkpoint_config.get(param_name)

            if current_value != default_value:
                # User explicitly set this parameter
                sources['command_line'].append(f"{param_name}={current_value}")
            elif checkpoint_value is not None:
                # Use checkpoint value
                setattr(self.args, param_name, checkpoint_value)
                sources['checkpoint'].append(f"{param_name}={checkpoint_value}")
            else:
                # Use default value (already set)
                sources['defaults'].append(f"{param_name}={default_value}")

        # Print parameter sources
        if sources['command_line']:
            print(f"From command line: {', '.join(sources['command_line'])}")
        if sources['checkpoint']:
            print(f"From checkpoint: {', '.join(sources['checkpoint'])}")
        if sources['defaults']:
            print(f"Using defaults: {', '.join(sources['defaults'])}")

    def load_checkpoint(self):
        """Load checkpoint for resuming training with proper logic"""
        checkpoint_path = Path(self.args.resume)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        print(f"Loading checkpoint from: {checkpoint_path}")
        self.resume_checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Save current config to checkpoint for parameter merging
        if 'config' not in self.resume_checkpoint:
            self.resume_checkpoint['config'] = {}

        # Merge parameters with smart priority
        self.merge_parameters_with_checkpoint()

        # Check if this is an emergency checkpoint or completed epoch
        is_emergency = self.resume_checkpoint.get('is_emergency', False)
        epoch_completed = self.resume_checkpoint.get('epoch_completed', True)

        if is_emergency or not epoch_completed:
            # Emergency save or incomplete epoch - restart the same epoch
            self.start_epoch = self.resume_checkpoint['epoch']
            self.is_resuming_from_emergency = True
            print(f"Resuming from incomplete epoch {self.start_epoch + 1} (emergency/incomplete save)")
        else:
            # Normal completed epoch - start from next epoch
            self.start_epoch = self.resume_checkpoint['epoch'] + 1
            self.is_resuming_from_emergency = False
            print(
                f"Resuming from completed epoch {self.resume_checkpoint['epoch'] + 1}, starting epoch {self.start_epoch + 1}")

    def signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        print("\nInterrupt signal detected, emergency saving...")
        self.emergency_save = True

    def setup_directories(self):
        """Setup directory structure"""
        if self.args.resume:
            # For resumed training, create a new timestamped directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.is_resuming_from_emergency:
                experiment_name = f"{self.args.name}_resumed_emergency_fresh_{timestamp}"
            else:
                experiment_name = f"{self.args.name}_resumed_fresh_{timestamp}"
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

        # Set random seed (use same seed for consistent split when resuming)
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

        # Load checkpoint states if resuming - SMART PARAMETER PRIORITY
        if self.resume_checkpoint:
            print("Loading model state from checkpoint...")
            self.model.load_state_dict(self.resume_checkpoint['model_state_dict'])

            print("Loading optimizer with smart parameter merging...")

            # Get the checkpoint's optimizer state and current args
            checkpoint_optimizer_state = self.resume_checkpoint['optimizer_state_dict']

            # Load optimizer state but update hyperparameters based on current args
            self.optimizer.load_state_dict(checkpoint_optimizer_state)

            # Update hyperparameters that may have been overridden by args merging
            current_lr = self.args.lr
            current_wd = self.args.wd

            # Apply current args values to loaded optimizer
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
                param_group['weight_decay'] = current_wd

            # Print what was used
            if 'param_groups' in checkpoint_optimizer_state and len(checkpoint_optimizer_state['param_groups']) > 0:
                checkpoint_lr = checkpoint_optimizer_state['param_groups'][0].get('lr', 'unknown')
                checkpoint_wd = checkpoint_optimizer_state['param_groups'][0].get('weight_decay', 'unknown')
                print(f"Final learning rate: {current_lr} (checkpoint: {checkpoint_lr})")
                print(f"Final weight decay: {current_wd} (checkpoint: {checkpoint_wd})")

            # Scheduler: Use current args parameters
            if self.scheduler and self.args.scheduler != 'None':
                if self.resume_checkpoint.get('scheduler_state_dict') and not any([
                    getattr(self.args, 'factor') != self.args._defaults['factor'],
                    getattr(self.args, 'patience') != self.args._defaults['patience'],
                    getattr(self.args, 'min_lr') != self.args._defaults['min_lr']
                ]):
                    # Load scheduler state only if no scheduler params were overridden
                    print("Loading scheduler state from checkpoint...")
                    self.scheduler.load_state_dict(self.resume_checkpoint['scheduler_state_dict'])
                else:
                    print("Scheduler reset with new parameters from command line/defaults")

            print(f"Successfully resumed from epoch {self.start_epoch + 1} with merged parameters")

    def emergency_checkpoint(self, epoch, train_loss, val_loss):
        """Emergency save checkpoint"""
        print("Performing emergency save...")

        # Save emergency checkpoint
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

        checkpoint_path, _ = self.checkpoint.save_checkpoint(
            self.model, self.optimizer, self.scheduler, epoch, metrics,
            self.monitor.get_history(), is_emergency=True, config=vars(self.args)
        )

        # Save training curves
        plot_path = self.save_dir / "plots" / f"emergency_training_curves_epoch_{epoch + 1}.png"
        self.monitor.plot_training_curves(plot_path)

        print(f"Emergency save completed: {checkpoint_path}")

    def print_parameter_summary(self, is_emergency=False, current_epoch=None):
        """Print final parameter summary"""
        if is_emergency:
            print("\n" + "=" * 50)
            print("EMERGENCY SAVE PARAMETERS SUMMARY:")
            print("=" * 50)
            print(f"Dataset: {self.args.data}")
            print(f"Interrupted at epoch: {current_epoch + 1}/{self.args.epochs}")
            print(f"Batch size: {self.args.batch}")
            print(f"Learning rate at interrupt: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"Weight decay: {self.optimizer.param_groups[0]['weight_decay']}")
            print(f"Optimizer: {self.args.optimizer}")
            print(f"Scheduler: {self.args.scheduler}")
            print(f"Device: {self.device}")
            print("=" * 50)
            print(f"Emergency save location: {self.save_dir}")
        else:
            print("\n" + "=" * 50)
            print("FINAL TRAINING PARAMETERS SUMMARY:")
            print("=" * 50)
            print(f"Dataset: {self.args.data}")
            print(f"Total epochs completed: {self.args.epochs}")
            print(f"Batch size: {self.args.batch}")
            print(f"Final learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"Weight decay: {self.optimizer.param_groups[0]['weight_decay']}")
            print(f"Optimizer: {self.args.optimizer}")
            print(f"Scheduler: {self.args.scheduler}")
            if self.args.scheduler != 'None':
                print(f"  - Factor: {self.args.factor}")
                print(f"  - Patience: {self.args.patience}")
                print(f"  - Min LR: {self.args.min_lr}")
            print(f"Device: {self.device}")
            print(f"Data split: {self.args.split:.2f}")
            print(f"Random seed: {self.args.seed}")
            print("=" * 50)
            print(f"All results saved to: {self.save_dir}")

    def train(self):
        """Main training loop with fixed resume logic"""
        print(f"Starting training...")
        print(f"Using device: {self.device}")
        if self.args.resume:
            if self.is_resuming_from_emergency:
                print(f"Resuming from emergency save - restarting epoch {self.start_epoch + 1}")
            else:
                print(f"Resuming from completed epoch - starting epoch {self.start_epoch + 1}")
            print("NOTE: All parameters use command line values. Training plots start fresh.")
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

            # Training phase
            with tqdm(total=len(self.train_loader), desc="Training", ncols=80) as pbar:
                self.model.train()
                total_loss = 0.0

                for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                    if self.emergency_save:
                        # Perform emergency save with incomplete epoch data
                        val_loss = self.validate() if total_loss > 0 else float('inf')
                        train_loss = total_loss / (batch_idx + 1) if batch_idx > 0 else float('inf')

                        metrics = {
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'learning_rate': self.optimizer.param_groups[0]['lr']
                        }

                        checkpoint_path, _ = self.checkpoint.save_checkpoint(
                            self.model, self.optimizer, self.scheduler, epoch, metrics,
                            self.monitor.get_history(), is_emergency=True, config=vars(self.args)
                        )

                        plot_path = self.save_dir / "plots" / f"emergency_training_curves_epoch_{epoch + 1}.png"
                        self.monitor.plot_training_curves(plot_path)
                        print(f"Emergency save completed: {checkpoint_path}")

                        # Print parameter summary for emergency save
                        self.print_parameter_summary(is_emergency=True, current_epoch=epoch)
                        return

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

            # Validation phase
            if self.emergency_save:
                metrics = {
                    'train_loss': train_loss,
                    'val_loss': float('inf'),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                checkpoint_path, _ = self.checkpoint.save_checkpoint(
                    self.model, self.optimizer, self.scheduler, epoch, metrics,
                    self.monitor.get_history(), is_emergency=True, config=vars(self.args)
                )
                plot_path = self.save_dir / "plots" / f"emergency_training_curves_epoch_{epoch + 1}.png"
                self.monitor.plot_training_curves(plot_path)
                print(f"Emergency save completed: {checkpoint_path}")

                # Print parameter summary for emergency save
                self.print_parameter_summary(is_emergency=True, current_epoch=epoch)
                return

            with tqdm(total=len(self.val_loader), desc="Validation", ncols=80) as pbar:
                val_loss = self.validate()
                pbar.update(len(self.val_loader))
                pbar.set_postfix({'loss': f'{val_loss:.6f}'})

            # Check for emergency save after validation
            if self.emergency_save:
                metrics = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                checkpoint_path, _ = self.checkpoint.save_checkpoint(
                    self.model, self.optimizer, self.scheduler, epoch, metrics,
                    self.monitor.get_history(), is_emergency=True, config=vars(self.args)
                )
                plot_path = self.save_dir / "plots" / f"emergency_training_curves_epoch_{epoch + 1}.png"
                self.monitor.plot_training_curves(plot_path)
                print(f"Emergency save completed: {checkpoint_path}")

                # Print parameter summary for emergency save
                self.print_parameter_summary(is_emergency=True, current_epoch=epoch)
                return

            # Update epoch loss record
            self.monitor.update_epoch_loss(train_loss, val_loss)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Print epoch results
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")

            # Update learning rate
            if self.scheduler:
                self.scheduler.step(val_loss)

            # Save checkpoint (normal save - epoch completed)
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr
            }

            checkpoint_path, is_best = self.checkpoint.save_checkpoint(
                self.model, self.optimizer, self.scheduler, epoch, metrics,
                self.monitor.get_history(), is_emergency=False, config=vars(self.args)
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

        # Training completed normally
        if not self.emergency_save:
            print("\nTraining completed successfully!")

            # Save final training curves
            final_plot_path = self.save_dir / "plots" / "final_training_curves.png"
            self.monitor.plot_training_curves(final_plot_path)

            # Print final parameter summary
            self.print_parameter_summary(is_emergency=False)

        # Training completed normally
        if not self.emergency_save:
            print("\nTraining completed successfully!")

            # Save final training curves
            final_plot_path = self.save_dir / "plots" / "final_training_curves.png"
            self.monitor.plot_training_curves(final_plot_path)

            print(f"All results saved to: {self.save_dir}")

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

        avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        return avg_loss


# ================== Main Program Entry ==================
if __name__ == "__main__":
    args = parse_arguments()
    trainer = Trainer(args)
    trainer.train()
