#!/usr/bin/env python3
"""
TrackNetV4 Badminton Tracking Network Training Script
- Supports CUDA/MPS/CPU automatic selection
- Supports training from scratch and resume from checkpoint
- MIMO design with automatic saving per epoch
- Auto-save on forced termination with signal handling
- Optimized plotting strategy: only plot when saving
- Early directory structure setup
"""

import argparse
import json
import logging
import time
import signal
import sys
import atexit
from pathlib import Path
from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset_controller.ball_tracking_data_reader import BallTrackingDataset
from tracknet import TrackNetV4, WeightedBCELoss

# Global variable for signal handling
_trainer_instance = None

# Default configuration
DEFAULT_CONFIG = {
    "batch_size": 2,
    "num_epochs": 30,
    "learning_rate": 1.0,
    "weight_decay": 0.0,
    "gradient_clip": 1.0,
    "input_height": 288,
    "input_width": 512,
    "heatmap_radius": 3,
    "detection_threshold": 0.5,
    "distance_threshold": 4,
    "scheduler_factor": 0.5,
    "scheduler_patience": 8,
    "early_stop_patience": 15,
    "train_split": 0.8,
    "save_interval": 1,  # Save every epoch
}

DATASET_CONFIG = {
    "input_frames": 3,
    "output_frames": 3,  # MIMO design
    "normalize_coords": True,
    "normalize_pixels": True,
    "video_ext": ".mp4",
    "csv_suffix": "_ball.csv"
}


def setup_directories(save_dir):
    """Setup all required directories at script startup"""
    save_path = Path(save_dir)

    # Create main save directory
    save_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for different types of saves
    checkpoints_dir = save_path / "checkpoints"
    plots_dir = save_path / "plots"
    logs_dir = save_path / "logs"

    checkpoints_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    print(f"✓ Directory structure created: {save_path}")
    print(f"  - Checkpoints: {checkpoints_dir}")
    print(f"  - Plots: {plots_dir}")
    print(f"  - Logs: {logs_dir}")

    return save_path, checkpoints_dir, plots_dir, logs_dir


def signal_handler(signum, frame):
    """Signal handler for graceful termination on Ctrl+C or SIGTERM"""
    global _trainer_instance

    signal_names = {
        signal.SIGINT: "SIGINT (Ctrl+C)",
        signal.SIGTERM: "SIGTERM"
    }

    signal_name = signal_names.get(signum, f"Signal {signum}")
    print(f"\n⚠️  Received {signal_name} signal, safely saving model and plots...")

    if _trainer_instance is not None:
        try:
            # Save emergency checkpoint with model state and training plots
            emergency_path = _trainer_instance.checkpoints_dir / f'emergency_save_epoch_{_trainer_instance.current_epoch:03d}.pth'
            _trainer_instance.save_emergency_checkpoint(emergency_path)
            print(f"✓ Emergency save completed: {emergency_path}")
            print(f"✓ Training plots updated")
        except Exception as e:
            print(f"❌ Emergency save failed: {e}")

    print("🔄 Process exiting safely")
    sys.exit(0)


def cleanup_on_exit():
    """Cleanup function called on program exit"""
    global _trainer_instance
    if _trainer_instance is not None and hasattr(_trainer_instance, 'training_in_progress'):
        if _trainer_instance.training_in_progress:
            print("\n🔄 Program exiting normally, performing final save...")
            try:
                exit_path = _trainer_instance.checkpoints_dir / f'exit_save_epoch_{_trainer_instance.current_epoch:03d}.pth'
                _trainer_instance.save_emergency_checkpoint(exit_path)
                print(f"✓ Exit save completed: {exit_path}")
            except Exception as e:
                print(f"❌ Exit save failed: {e}")


def get_device_and_config():
    """Automatically select the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        config = {"num_workers": 4, "pin_memory": True, "persistent_workers": True}
        print(f"✓ CUDA: {torch.cuda.get_device_name()}")

        # Adjust batch size based on GPU memory
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if memory_gb < 8:
            config["batch_multiplier"] = 0.5

        # CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        config = {"num_workers": 2, "pin_memory": False, "persistent_workers": False}
        print("✓ MPS: Apple Silicon")

    else:
        device = torch.device('cpu')
        config = {"num_workers": 4, "pin_memory": False, "persistent_workers": True}
        print("⚠️ CPU mode")

    return device, config


def init_weights(m):
    """Initialize weights using uniform distribution as per paper requirements"""
    if isinstance(m, nn.Conv2d):
        nn.init.uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def create_gaussian_heatmap(x, y, visibility, height, width, radius=3.0):
    """Generate Gaussian heatmap for ball position"""
    heatmap = torch.zeros(height, width)
    if visibility < 0.5:
        return heatmap

    x_pixel = max(0, min(width - 1, int(x * width)))
    y_pixel = max(0, min(height - 1, int(y * height)))

    kernel_size = int(3 * radius)
    x_min, x_max = max(0, x_pixel - kernel_size), min(width, x_pixel + kernel_size + 1)
    y_min, y_max = max(0, y_pixel - kernel_size), min(height, y_pixel + kernel_size + 1)

    y_coords, x_coords = torch.meshgrid(
        torch.arange(y_min, y_max),
        torch.arange(x_min, x_max),
        indexing='ij'
    )

    dist_sq = (x_coords - x_pixel) ** 2 + (y_coords - y_pixel) ** 2
    gaussian_values = torch.exp(-dist_sq / (2 * radius ** 2))
    gaussian_values[gaussian_values < 0.01] = 0

    heatmap[y_min:y_max, x_min:x_max] = gaussian_values
    return heatmap


def collate_fn(batch):
    """Custom collate function for batch processing"""
    config = DEFAULT_CONFIG
    frames_list, heatmaps_list = [], []

    for frames, labels in batch:
        # Resize input frames
        frames = F.interpolate(
            frames.unsqueeze(0),
            size=(config["input_height"], config["input_width"]),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        frames_list.append(frames)

        # Generate heatmaps
        heatmaps = torch.zeros(len(labels), config["input_height"], config["input_width"])
        for i, label_dict in enumerate(labels):
            if isinstance(label_dict, dict):
                heatmap = create_gaussian_heatmap(
                    label_dict['x'].item(),
                    label_dict['y'].item(),
                    label_dict['visibility'].item(),
                    config["input_height"],
                    config["input_width"],
                    config["heatmap_radius"]
                )
                heatmaps[i] = heatmap
        heatmaps_list.append(heatmaps)

    return torch.stack(frames_list), torch.stack(heatmaps_list)


def load_dataset(data_dir):
    """Load and combine datasets from multiple match directories"""
    data_dir = Path(data_dir)
    match_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('match')])

    if not match_dirs:
        raise ValueError(f"No match directories found in: {data_dir}")

    combined_dataset = None
    for match_dir in match_dirs:
        try:
            dataset = BallTrackingDataset(str(match_dir), config=DATASET_CONFIG)
            if len(dataset) > 0:
                combined_dataset = dataset if combined_dataset is None else combined_dataset + dataset
                print(f"✓ {match_dir.name}: {len(dataset)} samples")
        except Exception as e:
            print(f"✗ {match_dir.name} loading failed: {e}")

    if combined_dataset is None:
        raise ValueError("No usable datasets found")

    print(f"Total: {len(combined_dataset)} samples")
    return combined_dataset


class Trainer:
    def __init__(self, args, device, device_config, save_dir, checkpoints_dir, plots_dir, logs_dir):
        self.args = args
        self.device = device
        self.device_config = device_config

        # Directory structure
        self.save_dir = save_dir
        self.checkpoints_dir = checkpoints_dir
        self.plots_dir = plots_dir
        self.logs_dir = logs_dir

        # Training state tracking
        self.current_epoch = 0
        self.current_batch = 0
        self.training_in_progress = False

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.logs_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Training state
        self.start_epoch = 0
        self.best_loss = float('inf')

        # Epoch-level metrics (for compatibility)
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        # Batch-level metrics (for detailed plotting)
        self.batch_train_losses = []
        self.batch_numbers = []
        self.epoch_boundaries = []  # Record batch positions at epoch end
        self.total_batches = 0

        # Initialize model
        self.setup_model()

        # Load checkpoint if resuming
        if args.resume:
            self.load_checkpoint(args.resume)

    def setup_model(self):
        """Initialize model and optimizer"""
        self.model = TrackNetV4()

        # Weight initialization
        self.model.apply(init_weights)
        self.model = self.model.to(self.device)

        # Loss function and optimizer
        self.criterion = WeightedBCELoss()
        self.optimizer = optim.Adadelta(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=DEFAULT_CONFIG["scheduler_factor"],
            patience=DEFAULT_CONFIG["scheduler_patience"]
        )

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model parameters: {total_params:,}")

    def save_checkpoint(self, epoch, is_best=False, checkpoint_type="regular"):
        """Save checkpoint and plot training curves when saving"""
        checkpoint = {
            'epoch': epoch,
            'current_batch': self.current_batch,
            'total_batches': self.total_batches,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,

            # Epoch-level data
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,

            # Batch-level data
            'batch_train_losses': self.batch_train_losses,
            'batch_numbers': self.batch_numbers,
            'epoch_boundaries': self.epoch_boundaries,

            'config': vars(self.args),
            'checkpoint_type': checkpoint_type,
            'save_time': time.time()
        }

        # Save latest
        torch.save(checkpoint, self.checkpoints_dir / f'epoch_{epoch:03d}.pth')
        torch.save(checkpoint, self.checkpoints_dir / 'latest.pth')

        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoints_dir / 'best.pth')
            self.logger.info(f"✓ Best model Epoch {epoch}: {self.best_loss:.6f}")

        # Plot training curves when saving
        try:
            if len(self.batch_train_losses) > 0:
                self.plot_curves(epoch)
                self.logger.info(f"✓ Training curves updated")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate training curves: {e}")

    def save_emergency_checkpoint(self, save_path):
        """Save emergency checkpoint including model state and training plots"""
        checkpoint = {
            'epoch': self.current_epoch,
            'current_batch': self.current_batch,
            'total_batches': self.total_batches,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,

            # Epoch-level data
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,

            # Batch-level data
            'batch_train_losses': self.batch_train_losses,
            'batch_numbers': self.batch_numbers,
            'epoch_boundaries': self.epoch_boundaries,

            'config': vars(self.args),
            'checkpoint_type': "emergency",
            'save_time': time.time()
        }

        # Save model checkpoint
        torch.save(checkpoint, save_path)
        # Also save as latest checkpoint
        torch.save(checkpoint, self.checkpoints_dir / 'latest.pth')

        # Generate training curves during emergency save
        try:
            if len(self.batch_train_losses) > 0:  # Ensure batch data exists for plotting
                self.plot_curves(self.current_epoch)
                self.logger.info(f"✓ Training curves saved")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate training curves: {e}")

        # Log detailed emergency save information
        self.logger.info(f"📊 Emergency save details:")
        self.logger.info(f"   - Model checkpoint: {save_path}")
        self.logger.info(f"   - Current Epoch: {self.current_epoch}")
        self.logger.info(f"   - Current Batch: {self.current_batch}")
        self.logger.info(f"   - Total Batches: {self.total_batches}")
        if len(self.batch_train_losses) > 0:
            self.logger.info(f"   - Latest training loss: {self.batch_train_losses[-1]:.6f}")
        if len(self.val_losses) > 0:
            self.logger.info(f"   - Latest validation loss: {self.val_losses[-1]:.6f}")
        self.logger.info(f"   - Best validation loss: {self.best_loss:.6f}")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint from file"""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            self.logger.warning(f"Checkpoint does not exist: {checkpoint_path}")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']

        # Load epoch-level data
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.learning_rates = checkpoint['learning_rates']

        # Load batch-level data (backward compatibility)
        self.batch_train_losses = checkpoint.get('batch_train_losses', [])
        self.batch_numbers = checkpoint.get('batch_numbers', [])
        self.epoch_boundaries = checkpoint.get('epoch_boundaries', [])
        self.total_batches = checkpoint.get('total_batches', 0)

        # Display special info for emergency checkpoints
        checkpoint_type = checkpoint.get('checkpoint_type', 'regular')
        if checkpoint_type == 'emergency':
            self.logger.info(f"✓ Resumed from emergency checkpoint: Epoch {self.start_epoch}, Total Batch {self.total_batches}")
        else:
            self.logger.info(f"✓ Continuing from Epoch {self.start_epoch}, Total Batch {self.total_batches}")

    def train_epoch(self, epoch, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        self.current_epoch = epoch

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            self.current_batch = batch_idx
            self.total_batches += 1

            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()

            # Gradient clipping
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

            self.optimizer.step()

            total_loss += loss.item()

            # Record batch-level loss
            self.batch_train_losses.append(loss.item())
            self.batch_numbers.append(self.total_batches)

            # Update progress bar display without plotting
            pbar.set_postfix({'Loss': f'{loss.item():.6f}', 'Batch': self.total_batches})

        # Record epoch end position
        self.epoch_boundaries.append(self.total_batches)

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        """Validation phase"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def plot_curves(self, epoch):
        """Plot training curves - comprehensive batch-level and epoch-level charts"""
        if len(self.batch_train_losses) < 2:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Left plot: batch-level training loss curve
        if len(self.batch_train_losses) > 0:
            ax1.plot(self.batch_numbers, self.batch_train_losses, 'b-', alpha=0.7, linewidth=1, label='Training Loss (batch)')

            # Mark epoch boundaries
            for i, boundary in enumerate(self.epoch_boundaries):
                if boundary <= len(self.batch_numbers):
                    ax1.axvline(x=boundary, color='red', linestyle='--', alpha=0.6)
                    ax1.text(boundary, ax1.get_ylim()[1] * 0.95, f'E{i + 1}',
                             rotation=90, ha='right', va='top', fontsize=8, color='red')

            # Add epoch-level validation loss if available
            if len(self.val_losses) > 0:
                epoch_x_positions = self.epoch_boundaries[:len(self.val_losses)]
                ax1.plot(epoch_x_positions, self.val_losses, 'ro-', linewidth=2,
                         markersize=4, label='Validation Loss (epoch)', alpha=0.8)

        ax1.set_xlabel('Batch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves (Batch-level + Epoch Markers)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right plot: epoch-level learning rate curve
        if len(self.learning_rates) > 0:
            epochs_range = range(1, len(self.learning_rates) + 1)
            ax2.plot(epochs_range, self.learning_rates, 'g-', linewidth=2, marker='o', markersize=3)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save with epoch and batch information in filename
        filename = f'training_curves_epoch_{epoch:03d}_batch_{self.total_batches:05d}.png'
        plt.savefig(self.plots_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

    def train(self, train_dataset, val_dataset):
        """Main training loop"""
        # Mark training start
        self.training_in_progress = True

        # Data loaders
        data_kwargs = {
            'batch_size': self.args.batch_size,
            'num_workers': self.device_config['num_workers'],
            'pin_memory': self.device_config['pin_memory'],
            'persistent_workers': self.device_config['persistent_workers'],
            'collate_fn': collate_fn,
            'drop_last': True
        }

        train_loader = DataLoader(train_dataset, shuffle=False, **data_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **data_kwargs)

        self.logger.info(f"Training set: {len(train_dataset)}, Validation set: {len(val_dataset)}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info("⚠️  Press Ctrl+C to safely stop training and auto-save model")

        # Early stopping counter
        patience_counter = 0
        start_time = time.time()

        try:
            for epoch in range(self.start_epoch, self.args.epochs):
                # Training and validation
                train_loss = self.train_epoch(epoch, train_loader)
                val_loss = self.validate(val_loader)

                # Update learning rate
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']

                # Record metrics
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.learning_rates.append(current_lr)

                # Check for best model
                is_best = val_loss < self.best_loss
                if is_best:
                    self.best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Log progress
                self.logger.info(
                    f"Epoch {epoch:03d}: Train={train_loss:.6f}, "
                    f"Val={val_loss:.6f}, LR={current_lr:.2e}"
                    f"{' [BEST]' if is_best else ''}"
                )

                # Save checkpoint (will automatically plot curves)
                if epoch % self.args.save_interval == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)

                # Early stopping check
                if patience_counter >= DEFAULT_CONFIG["early_stop_patience"]:
                    self.logger.info(f"Early stopping triggered at Epoch {epoch}")
                    break

        except KeyboardInterrupt:
            self.logger.info("\n⚠️  Received keyboard interrupt signal")
            # Saving is handled by signal handler

        except Exception as e:
            self.logger.error(f"❌ Exception occurred during training: {e}")
            # Save checkpoint on exception, including plots
            try:
                exception_path = self.checkpoints_dir / f'exception_save_epoch_{self.current_epoch:03d}.pth'
                self.save_emergency_checkpoint(exception_path)
                self.logger.info(f"✓ Exception save completed: {exception_path}")
            except Exception as save_error:
                self.logger.error(f"❌ Exception save failed: {save_error}")
            raise

        finally:
            # Mark training end
            self.training_in_progress = False

        # Training completed
        total_time = time.time() - start_time
        self.logger.info("=" * 50)
        self.logger.info(f"Training completed! Duration: {total_time / 3600:.2f} hours")
        self.logger.info(f"Best validation loss: {self.best_loss:.6f}")
        self.logger.info("=" * 50)


def main():
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

    # Register exit cleanup function
    atexit.register(cleanup_on_exit)

    parser = argparse.ArgumentParser(description='TrackNetV2 Badminton Tracking Training')

    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Save directory')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--save_interval', type=int, default=1, help='Save interval')

    # Resume training parameter
    parser.add_argument('--resume', type=str, help='Checkpoint path to resume training')

    args = parser.parse_args()

    # Setup directory structure first
    save_dir, checkpoints_dir, plots_dir, logs_dir = setup_directories(args.save_dir)

    # Get device configuration
    device, device_config = get_device_and_config()

    # Adjust batch size based on device
    if 'batch_multiplier' in device_config:
        args.batch_size = max(1, int(args.batch_size * device_config['batch_multiplier']))
        print(f"Batch size adjusted to: {args.batch_size}")

    try:
        # Load dataset
        print(f"\nLoading dataset: {args.data_dir}")
        full_dataset = load_dataset(args.data_dir)

        # Split dataset
        total_size = len(full_dataset)
        train_size = int(DEFAULT_CONFIG['train_split'] * total_size)
        indices = torch.randperm(total_size).tolist()

        train_dataset = Subset(full_dataset, indices[:train_size])
        val_dataset = Subset(full_dataset, indices[train_size:])

        print(f"Training set: {len(train_dataset)}, Validation set: {len(val_dataset)}")

        # Setup global trainer instance (for signal handling)
        global _trainer_instance
        trainer = Trainer(args, device, device_config, save_dir, checkpoints_dir, plots_dir, logs_dir)
        _trainer_instance = trainer

        # Start training
        trainer.train(train_dataset, val_dataset)

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

    """
    Usage Examples:
    New training: python train.py --data_dir Dataset/Professional --save_dir checkpoints
    Resume training: python train.py --data_dir Dataset/Professional --resume checkpoints/checkpoints/latest.pth
    Full parameters: python train.py --data_dir Dataset/Professional --save_dir checkpoints --batch_size 2 --epochs 30 --lr 1.0 --weight_decay 0.0 --grad_clip 1.0 --save_interval 1
    """