#!/usr/bin/env python3
"""
TrackNet Badminton Tracking Network Training Script - FIXED VERSION
- Proper coordinate mapping with equal ratio scaling
- No normalization in dataset loading (keep raw pixels 0-255 and pixel coordinates)
- Precise transformation: raw -> equal ratio scale -> normalize
- Equal ratio image scaling (not cropping)
- CUDA/MPS/CPU automatic device selection
- MIMO design with comprehensive state management
"""

import argparse
import logging
import signal
import sys
import time
import atexit
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset_controller.ball_tracking_data_reader import BallTrackingDataset
from tracknet import TrackNet, WeightedBCELoss

# Global trainer instance for signal handling
_trainer_instance = None

# Training configuration
CONFIG = {
    "batch_size": 2,
    "num_epochs": 30,
    "learning_rate": 1.0,
    "weight_decay": 0.0,
    "gradient_clip": 1.0,
    "input_height": 288,
    "input_width": 512,
    "heatmap_radius": 3,
    "detection_threshold": 0.5,
    "scheduler_factor": 0.5,
    "scheduler_patience": 8,
    "early_stop_patience": 15,
    "train_split": 0.8,
    "save_interval": 1,
}

DATASET_CONFIG = {
    "input_frames": 3,
    "output_frames": 3,  # MIMO design
    "normalize_coords": False,  # ⭐ 关键：保持原始像素坐标
    "normalize_pixels": False,  # ⭐ 关键：保持原始像素值0-255
}


def setup_directories(save_dir):
    """Setup complete directory structure at startup"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = save_path / "checkpoints"
    plots_dir = save_path / "plots"
    logs_dir = save_path / "logs"

    for dir_path in [checkpoints_dir, plots_dir, logs_dir]:
        dir_path.mkdir(exist_ok=True)

    print(f"✓ Directory structure created: {save_path}")
    return save_path, checkpoints_dir, plots_dir, logs_dir


def signal_handler(signum, frame):
    """Graceful termination on Ctrl+C or SIGTERM"""
    global _trainer_instance

    signal_names = {signal.SIGINT: "SIGINT (Ctrl+C)", signal.SIGTERM: "SIGTERM"}
    signal_name = signal_names.get(signum, f"Signal {signum}")

    print(f"\n⚠️  Received {signal_name}, saving model and plots...")

    if _trainer_instance:
        try:
            emergency_path = _trainer_instance.checkpoints_dir / f'emergency_epoch_{_trainer_instance.current_epoch:03d}.pth'
            _trainer_instance.save_emergency_checkpoint(emergency_path)
            print(f"✓ Emergency save completed: {emergency_path}")
        except Exception as e:
            print(f"❌ Emergency save failed: {e}")

    print("🔄 Process exiting safely")
    sys.exit(0)


def cleanup_on_exit():
    """Cleanup function on program exit"""
    global _trainer_instance
    if _trainer_instance and hasattr(_trainer_instance, 'training_in_progress'):
        if _trainer_instance.training_in_progress:
            print("\n🔄 Program exiting, performing final save...")
            try:
                exit_path = _trainer_instance.checkpoints_dir / f'exit_epoch_{_trainer_instance.current_epoch:03d}.pth'
                _trainer_instance.save_emergency_checkpoint(exit_path)
                print(f"✓ Exit save completed: {exit_path}")
            except Exception as e:
                print(f"❌ Exit save failed: {e}")


def get_device_and_config():
    """Auto-select best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        config = {"num_workers": 4, "pin_memory": True, "persistent_workers": True}
        print(f"✓ CUDA: {torch.cuda.get_device_name()}")

        # Memory-based batch size adjustment
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if memory_gb < 8:
            config["batch_multiplier"] = 0.5

        torch.backends.cudnn.benchmark = True

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
    """Initialize weights per paper requirements"""
    if isinstance(m, nn.Conv2d):
        nn.init.uniform_(m.weight)
        if m.bias is not None:
            nn.init.uniform_(m.bias)


def calculate_equal_ratio_resize(original_height: int, original_width: int,
                                 target_height: int, target_width: int) -> Tuple[int, int]:
    """Calculate new size for equal ratio scaling (no distortion)"""
    # Calculate scaling ratios
    ratio_h = target_height / original_height
    ratio_w = target_width / original_width

    # Use minimum ratio to ensure both dimensions fit
    scale_ratio = min(ratio_h, ratio_w)

    new_height = int(original_height * scale_ratio)
    new_width = int(original_width * scale_ratio)

    return new_height, new_width, scale_ratio


def create_gaussian_heatmap(x_norm: float, y_norm: float, visibility: float,
                            height: int, width: int, radius: float = 3.0) -> torch.Tensor:
    """
    Generate Gaussian heatmap for ball position

    Args:
        x_norm: Normalized x coordinate (0-1)
        y_norm: Normalized y coordinate (0-1)
        visibility: Visibility flag (0 or 1)
        height: Heatmap height
        width: Heatmap width
        radius: Gaussian radius in pixels

    Returns:
        Gaussian heatmap tensor
    """
    heatmap = torch.zeros(height, width)
    if visibility < 0.5:
        return heatmap

    # Convert normalized coordinates to pixel coordinates
    x_pixel = max(0, min(width - 1, int(x_norm * width)))
    y_pixel = max(0, min(height - 1, int(y_norm * height)))

    # Define gaussian kernel size
    kernel_size = int(3 * radius)
    x_min = max(0, x_pixel - kernel_size)
    x_max = min(width, x_pixel + kernel_size + 1)
    y_min = max(0, y_pixel - kernel_size)
    y_max = min(height, y_pixel + kernel_size + 1)

    # Create coordinate grids
    y_coords, x_coords = torch.meshgrid(
        torch.arange(y_min, y_max),
        torch.arange(x_min, x_max),
        indexing='ij'
    )

    # Calculate gaussian values
    dist_sq = (x_coords - x_pixel) ** 2 + (y_coords - y_pixel) ** 2
    gaussian = torch.exp(-dist_sq / (2 * radius ** 2))

    # Remove very small values
    gaussian[gaussian < 0.01] = 0

    # Assign to heatmap
    heatmap[y_min:y_max, x_min:x_max] = gaussian

    return heatmap


def collate_fn(batch):
    """
    Custom collate function with proper coordinate mapping

    Process:
    1. Keep original image and coordinates
    2. Calculate equal ratio scaling
    3. Resize image with equal ratio
    4. Scale coordinates accordingly
    5. Normalize image pixels to [0, 1]
    6. Normalize coordinates to [0, 1]
    7. Generate Gaussian heatmaps
    """
    frames_list, heatmaps_list = [], []

    for frames, labels in batch:
        # Get original dimensions
        original_height, original_width = frames.shape[-2], frames.shape[-1]

        # Calculate equal ratio resize parameters
        new_height, new_width, scale_ratio = calculate_equal_ratio_resize(
            original_height, original_width,
            CONFIG["input_height"], CONFIG["input_width"]
        )

        # Step 1: Equal ratio resize image
        frames_resized = F.interpolate(
            frames.unsqueeze(0),
            size=(new_height, new_width),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # Step 2: Pad to target size if needed (center padding)
        if new_height != CONFIG["input_height"] or new_width != CONFIG["input_width"]:
            pad_h = CONFIG["input_height"] - new_height
            pad_w = CONFIG["input_width"] - new_width

            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            frames_resized = F.pad(frames_resized,
                                   (pad_left, pad_right, pad_top, pad_bottom),
                                   mode='constant', value=0)
        else:
            pad_left = pad_top = 0

        # Step 3: Normalize image pixels to [0, 1]
        frames_normalized = frames_resized.float() / 255.0
        frames_list.append(frames_normalized)

        # Step 4: Process coordinates and generate heatmaps
        heatmaps = torch.zeros(len(labels), CONFIG["input_height"], CONFIG["input_width"])

        for i, label_dict in enumerate(labels):
            if isinstance(label_dict, dict):
                # Get original pixel coordinates
                x_orig = label_dict['x'].item()  # Original pixel x
                y_orig = label_dict['y'].item()  # Original pixel y
                visibility = label_dict['visibility'].item()

                if visibility >= 0.5:
                    # Step 4a: Scale coordinates
                    x_scaled = x_orig * scale_ratio + pad_left
                    y_scaled = y_orig * scale_ratio + pad_top

                    # Step 4b: Normalize coordinates to [0, 1]
                    x_norm = x_scaled / CONFIG["input_width"]
                    y_norm = y_scaled / CONFIG["input_height"]

                    # Step 4c: Generate Gaussian heatmap
                    heatmap = create_gaussian_heatmap(
                        x_norm, y_norm, visibility,
                        CONFIG["input_height"], CONFIG["input_width"],
                        CONFIG["heatmap_radius"]
                    )
                    heatmaps[i] = heatmap

        heatmaps_list.append(heatmaps)

    return torch.stack(frames_list), torch.stack(heatmaps_list)


def load_dataset(data_dir):
    """Load and combine datasets from match directories"""
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
            print(f"✗ {match_dir.name} failed: {e}")

    if combined_dataset is None:
        raise ValueError("No usable datasets found")

    print(f"Total: {len(combined_dataset)} samples")
    return combined_dataset


class Trainer:
    def __init__(self, args, device, device_config, save_dir, checkpoints_dir, plots_dir, logs_dir):
        self.args = args
        self.device = device
        self.device_config = device_config

        # Directory paths
        self.save_dir = save_dir
        self.checkpoints_dir = checkpoints_dir
        self.plots_dir = plots_dir
        self.logs_dir = logs_dir

        # Training state
        self.current_epoch = 0
        self.current_batch = 0
        self.training_in_progress = False
        self.start_epoch = 0
        self.best_loss = float('inf')

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.batch_train_losses = []
        self.batch_numbers = []
        self.epoch_boundaries = []
        self.total_batches = 0

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

        self.setup_model()

        if args.resume:
            self.load_checkpoint(args.resume)

    def setup_model(self):
        """Initialize model per paper specifications"""
        self.model = TrackNet(
            input_frames=DATASET_CONFIG["input_frames"],
            output_frames=DATASET_CONFIG["output_frames"]
        )

        # Paper-specified weight initialization
        self.model.apply(init_weights)
        self.model = self.model.to(self.device)

        # Paper-specified optimizer and loss
        self.criterion = WeightedBCELoss()
        self.optimizer = optim.Adadelta(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=CONFIG["scheduler_factor"],
            patience=CONFIG["scheduler_patience"]
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model parameters: {total_params:,}")

    def save_checkpoint(self, epoch, is_best=False, checkpoint_type="regular"):
        """Save checkpoint with comprehensive state"""
        checkpoint = {
            'epoch': epoch,
            'current_batch': self.current_batch,
            'total_batches': self.total_batches,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'batch_train_losses': self.batch_train_losses,
            'batch_numbers': self.batch_numbers,
            'epoch_boundaries': self.epoch_boundaries,
            'config': vars(self.args),
            'checkpoint_type': checkpoint_type,
            'save_time': time.time()
        }

        # Save checkpoints
        torch.save(checkpoint, self.checkpoints_dir / f'epoch_{epoch:03d}.pth')
        torch.save(checkpoint, self.checkpoints_dir / 'latest.pth')

        if is_best:
            torch.save(checkpoint, self.checkpoints_dir / 'best.pth')
            self.logger.info(f"✓ Best model Epoch {epoch}: {self.best_loss:.6f}")

        # Generate plots
        try:
            if len(self.batch_train_losses) > 0:
                self.plot_curves(epoch)
                self.logger.info(f"✓ Training curves updated")
        except Exception as e:
            self.logger.warning(f"⚠️ Plot generation failed: {e}")

    def save_emergency_checkpoint(self, save_path):
        """Emergency save with full state preservation"""
        checkpoint = {
            'epoch': self.current_epoch,
            'current_batch': self.current_batch,
            'total_batches': self.total_batches,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'batch_train_losses': self.batch_train_losses,
            'batch_numbers': self.batch_numbers,
            'epoch_boundaries': self.epoch_boundaries,
            'config': vars(self.args),
            'checkpoint_type': "emergency",
            'save_time': time.time()
        }

        torch.save(checkpoint, save_path)
        torch.save(checkpoint, self.checkpoints_dir / 'latest.pth')

        # Generate emergency plots
        try:
            if len(self.batch_train_losses) > 0:
                self.plot_curves(self.current_epoch)
                self.logger.info(f"✓ Emergency plots saved")
        except Exception as e:
            self.logger.warning(f"⚠️ Emergency plot failed: {e}")

        # Log emergency details
        self.logger.info(f"📊 Emergency save details:")
        self.logger.info(f"   - Checkpoint: {save_path}")
        self.logger.info(f"   - Epoch: {self.current_epoch}, Batch: {self.current_batch}")
        self.logger.info(f"   - Total batches: {self.total_batches}")
        if self.batch_train_losses:
            self.logger.info(f"   - Latest train loss: {self.batch_train_losses[-1]:.6f}")
        if self.val_losses:
            self.logger.info(f"   - Latest val loss: {self.val_losses[-1]:.6f}")
        self.logger.info(f"   - Best loss: {self.best_loss:.6f}")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint with full state restoration"""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.learning_rates = checkpoint['learning_rates']

        # Restore batch-level tracking
        self.batch_train_losses = checkpoint.get('batch_train_losses', [])
        self.batch_numbers = checkpoint.get('batch_numbers', [])
        self.epoch_boundaries = checkpoint.get('epoch_boundaries', [])
        self.total_batches = checkpoint.get('total_batches', 0)

        checkpoint_type = checkpoint.get('checkpoint_type', 'regular')
        if checkpoint_type == 'emergency':
            self.logger.info(f"✓ Resumed from emergency checkpoint: Epoch {self.start_epoch}")
        else:
            self.logger.info(f"✓ Resumed from Epoch {self.start_epoch}")

    def train_epoch(self, epoch, train_loader):
        """Train single epoch with batch tracking"""
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

            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

            self.optimizer.step()

            total_loss += loss.item()

            # Track batch-level metrics
            self.batch_train_losses.append(loss.item())
            self.batch_numbers.append(self.total_batches)

            pbar.set_postfix({'Loss': f'{loss.item():.6f}', 'Batch': self.total_batches})

        # Mark epoch boundary
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
        """Comprehensive training curves with batch and epoch levels"""
        if len(self.batch_train_losses) < 2:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Batch-level loss curve with epoch markers
        if self.batch_train_losses:
            ax1.plot(self.batch_numbers, self.batch_train_losses, 'b-', alpha=0.7,
                     linewidth=1, label='Training Loss (batch)')

            # Mark epoch boundaries
            for i, boundary in enumerate(self.epoch_boundaries):
                if boundary <= len(self.batch_numbers):
                    ax1.axvline(x=boundary, color='red', linestyle='--', alpha=0.6)
                    ax1.text(boundary, ax1.get_ylim()[1] * 0.95, f'E{i + 1}',
                             rotation=90, ha='right', va='top', fontsize=8, color='red')

            # Overlay epoch-level validation loss
            if self.val_losses:
                epoch_positions = self.epoch_boundaries[:len(self.val_losses)]
                ax1.plot(epoch_positions, self.val_losses, 'ro-', linewidth=2,
                         markersize=4, label='Validation Loss (epoch)', alpha=0.8)

        ax1.set_xlabel('Batch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves (Batch-level + Epoch Markers)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Learning rate schedule
        if self.learning_rates:
            epochs = range(1, len(self.learning_rates) + 1)
            ax2.plot(epochs, self.learning_rates, 'g-', linewidth=2, marker='o', markersize=3)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        filename = f'training_curves_epoch_{epoch:03d}_batch_{self.total_batches:05d}.png'
        plt.savefig(self.plots_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

    def train(self, train_dataset, val_dataset):
        """Main training loop with comprehensive state management"""
        self.training_in_progress = True

        # Setup data loaders
        loader_kwargs = {
            'batch_size': self.args.batch_size,
            'num_workers': self.device_config['num_workers'],
            'pin_memory': self.device_config['pin_memory'],
            'persistent_workers': self.device_config['persistent_workers'],
            'collate_fn': collate_fn,
            'drop_last': True
        }

        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

        self.logger.info(f"Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info("⚠️  Press Ctrl+C to safely stop training")

        patience_counter = 0
        start_time = time.time()

        try:
            for epoch in range(self.start_epoch, self.args.epochs):
                # Training and validation
                train_loss = self.train_epoch(epoch, train_loader)
                val_loss = self.validate(val_loader)

                # Update scheduler
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']

                # Record metrics
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.learning_rates.append(current_lr)

                # Check for improvement
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

                # Save checkpoint
                if epoch % self.args.save_interval == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)

                # Early stopping
                if patience_counter >= CONFIG["early_stop_patience"]:
                    self.logger.info(f"Early stopping at Epoch {epoch}")
                    break

        except KeyboardInterrupt:
            self.logger.info("\n⚠️  Training interrupted by user")
        except Exception as e:
            self.logger.error(f"❌ Training exception: {e}")
            try:
                exception_path = self.checkpoints_dir / f'exception_epoch_{self.current_epoch:03d}.pth'
                self.save_emergency_checkpoint(exception_path)
                self.logger.info(f"✓ Exception save: {exception_path}")
            except Exception as save_error:
                self.logger.error(f"❌ Exception save failed: {save_error}")
            raise
        finally:
            self.training_in_progress = False

        # Training completion summary
        total_time = time.time() - start_time
        self.logger.info("=" * 50)
        self.logger.info(f"Training completed! Duration: {total_time / 3600:.2f} hours")
        self.logger.info(f"Best validation loss: {self.best_loss:.6f}")
        self.logger.info("=" * 50)


def main():
    # Register signal handlers and cleanup
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_on_exit)

    parser = argparse.ArgumentParser(description='TrackNet Badminton Tracking Training - Fixed Version')

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

    # Resume training
    parser.add_argument('--resume', type=str, help='Checkpoint path to resume')

    args = parser.parse_args()

    # Setup directories and device
    save_dir, checkpoints_dir, plots_dir, logs_dir = setup_directories(args.save_dir)
    device, device_config = get_device_and_config()

    # Adjust batch size for limited memory
    if 'batch_multiplier' in device_config:
        args.batch_size = max(1, int(args.batch_size * device_config['batch_multiplier']))
        print(f"Batch size adjusted: {args.batch_size}")

    try:
        # Load and split dataset
        print(f"\nLoading dataset: {args.data_dir}")
        print("⭐ Using RAW data: No coordinate/pixel normalization in dataset")
        full_dataset = load_dataset(args.data_dir)

        total_size = len(full_dataset)
        train_size = int(CONFIG['train_split'] * total_size)
        indices = torch.randperm(total_size).tolist()

        train_dataset = Subset(full_dataset, indices[:train_size])
        val_dataset = Subset(full_dataset, indices[train_size:])

        print(f"Training: {len(train_dataset)}, Validation: {len(val_dataset)}")

        # Initialize trainer and start training
        global _trainer_instance
        trainer = Trainer(args, device, device_config, save_dir, checkpoints_dir, plots_dir, logs_dir)
        _trainer_instance = trainer

        trainer.train(train_dataset, val_dataset)

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

"""
New training with proper coordinate mapping:
python train_fixed.py --data_dir Dataset/Professional --save_dir checkpoints

Resume training:
python train_fixed.py --data_dir Dataset/Professional --resume checkpoints/checkpoints/latest.pth

Custom parameters:
python train_fixed.py --data_dir Dataset/Professional --batch_size 4 --epochs 50 --lr 1.0
"""