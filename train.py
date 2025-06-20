#!/usr/bin/env python3
"""
TrackNetV2 羽毛球追踪网络训练脚本 - 支持强制结束时自动保存
- 支持CUDA/MPS/CPU自动选择
- 支持从头训练和断点续训
- MIMO设计，每epoch自动保存
- 强制结束时自动保存当前模型
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

# 全局变量，用于信号处理
_trainer_instance = None

# 默认配置
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
    "save_interval": 1,  # 每epoch保存
}

DATASET_CONFIG = {
    "input_frames": 3,
    "output_frames": 3,  # MIMO设计
    "normalize_coords": True,
    "normalize_pixels": True,
    "video_ext": ".mp4",
    "csv_suffix": "_ball.csv"
}


def signal_handler(signum, frame):
    """信号处理函数 - 处理Ctrl+C等强制结束信号"""
    global _trainer_instance

    signal_names = {
        signal.SIGINT: "SIGINT (Ctrl+C)",
        signal.SIGTERM: "SIGTERM"
    }

    signal_name = signal_names.get(signum, f"Signal {signum}")
    print(f"\n⚠️  收到{signal_name}信号，正在安全保存模型...")

    if _trainer_instance is not None:
        try:
            # 保存紧急检查点
            emergency_path = _trainer_instance.save_dir / f'emergency_save_epoch_{_trainer_instance.current_epoch:03d}.pth'
            _trainer_instance.save_emergency_checkpoint(emergency_path)
            print(f"✓ 紧急保存完成: {emergency_path}")
        except Exception as e:
            print(f"❌ 紧急保存失败: {e}")

    print("🔄 进程安全退出")
    sys.exit(0)


def cleanup_on_exit():
    """程序退出时的清理函数"""
    global _trainer_instance
    if _trainer_instance is not None and hasattr(_trainer_instance, 'training_in_progress'):
        if _trainer_instance.training_in_progress:
            print("\n🔄 程序正常退出，执行最后保存...")
            try:
                exit_path = _trainer_instance.save_dir / f'exit_save_epoch_{_trainer_instance.current_epoch:03d}.pth'
                _trainer_instance.save_emergency_checkpoint(exit_path)
                print(f"✓ 退出保存完成: {exit_path}")
            except Exception as e:
                print(f"❌ 退出保存失败: {e}")


def get_device_and_config():
    """自动选择最佳设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        config = {"num_workers": 4, "pin_memory": True, "persistent_workers": True}
        print(f"✓ CUDA: {torch.cuda.get_device_name()}")

        # 根据显存调整批次
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if memory_gb < 8:
            config["batch_multiplier"] = 0.5

        # CUDA优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        config = {"num_workers": 2, "pin_memory": False, "persistent_workers": False}
        print("✓ MPS: Apple Silicon")

    else:
        device = torch.device('cpu')
        config = {"num_workers": 4, "pin_memory": False, "persistent_workers": True}
        print("⚠️ CPU模式")

    return device, config


def init_weights(m):
    """权重初始化 - 按论文要求使用uniform"""
    if isinstance(m, nn.Conv2d):
        nn.init.uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def create_gaussian_heatmap(x, y, visibility, height, width, radius=3.0):
    """生成高斯热图"""
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
    """数据批处理"""
    config = DEFAULT_CONFIG
    frames_list, heatmaps_list = [], []

    for frames, labels in batch:
        # 调整输入尺寸
        frames = F.interpolate(
            frames.unsqueeze(0),
            size=(config["input_height"], config["input_width"]),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        frames_list.append(frames)

        # 生成热图
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
    """加载数据集"""
    data_dir = Path(data_dir)
    match_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('match')])

    if not match_dirs:
        raise ValueError(f"未找到match文件夹: {data_dir}")

    combined_dataset = None
    for match_dir in match_dirs:
        try:
            dataset = BallTrackingDataset(str(match_dir), config=DATASET_CONFIG)
            if len(dataset) > 0:
                combined_dataset = dataset if combined_dataset is None else combined_dataset + dataset
                print(f"✓ {match_dir.name}: {len(dataset)} 样本")
        except Exception as e:
            print(f"✗ {match_dir.name} 加载失败: {e}")

    if combined_dataset is None:
        raise ValueError("无可用数据集")

    print(f"总计: {len(combined_dataset)} 样本")
    return combined_dataset


class Trainer:
    def __init__(self, args, device, device_config):
        self.args = args
        self.device = device
        self.device_config = device_config

        # 创建目录
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # 训练状态追踪
        self.current_epoch = 0
        self.current_batch = 0
        self.training_in_progress = False

        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.save_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # 训练状态
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        # 初始化模型
        self.setup_model()

        # 加载检查点
        if args.resume:
            self.load_checkpoint(args.resume)

    def setup_model(self):
        """初始化模型和优化器"""
        self.model = TrackNetV4()
        # MIMO输出
        self.model.conv2d_18 = nn.Conv2d(64, DATASET_CONFIG['output_frames'], 1)

        # 权重初始化
        self.model.apply(init_weights)
        self.model = self.model.to(self.device)

        # 损失函数和优化器
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

        # 统计参数
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"模型参数: {total_params:,}")

    def save_checkpoint(self, epoch, is_best=False, checkpoint_type="regular"):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'current_batch': self.current_batch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'config': vars(self.args),
            'checkpoint_type': checkpoint_type,
            'save_time': time.time()
        }

        # 保存最新
        torch.save(checkpoint, self.save_dir / f'epoch_{epoch:03d}.pth')
        torch.save(checkpoint, self.save_dir / 'latest.pth')

        # 保存最佳
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best.pth')
            self.logger.info(f"✓ 最佳模型 Epoch {epoch}: {self.best_loss:.6f}")

    def save_emergency_checkpoint(self, save_path):
        """保存紧急检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'current_batch': self.current_batch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'config': vars(self.args),
            'checkpoint_type': "emergency",
            'save_time': time.time()
        }

        torch.save(checkpoint, save_path)
        # 同时保存为最新检查点
        torch.save(checkpoint, self.save_dir / 'latest.pth')

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            self.logger.warning(f"检查点不存在: {checkpoint_path}")
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

        # 如果是紧急保存的检查点，显示特殊信息
        checkpoint_type = checkpoint.get('checkpoint_type', 'regular')
        if checkpoint_type == 'emergency':
            self.logger.info(f"✓ 从紧急保存的检查点恢复: Epoch {self.start_epoch}")
        else:
            self.logger.info(f"✓ 从Epoch {self.start_epoch}继续训练")

    def train_epoch(self, epoch, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        self.current_epoch = epoch

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            self.current_batch = batch_idx

            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()

            # 梯度裁剪
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="验证"):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def plot_curves(self, epoch):
        """绘制训练曲线"""
        if len(self.train_losses) < 2:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        epochs = range(1, len(self.train_losses) + 1)

        # 损失曲线
        ax1.plot(epochs, self.train_losses, 'b-', label='训练', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='验证', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('损失曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 学习率曲线
        ax2.plot(epochs, self.learning_rates, 'g-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('学习率变化')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / f'curves_epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()

    def train(self, train_dataset, val_dataset):
        """主训练循环"""
        # 标记训练开始
        self.training_in_progress = True

        # 数据加载器
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

        self.logger.info(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
        self.logger.info(f"设备: {self.device}")
        self.logger.info("⚠️  按Ctrl+C可安全停止训练并自动保存模型")

        # 早停计数器
        patience_counter = 0
        start_time = time.time()

        try:
            for epoch in range(self.start_epoch, self.args.epochs):
                # 训练和验证
                train_loss = self.train_epoch(epoch, train_loader)
                val_loss = self.validate(val_loader)

                # 更新学习率
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']

                # 记录指标
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.learning_rates.append(current_lr)

                # 检查最佳模型
                is_best = val_loss < self.best_loss
                if is_best:
                    self.best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                # 记录进度
                self.logger.info(
                    f"Epoch {epoch:03d}: 训练={train_loss:.6f}, "
                    f"验证={val_loss:.6f}, LR={current_lr:.2e}"
                    f"{' [BEST]' if is_best else ''}"
                )

                # 保存检查点和图表
                if epoch % self.args.save_interval == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)
                    self.plot_curves(epoch)

                # 早停检查
                if patience_counter >= DEFAULT_CONFIG["early_stop_patience"]:
                    self.logger.info(f"早停触发，Epoch {epoch}")
                    break

        except KeyboardInterrupt:
            self.logger.info("\n⚠️  收到键盘中断信号")
            # 这里的保存由信号处理器处理

        except Exception as e:
            self.logger.error(f"❌ 训练过程中出现异常: {e}")
            # 保存异常时的检查点
            try:
                exception_path = self.save_dir / f'exception_save_epoch_{self.current_epoch:03d}.pth'
                self.save_emergency_checkpoint(exception_path)
                self.logger.info(f"✓ 异常保存完成: {exception_path}")
            except Exception as save_error:
                self.logger.error(f"❌ 异常保存失败: {save_error}")
            raise

        finally:
            # 标记训练结束
            self.training_in_progress = False

        # 训练完成
        total_time = time.time() - start_time
        self.logger.info("=" * 50)
        self.logger.info(f"训练完成! 用时: {total_time / 3600:.2f}小时")
        self.logger.info(f"最佳验证损失: {self.best_loss:.6f}")
        self.logger.info("=" * 50)


def main():
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # 终止信号

    # 注册退出清理函数
    atexit.register(cleanup_on_exit)

    parser = argparse.ArgumentParser(description='TrackNetV2 羽毛球追踪训练')

    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True, help='数据集目录')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='保存目录')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1.0, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='权重衰减')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪')
    parser.add_argument('--save_interval', type=int, default=1, help='保存间隔')

    # 续训参数
    parser.add_argument('--resume', type=str, help='继续训练的检查点路径')

    args = parser.parse_args()

    # 获取设备
    device, device_config = get_device_and_config()

    # 根据设备调整批次大小
    if 'batch_multiplier' in device_config:
        args.batch_size = max(1, int(args.batch_size * device_config['batch_multiplier']))
        print(f"批次大小调整为: {args.batch_size}")

    try:
        # 加载数据集
        print(f"\n加载数据集: {args.data_dir}")
        full_dataset = load_dataset(args.data_dir)

        # 分割数据集
        total_size = len(full_dataset)
        train_size = int(DEFAULT_CONFIG['train_split'] * total_size)
        indices = torch.randperm(total_size).tolist()

        train_dataset = Subset(full_dataset, indices[:train_size])
        val_dataset = Subset(full_dataset, indices[train_size:])

        print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

        # 设置全局trainer实例（用于信号处理）
        global _trainer_instance
        trainer = Trainer(args, device, device_config)
        _trainer_instance = trainer

        # 开始训练
        trainer.train(train_dataset, val_dataset)

    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

    """
    新模型训练：python train.py --data_dir Dataset/Professional --save_dir checkpoints
    继续训练：python train.py --data_dir Dataset/Professional --resume checkpoints/latest.pth
    全参数训练：python train.py --data_dir Dataset/Professional --save_dir checkpoints --batch_size 2 --epochs 30 --lr 1.0 --weight_decay 0.0 --grad_clip 1.0 --save_interval 1
    
    强制结束时会自动保存模型到以下位置：
    - emergency_save_epoch_XXX.pth (Ctrl+C或SIGTERM信号)
    - exception_save_epoch_XXX.pth (程序异常)
    - exit_save_epoch_XXX.pth (正常退出)
    - latest.pth (总是更新为最新状态)
    """
