import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import time
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from tracknet import TrackNet, WeightedBCELoss
from dataset_controller.ball_tracking_data_reader import BallTrackingDataset

# ======================== TrackNetV2 论文配置参数 ========================
TRAINING_CONFIG = {
    "training": {
        "batch_size": 2,  # 根据论文，小批次保证稳定训练
        "num_epochs": 30,  # 论文中使用30个epoch
        "learning_rate": 1.0,  # 论文中Adadelta使用lr=1.0
        "weight_decay": 0.0,  # 论文中未提及，保持为0
        "gradient_clip_value": 1.0,
        "tolerance_variable": 4  # 论文中的tolerance variable
    },
    "model": {
        "input_height": 288,  # 论文中从640×360降至512×288
        "input_width": 512,
        "heatmap_radius": 3,  # 高斯热图半径
        "detection_threshold": 0.5,  # 论文中使用0.5作为阈值
        "distance_threshold": 4  # 论文中使用4像素作为距离阈值
    },
    "optimization": {
        "scheduler_factor": 0.5,
        "scheduler_patience": 8,
        "min_lr": 1e-6
    },
    "early_stopping": {
        "enabled": True,
        "patience": 15,
        "min_delta": 1e-6
    },
    "logging": {
        "save_interval": 5,
        "plot_interval": 5,
        "log_level": "INFO"
    },
    "paths": {
        "save_dir": "checkpoints",
        "log_dir": "logs"
    },
    "data": {
        "num_workers": 2,  # MPS设备建议较少workers
        "pin_memory": False,  # MPS设备不支持pin_memory
        "persistent_workers": False,  # MPS设备建议关闭
        "train_split": 0.8
    }
}

# 论文中的数据集配置
DATASET_CONFIGS = {
    "3in3out": {  # 论文中的MIMO设计
        "input_frames": 3,
        "output_frames": 3,
        "normalize_coords": True,
        "normalize_pixels": True,
        "video_ext": ".mp4",
        "csv_suffix": "_ball.csv"
    },
    "3in1out": {  # 传统MISO设计作为对比
        "input_frames": 3,
        "output_frames": 1,
        "normalize_coords": True,
        "normalize_pixels": True,
        "video_ext": ".mp4",
        "csv_suffix": "_ball.csv"
    }
}


def setup_logging(log_dir: Path, config_name: str) -> logging.Logger:
    """设置日志系统"""
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(f'tracknetv2_{config_name}')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # 文件handler
    file_handler = logging.FileHandler(log_dir / f'training_{config_name}.log')
    file_handler.setLevel(logging.INFO)

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_device() -> torch.device:
    """获取最佳可用设备，处理MPS兼容性"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name()
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ 使用CUDA设备: {device_name} ({memory:.1f}GB)")
        return device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✓ 使用MPS设备 (Apple Silicon)")
        print("  注意: MPS设备将自动禁用pin_memory和persistent_workers")
        return device
    else:
        device = torch.device('cpu')
        print("✓ 使用CPU设备")
        return device


def load_all_matches(professional_dir: Path, config: Dict) -> BallTrackingDataset:
    """加载所有match文件夹并合并数据集"""
    professional_dir = Path(professional_dir)
    match_dirs = sorted([
        d for d in professional_dir.iterdir()
        if d.is_dir() and d.name.startswith('match')
    ])

    if not match_dirs:
        raise ValueError(f"在 {professional_dir} 中未找到match文件夹")

    combined_dataset = None
    total_samples = 0

    print(f"正在加载数据集...")
    for match_dir in match_dirs:
        try:
            dataset = BallTrackingDataset(str(match_dir), config=config)
            if len(dataset) > 0:
                if combined_dataset is None:
                    combined_dataset = dataset
                else:
                    combined_dataset = combined_dataset + dataset
                total_samples += len(dataset)
                print(f"  ✓ {match_dir.name}: {len(dataset)} 个样本")
        except Exception as e:
            print(f"  ✗ {match_dir.name} 加载失败: {e}")

    if combined_dataset is None:
        raise ValueError("没有成功加载任何数据集")

    print(f"数据集加载完成，总计: {total_samples} 个样本")
    return combined_dataset


def create_gaussian_heatmap(
        x: float, y: float, visibility: float,
        height: int, width: int, radius: float = 3.0
) -> torch.Tensor:
    """按照论文创建高斯热图（实值2D数组而非one-hot编码）"""
    heatmap = torch.zeros(height, width, dtype=torch.float32)

    if visibility < 0.5:
        return heatmap

    # 计算像素坐标
    x_pixel = max(0, min(width - 1, int(x * width)))
    y_pixel = max(0, min(height - 1, int(y * height)))

    # 优化计算：仅在有效区域计算高斯值
    kernel_size = int(3 * radius)
    x_min = max(0, x_pixel - kernel_size)
    x_max = min(width, x_pixel + kernel_size + 1)
    y_min = max(0, y_pixel - kernel_size)
    y_max = min(height, y_pixel + kernel_size + 1)

    if x_max <= x_min or y_max <= y_min:
        return heatmap

    # 在有效区域生成高斯分布
    y_coords, x_coords = torch.meshgrid(
        torch.arange(y_min, y_max, dtype=torch.float32),
        torch.arange(x_min, x_max, dtype=torch.float32),
        indexing='ij'
    )

    dist_sq = (x_coords - x_pixel) ** 2 + (y_coords - y_pixel) ** 2
    gaussian_values = torch.exp(-dist_sq / (2 * radius ** 2))

    # 论文中提到的阈值处理
    gaussian_values[gaussian_values < 0.01] = 0

    heatmap[y_min:y_max, x_min:x_max] = gaussian_values

    return heatmap


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
    """TrackNetV2数据整理函数"""
    config = TRAINING_CONFIG["model"]
    target_height = config["input_height"]
    target_width = config["input_width"]
    radius = config["heatmap_radius"]

    frames_list = []
    heatmaps_list = []

    for frames, labels in batch:
        # 调整输入尺寸到论文指定的512×288
        frames = frames.unsqueeze(0)
        frames_resized = F.interpolate(
            frames,
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=False,
            antialias=True
        )
        frames_resized = frames_resized.squeeze(0)
        frames_list.append(frames_resized)

        # 按照论文生成实值热图
        num_frames = len(labels)
        heatmaps = torch.zeros(num_frames, target_height, target_width, dtype=torch.float32)

        for i, label_dict in enumerate(labels):
            if isinstance(label_dict, dict):
                x = label_dict['x'].item()
                y = label_dict['y'].item()
                visibility = label_dict['visibility'].item()

                heatmap = create_gaussian_heatmap(
                    x, y, visibility, target_height, target_width, radius
                )
                heatmaps[i] = heatmap

        heatmaps_list.append(heatmaps)

    return torch.stack(frames_list), torch.stack(heatmaps_list)


class WeightedBCELossV2(nn.Module):
    """论文中的加权二值交叉熵损失函数"""

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        实现论文中的WBCE损失函数:
        WBCE = -Σ[(1-w)²*y_true*log(y_pred) + w²*(1-y_true)*log(1-y_pred)]
        其中 w = y_true (ground truth标签)
        """
        # 防止log(0)
        eps = 1e-7
        y_pred = torch.clamp(y_pred, eps, 1 - eps)

        # 论文中的权重系数 w = y_true
        w = y_true

        # 计算加权交叉熵
        term1 = (1 - w) ** 2 * y_true * torch.log(y_pred)
        term2 = w ** 2 * (1 - y_true) * torch.log(1 - y_pred)

        loss = -(term1 + term2)

        return loss.mean()


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 15, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


class MetricsTracker:
    """训练指标跟踪器"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def update(self, train_loss: float, val_loss: float, lr: float, epoch: int):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(lr)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch

    def get_summary(self) -> Dict:
        return {
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'total_epochs': len(self.train_losses)
        }


class TrackNetV2Trainer:
    """TrackNetV2训练器（严格按照论文实现）"""

    def __init__(self, config_name: str, config: Dict = None):
        self.config = config or TRAINING_CONFIG
        self.config_name = config_name
        self.device = get_device()

        # 设置日志
        self.logger = setup_logging(self.log_dir, config_name)

        # 处理MPS设备特殊配置
        self._configure_for_device()

        # 创建目录
        self.save_dir = Path(self.config["paths"]["save_dir"])
        self.log_dir = Path(self.config["paths"]["log_dir"])
        self.save_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

        # 初始化组件
        self.metrics = MetricsTracker()
        self.early_stopping = EarlyStopping(
            patience=self.config['early_stopping']['patience'],
            min_delta=self.config['early_stopping']['min_delta']
        )

        # 设置模型
        self.setup_model()

        # 保存配置
        self.save_config()

        self.logger.info("TrackNetV2训练器初始化完成")
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"配置: {config_name}")

    def _configure_for_device(self):
        """根据设备类型配置参数"""
        if self.device.type == 'mps':
            # MPS设备不支持某些功能
            self.config['data']['pin_memory'] = False
            self.config['data']['persistent_workers'] = False
            # 警告用户
            warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*")
            self.logger.info("MPS设备检测到，已自动禁用pin_memory和persistent_workers")
        elif self.device.type == 'cuda':
            # CUDA设备启用性能优化
            self.config['data']['pin_memory'] = True
            self.config['data']['persistent_workers'] = True
            self.config['data']['num_workers'] = min(4, self.config['data']['num_workers'])

    def save_config(self):
        """保存训练配置"""
        config_path = self.log_dir / f'config_{self.config_name}.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'training_config': self.config,
                'dataset_config': DATASET_CONFIGS[self.config_name],
                'paper_reference': 'TrackNetV2: Efficient Shuttlecock Tracking Network (ICPAI 2020)'
            }, f, indent=2, ensure_ascii=False)

    def setup_model(self):
        """按照论文设置模型和优化器"""
        dataset_config = DATASET_CONFIGS[self.config_name]

        # 初始化TrackNet模型
        self.model = TrackNet()

        # 根据输出帧数调整最后一层（MIMO设计）
        if dataset_config['output_frames'] != 3:
            # 替换最后的卷积层以支持不同的输出帧数
            self.model.conv2d_18 = nn.Conv2d(64, dataset_config['output_frames'], 1)

        self.model = self.model.to(self.device)

        # 模型参数统计
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.logger.info(f"模型总参数: {total_params:,}")
        self.logger.info(f"可训练参数: {trainable_params:,}")

        # 使用论文中的加权BCE损失
        self.criterion = WeightedBCELossV2()

        # 论文中使用Adadelta优化器，lr=1.0
        self.optimizer = optim.Adadelta(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config['optimization']['scheduler_factor'],
            patience=self.config['optimization']['scheduler_patience'],
            min_lr=self.config['optimization']['min_lr'],
            verbose=True
        )

        self.logger.info("模型设置完成")
        self.logger.info(f"优化器: Adadelta (lr={self.config['training']['learning_rate']})")
        self.logger.info(f"损失函数: 加权二值交叉熵 (WBCE)")

    def train_epoch(self, epoch: int, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        progress_bar = tqdm(
            train_loader,
            desc=f"训练 Epoch {epoch + 1}/{self.config['training']['num_epochs']}",
            leave=False
        )

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # 确保输出经过sigmoid激活（论文中强调）
            if not hasattr(self.model, 'final_activation_applied'):
                outputs = torch.sigmoid(outputs)

            loss = self.criterion(outputs, targets)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if self.config['training']['gradient_clip_value'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip_value']
                )

            self.optimizer.step()

            total_loss += loss.item()

            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg': f'{total_loss / (batch_idx + 1):.6f}'
            })

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate_epoch(self, val_loader: DataLoader) -> float:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="验证", leave=False)
            for inputs, targets in progress_bar:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(inputs)

                # 确保输出经过sigmoid激活
                if not hasattr(self.model, 'final_activation_applied'):
                    outputs = torch.sigmoid(outputs)

                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                progress_bar.set_postfix({'Loss': f'{loss.item():.6f}'})

        avg_loss = total_loss / num_batches
        return avg_loss

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': self.metrics.__dict__,
            'config_name': self.config_name,
            'config': self.config,
            'paper_info': 'TrackNetV2: Efficient Shuttlecock Tracking Network'
        }

        # 保存最新模型
        latest_path = self.save_dir / f'latest_tracknetv2_{self.config_name}.pth'
        torch.save(checkpoint, latest_path)

        # 保存最佳模型
        if is_best:
            best_path = self.save_dir / f'best_tracknetv2_{self.config_name}.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"✓ 保存最佳模型! 验证损失: {self.metrics.best_val_loss:.6f}")

    def plot_training_curves(self):
        """绘制训练曲线"""
        if len(self.metrics.train_losses) < 2:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'TrackNetV2训练过程 - {self.config_name}', fontsize=16)

        epochs = range(1, len(self.metrics.train_losses) + 1)

        # 损失曲线
        axes[0, 0].plot(epochs, self.metrics.train_losses, 'b-', label='训练损失', linewidth=2)
        axes[0, 0].plot(epochs, self.metrics.val_losses, 'r-', label='验证损失', linewidth=2)
        axes[0, 0].axvline(x=self.metrics.best_epoch + 1, color='g', linestyle='--', alpha=0.7, label='最佳模型')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('WBCE Loss')
        axes[0, 0].set_title('加权二值交叉熵损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 学习率曲线
        if self.metrics.learning_rates:
            axes[0, 1].plot(epochs, self.metrics.learning_rates, 'g-', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_title('Adadelta学习率变化')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)

        # 最近epochs的损失
        if len(epochs) > 20:
            recent_epochs = 20
            recent_range = epochs[-recent_epochs:]
            axes[1, 0].plot(recent_range, self.metrics.train_losses[-recent_epochs:], 'b-', label='训练', linewidth=2)
            axes[1, 0].plot(recent_range, self.metrics.val_losses[-recent_epochs:], 'r-', label='验证', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title(f'最近{recent_epochs}轮进展')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 收敛分析
        if len(self.metrics.val_losses) > 5:
            # 计算移动平均
            window = min(5, len(self.metrics.val_losses))
            moving_avg = np.convolve(self.metrics.val_losses, np.ones(window) / window, mode='valid')
            moving_epochs = epochs[window - 1:]
            axes[1, 1].plot(epochs, self.metrics.val_losses, 'r-', alpha=0.3, label='原始验证损失')
            axes[1, 1].plot(moving_epochs, moving_avg, 'r-', linewidth=2, label=f'{window}点移动平均')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Validation Loss')
            axes[1, 1].set_title('收敛分析')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片
        plot_path = self.log_dir / f'training_curves_tracknetv2_{self.config_name}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    def train(self, train_dataset, val_dataset):
        """主训练循环"""
        # 数据加载器配置
        data_config = self.config['data']

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=data_config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=data_config['pin_memory'],
            persistent_workers=data_config['persistent_workers'],
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=data_config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=data_config['pin_memory'],
            persistent_workers=data_config['persistent_workers'],
            drop_last=True
        )

        # 记录训练信息
        self.logger.info("=" * 60)
        self.logger.info("开始TrackNetV2训练")
        self.logger.info("=" * 60)
        self.logger.info(f"训练集大小: {len(train_dataset)}")
        self.logger.info(f"验证集大小: {len(val_dataset)}")
        self.logger.info(f"训练批次数: {len(train_loader)}")
        self.logger.info(f"验证批次数: {len(val_loader)}")
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"配置: {self.config_name}")
        self.logger.info(f"输入尺寸: {self.config['model']['input_width']}×{self.config['model']['input_height']}")
        self.logger.info(f"目标epoch数: {self.config['training']['num_epochs']}")

        start_time = time.time()

        for epoch in range(self.config['training']['num_epochs']):
            epoch_start_time = time.time()

            # 训练和验证
            train_loss = self.train_epoch(epoch, train_loader)
            val_loss = self.validate_epoch(val_loader)

            # 学习率调度
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # 更新指标
            self.metrics.update(train_loss, val_loss, current_lr, epoch)

            # 计算epoch时间
            epoch_time = time.time() - epoch_start_time

            # 检查是否是最佳模型
            is_best = val_loss < self.metrics.best_val_loss

            # 记录进度
            self.logger.info(
                f"Epoch {epoch + 1:3d}/{self.config['training']['num_epochs']}: "
                f"训练={train_loss:.6f}, 验证={val_loss:.6f}, "
                f"LR={current_lr:.2e}, 时间={epoch_time:.1f}s"
                f"{' [BEST]' if is_best else ''}"
            )

            # 保存检查点
            if (epoch + 1) % self.config['logging']['save_interval'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

            # 绘制曲线
            if (epoch + 1) % self.config['logging']['plot_interval'] == 0:
                self.plot_training_curves()

            # 早停检查
            if self.config['early_stopping']['enabled']:
                if self.early_stopping(val_loss):
                    self.logger.info(f"早停触发! 在第 {epoch + 1} 轮停止训练")
                    break

        # 训练完成
        total_time = time.time() - start_time
        summary = self.metrics.get_summary()

        self.logger.info("=" * 60)
        self.logger.info("TrackNetV2训练完成!")
        self.logger.info("=" * 60)
        self.logger.info(f"总用时: {total_time / 3600:.2f} 小时")
        self.logger.info(f"最佳验证损失: {summary['best_val_loss']:.6f} (Epoch {summary['best_epoch'] + 1})")
        self.logger.info(f"最终训练损失: {summary['final_train_loss']:.6f}")
        self.logger.info(f"最终验证损失: {summary['final_val_loss']:.6f}")

        # 论文性能对比信息
        self.logger.info("\n论文TrackNetV2性能指标:")
        self.logger.info("- 训练集: 准确率96.3%, 精确度97.0%, 召回率98.7%")
        self.logger.info("- 测试集: 准确率85.2%, 精确度97.2%, 召回率85.4%")
        self.logger.info("- 处理速度: 31.84 FPS (3-in-3-out)")
        self.logger.info("=" * 60)

        # 最终保存
        self.save_checkpoint(epoch, False)
        self.plot_training_curves()

        return summary


def main():
    """主函数"""
    print("=" * 70)
    print("TrackNetV2: Efficient Shuttlecock Tracking Network")
    print("基于论文: TrackNetV2 (ICPAI 2020)")
    print("=" * 70)

    # 获取数据集路径
    base_dir = Path(__file__).resolve().parent
    professional_dir = base_dir / 'Dataset' / 'Professional'

    if not professional_dir.exists():
        print(f"❌ 错误: 数据集目录不存在: {professional_dir}")
        return

    # 选择配置
    print("\n可用配置 (基于论文设计):")
    print("1. 3in3out: MIMO设计 - 3输入帧 -> 3输出帧 (论文推荐，性能最佳)")
    print("2. 3in1out: MISO设计 - 3输入帧 -> 1输出帧 (传统设计对比)")

    while True:
        try:
            choice = input(f"\n请选择配置 (1-2): ").strip()
            if choice == "1":
                config_name = "3in3out"
                break
            elif choice == "2":
                config_name = "3in1out"
                break
            else:
                print("无效输入，请输入1或2")
        except KeyboardInterrupt:
            print("\n训练取消")
            return

    print(f"\n✓ 已选择配置: {config_name}")
    if config_name == "3in3out":
        print("  📊 MIMO设计将显著提升处理速度（论文中从2.6 FPS提升到31.8 FPS）")

    try:
        # 加载数据集
        print(f"\n正在加载数据集...")
        dataset_config = DATASET_CONFIGS[config_name]
        full_dataset = load_all_matches(professional_dir, dataset_config)

        # 分割数据集
        total_size = len(full_dataset)
        train_size = int(TRAINING_CONFIG['data']['train_split'] * total_size)
        val_size = total_size - train_size

        # 创建随机分割
        indices = torch.randperm(total_size).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

        print(f"\n✓ 数据集分割完成:")
        print(f"  训练集: {len(train_dataset)} 样本 ({len(train_dataset) / total_size:.1%})")
        print(f"  验证集: {len(val_dataset)} 样本 ({len(val_dataset) / total_size:.1%})")
        print(f"  论文数据集: 55,563帧来自18个羽毛球比赛视频")

        # 初始化训练器
        print(f"\n正在初始化TrackNetV2训练器...")
        trainer = TrackNetV2Trainer(config_name)

        print(f"\n🚀 开始训练...")
        summary = trainer.train(train_dataset, val_dataset)

        print(f"\n✅ 训练成功完成!")
        print(f"📁 最佳模型保存在: {trainer.save_dir / f'best_tracknetv2_{config_name}.pth'}")
        print(f"📊 训练日志保存在: {trainer.log_dir}")

    except KeyboardInterrupt:
        print(f"\n⏹ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
