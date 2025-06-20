import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from tracknet import TrackNet, WeightedBCELoss
from dataset_controller.ball_tracking_data_reader import BallTrackingDataset

# ======================== 配置参数 ========================
TRAINING_CONFIG = {
    "training": {
        "batch_size": 2,
        "num_epochs": 30,
        "learning_rate": 1.0,
        "weight_decay": 0.0
    },
    "model": {
        "input_height": 288,
        "input_width": 512,
        "heatmap_radius": 3,
        "detection_threshold": 0.5
    },
    "optimization": {
        "scheduler_factor": 0.5,
        "scheduler_patience": 5
    },
    "early_stopping": {
        "enabled": True,
        "patience": 15
    },
    "logging": {
        "save_interval": 10,
        "plot_interval": 5
    },
    "paths": {
        "save_dir": "checkpoints",
        "log_dir": "logs"
    }
}

# Dataset configurations
DATASET_CONFIGS = {
    "3in3out": {
        "input_frames": 3,
        "output_frames": 3,
        "normalize_coords": False,
        "normalize_pixels": False,
        "video_ext": ".mp4",
        "csv_suffix": "_ball.csv"
    },
    "3in1out": {
        "input_frames": 3,
        "output_frames": 1,
        "normalize_coords": False,
        "normalize_pixels": False,
        "video_ext": ".mp4",
        "csv_suffix": "_ball.csv"
    }
}


# =========================================================


def load_all_matches(professional_dir, config):
    """加载professional文件夹中所有match文件夹并拼接"""
    professional_dir = Path(professional_dir)
    match_dirs = sorted([d for d in professional_dir.iterdir()
                         if d.is_dir() and d.name.startswith('match')])

    if not match_dirs:
        raise ValueError("未找到match文件夹")

    combined_dataset = None
    for match_dir in match_dirs:
        try:
            dataset = BallTrackingDataset(str(match_dir), config=config)
            if len(dataset) > 0:
                if combined_dataset is None:
                    combined_dataset = dataset
                else:
                    combined_dataset = combined_dataset + dataset
                print(f"已添加 {match_dir.name}: {len(dataset)} 个样本")
        except Exception as e:
            print(f"加载 {match_dir.name} 时出错: {e}")

    return combined_dataset


def create_gaussian_heatmap(x, y, visibility, height, width, radius=3):
    """创建高斯热图"""
    heatmap = torch.zeros(height, width)
    if visibility < 0.5:
        return heatmap

    x_pixel = int(x * width)
    y_pixel = int(y * height)
    x_pixel = max(0, min(width - 1, x_pixel))
    y_pixel = max(0, min(height - 1, y_pixel))

    y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    dist_sq = (x_coords - x_pixel) ** 2 + (y_coords - y_pixel) ** 2
    heatmap = torch.exp(-dist_sq / (2 * radius ** 2))
    heatmap[heatmap < 0.01] = 0

    return heatmap


def collate_fn(batch):
    """数据整理函数：720×1280 -> 288×512"""
    config = TRAINING_CONFIG["model"]
    target_height = config["input_height"]
    target_width = config["input_width"]

    frames_list = []
    heatmaps_list = []

    for frames, labels in batch:
        # 调整输入尺寸
        frames = frames.unsqueeze(0)
        frames_resized = F.interpolate(frames, size=(target_height, target_width),
                                       mode='bilinear', align_corners=False)
        frames_resized = frames_resized.squeeze(0)
        frames_list.append(frames_resized)

        # 生成热图
        num_frames = len(labels)
        heatmaps = torch.zeros(num_frames, target_height, target_width)

        for i, label_dict in enumerate(labels):
            if isinstance(label_dict, dict):
                x = label_dict['x'].item()
                y = label_dict['y'].item()
                visibility = label_dict['visibility'].item()

                heatmap = create_gaussian_heatmap(x, y, visibility,
                                                  target_height, target_width,
                                                  config["heatmap_radius"])
                heatmaps[i] = heatmap

        heatmaps_list.append(heatmaps)

    return torch.stack(frames_list), torch.stack(heatmaps_list)


class Trainer:
    def __init__(self, config_name):
        self.config = TRAINING_CONFIG
        self.config_name = config_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建目录
        Path(self.config["paths"]["save_dir"]).mkdir(exist_ok=True)
        Path(self.config["paths"]["log_dir"]).mkdir(exist_ok=True)

        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0

        # 设置模型
        self.setup_model()

    def setup_model(self):
        """设置模型和优化器"""
        dataset_config = DATASET_CONFIGS[self.config_name]

        self.model = TrackNet()
        if dataset_config['output_frames'] != 3:
            self.model.conv2d_18 = nn.Conv2d(64, dataset_config['output_frames'], 1)
        self.model = self.model.to(self.device)

        self.criterion = WeightedBCELoss()
        self.optimizer = optim.Adadelta(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config['optimization']['scheduler_factor'],
            patience=self.config['optimization']['scheduler_patience'],
        )

    def train_epoch(self, epoch, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for inputs, targets in progress_bar:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.6f}'})

        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate_epoch(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="验证"):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss

    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config_name': self.config_name
        }

        save_dir = Path(self.config["paths"]["save_dir"])
        torch.save(checkpoint, save_dir / f'latest_{self.config_name}.pth')

        if is_best:
            torch.save(checkpoint, save_dir / f'best_{self.config_name}.pth')
            print(f"保存最佳模型! 验证损失: {self.best_val_loss:.6f}")

    def plot_losses(self):
        """绘制损失曲线"""
        if len(self.train_losses) < 2:
            return

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='训练损失', linewidth=2)
        plt.plot(self.val_losses, label='验证损失', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'TrackNetV2 - {self.config_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        if len(self.train_losses) > 10:
            recent = 20
            epochs = range(len(self.train_losses) - recent, len(self.train_losses))
            plt.plot(epochs, self.train_losses[-recent:], label='训练', linewidth=2)
            plt.plot(epochs, self.val_losses[-recent:], label='验证', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('最近进度')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        log_dir = Path(self.config["paths"]["log_dir"])
        plt.savefig(log_dir / f'loss_{self.config_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def train(self, train_dataset, val_dataset):
        """主训练循环"""
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
        print(f"设备: {self.device}, 配置: {self.config_name}")
        print(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")

        start_time = time.time()

        for epoch in range(self.config['training']['num_epochs']):
            # 训练和验证
            train_loss = self.train_epoch(epoch, train_loader)
            val_loss = self.validate_epoch(val_loader)

            # 学习率调度
            self.scheduler.step(val_loss)

            # 打印进度
            print(f"Epoch {epoch + 1:3d}: 训练={train_loss:.6f}, 验证={val_loss:.6f}, "
                  f"LR={self.optimizer.param_groups[0]['lr']:.2e}")

            # 检查最佳模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            # 保存检查点
            if (epoch + 1) % self.config['logging']['save_interval'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

            # 绘制损失曲线
            if (epoch + 1) % self.config['logging']['plot_interval'] == 0:
                self.plot_losses()

            # 早停检查
            if (self.config['early_stopping']['enabled'] and
                    self.early_stop_counter >= self.config['early_stopping']['patience']):
                print(f"早停触发! 第 {epoch + 1} 轮停止")
                break

        total_time = time.time() - start_time
        print(f"训练完成! 用时: {total_time / 3600:.2f}h, 最佳验证损失: {self.best_val_loss:.6f}")
        self.save_checkpoint(epoch, False)
        self.plot_losses()


if __name__ == "__main__":
    # 加载数据集
    base_dir = Path(__file__).resolve().parent
    professional_dir = base_dir / 'Dataset' / 'Professional'

    # 选择配置
    print("选择配置: 1-3in3out, 2-3in1out")
    choice = input("输入选择 (1-2): ").strip()
    config_name = "3in3out" if choice == "1" else "3in1out"

    print(f"加载数据集 - 配置: {config_name}")
    dataset_config = DATASET_CONFIGS[config_name]
    full_dataset = load_all_matches(professional_dir, dataset_config)

    # 分割数据集 (80/20)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_dataset = torch.utils.data.Subset(full_dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(full_dataset, range(train_size, total_size))

    print(f"数据集分割完成: 训练={len(train_dataset)}, 验证={len(val_dataset)}")

    # 开始训练
    trainer = Trainer(config_name)
    trainer.train(train_dataset, val_dataset)
