import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import numpy as np
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# 导入模型和数据集
from tracknet import TrackNet, WeightedBCELoss, postprocess_heatmap
from dataset import BallTrackingDataset

# ======================== 根据论文的配置参数 ========================
TRAINING_CONFIG = {
    "dataset": {
        "base_dir": ".",
        "match_dir": "Dataset/Professional/match1",
        "input_height": 288,  # 论文中从640×360改为512×288
        "input_width": 512,
        "configs": {
            "3in3out": {
                "input_frames": 3,
                "output_frames": 3,  # MIMO设计: 3-in 3-out
                "normalize_coords": False,
                "normalize_pixels": False,
                "video_ext": ".mp4",
                "csv_suffix": "_ball.csv"
            },
            "3in1out": {
                "input_frames": 3,
                "output_frames": 1,  # MISO设计: 3-in 1-out
                "normalize_coords": False,
                "normalize_pixels": False,
                "video_ext": ".mp4",
                "csv_suffix": "_ball.csv"
            }
        }
    },
    "training": {
        "batch_size": 2,  # 根据论文和GPU内存调整
        "num_epochs": 30,  # 论文中使用30个epochs
        "learning_rate": 1.0,  # 论文中使用1.0
        "weight_decay": 0.0,
        "train_ratio": 0.8,
        "val_ratio": 0.2
    },
    "model": {
        "heatmap_radius": 3,  # 高斯热图半径
        "detection_threshold": 0.5,  # 论文中使用0.5阈值
        "tolerance_pixels": 4  # 论文中使用4像素容忍度
    },
    "optimization": {
        "optimizer": "Adadelta",  # 论文指定使用Adadelta
        "scheduler": {
            "type": "ReduceLROnPlateau",
            "mode": "min",
            "factor": 0.5,
            "patience": 5,
            "verbose": True
        }
    },
    "early_stopping": {
        "enabled": True,
        "patience": 15,
        "min_delta": 1e-4
    },
    "logging": {
        "save_interval": 10,
        "print_interval": 10,
        "plot_interval": 5
    },
    "paths": {
        "save_dir": "checkpoints",
        "log_dir": "logs"
    }
}


# ================================================================


def create_gaussian_heatmap(x, y, visibility, height, width, radius=3):
    """根据论文创建高斯热图 - 2D Gaussian distribution"""
    heatmap = torch.zeros(height, width)

    if visibility < 0.5:  # 球不可见
        return heatmap

    # 转换归一化坐标到像素坐标
    x_pixel = int(x * width)
    y_pixel = int(y * height)

    # 确保坐标在边界内
    x_pixel = max(0, min(width - 1, x_pixel))
    y_pixel = max(0, min(height - 1, y_pixel))

    # 创建高斯分布 - 论文中使用的amplified 2D Gaussian
    y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')

    # 计算距离平方
    dist_sq = (x_coords - x_pixel) ** 2 + (y_coords - y_pixel) ** 2

    # 生成高斯热图
    heatmap = torch.exp(-dist_sq / (2 * radius ** 2))

    # 阈值处理 - 论文中提到的处理方式
    heatmap[heatmap < 0.01] = 0

    return heatmap


def collate_fn(batch):
    """根据论文要求的数据处理: 720×1280 -> 288×512"""
    config = TRAINING_CONFIG["dataset"]
    target_height = config["input_height"]
    target_width = config["input_width"]

    frames_list = []
    heatmaps_list = []

    for frames, labels in batch:
        # 处理输入帧：调整尺寸到论文要求的512×288
        frames = frames.unsqueeze(0)  # [1, 9, H, W]
        frames_resized = F.interpolate(frames, size=(target_height, target_width),
                                       mode='bilinear', align_corners=False)
        frames_resized = frames_resized.squeeze(0)  # [9, 288, 512]
        frames_list.append(frames_resized)

        # 处理标签：从坐标字典转换为热图
        num_frames = len(labels)
        heatmaps = torch.zeros(num_frames, target_height, target_width)

        for i, label_dict in enumerate(labels):
            if isinstance(label_dict, dict):
                x = label_dict['x'].item()
                y = label_dict['y'].item()
                visibility = label_dict['visibility'].item()

                heatmap = create_gaussian_heatmap(x, y, visibility,
                                                  target_height, target_width,
                                                  TRAINING_CONFIG["model"]["heatmap_radius"])
                heatmaps[i] = heatmap

        heatmaps_list.append(heatmaps)

    batch_frames = torch.stack(frames_list)
    batch_heatmaps = torch.stack(heatmaps_list)

    return batch_frames, batch_heatmaps


class TrackNetV2Trainer:
    def __init__(self):
        self.config = TRAINING_CONFIG
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建保存目录
        Path(self.config["paths"]["save_dir"]).mkdir(exist_ok=True)
        Path(self.config["paths"]["log_dir"]).mkdir(exist_ok=True)

        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0

    def print_banner(self):
        """打印程序标题"""
        print("=" * 60)
        print("        TrackNetV2 羽毛球追踪训练程序")
        print("        基于论文: TrackNetV2: Efficient Shuttlecock Tracking Network")
        print("=" * 60)
        print(f"设备: {self.device}")
        print(f"数据目录: {self.config['dataset']['match_dir']}")
        print(f"输入尺寸: {self.config['dataset']['input_width']}×{self.config['dataset']['input_height']}")
        print(f"优化器: {self.config['optimization']['optimizer']}")
        print(f"学习率: {self.config['training']['learning_rate']}")
        print(f"训练轮数: {self.config['training']['num_epochs']}")
        print()

    def select_model_config(self):
        """选择模型配置"""
        print("请选择TrackNetV2配置:")
        print("1. 3-in-3-out (MIMO): 更高吞吐量, 论文推荐配置")
        print("2. 3-in-1-out (MISO): 传统配置")

        while True:
            choice = input("请输入选择 (1-2): ").strip()
            if choice == "1":
                return "3in3out"
            elif choice == "2":
                return "3in1out"
            else:
                print("无效选择，请重新输入!")

    def setup_model_and_optimizer(self, config_name):
        """根据论文配置设置模型和优化器"""
        dataset_config = self.config["dataset"]["configs"][config_name]

        # 创建TrackNet模型
        self.model = TrackNet()

        # 根据输出帧数调整最后一层 - 论文中的MIMO设计
        if dataset_config['output_frames'] != 3:
            self.model.conv2d_18 = nn.Conv2d(64, dataset_config['output_frames'], 1)

        self.model = self.model.to(self.device)

        # 论文中的损失函数
        self.criterion = WeightedBCELoss()

        # 论文中指定的优化器
        self.optimizer = optim.Adadelta(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        # 学习率调度器
        scheduler_config = self.config['optimization']['scheduler']
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=scheduler_config['mode'],
            factor=scheduler_config['factor'],
            patience=scheduler_config['patience'],
            verbose=scheduler_config['verbose']
        )

        print(f"✓ 模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"✓ 配置: {config_name} ({dataset_config['input_frames']}进{dataset_config['output_frames']}出)")

    def setup_data_loaders(self, config_name):
        """设置数据加载器"""
        dataset_config = self.config["dataset"]["configs"][config_name]
        match_dir = Path(self.config["dataset"]["base_dir"]) / self.config["dataset"]["match_dir"]

        print(f"\n设置数据加载器...")
        print(f"数据目录: {match_dir}")

        try:
            dataset = BallTrackingDataset(str(match_dir), config=dataset_config)
            print(f"✓ 数据集大小: {len(dataset)}")
        except Exception as e:
            print(f"✗ 创建数据集失败: {e}")
            return False

        # 分割数据集
        train_size = int(self.config['training']['train_ratio'] * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        print(f"✓ 训练集: {train_size}, 验证集: {val_size}")

        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        print(f"✓ 训练批次: {len(self.train_loader)}, 验证批次: {len(self.val_loader)}")
        return True

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config['training']['num_epochs']}")

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # 更新进度条
            if batch_idx % self.config['logging']['print_interval'] == 0:
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'Avg': f'{total_loss / (batch_idx + 1):.6f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate_epoch(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="验证中"):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss

    def save_checkpoint(self, epoch, config_name, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config_name': config_name,
            'training_config': self.config
        }

        save_dir = Path(self.config["paths"]["save_dir"])

        # 保存最新检查点
        latest_path = save_dir / f'latest_{config_name}.pth'
        torch.save(checkpoint, latest_path)

        # 保存最佳模型
        if is_best:
            best_path = save_dir / f'best_{config_name}.pth'
            torch.save(checkpoint, best_path)
            print(f"🏆 保存最佳模型! 验证损失: {self.best_val_loss:.6f}")

        # 定期保存
        if (epoch + 1) % self.config['logging']['save_interval'] == 0:
            epoch_path = save_dir / f'checkpoint_{config_name}_epoch_{epoch + 1}.pth'
            torch.save(checkpoint, epoch_path)

    def plot_losses(self, config_name):
        """绘制损失曲线"""
        if len(self.train_losses) < 2:
            return

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='训练损失', color='blue', linewidth=2)
        plt.plot(self.val_losses, label='验证损失', color='red', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Weighted BCE Loss')
        plt.title(f'TrackNetV2 训练曲线 - {config_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        if len(self.train_losses) > 10:
            recent_epochs = min(20, len(self.train_losses))
            epochs = range(len(self.train_losses) - recent_epochs, len(self.train_losses))
            plt.plot(epochs, self.train_losses[-recent_epochs:],
                     label=f'训练损失 (最近{recent_epochs}轮)', color='blue', linewidth=2)
            plt.plot(epochs, self.val_losses[-recent_epochs:],
                     label=f'验证损失 (最近{recent_epochs}轮)', color='red', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('最近训练进度')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        log_dir = Path(self.config["paths"]["log_dir"])
        plt.savefig(log_dir / f'loss_curves_{config_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def train_model(self, config_name):
        """主训练循环"""
        print(f"\n🚀 开始训练 TrackNetV2 - {config_name}")
        print("-" * 60)

        start_time = time.time()

        for epoch in range(self.config['training']['num_epochs']):
            epoch_start = time.time()

            # 训练
            train_loss = self.train_epoch(epoch)

            # 验证
            val_loss = self.validate_epoch()

            # 学习率调度
            self.scheduler.step(val_loss)

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']

            # 打印进度
            print(f"Epoch {epoch + 1:3d}/{self.config['training']['num_epochs']}")
            print(f"  📈 训练损失: {train_loss:.6f}")
            print(f"  📊 验证损失: {val_loss:.6f}")
            print(f"  ⏱️  用时: {epoch_time:.1f}s, 学习率: {current_lr:.2e}")

            # 检查最佳模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            # 保存检查点
            self.save_checkpoint(epoch, config_name, is_best)

            # 绘制损失曲线
            if (epoch + 1) % self.config['logging']['plot_interval'] == 0:
                self.plot_losses(config_name)

            # 早停检查
            if (self.config['early_stopping']['enabled'] and
                    self.early_stop_counter >= self.config['early_stopping']['patience']):
                print(f"⏰ 早停触发! 在第 {epoch + 1} 轮停止训练")
                break

            print("-" * 60)

        total_time = time.time() - start_time
        print(f"\n🎉 训练完成!")
        print(f"⏱️ 总用时: {total_time / 3600:.2f} 小时")
        print(f"🏆 最佳验证损失: {self.best_val_loss:.6f}")

        # 最终保存
        self.save_checkpoint(epoch, config_name, False)
        self.plot_losses(config_name)

    def test_data_loading(self, config_name):
        """测试数据加载"""
        print(f"\n🧪 测试数据加载和模型前向传播...")

        try:
            for inputs, targets in self.train_loader:
                print(f"✓ 输入形状: {inputs.shape}")
                print(f"✓ 目标形状: {targets.shape}")
                print(f"✓ 输入范围: [{inputs.min():.3f}, {inputs.max():.3f}]")
                print(f"✓ 目标范围: [{targets.min():.3f}, {targets.max():.3f}]")

                # 测试模型前向传播
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                with torch.no_grad():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                print(f"✓ 输出形状: {outputs.shape}")
                print(f"✓ 损失值: {loss.item():.6f}")
                print("✅ 数据加载和模型测试成功!")
                return True

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False

    def run(self):
        """主运行函数"""
        self.print_banner()

        # 1. 选择模型配置
        config_name = self.select_model_config()
        print(f"✅ 选择配置: {config_name}")

        # 2. 设置模型和优化器
        self.setup_model_and_optimizer(config_name)

        # 3. 设置数据加载器
        if not self.setup_data_loaders(config_name):
            print("❌ 数据加载器设置失败，程序退出")
            return

        # 4. 测试数据加载
        if not self.test_data_loading(config_name):
            print("❌ 数据加载测试失败，请检查数据集!")
            return

        # 5. 开始训练
        print(f"\n🎯 论文配置总结:")
        print(f"   - 输入尺寸: 512×288×9")
        print(f"   - 输出: {config_name}")
        print(f"   - 损失函数: Weighted BCE (论文公式)")
        print(f"   - 优化器: Adadelta (lr=1.0)")
        print(f"   - 轮数: 30")

        input("\n按回车键开始训练...")
        self.train_model(config_name)

        print("\n🏁 训练程序结束!")


if __name__ == "__main__":
    trainer = TrackNetV2Trainer()
    trainer.run()
