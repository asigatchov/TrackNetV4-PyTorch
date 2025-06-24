#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TrackNet 训练脚本 - 优化版
"""

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

from TrackNet import TrackNet
from dataset_preprocess.dataset_frame_heatmap import FrameHeatmapDataset

# ================== 配置 ==================
CONFIG = {
    # 数据集配置
    "dataset": {
        "root_dir": "dataset/Professional_reorg_train",
        "train_ratio": 0.8,
        "random_seed": 26
    },

    # 训练配置
    "training": {
        "batch_size": 3,
        "num_epochs": 30,
        "num_workers": 0,
        "device": "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    },

    # 优化器配置
    "optimizer": {
        "type": "Adadelta",
        "lr": 1.0,
        "weight_decay": 0
    },

    # 学习率调度器配置
    "lr_scheduler": {
        "type": "ReduceLROnPlateau",
        "mode": "min",
        "factor": 0.5,
        "patience": 3,
        "min_lr": 1e-6
    },

    # 日志和保存配置
    "logging": {
        "plot_interval_batches": 10,  # 每x个batch记录一次用于绘图
        "save_dir": "training_outputs",
        "experiment_name": "tracknet_experiment"
    }
}


class WeightedBinaryCrossEntropy(nn.Module):
    """
    论文中定义的加权二元交叉熵损失函数
    WBCE = -Σ[(1-w)² * ŷ * log(y) + w² * (1-ŷ) * log(1-y)]
    其中 w = y (预测值本身作为权重)
    """

    def __init__(self, epsilon=1e-7):
        super(WeightedBinaryCrossEntropy, self).__init__()
        self.epsilon = epsilon  # 防止log(0)

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: 模型预测 [B, 3, H, W]，值域[0,1]
            y_true: 真实标签 [B, 3, H, W]，值域{0,1}
        Returns:
            loss: 标量损失值
        """
        # 确保预测值在有效范围内，避免log(0)
        y_pred = torch.clamp(y_pred, self.epsilon, 1 - self.epsilon)

        # w = y (论文定义：权重等于预测值)
        w = y_pred

        # 计算加权二元交叉熵
        # WBCE = -Σ[(1-w)² * ŷ * log(y) + w² * (1-ŷ) * log(1-y)]
        term1 = (1 - w) ** 2 * y_true * torch.log(y_pred)
        term2 = w ** 2 * (1 - y_true) * torch.log(1 - y_pred)

        # 负号在前，求和
        wbce = -(term1 + term2)

        # 返回批次平均损失
        return wbce.mean()


class TrainingMonitor:
    """训练监控器"""

    def __init__(self, config, save_dir):
        self.config = config
        self.save_dir = save_dir

        # 训练历史记录
        self.batch_losses = []  # 批次损失
        self.batch_steps = []  # 批次步数
        self.batch_lrs = []  # 批次学习率

        self.epoch_train_losses = []  # epoch训练损失
        self.epoch_val_losses = []  # epoch验证损失
        self.epoch_steps = []  # epoch对应的批次步数

        self.current_batch = 0

        # 设置日志
        self.setup_logger()

    def setup_logger(self):
        """设置日志记录器"""
        log_file = self.save_dir / "training.log"

        # 配置日志格式，移除控制台输出以减少啰嗦
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[logging.FileHandler(log_file)]
        )
        self.logger = logging.getLogger(__name__)

    def update_batch_loss(self, loss, lr):
        """更新批次损失（用于绘图）"""
        self.current_batch += 1

        # 按照配置的间隔记录
        if self.current_batch % self.config["logging"]["plot_interval_batches"] == 0:
            self.batch_losses.append(loss)
            self.batch_steps.append(self.current_batch)
            self.batch_lrs.append(lr)

    def update_epoch_loss(self, train_loss, val_loss):
        """更新epoch损失"""
        self.epoch_train_losses.append(train_loss)
        self.epoch_val_losses.append(val_loss)
        self.epoch_steps.append(self.current_batch)

    def plot_training_curves(self, save_path):
        """绘制训练曲线（英文标签）"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 损失曲线
        # 绘制批次损失
        if self.batch_losses:
            ax1.plot(self.batch_steps, self.batch_losses, 'b-', alpha=0.3,
                     label=f'Batch Loss (every {self.config["logging"]["plot_interval_batches"]} batches)')

        # 绘制epoch损失
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

        # 学习率曲线
        if self.batch_lrs:
            ax2.plot(self.batch_steps, self.batch_lrs, 'g-', linewidth=2)
            ax2.set_xlabel(f'Batch Number (every {self.config["logging"]["plot_interval_batches"]} batches)')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


class ModelCheckpoint:
    """模型检查点管理器"""

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.best_loss = float('inf')

    def save_checkpoint(self, model, optimizer, scheduler, epoch, metrics):
        """保存检查点"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'timestamp': timestamp
        }

        # 每个epoch都保存
        filename = f"checkpoint_epoch_{epoch + 1}_{timestamp}.pth"
        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)

        # 如果是最佳模型，覆盖保存best_model.pth
        if metrics['val_loss'] < self.best_loss:
            self.best_loss = metrics['val_loss']
            best_path = self.save_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            return filepath, True

        return filepath, False


class Trainer:
    """主训练器类"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["training"]["device"])

        # 创建保存目录
        self.setup_directories()

        # 初始化组件
        self.monitor = TrainingMonitor(config, self.save_dir)
        self.checkpoint = ModelCheckpoint(self.save_dir / "checkpoints")

        # 设置中断处理
        self.emergency_save = False
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """中断信号处理"""
        print("\n检测到中断信号，正在紧急保存...")
        self.emergency_save = True

    def setup_directories(self):
        """设置目录结构"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{self.config['logging']['experiment_name']}_{timestamp}"

        self.save_dir = Path(self.config["logging"]["save_dir"]) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        (self.save_dir / "checkpoints").mkdir(exist_ok=True)
        (self.save_dir / "plots").mkdir(exist_ok=True)
        (self.save_dir / "configs").mkdir(exist_ok=True)

        # 保存配置
        config_path = self.save_dir / "configs" / "training_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)

    def prepare_data(self):
        """准备数据集和数据加载器"""
        print("正在加载数据集...")

        # 加载数据集
        dataset = FrameHeatmapDataset(self.config["dataset"]["root_dir"])
        print(f"数据集大小: {len(dataset)}")

        # 设置随机种子
        torch.manual_seed(self.config["dataset"]["random_seed"])

        # 分割数据集
        train_size = int(self.config["dataset"]["train_ratio"] * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=self.config["training"]["num_workers"],
            pin_memory=True if self.device.type == 'cuda' else False
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config["training"]["num_workers"],
            pin_memory=True if self.device.type == 'cuda' else False
        )

        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")

    def prepare_model(self):
        """准备模型、损失函数和优化器"""
        # 创建模型
        self.model = TrackNet().to(self.device)
        self.criterion = WeightedBinaryCrossEntropy()

        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"模型参数量: {total_params:,}")

        # 创建优化器
        if self.config["optimizer"]["type"] == "Adadelta":
            self.optimizer = torch.optim.Adadelta(
                self.model.parameters(),
                lr=self.config["optimizer"]["lr"],
                weight_decay=self.config["optimizer"]["weight_decay"]
            )

        # 创建学习率调度器
        if self.config["lr_scheduler"]["type"] == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=self.config["lr_scheduler"]["mode"],
                factor=self.config["lr_scheduler"]["factor"],
                patience=self.config["lr_scheduler"]["patience"],
                min_lr=self.config["lr_scheduler"]["min_lr"]
                # 移除了 verbose=False，因为 ReduceLROnPlateau 没有这个参数
            )
        else:
            self.scheduler = None

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        batch_count = 0

        for inputs, targets in self.train_loader:
            if self.emergency_save:
                break

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 记录损失
            batch_loss = loss.item()
            total_loss += batch_loss
            batch_count += 1

            # 更新批次损失（用于绘图）
            current_lr = self.optimizer.param_groups[0]['lr']
            self.monitor.update_batch_loss(batch_loss, current_lr)

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        return avg_loss

    def validate(self):
        """验证模型"""
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
        """紧急保存检查点"""
        # 创建紧急保存目录
        emergency_dir = Path("emergency_saves") / datetime.now().strftime("%Y%m%d_%H%M%S")
        emergency_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型
        checkpoint_path = emergency_dir / f"emergency_checkpoint_epoch_{epoch + 1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss
        }, checkpoint_path)

        # 保存训练曲线
        plot_path = emergency_dir / "training_curves.png"
        self.monitor.plot_training_curves(plot_path)

        # 保存配置
        config_path = emergency_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)

        print(f"紧急保存完成: {emergency_dir}")

    def train(self):
        """主训练循环"""
        print(f"开始训练...")
        print(f"使用设备: {self.device}")
        print("-" * 50)

        # 准备数据和模型
        self.prepare_data()
        self.prepare_model()

        # 训练循环
        for epoch in range(self.config["training"]["num_epochs"]):
            if self.emergency_save:
                break

            epoch_start_time = time.time()

            # 显示epoch进度
            print(f"\nEpoch [{epoch + 1}/{self.config['training']['num_epochs']}]")

            # 训练
            with tqdm(total=len(self.train_loader), desc="训练", ncols=80) as pbar:
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

                    # 更新批次损失记录
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.monitor.update_batch_loss(batch_loss, current_lr)

                    pbar.update(1)
                    pbar.set_postfix({'loss': f'{batch_loss:.6f}'})

                train_loss = total_loss / len(self.train_loader)

            # 验证
            with tqdm(total=len(self.val_loader), desc="验证", ncols=80) as pbar:
                val_loss = self.validate()
                pbar.update(len(self.val_loader))
                pbar.set_postfix({'loss': f'{val_loss:.6f}'})

            # 更新epoch损失记录
            self.monitor.update_epoch_loss(train_loss, val_loss)

            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']

            # 打印epoch结果
            print(f"训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}, 学习率: {current_lr:.6f}")

            # 更新学习率
            if self.scheduler:
                self.scheduler.step(val_loss)

            # 保存检查点
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr
            }

            checkpoint_path, is_best = self.checkpoint.save_checkpoint(
                self.model, self.optimizer, self.scheduler, epoch, metrics
            )

            if is_best:
                print(f"保存最佳模型！验证损失: {val_loss:.6f}")

            # 保存训练曲线
            plot_path = self.save_dir / "plots" / f"training_curves_epoch_{epoch + 1}.png"
            self.monitor.plot_training_curves(plot_path)

            # 记录日志
            self.monitor.logger.info(
                f"Epoch {epoch + 1}/{self.config['training']['num_epochs']}: "
                f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                f"lr={current_lr:.6f}, time={time.time() - epoch_start_time:.2f}s"
            )

        # 处理中断或正常结束
        if self.emergency_save:
            self.emergency_checkpoint(epoch, train_loss, val_loss)
        else:
            print("\n训练完成！")

            # 保存最终训练曲线
            final_plot_path = self.save_dir / "plots" / "final_training_curves.png"
            self.monitor.plot_training_curves(final_plot_path)

            print(f"所有结果已保存到: {self.save_dir}")


# ================== 主程序入口 ==================
if __name__ == "__main__":
    trainer = Trainer(CONFIG)
    trainer.train()
