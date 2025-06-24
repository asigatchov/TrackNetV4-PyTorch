#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TrackNet 训练脚本 - 完全重构版本
支持功能：
- 进度条显示
- 自动保存模型和训练状态
- 损失曲线和学习率曲线可视化
- 中断保护和紧急保存
- 配置管理
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import sys
import json
import signal
import time
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset_preprocessing.dataset_generator import FrameHeatmapDataset
from TrackNet import TrackNet, WeightedBinaryCrossEntropy

# ================== 配置区域 ==================
CONFIG = {
    # 数据集配置
    "dataset": {
        "root_dir": "dataset/Professional_reorg_train",
        "train_ratio": 0.8,
        "random_seed": 26
    },

    # 训练配置
    "training": {
        "batch_size": 2,
        "num_epochs": 5,
        "num_workers": 2,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
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
        "min_lr": 1e-6,
        "verbose": False
    },

    # 日志和保存配置
    "logging": {
        "log_interval_batches": 10,  # 每x个batch记录一次
        "plot_interval_batches": 50,  # 每x个batch更新一次图表
        "save_dir": "training_outputs",
        "experiment_name": "tracknet_experiment"
    },

    # 早停配置
    "early_stopping": {
        "patience": 10,
        "min_delta": 1e-4
    }
}


class TrainingMonitor:
    """训练监控器，负责记录和可视化训练过程"""

    def __init__(self, config, save_dir):
        self.config = config
        self.save_dir = save_dir

        # 训练历史记录
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.batch_train_losses = []
        self.batch_numbers = []

        # 设置日志
        self.setup_logger()

    def setup_logger(self):
        """设置日志记录器"""
        log_file = self.save_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log_batch(self, epoch, batch_idx, total_batches, loss, lr):
        """记录批次信息"""
        global_batch = epoch * total_batches + batch_idx
        self.batch_train_losses.append(loss)
        self.batch_numbers.append(global_batch)
        self.learning_rates.append(lr)

        if batch_idx % self.config["logging"]["log_interval_batches"] == 0:
            self.logger.info(
                f"Epoch [{epoch + 1}/{self.config['training']['num_epochs']}] "
                f"Batch [{batch_idx}/{total_batches}] "
                f"Loss: {loss:.6f} LR: {lr:.6f}"
            )

    def log_epoch(self, epoch, train_loss, val_loss, lr):
        """记录epoch信息"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        self.logger.info(
            f"\nEpoch [{epoch + 1}/{self.config['training']['num_epochs']}] 完成\n"
            f"训练损失: {train_loss:.6f}\n"
            f"验证损失: {val_loss:.6f}\n"
            f"学习率: {lr:.6f}\n"
            f"{'-' * 50}"
        )

    def plot_training_curves(self, save_path):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 绘制损失曲线
        ax1.plot(self.train_losses, 'b-', label='训练损失', linewidth=2)
        ax1.plot(self.val_losses, 'r-', label='验证损失', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失')
        ax1.set_title('训练和验证损失曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 绘制学习率曲线
        plot_interval = self.config["logging"]["plot_interval_batches"]
        plot_indices = range(0, len(self.learning_rates), plot_interval)
        plot_lrs = [self.learning_rates[i] for i in plot_indices if i < len(self.learning_rates)]
        plot_batches = [self.batch_numbers[i] for i in plot_indices if i < len(self.batch_numbers)]

        ax2.plot(plot_batches, plot_lrs, 'g-', linewidth=2)
        ax2.set_xlabel(f'批次 (每{plot_interval}个批次)')
        ax2.set_ylabel('学习率')
        ax2.set_title('学习率调整曲线')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


class ModelCheckpoint:
    """模型检查点管理器"""

    def __init__(self, save_dir, monitor='val_loss', mode='min'):
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')

    def save_checkpoint(self, model, optimizer, scheduler, epoch, metrics, is_best=False):
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

        # 保存最新的检查点
        filename = f"checkpoint_epoch_{epoch + 1}_{timestamp}.pth"
        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)

        # 如果是最佳模型，额外保存一份
        if is_best:
            best_path = self.save_dir / "best_model.pth"
            torch.save(checkpoint, best_path)

        return filepath

    def is_best(self, current_score):
        """检查是否是最佳分数"""
        if self.mode == 'min':
            is_best = current_score < self.best_score
        else:
            is_best = current_score > self.best_score

        if is_best:
            self.best_score = current_score

        return is_best


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
        self.setup_signal_handlers()

        # 标记是否需要紧急保存
        self.emergency_save = False

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

    def setup_signal_handlers(self):
        """设置信号处理器用于紧急保存"""

        def signal_handler(signum, frame):
            self.monitor.logger.warning("\n检测到中断信号，正在紧急保存...")
            self.emergency_save = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def prepare_data(self):
        """准备数据集和数据加载器"""
        self.monitor.logger.info("正在加载数据集...")

        # 加载数据集
        dataset = FrameHeatmapDataset(self.config["dataset"]["root_dir"])
        self.monitor.logger.info(f"数据集大小: {len(dataset)}")

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

        self.monitor.logger.info(f"训练集大小: {len(train_dataset)}")
        self.monitor.logger.info(f"验证集大小: {len(val_dataset)}")

    def prepare_model(self):
        """准备模型、损失函数和优化器"""
        # 创建模型
        self.model = TrackNet().to(self.device)
        self.criterion = WeightedBinaryCrossEntropy()

        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.monitor.logger.info(f"模型总参数量: {total_params:,}")
        self.monitor.logger.info(f"可训练参数量: {trainable_params:,}")

        # 创建优化器
        optimizer_config = self.config["optimizer"]
        if optimizer_config["type"] == "Adadelta":
            self.optimizer = torch.optim.Adadelta(
                self.model.parameters(),
                lr=optimizer_config["lr"],
                weight_decay=optimizer_config["weight_decay"]
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_config['type']}")

        # 创建学习率调度器
        scheduler_config = self.config["lr_scheduler"]
        if scheduler_config["type"] == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=scheduler_config["mode"],
                factor=scheduler_config["factor"],
                patience=scheduler_config["patience"],
                min_lr=scheduler_config["min_lr"],
                verbose=scheduler_config["verbose"]
            )
        else:
            self.scheduler = None

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0

        # 创建进度条
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config['training']['num_epochs']} [训练]",
            ncols=100,
            leave=False
        )

        for batch_idx, (inputs, targets) in enumerate(pbar):
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

            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录批次信息
            self.monitor.log_batch(epoch, batch_idx, len(self.train_loader), batch_loss, current_lr)

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'lr': f'{current_lr:.2e}'
            })

        pbar.close()

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self, epoch):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0

        # 创建进度条
        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch + 1}/{self.config['training']['num_epochs']} [验证]",
            ncols=100,
            leave=False
        )

        with torch.no_grad():
            for inputs, targets in pbar:
                if self.emergency_save:
                    break

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                batch_loss = loss.item()
                total_loss += batch_loss

                pbar.set_postfix({'loss': f'{batch_loss:.4f}'})

        pbar.close()

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def emergency_checkpoint(self, epoch, train_loss, val_loss):
        """紧急保存检查点"""
        self.monitor.logger.warning("正在执行紧急保存...")

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

        self.monitor.logger.warning(f"紧急保存完成: {emergency_dir}")

    def train(self):
        """主训练循环"""
        self.monitor.logger.info("开始训练...")
        self.monitor.logger.info(f"使用设备: {self.device}")

        # 准备数据和模型
        self.prepare_data()
        self.prepare_model()

        # 训练循环
        for epoch in range(self.config["training"]["num_epochs"]):
            if self.emergency_save:
                break

            epoch_start_time = time.time()

            # 训练一个epoch
            train_loss = self.train_epoch(epoch)

            # 验证
            val_loss = self.validate(epoch)

            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录epoch信息
            self.monitor.log_epoch(epoch, train_loss, val_loss, current_lr)

            # 更新学习率
            if self.scheduler:
                self.scheduler.step(val_loss)

            # 保存检查点
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr
            }

            is_best = self.checkpoint.is_best(val_loss)
            checkpoint_path = self.checkpoint.save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                epoch, metrics, is_best
            )

            if is_best:
                self.monitor.logger.info(f"新的最佳模型！验证损失: {val_loss:.6f}")

            # 保存训练曲线
            plot_path = self.save_dir / "plots" / f"training_curves_epoch_{epoch + 1}.png"
            self.monitor.plot_training_curves(plot_path)

            # 记录epoch时间
            epoch_time = time.time() - epoch_start_time
            self.monitor.logger.info(f"Epoch时间: {epoch_time:.2f}秒\n")

        # 检查是否需要紧急保存
        if self.emergency_save:
            self.emergency_checkpoint(epoch, train_loss, val_loss)
        else:
            self.monitor.logger.info("训练完成！")

            # 保存最终的训练曲线
            final_plot_path = self.save_dir / "plots" / "final_training_curves.png"
            self.monitor.plot_training_curves(final_plot_path)

            self.monitor.logger.info(f"所有结果已保存到: {self.save_dir}")


# ================== 主程序入口 ==================
if __name__ == "__main__":
    # 创建训练器并开始训练
    trainer = Trainer(CONFIG)
    trainer.train()
