#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TrackNet 训练脚本 - 完全重构版本
支持功能：
- 进度条显示
- 自动保存模型和训练状态（仅每个 epoch）
- 紧急保存
- 损失曲线和学习率曲线可视化
- 配置管理
"""

import json
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from TrackNet import TrackNet, WeightedBinaryCrossEntropy
from dataset_preprocessing.dataset_generator import FrameHeatmapDataset

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
        "num_workers": 0,
        # 使用 MPS/CUDA/CPU
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
        "log_interval_batches": 10,  # 每 x 个 batch 记录一次
        "plot_interval_batches": 50,  # 绘制学习率曲线的间隔
        "save_dir": "training_outputs",
        "experiment_name": "tracknet_experiment"
    },

    # 早停（未实现，可扩展）
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
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.batch_train_losses = []
        self.batch_numbers = []
        self.setup_logger()

    def setup_logger(self):
        log_file = self.save_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stderr)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log_batch(self, epoch, batch_idx, total_batches, loss, lr):
        global_batch = epoch * total_batches + batch_idx
        self.batch_train_losses.append(loss)
        self.batch_numbers.append(global_batch)
        self.learning_rates.append(lr)
        if batch_idx % self.config["logging"]["log_interval_batches"] == 0:
            self.logger.info(
                f"Epoch [{epoch + 1}/{self.config['training']['num_epochs']}] "
                f"Batch [{batch_idx}/{total_batches}] Loss: {loss:.6f} LR: {lr:.6f}"
            )

    def log_epoch(self, epoch, train_loss, val_loss, lr):
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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Loss curves
        ax1.plot(self.train_losses, label='Training Loss', linewidth=2)
        ax1.plot(self.val_losses,   label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training vs Validation Loss')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Learning rate curve
        interval = self.config["logging"]["plot_interval_batches"]
        indices = list(range(0, len(self.learning_rates), interval))
        ax2.plot(
            [self.batch_numbers[i]     for i in indices],
            [self.learning_rates[i]    for i in indices],
            linewidth=2
        )
        ax2.set_xlabel(f'Batch (every {interval} steps)')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(alpha=0.3)
        ax2.set_yscale('log')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

class ModelCheckpoint:
    """模型检查点，仅每 epoch 或紧急保存时执行"""

    def __init__(self, save_dir, monitor='val_loss', mode='min'):
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')

    def save_checkpoint(self, model, optimizer, scheduler, epoch, metrics, is_best=False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'timestamp': timestamp
        }
        path = self.save_dir / f"checkpoint_epoch_{epoch + 1}_{timestamp}.pth"
        torch.save(ckpt, path)
        if is_best:
            torch.save(ckpt, self.save_dir / "best_model.pth")
        return path

    def is_best(self, current_score):
        is_better = (current_score < self.best_score) if self.mode == 'min' else (current_score > self.best_score)
        if is_better:
            self.best_score = current_score
        return is_better


class Trainer:
    """主训练器：包含数据准备、模型训练、验证、保存逻辑"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['training']['device'])
        self.setup_directories()
        self.monitor = TrainingMonitor(config, self.save_dir)
        self.checkpoint = ModelCheckpoint(self.save_dir / 'checkpoints')
        self.setup_signal_handlers()
        self.emergency_save = False

    def setup_directories(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{self.config['logging']['experiment_name']}_{timestamp}"
        self.save_dir = Path(self.config['logging']['save_dir']) / name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.save_dir / 'plots').mkdir(exist_ok=True)
        (self.save_dir / 'configs').mkdir(exist_ok=True)
        with open(self.save_dir / 'configs' / 'training_config.json', 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)

    def setup_signal_handlers(self):
        def handler(signum, frame):
            self.monitor.logger.warning("检测到中断，触发紧急保存")
            self.emergency_save = True

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def prepare_data(self):
        """加载并切分数据集，构建 DataLoader"""
        self.monitor.logger.info("加载数据集...")
        dataset = FrameHeatmapDataset(self.config['dataset']['root_dir'])
        torch.manual_seed(self.config['dataset']['random_seed'])
        train_size = int(self.config['dataset']['train_ratio'] * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        self.train_loader = DataLoader(train_ds,
                                       batch_size=self.config['training']['batch_size'],
                                       shuffle=True,
                                       num_workers=self.config['training']['num_workers'],
                                       pin_memory=self.device.type == 'cuda')
        self.val_loader = DataLoader(val_ds,
                                     batch_size=self.config['training']['batch_size'],
                                     shuffle=False,
                                     num_workers=self.config['training']['num_workers'],
                                     pin_memory=self.device.type == 'cuda')
        self.monitor.logger.info(f"训练集: {len(train_ds)} 样本, 验证集: {len(val_ds)} 样本")

    def prepare_model(self):
        """初始化模型、损失函数、优化器及调度器"""
        self.model = TrackNet().to(self.device)
        self.criterion = WeightedBinaryCrossEntropy()
        optimizer_cfg = self.config['optimizer']
        if optimizer_cfg['type'] == 'Adadelta':
            self.optimizer = torch.optim.Adadelta(self.model.parameters(),
                                                  lr=optimizer_cfg['lr'],
                                                  weight_decay=optimizer_cfg['weight_decay'])
        else:
            raise ValueError(f"未知优化器: {optimizer_cfg['type']}")
        sched_cfg = self.config['lr_scheduler']
        if sched_cfg['type'] == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer,
                                               mode=sched_cfg['mode'],
                                               factor=sched_cfg['factor'],
                                               patience=sched_cfg['patience'],
                                               min_lr=sched_cfg['min_lr'])
        else:
            self.scheduler = None

    def train_epoch(self, epoch):
        """执行一个 epoch 的训练"""
        self.model.train()
        running_loss = 0.0
        pbar = tqdm(self.train_loader,
                    desc=f"Epoch {epoch + 1}/{self.config['training']['num_epochs']} [训练]",
                    ncols=100, leave=False)
        for idx, (x, y) in enumerate(pbar):
            if self.emergency_save: break
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()
            batch_loss = loss.item()
            running_loss += batch_loss
            lr = self.optimizer.param_groups[0]['lr']
            self.monitor.log_batch(epoch, idx, len(self.train_loader), batch_loss, lr)
            pbar.set_postfix({'loss': f'{batch_loss:.6f}', 'lr': f'{lr:.2e}'})
        pbar.close()
        return running_loss / len(self.train_loader)

    def validate(self, epoch):
        """执行一个 epoch 的验证"""
        self.model.eval()
        val_loss = 0.0
        pbar = tqdm(self.val_loader,
                    desc=f"Epoch {epoch + 1}/{self.config['training']['num_epochs']} [验证]",
                    ncols=100, leave=False)
        with torch.no_grad():
            for x, y in pbar:
                if self.emergency_save: break
                x, y = x.to(self.device), y.to(self.device)
                loss = self.criterion(self.model(x), y)
                val_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        pbar.close()
        return val_loss / len(self.val_loader)

    def emergency_checkpoint(self, epoch, train_loss, val_loss):
        """紧急中断时保存最新状态"""
        self.monitor.logger.warning("执行紧急保存...")
        em_dir = Path("emergency_saves") / datetime.now().strftime("%Y%m%d_%H%M%S")
        em_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss
        }, em_dir / f"emergency_epoch_{epoch + 1}.pth")
        self.monitor.plot_training_curves(em_dir / "training_curves.png")
        with open(em_dir / "config.json", 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)
        self.monitor.logger.warning(f"紧急保存完成: {em_dir}")

    def train(self):
        """主训练流程"""
        self.monitor.logger.info(f"使用设备: {self.device}")
        self.prepare_data()
        self.prepare_model()
        for epoch in range(self.config['training']['num_epochs']):
            if self.emergency_save: break
            start = time.time()
            tr_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            lr = self.optimizer.param_groups[0]['lr']
            self.monitor.log_epoch(epoch, tr_loss, val_loss, lr)
            if self.scheduler: self.scheduler.step(val_loss)
            metrics = {'train_loss': tr_loss, 'val_loss': val_loss, 'lr': lr}
            is_best = self.checkpoint.is_best(val_loss)
            self.checkpoint.save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, metrics, is_best)
            plot_file = self.save_dir / 'plots' / f'training_curves_epoch_{epoch + 1}.png'
            self.monitor.plot_training_curves(plot_file)
            self.monitor.logger.info(f"Epoch 用时: {time.time() - start:.2f}秒")
        if self.emergency_save:
            self.emergency_checkpoint(epoch, tr_loss, val_loss)
        else:
            final_plot = self.save_dir / 'plots' / 'final_training_curves.png'
            self.monitor.plot_training_curves(final_plot)
            self.monitor.logger.info(f"训练完成，结果保存在 {self.save_dir}")


if __name__ == '__main__':
    Trainer(CONFIG).train()
