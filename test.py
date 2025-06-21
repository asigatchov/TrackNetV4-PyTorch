#!/usr/bin/env python3
"""
TrackNetV2 测试脚本 - 基于论文结构
论文: TrackNetV2: Efficient Shuttlecock Tracking Network
支持 3-in-3-out MIMO 设计，512×288 输入分辨率
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json

from dataset_controller.ball_tracking_data_reader import BallTrackingDataset
from tracknet import TrackNetV4, postprocess_heatmap

# ==================== 配置参数 ====================
CONFIG = {
    # 数据集配置 - 按照论文TrackNetV2设置
    "dataset": {
        "input_frames": 3,           # 3帧输入
        "output_frames": 3,          # 3帧输出 (MIMO)
        "normalize_coords": True,
        "normalize_pixels": True,
        "video_ext": ".mp4",
        "csv_suffix": "_ball.csv"
    },

    # 网络配置 - 按照论文Table I
    "network": {
        "input_height": 288,         # 论文中的512×288
        "input_width": 512,
        "heatmap_radius": 3,         # 高斯热图半径
        "output_channels": 3         # MIMO输出3个热图
    },

    # 测试配置 - 按照论文实验设置
    "test": {
        "batch_size": 2,
        "pixel_threshold": 4.0,      # 论文中的4像素阈值
        "heatmap_threshold": 0.5,    # 论文中的0.5阈值
        "distance_metric": "euclidean"
    },

    # 设备配置
    "device": {
        "cuda_workers": 4,
        "cuda_pin_memory": True,
        "mps_workers": 2,
        "mps_pin_memory": False,
        "cpu_workers": 4,
        "cpu_pin_memory": False,
        "persistent_workers": True
    },

    # 路径配置 (硬编码)
    "paths": {
        "test_data_dir": "dataset/Test",
        "checkpoint_path": "best.pth"
    }
}

print("=" * 60)
print("TrackNetV2 测试配置:")
print(json.dumps(CONFIG, indent=2, ensure_ascii=False))
print("=" * 60)


def get_device_and_loader_config():
    """设备检测和数据加载器配置"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        loader_config = {
            "num_workers": CONFIG["device"]["cuda_workers"],
            "pin_memory": CONFIG["device"]["cuda_pin_memory"],
            "persistent_workers": CONFIG["device"]["persistent_workers"]
        }
        print(f"✓ 使用CUDA: {torch.cuda.get_device_name()}")
        # 启用CUDA优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        loader_config = {
            "num_workers": CONFIG["device"]["mps_workers"],
            "pin_memory": CONFIG["device"]["mps_pin_memory"],
            "persistent_workers": False
        }
        print("✓ 使用MPS: Apple Silicon")

    else:
        device = torch.device('cpu')
        loader_config = {
            "num_workers": CONFIG["device"]["cpu_workers"],
            "pin_memory": CONFIG["device"]["cpu_pin_memory"],
            "persistent_workers": CONFIG["device"]["persistent_workers"]
        }
        print("⚠️ 使用CPU模式")

    return device, loader_config


def create_gaussian_heatmap(x, y, visibility, height, width, radius):
    """
    生成高斯热图 - 按照论文TrackNetV2的方法
    Ground truth is an amplified 2D Gaussian distribution function
    """
    heatmap = torch.zeros(height, width, dtype=torch.float32)

    if visibility < 0.5:  # 不可见球
        return heatmap

    # 转换为像素坐标
    x_pixel = max(0, min(width - 1, int(x * width)))
    y_pixel = max(0, min(height - 1, int(y * height)))

    # 计算高斯核范围
    kernel_size = int(3 * radius)
    x_min = max(0, x_pixel - kernel_size)
    x_max = min(width, x_pixel + kernel_size + 1)
    y_min = max(0, y_pixel - kernel_size)
    y_max = min(height, y_pixel + kernel_size + 1)

    # 生成网格
    y_coords, x_coords = torch.meshgrid(
        torch.arange(y_min, y_max),
        torch.arange(x_min, x_max),
        indexing='ij'
    )

    # 计算高斯分布
    dist_sq = (x_coords - x_pixel) ** 2 + (y_coords - y_pixel) ** 2
    gaussian_values = torch.exp(-dist_sq / (2 * radius ** 2))

    # 阈值化减少噪声
    gaussian_values[gaussian_values < 0.01] = 0

    heatmap[y_min:y_max, x_min:x_max] = gaussian_values
    return heatmap


def tracknetv2_collate_fn(batch):
    """
    TrackNetV2 批处理函数
    按照论文描述处理数据：输入调整为512×288，生成3个输出热图
    """
    frames_list = []
    heatmaps_list = []

    net_config = CONFIG["network"]
    dataset_config = CONFIG["dataset"]

    for frames, labels in batch:
        # 调整输入帧到论文指定尺寸 512×288
        frames_resized = F.interpolate(
            frames.unsqueeze(0),
            size=(net_config["input_height"], net_config["input_width"]),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        frames_list.append(frames_resized)

        # 生成MIMO输出热图 (3个热图对应3帧输出)
        output_frames = dataset_config["output_frames"]
        heatmaps = torch.zeros(
            output_frames,
            net_config["input_height"],
            net_config["input_width"],
            dtype=torch.float32
        )

        for i, label_dict in enumerate(labels):
            if i < output_frames and isinstance(label_dict, dict):
                heatmap = create_gaussian_heatmap(
                    label_dict['x'].item(),
                    label_dict['y'].item(),
                    label_dict['visibility'].item(),
                    net_config["input_height"],
                    net_config["input_width"],
                    net_config["heatmap_radius"]
                )
                heatmaps[i] = heatmap

        heatmaps_list.append(heatmaps)

    return torch.stack(frames_list), torch.stack(heatmaps_list)


def load_test_dataset():
    """加载测试数据集"""
    data_dir = Path(CONFIG["paths"]["test_data_dir"])
    if not data_dir.exists():
        raise FileNotFoundError(f"测试数据目录不存在: {data_dir}")

    # 查找所有match目录
    match_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith('match')
    ])

    if not match_dirs:
        raise ValueError(f"在 {data_dir} 中未找到match目录")

    print(f"\n加载测试数据集: {data_dir}")
    combined_dataset = None

    for match_dir in match_dirs:
        try:
            dataset = BallTrackingDataset(
                str(match_dir),
                config=CONFIG["dataset"]
            )

            if len(dataset) > 0:
                if combined_dataset is None:
                    combined_dataset = dataset
                else:
                    combined_dataset = combined_dataset + dataset
                print(f"✓ {match_dir.name}: {len(dataset)} 样本")
            else:
                print(f"⚠️ {match_dir.name}: 无有效样本")

        except Exception as e:
            print(f"✗ {match_dir.name} 加载失败: {e}")

    if combined_dataset is None or len(combined_dataset) == 0:
        raise ValueError("未找到有效的测试数据")

    print(f"总计: {len(combined_dataset)} 个测试样本")
    return combined_dataset


def load_tracknetv2_model(device):
    """加载TrackNetV2模型"""
    checkpoint_path = Path(CONFIG["paths"]["checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"模型检查点不存在: {checkpoint_path}")

    print(f"\n加载TrackNetV2模型: {checkpoint_path}")

    # 创建模型实例
    model = TrackNetV4()  # 使用现有的TrackNetV4作为TrackNetV2

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'Unknown')
        best_loss = checkpoint.get('best_loss', 'Unknown')
        print(f"✓ 从检查点加载模型 (Epoch: {epoch}, Best Loss: {best_loss})")
    else:
        model.load_state_dict(checkpoint)
        print("✓ 直接加载模型权重")

    model.to(device)
    model.eval()

    # 显示模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    return model


def evaluate_tracknetv2(model, test_loader, device, loader_config):
    """
    TrackNetV2评估
    按照论文方法：使用欧几里得距离和4像素阈值
    """
    print(f"\n开始TrackNetV2评估")
    print(f"批次数量: {len(test_loader)}")
    print(f"批次大小: {CONFIG['test']['batch_size']}")
    print(f"像素阈值: {CONFIG['test']['pixel_threshold']} px")
    print(f"热图阈值: {CONFIG['test']['heatmap_threshold']}")

    test_config = CONFIG["test"]
    output_frames = CONFIG["dataset"]["output_frames"]

    # 统计变量
    total_predictions = 0
    true_positives = 0
    false_positives_type1 = 0  # FP1: 预测和真实都有球，但距离超出阈值
    false_positives_type2 = 0  # FP2: 预测有球，真实无球
    false_negatives = 0        # FN: 预测无球，真实有球
    true_negatives = 0         # TN: 预测和真实都无球

    all_distances = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, desc="评估中")):
            batch_size = inputs.size(0)

            # GPU推理
            inputs = inputs.to(device, non_blocking=loader_config["pin_memory"])
            targets = targets.to(device, non_blocking=loader_config["pin_memory"])

            # 前向传播 - MIMO输出
            outputs = model(inputs)  # [B, 3, H, W]

            # 后处理得到预测坐标
            predicted_coords = postprocess_heatmap(
                outputs.cpu(),
                threshold=test_config["heatmap_threshold"]
            )

            # 从真实热图提取坐标
            true_coords = postprocess_heatmap(
                targets.cpu(),
                threshold=0.1  # 更低阈值提取真实坐标
            )

            # 逐样本逐帧评估
            for b in range(batch_size):
                for f in range(output_frames):
                    pred_coord = predicted_coords[b][f]
                    true_coord = true_coords[b][f]

                    total_predictions += 1

                    if pred_coord is not None and true_coord is not None:
                        # 都检测到球 - 计算距离
                        distance = np.sqrt(
                            (pred_coord[0] - true_coord[0]) ** 2 +
                            (pred_coord[1] - true_coord[1]) ** 2
                        )
                        all_distances.append(distance)

                        if distance <= test_config["pixel_threshold"]:
                            true_positives += 1  # TP
                        else:
                            false_positives_type1 += 1  # FP1

                    elif pred_coord is not None and true_coord is None:
                        # 误检 - 预测有球但真实无球
                        false_positives_type2 += 1  # FP2
                        all_distances.append(float('inf'))

                    elif pred_coord is None and true_coord is not None:
                        # 漏检 - 预测无球但真实有球
                        false_negatives += 1  # FN
                        all_distances.append(float('inf'))

                    else:
                        # 都无球 - 正确
                        true_negatives += 1  # TN
                        all_distances.append(0.0)

            # 显示第一个batch的详细结果
            if batch_idx == 0:
                print(f"\n第一批次预测示例:")
                for b in range(min(2, batch_size)):
                    for f in range(output_frames):
                        pred = predicted_coords[b][f]
                        true = true_coords[b][f]
                        if pred and true:
                            dist = np.sqrt((pred[0]-true[0])**2 + (pred[1]-true[1])**2)
                            print(f"  样本{b}帧{f}: 预测{pred} vs 真实{true} (距离:{dist:.1f}px)")
                        else:
                            print(f"  样本{b}帧{f}: 预测{pred} vs 真实{true}")

    # 计算性能指标
    accuracy = (true_positives + true_negatives) / total_predictions
    precision = true_positives / (true_positives + false_positives_type1 + false_positives_type2) if (true_positives + false_positives_type1 + false_positives_type2) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # 距离统计
    finite_distances = [d for d in all_distances if np.isfinite(d)]
    avg_distance = np.mean(finite_distances) if finite_distances else float('nan')
    median_distance = np.median(finite_distances) if finite_distances else float('nan')

    return {
        'total_predictions': total_predictions,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives_type1': false_positives_type1,
        'false_positives_type2': false_positives_type2,
        'false_negatives': false_negatives,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'avg_distance': avg_distance,
        'median_distance': median_distance,
        'finite_distances': len(finite_distances),
        'infinite_distances': len(all_distances) - len(finite_distances)
    }


def main():
    """主函数"""
    print("TrackNetV2 测试脚本启动")
    print("基于论文: TrackNetV2: Efficient Shuttlecock Tracking Network")

    try:
        # 1. 设备配置
        device, loader_config = get_device_and_loader_config()

        # 2. 加载数据集
        test_dataset = load_test_dataset()

        # 3. 创建数据加载器
        test_loader = DataLoader(
            test_dataset,
            batch_size=CONFIG["test"]["batch_size"],
            shuffle=False,
            collate_fn=tracknetv2_collate_fn,
            **loader_config
        )

        # 4. 加载模型
        model = load_tracknetv2_model(device)

        # 5. 评估模型
        print("\n" + "="*60)
        results = evaluate_tracknetv2(model, test_loader, device, loader_config)

        # 6. 输出结果
        print("\n" + "="*25 + " TrackNetV2 测试结果 " + "="*25)
        print(f"数据集配置:")
        print(f"  - 测试样本数: {len(test_dataset)}")
        print(f"  - 输入尺寸: {CONFIG['network']['input_width']}×{CONFIG['network']['input_height']}")
        print(f"  - MIMO设计: {CONFIG['dataset']['input_frames']}-in-{CONFIG['dataset']['output_frames']}-out")

        print(f"\n评估配置:")
        print(f"  - 像素阈值: {CONFIG['test']['pixel_threshold']} px")
        print(f"  - 热图阈值: {CONFIG['test']['heatmap_threshold']}")
        print(f"  - 批次大小: {CONFIG['test']['batch_size']}")

        print(f"\n混淆矩阵统计:")
        print(f"  - 总预测数: {results['total_predictions']}")
        print(f"  - 真阳性 (TP): {results['true_positives']}")
        print(f"  - 真阴性 (TN): {results['true_negatives']}")
        print(f"  - 假阳性1 (FP1): {results['false_positives_type1']} (距离超阈值)")
        print(f"  - 假阳性2 (FP2): {results['false_positives_type2']} (误检)")
        print(f"  - 假阴性 (FN): {results['false_negatives']} (漏检)")

        print(f"\n性能指标:")
        print(f"  - 准确率 (Accuracy): {results['accuracy']*100:.2f}%")
        print(f"  - 精确率 (Precision): {results['precision']*100:.2f}%")
        print(f"  - 召回率 (Recall): {results['recall']*100:.2f}%")
        print(f"  - F1分数: {results['f1_score']*100:.2f}%")

        print(f"\n距离统计:")
        print(f"  - 平均距离: {results['avg_distance']:.3f} px")
        print(f"  - 中位数距离: {results['median_distance']:.3f} px")
        print(f"  - 有效检测: {results['finite_distances']}")
        print(f"  - 失败检测: {results['infinite_distances']}")

        print("\n" + "="*70)
        print("🎯 TrackNetV2测试完成!")

        # 与论文结果对比
        print(f"\n📊 论文TrackNetV2结果对比:")
        print(f"论文 (3-in-3-out): 准确率85.2%, 精确率97.2%, 召回率85.4%")
        print(f"当前测试结果: 准确率{results['accuracy']*100:.1f}%, 精确率{results['precision']*100:.1f}%, 召回率{results['recall']*100:.1f}%")

    except Exception as e:
        print(f"❌ TrackNetV2测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()