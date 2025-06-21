#!/usr/bin/env python3
"""
TrackNet 坐标映射和热图生成测试工具
直接调用train.py文件中的函数，避免重复定义
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# 导入train.py中的函数
try:
    from train import (
        calculate_equal_ratio_resize,
        create_gaussian_heatmap,
        collate_fn,
        CONFIG,
        DATASET_CONFIG
    )

    print("✓ 成功导入train.py中的函数")
except ImportError as e:
    print(f"❌ 无法导入train.py: {e}")
    print("请确保train.py文件在同一目录下")
    sys.exit(1)

# ===== 配置区域 =====
USER_CONFIG = {
    "data_dir": "Dataset/Professional",  # 数据集目录路径
    "auto_play": True,  # 是否自动播放
    "playback_speed": 10,  # 播放速度(ms)
    "start_sample": 0,  # 起始样本索引
    "match_index": 0  # 使用第几个match目录 (0表示第一个)
}


def process_single_sample_like_train(frames, labels):
    """
    使用train.py中的collate_fn逻辑处理单个样本
    模拟batch处理但只处理一个样本
    """
    # 创建单样本batch
    batch = [(frames, labels)]

    # 使用train.py的collate_fn处理
    processed_frames, processed_heatmaps = collate_fn(batch)

    # 提取第一个(也是唯一一个)样本的结果
    sample_frames = processed_frames[0]  # shape: (3, H, W)
    sample_heatmaps = processed_heatmaps[0]  # shape: (3, H, W)

    # 计算用于可视化的额外信息
    original_height, original_width = frames.shape[-2], frames.shape[-1]
    new_height, new_width, scale_ratio = calculate_equal_ratio_resize(
        original_height, original_width,
        CONFIG["input_height"], CONFIG["input_width"]
    )

    # 计算padding
    if new_height != CONFIG["input_height"] or new_width != CONFIG["input_width"]:
        pad_h = CONFIG["input_height"] - new_height
        pad_w = CONFIG["input_width"] - new_width
        pad_top = pad_h // 2
        pad_left = pad_w // 2
    else:
        pad_left = pad_top = 0

    # 处理坐标信息用于可视化
    processed_coords = []
    for i, label_dict in enumerate(labels):
        if isinstance(label_dict, dict):
            x_orig = label_dict['x'].item()
            y_orig = label_dict['y'].item()
            visibility = label_dict['visibility'].item()

            if visibility >= 0.5:
                # 计算缩放后坐标
                x_scaled = x_orig * scale_ratio + pad_left
                y_scaled = y_orig * scale_ratio + pad_top

                # 计算归一化坐标
                x_norm = x_scaled / CONFIG["input_width"]
                y_norm = y_scaled / CONFIG["input_height"]

                processed_coords.append({
                    'x_orig': x_orig,
                    'y_orig': y_orig,
                    'x_scaled': x_scaled,
                    'y_scaled': y_scaled,
                    'x_norm': x_norm,
                    'y_norm': y_norm,
                    'visibility': visibility
                })
            else:
                processed_coords.append({
                    'x_orig': x_orig,
                    'y_orig': y_orig,
                    'x_scaled': -1,
                    'y_scaled': -1,
                    'x_norm': -1,
                    'y_norm': -1,
                    'visibility': visibility
                })

    return {
        'original_size': (original_height, original_width),
        'new_size': (new_height, new_width),
        'scale_ratio': scale_ratio,
        'pad_left': pad_left,
        'pad_top': pad_top,
        'frames_normalized': sample_frames,
        'heatmaps': sample_heatmaps,
        'coords': processed_coords
    }


class TrackNetVisualizer:
    def __init__(self):
        self.data_dir = Path(USER_CONFIG["data_dir"])
        self.current_sample = USER_CONFIG["start_sample"]
        self.playing = USER_CONFIG["auto_play"]
        self.delay = USER_CONFIG["playback_speed"]  # ms

        # 加载数据集
        try:
            print(f"🔍 正在加载数据集...")
            sys.path.append('.')  # 添加当前目录到路径
            from dataset_controller.ball_tracking_data_reader import BallTrackingDataset
            print(f"✓ 成功导入BallTrackingDataset")

            # 检查数据目录是否存在
            if not self.data_dir.exists():
                raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
            print(f"✓ 数据目录存在: {self.data_dir}")

            # 找到match目录
            match_dirs = sorted([d for d in self.data_dir.iterdir()
                                 if d.is_dir() and d.name.startswith('match')])

            if not match_dirs:
                raise ValueError(f"在目录中未找到match目录: {self.data_dir}")

            print(f"✓ 找到 {len(match_dirs)} 个match目录: {[d.name for d in match_dirs]}")

            # 使用配置指定的match目录
            match_index = USER_CONFIG["match_index"]
            if match_index >= len(match_dirs):
                print(f"⚠️ match_index {match_index} 超出范围，使用第0个")
                match_index = 0

            selected_match = match_dirs[match_index]
            print(f"🎯 正在加载: {selected_match}")

            self.dataset = BallTrackingDataset(str(selected_match), config=DATASET_CONFIG)

            if len(self.dataset) == 0:
                raise ValueError(f"数据集为空: {selected_match}")

            print(f"✓ 数据集配置:")
            print(f"  - 数据目录: {self.data_dir}")
            print(f"  - 加载match: {selected_match.name} ({match_index + 1}/{len(match_dirs)})")
            print(f"  - 样本总数: {len(self.dataset)}")
            print(f"  - 起始样本: {self.current_sample}")
            print(f"  - 自动播放: {self.playing}")
            print(f"  - 播放速度: {self.delay}ms")

            # 显示使用的配置
            print(f"✓ 使用train.py中的配置:")
            print(f"  - 输入尺寸: {CONFIG['input_height']}x{CONFIG['input_width']}")
            print(f"  - 热图半径: {CONFIG['heatmap_radius']}")
            print(f"  - 输入帧数: {DATASET_CONFIG['input_frames']}")
            print(f"  - 输出帧数: {DATASET_CONFIG['output_frames']}")

        except ImportError as e:
            print(f"❌ 导入错误: {e}")
            print("请确保dataset_controller.ball_tracking_data_reader模块可用")
            print("检查是否在正确的工作目录中运行此脚本")
            sys.exit(1)
        except FileNotFoundError as e:
            print(f"❌ 文件错误: {e}")
            print("请检查USER_CONFIG中的data_dir路径是否正确")
            sys.exit(1)
        except Exception as e:
            print(f"❌ 加载数据集失败: {e}")
            print(f"错误类型: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def run(self):
        """主运行循环"""
        print("\n🎮 控制说明:")
        print("  Space: Play/Pause")
        print("  A/D: Previous/Next sample")
        print("  Q: Quit")
        print("  +/-: Increase/Decrease playback speed")
        print("\n✅ 开始运行...")

        while True:
            if self.current_sample >= len(self.dataset):
                self.current_sample = 0

            # 获取当前样本
            frames, labels = self.dataset[self.current_sample]

            print(f"🔍 Debug - Sample {self.current_sample}:")
            print(f"  - frames shape: {frames.shape}")
            print(f"  - frames dtype: {frames.dtype}")
            print(f"  - labels type: {type(labels)}")
            if isinstance(labels, (list, tuple)):
                print(f"  - labels length: {len(labels)}")
                for i, label in enumerate(labels):
                    if isinstance(label, dict):
                        print(
                            f"    - label[{i}]: x={label.get('x', 'N/A')}, y={label.get('y', 'N/A')}, vis={label.get('visibility', 'N/A')}")

            # 使用train.py的处理逻辑
            result = process_single_sample_like_train(frames, labels)

            # 可视化
            self.visualize_sample(result)

            # 处理按键
            key = cv2.waitKey(self.delay) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                self.playing = not self.playing
            elif key == ord('a'):
                self.current_sample = max(0, self.current_sample - 1)
                self.playing = False
            elif key == ord('d'):
                self.current_sample = min(len(self.dataset) - 1, self.current_sample + 1)
                self.playing = False
            elif key == ord('+') or key == ord('='):
                self.delay = max(10, self.delay - 20)
            elif key == ord('-'):
                self.delay = min(1000, self.delay + 20)

            if self.playing:
                self.current_sample += 1

        cv2.destroyAllWindows()

    def visualize_sample(self, result):
        """可视化样本"""
        frames_normalized = result['frames_normalized']
        heatmaps = result['heatmaps']
        coords = result['coords']

        # 转换为numpy用于显示
        frames_np = (frames_normalized * 255).clamp(0, 255).byte().numpy()

        print(f"🔍 Visualize Debug:")
        print(f"  - frames_normalized shape: {frames_normalized.shape}")
        print(f"  - frames_np shape: {frames_np.shape}")
        print(f"  - heatmaps shape: {heatmaps.shape}")

        display_images = []

        for i in range(DATASET_CONFIG["input_frames"]):  # 使用配置中的帧数
            # 获取当前帧
            if len(frames_np.shape) == 4:  # (B, C, H, W)
                frame = frames_np[0, i]  # 取第一个batch的第i帧
            elif len(frames_np.shape) == 3:  # (C, H, W)
                frame = frames_np[i]  # 直接取第i帧
            else:
                print(f"❌ Unexpected frame shape: {frames_np.shape}")
                return

            print(f"  - frame[{i}] shape: {frame.shape}")

            # 处理灰度图 -> RGB
            if len(frame.shape) == 2:  # 灰度图 (H, W)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif len(frame.shape) == 3:  # 彩色图 (C, H, W) -> (H, W, C)
                if frame.shape[0] == 3:  # RGB channels first
                    frame_rgb = frame.transpose(1, 2, 0)
                else:  # Already (H, W, C)
                    frame_rgb = frame
                if frame_rgb.shape[2] == 1:  # Single channel
                    frame_rgb = cv2.cvtColor(frame_rgb.squeeze(), cv2.COLOR_GRAY2BGR)
            else:
                print(f"❌ Cannot handle frame shape: {frame.shape}")
                continue

            # 在图像上绘制坐标点
            frame_with_coords = frame_rgb.copy()
            coord = coords[i] if i < len(coords) else {'visibility': 0, 'x_orig': -1, 'y_orig': -1, 'x_scaled': -1,
                                                       'y_scaled': -1, 'x_norm': -1, 'y_norm': -1}

            if coord['visibility'] >= 0.5:
                x_scaled = int(coord['x_scaled'])
                y_scaled = int(coord['y_scaled'])

                # 绘制十字标记
                cv2.drawMarker(frame_with_coords, (x_scaled, y_scaled),
                               (0, 255, 0), cv2.MARKER_CROSS, 10, 2)

                # 绘制圆圈
                cv2.circle(frame_with_coords, (x_scaled, y_scaled),
                           CONFIG["heatmap_radius"], (255, 0, 0), 2)

            # 添加分辨率信息
            h, w = frame_with_coords.shape[:2]
            cv2.putText(frame_with_coords, f"{w}x{h}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 添加坐标信息
            coord_text = f"Orig:({coord['x_orig']:.0f},{coord['y_orig']:.0f})"
            cv2.putText(frame_with_coords, coord_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if coord['visibility'] >= 0.5:
                scaled_text = f"Scaled:({coord['x_scaled']:.0f},{coord['y_scaled']:.0f})"
                cv2.putText(frame_with_coords, scaled_text, (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                norm_text = f"Norm:({coord['x_norm']:.3f},{coord['y_norm']:.3f})"
                cv2.putText(frame_with_coords, norm_text, (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 获取对应的热力图
            heatmap = heatmaps[i].numpy()
            heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

            # 水平拼接图像和热力图
            combined = np.hstack([frame_with_coords, heatmap_colored])
            display_images.append(combined)

        # 垂直拼接所有帧
        final_display = np.vstack(display_images)

        # 添加整体信息
        info_text = f"Sample: {self.current_sample}/{len(self.dataset) - 1} | "
        info_text += f"{'Playing' if self.playing else 'Paused'} | "
        info_text += f"Speed: {1000 // self.delay}fps | "
        info_text += f"Using train.py functions"

        cv2.putText(final_display, info_text, (10, final_display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 显示
        cv2.imshow('TrackNet Test (Using train.py functions)', final_display)


def main():
    print("=" * 70)
    print("TrackNet坐标映射和热图测试工具 - 调用train.py函数版本")
    print("=" * 70)

    # 显示当前配置
    print("📋 当前USER_CONFIG:")
    for key, value in USER_CONFIG.items():
        print(f"  - {key}: {value}")

    print("\n📋 使用train.py中的配置:")
    print(f"  - CONFIG: batch_size={CONFIG['batch_size']}, input_size={CONFIG['input_height']}x{CONFIG['input_width']}")
    print(
        f"  - DATASET_CONFIG: input_frames={DATASET_CONFIG['input_frames']}, output_frames={DATASET_CONFIG['output_frames']}")
    print()

    visualizer = TrackNetVisualizer()
    visualizer.run()


if __name__ == "__main__":
    main()

"""
TrackNet坐标映射和热图测试工具 - 重构版本

🔄 主要改动:
- 直接导入train.py中的函数: calculate_equal_ratio_resize, create_gaussian_heatmap, collate_fn
- 移除了重复的函数定义
- 使用train.py中的CONFIG和DATASET_CONFIG
- 新增process_single_sample_like_train函数，使用train.py的collate_fn处理单样本

📝 配置说明:
- data_dir: 数据集根目录路径
- auto_play: 是否启动时自动播放
- playback_speed: 播放间隔(毫秒)
- start_sample: 起始样本索引
- match_index: 使用哪个match目录(0=第一个)

🎮 控制键:
- Space: 播放/暂停
- A/D: 上一个/下一个样本
- Q: 退出
- +/-: 增加/减少播放速度

📺 显示内容:
- 每帧显示：原图+坐标点 | 热力图
- 左上角显示分辨率信息
- 显示原始坐标、缩放后坐标、归一化坐标
- 底部显示当前样本和播放状态
- 现在显示"Using train.py functions"表示使用了train.py的函数

⚙️ 修改配置：
编辑文件顶部的 USER_CONFIG 字典即可

🔧 依赖要求:
- 需要train.py文件在同一目录下
- train.py必须包含以下函数: calculate_equal_ratio_resize, create_gaussian_heatmap, collate_fn
- train.py必须包含以下配置: CONFIG, DATASET_CONFIG
"""
