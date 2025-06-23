#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像序列播放器 - 热力图透明叠加版本

功能说明:
- 自动扫描match文件夹下的inputs和heatmaps子目录
- 匹配同名文件夹中的图像序列（如1_05_03, 2_10_07等）
- 将热力图以透明方式叠加在原图上显示
- 支持连续播放、暂停、切换序列等交互控制
- 支持保存当前帧、调整透明度、快进快退等功能

调用示例:
1. 基本用法:
   python dataset_train_player.py /path/to/match1

2. 指定播放帧率:
   python dataset_train_player.py /path/to/match1 --fps 15

3. 指定透明度:
   python dataset_train_player.py /path/to/match1 --alpha 0.4

文件夹结构要求:
match1/
├── inputs/
│   ├── 1_05_03/     # 序列标识符（任意命名）
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   └── 2_10_07/
│       └── ...
└── heatmaps/
    ├── 1_05_03/     # 必须与inputs中的文件夹名对应
    │   ├── 0.jpg
    │   ├── 1.jpg
    │   └── ...
    └── 2_10_07/
        └── ...
"""

import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path
import glob
from typing import List, Tuple, Optional


class SequencePlayer:
    def __init__(self, match_path: str, alpha: float = 0.3):
        self.match_path = Path(match_path)
        self.current_sequence_index = 0
        self.sequence_folders = []
        self.fps = 30  # 默认帧率
        self.alpha = alpha  # 热力图透明度
        self.show_original_only = False  # 是否只显示原图

    def scan_sequence_folders(self) -> List[str]:
        """扫描所有图像序列文件夹"""
        inputs_path = self.match_path / "inputs"
        heatmaps_path = self.match_path / "heatmaps"

        if not inputs_path.exists() or not heatmaps_path.exists():
            print(f"错误: inputs 或 heatmaps 文件夹不存在于 {self.match_path}")
            return []

        # 获取inputs中的所有序列文件夹
        sequence_folders = []
        for folder in inputs_path.iterdir():
            if folder.is_dir():
                # 检查对应的heatmaps文件夹是否存在
                corresponding_heatmap = heatmaps_path / folder.name
                if corresponding_heatmap.exists():
                    sequence_folders.append(folder.name)
                else:
                    print(f"警告: {folder.name} 在heatmaps中没有对应文件夹")

        sequence_folders.sort()  # 按名称排序
        return sequence_folders

    def load_image_sequence(self, folder_path: Path) -> List[np.ndarray]:
        """加载图像序列"""
        images = []

        # 支持多种图像格式
        image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []

        for pattern in image_patterns:
            image_files.extend(glob.glob(str(folder_path / pattern)))

        # 按数字顺序排序
        def extract_number(filename):
            try:
                return int(Path(filename).stem)
            except ValueError:
                return float('inf')

        image_files.sort(key=extract_number)

        for img_file in image_files:
            img = cv2.imread(img_file)
            if img is not None:
                images.append(img)
            else:
                print(f"警告: 无法读取图像 {img_file}")

        return images

    def resize_to_match(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """调整图像大小以匹配（使用原图的尺寸作为目标）"""
        h1, w1 = img1.shape[:2]  # 原图尺寸
        h2, w2 = img2.shape[:2]  # 热力图尺寸

        # 将热力图调整为与原图相同的尺寸
        img2_resized = cv2.resize(img2, (w1, h1))

        return img1, img2_resized

    def apply_colormap_to_heatmap(self, heatmap: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """为热力图应用颜色映射"""
        # 如果是彩色图像，转换为灰度
        if len(heatmap.shape) == 3:
            heatmap_gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        else:
            heatmap_gray = heatmap

        # 归一化到0-255范围
        heatmap_norm = cv2.normalize(heatmap_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 应用颜色映射
        heatmap_colored = cv2.applyColorMap(heatmap_norm, colormap)

        return heatmap_colored

    def overlay_images(self, input_img: np.ndarray, heatmap_img: np.ndarray,
                       alpha: float = None) -> np.ndarray:
        """将热力图叠加在原图上"""
        if alpha is None:
            alpha = self.alpha

        # 调整图像大小以匹配
        input_resized, heatmap_resized = self.resize_to_match(input_img, heatmap_img)

        # 如果只显示原图
        if self.show_original_only:
            return input_resized

        # 为热力图应用颜色映射（如果需要）
        heatmap_colored = self.apply_colormap_to_heatmap(heatmap_resized)

        # 使用alpha混合进行叠加
        # result = (1-alpha) * input + alpha * heatmap
        overlayed = cv2.addWeighted(input_resized, 1 - alpha, heatmap_colored, alpha, 0)

        return overlayed

    def play_sequence(self, sequence_name: str):
        """播放单个图像序列"""
        inputs_path = self.match_path / "inputs" / sequence_name
        heatmaps_path = self.match_path / "heatmaps" / sequence_name

        print(f"加载序列: {sequence_name}")

        # 加载图像序列
        input_images = self.load_image_sequence(inputs_path)
        heatmap_images = self.load_image_sequence(heatmaps_path)

        if not input_images or not heatmap_images:
            print(f"错误: 无法加载 {sequence_name} 的图像序列")
            return False

        # 确保两个序列长度相同
        min_length = min(len(input_images), len(heatmap_images))
        if len(input_images) != len(heatmap_images):
            print(f"警告: inputs({len(input_images)}) 和 heatmaps({len(heatmap_images)}) 图像数量不同，使用较短的序列")

        print(f"播放 {min_length} 帧图像")

        # 创建窗口
        window_name = f"Heatmap Overlay Player - {sequence_name} ({self.current_sequence_index + 1}/{len(self.sequence_folders)})"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        frame_index = 0
        paused = False

        while frame_index < min_length:
            if not paused:
                # 叠加图像
                combined_frame = self.overlay_images(
                    input_images[frame_index],
                    heatmap_images[frame_index]
                )

                # 添加信息文本
                alpha_text = "Original Only" if self.show_original_only else f"Alpha: {self.alpha:.2f}"
                info_text = f"Frame: {frame_index + 1}/{min_length} | {alpha_text} | Sequence: {sequence_name}"

                # 添加文本背景以提高可读性
                text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                cv2.rectangle(combined_frame, (5, 5), (text_size[0] + 15, 35), (0, 0, 0), -1)
                cv2.putText(combined_frame, info_text, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                cv2.imshow(window_name, combined_frame)
                frame_index += 1

            # 按键处理
            key = cv2.waitKey(int(1000 / self.fps)) & 0xFF

            if key == ord('q') or key == 27:  # q 或 ESC 退出
                cv2.destroyWindow(window_name)
                return False
            elif key == ord(' '):  # 空格暂停/继续
                paused = not paused
                print("暂停" if paused else "继续")
            elif key == ord('n') or key == ord('.'):  # n 或 . 下一个序列
                cv2.destroyWindow(window_name)
                return True
            elif key == ord('p') or key == ord(','):  # p 或 , 上一个序列
                cv2.destroyWindow(window_name)
                return "previous"
            elif key == ord('r'):  # r 重新开始当前序列
                frame_index = 0
                paused = False
            elif key == ord('o'):  # o 切换显示模式（叠加/仅原图）
                self.show_original_only = not self.show_original_only
                print(f"显示模式: {'仅原图' if self.show_original_only else '热力图叠加'}")
            elif key == ord('s'):  # s 保存当前帧
                save_path = f"frame_{sequence_name}_{frame_index}.jpg"
                cv2.imwrite(save_path, combined_frame)
                print(f"保存帧到: {save_path}")
            elif key == ord('f'):  # f 快进
                frame_index = min(frame_index + 10, min_length - 1)
            elif key == ord('b'):  # b 快退
                frame_index = max(frame_index - 10, 0)
            elif key == ord('+') or key == ord('='):  # + 增加透明度
                self.alpha = min(1.0, self.alpha + 0.05)
                print(f"透明度: {self.alpha:.2f}")
            elif key == ord('-') or key == ord('_'):  # - 减少透明度
                self.alpha = max(0.0, self.alpha - 0.05)
                print(f"透明度: {self.alpha:.2f}")

        # 序列播放完毕
        print(f"序列 {sequence_name} 播放完毕")
        cv2.destroyWindow(window_name)
        return True

    def run(self):
        """运行播放器"""
        if not self.match_path.exists():
            print(f"错误: 路径不存在 {self.match_path}")
            return

        print(f"扫描文件夹: {self.match_path}")
        self.sequence_folders = self.scan_sequence_folders()

        if not self.sequence_folders:
            print("没有找到有效的图像序列文件夹")
            return

        print(f"找到 {len(self.sequence_folders)} 个图像序列文件夹:")
        for i, folder in enumerate(self.sequence_folders):
            print(f"  {i + 1}. {folder}")

        print(f"\n初始透明度: {self.alpha:.2f}")
        print("\n控制说明:")
        print("  空格键: 暂停/继续")
        print("  n 或 .: 下一个序列")
        print("  p 或 ,: 上一个序列")
        print("  r: 重新开始当前序列")
        print("  o: 切换显示模式 (叠加/仅原图)")
        print("  + 或 =: 增加热力图透明度")
        print("  - 或 _: 减少热力图透明度")
        print("  s: 保存当前帧")
        print("  f: 快进10帧")
        print("  b: 快退10帧")
        print("  q 或 ESC: 退出")
        print()

        # 播放所有序列
        while self.current_sequence_index < len(self.sequence_folders):
            sequence_name = self.sequence_folders[self.current_sequence_index]
            result = self.play_sequence(sequence_name)

            if result is False:  # 用户退出
                break
            elif result == "previous":  # 上一个序列
                self.current_sequence_index = max(0, self.current_sequence_index - 1)
            else:  # 下一个序列
                self.current_sequence_index += 1

        cv2.destroyAllWindows()
        print("播放器退出")


def main():
    parser = argparse.ArgumentParser(description='图像序列播放器 - 热力图透明叠加版本')
    parser.add_argument('match_path', help='match文件夹路径')
    parser.add_argument('--fps', type=int, default=30, help='播放帧率 (默认: 30)')
    parser.add_argument('--alpha', type=float, default=0.3, help='热力图透明度 (0.0-1.0, 默认: 0.3)')

    args = parser.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        print("错误: alpha值必须在0.0到1.0之间")
        return

    player = SequencePlayer(args.match_path, args.alpha)
    player.fps = args.fps
    player.run()


if __name__ == "__main__":
    main()
