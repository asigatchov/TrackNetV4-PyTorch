#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_reorg_processor.py
数据集重组处理脚本

用法示例:
    python dataset_reorg_processor.py /path/to/dataset_reorg
    python dataset_reorg_processor.py /path/to/dataset_reorg --output_dir /output/path

依赖安装:
    pip install opencv-python pandas numpy scipy

功能说明:
1. 自动遍历dataset_reorg文件夹中的所有match
2. 将原始图像按比例缩放到512×288分辨率（保持宽高比，不拉伸）
3. 根据缩放比例转换CSV标注坐标
4. 为每帧生成对应的热力图：
   - Visibility=1: 生成以标注点为中心的高斯分布热力图（σ=3像素）
   - Visibility=0: 生成全零热力图
5. 创建训练数据集，包含inputs（缩放图像）和heatmaps（热力图）两个文件夹
6. 严格匹配：只处理同时存在图像文件和CSV记录的帧

输入结构:
dataset_reorg/
├── match1/
│   ├── inputs/video1/0.jpg,1.jpg...
│   └── labels/video1.csv
└── match2/...

输出结构:
dataset_reorg_train/
├── match1/
│   ├── inputs/video1/0.jpg,1.jpg... (512×288)
│   └── heatmaps/video1/0.jpg,1.jpg... (热力图)
└── match2/...

示例CSV格式 (video1.csv):
Frame,Visibility,X,Y
0,1,637.0,346.0
1,1,639.0,346.0
2,0,640.0,345.0  # Visibility=0生成全零热力图
"""

import argparse
import glob
import os

import cv2
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


def create_gaussian_heatmap(center_x, center_y, width=512, height=288, sigma=3):
    """
    生成以指定坐标为中心的2D高斯热力图

    Args:
        center_x: 中心点x坐标
        center_y: 中心点y坐标
        width: 热力图宽度
        height: 热力图高度
        sigma: 高斯分布的标准差

    Returns:
        numpy.ndarray: 归一化的热力图 (0-255)
    """
    # 创建坐标网格
    x = np.arange(0, width)
    y = np.arange(0, height)
    xx, yy = np.meshgrid(x, y)

    # 将坐标转换为列向量
    pos = np.dstack((xx, yy))

    # 定义多元高斯分布
    mean = [center_x, center_y]
    cov = [[sigma ** 2, 0], [0, sigma ** 2]]

    # 生成高斯分布
    rv = multivariate_normal(mean, cov)
    heatmap = rv.pdf(pos)

    # 归一化到0-255范围
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = (heatmap * 255).astype(np.uint8)

    return heatmap


def resize_image_keep_ratio(image, target_width=512, target_height=288):
    """
    按比例缩放图像到目标尺寸，保持宽高比

    Args:
        image: 输入图像
        target_width: 目标宽度
        target_height: 目标高度

    Returns:
        tuple: (缩放后的图像, 缩放比例)
    """
    h, w = image.shape[:2]

    # 计算缩放比例
    scale_w = target_width / w
    scale_h = target_height / h
    scale = min(scale_w, scale_h)  # 选择较小的比例以保持宽高比

    # 计算新的尺寸
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 缩放图像
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 创建目标尺寸的画布并居中放置图像
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # 计算居中位置
    start_x = (target_width - new_w) // 2
    start_y = (target_height - new_h) // 2

    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized

    return canvas, scale, start_x, start_y


def transform_coordinates(x, y, scale, offset_x, offset_y):
    """
    根据图像缩放比例转换坐标

    Args:
        x, y: 原始坐标
        scale: 缩放比例
        offset_x, offset_y: 在新画布中的偏移量

    Returns:
        tuple: 转换后的坐标
    """
    new_x = x * scale + offset_x
    new_y = y * scale + offset_y
    return new_x, new_y


def process_video(input_dir, label_file, output_inputs_dir, output_heatmaps_dir, video_name):
    """
    处理单个video的所有帧

    Args:
        input_dir: 输入图像文件夹路径
        label_file: 标注CSV文件路径
        output_inputs_dir: 输出图像文件夹路径
        output_heatmaps_dir: 输出热力图文件夹路径
        video_name: video名称
    """
    print(f"    处理 {video_name}...")

    # 创建输出目录
    video_inputs_dir = os.path.join(output_inputs_dir, video_name)
    video_heatmaps_dir = os.path.join(output_heatmaps_dir, video_name)
    os.makedirs(video_inputs_dir, exist_ok=True)
    os.makedirs(video_heatmaps_dir, exist_ok=True)

    # 读取标注文件
    if not os.path.exists(label_file):
        print(f"      警告: 标注文件不存在 {label_file}")
        return

    try:
        df = pd.read_csv(label_file)
    except Exception as e:
        print(f"      错误: 无法读取标注文件 {label_file}: {e}")
        return

    # 获取所有图像文件
    image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    image_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    # 获取所有图像帧号
    image_frames = set()
    for image_file in image_files:
        frame_num = int(os.path.splitext(os.path.basename(image_file))[0])
        image_frames.add(frame_num)

    # 获取所有CSV中的帧号（现在直接对应，无需转换）
    csv_frames = set(df['Frame'].values)

    # 找出严格匹配的帧
    matched_frames = image_frames & csv_frames
    only_image_frames = image_frames - csv_frames  # 有图像无标注
    only_csv_frames = csv_frames - image_frames  # 有标注无图像

    processed_count = 0

    # 只处理匹配的帧
    for frame_num in sorted(matched_frames):
        image_file = os.path.join(input_dir, f"{frame_num}.jpg")

        # 读取图像
        image = cv2.imread(image_file)
        if image is None:
            print(f"      警告: 无法读取图像 {image_file}")
            continue

        # 获取对应的标注（现在直接对应）
        frame_data = df[df['Frame'] == frame_num]
        frame_row = frame_data.iloc[0]

        # 缩放图像
        resized_image, scale, offset_x, offset_y = resize_image_keep_ratio(image)

        # 检查可见性，决定热力图生成方式
        if frame_row['Visibility'] == 1:
            # 可见帧：生成有标注点的热力图
            orig_x = frame_row['X']
            orig_y = frame_row['Y']

            # 检查坐标是否有效
            if pd.isna(orig_x) or pd.isna(orig_y):
                print(f"      警告: 帧 {frame_num} 坐标无效，生成空热力图")
                heatmap = np.zeros((288, 512), dtype=np.uint8)
            else:
                # 转换坐标
                new_x, new_y = transform_coordinates(orig_x, orig_y, scale, offset_x, offset_y)

                # 确保坐标在有效范围内
                new_x = max(0, min(511, new_x))
                new_y = max(0, min(287, new_y))

                # 生成热力图
                heatmap = create_gaussian_heatmap(new_x, new_y)
        else:
            # 不可见帧：生成全零热力图
            heatmap = np.zeros((288, 512), dtype=np.uint8)

        # 保存文件
        output_image_path = os.path.join(video_inputs_dir, f"{frame_num}.jpg")
        output_heatmap_path = os.path.join(video_heatmaps_dir, f"{frame_num}.jpg")

        cv2.imwrite(output_image_path, resized_image)
        cv2.imwrite(output_heatmap_path, heatmap)

        processed_count += 1

    # 每个video处理完就总结
    print(f"      完成处理 {processed_count} 帧", end="")

    # 报告不匹配的情况
    issues = []
    if only_image_frames:
        frames_str = ",".join(map(str, sorted(only_image_frames)))
        issues.append(f"图像文件无对应CSV记录: {frames_str}")

    if only_csv_frames:
        frames_str = ",".join(map(str, sorted(only_csv_frames)))
        issues.append(f"CSV记录无对应图像文件: {frames_str}")

    if issues:
        print(f"，跳过 {'; '.join(issues)}")
    else:
        print()


def process_match(match_dir, output_dir):
    """
    处理单个match文件夹

    Args:
        match_dir: 输入match文件夹路径
        output_dir: 输出根目录路径
    """
    match_name = os.path.basename(match_dir)
    print(f"  处理 {match_name}...")

    # 创建输出文件夹（match保持原名）
    output_match_dir = os.path.join(output_dir, match_name)
    output_inputs_dir = os.path.join(output_match_dir, "inputs")
    output_heatmaps_dir = os.path.join(output_match_dir, "heatmaps")

    os.makedirs(output_inputs_dir, exist_ok=True)
    os.makedirs(output_heatmaps_dir, exist_ok=True)

    # 获取inputs和labels目录
    inputs_dir = os.path.join(match_dir, "inputs")
    labels_dir = os.path.join(match_dir, "labels")

    if not os.path.exists(inputs_dir):
        print(f"    警告: inputs目录不存在 {inputs_dir}")
        return

    if not os.path.exists(labels_dir):
        print(f"    警告: labels目录不存在 {labels_dir}")
        return

    # 获取所有video文件夹
    video_dirs = [d for d in os.listdir(inputs_dir)
                  if os.path.isdir(os.path.join(inputs_dir, d))]
    video_dirs.sort()

    for video_name in video_dirs:
        video_input_dir = os.path.join(inputs_dir, video_name)
        video_label_file = os.path.join(labels_dir, f"{video_name}.csv")

        process_video(video_input_dir, video_label_file,
                      output_inputs_dir, output_heatmaps_dir, video_name)


def main():
    parser = argparse.ArgumentParser(description='处理数据集生成训练数据')
    parser.add_argument('dataset_path', type=str, help='dataset_reorg文件夹的路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录路径（默认为dataset_path的父目录）')

    args = parser.parse_args()

    dataset_path = args.dataset_path

    if not os.path.exists(dataset_path):
        print(f"错误: 数据集路径不存在 {dataset_path}")
        return

    # 设置输出目录（顶层文件夹加_train后缀）
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # 从dataset_reorg得到dataset_reorg_train
        dataset_name = os.path.basename(dataset_path.rstrip('/'))
        parent_dir = os.path.dirname(dataset_path)
        output_dir = os.path.join(parent_dir, f"{dataset_name}_train")

    os.makedirs(output_dir, exist_ok=True)

    print(f"开始处理数据集: {dataset_path}")
    print(f"输出目录: {output_dir}")

    # 获取所有match文件夹
    match_dirs = [os.path.join(dataset_path, d)
                  for d in os.listdir(dataset_path)
                  if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('match')]
    match_dirs.sort()

    if not match_dirs:
        print("错误: 没有找到match文件夹")
        return

    print(f"找到 {len(match_dirs)} 个match文件夹")

    # 处理每个match
    for match_dir in match_dirs:
        try:
            process_match(match_dir, output_dir)
        except Exception as e:
            print(f"  错误: 处理 {os.path.basename(match_dir)} 时出错: {e}")
            continue

    print("数据集处理完成!")


if __name__ == "__main__":
    main()
