#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import torch
import os
import shutil
from TrackNet import TrackNet


class TrackNetPredictor:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrackNet().to(self.device)

        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.model.load_state_dict(state_dict)

        self.model.eval()

    def preprocess_frames(self, frames):
        """输入3帧，输出 (1,9,288,512)"""
        processed = []
        for frame in frames:
            # 缩放到512x288
            frame = cv2.resize(frame, (512, 288))
            # 归一化
            frame = frame.astype(np.float32) / 255.0
            # (H,W,C) -> (C,H,W)
            frame = torch.from_numpy(frame).permute(2, 0, 1)
            processed.append(frame)

        # 合并为 (9,288,512)
        input_tensor = torch.cat(processed, dim=0)
        # 添加batch维度 (1,9,288,512)
        return input_tensor.unsqueeze(0).to(self.device)

    def predict(self, frames):
        """输入3帧，返回3张热力图"""
        input_tensor = self.preprocess_frames(frames)

        with torch.no_grad():
            output = self.model(input_tensor)  # (1,3,288,512)

        return output.squeeze(0).cpu().numpy()  # (3,288,512)

    def detect_ball(self, heatmap, threshold=0.5):
        """检测热力图中是否有球，返回位置(x,y)或None"""
        if heatmap.max() < threshold:
            return None

        # 找最大值位置
        max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        return (max_pos[1], max_pos[0])  # (x,y)


class VideoProcessor:
    def __init__(self, model_path, work_dir="video_processing"):
        """初始化视频处理器"""
        self.predictor = TrackNetPredictor(model_path)
        self.work_dir = work_dir
        self.temp_dir = os.path.join(work_dir, "temp")

    def setup_directories(self):
        """创建工作目录"""
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

    def extract_frames(self, video_path):
        """从视频中提取帧并返回帧信息"""
        cap = cv2.VideoCapture(video_path)

        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        frame_count = 0

        print(f"开始提取视频帧，总帧数: {total_frames}, 分辨率: {width}x{height}, FPS: {fps}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames.append(frame)
            frame_count += 1

            if frame_count % 100 == 0:
                print(f"已提取 {frame_count} 帧...")

        cap.release()

        print(f"帧提取完成，共 {frame_count} 帧")
        return frames, (width, height), fps

    def group_frames(self, frames):
        """将帧按3帧一组分组，不足3帧的丢弃"""
        groups = []
        group_indices = []

        for i in range(0, len(frames) - 2, 3):  # 每3帧一组，步长为3
            group = frames[i:i + 3]
            if len(group) == 3:
                groups.append(group)
                group_indices.append(list(range(i, i + 3)))

        discarded = len(frames) - len(groups) * 3
        print(f"总共 {len(frames)} 帧，分成 {len(groups)} 组，丢弃 {discarded} 帧")
        return groups, group_indices

    def scale_coordinates(self, coords, model_size=(512, 288), original_size=(1920, 1080)):
        """将模型输出的坐标缩放到原始分辨率"""
        if coords is None:
            return None

        x, y = coords
        scale_x = original_size[0] / model_size[0]
        scale_y = original_size[1] / model_size[1]

        scaled_x = int(x * scale_x)
        scaled_y = int(y * scale_y)

        return (scaled_x, scaled_y)

    def draw_ball(self, frame, ball_pos, radius=8, color=(0, 255, 0), thickness=2):
        """在帧上绘制球的位置"""
        if ball_pos is not None:
            # 绘制实心圆
            cv2.circle(frame, ball_pos, radius, color, -1)
            # 绘制外圆环增强可见性
            cv2.circle(frame, ball_pos, radius + 2, (255, 255, 255), thickness)
        return frame

    def process_video(self, video_path, output_filename="processed_video.mp4"):
        """处理整个视频"""
        # 设置工作目录
        self.setup_directories()

        try:
            # 提取帧
            frames, original_size, fps = self.extract_frames(video_path)

            # 分组
            frame_groups, group_indices = self.group_frames(frames)

            processed_frames = []

            # 处理每一组
            for group_idx, (group, indices) in enumerate(zip(frame_groups, group_indices)):
                if (group_idx + 1) % 10 == 0:
                    print(f"处理进度: {group_idx + 1}/{len(frame_groups)} 组")

                # 预测热力图
                heatmaps = self.predictor.predict(group)

                # 处理每一帧
                for frame_idx, (frame, heatmap, original_idx) in enumerate(zip(group, heatmaps, indices)):
                    # 检测球位置（模型坐标）
                    ball_pos_model = self.predictor.detect_ball(heatmap)

                    # 缩放坐标到原始分辨率
                    ball_pos_original = self.scale_coordinates(
                        ball_pos_model,
                        model_size=(512, 288),
                        original_size=original_size
                    )

                    # 在原始帧上绘制球
                    processed_frame = self.draw_ball(frame.copy(), ball_pos_original)
                    processed_frames.append(processed_frame)

                    # 打印检测结果
                    if ball_pos_original:
                        print(f"帧 {original_idx + 1}: 球位置 {ball_pos_original}")
                    else:
                        print(f"帧 {original_idx + 1}: 未检测到球")

            # 保存处理后的视频
            output_path = os.path.join(self.work_dir, output_filename)
            self.save_video(processed_frames, output_path, fps, original_size)

            print(f"视频处理完成！输出文件: {output_path}")
            return output_path

        finally:
            # 清理临时文件
            self.cleanup_temp()

    def save_video(self, frames, output_path, fps, size):
        """将帧序列保存为MP4视频"""
        # 使用H.264编码器，兼容性更好
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, size)

        if not out.isOpened():
            raise RuntimeError(f"无法创建视频文件: {output_path}")

        print(f"开始保存视频，共 {len(frames)} 帧...")

        for i, frame in enumerate(frames):
            out.write(frame)
            if (i + 1) % 100 == 0:
                print(f"已保存 {i + 1}/{len(frames)} 帧")

        out.release()
        print("视频保存完成")

    def cleanup_temp(self):
        """清理临时文件夹"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print("临时文件清理完成")


def main():
    # 硬编码路径配置
    model_path = "best_model.pth"  # 模型文件路径
    input_video = "dataset_predict/test.mp4"  # 输入视频路径
    output_video = "dataset_predict/processed_video.mp4"  # 输出视频文件名
    work_directory = "dataset_predict"  # 工作目录名

    print("=== 视频球位置检测与跟踪 ===")
    print(f"模型路径: {model_path}")
    print(f"输入视频: {input_video}")
    print(f"工作目录: {work_directory}")

    # 检查输入文件是否存在
    if not os.path.exists(input_video):
        print(f"错误: 输入视频文件不存在: {input_video}")
        return

    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return

    try:
        # 创建视频处理器
        processor = VideoProcessor(model_path, work_directory)

        # 处理视频
        output_path = processor.process_video(input_video, output_video)

        print(f"\n处理完成!")
        print(f"输出视频: {output_path}")
        print(f"工作目录中只保留最终的MP4文件，临时文件已清理")

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
