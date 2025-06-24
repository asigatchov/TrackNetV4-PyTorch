#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from tracknet import TrackNet


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

    def save_heatmaps(self, heatmaps, output_dir="outputs"):
        """保存热力图"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        for i, heatmap in enumerate(heatmaps):
            plt.figure(figsize=(8, 6))
            plt.imshow(heatmap, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title(f'Heatmap {i + 1}')
            plt.axis('off')

            save_path = f"{output_dir}/heatmap_{i + 1}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"保存热力图: {save_path}")


def main():
    # 硬编码模型路径
    model_path = "best_model.pth"

    # 初始化预测器
    predictor = TrackNetPredictor(model_path)

    # 硬编码加载三张JPG
    frame1 = cv2.imread("predict/266.jpg")
    frame2 = cv2.imread("predict/267.jpg")
    frame3 = cv2.imread("predict/268.jpg")

    frames = [frame1, frame2, frame3]

    # 预测
    heatmaps = predictor.predict(frames)
    print(f"输出热力图: {heatmaps.shape}")

    # 硬编码输出路径并保存热力图
    output_dir = "predict/heatmap_outputs"
    predictor.save_heatmaps(heatmaps, output_dir)

    # 检测球位置
    for i, heatmap in enumerate(heatmaps):
        ball_pos = predictor.detect_ball(heatmap)
        if ball_pos:
            print(f"帧{i + 1}: 球在 ({ball_pos[0]}, {ball_pos[1]})")
        else:
            print(f"帧{i + 1}: 无球")


if __name__ == "__main__":
    main()
