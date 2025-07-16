#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import torch
import os
from tqdm import tqdm
from tracknet_v4 import TrackNetV4 as TrackNet


class TrackNetPredictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrackNet().to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def preprocess_frames(self, frames):
        processed = []
        for frame in frames:
            frame = cv2.resize(frame, (512, 288))
            frame = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1)
            processed.append(frame)
        return torch.cat(processed, dim=0).unsqueeze(0).to(self.device)

    def predict(self, frames):
        input_tensor = self.preprocess_frames(frames)
        with torch.no_grad():
            output = self.model(input_tensor)
        return output.squeeze(0).cpu().numpy()

    def detect_ball(self, heatmap, threshold=0.5):
        if heatmap.max() < threshold:
            return None
        max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        return (max_pos[1], max_pos[0])


class VideoProcessor:
    def __init__(self, model_path, dot_size=3):
        self.predictor = TrackNetPredictor(model_path)
        self.dot_size = dot_size  # 红点大小参数

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        with tqdm(total=total_frames, desc="🏸 Extracting frames", unit="frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                pbar.update(1)

        cap.release()
        print(f"✅ Extraction complete: {len(frames)} frames, {width}x{height}, {fps:.1f}FPS")
        return frames, (width, height), fps

    def group_frames(self, frames):
        groups = []
        for i in range(0, len(frames) - 2, 3):
            if i + 3 <= len(frames):
                groups.append(frames[i:i + 3])

        discarded = len(frames) - len(groups) * 3
        print(f"🏸 Grouping complete: {len(groups)} groups (3 frames/group), {discarded} frames discarded")
        return groups

    def scale_coordinates(self, coords, original_size):
        if coords is None:
            return None
        x, y = coords
        scale_x = original_size[0] / 512
        scale_y = original_size[1] / 288
        return (int(x * scale_x), int(y * scale_y))

    def draw_ball(self, frame, ball_pos):
        """Draw a red dot on the frame with adjustable size"""
        if ball_pos is not None:
            cv2.circle(frame, ball_pos, self.dot_size, (0, 0, 255), -1)  # 使用可调节的红点大小
        return frame

    def process_video(self, video_path, output_path="processed_video.mp4"):
        print("🏸 Starting shuttlecock detection...")
        print(f"🔴 Red dot size: {self.dot_size} pixels")
        frames, original_size, fps = self.extract_frames(video_path)
        frame_groups = self.group_frames(frames)
        processed_frames = []

        ball_detected_count = 0
        total_processed_frames = 0

        with tqdm(total=len(frame_groups), desc="🏸 Detecting shuttlecock", unit="groups") as pbar:
            for group in frame_groups:
                heatmaps = self.predictor.predict(group)

                for frame, heatmap in zip(group, heatmaps):
                    ball_pos_model = self.predictor.detect_ball(heatmap)
                    ball_pos_original = self.scale_coordinates(ball_pos_model, original_size)
                    processed_frame = self.draw_ball(frame.copy(), ball_pos_original)
                    processed_frames.append(processed_frame)

                    total_processed_frames += 1
                    if ball_pos_original:
                        ball_detected_count += 1

                pbar.update(1)

        detection_rate = (ball_detected_count / total_processed_frames) * 100
        print(f"🎯 Detection stats: {ball_detected_count}/{total_processed_frames} frames ({detection_rate:.1f}%)")

        self.save_video(processed_frames, output_path, fps, original_size)
        return output_path

    def save_video(self, frames, output_path, fps, size):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, size)

        if not out.isOpened():
            raise RuntimeError(f"Cannot create video file: {output_path}")

        with tqdm(total=len(frames), desc="🏸 Saving video", unit="frames") as pbar:
            for frame in frames:
                out.write(frame)
                pbar.update(1)

        out.release()
        print(f"✅ Video saved successfully: {output_path}")


def main():
    model_path = "best_model.pth"
    input_video = "dataset_predict/test.mp4"
    output_video = "dataset_predict/processed_video.mp4"

    # 硬编码红点大小参数 (像素)
    RED_DOT_SIZE = 7  # 可以调整这个值：1=很小, 3=小, 5=中等, 8=大, 12=很大

    print("=" * 60)
    print("🏸 Badminton Shuttlecock Detection & Tracking System")
    print("=" * 60)
    print(f"📂 Model file: {model_path}")
    print(f"🎬 Input video: {input_video}")
    print(f"💾 Output video: {output_video}")
    print(f"🔴 Red dot size: {RED_DOT_SIZE} pixels")
    print("-" * 60)

    if not os.path.exists(input_video):
        print(f"❌ Error: Input video file not found: {input_video}")
        return

    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found: {model_path}")
        return

    try:
        processor = VideoProcessor(model_path, dot_size=RED_DOT_SIZE)
        output_path = processor.process_video(input_video, output_video)
        print("=" * 60)
        print(f"🎉 Processing complete! Output video: {output_path}")
        print("=" * 60)
    except Exception as e:
        print(f"❌ Processing error: {str(e)}")


if __name__ == "__main__":
    main()
