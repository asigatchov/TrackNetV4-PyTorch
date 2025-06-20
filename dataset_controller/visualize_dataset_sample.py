from pathlib import Path

import cv2
import numpy as np
import torch
from ball_tracking_data_reader import BallTrackingDataset, CONFIG

# Locate the project root directory
base_dir = Path(__file__).resolve().parent.parent


class ConfigurableDatasetViewer:
    """配置化数据集可视化器，支持动态输入/输出帧数"""

    def __init__(self, dataset):
        self.dataset = dataset
        self.config = dataset.config
        self.input_frames = self.config["input_frames"]
        self.output_frames = self.config["output_frames"]
        self.channels_per_frame = 3

        print(f"Viewer initialized: {self.input_frames} input frames -> {self.output_frames} output labels")

    def _reshape_frames(self, frames):
        """将帧数据从 (C*input_frames, H, W) 重塑为 (input_frames, H, W, C)"""
        if isinstance(frames, torch.Tensor):
            frames = frames.detach().cpu().numpy()

        H, W = frames.shape[1], frames.shape[2]

        # Reshape from (C*input_frames, H, W) to (input_frames, C, H, W)
        frames_reshaped = frames.reshape(self.input_frames, self.channels_per_frame, H, W)

        # Convert to (input_frames, H, W, C) for visualization
        frames_hwc = np.transpose(frames_reshaped, (0, 2, 3, 1))

        return frames_hwc

    def _normalize_frame(self, frame):
        """标准化帧数据到 [0, 255] uint8"""
        if frame.dtype in (np.float32, np.float64):
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    def _get_output_frame_indices(self, center_frame_idx):
        """获取输出帧在输入帧中的索引"""
        input_half = self.input_frames // 2
        center_input_idx = input_half  # 输入帧中的中心索引

        if self.output_frames == self.input_frames:
            # 匹配情况：所有输入帧都有标签
            return list(range(self.input_frames))
        elif self.output_frames == 1:
            # 单输出：中心帧
            return [center_input_idx]
        elif self.output_frames == 2:
            # 双输出：中心两帧
            return [center_input_idx, center_input_idx + 1] if center_input_idx + 1 < self.input_frames else [
                center_input_idx - 1, center_input_idx]
        else:
            # 多输出：以中心为核心
            output_half = self.output_frames // 2
            start_idx = max(0, center_input_idx - output_half)
            end_idx = min(self.input_frames, start_idx + self.output_frames)
            return list(range(start_idx, end_idx))

    def _draw_ball_positions(self, frame_bgr, labels, frame_indices, W, current_view):
        """在帧上绘制球的位置"""
        if not isinstance(labels, list):
            labels = [labels]

        output_indices = self._get_output_frame_indices(self.input_frames // 2)

        for i, (frame_idx, label) in enumerate(zip(output_indices, labels)):
            if frame_idx not in frame_indices:
                continue

            if isinstance(label, dict) and label.get('visibility', 0).item() == 1:
                x = label.get('x')
                y = label.get('y')
                if x is not None and y is not None:
                    xi, yi = int(x.item()), int(y.item())

                    # 调整坐标位置
                    if current_view == 'all':
                        # 找到该帧在显示中的位置
                        display_idx = frame_indices.index(frame_idx)
                        xi_adjusted = xi + display_idx * W
                        yi_adjusted = yi
                    else:
                        xi_adjusted, yi_adjusted = xi, yi

                    # 绘制球的位置
                    color = (0, 0, 255) if i == len(labels) // 2 else (0, 255, 255)  # 中心帧红色，其他黄色
                    cv2.circle(frame_bgr, (xi_adjusted, yi_adjusted), 5, color, -1)
                    cv2.putText(
                        frame_bgr,
                        f"({xi},{yi})",
                        (xi_adjusted + 5, yi_adjusted - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                        cv2.LINE_AA
                    )

    def play_dataset(self, delay_ms: int = 30, show_frame: str = 'center'):
        """
        播放数据集中的所有帧

        :param delay_ms: 帧间延迟（毫秒）
        :param show_frame: 显示模式 ('center', 'all', 或帧索引)
        """
        window_name = f"Dataset Viewer ({self.input_frames}in{self.output_frames}out) - q:quit, p:pause, a:all, 0-{self.input_frames - 1}:frame"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        paused = False
        current_show_frame = show_frame

        idx = 0
        while idx < len(self.dataset):
            if not paused:
                frames, labels = self.dataset[idx]
                frames_hwc = self._reshape_frames(frames)

                H, W = frames_hwc.shape[1], frames_hwc.shape[2]

                # 选择显示的帧
                if current_show_frame == 'all':
                    # 显示所有输入帧
                    display_frames = [self._normalize_frame(frames_hwc[i]) for i in range(self.input_frames)]
                    display_frame = np.hstack(display_frames)
                    title_suffix = f" - All {self.input_frames} Frames"
                    frame_indices = list(range(self.input_frames))
                elif current_show_frame == 'center':
                    # 显示中心帧
                    center_idx = self.input_frames // 2
                    display_frame = self._normalize_frame(frames_hwc[center_idx])
                    title_suffix = f" - Center Frame ({center_idx})"
                    frame_indices = [center_idx]
                elif current_show_frame.isdigit():
                    # 显示指定索引的帧
                    frame_idx = int(current_show_frame)
                    if 0 <= frame_idx < self.input_frames:
                        display_frame = self._normalize_frame(frames_hwc[frame_idx])
                        title_suffix = f" - Frame {frame_idx}"
                        frame_indices = [frame_idx]
                    else:
                        display_frame = self._normalize_frame(frames_hwc[self.input_frames // 2])
                        title_suffix = " - Center Frame (default)"
                        frame_indices = [self.input_frames // 2]
                else:
                    # 默认显示中心帧
                    center_idx = self.input_frames // 2
                    display_frame = self._normalize_frame(frames_hwc[center_idx])
                    title_suffix = " - Center Frame"
                    frame_indices = [center_idx]

                # 转换为BGR用于OpenCV显示
                if display_frame.ndim == 3 and display_frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = display_frame.copy()

                # 绘制球的位置
                self._draw_ball_positions(frame_bgr, labels, frame_indices, W, current_show_frame)

                # 添加帧信息
                info_text = f"Sample {idx + 1}/{len(self.dataset)}{title_suffix}"
                cv2.putText(
                    frame_bgr,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                # 添加配置信息
                config_text = f"Config: {self.input_frames}in -> {self.output_frames}out"
                cv2.putText(
                    frame_bgr,
                    config_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA
                )

                # 显示帧
                cv2.imshow(window_name, frame_bgr)

            # 处理按键
            key = cv2.waitKey(delay_ms if not paused else 0) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('p') or key == 32:  # 'p' 或空格
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('a'):
                current_show_frame = 'all'
                print(f"Showing all {self.input_frames} frames")
            elif key == ord('c'):
                current_show_frame = 'center'
                print("Showing center frame")
            elif chr(key).isdigit():
                frame_idx = int(chr(key))
                if 0 <= frame_idx < self.input_frames:
                    current_show_frame = str(frame_idx)
                    print(f"Showing frame {frame_idx}")
                else:
                    print(f"Frame index {frame_idx} out of range (0-{self.input_frames - 1})")

            if not paused:
                idx += 1

        cv2.destroyAllWindows()

    def visualize_sample(self, sample_idx: int = 0):
        """
        可视化单个样本，显示所有输入帧和标签信息
        """
        if sample_idx >= len(self.dataset):
            print(f"Sample index {sample_idx} out of range. Dataset has {len(self.dataset)} samples.")
            return

        frames, labels = self.dataset[sample_idx]
        frames_hwc = self._reshape_frames(frames)

        H, W = frames_hwc.shape[1], frames_hwc.shape[2]

        # 标准化所有帧
        for i in range(self.input_frames):
            frames_hwc[i] = self._normalize_frame(frames_hwc[i])

        # 创建可视化
        fig_width = W * self.input_frames
        fig_height = H + 60  # 为标签信息预留空间
        combined = np.zeros((fig_height, fig_width, 3), dtype=np.uint8)

        # 并排放置帧
        for i in range(self.input_frames):
            start_x = i * W
            end_x = (i + 1) * W
            combined[60:, start_x:end_x, :] = frames_hwc[i]

        # 转换为BGR用于OpenCV
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

        # 添加帧标签
        output_indices = self._get_output_frame_indices(self.input_frames // 2)
        for i in range(self.input_frames):
            x_pos = i * W + 10
            frame_label = f"Frame {i}"
            if i in output_indices:
                label_idx = output_indices.index(i)
                frame_label += f" (Label {label_idx})"

            cv2.putText(combined_bgr, frame_label, (x_pos, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 绘制球的位置
        if not isinstance(labels, list):
            labels = [labels]

        for i, label_idx in enumerate(output_indices):
            if i < len(labels) and isinstance(labels[i], dict) and labels[i].get('visibility', 0).item() == 1:
                x = int(labels[i].get('x').item())
                y = int(labels[i].get('y').item())

                # 调整坐标到对应帧
                x_adjusted = x + label_idx * W
                y_adjusted = y + 60  # 偏移标签区域

                color = (0, 0, 255) if label_idx == self.input_frames // 2 else (0, 255, 255)
                cv2.circle(combined_bgr, (x_adjusted, y_adjusted), 5, color, -1)
                cv2.putText(combined_bgr, f"({x},{y})", (x_adjusted + 5, y_adjusted - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 添加配置信息
        config_info = f"Sample {sample_idx} - Config: {self.input_frames} input frames -> {self.output_frames} output labels"
        cv2.putText(combined_bgr, config_info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 显示
        window_name = f"Sample {sample_idx} - {self.input_frames} Input Frames"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, combined_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def play_dataset(dataset, delay_ms: int = 30, show_frame: str = 'center'):
    """便捷函数：播放数据集"""
    viewer = ConfigurableDatasetViewer(dataset)
    viewer.play_dataset(delay_ms, show_frame)


def visualize_sample(dataset, sample_idx: int = 0):
    """便捷函数：可视化单个样本"""
    viewer = ConfigurableDatasetViewer(dataset)
    viewer.visualize_sample(sample_idx)


if __name__ == "__main__":
    # 测试不同配置
    match_dir = base_dir / 'Dataset' / 'Professional' / 'match1'

    # 配置1：3进3出
    config_3in3out = {
        "input_frames": 3,
        "output_frames": 3,
        "normalize_coords": False,
        "normalize_pixels": False,
        "video_ext": ".mp4",
        "csv_suffix": "_ball.csv"
    }

    # 3进1出
    config_3in1out = {
        "input_frames": 3,
        "output_frames": 1,
        "normalize_coords": False,
        "normalize_pixels": False,
        "video_ext": ".mp4",
        "csv_suffix": "_ball.csv"
    }

    dataset1 = BallTrackingDataset(str(match_dir), config=config_3in3out)
    dataset2 = BallTrackingDataset(str(match_dir), config=config_3in3out)

    dataset = dataset1 + dataset2  # 合并两个数据集

    print(dataset[0])
    #play_dataset(dataset, delay_ms=10)
