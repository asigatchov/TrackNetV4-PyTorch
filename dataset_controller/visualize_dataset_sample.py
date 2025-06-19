from pathlib import Path

import cv2
import numpy as np
import torch
from data_reader import BallTrackingDataset

# 定位到项目根目录
base_dir = Path(__file__).resolve().parent.parent


def play_dataset(dataset, delay_ms: int = 30):
    """
    以“视频”形式播放 dataset 中的所有帧：
      - 每帧显示原始图像
      - 如果 visibility>0.5，则绘制红点＋坐标
      - 按 q 键可退出
    :param dataset: 一个继承 torch.utils.data.Dataset、返回 (frame, label) 的 dataset
    :param delay_ms: 每帧停留的毫秒数
    """
    window_name = "Dataset Viewer (press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for idx in range(len(dataset)):
        frame, label = dataset[idx]

        # —— Tensor -> HWC numpy uint8 ——
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
            if frame.ndim == 3 and frame.shape[0] in (1, 3):
                frame = np.transpose(frame, (1, 2, 0))

        # 自动检测动态范围并转为 uint8
        if frame.dtype in (np.float32, np.float64):
            vmin, vmax = frame.min(), frame.max()
            if vmax <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

        # RGB -> BGR（OpenCV 默认为 BGR）
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame

        # —— 叠加 label ——
        if isinstance(label, dict):
            vis = label.get('visibility', None)
            x = label.get('x', None)
            y = label.get('y', None)
            if vis is not None and vis.item() > 0.5 and x is not None and y is not None:
                xi, yi = int(x.item()), int(y.item())
                cv2.circle(frame_bgr, (xi, yi), 5, (0, 0, 255), -1)
                cv2.putText(
                    frame_bgr,
                    f"({xi},{yi})",
                    (xi + 5, yi - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA
                )

        # 显示并等待
        cv2.imshow(window_name, frame_bgr)
        if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    match_dir = base_dir / 'Dataset' / 'Professional' / 'match1'
    dataset1 = BallTrackingDataset(str(match_dir))

    print(f"Dataset loaded: {len(dataset1)} frames")
    # 播放全量 dataset，30ms／帧
    play_dataset(dataset1, delay_ms=30)
