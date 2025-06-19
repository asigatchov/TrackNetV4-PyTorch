from pathlib import Path

import cv2
import numpy as np
import torch
from data_reader import BallTrackingDataset

# Locate the project root directory
base_dir = Path(__file__).resolve().parent.parent


def play_dataset(dataset, delay_ms: int = 30):
    """
    Play all frames in the dataset as a video using OpenCV.
    Each frame displays the raw image and overlays the ball position when visibility == 1.
    Press 'q' to quit playback.

    :param dataset: torch.utils.data.Dataset returning (frame, label) tuples
    :param delay_ms: delay in milliseconds between frames
    """
    window_name = "Dataset Viewer (press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for idx in range(len(dataset)):
        frame, label = dataset[idx]

        # Convert torch.Tensor to numpy.ndarray
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
            # Convert from [C, H, W] to [H, W, C] if necessary
            if frame.ndim == 3 and frame.shape[0] in (1, 3):
                frame = np.transpose(frame, (1, 2, 0))

        # Auto-detect dynamic range and convert to uint8
        if frame.dtype in (np.float32, np.float64):
            vmin, vmax = frame.min(), frame.max()
            if vmax <= 1.0:
                # Data is in [0,1]
                frame = (frame * 255).astype(np.uint8)
            else:
                # Data is already in [0,255]
                frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Convert RGB to BGR for OpenCV display
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame

        # Overlay the ball position only when visibility == 1
        if isinstance(label, dict) and label.get('visibility', 0).item() == 1:
            x = label.get('x')
            y = label.get('y')
            if x is not None and y is not None:
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

        # Show the frame and handle the quit key
        cv2.imshow(window_name, frame_bgr)
        if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Initialize the dataset
    match_dir = base_dir / 'Dataset' / 'Professional' / 'match2'
    dataset1 = BallTrackingDataset(str(match_dir))

    print(f"Dataset loaded: {len(dataset1)} frames")
    # Play the entire dataset as a video with 30 ms per frame
    play_dataset(dataset1, delay_ms=30)
