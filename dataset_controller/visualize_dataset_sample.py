from pathlib import Path

import cv2
import numpy as np
import torch
from ball_tracking_data_reader import BallTrackingDataset

# Locate the project root directory
base_dir = Path(__file__).resolve().parent.parent


def play_dataset(dataset, delay_ms: int = 30, show_frame: str = 'center'):
    """
    Play all frames in the dataset as a video using OpenCV.
    Each frame displays the selected frame from 3 consecutive frames and overlays ball position.
    Press 'q' to quit, 'p' to pause/resume, '1'/'2'/'3' to switch frame view.

    :param dataset: torch.utils.data.Dataset returning (frames, label) tuples
    :param delay_ms: delay in milliseconds between frames
    :param show_frame: which frame to show ('prev', 'center', 'next', 'all')
    """
    window_name = "Dataset Viewer (q:quit, p:pause, 1/2/3:frame select, a:all frames)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    paused = False
    current_show_frame = show_frame

    idx = 0
    while idx < len(dataset):
        if not paused:
            frames, label = dataset[idx]

            # Convert torch.Tensor to numpy.ndarray
            if isinstance(frames, torch.Tensor):
                frames = frames.detach().cpu().numpy()

            # Reshape from (9, H, W) to (3, 3, H, W) then to (3, H, W, 3)
            num_frames = 3
            channels_per_frame = 3
            H, W = frames.shape[1], frames.shape[2]

            # Reshape to (3, 3, H, W)
            frames_reshaped = frames.reshape(num_frames, channels_per_frame, H, W)

            # Convert to (3, H, W, 3) for visualization
            frames_hwc = np.transpose(frames_reshaped, (0, 2, 3, 1))

            # Select which frame(s) to display
            if current_show_frame == 'prev':
                display_frame = frames_hwc[0]  # Previous frame
                title_suffix = " - Previous Frame"
            elif current_show_frame == 'center':
                display_frame = frames_hwc[1]  # Center frame (labeled)
                title_suffix = " - Center Frame (Labeled)"
            elif current_show_frame == 'next':
                display_frame = frames_hwc[2]  # Next frame
                title_suffix = " - Next Frame"
            elif current_show_frame == 'all':
                # Concatenate all 3 frames horizontally
                display_frame = np.hstack([frames_hwc[0], frames_hwc[1], frames_hwc[2]])
                title_suffix = " - All 3 Frames"
            else:
                display_frame = frames_hwc[1]  # Default to center
                title_suffix = " - Center Frame"

            # Auto-detect dynamic range and convert to uint8
            if display_frame.dtype in (np.float32, np.float64):
                vmin, vmax = display_frame.min(), display_frame.max()
                if vmax <= 1.0:
                    # Data is in [0,1]
                    display_frame = (display_frame * 255).astype(np.uint8)
                else:
                    # Data is already in [0,255]
                    display_frame = np.clip(display_frame, 0, 255).astype(np.uint8)

            # Convert RGB to BGR for OpenCV display
            if display_frame.ndim == 3 and display_frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = display_frame.copy()

            # Overlay the ball position only when visibility == 1
            # Note: coordinates are for the center frame
            if isinstance(label, dict) and label.get('visibility', 0).item() == 1:
                x = label.get('x')
                y = label.get('y')
                if x is not None and y is not None:
                    xi, yi = int(x.item()), int(y.item())

                    # Adjust coordinates if showing all frames
                    if current_show_frame == 'all':
                        # Ball position is on the center frame (middle third)
                        xi_adjusted = xi + W  # Offset by one frame width
                        yi_adjusted = yi
                    else:
                        xi_adjusted, yi_adjusted = xi, yi

                    # Only draw ball on center frame or adjusted position for 'all' view
                    if current_show_frame != 'all' or current_show_frame == 'all':
                        cv2.circle(frame_bgr, (xi_adjusted, yi_adjusted), 5, (0, 0, 255), -1)
                        cv2.putText(
                            frame_bgr,
                            f"({xi},{yi})",
                            (xi_adjusted + 5, yi_adjusted - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA
                        )

            # Add frame info
            info_text = f"Frame {idx + 1}/{len(dataset)}{title_suffix}"
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

            # Show the frame
            cv2.imshow(window_name, frame_bgr)

        # Handle key presses
        key = cv2.waitKey(delay_ms if not paused else 0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('1'):
            current_show_frame = 'prev'
            print("Showing previous frame")
        elif key == ord('2'):
            current_show_frame = 'center'
            print("Showing center frame (labeled)")
        elif key == ord('3'):
            current_show_frame = 'next'
            print("Showing next frame")
        elif key == ord('a'):
            current_show_frame = 'all'
            print("Showing all 3 frames")
        elif key == 32:  # Spacebar
            paused = not paused
            print("Paused" if paused else "Resumed")

        if not paused:
            idx += 1

    cv2.destroyAllWindows()


def visualize_sample(dataset, sample_idx: int = 0):
    """
    Visualize a single sample showing all 3 frames side by side

    :param dataset: BallTrackingDataset
    :param sample_idx: index of sample to visualize
    """
    if sample_idx >= len(dataset):
        print(f"Sample index {sample_idx} out of range. Dataset has {len(dataset)} samples.")
        return

    frames, label = dataset[sample_idx]

    # Convert torch.Tensor to numpy.ndarray
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()

    # Reshape from (9, H, W) to (3, 3, H, W) then to (3, H, W, 3)
    num_frames = 3
    channels_per_frame = 3
    H, W = frames.shape[1], frames.shape[2]

    frames_reshaped = frames.reshape(num_frames, channels_per_frame, H, W)
    frames_hwc = np.transpose(frames_reshaped, (0, 2, 3, 1))

    # Normalize to [0, 255] if needed
    for i in range(3):
        if frames_hwc[i].dtype in (np.float32, np.float64):
            if frames_hwc[i].max() <= 1.0:
                frames_hwc[i] = (frames_hwc[i] * 255).astype(np.uint8)
            else:
                frames_hwc[i] = np.clip(frames_hwc[i], 0, 255).astype(np.uint8)

    # Create visualization
    fig_width = W * 3
    fig_height = H
    combined = np.zeros((fig_height, fig_width, 3), dtype=np.uint8)

    # Place frames side by side
    for i in range(3):
        start_x = i * W
        end_x = (i + 1) * W
        combined[:, start_x:end_x, :] = frames_hwc[i]

    # Convert to BGR for OpenCV
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

    # Add labels
    frame_labels = ['Previous', 'Center (Labeled)', 'Next']
    for i, frame_label in enumerate(frame_labels):
        x_pos = i * W + 10
        cv2.putText(combined_bgr, frame_label, (x_pos, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw ball position on center frame if visible
    if isinstance(label, dict) and label.get('visibility', 0).item() == 1:
        x = int(label.get('x').item())
        y = int(label.get('y').item())
        # Adjust for center frame position
        x_center = x + W
        cv2.circle(combined_bgr, (x_center, y), 5, (0, 0, 255), -1)
        cv2.putText(combined_bgr, f"Ball({x},{y})", (x_center + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Display
    window_name = f"Sample {sample_idx} - 3 Consecutive Frames"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, combined_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Initialize the dataset
    match_dir = base_dir / 'Dataset' / 'Professional' / 'match1'
    dataset1 = BallTrackingDataset(str(match_dir))

    match_dir = base_dir / 'Dataset' / 'Professional' / 'match2'
    dataset2 = BallTrackingDataset(str(match_dir))

    dataset = dataset1 + dataset2

    print(f"Dataset loaded: {len(dataset)} frames")
    print("Controls:")
    print("- 'q': quit")
    print("- 'p' or spacebar: pause/resume")
    print("- '1': show previous frame")
    print("- '2': show center frame (labeled)")
    print("- '3': show next frame")
    print("- 'a': show all 3 frames side by side")

    # Option 1: Play the dataset as video
    play_dataset(dataset, delay_ms=1)
