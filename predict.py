#!/usr/bin/env python3
"""
TrackNet Ball Tracking Prediction Script
"""

from pathlib import Path
from typing import Tuple, List, Dict, Any

import cv2
import torch
import torch.nn.functional as F

from tracknet import TrackNet


def load_model(checkpoint_path: str, device: torch.device) -> TrackNet:
    """Load trained model"""
    model = TrackNet(input_frames=3, output_frames=3)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded: {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['best_loss']:.6f}")

    return model


def calculate_resize_params(orig_h: int, orig_w: int, target_h: int, target_w: int) -> Tuple[int, int, float]:
    """Calculate equal ratio resize parameters"""
    ratio = min(target_h / orig_h, target_w / orig_w)
    new_h = int(orig_h * ratio)
    new_w = int(orig_w * ratio)
    return new_h, new_w, ratio


def preprocess_frames(frame_paths: List[str], h: int = 288, w: int = 512) -> Tuple[torch.Tensor, List[Dict]]:
    """Preprocess frames with same pipeline as training"""
    frames = []
    transform_info = []

    for i, path in enumerate(frame_paths):
        # Load image
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Cannot load: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        # Equal ratio resize
        new_h, new_w, ratio = calculate_resize_params(orig_h, orig_w, h, w)

        # Convert to tensor and resize
        tensor = torch.from_numpy(img).float().permute(2, 0, 1)
        tensor = F.interpolate(tensor.unsqueeze(0), size=(new_h, new_w),
                               mode='bilinear', align_corners=False).squeeze(0)

        # Padding
        pad_h, pad_w = h - new_h, w - new_w
        pad_top, pad_left = pad_h // 2, pad_w // 2

        if pad_h != 0 or pad_w != 0:
            tensor = F.pad(tensor, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top))

        # Normalize to [0,1]
        tensor = tensor / 255.0
        frames.append(tensor)

        # Store transform info
        transform_info.append({
            'ratio': ratio, 'pad_left': pad_left, 'pad_top': pad_top,
            'orig_h': orig_h, 'orig_w': orig_w
        })

        print(f"Frame {i + 1}: {orig_w}x{orig_h} -> {new_w}x{new_h} -> {w}x{h}")

    # Stack and reshape: [3,3,H,W] -> [1,9,H,W]
    input_tensor = torch.stack(frames).view(-1, h, w).unsqueeze(0)
    return input_tensor, transform_info


def postprocess_heatmaps(heatmaps: torch.Tensor, transform_info: List[Dict],
                         threshold: float = 0.5) -> List[Dict]:
    """Convert heatmaps to coordinates"""
    coordinates = []

    for c in range(heatmaps.shape[1]):
        heatmap = heatmaps[0, c]
        transform = transform_info[c]

        # Find peaks
        binary = (heatmap > threshold).float()

        if binary.sum() > 0:
            y_idx, x_idx = torch.where(binary > 0)
            center_x = x_idx.float().mean().item()
            center_y = y_idx.float().mean().item()

            # Reverse transform to original coordinates
            orig_x = (center_x - transform['pad_left']) / transform['ratio']
            orig_y = (center_y - transform['pad_top']) / transform['ratio']

            # Clamp to valid range
            orig_x = max(0, min(transform['orig_w'] - 1, orig_x))
            orig_y = max(0, min(transform['orig_h'] - 1, orig_y))

            confidence = heatmap.max().item()

            coordinates.append({
                'x': orig_x, 'y': orig_y, 'confidence': confidence, 'detected': True
            })

            print(f"Frame {c + 1}: Ball at ({orig_x:.1f}, {orig_y:.1f}), conf: {confidence:.3f}")
        else:
            coordinates.append({
                'x': None, 'y': None, 'confidence': 0.0, 'detected': False
            })
            print(f"Frame {c + 1}: Not detected")

    return coordinates


def predict(model: TrackNet, frame_paths: List[str], device: torch.device) -> List[Dict]:
    """Run prediction"""
    input_tensor, transform_info = preprocess_frames(frame_paths)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        heatmaps = model(input_tensor)

    return postprocess_heatmaps(heatmaps, transform_info)


def visualize(frame_paths: List[str], predictions: List[Dict], output_dir: str = ".") -> None:
    """Save visualization results"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    for i, (path, pred) in enumerate(zip(frame_paths, predictions)):
        img = cv2.imread(str(path))
        if img is None:
            continue

        if pred['detected']:
            x, y = int(pred['x']), int(pred['y'])
            conf = pred['confidence']

            cv2.circle(img, (x, y), 8, (0, 255, 0), -1)
            cv2.circle(img, (x, y), 12, (0, 255, 0), 2)
            cv2.putText(img, f"Ball ({conf:.2f})", (x + 15, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f"({x}, {y})", (x + 15, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(img, "Not Detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(img, f"Frame {i + 1}", (50, img.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        result_path = output_path / f"result_frame_{i + 1:03d}.jpg"
        cv2.imwrite(str(result_path), img)
        print(f"Saved: {result_path}")


def main():
    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Paths - UPDATE THESE
    checkpoint_path = "latest.pth"
    frame_paths = [
        "test/1.png",
        "test/2.png",
        "test/3.png"
    ]

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Frames: {frame_paths}")

    # Check files exist
    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return

    missing = [p for p in frame_paths if not Path(p).exists()]
    if missing:
        print(f"ERROR: Missing frames: {missing}")
        return

    # Run prediction
    model = load_model(checkpoint_path, device)
    predictions = predict(model, frame_paths, device)

    # Results
    detected = sum(1 for p in predictions if p['detected'])
    print(f"\nResults: {detected}/{len(predictions)} frames detected")

    for i, pred in enumerate(predictions):
        if pred['detected']:
            print(f"Frame {i + 1}: ({pred['x']:.1f}, {pred['y']:.1f}) conf={pred['confidence']:.3f}")
        else:
            print(f"Frame {i + 1}: Not detected")

    # Save visualization
    visualize(frame_paths, predictions)
    print("\nDone! Check result_frame_*.jpg files")


if __name__ == "__main__":
    main()
