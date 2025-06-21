#!/usr/bin/env python3
"""
TrackNetV4 Ball Tracking Prediction Script
Input: 3 consecutive frames
Output: Ball coordinates for 3 frames
"""

from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

from tracknet import TrackNetV4


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    model = TrackNetV4()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded from: {checkpoint_path}")
    print(f"✓ Training epoch: {checkpoint['epoch']}")
    print(f"✓ Best validation loss: {checkpoint['best_loss']:.6f}")

    return model


def preprocess_frames(frame_paths, input_height=288, input_width=512):
    """
    Load and preprocess 3 consecutive frames
    Args:
        frame_paths: List of 3 image file paths
        input_height: Target height for model input (288)
        input_width: Target width for model input (512)
    Returns:
        preprocessed_tensor: [1, 9, H, W] tensor ready for model
        original_sizes: List of (height, width) for each frame
    """
    frames = []
    original_sizes = []

    print("Loading and preprocessing frames...")
    for i, frame_path in enumerate(frame_paths):
        # Load image
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise ValueError(f"Cannot load image: {frame_path}")

        print(f"  Frame {i + 1}: {frame_path} -> {frame.shape}")

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_sizes.append(frame.shape[:2])  # (height, width)

        # Convert to tensor and normalize to [0, 1]
        frame_tensor = torch.from_numpy(frame).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]

        frames.append(frame_tensor)

    # Stack frames: [3, 3, H, W]
    frames_tensor = torch.stack(frames, dim=0)

    # Resize to model input size
    frames_tensor = F.interpolate(
        frames_tensor,
        size=(input_height, input_width),
        mode='bilinear',
        align_corners=False
    )

    # Reshape to [9, H, W] for model input (3 frames × 3 RGB channels)
    input_tensor = frames_tensor.view(-1, input_height, input_width)

    # Add batch dimension: [1, 9, H, W]
    input_tensor = input_tensor.unsqueeze(0)

    print(f"✓ Preprocessed tensor shape: {input_tensor.shape}")

    return input_tensor, original_sizes


def postprocess_predictions(heatmaps, original_sizes, threshold=0.5):
    """
    Convert heatmaps to ball coordinates in original image coordinates
    Args:
        heatmaps: [1, 3, H, W] model output heatmaps
        original_sizes: List of (height, width) for each frame
        threshold: Detection threshold
    Returns:
        coordinates: List of (x, y) coordinates for each frame, None if not detected
    """
    batch_size, channels, height, width = heatmaps.shape
    coordinates = []

    print(f"Postprocessing heatmaps: {heatmaps.shape}")

    for c in range(channels):
        heatmap = heatmaps[0, c]  # [H, W]

        # Find peaks above threshold
        binary_map = (heatmap > threshold).float()

        if binary_map.sum() > 0:
            # Find center of mass
            y_indices, x_indices = torch.where(binary_map > 0)
            center_x = x_indices.float().mean().item()
            center_y = y_indices.float().mean().item()

            # Convert to original image coordinates
            orig_height, orig_width = original_sizes[c % len(original_sizes)]
            # Scale from model coordinates (288x512) to original coordinates
            orig_x = center_x * orig_width / width
            orig_y = center_y * orig_height / height

            # Get confidence score (max value in heatmap)
            confidence = heatmap.max().item()

            coordinates.append({
                'x': orig_x,
                'y': orig_y,
                'confidence': confidence,
                'detected': True
            })

            print(f"  Frame {c + 1}: Ball detected at ({orig_x:.1f}, {orig_y:.1f}), confidence: {confidence:.3f}")
        else:
            coordinates.append({
                'x': None,
                'y': None,
                'confidence': 0.0,
                'detected': False
            })
            print(f"  Frame {c + 1}: Ball not detected")

    return coordinates


def predict_ball_trajectory(model, frame_paths, device):
    """
    Predict ball trajectory for 3 consecutive frames
    Args:
        model: Trained TrackNetV4 model
        frame_paths: List of 3 image file paths
        device: Torch device
    Returns:
        coordinates: List of prediction dictionaries for each frame
    """
    print("Running inference...")

    # Preprocess input frames
    input_tensor, original_sizes = preprocess_frames(frame_paths)
    input_tensor = input_tensor.to(device)

    # Model inference
    with torch.no_grad():
        heatmaps = model(input_tensor)  # [1, 3, H, W]

    print(f"✓ Model output shape: {heatmaps.shape}")

    # Postprocess predictions
    coordinates = postprocess_predictions(heatmaps, original_sizes)

    return coordinates


def visualize_results(frame_paths, predictions, save_output=True):
    """
    Visualize prediction results on frames
    Args:
        frame_paths: List of frame paths
        predictions: List of prediction dictionaries
        save_output: Whether to save output images
    """
    print("\nGenerating visualization...")

    for i, (frame_path, pred) in enumerate(zip(frame_paths, predictions)):
        try:
            # Load frame
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            # Draw prediction
            if pred['detected']:
                x, y = int(pred['x']), int(pred['y'])
                confidence = pred['confidence']

                # Draw circle and confidence
                cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)  # Green filled circle
                cv2.circle(frame, (x, y), 12, (0, 255, 0), 2)  # Green outline

                # Add text with confidence
                text = f"Ball ({confidence:.2f})"
                cv2.putText(frame, text, (x + 15, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Add coordinates
                coord_text = f"({x}, {y})"
                cv2.putText(frame, coord_text, (x + 15, y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                # Not detected
                cv2.putText(frame, "Ball Not Detected", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Add frame number
            cv2.putText(frame, f"Frame {i + 1}", (50, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if save_output:
                # Save result
                output_path = f"prediction_result_frame_{i + 1:03d}.jpg"
                cv2.imwrite(output_path, frame)
                print(f"✓ Saved: {output_path}")

        except Exception as e:
            print(f"⚠️  Failed to process frame {i + 1}: {e}")


def main():
    """Main prediction function"""
    device = 'cpu'
    # Configuration
    if torch.cuda.is_available():
        device = 'cuda'
        print("Using GPU for inference")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Using Apple Silicon GPU for inference")
    else:
        print("Using CPU for inference")

    checkpoint_path = "checkpoints/checkpoints/best.pth"  # Update this path as needed

    # Hardcoded input frames - UPDATE THESE PATHS WITH YOUR ACTUAL IMAGES
    frame_paths = [
        "test/1.png",  # Frame 1
        "test/2.png",  # Frame 2
        "test/3.png"  # Frame 3
    ]

    print("TrackNetV4 Ball Trajectory Prediction")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Input frames: {frame_paths}")
    print("=" * 50)

    try:
        # Check if checkpoint exists
        if not Path(checkpoint_path).exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            print("Please update checkpoint_path in the script")
            return

        # Check if input frames exist
        missing_frames = []
        for frame_path in frame_paths:
            if not Path(frame_path).exists():
                missing_frames.append(frame_path)

        if missing_frames:
            print(f"❌ Missing frames: {missing_frames}")
            print("Please update frame_paths in the script with actual image paths")
            return

        # Load model
        model = load_model(checkpoint_path, device)

        # Predict ball trajectory
        predictions = predict_ball_trajectory(model, frame_paths, device)

        # Display results summary
        print("\n" + "=" * 50)
        print("PREDICTION RESULTS SUMMARY")
        print("=" * 50)

        detected_count = sum(1 for pred in predictions if pred['detected'])
        print(f"Frames processed: {len(predictions)}")
        print(f"Ball detected in: {detected_count}/{len(predictions)} frames")
        print()

        for i, pred in enumerate(predictions):
            if pred['detected']:
                print(f"Frame {i + 1}: ✓ Ball at ({pred['x']:.1f}, {pred['y']:.1f}) [conf: {pred['confidence']:.3f}]")
            else:
                print(f"Frame {i + 1}: ✗ Ball not detected")

        # Generate visualization
        visualize_results(frame_paths, predictions)

        print("\n" + "=" * 50)
        print("✓ Prediction completed successfully!")
        print("Check the generated 'prediction_result_frame_*.jpg' files")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

"""
Usage Instructions:
1. Update checkpoint_path to point to your trained model
2. Update frame_paths to point to your 3 consecutive input frames
3. Run: python predict.py

The script will:
- Load the trained TrackNetV4 model
- Process 3 input frames 
- Output ball coordinates for each frame
- Generate visualization images with predictions
- Print detailed results to console

Input Requirements:
- 3 consecutive frames (any common image format)
- Trained TrackNetV4 model checkpoint (.pth file)

Output:
- Ball coordinates in original image coordinate system
- Confidence scores for each detection
- Visualization images with marked ball positions
"""
