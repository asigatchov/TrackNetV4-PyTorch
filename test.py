#!/usr/bin/env python3
"""
TrackNet Test Script
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm

from dataset_controller.ball_tracking_data_reader import BallTrackingDataset
from tracknet import TrackNet, postprocess_heatmap

# Config
CONFIG = {
    "data_dir": "dataset/Test",
    "checkpoint": "best.pth",
    "batch_size": 2,
    "pixel_threshold": 4.0,
    "heatmap_threshold": 0.5,
    "input_height": 288,
    "input_width": 512,
    "heatmap_radius": 3,
}

DATASET_CONFIG = {
    "input_frames": 3,
    "output_frames": 3,
    "normalize_coords": True,
    "normalize_pixels": True,
}


def get_device():
    """Device setup"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        config = {"num_workers": 4, "pin_memory": True}
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        torch.backends.cudnn.benchmark = True
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        config = {"num_workers": 2, "pin_memory": False}
        print("Using MPS: Apple Silicon")
    else:
        device = torch.device('cpu')
        config = {"num_workers": 4, "pin_memory": False}
        print("Using CPU")

    return device, config


def create_gaussian_heatmap(x, y, visibility, height, width, radius=3):
    """Generate Gaussian heatmap"""
    heatmap = torch.zeros(height, width)

    if visibility < 0.5:
        return heatmap

    x_pixel = max(0, min(width - 1, int(x * width)))
    y_pixel = max(0, min(height - 1, int(y * height)))

    kernel_size = int(3 * radius)
    x_min = max(0, x_pixel - kernel_size)
    x_max = min(width, x_pixel + kernel_size + 1)
    y_min = max(0, y_pixel - kernel_size)
    y_max = min(height, y_pixel + kernel_size + 1)

    y_coords, x_coords = torch.meshgrid(
        torch.arange(y_min, y_max),
        torch.arange(x_min, x_max),
        indexing='ij'
    )

    dist_sq = (x_coords - x_pixel) ** 2 + (y_coords - y_pixel) ** 2
    gaussian = torch.exp(-dist_sq / (2 * radius ** 2))
    gaussian[gaussian < 0.01] = 0

    heatmap[y_min:y_max, x_min:x_max] = gaussian
    return heatmap


def test_collate_fn(batch):
    """Test collate function"""
    frames_list, heatmaps_list = [], []

    for frames, labels in batch:
        # Resize frames
        frames_resized = F.interpolate(
            frames.unsqueeze(0),
            size=(CONFIG["input_height"], CONFIG["input_width"]),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        frames_list.append(frames_resized)

        # Generate heatmaps
        heatmaps = torch.zeros(
            DATASET_CONFIG["output_frames"],
            CONFIG["input_height"],
            CONFIG["input_width"]
        )

        for i, label_dict in enumerate(labels):
            if i < DATASET_CONFIG["output_frames"] and isinstance(label_dict, dict):
                heatmap = create_gaussian_heatmap(
                    label_dict['x'].item(),
                    label_dict['y'].item(),
                    label_dict['visibility'].item(),
                    CONFIG["input_height"],
                    CONFIG["input_width"],
                    CONFIG["heatmap_radius"]
                )
                heatmaps[i] = heatmap

        heatmaps_list.append(heatmaps)

    return torch.stack(frames_list), torch.stack(heatmaps_list)


def load_test_dataset():
    """Load test dataset"""
    data_dir = Path(CONFIG["data_dir"])
    if not data_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {data_dir}")

    match_dirs = sorted([d for d in data_dir.iterdir()
                         if d.is_dir() and d.name.startswith('match')])

    if not match_dirs:
        raise ValueError(f"No match directories found in: {data_dir}")

    print(f"Loading test dataset: {data_dir}")
    combined_dataset = None

    for match_dir in match_dirs:
        try:
            dataset = BallTrackingDataset(str(match_dir), config=DATASET_CONFIG)
            if len(dataset) > 0:
                combined_dataset = dataset if combined_dataset is None else combined_dataset + dataset
                print(f"  {match_dir.name}: {len(dataset)} samples")
        except Exception as e:
            print(f"  {match_dir.name} failed: {e}")

    if combined_dataset is None:
        raise ValueError("No valid test data found")

    print(f"Total: {len(combined_dataset)} test samples")
    return combined_dataset


def load_model(device):
    """Load model"""
    checkpoint_path = Path(CONFIG["checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading model: {checkpoint_path}")

    # Create model - FIXED: Use TrackNet with correct parameters
    model = TrackNet(input_frames=3, output_frames=3)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'Unknown')
        best_loss = checkpoint.get('best_loss', 'Unknown')
        print(f"  Loaded from checkpoint (Epoch: {epoch}, Loss: {best_loss})")
    else:
        model.load_state_dict(checkpoint)
        print("  Loaded model weights")

    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    return model


def evaluate_model(model, test_loader, device, loader_config):
    """Evaluate model"""
    print(f"\nStarting evaluation")
    print(f"Batches: {len(test_loader)}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Pixel threshold: {CONFIG['pixel_threshold']} px")
    print(f"Heatmap threshold: {CONFIG['heatmap_threshold']}")

    # Metrics
    total_predictions = 0
    true_positives = 0
    false_positives_type1 = 0  # FP1: both detected but distance > threshold
    false_positives_type2 = 0  # FP2: predicted ball but no ground truth
    false_negatives = 0  # FN: no prediction but ground truth exists
    true_negatives = 0  # TN: both no ball

    all_distances = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, desc="Testing")):
            batch_size = inputs.size(0)

            inputs = inputs.to(device, non_blocking=loader_config["pin_memory"])
            targets = targets.to(device, non_blocking=loader_config["pin_memory"])

            # Forward pass
            outputs = model(inputs)  # [B, 3, H, W]

            # Get predictions
            predicted_coords = postprocess_heatmap(
                outputs.cpu(),
                threshold=CONFIG["heatmap_threshold"]
            )

            # Get ground truth
            true_coords = postprocess_heatmap(
                targets.cpu(),
                threshold=0.1
            )

            # Evaluate each sample and frame
            for b in range(batch_size):
                for f in range(DATASET_CONFIG["output_frames"]):
                    pred_coord = predicted_coords[b][f]
                    true_coord = true_coords[b][f]

                    total_predictions += 1

                    if pred_coord is not None and true_coord is not None:
                        # Both detected - check distance
                        distance = np.sqrt(
                            (pred_coord[0] - true_coord[0]) ** 2 +
                            (pred_coord[1] - true_coord[1]) ** 2
                        )
                        all_distances.append(distance)

                        if distance <= CONFIG["pixel_threshold"]:
                            true_positives += 1
                        else:
                            false_positives_type1 += 1

                    elif pred_coord is not None and true_coord is None:
                        # False positive - predicted but no ground truth
                        false_positives_type2 += 1
                        all_distances.append(float('inf'))

                    elif pred_coord is None and true_coord is not None:
                        # False negative - missed detection
                        false_negatives += 1
                        all_distances.append(float('inf'))

                    else:
                        # Both no ball - correct
                        true_negatives += 1
                        all_distances.append(0.0)

            # Show first batch details
            if batch_idx == 0:
                print(f"\nFirst batch predictions:")
                for b in range(min(2, batch_size)):
                    for f in range(DATASET_CONFIG["output_frames"]):
                        pred = predicted_coords[b][f]
                        true = true_coords[b][f]
                        if pred and true:
                            dist = np.sqrt((pred[0] - true[0]) ** 2 + (pred[1] - true[1]) ** 2)
                            print(f"  Sample{b} Frame{f}: pred{pred} vs true{true} (dist:{dist:.1f}px)")
                        else:
                            print(f"  Sample{b} Frame{f}: pred{pred} vs true{true}")

    # Calculate metrics
    accuracy = (true_positives + true_negatives) / total_predictions
    precision = true_positives / (true_positives + false_positives_type1 + false_positives_type2) if (
                                                                                                             true_positives + false_positives_type1 + false_positives_type2) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Distance stats
    finite_distances = [d for d in all_distances if np.isfinite(d)]
    avg_distance = np.mean(finite_distances) if finite_distances else float('nan')
    median_distance = np.median(finite_distances) if finite_distances else float('nan')

    return {
        'total_predictions': total_predictions,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives_type1': false_positives_type1,
        'false_positives_type2': false_positives_type2,
        'false_negatives': false_negatives,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'avg_distance': avg_distance,
        'median_distance': median_distance,
        'finite_distances': len(finite_distances),
        'infinite_distances': len(all_distances) - len(finite_distances)
    }


def main():
    """Main function"""
    print("TrackNet Test Script")
    print("=" * 50)

    try:
        # Setup
        device, loader_config = get_device()

        # Load dataset
        test_dataset = load_test_dataset()

        # Create data loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            collate_fn=test_collate_fn,
            **loader_config
        )

        # Load model
        model = load_model(device)

        # Evaluate
        results = evaluate_model(model, test_loader, device, loader_config)

        # Results
        print("\n" + "=" * 20 + " Test Results " + "=" * 20)
        print(f"Dataset: {len(test_dataset)} samples")
        print(f"Input size: {CONFIG['input_width']}×{CONFIG['input_height']}")
        print(f"MIMO: {DATASET_CONFIG['input_frames']}-in-{DATASET_CONFIG['output_frames']}-out")

        print(f"\nEvaluation:")
        print(f"  Pixel threshold: {CONFIG['pixel_threshold']} px")
        print(f"  Heatmap threshold: {CONFIG['heatmap_threshold']}")
        print(f"  Batch size: {CONFIG['batch_size']}")

        print(f"\nConfusion Matrix:")
        print(f"  Total predictions: {results['total_predictions']}")
        print(f"  True Positives (TP): {results['true_positives']}")
        print(f"  True Negatives (TN): {results['true_negatives']}")
        print(f"  False Positives 1 (FP1): {results['false_positives_type1']} (distance > threshold)")
        print(f"  False Positives 2 (FP2): {results['false_positives_type2']} (false detection)")
        print(f"  False Negatives (FN): {results['false_negatives']} (missed detection)")

        print(f"\nMetrics:")
        print(f"  Accuracy: {results['accuracy'] * 100:.2f}%")
        print(f"  Precision: {results['precision'] * 100:.2f}%")
        print(f"  Recall: {results['recall'] * 100:.2f}%")
        print(f"  F1 Score: {results['f1_score'] * 100:.2f}%")

        print(f"\nDistance Stats:")
        print(f"  Average distance: {results['avg_distance']:.3f} px")
        print(f"  Median distance: {results['median_distance']:.3f} px")
        print(f"  Valid detections: {results['finite_distances']}")
        print(f"  Failed detections: {results['infinite_distances']}")

        print("\n" + "=" * 53)
        print("Test completed!")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
