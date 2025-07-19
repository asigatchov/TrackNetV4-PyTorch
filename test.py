"""
TrackNet Testing Script - Evaluation Version

Usage Examples:
1. Basic testing:
   python test.py --model best_model.pth --data dataset/test

2. Custom testing:
   python test.py --model checkpoint.pth --data dataset/test --batch 8 --threshold 0.5 --device cuda

3. Detailed evaluation:
   python test.py --model best_model.pth --data dataset/test --threshold 0.3 --tolerance 4 --out test_results --report detailed

4. Quick validation:
   python test.py --model model.pth --data dataset/val --batch 16 --report summary

Parameter Functions:
- model: Path to trained model checkpoint file (required)
- data: Path to test dataset directory (required)
- batch: Number of samples per batch (default: 4)
- threshold: Detection threshold for heatmap (default: 0.5)
- tolerance: Distance tolerance in pixels (default: 4)
- device: Computing device - auto/cpu/cuda/mps (default: auto)
- out: Directory to save test results (default: test_results)
- report: Report detail level - summary/detailed (default: detailed)
- save_predictions: Save prediction coordinates to file (default: False)
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.tracknet import TrackNet
from preprocessing.tracknet_dataset import FrameHeatmapDataset


def parse_args():
    parser = argparse.ArgumentParser(description="TrackNet Testing and Evaluation")

    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to test dataset directory')
    parser.add_argument('--batch', type=int, default=4, help='Batch size for testing (default: 4)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold (default: 0.5)')
    parser.add_argument('--tolerance', type=int, default=4, help='Distance tolerance in pixels (default: 4)')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto/cpu/cuda/mps (default: auto)')
    parser.add_argument('--out', type=str, default='test_results', help='Output directory (default: test_results)')
    parser.add_argument('--report', type=str, default='detailed', choices=['summary', 'detailed'],
                        help='Report detail level (default: detailed)')
    parser.add_argument('--save_predictions', action='store_true', help='Save prediction coordinates')

    return parser.parse_args()


class TrackNetTester:
    def __init__(self, args):
        self.args = args
        self.device = self._setup_device()
        self._setup_dirs()
        self._load_model()

        self.predictions = []
        self.ground_truths = []
        self.results = {
            'tp': 0, 'tn': 0, 'fp1': 0, 'fp2': 0, 'fn': 0,
            'total_frames': 0, 'detected_frames': 0
        }

    def _setup_device(self):
        if self.args.device == 'auto':
            if torch.backends.mps.is_available():
                return torch.device('mps')
            elif torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        return torch.device(self.args.device)

    def _setup_dirs(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(self.args.out) / f"test_{timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        with open(self.save_dir / "test_config.json", 'w') as f:
            json.dump(vars(self.args), f, indent=2)

    def _load_model(self):
        self.model = TrackNet().to(self.device)
        checkpoint = torch.load(self.args.model, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"Model loaded from: {self.args.model}")

    def _setup_data(self):
        self.test_dataset = FrameHeatmapDataset(self.args.data)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.args.batch,
            shuffle=False,
            num_workers=0,
            pin_memory=self.device.type == 'cuda'
        )
        print(f"Test dataset: {len(self.test_dataset)} samples")

    def _extract_coordinates(self, heatmap):
        if heatmap.max() < self.args.threshold:
            return None
        max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        return (max_pos[1], max_pos[0])

    def _extract_ground_truth_coordinates(self, target_heatmap):
        if target_heatmap.max() < 0.1:
            return None
        max_pos = np.unravel_index(np.argmax(target_heatmap), target_heatmap.shape)
        return (max_pos[1], max_pos[0])

    def _calculate_distance(self, pred_coord, gt_coord):
        if pred_coord is None or gt_coord is None:
            return float('inf')
        return np.sqrt((pred_coord[0] - gt_coord[0]) ** 2 + (pred_coord[1] - gt_coord[1]) ** 2)

    def _classify_prediction(self, pred_coord, gt_coord):
        has_pred = pred_coord is not None
        has_gt = gt_coord is not None

        if not has_pred and not has_gt:
            return 'tn'
        elif not has_pred and has_gt:
            return 'fn'
        elif has_pred and not has_gt:
            return 'fp2'
        else:
            distance = self._calculate_distance(pred_coord, gt_coord)
            if distance <= self.args.tolerance:
                return 'tp'
            else:
                return 'fp1'

    def _evaluate_batch(self, outputs, targets):
        batch_size = outputs.shape[0]

        for b in range(batch_size):
            for f in range(3):
                pred_heatmap = outputs[b, f].cpu().numpy()
                gt_heatmap = targets[b, f].cpu().numpy()

                pred_coord = self._extract_coordinates(pred_heatmap)
                gt_coord = self._extract_ground_truth_coordinates(gt_heatmap)

                classification = self._classify_prediction(pred_coord, gt_coord)
                self.results[classification] += 1
                self.results['total_frames'] += 1

                if pred_coord is not None:
                    self.results['detected_frames'] += 1

                if self.args.save_predictions:
                    self.predictions.append({
                        'predicted': pred_coord,
                        'ground_truth': gt_coord,
                        'classification': classification,
                        'distance': self._calculate_distance(pred_coord, gt_coord) if pred_coord and gt_coord else None
                    })

    def _calculate_metrics(self):
        tp = self.results['tp']
        tn = self.results['tn']
        fp1 = self.results['fp1']
        fp2 = self.results['fp2']
        fn = self.results['fn']

        total_fp = fp1 + fp2
        total_predictions = tp + total_fp
        total_positives = tp + fn
        total_negatives = tn + total_fp
        total_samples = tp + tn + total_fp + fn

        accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
        precision = tp / total_predictions if total_predictions > 0 else 0
        recall = tp / total_positives if total_positives > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        detection_rate = self.results['detected_frames'] / self.results['total_frames']

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'detection_rate': detection_rate
        }

    def _save_results(self, metrics):
        results_summary = {
            'test_config': vars(self.args),
            'confusion_matrix': {
                'tp': self.results['tp'],
                'tn': self.results['tn'],
                'fp1': self.results['fp1'],
                'fp2': self.results['fp2'],
                'fn': self.results['fn']
            },
            'metrics': metrics,
            'statistics': {
                'total_frames': self.results['total_frames'],
                'detected_frames': self.results['detected_frames'],
                'detection_rate': metrics['detection_rate']
            },
            'timestamp': datetime.now().isoformat()
        }

        with open(self.save_dir / "test_results.json", 'w') as f:
            json.dump(results_summary, f, indent=2)

        if self.args.save_predictions:
            with open(self.save_dir / "predictions.json", 'w') as f:
                json.dump(self.predictions, f, indent=2)

    def _print_results(self, metrics):
        print("\n" + "=" * 60)
        print("TrackNet Test Results")
        print("=" * 60)

        if self.args.report == 'detailed':
            print(f"Test Dataset: {self.args.data}")
            print(f"Model: {self.args.model}")
            print(f"Device: {self.device}")
            print(f"Detection Threshold: {self.args.threshold}")
            print(f"Distance Tolerance: {self.args.tolerance} pixels")
            print("-" * 60)

            print("Confusion Matrix:")
            print(f"  True Positives (TP): {self.results['tp']}")
            print(f"  True Negatives (TN): {self.results['tn']}")
            print(f"  False Positives (FP1 - wrong position): {self.results['fp1']}")
            print(f"  False Positives (FP2 - false detection): {self.results['fp2']}")
            print(f"  False Negatives (FN): {self.results['fn']}")
            print("-" * 60)

        print("Performance Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f} ({metrics['accuracy'] * 100:.1f}%)")
        print(f"  Precision: {metrics['precision']:.3f} ({metrics['precision'] * 100:.1f}%)")
        print(f"  Recall:    {metrics['recall']:.3f} ({metrics['recall'] * 100:.1f}%)")
        print(f"  F1-Score:  {metrics['f1_score']:.3f}")
        print(f"  Detection Rate: {metrics['detection_rate']:.3f} ({metrics['detection_rate'] * 100:.1f}%)")

        print("-" * 60)
        print(f"Total Frames: {self.results['total_frames']}")
        print(f"Results saved to: {self.save_dir}")
        print("=" * 60)

    def run_test(self):
        print(f"Starting TrackNet evaluation on {self.device}")
        self._setup_data()

        start_time = time.time()

        with torch.no_grad():
            with tqdm(total=len(self.test_loader), desc="Testing") as pbar:
                for inputs, targets in self.test_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.model(inputs)
                    self._evaluate_batch(outputs, targets)

                    pbar.update(1)

        test_time = time.time() - start_time
        metrics = self._calculate_metrics()

        print(f"\nTesting completed in {test_time:.1f}s")
        print(f"Processing speed: {self.results['total_frames'] / test_time:.1f} FPS")

        self._save_results(metrics)
        self._print_results(metrics)

        return metrics


def main():
    args = parse_args()

    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return

    if not Path(args.data).exists():
        print(f"Error: Test data directory not found: {args.data}")
        return

    tester = TrackNetTester(args)
    metrics = tester.run_test()

    return metrics


if __name__ == "__main__":
    main()
