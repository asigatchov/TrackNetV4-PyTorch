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
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import blue, black, green, red
from reportlab.lib.units import inch

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
        self.results = {'tp': 0, 'tn': 0, 'fp1': 0, 'fp2': 0, 'fn': 0, 'total_frames': 0, 'detected_frames': 0}
        self.distances = []

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

    def _load_model(self):
        print("Loading model...")
        self.model = TrackNet().to(self.device)
        checkpoint = torch.load(self.args.model, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"\033[92m✓\033[0m Model loaded: {self.args.model}")

    def _setup_data(self):
        print("Loading dataset...")
        self.test_dataset = FrameHeatmapDataset(self.args.data)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.args.batch, shuffle=False,
            num_workers=0, pin_memory=self.device.type == 'cuda'
        )
        print(f"\033[92m✓\033[0m Dataset loaded: {len(self.test_dataset)} samples")

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

                distance = self._calculate_distance(pred_coord, gt_coord)
                if distance != float('inf'):
                    self.distances.append(distance)

                if self.args.save_predictions:
                    self.predictions.append({
                        'predicted': pred_coord, 'ground_truth': gt_coord,
                        'classification': classification, 'distance': distance if distance != float('inf') else None
                    })

    def _calculate_metrics(self):
        tp, tn, fp1, fp2, fn = self.results['tp'], self.results['tn'], self.results['fp1'], self.results['fp2'], \
            self.results['fn']

        total_fp = fp1 + fp2
        total_predictions = tp + total_fp
        total_positives = tp + fn
        total_samples = tp + tn + total_fp + fn

        accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
        precision = tp / total_predictions if total_predictions > 0 else 0
        recall = tp / total_positives if total_positives > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        detection_rate = self.results['detected_frames'] / self.results['total_frames']

        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score,
                'detection_rate': detection_rate}

    def _generate_visualizations(self, metrics):
        print("Generating visualizations...")

        plt.style.use('default')

        # Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        cm_data = np.array([[self.results['tp'], self.results['fp1'] + self.results['fp2']],
                            [self.results['fn'], self.results['tn']]])
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Positive', 'Predicted Negative'],
                    yticklabels=['Actual Positive', 'Actual Negative'], ax=ax)
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()

        # PDF Report
        self._generate_pdf_report(metrics)
        print(f"\033[92m✓\033[0m Report saved to {self.save_dir}")

    def _generate_pdf_report(self, metrics):
        c = canvas.Canvas(str(self.save_dir / 'test_report.pdf'), pagesize=A4)
        width, height = A4

        # Header
        c.setFont("Helvetica-Bold", 20)
        c.setFillColor(blue)
        c.drawCentredText(width / 2, height - 80, "TrackNet Test Results")

        c.setStrokeColor(blue)
        c.line(50, height - 100, width - 50, height - 100)

        # Configuration
        y = height - 140
        c.setFont("Helvetica-Bold", 12)
        c.setFillColor(black)
        c.drawString(50, y, f"Test Dataset: {self.args.data}")
        y -= 20
        c.drawString(50, y, f"Model: {self.args.model}")
        y -= 20
        c.drawString(50, y, f"Device: {self.device}")
        y -= 20
        c.drawString(50, y, f"Detection Threshold: {self.args.threshold}")
        y -= 20
        c.drawString(50, y, f"Distance Tolerance: {self.args.tolerance} pixels")

        # Separator
        y -= 30
        c.line(50, y, width - 50, y)

        # Confusion Matrix
        y -= 40
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "Confusion Matrix:")

        y -= 25
        c.setFont("Helvetica", 11)
        c.setFillColor(green)
        c.drawString(70, y, f"True Positives (TP): {self.results['tp']}")
        y -= 20
        c.setFillColor(black)
        c.drawString(70, y, f"True Negatives (TN): {self.results['tn']}")
        y -= 20
        c.setFillColor(red)
        c.drawString(70, y, f"False Positives (FP1 - wrong position): {self.results['fp1']}")
        y -= 20
        c.drawString(70, y, f"False Positives (FP2 - false detection): {self.results['fp2']}")
        y -= 20
        c.drawString(70, y, f"False Negatives (FN): {self.results['fn']}")

        # Separator
        y -= 30
        c.setFillColor(black)
        c.line(50, y, width - 50, y)

        # Performance Metrics
        y -= 40
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "Performance Metrics:")

        y -= 25
        c.setFont("Helvetica", 11)
        c.drawString(70, y, f"Accuracy:       {metrics['accuracy']:.3f} ({metrics['accuracy'] * 100:.1f}%)")
        y -= 20
        c.drawString(70, y, f"Precision:      {metrics['precision']:.3f} ({metrics['precision'] * 100:.1f}%)")
        y -= 20
        c.drawString(70, y, f"Recall:         {metrics['recall']:.3f} ({metrics['recall'] * 100:.1f}%)")
        y -= 20
        c.drawString(70, y, f"F1-Score:       {metrics['f1_score']:.3f}")
        y -= 20
        c.drawString(70, y, f"Detection Rate: {metrics['detection_rate']:.3f} ({metrics['detection_rate'] * 100:.1f}%)")

        # Separator
        y -= 30
        c.line(50, y, width - 50, y)

        # Total
        y -= 30
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"Total Frames: {self.results['total_frames']}")

        # Footer
        c.setFont("Helvetica", 8)
        c.setFillColor(black)
        c.drawCentredText(width / 2, 30, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        c.save()

    def _save_predictions(self):
        if self.args.save_predictions:
            with open(self.save_dir / "predictions.json", 'w') as f:
                json.dump(self.predictions, f, indent=2)

    def _print_results(self, metrics):
        print("\n" + "=" * 60)
        print("\033[96mTrackNet Test Results\033[0m")
        print("=" * 60)

        if self.args.report == 'detailed':
            print(f"Test Dataset: {self.args.data}")
            print(f"Model: {self.args.model}")
            print(f"Device: {self.device}")
            print(f"Detection Threshold: {self.args.threshold}")
            print(f"Distance Tolerance: {self.args.tolerance} pixels")
            print("-" * 60)

            print("Confusion Matrix:")
            print(f"  \033[92mTrue Positives (TP):\033[0m {self.results['tp']}")
            print(f"  True Negatives (TN): {self.results['tn']}")
            print(f"  \033[93mFalse Positives (FP1 - wrong position):\033[0m {self.results['fp1']}")
            print(f"  \033[93mFalse Positives (FP2 - false detection):\033[0m {self.results['fp2']}")
            print(f"  \033[91mFalse Negatives (FN):\033[0m {self.results['fn']}")
            print("-" * 60)

        print("Performance Metrics:")
        print(f"  Accuracy:       \033[92m{metrics['accuracy']:.3f}\033[0m ({metrics['accuracy'] * 100:.1f}%)")
        print(f"  Precision:      \033[94m{metrics['precision']:.3f}\033[0m ({metrics['precision'] * 100:.1f}%)")
        print(f"  Recall:         \033[94m{metrics['recall']:.3f}\033[0m ({metrics['recall'] * 100:.1f}%)")
        print(f"  F1-Score:       \033[95m{metrics['f1_score']:.3f}\033[0m")
        print(f"  Detection Rate: {metrics['detection_rate']:.3f} ({metrics['detection_rate'] * 100:.1f}%)")

        print("-" * 60)
        print(f"Total Frames: {self.results['total_frames']}")
        print(f"\033[92m✓ Results saved to: {self.save_dir}\033[0m")
        print("=" * 60)

    def run_test(self):
        print(f"Starting TrackNet evaluation on \033[96m{self.device}\033[0m")
        self._setup_data()

        start_time = time.time()

        with torch.no_grad():
            with tqdm(total=len(self.test_loader), desc="Testing Progress",
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                for inputs, targets in self.test_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.model(inputs)
                    self._evaluate_batch(outputs, targets)

                    pbar.update(1)

        test_time = time.time() - start_time
        metrics = self._calculate_metrics()

        print(f"\n\033[92m✓\033[0m Testing completed in {test_time:.1f}s")
        print(f"Processing speed: {self.results['total_frames'] / test_time:.1f} FPS")

        self._generate_visualizations(metrics)
        self._save_predictions()
        self._print_results(metrics)

        return metrics


def main():
    args = parse_args()

    if not Path(args.model).exists():
        print(f"\033[91mError: Model file not found: {args.model}\033[0m")
        return

    if not Path(args.data).exists():
        print(f"\033[91mError: Test data directory not found: {args.data}\033[0m")
        return

    tester = TrackNetTester(args)
    metrics = tester.run_test()

    return metrics


if __name__ == "__main__":
    main()
