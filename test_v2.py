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

DETECTION_THRESHOLD = 0.5
GROUND_TRUTH_THRESHOLD = 0.1
DISTANCE_TOLERANCE = 4
DEFAULT_BATCH_SIZE = 4
DEFAULT_DEVICE = 'auto'
DEFAULT_OUTPUT_DIR = 'test_results'
DEFAULT_REPORT_LEVEL = 'detailed'
FIGURE_DPI = 150
REPORT_BACKGROUND_COLOR = '#f8f9fa'

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
from preprocessing.tracknet_dataset import FrameHeatmapDataset

# Choose the version of TrackNet model you want to use
from model.tracknet_v4 import TrackNet
from model.vballnet_v1 import VballNetV1 
from model.vballnet_v1c import VballNetV1c 
from model.vballnet_v1d import VballNetV1d 
from model.vballnetfast_v1 import VballNetFastV1  # Import the fast version


def parse_args():
    parser = argparse.ArgumentParser(description="TrackNet Testing and Evaluation")
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to test dataset directory')
    parser.add_argument('--batch', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Batch size for testing (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--threshold', type=float, default=DETECTION_THRESHOLD,
                        help=f'Detection threshold (default: {DETECTION_THRESHOLD})')
    parser.add_argument('--tolerance', type=int, default=DISTANCE_TOLERANCE,
                        help=f'Distance tolerance in pixels (default: {DISTANCE_TOLERANCE})')
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE,
                        help=f'Device: auto/cpu/cuda/mps (default: {DEFAULT_DEVICE})')
    parser.add_argument('--out', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--report', type=str, default=DEFAULT_REPORT_LEVEL, choices=['summary', 'detailed'],
                        help=f'Report detail level (default: {DEFAULT_REPORT_LEVEL})')
    parser.add_argument('--save_predictions', action='store_true', help='Save prediction coordinates')
    return parser.parse_args()


def parse_model_params_from_name(model_path):
    """
    Parse model name for seq and grayscale mode.
    Example: VballNetV1b_seq9_grayscale_best -> seq=9, grayscale=True
    """
    import os
    basename = os.path.basename(model_path)
    seq = 3
    grayscale = False
    if "seq" in basename:
        import re
        m = re.search(r"seq(\d+)", basename)
        if m:
            seq = int(m.group(1))
    if "grayscale" in basename.lower():
        grayscale = True
    return seq, grayscale


class TrackNetTester:
    def __init__(self, args):
        self.args = args
        self.device = self._setup_device()
        self._setup_dirs()
        self._load_model()

        self.frame_predictions = {}
        self.results = {'tp': 0, 'tn': 0, 'fp1': 0, 'fp2': 0, 'fn': 0, 'total_frames': 0, 'detected_frames': 0}
        self.predictions = []

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
        print("Loading model...")
        seq, grayscale = parse_model_params_from_name(self.args.model)
        self.seq = seq
        self.grayscale = grayscale
        if 'VballNetV1c' in self.args.model:
            # VballNetV1c with context (GRU)
            in_dim = seq if grayscale else seq * 3
            out_dim = seq
            self.model = VballNetV1c(
                height=288,
                width=512,
                in_dim=in_dim,
                out_dim=out_dim,
                fusion_layer_type="TypeA"
            ).to(self.device)
            self.model._model_type = "VballNetV1c"
        elif 'VballNetV1' in self.args.model:
            in_dim = seq if grayscale else seq * 3
            out_dim = seq
            self.model = VballNetV1(
                height=288,
                width=512,
                in_dim=in_dim,
                out_dim=out_dim,
                fusion_layer_type="TypeA"
            ).to(self.device)
        elif 'VballNetV1d' in self.args.model:
            in_dim = seq if grayscale else seq * 3
            out_dim = seq
            self.model = VballNetV1d(
                height=288,
                width=512,
                in_dim=in_dim,
                out_dim=out_dim,
                fusion_layer_type="TypeA"
            ).to(self.device)
            self.model._model_type = "VballNetV1d"

        elif 'VballNetFastV1' in self.args.model:
            in_dim = seq if grayscale else seq * 3
            out_dim = seq
            self.model = VballNetFastV1(
                input_height=288,
                input_width=512,
                in_dim=in_dim,
                out_dim=out_dim,
                channels=[8, 16, 32],
                bottleneck_channels=64,
                dropout_p=0.2
            ).to(self.device)
            self.model._model_type = "VballNetFastV1"

        else:
            self.model = TrackNet().to(self.device)
        checkpoint = torch.load(self.args.model, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"Model loaded: {self.args.model} (seq={seq}, grayscale={grayscale})")

    def _setup_data(self):
        print("Loading dataset...")
        seq, grayscale = parse_model_params_from_name(self.args.model)
        
        self.test_dataset = FrameHeatmapDataset(self.args.data, grayscale=grayscale, seq=seq)
        
        
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.args.batch, shuffle=False,
            num_workers=0, pin_memory=self.device.type == 'cuda'
        )


        print(f"Dataset loaded: {len(self.test_dataset)} samples")

    def _extract_coordinates(self, heatmap):
        if heatmap.max() < self.args.threshold:
            return None
        max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        return (max_pos[1], max_pos[0])

    def _extract_ground_truth_coordinates(self, target_heatmap):
        if target_heatmap.max() < GROUND_TRUTH_THRESHOLD:
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

    def _collect_predictions(self, outputs, targets, batch_start_idx):
        batch_size = outputs.shape[0]

        for b in range(batch_size):
            item_info = self.test_dataset.get_info(batch_start_idx + b)
            base_frame_idx = item_info['idx']
            match_name = item_info['match']
            frame_name = item_info['frame']

            for f in range(3):
                pred_heatmap = outputs[b, f].cpu().numpy()
                gt_heatmap = targets[b, f].cpu().numpy()

                pred_coord = self._extract_coordinates(pred_heatmap)
                gt_coord = self._extract_ground_truth_coordinates(gt_heatmap)

                frame_idx = base_frame_idx + f
                frame_key = f"{match_name}_{frame_name}_{frame_idx}"

                if frame_key not in self.frame_predictions:
                    self.frame_predictions[frame_key] = []

                self.frame_predictions[frame_key].append({
                    'pred': pred_coord,
                    'gt': gt_coord,
                    'is_center': f == 1,
                    'distance': self._calculate_distance(pred_coord, gt_coord)
                })

    def _process_center_frame_predictions(self):
        print("Processing center frame predictions...")

        for frame_key, predictions in self.frame_predictions.items():
            center_pred = None
            for pred in predictions:
                if pred['is_center']:
                    center_pred = pred
                    break

            if center_pred:
                classification = self._classify_prediction(center_pred['pred'], center_pred['gt'])
                self.results[classification] += 1
                self.results['total_frames'] += 1

                if center_pred['pred'] is not None:
                    self.results['detected_frames'] += 1

                if self.args.save_predictions:
                    distance = center_pred['distance'] if center_pred['distance'] != float('inf') else None
                    self.predictions.append({
                        'frame_key': frame_key,
                        'predicted': center_pred['pred'],
                        'ground_truth': center_pred['gt'],
                        'classification': classification,
                        'distance': distance
                    })

        print(f"Processed {self.results['total_frames']} center frame predictions")

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

        fig, ax = plt.subplots(figsize=(8, 6))
        cm_data = np.array([[self.results['tp'], self.results['fp1'] + self.results['fp2']],
                            [self.results['fn'], self.results['tn']]])
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Positive', 'Predicted Negative'],
                    yticklabels=['Actual Positive', 'Actual Negative'], ax=ax)
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'confusion_matrix.png', dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()

        fig = plt.figure(figsize=(14, 10))
        fig.patch.set_facecolor(REPORT_BACKGROUND_COLOR)

        fig.suptitle('TrackNet Test Results - Center Frame Evaluation', fontsize=24, fontweight='bold', y=0.95,
                     color='#2c3e50')

        ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=1)
        ax1.axis('off')
        config_text = f"""Dataset: {self.args.data}
Model: {self.args.model}
Device: {self.device}
Threshold: {self.args.threshold} | Tolerance: {self.args.tolerance}px
Evaluation: Center Frame Only"""
        ax1.text(0.05, 0.5, config_text, fontsize=12, verticalalignment='center',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='#e8f4fd', alpha=0.8))

        ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2, rowspan=1)
        ax2.axis('off')
        cm_text = f"""True Positives: {self.results['tp']:,}
True Negatives: {self.results['tn']:,}
False Positives (Wrong): {self.results['fp1']:,}
False Positives (False): {self.results['fp2']:,}
False Negatives: {self.results['fn']:,}"""
        ax2.text(0.05, 0.5, cm_text, fontsize=12, verticalalignment='center',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='#fff2e8', alpha=0.8))

        ax3 = plt.subplot2grid((3, 4), (1, 0), colspan=4, rowspan=1)
        ax3.axis('off')

        metrics_data = [
            ('Accuracy', metrics['accuracy'], '#27ae60'),
            ('Precision', metrics['precision'], '#3498db'),
            ('Recall', metrics['recall'], '#e74c3c'),
            ('F1-Score', metrics['f1_score'], '#9b59b6'),
            ('Detection Rate', metrics['detection_rate'], '#f39c12')
        ]

        y_pos = 0.8
        for name, value, color in metrics_data:
            ax3.barh(y_pos, 1, height=0.08, color='#ecf0f1', alpha=0.5)
            ax3.barh(y_pos, value, height=0.08, color=color, alpha=0.8)
            ax3.text(0.02, y_pos, name, fontsize=11, fontweight='bold', va='center')
            ax3.text(0.98, y_pos, f'{value:.3f} ({value * 100:.1f}%)', fontsize=11,
                     fontweight='bold', va='center', ha='right')
            y_pos -= 0.15

        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)

        ax4 = plt.subplot2grid((3, 4), (2, 0), colspan=4, rowspan=1)
        ax4.axis('off')

        total_frames = self.results['total_frames']
        summary_text = f"""Total Center Frames Processed: {total_frames:,}
Test Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

        ax4.text(0.5, 0.5, summary_text, fontsize=14, fontweight='bold',
                 ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.8", facecolor='#d5f4e6', alpha=0.8))

        plt.tight_layout()
        plt.savefig(self.save_dir / 'test_report.png', dpi=FIGURE_DPI, bbox_inches='tight',
                    facecolor=REPORT_BACKGROUND_COLOR, edgecolor='none')
        plt.close()

        print(f"Visualizations saved to {self.save_dir}")

    def _save_results(self, metrics):
        if self.args.save_predictions:
            with open(self.save_dir / "predictions.json", 'w') as f:
                json.dump(self.predictions, f, indent=2)

    def _print_results(self, metrics):
        print("\n" + "=" * 60)
        print("TrackNet Test Results - Center Frame Evaluation")
        print("=" * 60)

        if self.args.report == 'detailed':
            print(f"Test Dataset: {self.args.data}")
            print(f"Model: {self.args.model}")
            print(f"Device: {self.device}")
            print(f"Detection Threshold: {self.args.threshold}")
            print(f"Distance Tolerance: {self.args.tolerance} pixels")
            print(f"Evaluation Method: Center Frame Only")
            print("-" * 60)

            print("Confusion Matrix:")
            print(f"  True Positives (TP): {self.results['tp']}")
            print(f"  True Negatives (TN): {self.results['tn']}")
            print(f"  False Positives (FP1 - wrong position): {self.results['fp1']}")
            print(f"  False Positives (FP2 - false detection): {self.results['fp2']}")
            print(f"  False Negatives (FN): {self.results['fn']}")
            print("-" * 60)

        print("Performance Metrics:")
        print(f"  Accuracy:       {metrics['accuracy']:.3f} ({metrics['accuracy'] * 100:.1f}%)")
        print(f"  Precision:      {metrics['precision']:.3f} ({metrics['precision'] * 100:.1f}%)")
        print(f"  Recall:         {metrics['recall']:.3f} ({metrics['recall'] * 100:.1f}%)")
        print(f"  F1-Score:       {metrics['f1_score']:.3f}")
        print(f"  Detection Rate: {metrics['detection_rate']:.3f} ({metrics['detection_rate'] * 100:.1f}%)")

        print("-" * 60)
        print(f"Total Center Frames: {self.results['total_frames']}")
        print(f"Results saved to: {self.save_dir}")
        print("=" * 60)

    def run_test(self):
        print(f"Starting TrackNet evaluation on {self.device}")
        print("Using center frame evaluation strategy")
        self._setup_data()

        start_time = time.time()
        batch_start_idx = 0

        use_gru = hasattr(self.model, '_model_type') and self.model._model_type == "VballNetV1c"
        h0 = None

        with torch.no_grad():
            with tqdm(total=len(self.test_loader), desc="Testing Progress",
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                for inputs, targets in self.test_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    if use_gru:
                        try:
                            outputs, hn = self.model(inputs, h0=h0)
                            h0 = hn.detach()
                        except Exception as e:
                            print('Error during GRU forward pass:', e)
                            outputs, hn = self.model(inputs, h0=None)
                            h0 = hn.detach()
                    else:
                        outputs = self.model(inputs)

                    self._collect_predictions(outputs, targets, batch_start_idx)

                    batch_start_idx += inputs.shape[0]
                    pbar.update(1)

        self._process_center_frame_predictions()

        test_time = time.time() - start_time
        metrics = self._calculate_metrics()

        print(f"\nTesting completed in {test_time:.1f}s")
        print(f"Processing speed: {self.results['total_frames'] / test_time:.1f} FPS")

        self._generate_visualizations(metrics)
        self._save_results(metrics)
        self._print_results(metrics)

        return metrics


def main():
    print("\n" + "=" * 50)
    print("TrackNet Testing Configuration")
    print("=" * 50)
    print(f"DETECTION_THRESHOLD:   {DETECTION_THRESHOLD}")
    print(f"GROUND_TRUTH_THRESHOLD: {GROUND_TRUTH_THRESHOLD}")
    print(f"DISTANCE_TOLERANCE:    {DISTANCE_TOLERANCE}")
    print(f"DEFAULT_BATCH_SIZE:    {DEFAULT_BATCH_SIZE}")
    print(f"EVALUATION_METHOD:     Center Frame Only")
    print("=" * 50 + "\n")

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
