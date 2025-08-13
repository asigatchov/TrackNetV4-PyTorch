"""
TrackNet Dataset Loader (v2) with RAM Caching

Loads pre-augmented badminton dataset for training, caching all frames and heatmaps in RAM.
Designed for sequence-based input (e.g., 3 consecutive frames).
Supports grayscale or RGB input.

Expected dataset structure:
    dataset/
    ├── match1/
    │   ├── inputs/
    │   │   ├── rally1_orig/0.jpg,1.jpg...
    │   │   ├── rally1_flip/0.jpg,1.jpg...
    │   │   ├── rally1_rot_p10/0.jpg,1.jpg...
    │   │   ├── rally1_rot_m10/0.jpg,1.jpg...
    │   └── heatmaps/
    │       ├── rally1_orig/0.jpg,1.jpg...
    │       ├── rally1_flip/0.jpg,1.jpg...
    │       ├── rally1_rot_p10/0.jpg,1.jpg...
    │       └── rally1_rot_m10/0.jpg,1.jpg...
    └── match2/...

Usage:
    dataset = FrameHeatmapDataset(data_path, seq=3, grayscale=True)
"""

import os
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class FrameHeatmapDataset(Dataset):
    def __init__(self, data_path, seq=3, grayscale=False):
        """
        Initialize dataset, caching all images and heatmaps in RAM.

        Args:
            data_path (str): Path to preprocessed dataset directory.
            seq (int): Number of consecutive frames per sample (default: 3).
            grayscale (bool): Use grayscale input (True) or RGB (False).
        """
        self.data_path = Path(data_path)
        self.seq = seq
        self.grayscale = grayscale
        self.image_cache = []  # List of (image, heatmap) tuples
        self.sequence_indices = (
            []
        )  # List of (match, rally, frame_idx) for valid sequences
        self._load_dataset()

    def _load_dataset(self):
        """Load and cache all images and heatmaps in RAM."""
        print(f"Loading dataset from {self.data_path} into RAM...")
        matches = [
            d
            for d in os.listdir(self.data_path)
            if os.path.isdir(self.data_path / d) and d.startswith("match")
        ]

        for match in tqdm(matches, desc="Caching matches"):
            inputs_path = self.data_path / match / "inputs"
            heatmaps_path = self.data_path / match / "heatmaps"
            if not (inputs_path.exists() and heatmaps_path.exists()):
                continue

            rallies = [
                d for d in os.listdir(inputs_path) if os.path.isdir(inputs_path / d)
            ]
            for rally in rallies:
                input_rally_path = inputs_path / rally
                heatmap_rally_path = heatmaps_path / rally
                frames = sorted(
                    [f for f in os.listdir(input_rally_path) if f.endswith(".jpg")],
                    key=lambda x: int(x.split(".")[0]),
                )

                # Load all frames and heatmaps for this rally into RAM
                rally_images = []
                for frame in frames:
                    img_path = input_rally_path / frame
                    heatmap_path = heatmap_rally_path / frame

                    # Load image
                    img = cv2.imread(
                        str(img_path),
                        cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR,
                    )
                    if img is None:
                        continue
                    if not self.grayscale:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype(np.uint8)  # Use uint8 to save memory

                    # Load heatmap
                    heatmap = cv2.imread(str(heatmap_path), cv2.IMREAD_GRAYSCALE)
                    if heatmap is None:
                        continue
                    heatmap = heatmap.astype(np.uint8)

                    rally_images.append((img, heatmap))

                # Cache valid sequences
                for i in range(len(frames) - self.seq + 1):
                    self.sequence_indices.append((match, rally, i))
                self.image_cache.extend(rally_images)

        print(
            f"Cached {len(self.image_cache)} frames and {len(self.sequence_indices)} sequences in RAM."
        )
        if not self.sequence_indices:
            raise ValueError("No valid sequences found in dataset.")

    def __len__(self):
        """Return the number of valid sequences."""
        return len(self.sequence_indices)

    def __getitem__(self, idx):
        """
        Get a sequence of frames and corresponding heatmaps.

        Returns:
            tuple: (inputs, targets)
                - inputs: torch.Tensor of shape (seq*C, H, W) or (C, H, W) if seq=1
                - targets: torch.Tensor of shape (seq, H, W)
        """
        match, rally, start_idx = self.sequence_indices[idx]

        # Get sequence of frames and heatmaps from cache
        images = []
        heatmaps = []
        for i in range(start_idx, start_idx + self.seq):
            img, heatmap = self.image_cache[i]
            images.append(img)
            heatmaps.append(heatmap)

        # Stack images
        if self.grayscale:
            inputs = np.stack(images, axis=0)  # (seq, H, W)
        else:
            inputs = np.stack(images, axis=0)  # (seq, H, W, 3)
            inputs = inputs.transpose(0, 3, 1, 2)  # (seq, 3, H, W)
            inputs = inputs.reshape(
                -1, inputs.shape[2], inputs.shape[3]
            )  # (seq*3, H, W)

        # Stack heatmaps
        targets = np.stack(heatmaps, axis=0)  # (seq, H, W)

        # Convert to tensors and normalize
        inputs = torch.from_numpy(inputs).float() / 255.0
        targets = torch.from_numpy(targets).float() / 255.0

        return inputs, targets
