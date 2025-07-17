"""
Frame Heatmap Dataset for PyTorch

Processes frame images and corresponding heatmaps for TrackNet training.

Dataset Structure:
dataset_reorg_train/
├── match1/
│   ├── inputs/frame1/0.jpg,1.jpg... (512×288)
│   └── heatmaps/frame1/0.jpg,1.jpg... (heatmaps)
└── match2/...

Output Format:
- inputs: (9, 288, 512) - 3 RGB images concatenated, normalized to [0,1]
- heatmaps: (3, 288, 512) - 3 grayscale heatmaps concatenated, normalized to [0,1]

Author: Generated for TrackNet training
"""

import glob
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class FrameHeatmapDataset(Dataset):
    def __init__(self, root_dir, transform=None, heatmap_transform=None):
        """
        Args:
            root_dir: Root directory of dataset
            transform: Transform for input images (default: normalize to [0,1])
            heatmap_transform: Transform for heatmaps (default: normalize to [0,1])
        """
        self.root_dir = Path(root_dir)
        self.transform = transform or transforms.ToTensor()
        self.heatmap_transform = heatmap_transform or transforms.ToTensor()
        self.data_items = self._scan_dataset()

    def _scan_dataset(self):
        """Scan dataset and build index"""
        items = []
        match_dirs = sorted(d for d in self.root_dir.iterdir()
                            if d.is_dir() and d.name.startswith('match'))

        print(f"Scanning {len(match_dirs)} match folders...")

        for match_dir in match_dirs:
            items.extend(self._process_match(match_dir))

        print(f"Found {len(items)} valid samples")
        return items

    def _process_match(self, match_dir):
        """Process single match directory"""
        inputs_dir = match_dir / 'inputs'
        heatmaps_dir = match_dir / 'heatmaps'

        if not (inputs_dir.exists() and heatmaps_dir.exists()):
            return []

        items = []
        common_frames = self._get_common_frames(inputs_dir, heatmaps_dir)

        for frame_name in sorted(common_frames):
            items.extend(self._process_frame(match_dir, frame_name))

        return items

    def _get_common_frames(self, inputs_dir, heatmaps_dir):
        """Get frame folders that exist in both inputs and heatmaps"""
        input_frames = {d.name for d in inputs_dir.iterdir() if d.is_dir()}
        heatmap_frames = {d.name for d in heatmaps_dir.iterdir() if d.is_dir()}
        return input_frames.intersection(heatmap_frames)

    def _process_frame(self, match_dir, frame_name):
        """Process single frame directory"""
        input_dir = match_dir / 'inputs' / frame_name
        heatmap_dir = match_dir / 'heatmaps' / frame_name

        input_files = self._get_sorted_images(input_dir)
        heatmap_files = self._get_sorted_images(heatmap_dir)

        if len(input_files) != len(heatmap_files) or len(input_files) < 3:
            return []

        # Generate 3-frame sequences
        return [
            {
                'inputs': input_files[i:i + 3],
                'heatmaps': heatmap_files[i:i + 3],
                'match': match_dir.name,
                'frame': frame_name,
                'idx': i
            }
            for i in range(len(input_files) - 2)
        ]

    def _get_sorted_images(self, directory):
        """Get sorted image files by numeric stem"""
        return sorted(glob.glob(str(directory / "*.jpg")),
                      key=lambda x: int(Path(x).stem))

    def _load_image(self, image_path, is_heatmap=False):
        """Load and transform image"""
        try:
            image = Image.open(image_path)
            if is_heatmap:
                image = image.convert('L')
                return self.heatmap_transform(image)
            else:
                image = image.convert('RGB')
                return self.transform(image)
        except Exception as e:
            print(f"Failed to load image: {image_path}")
            channels = 1 if is_heatmap else 3
            return torch.zeros(channels, 288, 512)

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        """
        Returns:
            inputs: (9, 288, 512) - 3 RGB images, [0,1]
            heatmaps: (3, 288, 512) - 3 grayscale heatmaps, [0,1]
        """
        item = self.data_items[idx]

        inputs = torch.cat([self._load_image(path, False) for path in item['inputs']], dim=0)
        heatmaps = torch.cat([self._load_image(path, True) for path in item['heatmaps']], dim=0)

        return inputs, heatmaps

    def get_info(self, idx):
        """Get sample information"""
        return self.data_items[idx]


if __name__ == "__main__":
    # Usage example
    root_dir = "../dataset/Test_preprocessed"

    # Create dataset
    dataset = FrameHeatmapDataset(root_dir)
    print(f"Dataset size: {len(dataset)}")

    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=2
    )

    # Test data loading
    print("\nTesting data loading:")
    for batch_idx, (inputs, heatmaps) in enumerate(dataloader):
        print(f"Batch {batch_idx}: inputs{inputs.shape}, heatmaps{heatmaps.shape}")
        print(f"  Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
        print(f"  Heatmap range: [{heatmaps.min():.3f}, {heatmaps.max():.3f}]")

        if batch_idx == 0:
            info = dataset.get_info(0)
            print(f"  Sample info: {info['match']}/{info['frame']}, start index {info['idx']}")
        break
