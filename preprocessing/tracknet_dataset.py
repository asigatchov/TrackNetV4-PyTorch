import glob
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import random
import cv2
import numpy as np
import argparse

class FrameHeatmapDataset(Dataset):
    def __init__(self, root_dir, transform=None, heatmap_transform=None, seq=3, grayscale=False):
        """
        Args:
            root_dir: Root directory of dataset
            transform: Transform for input images (default: includes augmentation and normalize to [0,1])
            heatmap_transform: Transform for heatmaps (default: includes same augmentation and normalize to [0,1])
            seq: sequence length (default: 3)
            grayscale: if True, return grayscale input (C=seq), else RGB (C=seq*3)
        """
        self.root_dir = Path(root_dir)
        self.seq = seq
        self.grayscale = grayscale

        # Define augmentation pipeline for inputs (RGB images)
        self.transform = transform or transforms.Compose([
    #       transforms.RandomHorizontalFlip(p=0.5),  # Зеркалирование лево-право с вероятностью 50%
    #       transforms.RandomRotation(degrees=10),   # Поворот на ±10 градусов
            transforms.ToTensor()                   # Нормализация в [0,1]
        ])

        # Define augmentation pipeline for heatmaps (grayscale)
        self.heatmap_transform = heatmap_transform or transforms.Compose([
    #        transforms.RandomHorizontalFlip(p=0.5),  # Зеркалирование лево-право с вероятностью 50%
    #        transforms.RandomRotation(degrees=10),   # Поворот на ±10 градусов
            transforms.ToTensor()                   # Нормализация в [0,1]
        ])

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

        if len(input_files) != len(heatmap_files) or len(input_files) < self.seq:
            return []

        # Generate seq-frame sequences

        print(f"Processing frame {frame_name} with {len(input_files)} images seq={self.seq}...")
        return [
            {
                'inputs': input_files[i:i + self.seq],
                'heatmaps': heatmap_files[i:i + self.seq],
                'match': match_dir.name,
                'frame': frame_name,
                'idx': i
            }
            for i in range(len(input_files) - self.seq + 1)
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


    def _load_images_synced(self, input_paths, heatmap_paths):
        """
        Load and augment images and heatmaps synchronously.
        Returns:
            inputs: (seq*3, 288, 512) or (seq, 288, 512) tensor
            heatmaps: (seq, 288, 512) tensor
        """
        input_imgs = [Image.open(p).convert('RGB') for p in input_paths]
        heatmap_imgs = [Image.open(p).convert('L') for p in heatmap_paths]

        input_np = [np.array(img) for img in input_imgs]  # (H, W, 3)
        heatmap_np = [np.array(img) for img in heatmap_imgs]  # (H, W)

        frames = np.concatenate(input_np, axis=2)  # (H, W, seq*3)
        heatmaps = np.stack(heatmap_np, axis=2)    # (H, W, seq)
        combined = np.concatenate([frames, heatmaps], axis=2)  # (H, W, seq*3+seq)

        # --- Apply augmentations synchronously ---
        # Random horizontal flip
        if random.random() < 0.5:
            combined = np.ascontiguousarray(np.flip(combined, axis=1))

        # Random rotation in [-10, 10] degrees
        angle = random.uniform(-10, 10)
        h, w = combined.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        combined = cv2.warpAffine(combined, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        # Split back
        frames_aug = combined[:, :, :self.seq*3]  # (H, W, seq*3)
        heatmaps_aug = combined[:, :, self.seq*3:]  # (H, W, seq)

        # To torch tensors, normalize to [0,1], permute to (C, H, W)
        frames_aug = frames_aug.astype(np.float32) / 255.0
        heatmaps_aug = heatmaps_aug.astype(np.float32) / 255.0

        if self.grayscale:
            # Возвращаем (seq, H, W) - усредняем по каналам RGB для каждого кадра
            frames_aug = frames_aug.reshape((frames_aug.shape[0], frames_aug.shape[1], self.seq, 3))
            frames_aug = frames_aug.mean(axis=3)  # (H, W, seq)
            frames_aug = torch.from_numpy(frames_aug).permute(2, 0, 1)  # (seq, H, W)
        else:
            frames_aug = torch.from_numpy(frames_aug).permute(2, 0, 1)  # (seq*3, H, W)

        heatmaps_aug = torch.from_numpy(heatmaps_aug).permute(2, 0, 1)  # (seq, H, W)

        return frames_aug, heatmaps_aug

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        """
        Returns:
            inputs: (9, 288, 512) - 3 RGB images, [0,1]
            heatmaps: (3, 288, 512) - 3 grayscale heatmaps, [0,1]
        """
        item = self.data_items[idx]
        inputs, heatmaps = self._load_images_synced(item['inputs'], item['heatmaps'])
        return inputs, heatmaps

    def get_info(self, idx):
        """Get sample information"""
        return self.data_items[idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='TrackNet', choices=['TrackNet', 'VballNetV1b'])
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--seq', type=int, default=3)
    args = parser.parse_args()

    # Model input/output dims
    if args.grayscale:
        in_dim = args.seq
        out_dim = args.seq
    else:
        in_dim = args.seq * 3
        out_dim = args.seq

    print(f"Model: {args.model_name}, grayscale: {args.grayscale}, seq: {args.seq}")
    print(f"in_dim: {in_dim}, out_dim: {out_dim}")

    # Usage example
    root_dir = "./dataset/test"

    # Create dataset with augmentations
    dataset = FrameHeatmapDataset(root_dir, grayscale=args.grayscale, seq=args.seq  )
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

        if batch_idx < 5:
            info = dataset.get_info(0)
            print(f"  Sample info: {info['match']}/{info['frame']}, start index {info['idx']}")

            # --- Визуализация через cv2: единое полотно ---
            b = 0  # первый элемент в батче
            inp = inputs[b]  # (in_dim, 288, 512)
            hm = heatmaps[b]  # (out_dim, 288, 512)

            imgs = []
            overlays = []
            hmaps = []
            n_vis = min(args.seq, hm.shape[0])
            for i in range(n_vis):
                if args.grayscale:
                    # Grayscale: inp shape (seq, H, W)
                    rgb = np.stack([inp[i].cpu().numpy()]*3, axis=2)  # (288, 512, 3)
                    rgb = (rgb * 255).astype(np.uint8)
                else:
                    # RGB: inp shape (seq*3, H, W)
                    rgb = inp[i*3:(i+1)*3].permute(1, 2, 0).cpu().numpy()  # (288, 512, 3)
                    rgb = (rgb * 255).astype(np.uint8)
                imgs.append(rgb)
                # Heatmap
                hm_img = hm[i].cpu().numpy()  # (288, 512)
                hm_img_uint8 = (hm_img * 255).astype(np.uint8)
                hm_color = cv2.applyColorMap(hm_img_uint8, cv2.COLORMAP_JET)
                hmaps.append(hm_color)
                # Overlay heatmap на RGB
                overlay = cv2.addWeighted(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.6, hm_color, 0.4, 0)
                overlays.append(overlay)

            # Собираем полотно: 3 ряда по n_vis изображений
            row1 = np.hstack([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs])
            row2 = np.hstack(overlays)
            row3 = np.hstack(hmaps)
            canvas = np.vstack([row1, row2, row3])

            cv2.imshow('Batch Visualization', canvas)
            print("Press any key in the image window to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # --- конец визуализации ---
        else:
            break
