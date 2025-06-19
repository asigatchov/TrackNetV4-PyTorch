import logging
from pathlib import Path
from typing import Tuple, Dict, List

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

base_dir = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class BallTrackingDataset(Dataset):
    """
    Ball tracking dataset with 3-frame input (9 channels total)

    Structure:
    match_folder/
    ├── csv/
    │   ├── filename_ball.csv  # Frame, Visibility, X, Y
    │   └── ...
    └── video/
        ├── filename.mp4
        └── ...
    """

    def __init__(self,
                 match_folder: str = None,
                 video_ext: str = '.mp4',
                 csv_suffix: str = '_ball.csv',
                 normalize_coords: bool = False,
                 normalize_pixels: bool = False,
                 num_frames: int = 3,
                 _internal_data: Dict = None):
        """
        Args:
            match_folder: Path containing csv and video subfolders
            video_ext: Video file extension
            csv_suffix: CSV file suffix
            normalize_coords: Whether to normalize coordinates to [0,1]
            normalize_pixels: Whether to normalize pixel values to [0,1]
            num_frames: Number of consecutive frames (default: 3)
            _internal_data: Internal use for dataset merging
        """
        self.video_ext = video_ext
        self.csv_suffix = csv_suffix
        self.normalize_coords = normalize_coords
        self.normalize_pixels = normalize_pixels
        self.num_frames = num_frames

        self._video_info_cache = {}
        self._label_cache = {}

        if _internal_data is not None:
            self._init_from_internal_data(_internal_data)
        else:
            if match_folder is None:
                raise ValueError("match_folder is required when not using _internal_data")
            self._init_from_folder(match_folder)

    def _init_from_folder(self, match_folder: str):
        """Initialize dataset from folder"""
        self.match_path = Path(match_folder)
        self.video_folder = self.match_path / 'video'
        self.csv_folder = self.match_path / 'csv'

        self._validate_structure()
        self.video_pairs = self._discover_pairs()
        self.frame_index = self._build_frame_index()

        print(f"Dataset loaded: {len(self.video_pairs)} videos, {len(self.frame_index)} labeled frames")

    def _init_from_internal_data(self, data: Dict):
        """Initialize from internal data (for merging)"""
        self.video_pairs = data['video_pairs']
        self.frame_index = data['frame_index']
        self._video_info_cache = data['video_info_cache']
        self._label_cache = data['label_cache']

        print(f"Dataset merged: {len(self.video_pairs)} videos, {len(self.frame_index)} labeled frames")

    def _validate_structure(self) -> None:
        """Validate folder structure"""
        for folder in [self.match_path, self.video_folder, self.csv_folder]:
            if not folder.exists():
                raise FileNotFoundError(f"Required folder not found: {folder}")

    def _discover_pairs(self) -> List[Dict[str, Path]]:
        """Discover and validate video-CSV file pairs"""
        pairs = []
        video_files = list(self.video_folder.glob(f"*{self.video_ext}"))

        if not video_files:
            raise ValueError(f"No video files found in {self.video_folder}")

        for video_path in video_files:
            csv_path = self.csv_folder / f"{video_path.stem}{self.csv_suffix}"

            if not csv_path.exists():
                logger.warning(f"Missing CSV for video: {video_path.name}")
                continue

            try:
                df = pd.read_csv(csv_path)
                required_cols = ['Frame', 'Visibility', 'X', 'Y']
                if not all(col in df.columns for col in required_cols):
                    logger.warning(f"Invalid CSV format: {csv_path.name}")
                    continue
            except Exception as e:
                logger.warning(f"Cannot read CSV {csv_path.name}: {e}")
                continue

            pairs.append({
                'video_path': video_path,
                'csv_path': csv_path,
                'stem': video_path.stem
            })

        if not pairs:
            raise ValueError("No valid video-CSV pairs found")

        return sorted(pairs, key=lambda x: x['stem'])

    def _get_video_info(self, video_path: Path) -> Dict[str, int]:
        """Get video info with caching"""
        if video_path not in self._video_info_cache:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            info = {
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            }
            cap.release()

            if info['frame_count'] <= 0:
                raise ValueError(f"Invalid frame count for video: {video_path}")

            self._video_info_cache[video_path] = info

        return self._video_info_cache[video_path]

    def _load_labels(self, csv_path: Path) -> pd.DataFrame:
        """Load and cache label data"""
        if csv_path not in self._label_cache:
            df = pd.read_csv(csv_path)
            df['Frame'] = df['Frame'].astype(int)
            self._label_cache[csv_path] = df

        return self._label_cache[csv_path]

    def _build_frame_index(self) -> List[Tuple[int, int]]:
        """Build frame index ensuring sufficient frames for multi-frame input"""
        index = []
        half_frames = self.num_frames // 2

        for video_idx, pair in enumerate(self.video_pairs):
            try:
                video_info = self._get_video_info(pair['video_path'])
                df = self._load_labels(pair['csv_path'])

                labeled_frames = set(df['Frame'].astype(int).tolist())

                for frame_idx in labeled_frames:
                    # Check if we have enough frames before and after
                    start_frame = frame_idx - half_frames
                    end_frame = frame_idx + half_frames

                    if start_frame < 0 or end_frame >= video_info['frame_count']:
                        continue

                    # Validate label data
                    frame_data = df[df['Frame'] == frame_idx]
                    if len(frame_data) != 1:
                        continue

                    row = frame_data.iloc[0]
                    if pd.isna(row['Visibility']) or pd.isna(row['X']) or pd.isna(row['Y']):
                        continue

                    index.append((video_idx, frame_idx))

            except Exception as e:
                logger.error(f"Error processing video {pair['video_path']}: {e}")
                raise

        if not index:
            raise ValueError("No valid labeled frames found in dataset")

        return sorted(index)

    def _read_frame(self, video_path: Path, frame_idx: int) -> np.ndarray:
        """Read specific frame"""
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Cannot read frame {frame_idx} from {video_path}")

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _read_consecutive_frames(self, video_path: Path, center_frame: int) -> np.ndarray:
        """Read consecutive frames centered on center_frame"""
        half_frames = self.num_frames // 2
        start_frame = center_frame - half_frames
        end_frame = center_frame + half_frames

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frames = []
        for frame_idx in range(start_frame, end_frame + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                cap.release()
                raise ValueError(f"Cannot read frame {frame_idx} from {video_path}")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()
        return np.stack(frames, axis=0)  # Shape: (num_frames, H, W, C)

    def _get_frame_label(self, csv_path: Path, frame_idx: int, video_info: Dict[str, int]) -> Dict[str, float]:
        """Get label for specific frame"""
        df = self._load_labels(csv_path)
        frame_data = df[df['Frame'] == frame_idx]

        if len(frame_data) == 0:
            raise ValueError(f"No label found for frame {frame_idx} in {csv_path.name}")

        if len(frame_data) > 1:
            raise ValueError(f"Multiple labels found for frame {frame_idx} in {csv_path.name}")

        row = frame_data.iloc[0]

        if pd.isna(row['Visibility']) or pd.isna(row['X']) or pd.isna(row['Y']):
            raise ValueError(f"Invalid/missing data for frame {frame_idx} in {csv_path.name}")

        visibility = float(row['Visibility'])
        x, y = float(row['X']), float(row['Y'])

        if visibility not in [0.0, 1.0]:
            raise ValueError(f"Invalid visibility value {visibility} for frame {frame_idx} in {csv_path.name}")

        if visibility > 0:
            if x < 0 or y < 0:
                raise ValueError(
                    f"Invalid coordinates ({x}, {y}) for visible ball at frame {frame_idx} in {csv_path.name}")

            if self.normalize_coords:
                x /= video_info['width']
                y /= video_info['height']

        return {'visibility': visibility, 'x': x, 'y': y}

    def __len__(self) -> int:
        return len(self.frame_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            frames: (C*num_frames, H, W) tensor - 3 consecutive frames = 9 channels
            labels: {'visibility': scalar, 'x': scalar, 'y': scalar} for center frame
        """
        if idx >= len(self.frame_index):
            raise IndexError(f"Index {idx} out of range")

        video_idx, frame_idx = self.frame_index[idx]
        pair = self.video_pairs[video_idx]

        # Read consecutive frames
        frames = self._read_consecutive_frames(pair['video_path'], frame_idx)  # (num_frames, H, W, C)
        video_info = self._get_video_info(pair['video_path'])

        # Get label for center frame only
        label = self._get_frame_label(pair['csv_path'], frame_idx, video_info)

        # Convert to tensor and reshape
        frames_tensor = torch.from_numpy(frames).float()  # (num_frames, H, W, C)

        if self.normalize_pixels:
            frames_tensor = frames_tensor / 255.0

        # Reshape to (C*num_frames, H, W)
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # (num_frames, C, H, W)
        frames_tensor = frames_tensor.reshape(-1, frames_tensor.shape[2],
                                              frames_tensor.shape[3])  # (C*num_frames, H, W)

        label_tensors = {
            'visibility': torch.tensor(label['visibility'], dtype=torch.float32),
            'x': torch.tensor(label['x'], dtype=torch.float32),
            'y': torch.tensor(label['y'], dtype=torch.float32)
        }

        return frames_tensor, label_tensors

    def get_video_info(self, video_idx: int) -> Dict:
        """Get info for specific video"""
        if video_idx >= len(self.video_pairs):
            raise IndexError(f"Video index {video_idx} out of range")

        pair = self.video_pairs[video_idx]
        info = self._get_video_info(pair['video_path'])
        info['stem'] = pair['stem']
        return info

    def _validate_compatibility(self, other: 'BallTrackingDataset') -> None:
        """Validate compatibility for merging"""
        if not isinstance(other, BallTrackingDataset):
            raise TypeError("Can only merge with another BallTrackingDataset")

        attrs_to_check = ['video_ext', 'csv_suffix', 'normalize_coords', 'normalize_pixels', 'num_frames']
        for attr in attrs_to_check:
            if getattr(self, attr) != getattr(other, attr):
                raise ValueError(f"{attr} settings don't match: {getattr(self, attr)} vs {getattr(other, attr)}")

    def __add__(self, other: 'BallTrackingDataset') -> 'BallTrackingDataset':
        """Merge two datasets"""
        self._validate_compatibility(other)

        combined_video_pairs = self.video_pairs + other.video_pairs
        combined_frame_index = list(self.frame_index)
        video_offset = len(self.video_pairs)

        for video_idx, frame_idx in other.frame_index:
            combined_frame_index.append((video_idx + video_offset, frame_idx))

        combined_video_info_cache = {**self._video_info_cache, **other._video_info_cache}
        combined_label_cache = {**self._label_cache, **other._label_cache}

        internal_data = {
            'video_pairs': combined_video_pairs,
            'frame_index': combined_frame_index,
            'video_info_cache': combined_video_info_cache,
            'label_cache': combined_label_cache
        }

        return BallTrackingDataset(
            video_ext=self.video_ext,
            csv_suffix=self.csv_suffix,
            normalize_coords=self.normalize_coords,
            normalize_pixels=self.normalize_pixels,
            num_frames=self.num_frames,
            _internal_data=internal_data
        )

    def __radd__(self, other):
        """Support sum([dataset1, dataset2, dataset3])"""
        if other == 0:
            return self
        return self.__add__(other)

    @classmethod
    def merge_multiple(cls, datasets: List['BallTrackingDataset']) -> 'BallTrackingDataset':
        """Merge multiple datasets"""
        if not datasets:
            raise ValueError("Empty dataset list")
        if len(datasets) == 1:
            return datasets[0]

        from functools import reduce
        return reduce(lambda a, b: a + b, datasets)


# Usage example
if __name__ == "__main__":
    match_dir = base_dir / 'Dataset' / 'Professional' / 'match1'

    # 3-frame input dataset (9 channels total)
    dataset = BallTrackingDataset(str(match_dir), normalize_coords=True, normalize_pixels=True)

    frames, labels = dataset[0]
    print(f"Frames shape: {frames.shape}")  # Should be (9, H, W)
    print(dataset[0])
