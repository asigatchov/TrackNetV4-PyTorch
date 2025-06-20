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

# Configuration for input/output frames
CONFIG = {
    "input_frames": 3,  # Number of input frames (channels = input_frames * 3)
    "output_frames": 3,  # Number of output frames/labels
    "normalize_coords": True,
    "normalize_pixels": True,
    "video_ext": ".mp4",
    "csv_suffix": "_ball.csv"
}


class BallTrackingDataset(Dataset):
    """
    Ball tracking dataset with configurable input/output frames

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
                 config: Dict = None,
                 _internal_data: Dict = None):
        """
        Args:
            match_folder: Path containing csv and video subfolders
            config: Configuration dict with input_frames, output_frames, etc.
            _internal_data: Internal use for dataset merging
        """
        # Use provided config or default CONFIG
        self.config = config if config is not None else CONFIG.copy()

        self.input_frames = self.config.get("input_frames", 3)
        self.output_frames = self.config.get("output_frames", 3)
        self.video_ext = self.config.get("video_ext", ".mp4")
        self.csv_suffix = self.config.get("csv_suffix", "_ball.csv")
        self.normalize_coords = self.config.get("normalize_coords", True)
        self.normalize_pixels = self.config.get("normalize_pixels", True)

        # Validate frame configuration
        if self.input_frames < 1 or self.output_frames < 1:
            raise ValueError("input_frames and output_frames must be >= 1")

        if self.input_frames % 2 == 0:
            raise ValueError("input_frames must be odd number for center alignment")

        self._video_info_cache = {}
        self._label_cache = {}

        if _internal_data is not None:
            self._init_from_internal_data(_internal_data)
        else:
            if match_folder is None:
                raise ValueError("match_folder is required when not using _internal_data")
            self._init_from_folder(match_folder)

        print(f"Dataset config: {self.input_frames} input frames -> {self.output_frames} output frames")

    def _init_from_folder(self, match_folder: str):
        """Initialize dataset from folder"""
        self.match_path = Path(match_folder)
        self.video_folder = self.match_path / 'video'
        self.csv_folder = self.match_path / 'csv'

        self._validate_structure()
        self.video_pairs = self._discover_pairs()
        self.frame_index = self._build_frame_index()

        print(f"Dataset loaded: {len(self.video_pairs)} videos, {len(self.frame_index)} labeled frame sequences")

    def _init_from_internal_data(self, data: Dict):
        """Initialize from internal data (for merging)"""
        self.video_pairs = data['video_pairs']
        self.frame_index = data['frame_index']
        self._video_info_cache = data['video_info_cache']
        self._label_cache = data['label_cache']

        print(f"Dataset merged: {len(self.video_pairs)} videos, {len(self.frame_index)} labeled frame sequences")

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

    def _get_output_frame_indices(self, center_frame: int) -> List[int]:
        """Get frame indices for output based on input/output frame configuration"""
        input_half = self.input_frames // 2

        if self.output_frames == self.input_frames:
            # Matched case: output all input frames
            start_frame = center_frame - input_half
            return list(range(start_frame, start_frame + self.input_frames))
        else:
            # Unmatched case: output middle frame(s)
            if self.output_frames == 1:
                # Single output: center frame
                return [center_frame]
            elif self.output_frames == 2:
                # Two outputs: center two frames (for even input_frames, this is the middle two)
                return [center_frame, center_frame + 1]
            else:
                # Multiple outputs: centered around middle
                output_half = self.output_frames // 2
                start_frame = center_frame - output_half
                return list(range(start_frame, start_frame + self.output_frames))

    def _build_frame_index(self) -> List[Tuple[int, int]]:
        """Build frame index ensuring sufficient frames for input and required labels for output"""
        index = []
        input_half = self.input_frames // 2

        for video_idx, pair in enumerate(self.video_pairs):
            try:
                video_info = self._get_video_info(pair['video_path'])
                df = self._load_labels(pair['csv_path'])
                labeled_frames = set(df['Frame'].astype(int).tolist())

                for center_frame in labeled_frames:
                    # Check input frame range
                    input_start = center_frame - input_half
                    input_end = center_frame + input_half

                    if input_start < 0 or input_end >= video_info['frame_count']:
                        continue

                    # Get required output frame indices
                    output_indices = self._get_output_frame_indices(center_frame)

                    # Check that all required output frames have labels
                    all_output_labeled = True
                    for output_idx in output_indices:
                        if output_idx not in labeled_frames:
                            all_output_labeled = False
                            break

                        # Validate each output frame's label data
                        frame_data = df[df['Frame'] == output_idx]
                        if len(frame_data) != 1:
                            all_output_labeled = False
                            break

                        row = frame_data.iloc[0]
                        if pd.isna(row['Visibility']) or pd.isna(row['X']) or pd.isna(row['Y']):
                            all_output_labeled = False
                            break

                    if all_output_labeled:
                        index.append((video_idx, center_frame))

            except Exception as e:
                logger.error(f"Error processing video {pair['video_path']}: {e}")
                raise

        if not index:
            raise ValueError("No valid labeled frame sequences found in dataset")

        return sorted(index)

    def _read_consecutive_frames(self, video_path: Path, center_frame: int) -> np.ndarray:
        """Read consecutive input frames centered on center_frame"""
        input_half = self.input_frames // 2
        start_frame = center_frame - input_half
        end_frame = center_frame + input_half

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
        return np.stack(frames, axis=0)  # Shape: (input_frames, H, W, C)

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

    def _get_output_labels(self, csv_path: Path, center_frame: int, video_info: Dict[str, int]) -> List[
        Dict[str, float]]:
        """Get labels for output frames based on configuration"""
        output_indices = self._get_output_frame_indices(center_frame)

        labels = []
        for frame_idx in output_indices:
            label = self._get_frame_label(csv_path, frame_idx, video_info)
            labels.append(label)

        return labels

    def __len__(self) -> int:
        return len(self.frame_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Returns:
            frames: (C*input_frames, H, W) tensor
            labels: List of output_frames dicts, each containing {'visibility': scalar, 'x': scalar, 'y': scalar}
        """
        if idx >= len(self.frame_index):
            raise IndexError(f"Index {idx} out of range")

        video_idx, center_frame = self.frame_index[idx]
        pair = self.video_pairs[video_idx]

        # Read input frames
        frames = self._read_consecutive_frames(pair['video_path'], center_frame)
        video_info = self._get_video_info(pair['video_path'])

        # Get output labels
        labels = self._get_output_labels(pair['csv_path'], center_frame, video_info)

        # Convert to tensor and reshape
        frames_tensor = torch.from_numpy(frames).float()  # (input_frames, H, W, C)

        if self.normalize_pixels:
            frames_tensor = frames_tensor / 255.0

        # Reshape to (C*input_frames, H, W)
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # (input_frames, C, H, W)
        frames_tensor = frames_tensor.reshape(-1, frames_tensor.shape[2],
                                              frames_tensor.shape[3])  # (C*input_frames, H, W)

        # Convert labels to tensors
        label_tensors = []
        for label in labels:
            label_tensor = {
                'visibility': torch.tensor(label['visibility'], dtype=torch.float32),
                'x': torch.tensor(label['x'], dtype=torch.float32),
                'y': torch.tensor(label['y'], dtype=torch.float32)
            }
            label_tensors.append(label_tensor)

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

        # Check if configs are compatible
        for key in ['input_frames', 'output_frames', 'video_ext', 'csv_suffix', 'normalize_coords', 'normalize_pixels']:
            if self.config.get(key) != other.config.get(key):
                raise ValueError(
                    f"Config {key} settings don't match: {self.config.get(key)} vs {other.config.get(key)}")

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
            match_folder=None,
            config=self.config,
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


def create_heatmaps_from_labels(labels: List[Dict], image_shape: Tuple[int, int],
                                sigma: float = 2.0, normalized_coords: bool = True) -> torch.Tensor:
    """
    Create multi-channel heatmap tensor from labels

    Args:
        labels: List of label dicts, each with 'visibility', 'x', 'y'
        image_shape: (height, width) of the image
        sigma: Standard deviation for Gaussian blob
        normalized_coords: Whether coordinates are normalized to [0,1]

    Returns:
        heatmaps: (num_labels, H, W) tensor with Gaussian blobs at ball positions
    """
    height, width = image_shape
    num_labels = len(labels)
    heatmaps = torch.zeros(num_labels, height, width)

    # Create coordinate grids
    y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')

    for i, label in enumerate(labels):
        if label['visibility'] > 0:
            # Get ball coordinates
            if normalized_coords:
                ball_x = label['x'] * width
                ball_y = label['y'] * height
            else:
                ball_x = label['x']
                ball_y = label['y']

            # Create Gaussian heatmap
            gaussian = torch.exp(-((x_coords - ball_x) ** 2 + (y_coords - ball_y) ** 2) / (2 * sigma ** 2))
            heatmaps[i] = gaussian

    return heatmaps


# Usage examples
if __name__ == "__main__":
    match_dir = base_dir / 'Dataset' / 'Professional' / 'match1'

    # Example 1: 3 input frames -> 3 output frames (matched)
    config_3in3out = {
        "input_frames": 3,
        "output_frames": 3,
        "normalize_coords": True,
        "normalize_pixels": True,
        "video_ext": ".mp4",
        "csv_suffix": "_ball.csv"
    }

    dataset1 = BallTrackingDataset(str(match_dir), config=config_3in3out)
    frames, labels = dataset1[0]
    print(f"3in3out - Frames shape: {frames.shape}, Labels count: {len(labels)}")
    print(dataset1[0])
