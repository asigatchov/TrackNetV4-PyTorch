"""
Frame Heatmap Dataset for PyTorch

处理帧图像和对应热力图的数据集，用于TrackNetV2训练。

数据集结构：
dataset_reorg_train/
├── match1/
│   ├── inputs/frame1/0.jpg,1.jpg... (512×288)
│   └── heatmaps/frame1/0.jpg,1.jpg... (热力图)
└── match2/...

数据输出格式：
- inputs: (9, 288, 512) - 3张RGB图片拼接，归一化至[-1,1]
- heatmaps: (3, 288, 512) - 3张灰度热力图拼接，归一化至[0,1]
- 严格保证输入输出顺序对应

Author: Generated for TrackNetV2 training
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import glob


class FrameHeatmapDataset(Dataset):
    def __init__(self, root_dir, transform=None, heatmap_transform=None):
        """
        Args:
            root_dir: 数据根目录
            transform: 输入图片变换（默认归一化至[-1,1]）
            heatmap_transform: 热力图变换（默认归一化至[0,1]）
        """
        self.root_dir = Path(root_dir)

        # 默认变换
        self.transform = transform or transforms.Compose([
            transforms.Resize((288, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [0,1] -> [-1,1]
        ])

        self.heatmap_transform = heatmap_transform or transforms.Compose([
            transforms.Resize((288, 512)),
            transforms.ToTensor()  # [0,1]
        ])

        # 扫描数据集
        self.data_items = self._scan_dataset()

    def _scan_dataset(self):
        """扫描数据集并构建索引"""
        data_items = []
        match_dirs = sorted([d for d in self.root_dir.iterdir()
                             if d.is_dir() and d.name.startswith('match')])

        print(f"扫描 {len(match_dirs)} 个match文件夹...")

        for match_dir in match_dirs:
            inputs_dir = match_dir / 'inputs'
            heatmaps_dir = match_dir / 'heatmaps'

            if not inputs_dir.exists() or not heatmaps_dir.exists():
                continue

            # 获取匹配的frame文件夹
            input_frames = {d.name for d in inputs_dir.iterdir() if d.is_dir()}
            heatmap_frames = {d.name for d in heatmaps_dir.iterdir() if d.is_dir()}
            common_frames = input_frames.intersection(heatmap_frames)

            for frame_name in sorted(common_frames):
                input_frame_dir = inputs_dir / frame_name
                heatmap_frame_dir = heatmaps_dir / frame_name

                # 获取并排序图片
                input_images = sorted(glob.glob(str(input_frame_dir / "*.jpg")),
                                      key=lambda x: int(Path(x).stem))
                heatmap_images = sorted(glob.glob(str(heatmap_frame_dir / "*.jpg")),
                                        key=lambda x: int(Path(x).stem))

                if len(input_images) != len(heatmap_images) or len(input_images) < 3:
                    continue

                # 三帧一组
                for i in range(len(input_images) - 2):
                    data_items.append({
                        'inputs': input_images[i:i + 3],
                        'heatmaps': heatmap_images[i:i + 3],
                        'match': match_dir.name,
                        'frame': frame_name,
                        'idx': i
                    })

        print(f"找到 {len(data_items)} 个有效样本")
        return data_items

    def _load_image(self, image_path, is_heatmap=False):
        """加载图片"""
        try:
            image = Image.open(image_path)
            if is_heatmap:
                # 热力图转灰度
                if image.mode != 'L':
                    image = image.convert('L')
                return self.heatmap_transform(image)
            else:
                # 输入图片转RGB
                image = image.convert('RGB')
                return self.transform(image)
        except Exception as e:
            print(f"图片加载失败: {image_path}")
            # 返回零张量
            channels = 1 if is_heatmap else 3
            return torch.zeros(channels, 288, 512)

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        """
        返回:
            inputs: (9, 288, 512) - 3张RGB图片，[-1,1]
            heatmaps: (3, 288, 512) - 3张灰度热力图，[0,1]
        """
        item = self.data_items[idx]

        # 加载3张输入图片
        inputs = [self._load_image(path, False) for path in item['inputs']]
        # 加载3张热力图
        heatmaps = [self._load_image(path, True) for path in item['heatmaps']]

        # 拼接
        inputs = torch.cat(inputs, dim=0)  # (9, 288, 512)
        heatmaps = torch.cat(heatmaps, dim=0)  # (3, 288, 512)

        return inputs, heatmaps

    def get_info(self, idx):
        """获取样本信息"""
        return self.data_items[idx]


if __name__ == "__main__":
    # 使用示例
    root_dir = "../dataset/Test_reorg_train"

    # 1. 基础使用
    dataset = FrameHeatmapDataset(root_dir)
    print(f"数据集大小: {len(dataset)}")

    # 2. 自定义变换
    custom_dataset = FrameHeatmapDataset(
        root_dir=root_dir,
        transform=transforms.Compose([
            transforms.Resize((288, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1,1]
        ]),
        heatmap_transform=transforms.Compose([
            transforms.Resize((288, 512)),
            transforms.ToTensor()  # [0,1]
        ])
    )

    # 3. 创建DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2
    )

    # 4. 测试数据加载
    print("\n测试数据加载:")
    for batch_idx, (inputs, heatmaps) in enumerate(dataloader):
        print(f"Batch {batch_idx}: inputs{inputs.shape}, heatmaps{heatmaps.shape}")
        print(f"  输入范围: [{inputs.min():.3f}, {inputs.max():.3f}]")
        print(f"  热力图范围: [{heatmaps.min():.3f}, {heatmaps.max():.3f}]")

        if batch_idx == 0:
            info = dataset.get_info(0)
            print(f"  样本信息: {info['match']}/{info['frame']}, 起始索引{info['idx']}")
        break
