import torch
import torchvision.transforms as transforms

from dataset_preprocessing.dataset_generator import FrameHeatmapDataset

if __name__ == "__main__":
    # 使用示例
    root_dir = "dataset/Professional_reorg_train"

    # 1. 基础使用
    dataset = FrameHeatmapDataset(root_dir)
    print(f"数据集大小: {len(dataset)}")

    # 2. 自定义变换
    origin_dataset = FrameHeatmapDataset(
        root_dir=root_dir,
        transform=transforms.Compose([
            transforms.Resize((288, 512)),
            transforms.ToTensor(),
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

    #分割为80%训练集和20%验证集
    train_size = int(0.8 * len(origin_dataset))
    val_size = len(origin_dataset) - train_size

    #设定随机种子以确保分割可重复性
    torch.manual_seed(26)
    # 分割数据集
    train_dataset, val_dataset = torch.utils.data.random_split(origin_dataset, [train_size, val_size])

    # 训练集和验证集形状和大小
    print(f"训练集大小: {len(train_dataset)}，训练集形状: {train_dataset[0][0].shape}, {train_dataset[0][1].shape}")
    print(f"验证集大小: {len(val_dataset)}，验证集形状: {val_dataset[0][0].shape}, {val_dataset[0][1].shape}")

    # DataLoader 加载数据
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2
    )
    print("DataLoader 准备完毕")
    # 测试加载数据
    for inputs, heatmaps in train_loader:
        print(f"输入形状: {inputs.shape}, 热力图形状: {heatmaps.shape}")
        break  # 只测试第一批数据

    for inputs, heatmaps in val_loader:
        print(f"验证集输入形状: {inputs.shape}, 验证集热力图形状: {heatmaps.shape}")
        break




