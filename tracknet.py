import torch
import torch.nn as nn
import torch.nn.functional as F


class TrackNet(nn.Module):
    """TrackNetV2: Efficient Shuttlecock Tracking Network
    Input: 512×288×9 -> Output: 512×288×3
    """

    def __init__(self):
        super(TrackNet, self).__init__()

        # Encoder
        self.conv2d_1 = nn.Conv2d(9, 64, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(64)
        self.conv2d_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(64)
        self.max_pooling_1 = nn.MaxPool2d(2, stride=2)

        self.conv2d_3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn_3 = nn.BatchNorm2d(128)
        self.conv2d_4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_4 = nn.BatchNorm2d(128)
        self.max_pooling_2 = nn.MaxPool2d(2, stride=2)

        self.conv2d_5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn_5 = nn.BatchNorm2d(256)
        self.conv2d_6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_6 = nn.BatchNorm2d(256)
        self.conv2d_7 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_7 = nn.BatchNorm2d(256)
        self.max_pooling_3 = nn.MaxPool2d(2, stride=2)

        # Bottleneck
        self.conv2d_8 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn_8 = nn.BatchNorm2d(512)
        self.conv2d_9 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn_9 = nn.BatchNorm2d(512)
        self.conv2d_10 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn_10 = nn.BatchNorm2d(512)

        # Decoder with skip connections
        self.up_sampling_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2d_11 = nn.Conv2d(768, 256, 3, padding=1)  # 512+256
        self.bn_11 = nn.BatchNorm2d(256)
        self.conv2d_12 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_12 = nn.BatchNorm2d(256)
        self.conv2d_13 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_13 = nn.BatchNorm2d(256)

        self.up_sampling_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2d_14 = nn.Conv2d(384, 128, 3, padding=1)  # 256+128
        self.bn_14 = nn.BatchNorm2d(128)
        self.conv2d_15 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_15 = nn.BatchNorm2d(128)

        self.up_sampling_3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2d_16 = nn.Conv2d(192, 64, 3, padding=1)  # 128+64
        self.bn_16 = nn.BatchNorm2d(64)
        self.conv2d_17 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_17 = nn.BatchNorm2d(64)

        # Output layer
        self.conv2d_18 = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.bn_1(self.conv2d_1(x)))
        x2 = F.relu(self.bn_2(self.conv2d_2(x1)))
        p1 = self.max_pooling_1(x2)

        x3 = F.relu(self.bn_3(self.conv2d_3(p1)))
        x4 = F.relu(self.bn_4(self.conv2d_4(x3)))
        p2 = self.max_pooling_2(x4)

        x5 = F.relu(self.bn_5(self.conv2d_5(p2)))
        x6 = F.relu(self.bn_6(self.conv2d_6(x5)))
        x7 = F.relu(self.bn_7(self.conv2d_7(x6)))
        p3 = self.max_pooling_3(x7)

        # Bottleneck
        x8 = F.relu(self.bn_8(self.conv2d_8(p3)))
        x9 = F.relu(self.bn_9(self.conv2d_9(x8)))
        x10 = F.relu(self.bn_10(self.conv2d_10(x9)))

        # Decoder
        up1 = self.up_sampling_1(x10)
        cat1 = torch.cat([up1, x7], dim=1)
        x11 = F.relu(self.bn_11(self.conv2d_11(cat1)))
        x12 = F.relu(self.bn_12(self.conv2d_12(x11)))
        x13 = F.relu(self.bn_13(self.conv2d_13(x12)))

        up2 = self.up_sampling_2(x13)
        cat2 = torch.cat([up2, x4], dim=1)
        x14 = F.relu(self.bn_14(self.conv2d_14(cat2)))
        x15 = F.relu(self.bn_15(self.conv2d_15(x14)))

        up3 = self.up_sampling_3(x15)
        cat3 = torch.cat([up3, x2], dim=1)
        x16 = F.relu(self.bn_16(self.conv2d_16(cat3)))
        x17 = F.relu(self.bn_17(self.conv2d_17(x16)))

        # Output heatmaps
        output = torch.sigmoid(self.conv2d_18(x17))
        return output


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss - 严格按照论文表述
    WBCE = -∑[(1-w)²ŷlog(y) + w²(1-ŷ)log(1-y)] where w = ŷ
    """

    def __init__(self):
        super(WeightedBCELoss, self).__init__()

    def forward(self, y_pred, y_true):
        # 按照论文：w = ŷ (ground truth)
        w = y_true

        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)

        # 严格按照论文公式实现
        term1 = (1 - w) ** 2 * y_true * torch.log(y_pred)
        term2 = w ** 2 * (1 - y_true) * torch.log(1 - y_pred)

        return -(term1 + term2).mean()


def postprocess_heatmap(heatmap, threshold=0.5):
    """Extract ball coordinates from heatmap"""
    batch_size, channels, height, width = heatmap.shape
    coordinates = []

    for b in range(batch_size):
        batch_coords = []
        for c in range(channels):
            binary_map = (heatmap[b, c] > threshold).float()
            if binary_map.sum() > 0:
                y_indices, x_indices = torch.where(binary_map > 0)
                center_x = x_indices.float().mean().item()
                center_y = y_indices.float().mean().item()
                batch_coords.append((center_x, center_y))
            else:
                batch_coords.append(None)
        coordinates.append(batch_coords)

    return coordinates


# Training setup
if __name__ == "__main__":
    model = TrackNet()
    criterion = WeightedBCELoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)

    # Example: 3 consecutive frames (9 channels) -> 3 heatmaps
    x = torch.randn(1, 9, 288, 512)
    y = torch.randint(0, 2, (1, 3, 288, 512)).float()

    output = model(x)
    loss = criterion(output, y)

    print(f"Input: {x.shape} -> Output: {output.shape}")
    print(f"Loss: {loss.item():.4f}")

    coords = postprocess_heatmap(output)
    print(f"Coordinates: {coords[0]}")
