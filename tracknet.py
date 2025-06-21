import torch
import torch.nn as nn
import torch.nn.functional as F


class TrackNet(nn.Module):
    """TrackNet with U-Net skip connections for shuttlecock tracking"""

    def __init__(self, input_frames=3, output_frames=3):
        super(TrackNet, self).__init__()

        # Encoder
        self.conv2d = nn.Conv2d(input_frames * 3, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.conv2d_1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(64)
        self.max_pooling2d = nn.MaxPool2d(2, stride=2)

        self.conv2d_2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(128)
        self.conv2d_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_3 = nn.BatchNorm2d(128)
        self.max_pooling2d_1 = nn.MaxPool2d(2, stride=2)

        self.conv2d_4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn_4 = nn.BatchNorm2d(256)
        self.conv2d_5 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_5 = nn.BatchNorm2d(256)
        self.conv2d_6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_6 = nn.BatchNorm2d(256)
        self.max_pooling2d_2 = nn.MaxPool2d(2, stride=2)

        self.conv2d_7 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn_7 = nn.BatchNorm2d(512)
        self.conv2d_8 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn_8 = nn.BatchNorm2d(512)
        self.conv2d_9 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn_9 = nn.BatchNorm2d(512)

        # Decoder with skip connections
        self.up_sampling2d = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2d_10 = nn.Conv2d(768, 256, 3, padding=1)  # 512 + 256 skip
        self.bn_10 = nn.BatchNorm2d(256)
        self.conv2d_11 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_11 = nn.BatchNorm2d(256)
        self.conv2d_12 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_12 = nn.BatchNorm2d(256)

        self.up_sampling2d_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2d_13 = nn.Conv2d(384, 128, 3, padding=1)  # 256 + 128 skip
        self.bn_13 = nn.BatchNorm2d(128)
        self.conv2d_14 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_14 = nn.BatchNorm2d(128)

        self.up_sampling2d_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2d_15 = nn.Conv2d(192, 64, 3, padding=1)  # 128 + 64 skip
        self.bn_15 = nn.BatchNorm2d(64)
        self.conv2d_16 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_16 = nn.BatchNorm2d(64)

        # MIMO output
        self.conv2d_17 = nn.Conv2d(64, output_frames, 1, padding=0)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight)
                if m.bias is not None:
                    nn.init.uniform_(m.bias)

    def forward(self, x):
        """Forward pass: [B, 9, 288, 512] -> [B, 3, 288, 512]"""
        # Encoder
        x1 = F.relu(self.bn(self.conv2d(x)))
        x2 = F.relu(self.bn_1(self.conv2d_1(x1)))
        p1 = self.max_pooling2d(x2)

        x3 = F.relu(self.bn_2(self.conv2d_2(p1)))
        x4 = F.relu(self.bn_3(self.conv2d_3(x3)))
        p2 = self.max_pooling2d_1(x4)

        x5 = F.relu(self.bn_4(self.conv2d_4(p2)))
        x6 = F.relu(self.bn_5(self.conv2d_5(x5)))
        x7 = F.relu(self.bn_6(self.conv2d_6(x6)))
        p3 = self.max_pooling2d_2(x7)

        x8 = F.relu(self.bn_7(self.conv2d_7(p3)))
        x9 = F.relu(self.bn_8(self.conv2d_8(x8)))
        x10 = F.relu(self.bn_9(self.conv2d_9(x9)))

        # Decoder
        up1 = self.up_sampling2d(x10)
        cat1 = torch.cat([x7, up1], dim=1)
        d1 = F.relu(self.bn_10(self.conv2d_10(cat1)))
        d2 = F.relu(self.bn_11(self.conv2d_11(d1)))
        d3 = F.relu(self.bn_12(self.conv2d_12(d2)))

        up2 = self.up_sampling2d_1(d3)
        cat2 = torch.cat([x4, up2], dim=1)
        d4 = F.relu(self.bn_13(self.conv2d_13(cat2)))
        d5 = F.relu(self.bn_14(self.conv2d_14(d4)))

        up3 = self.up_sampling2d_2(d5)
        cat3 = torch.cat([x2, up3], dim=1)
        d6 = F.relu(self.bn_15(self.conv2d_15(cat3)))
        d7 = F.relu(self.bn_16(self.conv2d_16(d6)))

        return torch.sigmoid(self.conv2d_17(d7))


class WeightedBCELoss(nn.Module):
    """Weighted BCE Loss: w = y_pred"""

    def forward(self, y_pred, y_true):
        eps = 1e-7
        y_pred = torch.clamp(y_pred, eps, 1 - eps)
        w = y_pred.detach()
        return -(
                (1 - w) ** 2 * y_true * torch.log(y_pred) +
                w ** 2 * (1 - y_true) * torch.log(1 - y_pred)
        ).mean()


def postprocess_heatmap(heatmap, threshold=0.5):
    """Extract coordinates from heatmap"""
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


if __name__ == "__main__":
    model = TrackNet(input_frames=3, output_frames=3)
    criterion = WeightedBCELoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)

    x = torch.randn(1, 9, 288, 512)
    y = torch.rand(1, 3, 288, 512)

    output = model(x)
    loss = criterion(output, y)

    print(f"Input: {x.shape} -> Output: {output.shape}")
    print(f"Loss: {loss.item():.4f}")

    coords = postprocess_heatmap(output)
    print(f"Coordinates: {coords[0]}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
