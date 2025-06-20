import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionPromptLayer(nn.Module):
    """Motion prompt layer with learnable parameters for motion attention."""

    def __init__(self):
        super(MotionPromptLayer, self).__init__()
        self.slope = nn.Parameter(torch.tensor(16.24))
        self.shift = nn.Parameter(torch.tensor(0.28))

    def forward(self, reshaped_input):
        # reshaped_input: [B, 3, 3, H, W]
        batch_size, frames, channels, height, width = reshaped_input.shape

        # Convert to grayscale and compute frame differences
        frame_0 = reshaped_input[:, 0].mean(dim=1, keepdim=True)  # [B, 1, H, W]
        frame_1 = reshaped_input[:, 1].mean(dim=1, keepdim=True)  # [B, 1, H, W]
        frame_2 = reshaped_input[:, 2].mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # Compute absolute differences
        diff_01 = torch.abs(frame_1 - frame_0)
        diff_12 = torch.abs(frame_2 - frame_1)
        frame_diff = torch.cat([diff_01, diff_12], dim=1)  # [B, 2, H, W]

        # Apply power normalization
        motion_attention = torch.sigmoid(self.slope * frame_diff + self.shift)

        # Return multiple outputs as in original architecture
        return [
            motion_attention.unsqueeze(1).expand(-1, 3, -1, -1, -1),  # [B, 2, 3, H, W]
            motion_attention,  # [B, 2, H, W]
            []  # Empty tuple as in original
        ]


class MotionIncorporationLayerV1(nn.Module):
    """Motion incorporation layer for fusing visual and motion features."""

    def __init__(self):
        super(MotionIncorporationLayerV1, self).__init__()

    def forward(self, visual_features, motion_attention):
        # visual_features: [B, 3, H, W] from conv2d_7
        # motion_attention: [B, 2, H, W] from motion_prompt_layer

        # Use mean of motion attention for all 3 channels
        motion_mean = motion_attention.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        motion_expanded = motion_mean.expand_as(visual_features)  # [B, 3, H, W]

        # Element-wise multiplication for motion-aware enhancement
        enhanced_features = visual_features * motion_expanded

        return enhanced_features


class TrackNetV4(nn.Module):
    """TrackNetV4: Enhanced object tracking with motion attention maps."""

    def __init__(self):
        super(TrackNetV4, self).__init__()

        # Encoder
        self.conv2d = nn.Conv2d(9, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.conv2d_1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(64)
        self.max_pooling2d = nn.MaxPool2d(2, stride=2)

        self.conv2d_2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(128)
        self.max_pooling2d_1 = nn.MaxPool2d(2, stride=2)

        self.conv2d_3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn_3 = nn.BatchNorm2d(256)
        self.max_pooling2d_2 = nn.MaxPool2d(2, stride=2)

        # Decoder
        self.up_sampling2d = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2d_4 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn_4 = nn.BatchNorm2d(256)

        self.up_sampling2d_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2d_5 = nn.Conv2d(384, 128, 3, padding=1)
        self.bn_5 = nn.BatchNorm2d(128)

        self.up_sampling2d_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2d_6 = nn.Conv2d(192, 64, 3, padding=1)
        self.bn_6 = nn.BatchNorm2d(64)

        self.conv2d_7 = nn.Conv2d(64, 3, 3, padding=1)

        # Motion modules
        self.motion_prompt_layer = MotionPromptLayer()
        self.motion_incorporation_layer_v1 = MotionIncorporationLayerV1()

    def forward(self, x):
        # Input: [B, 9, H, W]

        # Reshape for motion processing
        batch_size, channels, height, width = x.shape
        reshaped_input = x.view(batch_size, 3, 3, height, width)  # [B, 3, 3, H, W]

        # Encoder path
        x1 = F.relu(self.bn(self.conv2d(x)))
        x2 = F.relu(self.bn_1(self.conv2d_1(x1)))
        p1 = self.max_pooling2d(x2)

        x3 = F.relu(self.bn_2(self.conv2d_2(p1)))
        p2 = self.max_pooling2d_1(x3)

        x4 = F.relu(self.bn_3(self.conv2d_3(p2)))
        p3 = self.max_pooling2d_2(x4)

        # Decoder path with skip connections
        up1 = self.up_sampling2d(p3)
        cat1 = torch.cat([x4, up1], dim=1)  # Skip connection
        d1 = F.relu(self.bn_4(self.conv2d_4(cat1)))

        up2 = self.up_sampling2d_1(d1)
        cat2 = torch.cat([x3, up2], dim=1)  # Skip connection
        d2 = F.relu(self.bn_5(self.conv2d_5(cat2)))

        up3 = self.up_sampling2d_2(d2)
        cat3 = torch.cat([x2, up3], dim=1)  # Skip connection
        d3 = F.relu(self.bn_6(self.conv2d_6(cat3)))

        # Visual features before motion fusion
        visual_features = self.conv2d_7(d3)  # [B, 3, H, W]

        # Motion attention processing
        motion_outputs = self.motion_prompt_layer(reshaped_input)
        motion_attention = motion_outputs[1]  # [B, 2, H, W]

        # Motion-aware fusion
        enhanced_features = self.motion_incorporation_layer_v1(visual_features, motion_attention)

        # Final output with sigmoid
        output = torch.sigmoid(enhanced_features)
        return output


class WeightedBCELoss(nn.Module):
    """Weighted binary cross entropy loss for imbalanced data."""

    def __init__(self):
        super(WeightedBCELoss, self).__init__()

    def forward(self, y_pred, y_true):
        eps = 1e-7
        y_pred = torch.clamp(y_pred, eps, 1 - eps)
        w = y_pred.detach()  # 修正：w应该等于预测值
        loss = -(
                (1 - w) ** 2 * y_true * torch.log(y_pred) +
                w ** 2 * (1 - y_true) * torch.log(1 - y_pred)
        ).mean()
        return loss


def postprocess_heatmap(heatmap, threshold=0.5):
    """Extract ball coordinates from heatmap."""
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
    model = TrackNetV4()
    criterion = WeightedBCELoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)

    # Test forward pass
    x = torch.randn(1, 9, 288, 512)
    y = torch.randint(0, 2, (1, 3, 288, 512)).float()

    output = model(x)
    loss = criterion(output, y)

    print(f"Input: {x.shape} -> Output: {output.shape}")
    print(f"Loss: {loss.item():.4f}")

    coords = postprocess_heatmap(output)
    print(f"Coordinates: {coords[0]}")

    # Parameter count
    motion_params = sum(p.numel() for p in model.motion_prompt_layer.parameters())
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Motion prompt parameters: {motion_params}")
    print(f"Total parameters: {total_params:,}")
