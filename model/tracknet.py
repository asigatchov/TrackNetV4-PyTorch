import torch
import torch.nn as nn


class MotionPrompt(nn.Module):
    """Extract motion attention from frame differences"""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1))  # slope
        self.b = nn.Parameter(torch.randn(1))  # shift

    def forward(self, x):
        # x: [B,3,3,H,W] -> motion_info:[B,2,3,H,W], motion:[B,2,H,W], None
        B, T, C, H, W = x.shape
        gray = x.mean(dim=2)  # [B,3,H,W]

        # Frame diff: [t+1] - [t]
        diffs = [gray[:, i + 1] - gray[:, i] for i in range(T - 1)]
        D = torch.stack(diffs, dim=1)  # [B,2,H,W]

        # |D| -> attention
        A = torch.sigmoid(self.a * D.abs() + self.b)  # [B,2,H,W]

        return A.unsqueeze(2).expand(B, 2, 3, H, W), A, None


class MotionFusion(nn.Module):
    """Fuse visual features with motion attention"""

    def __init__(self):
        super().__init__()

    def forward(self, vis, mot):
        # vis:[B,3,H,W], mot:[B,2,H,W] -> [B,3,H,W]
        return torch.stack([vis[:, 0],
                            mot[:, 0] * vis[:, 1],
                            mot[:, 1] * vis[:, 2]], dim=1)


class TrackNet(nn.Module):
    """Motion-Enhanced Sports Tracking Network"""

    def __init__(self):
        super().__init__()

        # Motion modules
        self.motion = MotionPrompt()
        self.fusion = MotionFusion()

        # Encoder blocks
        self.conv1 = self._make_encoder_block(9, 64, 2)
        self.conv2 = self._make_encoder_block(64, 128, 2)
        self.conv3 = self._make_encoder_block(128, 256, 3)
        self.conv4 = self._make_encoder_block(256, 512, 3)

        # Decoder blocks
        self.up1 = self._make_decoder_block(768, 256, 3)
        self.up2 = self._make_decoder_block(384, 128, 2)
        self.up3 = self._make_decoder_block(192, 64, 2)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.out = nn.Conv2d(64, 3, 1)

    def _make_encoder_block(self, in_ch, out_ch, n_conv):
        layers = []
        for i in range(n_conv):
            ch_in = in_ch if i == 0 else out_ch
            layers.extend([
                nn.Conv2d(ch_in, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ])
        return nn.Sequential(*layers)

    def _make_decoder_block(self, in_ch, out_ch, n_conv):
        layers = []
        for i in range(n_conv):
            ch_in = in_ch if i == 0 else out_ch
            layers.extend([
                nn.Conv2d(ch_in, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ])
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [B,9,288,512] -> [B,3,288,512]
        B = x.size(0)

        # Motion analysis
        _, motion, _ = self.motion(x.view(B, 3, 3, 288, 512))

        # Encoder with skip connections
        x1 = self.conv1(x)  # [B,64,288,512]
        p1 = self.pool(x1)  # [B,64,144,256]

        x2 = self.conv2(p1)  # [B,128,144,256]
        p2 = self.pool(x2)  # [B,128,72,128]

        x3 = self.conv3(p2)  # [B,256,72,128]
        p3 = self.pool(x3)  # [B,256,36,64]

        x4 = self.conv4(p3)  # [B,512,36,64] (bottleneck)

        # Decoder with skip connections
        up1 = self.upsample(x4)  # [B,512,72,128]
        up1 = self.up1(torch.cat([up1, x3], dim=1))  # [B,256,72,128]

        up2 = self.upsample(up1)  # [B,256,144,256]
        up2 = self.up2(torch.cat([up2, x2], dim=1))  # [B,128,144,256]

        up3 = self.upsample(up2)  # [B,128,288,512]
        up3 = self.up3(torch.cat([up3, x1], dim=1))  # [B,64,288,512]

        # Motion-enhanced output
        visual = self.out(up3)  # [B,3,288,512]
        enhanced = self.fusion(visual, motion)  # [B,3,288,512]

        return torch.sigmoid(enhanced)


def gaussian_heatmap(size, center, sigma=5):
    """Generate 2D Gaussian heatmap"""
    H, W = size
    x, y = center
    X, Y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    return torch.exp(-((X.float() - x) ** 2 + (Y.float() - y) ** 2) / (2 * sigma ** 2))


if __name__ == "__main__":
    model = TrackNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test
    x = torch.randn(2, 9, 288, 512)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    print(f"Range: [{y.min():.3f}, {y.max():.3f}]")
    print("TrackNet ready!")
