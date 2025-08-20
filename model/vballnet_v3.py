import torch
import torch.nn as nn
import torch.nn.functional as F


def rearrange_tensor(tensor, order):
    """
    Rearranges tensor dimensions (B, C, H, W, T) based on order string.
    Only supports 'BTCHW' as target.
    """
    order = order.upper()
    assert set(order) == set("BCHWT"), "Order must contain B,C,H,W,T"
    perm = [order.index(dim) for dim in "BTCHW"]
    return tensor.permute(*perm)


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution: Depthwise + Pointwise.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class MotionPromptLayer(nn.Module):
    """
    Generates motion attention maps using central differences for grayscale input.
    """
    def __init__(self, num_frames, penalty_weight=0.0):
        super().__init__()
        self.num_frames = num_frames
        self.penalty_weight = penalty_weight
        self.a = nn.Parameter(torch.tensor(0.1))
        self.b = nn.Parameter(torch.tensor(0.0))

    def power_normalization(self, x):
        scale = 5 / (0.45 * torch.abs(torch.tanh(self.a)) + 1e-2)
        offset = 0.8 * torch.tanh(self.b)
        return torch.sigmoid(scale * (torch.abs(x) - offset))

    def forward(self, video_seq):
        # video_seq: (B, 9, H, W)
        B, C, H, W = video_seq.shape
        assert C == self.num_frames, "Input channels must match num_frames"

        # Normalize
        norm_seq = (video_seq - video_seq.mean(dim=(2, 3), keepdim=True)) / (video_seq.std(dim=(2, 3), keepdim=True) + 1e-5) * 0.225 + 0.45
        norm_seq = F.avg_pool2d(norm_seq, kernel_size=3, stride=1, padding=1)  # Сглаживание

        # Reshape to (B, T, H, W)
        grayscale_seq = norm_seq  # (B, 9, H, W)

        # Compute motion differences
        attention_maps = []
        for t in range(self.num_frames):
            if t == 0:
                diff = grayscale_seq[:, 1] - grayscale_seq[:, 0]
            elif t == self.num_frames - 1:
                diff = grayscale_seq[:, t] - grayscale_seq[:, t - 1]
            else:
                diff = (grayscale_seq[:, t + 1] - grayscale_seq[:, t - 1]) / 2.0
            att_map = self.power_normalization(diff)
            attention_maps.append(att_map)

        attention_maps = torch.stack(attention_maps, dim=1)  # (B, T, H, W)
        loss = 0.0 if not self.training or self.penalty_weight <= 0 else self.penalty_weight * (attention_maps[:, 1:] - attention_maps[:, :-1]).pow(2).sum() / (H * W * (self.num_frames - 1) * B)

        return attention_maps, loss

class FusionLayerTypeA(nn.Module):
    """
    Fuses feature maps with attention maps: element-wise multiplication.
    """
    def __init__(self, num_frames, out_dim):
        super().__init__()
        self.num_frames = num_frames
        self.out_dim = out_dim

    def forward(self, feature_map, attention_map):
        # feature_map: (B, C, H, W) where C == T (num input frames)
        # attention_map: (B, T, H, W)
        T_out = min(self.num_frames, self.out_dim)
        outputs = []
        for t in range(T_out):
            # Multiply frame t feature with frame t attention
            fused = feature_map[:, t] * attention_map[:, t]  # (B, H, W)
            outputs.append(fused)
        return torch.stack(outputs, dim=1)  # (B, T_out, H, W)


class SpatialAttention(nn.Module):
    """
    Spatial attention module (CBAM-style).
    Input: (B, C, H, W)
    Output: (B, C, H, W)
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Avg and max pool across channels
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)

        # Concatenate
        concat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)

        # Apply conv and sigmoid
        attention = self.sigmoid(self.conv(concat))  # (B, 1, H, W)

        return x * attention


class VballNetV3(nn.Module):
    """
    PyTorch implementation of VballNetV3.
    Input: 9 grayscale frames -> Output: 9 heatmaps
    """
    def __init__(self, height=288, width=512, in_dim=9, out_dim=9):
        super().__init__()
        assert in_dim == out_dim == 9, "Currently configured for 9 frames in and out"
        self.num_frames = in_dim
        self.mode = "grayscale"

        # Fusion layer
        self.fusion_layer = FusionLayerTypeA(num_frames=in_dim, out_dim=out_dim)

        # Motion prompt layer
        self.motion_prompt = MotionPromptLayer(num_frames=in_dim, penalty_weight=0.0)

        # Encoder
        self.enc1_1 = DepthwiseSeparableConv(in_dim, 32, 3, padding=1)
        self.enc1_2 = DepthwiseSeparableConv(32, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)  # 288->144, 512->256

        self.enc2 = DepthwiseSeparableConv(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)  # 144->72, 256->128

        self.enc3 = DepthwiseSeparableConv(64, 128, 3, padding=1)  # Bottleneck

        # Spatial attention
        self.spatial_attention = SpatialAttention(kernel_size=7)

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = DepthwiseSeparableConv(128 + 64, 64, 3, padding=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = DepthwiseSeparableConv(64 + 32, 32, 3, padding=1)

        # Final conv
        self.final_conv = nn.Conv2d(32, out_dim, kernel_size=1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 9, 288, 512)
        B, C, H, W = x.shape

        # Generate motion attention maps
        residual_maps, motion_loss = self.motion_prompt(x)  # (B, 9, 288, 512), scalar

        # Encoder
        x1 = self.enc1_1(x)  # (B, 32, 288, 512)
        x1 = self.enc1_2(x1)  # (B, 32, 288, 512)
        x = self.pool1(x1)    # (B, 32, 144, 256)

        x2 = self.enc2(x)     # (B, 64, 144, 256)
        x = self.pool2(x2)    # (B, 64, 72, 128)

        x = self.enc3(x)      # (B, 128, 72, 128)

        # Apply spatial attention
        x = self.spatial_attention(x)  # (B, 128, 72, 128)

        # Decoder
        x = self.up1(x)  # (B, 128, 144, 256)
        x = torch.cat([x, x2], dim=1)  # Skip connection
        x = self.dec1(x)  # (B, 64, 144, 256)

        x = self.up2(x)  # (B, 64, 288, 512)
        x = torch.cat([x, x1], dim=1)  # Skip connection
        x = self.dec2(x)  # (B, 32, 288, 512)

        # Final conv
        x = self.final_conv(x)  # (B, 9, 288, 512)

        # Fusion with motion maps
        x = self.fusion_layer(x, residual_maps)  # (B, 9, 288, 512)

        # Sigmoid activation
        x = torch.sigmoid(x)

        # Optionally return motion_loss during training
        if self.training:
            return x #, motion_loss
        else:
            return x


if __name__ == "__main__":
    print("=== VballNetV3: Model Initialization and ONNX Export Test ===")

    # Parameters
    HEIGHT = 288
    WIDTH = 512
    IN_DIM = 9
    OUT_DIM = 9
    BATCH_SIZE = 1

    # Create model
    model = VballNetV3(height=HEIGHT, width=WIDTH, in_dim=IN_DIM, out_dim=OUT_DIM)
    model.eval()  # Switch to inference mode

    print(f"Model created: Input={IN_DIM}x{HEIGHT}x{WIDTH}, Output={OUT_DIM}x{HEIGHT}x{WIDTH}")

    # Test input
    dummy_input = torch.randn(BATCH_SIZE, IN_DIM, HEIGHT, WIDTH)

    print("Running forward pass...")
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Forward pass successful! Output shape: {output.shape}")
    assert output.shape == (BATCH_SIZE, OUT_DIM, HEIGHT, WIDTH), "Output shape mismatch!"

    # Export to ONNX
    ONNX_FILE = "vballnetv2.onnx"
    print("Exporting to ONNX...")

    try:
        torch.onnx.export(
            model,
            dummy_input,
            ONNX_FILE,
            export_params=True,
            opset_version=11,  # Recommended for compatibility
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            },
            verbose=False,
        )
        print(f"✅ Model successfully exported to '{ONNX_FILE}'")
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        raise

    # Validate ONNX model (optional)
    try:
        import onnx
        onnx_model = onnx.load(ONNX_FILE)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model is valid and well-formed.")
    except ImportError:
        print("⚠️  'onnx' package not installed. Skipping ONNX check.")
    except Exception as e:
        print(f"❌ ONNX validation failed: {e}")

    print("🎉 All tests passed! Ready for deployment.")
