import torch
import torch.nn as nn
import torch.nn.functional as F

# Utility functions
def rearrange_tensor(input_tensor, order):
    """
    Rearranges the dimensions of a tensor according to the specified order.
    """
    order = order.upper()
    assert len(set(order)) == 5, "Order must be a 5 unique character string"
    assert all(dim in order for dim in "BCHWT"), "Order must contain all of BCHWT"
    perm = [order.index(dim) for dim in "BTCHW"]
    return input_tensor.permute(*perm)

def power_normalization(input, a, b):
    """
    Power normalization function for attention map generation.
    Vectorized to ensure ONNX compatibility.
    """
    return 1 / (1 + torch.exp(-(5 / (0.45 * torch.abs(torch.tanh(a)) + 1e-1)) * (torch.abs(input) - 0.6 * torch.tanh(b))))

# MotionPrompt Module
class MotionPrompt(nn.Module):
    """
    A module for generating attention maps from video sequences with GRU for temporal modeling.
    Uses central differences for motion detection and GRU to process temporal dependencies.
    Supports grayscale (N frames) and RGB (N×3 channels) modes.
    Optimized for ONNX compatibility and performance.
    """
    def __init__(self, num_frames, mode="grayscale", penalty_weight=0.0):
        super().__init__()
        self.num_frames = num_frames
        self.mode = mode.lower()
        assert self.mode in ["rgb", "grayscale"], "Mode must be 'rgb' or 'grayscale'"
        self.input_permutation = "BTCHW"
        self.input_color_order = "RGB" if self.mode == "rgb" else None
        self.color_map = {"R": 0, "G": 1, "B": 2}
        self.gray_scale = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)
        self.a = nn.Parameter(torch.tensor(0.1))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.lambda1 = penalty_weight

        # GRU для обработки последовательности разностей кадров
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )

        # Слои для подготовки входа и выхода GRU
        self.pool = nn.MaxPool2d(kernel_size=16, stride=16)  # Уменьшает 288x512 до 18x32
        self.pooled_height, self.pooled_width = 288 // 16, 512 // 16  # 18, 32
        self.reduced_channels = 1  # Фиксированное количество каналов для conv_reduce
        self.conv_reduce = nn.Conv2d(1, self.reduced_channels, kernel_size=1)
        self.linear_reduce = nn.Linear(self.reduced_channels * self.pooled_height * self.pooled_width, 256)
        self.linear_expand = nn.Linear(256, self.reduced_channels * self.pooled_height * self.pooled_width)
        self.conv_expand = nn.Conv2d(self.reduced_channels, 1, kernel_size=1)
        self.upsample = nn.Upsample(size=(288, 512), mode='bilinear', align_corners=False)

    def forward(self, video_seq):
        loss = torch.tensor(0.0, device=video_seq.device)
        video_seq = rearrange_tensor(video_seq, self.input_permutation)
        norm_seq = video_seq * 0.225 + 0.45

        if self.mode == "rgb":
            idx_list = [self.color_map[idx] for idx in self.input_color_order]
            weights = self.gray_scale[idx_list].to(video_seq.device)
            grayscale_video_seq = torch.einsum("btcwh,c->btwh", norm_seq, weights)
        else:  # grayscale mode
            grayscale_video_seq = norm_seq[:, :, 0, :, :]  # Single channel per frame
#            grayscale_video_seq = video_seq[:, :, 0, :, :]  # Single channel per frame
        # Compute central differences for frames t=0 to t=num_frames-1
        frame_diffs = []
        for t in range(self.num_frames):
            if t == 0:
                # Forward difference for first frame
                frame_diff = grayscale_video_seq[:, t + 1] - grayscale_video_seq[:, t]
            elif t == self.num_frames - 1:
                # Backward difference for last frame
                frame_diff = grayscale_video_seq[:, t] - grayscale_video_seq[:, t - 1]
            else:
                # Central difference for intermediate frames
                frame_diff = (grayscale_video_seq[:, t + 1] - grayscale_video_seq[:, t - 1]) / 2
            frame_diffs.append(frame_diff)

        frame_diffs = torch.stack(frame_diffs, dim=1)  # Shape: (batch, num_frames, height, width)

        # Подготовка для GRU
        batch_size, seq_len, height, width = frame_diffs.shape
        frame_diffs = frame_diffs.view(batch_size * seq_len, 1, height, width)  # (batch*seq_len, 1, 288, 512)
        pooled_diffs = self.pool(frame_diffs)  # (batch*seq_len, 1, 18, 32)
        reduced_diffs = self.conv_reduce(pooled_diffs)  # (batch*seq_len, reduced_channels, 18, 32)
        reduced_diffs = reduced_diffs.view(batch_size * seq_len, -1)  # (batch*seq_len, reduced_channels * 18 * 32)
        reduced_diffs = self.linear_reduce(reduced_diffs)  # (batch*seq_len, 256)
        reduced_diffs = reduced_diffs.view(batch_size, seq_len, 256)  # (batch, seq_len, 256)

        # Пропускаем через GRU
        gru_out, _ = self.gru(reduced_diffs)  # (batch, seq_len, 256)

        # Восстанавливаем формат для карт внимания
        gru_out = gru_out.reshape(batch_size * seq_len, 256)  # (batch*seq_len, 256)
        expanded_diffs = self.linear_expand(gru_out)  # (batch*seq_len, reduced_channels * 18 * 32)
        expanded_diffs = expanded_diffs.view(batch_size * seq_len, self.reduced_channels, self.pooled_height, self.pooled_width)  # (batch*seq_len, reduced_channels, 18, 32)
        expanded_diffs = self.conv_expand(expanded_diffs)  # (batch*seq_len, 1, 18, 32)
        expanded_diffs = self.upsample(expanded_diffs)  # (batch*seq_len, 1, 288, 512)
        expanded_diffs = expanded_diffs.view(batch_size, seq_len, height, width)  # (batch, seq_len, 288, 512)

        # Генерация карт внимания (векторизованная для ONNX)
        attention_map = power_normalization(expanded_diffs, self.a, self.b)  # Shape: (batch, num_frames, height, width)
        norm_attention = attention_map.unsqueeze(2)

        if self.training:
            B, T, H, W = grayscale_video_seq.shape
            temp_diff = norm_attention[:, 1:] - norm_attention[:, :-1]
            temporal_loss = torch.sum(temp_diff ** 2) / (H * W * (T - 1) * B)
            loss = self.lambda1 * temporal_loss

        return attention_map, loss

# FusionLayerTypeA Module
class FusionLayerTypeA(nn.Module):
    """
    A module that incorporates motion using attention maps - version 1.
    Applies attention map of current frame t to feature map of frame t.
    """
    def __init__(self, num_frames, out_dim):
        super().__init__()
        self.num_frames = num_frames
        self.out_dim = out_dim

    def forward(self, feature_map, attention_map):
        outputs = []
        for t in range(min(self.num_frames, self.out_dim)):
            outputs.append(feature_map[:, t, :, :] * attention_map[:, t, :, :])  # Use attention map of current frame
        return torch.stack(outputs, dim=1)

# VballNetV1 Model
class VballNetV1(nn.Module):
    """
    VballNetV1: Motion-enhanced U-Net for volleyball tracking.
    Supports Grayscale (N input frames, N output heatmaps) and RGB (N×3 input channels, N output heatmaps) modes.
    """
    def __init__(self, height=288, width=512, in_dim=9, out_dim=3, fusion_layer_type="TypeA"):
        super().__init__()
        assert fusion_layer_type == "TypeA", "Fusion layer must be 'TypeA'"
        mode = "grayscale" if in_dim == out_dim else "rgb"
        num_frames = in_dim if mode == "grayscale" else in_dim // 3

        # Fusion layer
        self.fusion_layer = FusionLayerTypeA(num_frames=num_frames, out_dim=out_dim)

        # Motion prompt
        self.motion_prompt = MotionPrompt(num_frames=num_frames, mode=mode)

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_dim, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.enc1_1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )

        # Output layer
        self.out_conv = nn.Conv2d(32, out_dim, kernel_size=1, padding=0)

    def forward(self, imgs_input):
        # Reshape input for motion prompt
        channels_per_frame = imgs_input.shape[1] // self.motion_prompt.num_frames
        motion_input = imgs_input.view(imgs_input.shape[0], self.motion_prompt.num_frames, channels_per_frame, imgs_input.shape[2], imgs_input.shape[3])

        # Motion prompt
        residual_maps, _ = self.motion_prompt(motion_input)

        # Encoder
        x1 = self.enc1(imgs_input)
        x1_1 = self.enc1_1(x1)
        x = self.pool1(x1_1)

        x2 = self.enc2(x)
        x = self.pool2(x2)

        x = self.enc3(x)

        # Decoder
        x = self.up1(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        x = torch.cat([x, x1_1], dim=1)
        x = self.dec2(x)

        # Output
        x = self.out_conv(x)
        x = self.fusion_layer(x, residual_maps)
        x = torch.sigmoid(x)

        return x

if __name__ == "__main__":
    # Model initialization and testing
    height, width, in_dim, out_dim = 288, 512, 9, 3
    model = VballNetV1(height, width, in_dim, out_dim)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"VballNetV1 initialized with {total_params:,} parameters")

    # Forward pass test
    test_input = torch.randn(2, in_dim, height, width)
    test_output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    print(f"Output range: [{test_output.min():.3f}, {test_output.max():.3f}]")
    print("✓ VballNetV1 ready for training!")
