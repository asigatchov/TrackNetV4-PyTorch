import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur


# --- Утилиты ---
def get_center_of_mass(heatmap):
    """
    Вычисляет центр массы по тепловой карте.
    heatmap: (H, W) or (B, H, W)
    Возвращает: (x, y) координаты в пикселях
    """
    if heatmap.dim() == 3:
        B, H, W = heatmap.shape
        xx = torch.arange(W, device=heatmap.device).view(1, 1, -1).expand(B, H, -1)
        yy = torch.arange(H, device=heatmap.device).view(1, -1, 1).expand(B, -1, W)
        coords = []
        for b in range(B):
            mass = heatmap[b]
            total_mass = mass.sum()
            if total_mass > 1e-6:
                cx = (xx[b] * mass).sum() / total_mass
                cy = (yy[b] * mass).sum() / total_mass
            else:
                cx = cy = -1.0
            coords.append([cx, cy])
        return torch.tensor(coords, device=heatmap.device)
    else:
        H, W = heatmap.shape
        xx = torch.arange(W, device=heatmap.device).view(1, -1).expand(H, -1)
        yy = torch.arange(H, device=heatmap.device).view(-1, 1).expand(-1, W)
        mass = heatmap
        total_mass = mass.sum()
        if total_mass > 1e-6:
            cx = (xx * mass).sum() / total_mass
            cy = (yy * mass).sum() / total_mass
            return torch.tensor([cx, cy], device=heatmap.device)
        return torch.tensor([-1., -1.], device=heatmap.device)


# --- Создание гауссова ядра ---
def gaussian_kernel(kernel_size=5, sigma=1.0, device='cpu'):
    x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device)
    x = x.repeat(kernel_size, 1)
    y = x.t()
    gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    return gaussian.view(1, 1, kernel_size, kernel_size)


# --- Enhanced MotionPrompt (векторизованная) ---
class EnhancedMotionPrompt(nn.Module):
    def __init__(self, num_frames, kernel_size=5, sigma=1.0):
        super().__init__()
        self.num_frames = num_frames
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.register_buffer('gaussian_kernel', gaussian_kernel(kernel_size, sigma))
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, video_seq):
        B, T, H, W = video_seq.shape
        # Размытие
        blurred = F.conv2d(
            video_seq.view(B * T, 1, H, W),
            self.gaussian_kernel,
            padding=self.kernel_size // 2
        ).view(B, T, H, W)

        # Центральные разности для всех кадров
        if T == 1:
            diff = torch.zeros_like(blurred)
        else:
            next_frames = torch.cat([blurred[:, 1:], blurred[:, -1:]], dim=1)
            prev_frames = torch.cat([blurred[:, :1], blurred[:, :-1]], dim=1)
            diff = 0.5 * (next_frames - prev_frames)

        mag = torch.abs(diff)
        motion_map = torch.sigmoid(self.a * (mag - self.b))
        # Фильтр: только если изменение > порог
        motion_map = motion_map * (mag > 0.1)

        return motion_map, None


# --- Fusion Layer Type B ---
class FusionLayerTypeB(nn.Module):
    def __init__(self, num_frames, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(out_dim * 2, out_dim, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, feature_map, attention_map):
        x = torch.cat([feature_map, attention_map], dim=1)
        x = self.conv(x)
        x = self.norm(x)
        return F.relu(x)


# --- ASPP ---
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3_6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.conv3x3_12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.conv3x3_18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_pool = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.final = nn.Conv2d(5 * out_channels, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        h, w = x.shape[2:]
        features1 = self.conv1x1(x)
        features2 = self.conv3x3_6(x)
        features3 = self.conv3x3_12(x)
        features4 = self.conv3x3_18(x)
        pooled = self.global_pool(x)
        pooled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
        pooled = self.conv1x1_pool(pooled)
        out = torch.cat([features1, features2, features3, features4, pooled], dim=1)
        out = self.final(out)
        out = self.norm(out)
        return F.relu(out)


# --- VballNetV3 with Deep Supervision and Temporal Consistency ---
class VballNetV3(nn.Module):
    def __init__(self, height=288, width=512, in_dim=9, out_dim=9):
        super().__init__()
        self.height = height
        self.width = width
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Motion Prompt
        self.motion_prompt = EnhancedMotionPrompt(num_frames=in_dim)

        # Fusion Layer
        self.fusion_layer = FusionLayerTypeB(num_frames=in_dim, out_dim=out_dim)

        # Encoder
        self.enc1 = self._conv_block(in_dim, 32)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.enc2 = self._conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.enc3 = self._conv_block(64, 128)

        # ASPP
        self.aspp = ASPP(128, 128)

        # Decoder с deep supervision
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = self._conv_block(128 + 64, 64)
        self.supervision1 = nn.Conv2d(64, out_dim, kernel_size=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = self._conv_block(64 + 32, 32)
        self.supervision2 = nn.Conv2d(32, out_dim, kernel_size=1)

        # Final output
        self.final_conv = nn.Conv2d(32, out_dim, kernel_size=1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        B, T, H, W = x.shape
        assert H == self.height and W == self.width, f"Input size must be ({self.height}, {self.width})"
        assert T == self.in_dim, f"Expected {self.in_dim} frames, got {T}"

        # Motion attention
        motion_maps, _ = self.motion_prompt(x)

        # Encoder
        x1 = self.enc1(x)
        x = self.pool1(x1)

        x2 = self.enc2(x)
        x = self.pool2(x2)

        x = self.enc3(x)
        x = self.aspp(x)

        # Decoder с deep supervision
        x = self.up1(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec1(x)
        out1 = self.supervision1(x)
        out1_up = F.interpolate(out1, size=(H, W), mode='bilinear', align_corners=False)

        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec2(x)
        out2 = self.supervision2(x)

        # Final output
        final = self.final_conv(x)

        # Фьюзия движения
        final = self.fusion_layer(final, motion_maps)
        out2 = self.fusion_layer(out2, motion_maps)

        # Deep supervision
        fused_output = final + out2 + out1_up
        output = torch.sigmoid(fused_output)

        return output  # (B, 9, H, W)

    def predict_centers(self, x, threshold=0.3, area_min=10):
        with torch.no_grad():
            heatmaps = self.forward(x)
            centers = []
            for b in range(heatmaps.shape[0]):
                batch_centers = []
                for t in range(9):
                    hmap = heatmaps[b, t]
                    if hmap.max() > threshold and (hmap > threshold).sum() > area_min:
                        center = get_center_of_mass(hmap)
                    else:
                        center = torch.tensor([-1., -1.], device=hmap.device)
                    batch_centers.append(center)
                centers.append(torch.stack(batch_centers))
            return torch.stack(centers)


# --- Post-processing ---
def postprocess_heatmap(heatmap, kernel_size=5, sigma=1.0):
    B, T, H, W = heatmap.shape
    kernel = gaussian_kernel(kernel_size, sigma, device=heatmap.device)
    blurred = F.conv2d(
        heatmap.view(B * T, 1, H, W),
        kernel,
        padding=kernel_size // 2
    ).view(B, T, H, W)
    return blurred


def smooth_temporal(heatmaps, window=3):
    B, T, H, W = heatmaps.shape
    smoothed = torch.zeros_like(heatmaps)
    for t in range(T):
        start = max(0, t - window//2)
        end = min(T, t + window//2 + 1)
        window_maps = heatmaps[:, start:end]
        smoothed[:, t] = window_maps.mean(dim=1)
    return smoothed


def non_max_suppression(heatmap, kernel_size=3, threshold=0.5):
    pooled = F.max_pool2d(heatmap, kernel_size, stride=1, padding=kernel_size//2)
    mask = (heatmap == pooled).float()
    return heatmap * mask * (heatmap > threshold)


# --- Loss ---
class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
        bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        pt = torch.exp(-bce)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = self.alpha * focal_weight * bce
        intersection = (pred * target).sum(dim=[2, 3])
        dice_loss = 1 - (2. * intersection + self.smooth) / (
            pred.sum(dim=[2, 3]) + target.sum(dim=[2, 3]) + self.smooth)
        return focal_loss.mean() + dice_loss.mean()


def temporal_consistency_loss(preds, alpha=0.1):
    B, T, H, W = preds.shape
    if T < 2:
        return 0.0
    diff = preds[:, 1:] - preds[:, :-1]
    loss = torch.sum(diff ** 2) / (H * W * (T - 1) * B)
    return alpha * loss


# --- Инициализация и конвертация ---
if __name__ == "__main__":
    # Параметры
    height, width = 288, 512
    in_dim, out_dim = 9, 9

    # Инициализация модели
    model = VballNetV3(height=height, width=width, in_dim=in_dim, out_dim=out_dim)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Компиляция (для скорости)
    try:
        compiled_model = torch.compile(model, mode="reduce-overhead")
        print("✓ Model compiled successfully.")
    except Exception as e:
        print(f"⚠️ Compilation failed: {e}")
        compiled_model = model

    # Демонстрация вывода
    dummy_input = torch.randn(1, in_dim, height, width)
    with torch.no_grad():
        output = compiled_model(dummy_input)
        print(f"Output shape: {output.shape}")

    # Post-processing
    processed = postprocess_heatmap(output)
    processed = smooth_temporal(processed)
    processed = non_max_suppression(processed, threshold=0.3)

    # Predict centers
    centers = compiled_model.predict_centers(dummy_input, threshold=0.3, area_min=10)
    print(f"Centers shape: {centers.shape}")  # (1, 9, 2)

    # Экспорт в ONNX
    onnx_path = "vballnet_v3.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        verbose=False
    )
    print(f"✓ Model exported to {onnx_path}")
