import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --- Утилиты ---
def get_center_of_mass(heatmap):
    """
    Вычисляет центр массы по тепловой карте.
    heatmap: (H, W) or (B, H, W)
    Возвращает: (x, y) координаты в пикселях
    """
    if heatmap.dim() == 3:
        B, H, W = heatmap.shape
        xx, yy = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
        xx = xx.float().to(heatmap.device)
        yy = yy.float().to(heatmap.device)
        coords = []
        for b in range(B):
            mass = heatmap[b]
            total_mass = mass.sum()
            if total_mass > 1e-6:
                cx = (xx * mass).sum() / total_mass
                cy = (yy * mass).sum() / total_mass
            else:
                cx = cy = -1  # мяч не найден
            coords.append([cx, cy])
        return torch.tensor(coords, device=heatmap.device)  # (B, 2)
    else:
        H, W = heatmap.shape
        xx, yy = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
        xx, yy = xx.float().to(heatmap.device), yy.float().to(heatmap.device)
        mass = heatmap
        total_mass = mass.sum()
        if total_mass > 1e-6:
            cx = (xx * mass).sum() / total_mass
            cy = (yy * mass).sum() / total_mass
            return torch.tensor([cx, cy], device=heatmap.device)
        return torch.tensor([-1., -1.], device=heatmap.device)


# --- Функция для создания гауссова ядра ---
def gaussian_kernel(kernel_size=5, sigma=1.0, device='cpu'):
    """
    Создает 2D гауссово ядро для свертки.
    kernel_size: размер ядра (нечетное число)
    sigma: стандартное отклонение гауссианы
    device: устройство для тензора
    Возвращает: тензор (1, 1, kernel_size, kernel_size)
    """
    x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device)
    x = x.repeat(kernel_size, 1)
    y = x.t()
    gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    return gaussian.view(1, 1, kernel_size, kernel_size)

# --- Enhanced MotionPrompt (векторизованная, ONNX-совместимая) ---
class EnhancedMotionPrompt2(nn.Module):
    def __init__(self, num_frames, kernel_size=5, sigma=1.0):
        super().__init__()
        self.num_frames = num_frames
        self.kernel_size = kernel_size
        self.sigma = sigma
        
        # Создаём гауссово ядро и сохраняем как буфер
        self.register_buffer('gaussian_kernel', gaussian_kernel(kernel_size, sigma))
        
        # Параметры для адаптивной активации
        self.a = nn.Parameter(torch.tensor(1.0))  # масштаб
        self.b = nn.Parameter(torch.tensor(0.0))  # смещение

    def forward(self, video_seq):
        # video_seq: (B, T, H, W)
        B, T, H, W = video_seq.shape

        # Применяем гауссово размытие ко всем кадрам
        blurred = F.conv2d(
            video_seq.view(B * T, 1, H, W),
            self.gaussian_kernel,
            padding=self.kernel_size // 2
        ).view(B, T, H, W)

        # --- Векторизованное вычисление разностей ---
        # Центральные разности: (blurred[:, t+1] - blurred[:, t-1]) / 2
        if T == 1:
            # Крайний случай: один кадр
            diff = torch.zeros_like(blurred)
        else:
            # Сдвиг вперёд: t+1 (для t=0..T-2), последний кадр дублируется
            next_frames = torch.cat([blurred[:, 1:], blurred[:, -1:]], dim=1)  # (B, T, H, W)
            # Сдвиг назад: t-1 (для t=1..T-1), первый кадр дублируется
            prev_frames = torch.cat([blurred[:, :1], blurred[:, :-1]], dim=1)  # (B, T, H, W)
            # Центральная разность
            diff = 0.5 * (next_frames - prev_frames)

        # Модуль разности
        mag = torch.abs(diff)

        # Адаптивная сигмоида: усиливает значимые изменения
        motion_maps = torch.sigmoid(self.a * (mag - self.b))

        return motion_maps, None  # (B, T, H, W)


# --- Enhanced MotionPrompt (ONNX-совместимая версия) ---
class EnhancedMotionPrompt(nn.Module):
    def __init__(self, num_frames, kernel_size=5, sigma=1.0):
        super().__init__()
        self.num_frames = num_frames
        self.kernel_size = kernel_size
        self.sigma = sigma
        # Регистрируем гауссово ядро как буфер
        self.register_buffer('gaussian_kernel', gaussian_kernel(kernel_size, sigma))
        # Обучаемые параметры a и b для сигмоиды
        self.a = nn.Parameter(torch.tensor(1.0))  # масштабирующий коэффициент
        self.b = nn.Parameter(torch.tensor(0.0))  # смещение

    def forward(self, video_seq):
        # video_seq: (B, T, H, W)
        B, T, H, W = video_seq.shape
        # Применяем гауссову свертку
        blurred = F.conv2d(
            video_seq.view(B * T, 1, H, W),
            self.gaussian_kernel,
            padding=self.kernel_size // 2
        ).view(B, T, H, W)

        # Вычисляем разницу между кадрами
        motion_maps = []
        for t in range(T):
            # Для первого и последнего кадра используем соседние кадры
            prev_idx = torch.clamp(torch.tensor(t - 1, device=video_seq.device), 0, T - 1)
            next_idx = torch.clamp(torch.tensor(t + 1, device=video_seq.device), 0, T - 1)
            diff = 0.5 * (blurred[:, next_idx] - blurred[:, prev_idx])
            mag = torch.abs(diff)
            motion_map = torch.sigmoid(self.a * (mag - self.b))
            motion_maps.append(motion_map)
        return torch.stack(motion_maps, dim=1), None  # (B, T, H, W)


# --- Fusion Layer Type B ---
class FusionLayerTypeB(nn.Module):
    def __init__(self, num_frames, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(out_dim * 2, out_dim, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, feature_map, attention_map):
        # feature_map: (B, T, H, W), attention_map: (B, T, H, W)
        x = torch.cat([feature_map, attention_map], dim=1)  # (B, 2T, H, W)
        x = self.conv(x)
        x = self.norm(x)
        return F.relu(x)


# --- ASPP (Atrous Spatial Pyramid Pooling) ---
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


# --- VballNetV2 with Deep Supervision ---
class VballNetV2(nn.Module):
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
        self.supervision1 = nn.Conv2d(64, out_dim, kernel_size=1)  # выход 1

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = self._conv_block(64 + 32, 32)
        self.supervision2 = nn.Conv2d(32, out_dim, kernel_size=1)  # выход 2

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
        # x: (B, 9, H, W)
        B, T, H, W = x.shape
        assert H == self.height and W == self.width, f"Input size must be ({self.height}, {self.width}), got ({H}, {W})"
        assert T == self.in_dim, f"Expected {self.in_dim} frames, got {T}"

        # Motion attention
        motion_maps, _ = self.motion_prompt(x)  # (B, T, H, W)

        # Encoder
        x1 = self.enc1(x)  # (B, 32, H, W)
        x = self.pool1(x1)

        x2 = self.enc2(x)  # (B, 64, H//2, W//2)
        x = self.pool2(x2)

        x = self.enc3(x)  # (B, 128, H//4, W//4)
        x = self.aspp(x)

        # Decoder с deep supervision
        x = self.up1(x)  # (B, 128, H//2, W//2)
        x = torch.cat([x, x2], dim=1)
        x = self.dec1(x)
        out1 = self.supervision1(x)  # (B, 9, H//2, W//2)
        out1_up = F.interpolate(out1, size=(H, W), mode='bilinear', align_corners=False)

        x = self.up2(x)  # (B, 64, H, W)
        x = torch.cat([x, x1], dim=1)
        x = self.dec2(x)
        out2 = self.supervision2(x)  # (B, 9, H, W)

        # Final output
        final = self.final_conv(x)  # (B, 9, H, W)

        # Финальная фьюзия движения
        final = self.fusion_layer(final, motion_maps)
        out2 = self.fusion_layer(out2, motion_maps)

        # Deep supervision: суммируем все выходы
        fused_output = final + out2 + out1_up
        output = torch.sigmoid(fused_output)

        return output  # (B, 9, H, W)

    def predict_centers(self, x, threshold=0.3):
        """
        Предсказать центры мяча для всех 9 кадров.
        Возвращает: (B, 9, 2) — (x, y) координаты или (-1, -1) если нет мяча.
        """
        with torch.no_grad():
            heatmaps = self.forward(x)  # (B, 9, H, W)
            centers = []
            for b in range(heatmaps.shape[0]):
                batch_centers = []
                for t in range(9):
                    hmap = heatmaps[b, t]
                    if hmap.max() > threshold:
                        center = get_center_of_mass(hmap)
                    else:
                        center = torch.tensor([-1.0, -1.0], device=hmap.device)
                    batch_centers.append(center)
                centers.append(torch.stack(batch_centers))
            return torch.stack(centers)  # (B, 9, 2)


if __name__ == "__main__":
    # Инициализация модели
    height, width = 288, 512
    in_dim, out_dim = 9, 9
    model = VballNetV2(height=height, width=width, in_dim=in_dim, out_dim=out_dim)

    # Перевод модели в режим оценки и на CPU (для ONNX)
    model.eval()
    model.to('cpu')

    # Создание примера входных данных
    dummy_input = torch.randn(1, in_dim, height, width)

    # Проверка модели на примере
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")  # Ожидается: torch.Size([1, 9, 288, 512])

    # Экспорт модели в ONNX
    onnx_path = "vballnet_v2.onnx"
    try:
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
        print(f"Model successfully exported to {onnx_path}")
    except Exception as e:
        print(f"Failed to export model to ONNX: {e}")
