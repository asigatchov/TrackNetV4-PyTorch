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


# def power_normalization(input, scale, threshold):
#     """
#     Улучшенная power_normalization.
#     Преобразует разность кубитов в карту внимания через сигмоиду.

#     Args:
#         input: тензор разностей кадров
#         scale: крутизна сигмоиды (обучаемый параметр)
#         threshold: порог активации (обучаемый параметр)

#     Returns:
#         attention_map: значения в [0, 1], где 1 — сильное движение
#     """
#     return torch.sigmoid(scale * (torch.abs(input) - threshold))


def power_normalization(input, scale, threshold):
    """
    Улучшенная power_normalization.
    Преобразует разность кубитов в карту внимания через сигмоиду.

    Args:
        input: тензор разностей кадров
        scale: крутизна сигмоиды (обучаемый параметр)
        threshold: порог активации (обучаемый параметр)

    Returns:
        attention_map: значения в [0, 1], где 1 — сильное движение
    """
    return torch.sigmoid(scale * (torch.abs(input) - threshold))


class MotionPrompt(nn.Module):
    """
    Улучшенный модуль MotionPrompt с:
    - Stateful GRU для онлайн-трекинга.
    - Стабильной power_normalization через обучаемые scale и threshold.
    - Оптимизированной обработкой признаков (без избыточных conv при reduced_channels=1).
    - Поддержкой ONNX.
    """

    def __init__(
        self, num_frames, mode="grayscale", penalty_weight=0.0, gru_hidden_size=256
    ):
        super().__init__()
        self.num_frames = num_frames
        self.mode = mode.lower()
        assert self.mode in ["rgb", "grayscale"], "Mode must be 'rgb' or 'grayscale'"
        self.input_permutation = "BTCHW"
        self.input_color_order = "RGB" if self.mode == "rgb" else None
        self.color_map = {"R": 0, "G": 1, "B": 2}
        self.gray_scale = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)
        self.lambda1 = penalty_weight
        self.hidden_size = gru_hidden_size

        # --- Улучшенная нормализация ---
        self.scale = nn.Parameter(torch.tensor(5.0))  # Управляет крутизной
        self.threshold = nn.Parameter(torch.tensor(0.6))  # Управляет порогом

        # --- GRU для временной динамики ---
        self.gru = nn.GRU(
            input_size=gru_hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # Сохранение скрытого состояния между вызовами
        self.hidden_state = None  # (num_layers=1, batch, hidden_size)

        # --- Сжатие пространства ---
        self.pool = nn.MaxPool2d(kernel_size=16, stride=16)  # 288x512 -> 18x32
        self.pooled_height, self.pooled_width = 288 // 16, 512 // 16  # 18, 32
        self.feature_dim = self.pooled_height * self.pooled_width  # 576

        # Заменяем conv_reduce и conv_expand на прямое преобразование
        # Если reduced_channels == 1, conv1x1 избыточен — работаем напрямую с пуллингом
        self.linear_reduce = nn.Linear(self.feature_dim, gru_hidden_size)
        self.linear_expand = nn.Linear(gru_hidden_size, self.feature_dim)

        # --- Восстановление разрешения ---
        self.upsample = nn.Upsample(
            size=(288, 512), mode="bilinear", align_corners=False
        )

    def forward(self, video_seq):
        """
        Args:
            video_seq: Tensor of shape (B, C, H, W) with C = N*T (T=num_frames)

        Returns:
            attention_map: (B, T, H, W)
            loss: scalar (0 if not training)
        """
        device = video_seq.device
        loss = torch.tensor(0.0, device=device)

        # --- Перестановка и нормализация ---
        # Убедитесь, что вы используете правильный порядок!
        # Предполагаем, что вход: (B, C, H, W), где C = num_frames * channels_per_frame
        # Мы хотим (B, T, C_per_frame, H, W) → но у нас уже есть motion_input из VballNetV1
        # Однако в этом forward мы получаем video_seq уже в формате (B, T, C, H, W) из view()
        # Если нет — используйте rearrange_tensor, но в текущем контексте VballNetV1 всё передаётся верно.

        # В текущем вызове motion_input имеет форму (B, T, C, H, W)
        # norm_seq = video_seq * 0.225 + 0.45  # ImageNet denorm
        norm_seq = video_seq * 0.225 + 0.45  # Уже нормализован

        # --- Конвертация в grayscale ---
        if self.mode == "rgb":
            idx_list = [self.color_map[idx] for idx in self.input_color_order]
            weights = self.gray_scale[idx_list].to(device)
            grayscale_video_seq = torch.einsum("btcwh,c->btwh", norm_seq, weights)
        else:
            grayscale_video_seq = norm_seq[:, :, 0, :, :]  # B, T, H, W

        B, T, H, W = grayscale_video_seq.shape

        if self.hidden_state is not None:
            _, hidden_batch_size, _ = self.hidden_state.shape
            if hidden_batch_size != B:
                # Размер батча изменился — сбрасываем состояние
                self.hidden_state = None

        # --- Центральные разности ---
        frame_diffs = []
        for t in range(T):
            if t == 0:
                diff = grayscale_video_seq[:, 1] - grayscale_video_seq[:, 0]
            elif t == T - 1:
                diff = grayscale_video_seq[:, -1] - grayscale_video_seq[:, -2]
            else:
                diff = (grayscale_video_seq[:, t + 1] - grayscale_video_seq[:, t - 1]) / 2
            frame_diffs.append(diff)
        frame_diffs = torch.stack(frame_diffs, dim=1)  # (B, T, H, W)

        # --- Подготовка к GRU ---
        x = frame_diffs.reshape(B * T, 1, H, W)  # (B*T, 1, 288, 512)
        x = self.pool(x)  # (B*T, 1, 18, 32)
        x = x.reshape(B * T, -1)  # (B*T, 576)
        x = self.linear_reduce(x)  # (B*T, hidden_size)
        x = x.reshape(B, T, -1)  # (B, T, hidden_size)

        # --- GRU с сохранением состояния ---
        gru_out, hidden_state = self.gru(x, self.hidden_state)
        self.hidden_state = hidden_state.detach()  # Сохраняем для следующего кадра

        # --- Восстановление пространственной структуры ---
        # ОШИБКА БЫЛА ЗДЕСЬ: gru_out может быть не contiguous после GRU
        x = gru_out.reshape(B * T, -1)  # ✅ ИСПРАВЛЕНО: используем reshape вместо view
        x = self.linear_expand(x)  # (B*T, 576)
        x = x.reshape(B * T, 1, self.pooled_height, self.pooled_width)  # (B*T, 1, 18, 32)
        x = self.upsample(x)  # (B*T, 1, 288, 512)
        x = x.reshape(B, T, H, W)  # (B, T, 288, 512)

        # --- Улучшенная power_normalization ---
        attention_map = torch.sigmoid(self.scale * (torch.abs(x) - self.threshold))

        # --- Temporal loss (только при обучении) ---
        if self.training:
            norm_attention = attention_map.unsqueeze(2)
            temp_diff = norm_attention[:, 1:] - norm_attention[:, :-1]
            temporal_loss = temp_diff.pow(2).sum() / (H * W * (T - 1) * B)
            loss = self.lambda1 * temporal_loss

        return attention_map, loss

    def reset_hidden_state(self):
        """Сбросить состояние GRU (например, при начале новой видео-последовательности)."""
        self.hidden_state = None

    def set_hidden_state(self, hidden_state):
        """Установить состояние вручную (для контроля)."""
        self.hidden_state = hidden_state


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
            outputs.append(
                feature_map[:, t, :, :] * attention_map[:, t, :, :]
            )  # Use attention map of current frame
        return torch.stack(outputs, dim=1)


# VballNetV1 Model
class VballNetV1c(nn.Module):
    """
    VballNetV1: Motion-enhanced U-Net for volleyball tracking.
    Supports Grayscale (N input frames, N output heatmaps) and RGB (N×3 input channels, N output heatmaps) modes.
    """

    def __init__(
        self, height=288, width=512, in_dim=9, out_dim=3, fusion_layer_type="TypeA"
    ):
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
            nn.BatchNorm2d(32),
        )
        self.enc1_1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
        )
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
        )

        # Output layer
        self.out_conv = nn.Conv2d(32, out_dim, kernel_size=1, padding=0)

    def forward(self, imgs_input):
        # Reshape input for motion prompt
        channels_per_frame = imgs_input.shape[1] // self.motion_prompt.num_frames
        motion_input = imgs_input.view(
            imgs_input.shape[0],
            self.motion_prompt.num_frames,
            channels_per_frame,
            imgs_input.shape[2],
            imgs_input.shape[3],
        )

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
    model = VballNetV1c(height, width, in_dim, out_dim)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"VballNetV1 initialized with {total_params:,} parameters")

    # Forward pass test
    test_input = torch.randn(2, in_dim, height, width)
    test_output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    print(f"Output range: [{test_output.min():.3f}, {test_output.max():.3f}]")
    print("✓ VballNetV1 ready for training!")
