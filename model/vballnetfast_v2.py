import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution с настраиваемым kernel_size."""
    def __init__(self, in_channels, out_channels, name, kernel_size=3, dropout_p=0.0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class Single2DConv(nn.Module):
    """Обертка для DepthwiseSeparableConv."""
    def __init__(self, in_channels, out_channels, name, kernel_size=3, dropout_p=0.0):
        super(Single2DConv, self).__init__()
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, name, kernel_size, dropout_p)
        
    def forward(self, x):
        return self.conv(x)

class VballNetFastV2(nn.Module):
    """
    Масштабируемая PyTorch-версия VballNet с GRU для учета траекторий между 9 grayscale кадрами.

    Args:
        input_height (int): Высота входного изображения.
        input_width (int): Ширина входного изображения.
        in_dim (int): Число входных каналов (по умолчанию 9 для grayscale кадров).
        out_dim (int): Число выходных тепловых карт (по умолчанию 9).
        channels (list): Список каналов для энкодера (декодер зеркально).
        bottleneck_channels (int): Число каналов в бутылочном горлышке.
        bottleneck_kernel_size (int): Размер ядра в бутылочном горлышке.
        gru_hidden_size (int): Размер скрытого состояния GRU.
        dropout_p (float): Вероятность Dropout.

    Returns:
        Model: PyTorch-модель VballNetFastV2 с GRU.
    """
    def __init__(self, input_height, input_width, in_dim=9, out_dim=9, 
                 channels=[8, 16, 32], bottleneck_channels=64, 
                 bottleneck_kernel_size=3, gru_hidden_size=64, dropout_p=0.2):
        super(VballNetFastV2, self).__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.channels = channels
        self.bottleneck_channels = bottleneck_channels
        self.gru_hidden_size = gru_hidden_size
        
        # Энкодер
        self.down_blocks = nn.ModuleList()
        current_channels = in_dim
        for i, out_channels in enumerate(channels):
            self.down_blocks.append(
                Single2DConv(current_channels, out_channels, f'down_block_{i+1}', kernel_size=3, dropout_p=0.0)
            )
            current_channels = out_channels
        
        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(len(channels))
        ])
        
        # Бутылочное горлышко
        self.bottleneck = Single2DConv(
            channels[-1], bottleneck_channels, 'bottleneck', 
            kernel_size=bottleneck_kernel_size, dropout_p=dropout_p
        )
        
        # GRU для обработки временных зависимостей
        self.gru = nn.GRU(
            input_size=bottleneck_channels,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True
        )
        # Переходной слой после GRU для восстановления каналов
        self.gru_transition = nn.Conv2d(
            gru_hidden_size, bottleneck_channels, kernel_size=1, bias=False
        )
        self.gru_bn = nn.BatchNorm2d(bottleneck_channels)
        self.gru_relu = nn.ReLU(inplace=True)
        
        # Декодер
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        decoder_channels = channels[::-1]  # [32, 16, 8]
        current_channels = bottleneck_channels
        
        for i, out_channels in enumerate(decoder_channels):
            in_channels = current_channels + channels[-(i+1)]
            self.upsamples.append(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            )
            self.up_blocks.append(
                Single2DConv(in_channels, out_channels, f'up_block_{i+1}', kernel_size=3, dropout_p=dropout_p)
            )
            current_channels = out_channels
        
        # Финальный предиктор: входные каналы = channels[0]
        self.predictor = nn.Conv2d(
            channels[0], out_dim, kernel_size=1
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        if x.size(2) != self.input_height or x.size(3) != self.input_width:
            x = F.interpolate(x, size=(self.input_height, self.input_width), 
                            mode='bilinear', align_corners=False)
        
        # Энкодер
        skip_connections = []
        for down_block, pool in zip(self.down_blocks, self.pools):
            x = down_block(x)
            skip_connections.append(x)
            x = pool(x)
        
        # Бутылочное горлышко
        x = self.bottleneck(x)  # (N, bottleneck_channels, H/16, W/16)
        
        # Подготовка для GRU: реорганизация в последовательность
        batch_size, channels, h, w = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, H/16, W/16, bottleneck_channels)
        x = x.view(batch_size, h * w, channels)  # (N, seq_len=H/16*W/16, bottleneck_channels)
        
        # GRU
        x, _ = self.gru(x)  # (N, seq_len, gru_hidden_size)
        
        # Восстановление пространственного формата
        x = x.view(batch_size, h, w, self.gru_hidden_size)  # (N, H/16, W/16, gru_hidden_size)
        x = x.permute(0, 3, 1, 2).contiguous()  # (N, gru_hidden_size, H/16, W/16)
        x = self.gru_transition(x)  # (N, bottleneck_channels, H/16, W/16)
        x = self.gru_bn(x)
        x = self.gru_relu(x)
        
        # Декодер
        for upsample, up_block, skip in zip(
            self.upsamples, self.up_blocks, skip_connections[::-1]
        ):
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)
            x = up_block(x)
        
        # Финальный предиктор
        x = self.predictor(x)
        x = self.sigmoid(x)
        
        return x

    def get_num_parameters(self):
        """Возвращает общее количество обучаемых параметров."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)