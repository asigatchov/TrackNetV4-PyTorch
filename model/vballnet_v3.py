import torch
import torch.nn as nn



class Conv2DBlock(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super(Conv2DBlock, self).__init__(**kwargs)
        self.conv = nn.Conv2d(
            in_dim, out_dim, kernel_size=3, padding=1, bias=False
        )  # padding=1 for 3x3 kernel
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Double2DConv(nn.Module):
    """ Conv2DBlock x 2 """
    def __init__(self, in_dim, out_dim):
        super(Double2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

class Triple2DConv(nn.Module):
    """ Conv2DBlock x 3 """
    def __init__(self, in_dim, out_dim):
        super(Triple2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)
        self.conv_3 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x

class Single2DConv(nn.Module):
    """ Conv2DBlock x 1 (for Nano) """
    def __init__(self, in_dim, out_dim):
        super(Single2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        return x

class VballNetV3(nn.Module):
    def __init__(self, in_dim, out_dim, height = 288, width = 512):
        super(VballNetV3, self).__init__()
        self.down_block_1 = Double2DConv(in_dim, 32)  # Уменьшено с 64 до 32
        self.down_block_2 = Double2DConv(32, 64)      # Уменьшено с 128 до 64
        self.down_block_3 = Double2DConv(64, 128)     # Triple2DConv -> Double2DConv, 256 -> 128
        self.bottleneck = Double2DConv(128, 256)      # Triple2DConv -> Double2DConv, 512 -> 256
        self.up_block_1 = Double2DConv(384, 128)      # 768 -> 384 (128+256), Triple -> Double
        self.up_block_2 = Double2DConv(192, 64)       # 384 -> 192 (64+128)
        self.up_block_3 = Double2DConv(96, 32)        # 192 -> 96 (32+64), 64 -> 32
        self.predictor = nn.Conv2d(32, out_dim, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.down_block_1(x)                                       # (N,   32,  288,   512)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x1)                     # (N,   32,  144,   256)
        x2 = self.down_block_2(x)                                       # (N,   64,  144,   256)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x2)                     # (N,   64,   72,   128)
        x3 = self.down_block_3(x)                                       # (N,  128,   72,   128)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x3)                     # (N,  128,   36,    64)
        x = self.bottleneck(x)                                          # (N,  256,   36,    64)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x3], dim=1)      # (N,  384,   72,   128)
        x = self.up_block_1(x)                                          # (N,  128,   72,   128)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x2], dim=1)      # (N,  192,  144,   256)
        x = self.up_block_2(x)                                          # (N,   64,  144,   256)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x1], dim=1)      # (N,   96,  288,   512)
        x = self.up_block_3(x)                                          # (N,   32,  288,   512)
        x = self.predictor(x)                                           # (N,    3,  288,   512)
        x = self.sigmoid(x)                                             # (N,    3,  288,   512)
        return x




if __name__ == "__main__":
    # Инициализация модели
    height, width = 288, 512
    in_dim, out_dim = 9, 9
    model = VballNetV3(height=height, width=width, in_dim=in_dim, out_dim=out_dim)

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
