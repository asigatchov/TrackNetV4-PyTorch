import torch
import torch.nn as nn
import numpy as np
import time
import torch
import torch.nn as nn
import numpy as np
import time

class VballNetV1d(nn.Module):
    def __init__(self, height=288, width=512, in_dim=9, out_dim=9):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = 128  # Увеличено

        # Stem: (B*T, 1, 288, 512) → (B*T, 32, 144, 256)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 288→144, 512→256
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # ✅ Фиксированный AvgPool2d вместо AdaptiveAvgPool2d
        # Вход после stem: (144, 256)
        # Цель: (8, 8) → kernel = (144//8, 256//8) = (18, 32)
        self.spatial_pool = nn.AvgPool2d(kernel_size=(18, 32), stride=(18, 32))  # → (B*T, 32, 8, 8)

        # Сжатие
        self.feature_flatten = nn.Linear(32 * 8 * 8, self.hidden_size)  # 2048 → 128

        # Временная обработка
        self.temporal_conv = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1)
        self.temporal_act = nn.ReLU(inplace=True)

        # Декодер
        self.hidden_to_features = nn.Linear(self.hidden_size, 32 * 8 * 8)  # 128 → 2048
        self.feature_unflatten = nn.Unflatten(1, (32, 8, 8))

        # Апскейл
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  # 8 → 16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  # 16 → 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 32 → 64
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(288, 512), mode='nearest'),
        )

        # Skip-connection
        self.skip_conv = nn.Conv2d(24, 16, kernel_size=1)  # обучаемый

        # Финальный слой
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)
    def forward(self, x):
        B, T, H, W = x.shape

        if H != 288 or W != 512:
            raise ValueError(f"Input size must be (288, 512), got ({H}, {W})")

        x = x.view(B * T, 1, H, W)  # (B*T, 1, 288, 512)

        # Skip connection: сохраняем до пулинга (24 каналов, 288x512)
        x_stem = self.stem[:4](x)  # (B*T, 24, 288, 512)
        x = self.stem[4:](x_stem)   # (B*T, 32, 144, 256)

        # Пространственное сжатие
        x = self.spatial_pool(x)  # → (B*T, 32, 8, 8)
        x = x.view(B * T, -1)     # flatten: (B*T, 2048)
        x = self.feature_flatten(x)  # (B*T, 128)
        x = x.view(B, T, -1)      # (B, T, 128)

        # Временная обработка
        x = x.transpose(1, 2)  # (B, 128, T)
        x = self.temporal_conv(x)
        x = self.temporal_act(x)
        x = x.transpose(1, 2)  # (B, T, 128)

        # Распаковка
        x = x.reshape(B * T, -1)  # (B*T, 128)
        x = self.hidden_to_features(x)  # (B*T, 2048)
        x = self.feature_unflatten(x)  # (B*T, 32, 8, 8)
        x = self.upsample(x)  # (B*T, 16, 288, 512)

        # Skip-connection: 24 → 16 каналов (обучаемый слой)
        x_skip = self.skip_conv(x_stem)  # (B*T, 16, 288, 512)
        x = x + x_skip  # residual

        # Финальный слой
        x = self.final_conv(x)  # (B*T, 1, 288, 512)
        x = x.reshape(B, T, 288, 512)

        # Коррекция out_dim
        if self.out_dim > T:
            x = torch.cat([x, x[:, -1:].expand(B, self.out_dim - T, 288, 512)], dim=1)
        elif self.out_dim < T:
            x = x[:, :self.out_dim]

        return x
    

if __name__ == "__main__":
    print("🚀 VballNetTiny_ONNX — модель для ONNX и высокой скорости на CPU\n")

    # Параметры
    BATCH_SIZE = 1
    IN_DIM = 9
    OUT_DIM = 9
    HEIGHT = 288
    WIDTH = 512
    ONNX_PATH = "vballnet_tiny_cpu.onnx"

    # Только CPU для совместимости с ONNX
    device = torch.device("cpu")
    print(f"🔧 Устройство: {device}")

    # Инициализация модели
    model = VballNetV1d(
        height=HEIGHT,
        width=WIDTH,
        in_dim=IN_DIM,
        out_dim=OUT_DIM
    ).to(device)

    # Подсчёт параметров
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧮 Количество параметров: {total_params:,}")

    # Тестовый ввод
    x_test = torch.randn(BATCH_SIZE, IN_DIM, HEIGHT, WIDTH)

    # Проверка forward
    with torch.no_grad():
        try:
            output = model(x_test)
            print(f"✅ Forward прошёл успешно. Выход: {output.shape}")
            assert output.shape == (BATCH_SIZE, OUT_DIM, HEIGHT, WIDTH), "Неверная форма выхода"
        except Exception as e:
            print(f"❌ Ошибка в forward: {e}")
            raise

    # ---------------------------------------------
    # 📦 ЭКСПОРТ В ONNX
    # ---------------------------------------------
    print(f"\n📦 Экспорт модели в ONNX: {ONNX_PATH}")

    model.eval()
    dynamic_axes = {
        'input': {0: 'batch', 1: 'time'},
        'output': {0: 'batch', 1: 'time'}
    }

    try:
        torch.onnx.export(
            model,
            x_test,
            ONNX_PATH,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        print(f"✅ Успешно экспортировано в {ONNX_PATH}")
    except Exception as e:
        print(f"❌ Ошибка экспорта в ONNX: {e}")
        raise

    # ---------------------------------------------
    # ⏱️ ЗАМЕР СКОРОСТИ В ONNX RUNTIME
    # ---------------------------------------------
    try:
        import onnxruntime as ort

        print("\n⏱️  Запуск ONNX Runtime (CPU) для замера скорости...")

        # Опции для оптимизации
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4  # настрой под число ядер
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        ort_session = ort.InferenceSession(ONNX_PATH, sess_options=sess_options, providers=['CPUExecutionProvider'])

        print(f"✅ ONNX Runtime запущен. Провайдер: {ort_session.get_providers()[0]}")

        # Подготовка данных
        x_np = np.random.randn(BATCH_SIZE, IN_DIM, HEIGHT, WIDTH).astype(np.float32)

        # Прогрев
        for _ in range(10):
            ort_session.run(None, {'input': x_np})

        # Замер
        n_runs = 100
        start = time.time()
        for _ in range(n_runs):
            ort_session.run(None, {'input': x_np})
        end = time.time()

        avg_time_ms = (end - start) / n_runs * 1000
        fps = 1000 / avg_time_ms

        print(f"\n📊 Результаты на CPU:")
        print(f"   Среднее время: {avg_time_ms:.3f} мс")
        print(f"   Скорость: {fps:.2f} FPS")
        print(f"✅ {'Достигнут 40+ FPS' if fps >= 40 else 'Не достигнут 40 FPS'}")

    except ImportError:
        print("\n⚠️  onnxruntime не установлен. Установи: pip install onnxruntime")
    except Exception as e:
        print(f"\n❌ Ошибка ONNX Runtime: {e}")

    print("\n🎉 Модель готова к использованию в production!")