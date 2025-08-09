import torch
import torch.nn as nn
import numpy as np
import time


class VballNetV1d(nn.Module):
    def __init__(self, height=288, width=512, in_dim=9, out_dim=9):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = 1024  # Увеличено до 1024 для поддержки (64, 8, 8)

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

        # Фиксированный AvgPool2d
        self.spatial_pool = nn.AvgPool2d(
            kernel_size=(18, 32), stride=(18, 32)
        )  # → (B*T, 32, 8, 8)

        # Сжатие
        self.feature_flatten = nn.Linear(32 * 8 * 8, self.hidden_size)  # 2048 → 1024

        # Временная обработка с ONNX-совместимым подходом
        self.temporal_conv1 = nn.Conv2d(
            self.hidden_size, self.hidden_size, kernel_size=3, padding=1
        )
        self.temporal_conv2 = nn.Conv2d(
            self.hidden_size, self.hidden_size, kernel_size=3, padding=1
        )
        self.temporal_act = nn.ReLU(inplace=True)

        # Декодер с увеличенными каналами
        self.hidden_to_features = nn.Linear(self.hidden_size, 64 * 8 * 8)  # 1024 → 4096
        self.feature_unflatten = nn.Unflatten(1, (64, 8, 8))

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # 8 → 16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # 16 → 32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # 32 → 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64 → 128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(288, 512), mode="nearest"),
        )

        # Skip-connection
        self.skip_conv = nn.Conv2d(
            24, 32, kernel_size=1
        )  # Обновлено для соответствия 32 каналам

        # Финальный слой
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        B, T, H, W = x.shape

        if H != 288 or W != 512:
            raise ValueError(f"Input size must be (288, 512), got ({H}, {W})")

        x = x.view(B * T, 1, H, W)  # (B*T, 1, 288, 512)

        # Skip connection: сохраняем до пулинга (24 каналов, 288x512)
        x_stem = self.stem[:4](x)  # (B*T, 24, 288, 512)
        x = self.stem[4:](x_stem)  # (B*T, 32, 144, 256)

        # Пространственное сжатие
        x = self.spatial_pool(x)  # → (B*T, 32, 8, 8)
        x = x.view(B * T, -1)  # flatten: (B*T, 2048)
        x = self.feature_flatten(x)  # (B*T, 1024)
        x = x.view(B, T, -1)  # (B, T, 1024)

        # Временная обработка (сохранение последовательности)
        batch_size = x.size(0)
        h_t = torch.zeros(batch_size, self.hidden_size, 1, 1).to(x.device)
        lstm_out = []

        for t in range(T):
            h_prev = h_t if t > 0 else torch.zeros_like(h_t)
            x_t = (
                x[:, t : t + 1, :]
                .transpose(1, 2)
                .view(batch_size, self.hidden_size, 1, 1)
            )
            h_t = self.temporal_conv1(h_prev + x_t)
            h_t = self.temporal_act(h_t)
            h_t = self.temporal_conv2(h_t)
            h_t = self.temporal_act(h_t)
            lstm_out.append(h_t)

        x = torch.stack(lstm_out, dim=1)  # (B, T, hidden_size, 1, 1)
        x = x.view(B * T, self.hidden_size, 1, 1)  # (B*T, hidden_size, 1, 1)

        # Распаковка и восстановление пространственных размеров
        x = x.view(B * T, -1)  # (B*T, hidden_size)
        x = self.hidden_to_features(x)  # (B*T, 64*8*8)
        x = self.feature_unflatten(x)  # (B*T, 64, 8, 8)
        x = self.upsample(x)  # (B*T, 32, 288, 512)

        # Skip-connection: 24 → 32 каналов
        x_skip = self.skip_conv(x_stem)  # (B*T, 32, 288, 512)
        x = x + x_skip  # residual

        # Финальный слой
        x = self.final_conv(x)  # (B*T, 1, 288, 512)
        x = x.reshape(B, T, 288, 512)

        # Коррекция out_dim
        if self.out_dim > T:
            x = torch.cat([x, x[:, -1:].expand(B, self.out_dim - T, 288, 512)], dim=1)
        elif self.out_dim < T:
            x = x[:, : self.out_dim]

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
    model = VballNetV1d(height=HEIGHT, width=WIDTH, in_dim=IN_DIM, out_dim=OUT_DIM).to(
        device
    )

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
            assert output.shape == (
                BATCH_SIZE,
                OUT_DIM,
                HEIGHT,
                WIDTH,
            ), "Неверная форма выхода"
        except Exception as e:
            print(f"❌ Ошибка в forward: {e}")
            raise

    # 📦 ЭКСПОРТ В ONNX
    print(f"\n📦 Экспорт модели в ONNX: {ONNX_PATH}")

    model.eval()
    dynamic_axes = {"input": {0: "batch", 1: "time"}, "output": {0: "batch", 1: "time"}}

    try:
        torch.onnx.export(
            model,
            x_test,
            ONNX_PATH,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            verbose=False,
        )
        print(f"✅ Успешно экспортировано в {ONNX_PATH}")
    except Exception as e:
        print(f"❌ Ошибка экспорта в ONNX: {e}")
        raise

    # ⏱️ ЗАМЕР СКОРОСТИ В ONNX RUNTIME
    try:
        import onnxruntime as ort

        print("\n⏱️  Запуск ONNX Runtime (CPU) для замера скорости...")

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        ort_session = ort.InferenceSession(
            ONNX_PATH, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )

        print(f"✅ ONNX Runtime запущен. Провайдер: {ort_session.get_providers()[0]}")

        x_np = np.random.randn(BATCH_SIZE, IN_DIM, HEIGHT, WIDTH).astype(np.float32)

        for _ in range(10):
            ort_session.run(None, {"input": x_np})

        n_runs = 100
        start = time.time()
        for _ in range(n_runs):
            ort_session.run(None, {"input": x_np})
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
