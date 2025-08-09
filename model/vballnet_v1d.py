import torch
import torch.nn as nn
import torch.nn.functional as F

class VballNetV1d(nn.Module):
    """
    UNet архитектура для детекции волейбольного мяча на видео
    Вход: 9 grayscale кадров 512x288
    Выход: 9 heatmap предсказаний для каждого мяча
    Оптимизирована для ONNX и CPU с fps > 60
    """
    
    def __init__(self, in_dim=9, out_dim=9, height=288, width=512):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.img_height = height
        self.img_width = width
        
        # Encoder (downsampling path)
        self.enc1 = self._make_encoder_block(in_dim, 32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self._make_encoder_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self._make_encoder_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = self._make_encoder_block(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._make_encoder_block(256, 512)
        
        # Decoder (upsampling path)
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._make_decoder_block(512, 256)
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._make_decoder_block(256, 128)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._make_decoder_block(128, 64)
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._make_decoder_block(64, 32)
        
        # Final output
        self.final_conv = nn.Conv2d(32, out_dim, kernel_size=1)
        
        # Инициализация весов
        self._initialize_weights()
    
    def _make_encoder_block(self, in_dim, out_dim):
        """Создание блока энкодера"""
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
    
    def _make_decoder_block(self, in_dim, out_dim):
        """Создание блока декодера"""
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
    
    def _initialize_weights(self):
        """Инициализация весов модели"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Вход: (B, 9, 288, 512)
        Выход: (B, 9, 288, 512)
        """
        B, T, H, W = x.shape
        
        # Flatten temporal dimension for processing
        # x = x.view(B * T, 1, H, W)  # Убираем преобразование в 1 канал
        # Мы работаем с 9 каналами сразу
        
        # Encoder
        enc1 = self.enc1(x)  # (B, 32, 288, 512)
        pool1 = self.pool1(enc1)  # (B, 32, 144, 256)
        
        enc2 = self.enc2(pool1)  # (B, 64, 144, 256)
        pool2 = self.pool2(enc2)  # (B, 64, 72, 128)
        
        enc3 = self.enc3(pool2)  # (B, 128, 72, 128)
        pool3 = self.pool3(enc3)  # (B, 128, 36, 64)
        
        enc4 = self.enc4(pool3)  # (B, 256, 36, 64)
        pool4 = self.pool4(enc4)  # (B, 256, 18, 32)
        
        # Bottleneck
        bottleneck = self.bottleneck(pool4)  # (B, 512, 18, 32)
        
        # Decoder
        up4 = self.upconv4(bottleneck)  # (B, 256, 36, 64)
        cat4 = torch.cat([up4, enc4], dim=1)  # (B, 512, 36, 64)
        dec4 = self.dec4(cat4)  # (B, 256, 36, 64)
        
        up3 = self.upconv3(dec4)  # (B, 128, 72, 128)
        cat3 = torch.cat([up3, enc3], dim=1)  # (B, 256, 72, 128)
        dec3 = self.dec3(cat3)  # (B, 128, 72, 128)
        
        up2 = self.upconv2(dec3)  # (B, 64, 144, 256)
        cat2 = torch.cat([up2, enc2], dim=1)  # (B, 128, 144, 256)
        dec2 = self.dec2(cat2)  # (B, 64, 144, 256)
        
        up1 = self.upconv1(dec2)  # (B, 32, 288, 512)
        cat1 = torch.cat([up1, enc1], dim=1)  # (B, 64, 288, 512)
        dec1 = self.dec1(cat1)  # (B, 32, 288, 512)
        
        # Final output
        heatmaps = self.final_conv(dec1)  # (B, 9, 288, 512)
        
        return heatmaps

# Создание модели
def create_model():
    model = VballNetV1d(
        in_dim=9,  # 9 grayscale frames
        out_dim=9,  # 9 heatmaps
        img_height=288,
        img_width=512
    )
    return model

# Тестирование модели
if __name__ == "__main__":
    print("🚀 Создание UNet модели для детекции волейбольного мяча...")
    
    # Параметры
    BATCH_SIZE = 1
    in_dim = 9
    out_dim = 9
    HEIGHT = 288
    WIDTH = 512
    
    # Создание модели
    model = create_model()
    
    # Подсчёт параметров
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧮 Количество параметров: {total_params:,}")
    
    # Тестовый ввод
    x_test = torch.randn(BATCH_SIZE, in_dim, HEIGHT, WIDTH)
    
    # Проверка forward
    with torch.no_grad():
        output = model(x_test)
        print(f"✅ Forward прошёл успешно. Выход: {output.shape}")
        assert output.shape == (BATCH_SIZE, out_dim, HEIGHT, WIDTH), "Неверная форма выхода"
    
    # Экспорт в ONNX
    print("\n📦 Экспорт модели в ONNX...")
    model.eval()
    
    try:
        torch.onnx.export(
            model,
            x_test,
            "volleyball_ball_unet_detector.onnx",
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch'},
                'output': {0: 'batch'}
            },
            verbose=False
        )
        print("✅ Успешно экспортировано в volleyball_ball_unet_detector.onnx")
    except Exception as e:
        print(f"❌ Ошибка экспорта в ONNX: {e}")
    
    print("\n🎉 Модель готова к использованию!")
