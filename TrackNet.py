import torch
import torch.nn as nn


class TrackNet(nn.Module):
    """
    TrackNet - 基于深度学习的高速小目标追踪网络 (MIMO版本)

    网络特点:
    - 编码器-解码器结构，基于VGG16设计
    - 采用U-Net风格的跳跃连接保持细节信息
    - MIMO设计：3帧输入，同时输出3帧预测，提升处理速度
    - 专门用于追踪羽毛球等高速小目标

    输入: [batch_size, 9, 288, 512] - 3帧图像×3通道
    输出: [batch_size, 3, 288, 512] - 3个热力图(对应3帧的预测)
    """

    def __init__(self):
        super(TrackNet, self).__init__()

        # ==================== 编码器部分 - VGG16风格 ====================

        # 第一个卷积块 (输入：3帧×3通道=9通道)
        self.conv1_1 = nn.Conv2d(9, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 尺寸减半：288x512 -> 144x256

        # 第二个卷积块
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 尺寸减半：144x256 -> 72x128

        # 第三个卷积块
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 尺寸减半：72x128 -> 36x64

        # 第四个卷积块（瓶颈层，特征提取的最深层）
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)

        # ==================== 解码器部分 - 带跳跃连接的上采样 ====================

        # 第一个上采样块 (从瓶颈层开始重建)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')  # 36x64 -> 72x128
        # 跳跃连接：512(上采样) + 256(conv3_3) = 768通道
        self.conv5_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(256)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(256)
        self.conv5_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(256)

        # 第二个上采样块
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')  # 72x128 -> 144x256
        # 跳跃连接：256(上采样) + 128(conv2_2) = 384通道
        self.conv6_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1)
        self.bn6_1 = nn.BatchNorm2d(128)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6_2 = nn.BatchNorm2d(128)

        # 第三个上采样块（恢复到原始尺寸）
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')  # 144x256 -> 288x512
        # 跳跃连接：128(上采样) + 64(conv1_2) = 192通道
        self.conv7_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1)
        self.bn7_1 = nn.BatchNorm2d(64)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn7_2 = nn.BatchNorm2d(64)

        # ==================== MIMO输出层 ====================
        # 1×1卷积生成3个热力图：每个对应一帧的预测结果
        self.conv_out = nn.Conv2d(64, 3, kernel_size=1)

        # ==================== 激活函数 ====================
        self.relu = nn.ReLU(inplace=True)  # 中间层使用ReLU
        self.sigmoid = nn.Sigmoid()  # 输出层使用Sigmoid，确保热力图值在[0,1]

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量 [batch_size, 9, 288, 512]
               9通道 = 3帧图像 × 3颜色通道(RGB)

        返回:
            out: 输出张量 [batch_size, 3, 288, 512]
                 3通道 = 3个热力图(分别对应输入的3帧)
        """
        # ==================== 编码器前向传播 ====================

        # 第一个卷积块 - 提取低级特征
        x1 = self.relu(self.bn1_1(self.conv1_1(x)))  # [B, 64, 288, 512]
        x1 = self.relu(self.bn1_2(self.conv1_2(x1)))  # [B, 64, 288, 512]
        x1_pool = self.pool1(x1)  # [B, 64, 144, 256]

        # 第二个卷积块 - 提取中级特征
        x2 = self.relu(self.bn2_1(self.conv2_1(x1_pool)))  # [B, 128, 144, 256]
        x2 = self.relu(self.bn2_2(self.conv2_2(x2)))  # [B, 128, 144, 256]
        x2_pool = self.pool2(x2)  # [B, 128, 72, 128]

        # 第三个卷积块 - 提取高级特征
        x3 = self.relu(self.bn3_1(self.conv3_1(x2_pool)))  # [B, 256, 72, 128]
        x3 = self.relu(self.bn3_2(self.conv3_2(x3)))  # [B, 256, 72, 128]
        x3 = self.relu(self.bn3_3(self.conv3_3(x3)))  # [B, 256, 72, 128]
        x3_pool = self.pool3(x3)  # [B, 256, 36, 64]

        # 第四个卷积块（瓶颈层）- 提取最高级语义特征
        x4 = self.relu(self.bn4_1(self.conv4_1(x3_pool)))  # [B, 512, 36, 64]
        x4 = self.relu(self.bn4_2(self.conv4_2(x4)))  # [B, 512, 36, 64]
        x4 = self.relu(self.bn4_3(self.conv4_3(x4)))  # [B, 512, 36, 64]

        # ==================== 解码器前向传播（带跳跃连接）====================

        # 第一个上采样 - 重建高分辨率特征
        up1 = self.upsample1(x4)  # [B, 512, 72, 128]
        up1 = torch.cat([up1, x3], dim=1)  # [B, 768, 72, 128] 跳跃连接
        up1 = self.relu(self.bn5_1(self.conv5_1(up1)))  # [B, 256, 72, 128]
        up1 = self.relu(self.bn5_2(self.conv5_2(up1)))  # [B, 256, 72, 128]
        up1 = self.relu(self.bn5_3(self.conv5_3(up1)))  # [B, 256, 72, 128]

        # 第二个上采样 - 继续重建分辨率
        up2 = self.upsample2(up1)  # [B, 256, 144, 256]
        up2 = torch.cat([up2, x2], dim=1)  # [B, 384, 144, 256] 跳跃连接
        up2 = self.relu(self.bn6_1(self.conv6_1(up2)))  # [B, 128, 144, 256]
        up2 = self.relu(self.bn6_2(self.conv6_2(up2)))  # [B, 128, 144, 256]

        # 第三个上采样 - 恢复到原始输入尺寸
        up3 = self.upsample3(up2)  # [B, 128, 288, 512]
        up3 = torch.cat([up3, x1], dim=1)  # [B, 192, 288, 512] 跳跃连接
        up3 = self.relu(self.bn7_1(self.conv7_1(up3)))  # [B, 64, 288, 512]
        up3 = self.relu(self.bn7_2(self.conv7_2(up3)))  # [B, 64, 288, 512]

        # ==================== MIMO输出层 ====================
        # 生成3个热力图，分别对应3帧的球位置预测
        out = self.conv_out(up3)  # [B, 3, 288, 512]
        out = self.sigmoid(out)  # 将输出限制在[0,1]范围，表示概率

        # 输出解释：
        # out[:, 0, :, :] = 第1帧的球位置热力图
        # out[:, 1, :, :] = 第2帧的球位置热力图
        # out[:, 2, :, :] = 第3帧的球位置热力图

        return out


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 创建MIMO版本的TrackNet模型
    model = TrackNet()
    print("TrackNet (MIMO版本) 模型创建成功")

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 测试输入输出尺寸
    # 模拟输入：1个batch，3帧图像(每帧3通道RGB)，尺寸288×512
    test_input = torch.randn(1, 9, 288, 512)

    print(f"\n输入张量尺寸: {test_input.shape}")
    print("输入含义: [batch_size, 3帧×3通道, height, width]")

    # 前向传播测试
    with torch.no_grad():  # 测试时不需要梯度计算
        output = model(test_input)

    print(f"\n输出张量尺寸: {output.shape}")
    print("输出含义: [batch_size, 3个热力图, height, width]")
    print("每个热力图对应一帧的球位置预测")

    # 检查输出值范围
    print(f"\n输出值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print("正常范围应该在[0, 1]之间（Sigmoid激活函数的输出）")

    print("\n✅ MIMO TrackNet测试完成！")
