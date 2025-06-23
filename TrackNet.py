import torch
import torch.nn as nn


class TrackNet(nn.Module):
    def __init__(self):
        super(TrackNet, self).__init__()

        # 编码器部分 - VGG16风格
        # 第一个卷积块
        self.conv1_1 = nn.Conv2d(9, 64, kernel_size=3, padding=1)  # 输入9通道（3帧×3通道）
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二个卷积块
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第三个卷积块
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第四个卷积块（瓶颈层）
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)

        # 解码器部分 - 带跳跃连接的上采样
        # 第一个上采样块
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1)  # 768通道输入（跳跃连接）
        self.bn5_1 = nn.BatchNorm2d(256)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(256)
        self.conv5_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(256)

        # 第二个上采样块
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv6_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1)  # 384通道输入（跳跃连接）
        self.bn6_1 = nn.BatchNorm2d(128)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6_2 = nn.BatchNorm2d(128)

        # 第三个上采样块
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv7_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1)  # 192通道输入（跳跃连接）
        self.bn7_1 = nn.BatchNorm2d(64)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn7_2 = nn.BatchNorm2d(64)

        # 最终输出层
        self.conv_out = nn.Conv2d(64, 3, kernel_size=1)  # 1×1卷积，输出3个热力图

        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()  # 输出层使用sigmoid

    def forward(self, x):
        # 编码器前向传播
        # 第一个块
        x1 = self.relu(self.bn1_1(self.conv1_1(x)))
        x1 = self.relu(self.bn1_2(self.conv1_2(x1)))
        x1_pool = self.pool1(x1)

        # 第二个块
        x2 = self.relu(self.bn2_1(self.conv2_1(x1_pool)))
        x2 = self.relu(self.bn2_2(self.conv2_2(x2)))
        x2_pool = self.pool2(x2)

        # 第三个块
        x3 = self.relu(self.bn3_1(self.conv3_1(x2_pool)))
        x3 = self.relu(self.bn3_2(self.conv3_2(x3)))
        x3 = self.relu(self.bn3_3(self.conv3_3(x3)))
        x3_pool = self.pool3(x3)

        # 第四个块（瓶颈）
        x4 = self.relu(self.bn4_1(self.conv4_1(x3_pool)))
        x4 = self.relu(self.bn4_2(self.conv4_2(x4)))
        x4 = self.relu(self.bn4_3(self.conv4_3(x4)))

        # 解码器前向传播
        # 第一个上采样
        up1 = self.upsample1(x4)
        up1 = torch.cat([up1, x3], dim=1)  # 跳跃连接
        up1 = self.relu(self.bn5_1(self.conv5_1(up1)))
        up1 = self.relu(self.bn5_2(self.conv5_2(up1)))
        up1 = self.relu(self.bn5_3(self.conv5_3(up1)))

        # 第二个上采样
        up2 = self.upsample2(up1)
        up2 = torch.cat([up2, x2], dim=1)  # 跳跃连接
        up2 = self.relu(self.bn6_1(self.conv6_1(up2)))
        up2 = self.relu(self.bn6_2(self.conv6_2(up2)))

        # 第三个上采样
        up3 = self.upsample3(up2)
        up3 = torch.cat([up3, x1], dim=1)  # 跳跃连接
        up3 = self.relu(self.bn7_1(self.conv7_1(up3)))
        up3 = self.relu(self.bn7_2(self.conv7_2(up3)))

        # 输出层
        out = self.conv_out(up3)
        out = self.sigmoid(out)  # 输出0-1之间的热力图值

        return out


# 创建模型实例
if __name__ == "__main__":
    model = TrackNet()

    # 打印模型结构验证
    print(f"TrackNet模型已创建")

    # 测试输入输出尺寸
    test_input = torch.randn(1, 9, 512, 288)  # batch_size=1, 3帧×3通道, H=512, W=288
    output = model(test_input)
    print(f"输入尺寸: {test_input.shape}")
    print(f"输出尺寸: {output.shape}")  # 应该是 [1, 3, 512, 288]
