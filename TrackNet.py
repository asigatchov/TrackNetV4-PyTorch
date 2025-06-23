import torch
import torch.nn as nn


class TrackNet(nn.Module):
    """
    TrackNet MIMO版本 - 羽毛球追踪网络
    输入: [B, 9, 288, 512] -> 输出: [B, 3, 288, 512]
    """

    def __init__(self):
        super(TrackNet, self).__init__()

        # 编码器 - VGG16风格
        # Block 1: 9->64->64 (288x512)
        self.conv1_1 = nn.Conv2d(9, 64, 3, padding=1)  # 3帧×3通道=9
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # -> 144x256

        # Block 2: 64->128->128 (144x256)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # -> 72x128

        # Block 3: 128->256->256->256 (72x128)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)  # -> 36x64

        # Block 4: 256->512->512->512 (瓶颈层 36x64)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)

        # 解码器 - 带跳跃连接
        # Up 1: 512+256=768->256 (72x128)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5_1 = nn.Conv2d(768, 256, 3, padding=1)  # 512+256跳跃连接
        self.bn5_1 = nn.BatchNorm2d(256)
        self.conv5_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(256)
        self.conv5_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(256)

        # Up 2: 256+128=384->128 (144x256)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv6_1 = nn.Conv2d(384, 128, 3, padding=1)  # 256+128跳跃连接
        self.bn6_1 = nn.BatchNorm2d(128)
        self.conv6_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6_2 = nn.BatchNorm2d(128)

        # Up 3: 128+64=192->64 (288x512)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv7_1 = nn.Conv2d(192, 64, 3, padding=1)  # 128+64跳跃连接
        self.bn7_1 = nn.BatchNorm2d(64)
        self.conv7_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn7_2 = nn.BatchNorm2d(64)

        # MIMO输出: 64->3 (3个热力图)
        self.conv_out = nn.Conv2d(64, 3, 1)  # 对应3帧预测

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: [B, 9, 288, 512] - 3帧×3通道
        return: [B, 3, 288, 512] - 3个热力图
        """
        # 编码器 - 逐步下采样提取特征
        x1 = self.relu(self.bn1_1(self.conv1_1(x)))  # [B, 64, 288, 512]
        x1 = self.relu(self.bn1_2(self.conv1_2(x1)))
        x1_pool = self.pool1(x1)  # [B, 64, 144, 256]

        x2 = self.relu(self.bn2_1(self.conv2_1(x1_pool)))  # [B, 128, 144, 256]
        x2 = self.relu(self.bn2_2(self.conv2_2(x2)))
        x2_pool = self.pool2(x2)  # [B, 128, 72, 128]

        x3 = self.relu(self.bn3_1(self.conv3_1(x2_pool)))  # [B, 256, 72, 128]
        x3 = self.relu(self.bn3_2(self.conv3_2(x3)))
        x3 = self.relu(self.bn3_3(self.conv3_3(x3)))
        x3_pool = self.pool3(x3)  # [B, 256, 36, 64]

        x4 = self.relu(self.bn4_1(self.conv4_1(x3_pool)))  # [B, 512, 36, 64] 瓶颈
        x4 = self.relu(self.bn4_2(self.conv4_2(x4)))
        x4 = self.relu(self.bn4_3(self.conv4_3(x4)))

        # 解码器 - 逐步上采样+跳跃连接
        up1 = self.upsample1(x4)  # [B, 512, 72, 128]
        up1 = torch.cat([up1, x3], dim=1)  # [B, 768, 72, 128] 512+256
        up1 = self.relu(self.bn5_1(self.conv5_1(up1)))  # [B, 256, 72, 128]
        up1 = self.relu(self.bn5_2(self.conv5_2(up1)))
        up1 = self.relu(self.bn5_3(self.conv5_3(up1)))

        up2 = self.upsample2(up1)  # [B, 256, 144, 256]
        up2 = torch.cat([up2, x2], dim=1)  # [B, 384, 144, 256] 256+128
        up2 = self.relu(self.bn6_1(self.conv6_1(up2)))  # [B, 128, 144, 256]
        up2 = self.relu(self.bn6_2(self.conv6_2(up2)))

        up3 = self.upsample3(up2)  # [B, 128, 288, 512]
        up3 = torch.cat([up3, x1], dim=1)  # [B, 192, 288, 512] 128+64
        up3 = self.relu(self.bn7_1(self.conv7_1(up3)))  # [B, 64, 288, 512]
        up3 = self.relu(self.bn7_2(self.conv7_2(up3)))

        # MIMO输出3个热力图
        out = self.conv_out(up3)  # [B, 3, 288, 512]
        out = self.sigmoid(out)  # 概率值[0,1]

        return out  # out[:,0]=第1帧, out[:,1]=第2帧, out[:,2]=第3帧


if __name__ == "__main__":
    model = TrackNet()

    # 参数量统计
    params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {params:,}")

    # 测试通道数变化
    test_input = torch.randn(1, 9, 288, 512)  # 3帧×3通道=9
    output = model(test_input)

    print(f"输入: {test_input.shape} (3帧×3通道)")
    print(f"输出: {output.shape} (3个热力图)")
    print(f"输出范围: [{output.min():.3f}, {output.max():.3f}]")

    # MIMO演示: 分离3帧预测
    frame1_pred = output[0, 0]  # [288, 512] 第1帧热力图
    frame2_pred = output[0, 1]  # [288, 512] 第2帧热力图
    frame3_pred = output[0, 2]  # [288, 512] 第3帧热力图
    print(f"分离后每帧: {frame1_pred.shape}")
