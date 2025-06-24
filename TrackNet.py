import torch
import torch.nn as nn


class WeightedBinaryCrossEntropy(nn.Module):
    """
    论文中定义的加权二元交叉熵损失函数
    WBCE = -Σ[(1-w)² * ŷ * log(y) + w² * (1-ŷ) * log(1-y)]
    其中 w = y (预测值本身作为权重)
    """

    def __init__(self, epsilon=1e-7):
        super(WeightedBinaryCrossEntropy, self).__init__()
        self.epsilon = epsilon  # 防止log(0)

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: 模型预测 [B, 3, H, W]，值域[0,1]
            y_true: 真实标签 [B, 3, H, W]，值域{0,1}
        Returns:
            loss: 标量损失值
        """
        # 确保预测值在有效范围内，避免log(0)
        y_pred = torch.clamp(y_pred, self.epsilon, 1 - self.epsilon)

        # w = y (论文定义：权重等于预测值)
        w = y_pred

        # 计算加权二元交叉熵
        # WBCE = -Σ[(1-w)² * ŷ * log(y) + w² * (1-ŷ) * log(1-y)]
        term1 = (1 - w) ** 2 * y_true * torch.log(y_pred)
        term2 = w ** 2 * (1 - y_true) * torch.log(1 - y_pred)

        # 负号在前，求和
        wbce = -(term1 + term2)

        # 返回批次平均损失
        return wbce.mean()


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


def generate_heatmap(size, center, sigma=5):
    """
    生成2D高斯热力图作为ground truth
    Args:
        size: (H, W) 热力图尺寸
        center: (x, y) 中心点坐标
        sigma: 高斯分布标准差
    Returns:
        heatmap: [H, W] 热力图
    """
    H, W = size
    x, y = center

    # 创建坐标网格
    X, Y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    X = X.float()
    Y = Y.float()

    # 计算高斯分布
    heatmap = torch.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * sigma ** 2))

    return heatmap


if __name__ == "__main__":
    # 创建模型和损失函数
    model = TrackNet()
    criterion = WeightedBinaryCrossEntropy()

    # 参数量统计
    params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {params:,}")

    # 测试前向传播
    batch_size = 2
    test_input = torch.randn(batch_size, 9, 288, 512)  # 3帧×3通道=9
    output = model(test_input)

    print(f"\n输入: {test_input.shape} (3帧×3通道)")
    print(f"输出: {output.shape} (3个热力图)")
    print(f"输出范围: [{output.min():.3f}, {output.max():.3f}]")

    # 创建测试用的ground truth热力图
    gt_heatmaps = torch.zeros(batch_size, 3, 288, 512)

    # 为每个批次和每帧生成随机的球位置
    for b in range(batch_size):
        for f in range(3):
            # 随机球位置
            ball_x = torch.randint(50, 462, (1,)).item()
            ball_y = torch.randint(50, 238, (1,)).item()

            # 生成高斯热力图
            heatmap = generate_heatmap((288, 512), (ball_x, ball_y))
            gt_heatmaps[b, f] = heatmap

    # 计算损失
    loss = criterion(output, gt_heatmaps)
    print(f"\n损失值: {loss.item():.4f}")

    # 演示加权效果
    print("\n演示加权机制:")
    # 创建一个简单的例子
    y_pred_demo = torch.tensor([[0.1, 0.9], [0.5, 0.5]])  # 预测值
    y_true_demo = torch.tensor([[0.0, 1.0], [1.0, 0.0]])  # 真实值

    # 计算权重 w = y
    w_demo = y_pred_demo

    # 显示每个像素的权重
    print(f"预测值: {y_pred_demo}")
    print(f"真实值: {y_true_demo}")
    print(f"权重 w: {w_demo}")

    # 计算各项
    term1 = (1 - w_demo) ** 2 * y_true_demo
    term2 = w_demo ** 2 * (1 - y_true_demo)

    print(f"背景权重 (1-w)²: {(1 - w_demo) ** 2}")
    print(f"前景权重 w²: {w_demo ** 2}")
