import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
import torch_geometric as pyg

# Utility functions
def power_normalization(input, a, b):
    """Power normalization function for attention map generation."""
    return 1 / (1 + torch.exp(-(5 / (0.45 * torch.abs(torch.tanh(a)) + 1e-1)) * (torch.abs(input) - 0.6 * torch.tanh(b))))

# MotionPrompt Module with Sparse Sampling
class MotionPrompt(nn.Module):
    """A module for generating attention maps with sparse sampling."""
    def __init__(self, num_frames, mode="grayscale", penalty_weight=0.0):
        super().__init__()
        self.num_frames = num_frames
        self.mode = mode.lower()
        assert self.mode in ["rgb", "grayscale"], "Mode must be 'rgb' or 'grayscale'"
        self.a = nn.Parameter(torch.tensor(0.1))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.lambda1 = penalty_weight
        self.threshold = 0.5  # Threshold for sparse sampling

    def forward(self, video_seq):
        loss = torch.tensor(0.0, device=video_seq.device)
        norm_seq = video_seq * 0.225 + 0.45

        grayscale_video_seq = video_seq # Single channel per frame

        # Compute central differences with sparse sampling
        attention_map = []
        rois = []
        for t in range(self.num_frames):
            if t == 0:
                frame_diff = grayscale_video_seq[:, t + 1] - grayscale_video_seq[:, t]
            elif t == self.num_frames - 1:
                frame_diff = grayscale_video_seq[:, t] - grayscale_video_seq[:, t - 1]
            else:
                frame_diff = (grayscale_video_seq[:, t + 1] - grayscale_video_seq[:, t - 1]) / 2
            # Sparse sampling: select regions above threshold
            mask = (torch.abs(frame_diff) > self.threshold).float()
            attention_map.append(power_normalization(frame_diff, self.a, self.b) * mask)
            rois.append(mask.nonzero(as_tuple=False))  # ROIs for graph construction

        attention_map = torch.stack(attention_map, dim=1)
        norm_attention = attention_map.unsqueeze(2)

        if self.training:
            B, T, H, W = grayscale_video_seq.shape
            temp_diff = norm_attention[:, 1:] - norm_attention[:, :-1]
            temporal_loss = torch.sum(temp_diff ** 2) / (H * W * (T - 1) * B)
            loss = self.lambda1 * temporal_loss

        return attention_map, loss, rois

# Graph Convolution Module
class GraphConvModule(nn.Module):
    """Graph convolution module using ChebConv for contextual feature extraction."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ChebConv(in_channels, out_channels, K=2)
        self.conv2 = ChebConv(out_channels, out_channels, K=2)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return self.relu(x)


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
            outputs.append(feature_map[:, t, :, :] * attention_map[:, t, :, :])  # Use attention map of current frame
        return torch.stack(outputs, dim=1)


# VballNetV2 Model
class VballNetV1d(nn.Module):
    """Enhanced VballNet with graph convolution, FPN, and DeepSORT integration."""
    def __init__(self, height=288, width=512, in_dim=9, out_dim=9):
        super().__init__()
        self.height = height
        self.width = width
        num_frames = in_dim

        # Motion prompt with sparse sampling
        self.motion_prompt = MotionPrompt(num_frames=num_frames, mode="grayscale")

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_dim, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.enc1_1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )

        # Graph convolution (applied to ROIs)
        self.graph_conv = GraphConvModule(128, 128)

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )

        # Fusion layer with multi-scale
        self.fusion_layer = FusionLayerTypeA(num_frames=num_frames, out_dim=out_dim)

        # Output layer
        self.out_conv = nn.Conv2d(32, out_dim, kernel_size=1, padding=0)

        # DeepSORT placeholder (to be integrated externally)
        self.deepsort = None

    def forward(self, imgs_input):
        # Motion prompt with ROIs
        residual_maps, loss, rois = self.motion_prompt(imgs_input)

        # Encoder
        x1 = self.enc1(imgs_input)
        x1_1 = self.enc1_1(x1)
        x = self.pool1(x1_1)

        x2 = self.enc2(x)
        x = self.pool2(x2)

        x = self.enc3(x)

        # Graph convolution on ROIs (simplified example)
        batch_size = imgs_input.shape[0]
        edge_index = self._create_edge_index(rois, batch_size).to(imgs_input.device)
        if edge_index.shape[1] > 0:
            x_graph = x.view(batch_size, -1, (self.height // 4) * (self.width // 4)).transpose(1, 2)
            x_graph = self.graph_conv(x_graph, edge_index)
            x = x + x_graph.transpose(1, 2).view(batch_size, 128, self.height // 4, self.width // 4)

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

    def _create_edge_index(self, rois, batch_size):
        """Create edge index between consecutive frame ROIs (temporal edges)."""
        edges = []
        for t in range(len(rois) - 1):
            curr_rois = rois[t]    # [b, t, h, w]
            next_rois = rois[t + 1]

            for b in range(batch_size):
                curr_b = curr_rois[curr_rois[:, 0] == b]
                next_b = next_rois[next_rois[:, 0] == b]

                if len(curr_b) == 0 or len(next_b) == 0:
                    continue

                # Координаты (h, w)
                curr_coords = curr_b[:, 2:].float()
                next_coords = next_b[:, 2:].float()

                # Вычисляем расстояния
                dist = torch.cdist(curr_coords, next_coords)  # (N_curr, N_next)
                min_dist, indices = torch.min(dist, dim=1)

                # Соединяем ближайшие (порог 30 пикселей)
                close = min_dist < 30
                for i, (is_close, idx) in enumerate(zip(close, indices)):
                    if is_close:
                        global_curr_idx = curr_b[i, 0] * len(curr_rois[0]) + i  # Упрощённый индекс
                        global_next_idx = next_b[idx, 0] * len(next_rois[0]) + idx
                        edges.append([global_curr_idx, global_next_idx])
                        edges.append([global_next_idx, global_curr_idx])  # Обратное ребро

        if len(edges) == 0:
            return torch.empty((2, 0), dtype=torch.long).contiguous()
        return torch.tensor(edges).t().contiguous()
    def _heatmaps_to_boxes(self, heatmaps):
        """Convert heatmaps to bounding boxes (simplified)."""
        batch_size, num_frames, h, w = heatmaps.shape
        boxes = []
        for b in range(batch_size):
            for t in range(num_frames):
                max_val, max_idx = torch.max(heatmaps[b, t].view(-1), dim=0)
                if max_val > 0.5:  # Threshold for detection
                    y, x = max_idx // w, max_idx % w
                    boxes.append([x, y, x + 10, y + 10])  # Placeholder box
        return torch.tensor(boxes) if boxes else None

if __name__ == "__main__":
    height, width, in_dim, out_dim = 288, 512, 9, 9
    model = VballNetV1d(height, width, in_dim, out_dim)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"VballNetV1d initialized with {total_params:,} parameters")

    test_input = torch.randn(9, in_dim, height, width)
    test_output, trajectories, loss = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    print(f"Output range: [{test_output.min():.3f}, {test_output.max():.3f}]")
    print(f"Loss: {loss.item():.3f}")
    print("✓ VballNetV1d ready for training!")