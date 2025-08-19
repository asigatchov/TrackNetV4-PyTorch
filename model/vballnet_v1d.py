import torch
import torch.nn as nn
import torch.onnx
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx


class Conv2DBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Single2DConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Single2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)

    def forward(self, x):
        return self.conv_1(x)


class GraphConvModule(nn.Module):
    """Graph convolution module using GAT for contextual feature extraction."""

    def __init__(self, in_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(
            in_channels, out_channels // heads, heads=heads, dropout=0.1
        )
        self.conv2 = GATConv(
            out_channels, out_channels // heads, heads=heads, dropout=0.1
        )
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return self.relu(x)


class ROIHead(nn.Module):
    """
    Обучаемый модуль для генерации ROI на основе признаков из bottleneck.
    Предсказывает heatmap активности размером (B, 1, H, W), затем выбирает top-k точек.
    """

    def __init__(self, in_channels, num_rois_per_frame=2):
        super(ROIHead, self).__init__()
        self.num_rois_per_frame = num_rois_per_frame
        self.heatmap = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x, num_rois_per_frame=None):
        """
        x: (B, C, H, W) — признаки из bottleneck
        Возвращает: list[tensor] — [ [b, t, h, w], ... ] для каждого кадра (t — индекс кадра)
        """
        B, C, H, W = x.shape
        device = x.device
        num_rois = num_rois_per_frame or self.num_rois_per_frame

        # Генерируем heatmap активности (единая для всего фрейма)
        heatmap = self.heatmap(x)  # (B, 1, H, W)
        heatmap = heatmap.squeeze(1)  # (B, H, W)

        rois = []
        for t in range(9):  # Предполагаем 9 кадров на входе
            frame_rois = []
            for b in range(B):
                prob_map = heatmap[b]  # (H, W)
                flat_probs = prob_map.view(-1)
                values, indices = torch.topk(
                    flat_probs, min(num_rois, flat_probs.numel())
                )
                coords = torch.stack([indices // W, indices % W], dim=1)  # (K, 2)

                for h, w in coords:
                    frame_rois.append([b, t, h.item(), w.item()])  # t — номер кадра

            if frame_rois:
                rois.append(torch.tensor(frame_rois, dtype=torch.float, device=device))
            else:
                rois.append(torch.empty((0, 4), dtype=torch.float, device=device))

        return rois


class VballNetV1d(nn.Module):
    def __init__(self, in_dim=9, out_dim=9, width=512, height=288):
        super(VballNetV1d, self).__init__()
        self.down_block_1 = Single2DConv(in_dim, 32)
        self.down_block_2 = Single2DConv(32, 64)
        self.down_block_3 = Single2DConv(64, 128)
        self.bottleneck = Single2DConv(128, 256)

        # ROI Head
        self.roi_head = ROIHead(in_channels=256, num_rois_per_frame=2)

        # Graph module
        self.graph_conv = GraphConvModule(in_channels=256, out_channels=256, heads=4)

        # Decoder
        self.up_block_1 = Single2DConv(384, 128)
        self.up_block_2 = Single2DConv(192, 64)
        self.up_block_3 = Single2DConv(96, 32)
        self.predictor = nn.Conv2d(32, out_dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _create_edge_index(self, rois, batch_size):
        edges = []
        node_offset = 0
        total_nodes = sum(len(roi) for roi in rois)

        for t in range(len(rois) - 1):
            curr_rois = rois[t]
            next_rois = rois[t + 1]

            for b in range(batch_size):
                curr_b = curr_rois[curr_rois[:, 0] == b]
                next_b = next_rois[next_rois[:, 0] == b]

                if len(curr_b) == 0 or len(next_b) == 0:
                    continue

                curr_coords = curr_b[:, 2:].float()
                next_coords = next_b[:, 2:].float()

                dist = torch.cdist(curr_coords, next_coords)
                min_dist, indices = torch.min(dist, dim=1)
                close = min_dist < 30

                for i, (is_close, idx) in enumerate(zip(close, indices)):
                    if is_close:
                        src = node_offset + i
                        dst = node_offset + len(curr_b) + idx
                        edges.append([src, dst])
                        edges.append([dst, src])

            node_offset += len(curr_b)

        if len(edges) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=curr_rois.device).contiguous()

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(curr_rois.device)

        # ✅ Исправление: dtype согласован
        if edge_index.size(1) > 0:
            num_nodes = total_nodes
            degrees = torch.zeros(num_nodes, device=edge_index.device, dtype=torch.float)
            ones = torch.ones_like(edge_index[0], dtype=torch.float)  # ← float
            degrees.scatter_add_(0, edge_index[0], ones)
            isolated = torch.where(degrees == 0)[0]
            if len(isolated) > 0:
                self_loops = torch.stack([isolated, isolated], dim=0).to(edge_index.device)
                edge_index = torch.cat([edge_index, self_loops], dim=1)

        return edge_index

    def _features_to_graph(self, x, rois, batch_size):
        node_features = []
        for b in range(batch_size):
            for t in range(len(rois)):
                frame_rois = rois[t]
                frame_b = frame_rois[frame_rois[:, 0] == b]
                for roi in frame_b:
                    h, w = int(roi[2]), int(roi[3])
                    if 0 <= h < x.size(2) and 0 <= w < x.size(3):
                        feat = x[b, :, h, w]
                        node_features.append(feat)
        if not node_features:
            return None, None
        node_features = torch.stack(node_features)
        return node_features, None  # batch не используется в GATConv без batch norm

    def _graph_to_features(self, graph_features, rois, batch_size, original_shape):
        N, C, H, W = original_shape
        output = torch.zeros(N, C, H, W, device=graph_features.device)
        node_idx = 0
        for b in range(batch_size):
            for t in range(len(rois)):
                frame_rois = rois[t]
                frame_b = frame_rois[frame_rois[:, 0] == b]
                for roi in frame_b:
                    h, w = int(roi[2]), int(roi[3])
                    if node_idx < len(graph_features) and 0 <= h < H and 0 <= w < W:
                        output[b, :, h, w] = graph_features[node_idx]
                        node_idx += 1
        return output

    def forward(self, x):
        batch_size = x.size(0)

        # Encoder
        x1 = self.down_block_1(x)
        x = nn.MaxPool2d(2)(x1)
        x2 = self.down_block_2(x)
        x = nn.MaxPool2d(2)(x2)
        x3 = self.down_block_3(x)
        x = nn.MaxPool2d(2)(x3)
        x = self.bottleneck(x)  # (B, 256, 36, 64)

        # Генерация ROI обучаемо
        rois = self.roi_head(x)  # обучаемая ветвь

        # GNN
        all_rois = (
            torch.cat([r for r in rois if r.size(0) > 0], dim=0)
            if any(r.size(0) > 0 for r in rois)
            else None
        )

        if all_rois is not None and len(all_rois) > 0:
            node_features, _ = self._features_to_graph(x, rois, batch_size)
            if node_features is not None and node_features.size(0) > 0:
                edge_index = self._create_edge_index(rois, batch_size)
                if edge_index.size(1) > 0:
                    graph_features = self.graph_conv(node_features, edge_index)
                    x_graph = self._graph_to_features(
                        graph_features, rois, batch_size, x.shape
                    )
                    x = x + x_graph

        # Decoder
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x3], dim=1)
        x = self.up_block_1(x)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x2], dim=1)
        x = self.up_block_2(x)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x1], dim=1)
        x = self.up_block_3(x)
        x = self.predictor(x)
        x = self.sigmoid(x)
        return x


# === Визуализация графа ===
def visualize_graph(rois, edge_index, height=288, width=512, frame_idx=0):
    """
    Визуализирует граф ROI для одного батча.
    """
    if len(rois) <= frame_idx or len(rois[frame_idx]) == 0:
        print("Нет ROI для визуализации.")
        return

    G = nx.Graph()

    # Добавляем узлы
    node_id = 0
    pos = {}
    colors = []
    for t, frame_rois in enumerate(rois):
        if frame_rois.size(0) == 0:
            continue
        for roi in frame_rois:
            b, t_idx, h, w = roi
            G.add_node(node_id, time=t_idx, pos=(w, height - h))
            pos[node_id] = (w, height - h)
            colors.append(t_idx)
            node_id += 1

    # Добавляем рёбра
    edges = edge_index.t().cpu().numpy()
    for src, dst in edges:
        if src < len(G.nodes) and dst < len(G.nodes):
            G.add_edge(src, dst)

    plt.figure(figsize=(10, 6))
    nx.draw(
        G,
        pos,
        node_color=colors,
        cmap=plt.cm.tab10,
        with_labels=False,
        node_size=100,
        edge_color="gray",
        alpha=0.8,
    )
    plt.title(f"Graph of ROIs (Frame {frame_idx})")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


# === Экспорт в ONNX ===
def export_to_onnx(model, input_shape=(2, 9, 288, 512), onnx_path="vballnet.onnx"):
    model.eval()
    dummy_input = torch.randn(input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        verbose=False,
        # Убедимся, что всё на CPU
        device=torch.device("cpu"),
    )
    print(f"Модель экспортирована в {onnx_path}")


# === Тест ===
if __name__ == "__main__":
    model = VballNetV1d().cpu()
    model.eval()

    x = torch.randn(2, 9, 288, 512)

    with torch.no_grad():
        output = model(x)
        print(f"Input: {x.shape} → Output: {output.shape}")

        # Пример визуализации
        rois = model.roi_head(
            model.bottleneck(
                model.down_block_3(
                    nn.MaxPool2d(2)(
                        model.down_block_2(nn.MaxPool2d(2)(model.down_block_1(x)))
                    )
                )
            )
        )
        edge_index = model._create_edge_index(rois, batch_size=2)
        if edge_index.size(1) > 0:
            visualize_graph(rois, edge_index, frame_idx=0)

        # Экспорт в ONNX
        export_to_onnx(model.cpu(), onnx_path="vballnet.onnx")
