import torch
import onnx
import argparse
from model.vballnet_v1 import VballNetV1  # Replace with the actual file/module where VballNetV1 is defined
from model.vballnet_v1c import VballNetV1c  # Import the new model version


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pth model checkpoint')
    parser.add_argument('--model_type', type=str, default='VballNetV1c', choices=['VballNetV1', 'VballNetV1c'])
    parser.add_argument('--height', type=int, default=288)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--in_dim', type=int, default=9)
    parser.add_argument('--out_dim', type=int, default=9)
    parser.add_argument('--batch_size', type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    height, width, in_dim, out_dim = args.height, args.width, args.in_dim, args.out_dim
    batch_size = args.batch_size

    # Initialize the model
    if args.model_type == 'VballNetV1c':
        model = VballNetV1c(height, width, in_dim, out_dim)
    else:
        model = VballNetV1(height, width, in_dim, out_dim)

    # Load the trained weights
    model_path = args.model_path
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Create a dummy input tensor for ONNX export
    dummy_input = torch.randn(batch_size, in_dim, height, width)
    h0 = torch.zeros(1, batch_size, 256)

    # Define the output ONNX file path (save next to model_path)
    onnx_path = model_path.replace('.pth', '.onnx')

    torch.onnx.export(
    # Экспорт
        model,
        (dummy_input, h0),
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input", "h0"],
        output_names=["heatmaps", "hn"],
        dynamic_axes={
            "input": {0: "batch"},
            "h0": {1: "batch"},
            "heatmaps": {0: "batch"},
            "hn": {1: "batch"},
        },
        verbose=False,
    )
    print(f"Model successfully exported to {onnx_path}")

    # Optional: Validate the ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation successful!")
