import torch
import onnx
from model.vballnet_v1 import VballNetV1  # Replace with the actual file/module where VballNetV1 is defined
from model.vballnet_v1c import VballNetV1c  # Import the new model version


if __name__ == "__main__":
    # Model parameters
    height, width, in_dim, out_dim = 288, 512, 9, 9

    # Initialize the model
    model = VballNetV1c(height, width, in_dim, out_dim)

    # Load the trained weights
    model_path = "weights/VballNetV1_gru2.pth"
    model_path = "outputs/exp_VballNetV1c_seq9_grayscale_20250804_235835/checkpoints/VballNetV1c_seq9_grayscale_best.pth"
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    # Extract the model_state_dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)  # Fallback in case the file contains only state_dict
    model.eval()  # Set to evaluation mode

    # Create a dummy input tensor for ONNX export
    dummy_input = torch.randn(1, in_dim, height, width)

    # Define the output ONNX file path
    onnx_path = model_path.replace('.pth', '.onnx')

    # Export the model to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=15,
        verbose=False
    )

    print(f"Model successfully exported to {onnx_path}")

    # Optional: Validate the ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation successful!")
