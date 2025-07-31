# TrackNet V4 PyTorch

PyTorch implementation of **TrackNet V4: Enhancing Fast Sports Object Tracking with Motion Attention Maps**.

TrackNet V4 is a deep learning model for real-time tracking of small, fast-moving objects in sports videos (e.g., tennis balls, ping-pong balls). The model uses motion attention maps to enhance tracking accuracy by focusing on temporal changes between consecutive frames.

## Features

- **Motion-aware tracking**: Uses frame differencing and attention mechanisms to track fast-moving objects
- **Real-time performance**: Optimized for video processing applications
- **Robust detection**: Handles occlusion and motion blur in sports scenarios
- **End-to-end training**: Direct optimization from raw video to object coordinates

## Installation

```bash
# Clone repository
git clone https://github.com/AnInsomniacy/tracknet-v4-pytorch.git
cd tracknet-v4-pytorch

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Preprocess your dataset
python preprocessing/video_to_heatmap.py --source dataset/raw --output dataset/preprocessed

# 2. Train the model
python train.py --data dataset/preprocessed --batch 4 --epochs 30

# 3. Run inference (modify paths in script as needed)
PYTHONPATH=. python predict/video_predict.py
```

## Quick Start for ubuntu cuda with UV manager 

```bash
git clone <repository-url>
cd tracknet-v4-pytorch
curl -LsSf https://astral.sh/uv/install.sh | sh

source ~/.bashrc

uv sync
uv run preprocessing/video_to_heatmap.py --source dataset/raw --output dataset/preprocessed
uv run train.py --data dataset/preprocessed --batch 4 --epochs 30

```

## Project Structure

```
tracknet-v4-pytorch/
├── model/
│   ├── tracknet_v4.py          # Main TrackNet V4 architecture
│   ├── tracknet_v2.py          # Legacy TrackNet V2
│   └── loss.py                 # Weighted Binary Cross Entropy loss
├── preprocessing/
│   ├── video_to_heatmap.py     # Video preprocessing pipeline
│   ├── tracknet_dataset.py     # PyTorch dataset loader
│   └── data_visualizer.py      # Data visualization tools
├── predict/
│   ├── single_frame_predict.py # Single frame inference
│   └── video_predict.py        # Video batch processing
├── train.py                    # Training script
├── test.py                     # Model evaluation
└── requirements.txt
```

## Usage

### Training

```bash
# Basic training
python train.py --data dataset/preprocessed

# Custom configuration
python train.py --data dataset/preprocessed --batch 8 --epochs 50 --lr 0.001 --optimizer Adam

# Resume from checkpoint
python train.py --resume checkpoints/model.pth --data dataset/preprocessed
```

### Testing & Evaluation

```bash
# Evaluate model performance
python test.py --model best_model.pth --data dataset/test

# Generate detailed evaluation report
python test.py --model best_model.pth --data dataset/test --report detailed --out results/
```

### Inference

```bash
# Set Python path and run prediction
PYTHONPATH=. python predict/video_predict.py

# For single frame inference
PYTHONPATH=. python predict/single_frame_predict.py
```

**Note**: The prediction scripts use predefined paths. Modify the model and input paths in the script files as needed.

## Data Format

The model expects preprocessed data with the following structure:

```
dataset/
├── inputs/          # RGB frames (288×512)
└── heatmaps/        # Ground truth heatmaps (288×512)
```

Each input consists of 3 consecutive frames concatenated into a 9-channel tensor. Heatmaps are Gaussian distributions centered on object locations.

## Model Architecture

TrackNet V4 introduces motion attention to improve tracking accuracy:

- **Input**: 3 consecutive RGB frames (9 channels, 288×512)
- **Motion Prompt Layer**: Extracts motion attention from frame differences
- **Encoder-Decoder**: VGG-style architecture with skip connections
- **Output**: Probability heatmaps for object detection (3 channels, 288×512)

The motion attention mechanism helps the model focus on regions with significant temporal changes, improving detection of fast-moving objects.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{raj2024tracknetv4,
    title={TrackNetV4: Enhancing Fast Sports Object Tracking with Motion Attention Maps},
    author={Raj, Arjun and Wang, Lei and Gedeon, Tom},
    journal={arXiv preprint arXiv:2409.14543},
    year={2024}
}
```

## License

This project is available for research and educational purposes.