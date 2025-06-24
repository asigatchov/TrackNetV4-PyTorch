# TrackNetV4-PyTorch

PyTorch implementation of **TrackNetV4: Enhancing Fast Sports Object Tracking with Motion Attention Maps** - a novel deep learning framework for high-speed shuttlecock and ball tracking in sports videos.

## Quick Start

### Installation
```bash
git clone <repository-url>
cd TrackNetV4-PyTorch
pip install torch torchvision opencv-python pandas numpy scipy tqdm matplotlib
```

### Basic Pipeline
1. **Preprocess**: `python dataset_preprocessor.py --source dataset/raw --output dataset/preprocessed`
2. **Train**: `python train.py --dataset_dir dataset/preprocessed --batch_size 4 --num_epochs 30`
3. **Visualize**: `python dataset_player.py --source dataset/preprocessed/match1`

## Project Structure
```
TrackNetV4-PyTorch/
├── TrackNet.py                    # TrackNetV4 model with motion attention
├── train.py                       # Training pipeline with WBCE loss
├── dataset_preprocessor.py        # Video preprocessing pipeline
├── dataset_frame_heatmap.py       # PyTorch dataset loader
├── dataset_player.py              # Interactive visualization tool
└── dataset/
    ├── raw/                      # Original videos + CSV annotations
    └── preprocessed/             # Processed frames + Gaussian heatmaps
```

## TrackNetV4 Architecture

### Core Innovation: Motion-Aware Fusion

TrackNetV4 enhances traditional TrackNet by integrating **learnable motion attention maps** with high-level visual features, addressing the challenge of tracking fast-moving small objects in sports videos.

### Key Components

**1. Motion Prompt Layer**
- Generates motion attention maps from frame differencing
- Uses absolute frame differences to capture both positive and negative intensity changes
- Applies Power Normalization with only 2 learnable parameters

**2. Motion-Aware Fusion Mechanism**
- Combines motion attention maps with visual features through element-wise multiplication
- Preserves ball location and trajectory information
- Fusion operation: `A ⊚ V = [V_t, A_t ⊙ V_(t+1), ..., A_(T-2) ⊙ V_(T-1)]`

**3. MIMO Architecture**
- **Input**: 3 consecutive RGB frames → 9 channels (288×512 resolution)
- **Output**: 3 probability heatmaps → temporal consistency
- **Encoder**: VGG16-style with batch normalization and skip connections
- **Decoder**: Symmetric upsampling for pixel-level precision

### Architecture Flow

1. **Temporal Block Formation**: Group 3 consecutive video frames
2. **Frame Differencing**: Compute absolute differences between adjacent frames
3. **Motion Attention Generation**: Apply motion prompt layer to highlight moving regions
4. **Visual Feature Extraction**: Extract high-level features using VGG16-based encoder
5. **Motion-Aware Fusion**: Combine motion attention with visual features
6. **Heatmap Generation**: Decode fused features to probability heatmaps

## Technical Principles

### Motion Attention Mechanism
- **Frame Differencing**: `D_t = |F_(t+1) - F_t|` captures motion dynamics
- **Attention Maps**: Highlight relevant motion regions while suppressing noise
- **Temporal Modeling**: Preserves motion information across multiple frames

### Weighted Binary Cross Entropy Loss
- Addresses class imbalance in heatmap prediction
- Weight `w = y_pred` emphasizes harder examples
- Formula: `WBCE = -[(1-w)² × y_true × log(y_pred) + w² × (1-y_true) × log(1-y_pred)]`

### Gaussian Heatmap Generation
- Ground truth represented as 2D Gaussian distributions
- Center at ball/shuttlecock position with configurable sigma
- Smooth probability distribution for robust training

## Data Format

### Input Dataset
```
dataset/raw/match1/
├── csv/rally1_ball.csv           # Frame,Visibility,X,Y annotations
└── video/rally1.mp4              # Original match video
```

### Processed Dataset
```
dataset/preprocessed/match1/
├── inputs/rally1/                # 512×288 RGB frames
└── heatmaps/rally1/              # Gaussian heatmaps (sigma=3.0)
```

## Performance Improvements

### Quantitative Results
- **Tennis Ball Tracking**: +0.6% accuracy, +0.8% F1-score over TrackNetV2
- **Shuttlecock Tracking**: +0.8% accuracy, +0.5% F1-score over TrackNetV2
- **Processing Speed**: ~160 FPS with minimal computational overhead

### Key Advantages
- **Lightweight**: Only 2 additional learnable parameters
- **Plug-and-Play**: Compatible with existing TrackNet architectures
- **Real-time**: Maintains high processing speed for live analysis
- **Robust**: Better handling of occlusion and low visibility scenarios

## Training Configuration

### Model Specifications
- **Parameters**: ~15M (base TrackNet) + 2 (motion prompt layer)
- **Memory Usage**: ~2GB GPU memory (batch size 8)
- **Optimal Settings**: Adadelta optimizer, ReduceLROnPlateau scheduler
- **Default Epochs**: 30 with early stopping

### Evaluation Metrics
- **Accuracy**: Correct predictions within 4-pixel tolerance
- **Precision/Recall/F1**: Standard classification metrics
- **Processing Speed**: Frames per second (FPS)

## Interactive Visualization

The dataset player provides real-time visualization of training data with heatmap overlays, supporting various controls for sequence navigation, transparency adjustment, and frame-by-frame analysis.

## Citation

```bibtex
@article{raj2024tracknetv4,
    title={TrackNetV4: Enhancing Fast Sports Object Tracking with Motion Attention Maps},
    author={Raj, Arjun and Wang, Lei and Gedeon, Tom},
    journal={arXiv preprint arXiv:2409.14543},
    year={2024}
}
```

## License

Open source implementation for academic and research purposes.