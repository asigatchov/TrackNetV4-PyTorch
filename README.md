# TrackNetV4-PyTorch

PyTorch implementation of TrackNetV4: Enhancing Fast Sports Object Tracking with Motion Attention Maps

## 🏆 Features

- **Motion-Aware Fusion**: Novel fusion mechanism combining visual features with learnable motion attention maps
- **Plug-and-Play Design**: Can be seamlessly integrated into existing TrackNet architectures (V2/V3)
- **Multi-Sport Support**: Optimized for tennis ball and shuttlecock tracking in broadcast videos
- **Real-time Performance**: Maintains high FPS while improving tracking accuracy
- **Multi-Ball Tracking**: Enhanced capability for challenging scenarios with multiple moving objects

## 🚀 Key Innovations

- **Motion Prompt Layer**: Generates motion attention maps from frame differencing
- **Element-wise Fusion**: Combines motion attention with high-level visual features
- **Improved Robustness**: Better performance in occlusion and low visibility scenarios
- **Lightweight Architecture**: Only 2 additional learnable parameters

## 📊 Performance

- Consistent improvements in accuracy, precision, recall, and F1-score
- Enhanced tracking of high-speed, small objects in sports videos
- Reduced false negatives and missed detections
- Superior performance on tennis ball and shuttlecock datasets

## 🏃‍♂️ Applications

- Sports video analysis and coaching
- Broadcast match video processing
- Player performance monitoring
- Automated sports statistics generation

## 📋 Datasets Supported

- Tennis ball tracking dataset
- Shuttlecock tracking dataset
- Custom multi-ball tracking scenarios