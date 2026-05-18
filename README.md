# WRC Road Boundary Detection

Real-time road boundary detection for EA Sports WRC gameplay using semantic segmentation and transfer learning.

## Overview

Part of a group project to build an autonomous driving bot for EA Sports WRC. This component handles road boundary detection — identifying the drivable surface in real time and computing a steering signal based on the car's position relative to the detected boundaries.

Classical approaches like Hough Line Transform were evaluated and rejected due to the unstructured nature of rally stages — no lane markings, irregular edges, and road surfaces that frequently blend with the surrounding environment. A deep learning approach using semantic segmentation was adopted instead.

## Demo

| Detection Overlay | Road Mask |
| Green = drivable region | White = road, Black = not road |
| Blue = left boundary | Red = right boundary |
| Cyan dot = car position | Status = ON ROAD / CORRECTING |

## How It Works

1. Each frame is passed through a MobileNetV2-based segmentation model
2. The model outputs a road probability map (0–1 per pixel)
3. The last 5 predictions are averaged for temporal stability
4. Morphological cleanup removes noise and fills gaps
5. Left and right boundary lines are extracted and averaged across frames
6. Steering offset is computed from the car's distance to each boundary
7. Out-of-bounds correction only triggers after 10 consecutive frames off-road

## Model Architecture

- **Encoder**: MobileNetV2 pretrained on ImageNet (frozen)
- **Decoder**: 5 transposed convolutional layers with BatchNorm, ReLU, and Dropout
- **Output**: Single channel road probability map via Sigmoid activation
- **Input size**: 256×256

## Dataset

- ~900 frames manually annotated using CVAT
- Extracted from 4K 60fps WRC gameplay footage
- Labels: binary polygon masks (road / not road)
- Augmentation: horizontal flip, brightness jitter, contrast adjustment, Gaussian blur

## Training

| Best val loss | 0.1658 | **0.1493** |
| Best epoch | 3 | 22 |
| Final train-val gap | 0.179 | 0.067 |

Overfitting was addressed by adding data augmentation and Dropout layers to the decoder. The model checkpoint with the lowest validation loss is used for inference.

## Installation

```bash
git clone https://github.com/yourusername/wrc-road-detection
cd wrc-road-detection
pip install torch torchvision opencv-python numpy matplotlib
```

## Usage

```bash
# Extract frames from video
python step1_extract_frames.py

# Convert CVAT annotations to masks
python step2_convert_masks.py

# Train the model
python step3_train.py

# Run inference on video
python inference_test_3.py path/to/video.mp4
```

## Limitations

- Trained on a single Rally Estonia stage — performance varies on unseen stages
- Struggles on snow/white terrain where road and environment are visually identical
- 256×256 input resolution loses fine detail from 4K source footage
- Threshold value (0.15) was tuned empirically and may need adjustment per stage

## Tech Stack

- Python, PyTorch, OpenCV, NumPy, CVAT, Matplotlib
- GPU: NVIDIA RTX 3050 (CUDA 12.9)