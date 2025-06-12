# YOLO Object Detector

[<img src="https://pjreddie.com/static/img/yologo.png" alt="You Only Look Once">](https://pjreddie.com/darknet/yolo/)

A comprehensive YOLO (You Only Look Once) object detection implementation using dlib with support for YOLOv5 and YOLOv7 architectures. This repository provides tools for training, testing, inference, data conversion, and visualization of object detection models.

## Features

- **Multiple YOLO Architectures**: Support for YOLOv5, YOLOv7, and YOLOv7-tiny models
- **Complete Training Pipeline**: From data preparation to model evaluation
- **Real-time Detection**: Webcam and video processing capabilities
- **Data Format Conversion**: Tools for converting between XML, COCO, and Darknet formats
- **Visualization Tools**: Draw bounding boxes and create training plots
- **GPU Acceleration**: Multi-GPU training support with optimized inference
- **Model Optimization**: Layer fusion and model compression utilities

## Tools Overview

### Core Detection Tools

- **`train`** - Train YOLO models with extensive configuration options
- **`test`** - Evaluate trained models and compute metrics (mAP, precision, recall)
- **`detect`** - Run inference on images, videos, or webcam feeds
- **`fuse`** - Optimize trained models by fusing layers for faster inference

### Data Conversion Utilities

- **`xml2coco`** - Convert XML datasets to COCO JSON format
- **`coco2xml`** - Convert COCO datasets to XML format
- **`xml2darknet`** - Convert XML datasets to Darknet format
- **`darknet2xml`** - Convert Darknet datasets to XML format
- **`convert_images`** - Batch convert and resize images to JPEG XL (supports BMP, GIF, JPEG, PNG and WebP)

### Auxiliary Tools

- **`compute_anchors`** - Calculate optimal anchor boxes for your dataset
- **`draw_boxes`** - Visualize bounding boxes from XML datasets
- **`evalcoco`** - Evaluate models on COCO test-dev dataset

## Building the Project

### Prerequisites

- CMake 3.14 or higher
- C++17 compatible compiler
- dlib library
- OpenCV (for video/webcam support)
- nlohmann/json library

### Build Instructions

The project includes a convenient build script:

```bash
# Build release version (default)
./build.sh

# Build debug version
./build.sh debug

# The executables will be in build/Release/ or build/Debug/
```

Manual build with CMake:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DUSE_AVX_INSTRUCTIONS=ON \
         -DUSE_SSE2_INSTRUCTIONS=ON \
         -DUSE_SSE4_INSTRUCTIONS=ON
make -j$(nproc)
```

## Quick Start

### 1. Prepare Your Dataset

Create XML dataset files using dlib's format or convert from other formats:

```bash
# Convert COCO to XML
./coco2xml annotations/instances_train2017.json

# Convert Darknet to XML
./darknet2xml --names classes.names --listing train.txt
```

Your dataset directory should contain:
- `training.xml` - Training dataset metadata
- `testing.xml` - Testing dataset metadata
- Image files referenced in the XML files

### 2. Train a Model

Basic training command:

```bash
./train /path/to/dataset/directory | tee training.log
```

Advanced training with custom parameters:

```bash
./train /path/to/dataset \
    --batch-gpu 16 \
    --gpus 2 \
    --size 640 \
    --epochs 100 \
    --learning-rate 0.001 \
    --mosaic 0.5 \
    --mixup 0.15 | tee training.log
```

### 3. Test Your Model

Evaluate the trained model:

```bash
./test /path/to/dataset/testing.xml --dnn model.dnn --conf 0.25
```

### 4. Run Inference

Detect objects in images:

```bash
# Single image
./detect --dnn model.dnn --image photo.jpg

# Directory of images
./detect --dnn model.dnn --images /path/to/images/

# Webcam (real-time)
./detect --dnn model.dnn --webcam 0

# Video file
./detect --dnn model.dnn --input video.mp4 --output processed_video.mp4
```

## Training Options

### Data Augmentation
- `--mosaic 0.5` - Mosaic augmentation probability
- `--mixup 0.15` - MixUp augmentation probability  
- `--mirror 0.5` - Horizontal flip probability
- `--angle 3` - Maximum rotation in degrees
- `--scale 0.5` - Scale variation factor
- `--color 0.5 0.2` - Color augmentation (gamma, magnitude)

### Model Configuration
- `--size 512` - Input image size for training
- `--batch-gpu 8` - Batch size per GPU
- `--gpus 1` - Number of GPUs to use
- `--backbone path.dnn` - Use pre-trained backbone

### Training Schedule
- `--epochs 100` - Total training epochs
- `--learning-rate 0.001` - Initial learning rate
- `--warmup 3` - Warm-up epochs
- `--cosine` - Use cosine learning rate schedule

## Detection Options

### Model Loading
- `--dnn model.dnn` - Load trained model
- `--sync model_sync` - Load from training checkpoint

### Detection Parameters
- `--conf 0.25` - Confidence threshold
- `--nms 0.45 1.0` - NMS IoU threshold and coverage ratio
- `--size 512` - Input image size for inference

### Visualization
- `--thickness 5` - Bounding box line thickness
- `--fill 128` - Fill boxes with transparency (0-255)
- `--font custom.bdf` - Use custom font for labels
- `--no-labels` - Hide class labels
- `--no-conf` - Hide confidence scores

## Dataset Formats

### XML Format (dlib)
Standard dlib image dataset metadata format with bounding box annotations.

### COCO Format
Convert to/from COCO JSON annotation format for compatibility with other tools.

### Darknet Format
Convert to/from Darknet format (text files with normalized coordinates).

## Model Architectures

The implementation supports multiple YOLO variants:

- **YOLOv5**: Efficient and accurate general-purpose detector
- **YOLOv7**: State-of-the-art accuracy with optimized architecture  
- **YOLOv7-tiny**: Lightweight version for resource-constrained environments

Models are automatically configured based on your dataset and training parameters.

## Performance Monitoring

Training generates detailed logs and visualizations:

- `training.log` - Detailed training metrics (append `| tee training.log` to your train command)
- `loss.png` - Loss curves and learning rate plots (via gnuplot)
- Model checkpoints saved periodically
- Best model saved based on validation mAP

Use the included `plot.gp` script to generate training visualizations:

```bash
gnuplot plot.gp
```

## Contributing

This project is built on top of dlib's DNN module and follows modern C++ practices. Contributions are welcome for:

- Additional YOLO architectures
- Performance optimizations  
- New data augmentation techniques
- Improved visualization tools

## License

See `LICENSE.txt` for licensing information.
