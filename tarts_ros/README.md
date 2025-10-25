# TARTS ROS2 Package

ROS2 wrapper for TARTS (Template-Assisted Reference-based Target Segmentation).

## Overview

This package provides ROS2 nodes for real-time object segmentation using TARTS, which combines DINOv3 feature extraction with SLIC superpixel segmentation for efficient template-based object detection.

## Architecture

- **tarts_core**: Core segmentation engine (independent of ROS)
- **tarts_ros**: ROS2 wrapper with multi-threaded pipeline for real-time performance

## Dependencies

- ROS2 (tested on Humble/Iron)
- Python 3.8+
- PyTorch (with CUDA support recommended)
- OpenCV
- fast-slic
- torchvision
- rclpy
- sensor_msgs
- cv_bridge

## Installation

1. Ensure `tarts_core` is in your Python path:
```bash
export PYTHONPATH=/path/to/TARTS:$PYTHONPATH
```

2. Build the ROS2 package:
```bash
cd /path/to/your/ros2_ws
colcon build --packages-select tarts_ros
source install/setup.bash
```

## Usage

### 1. Register a Prototype

First, register an object prototype from a reference image:

```bash
ros2 run tarts_ros tarts_register_prototype \
  --image /path/to/reference/image.jpg \
  --class_name my_object \
  --device cuda
```

This will save the prototype to `~/.ros/tarts_prototypes/my_object.pt`.

### 2. Run Segmentation Node

Launch the segmentation node:

```bash
# Using launch file (recommended)
ros2 launch tarts_ros tarts_segmentation.launch.py \
  class_name:=my_object \
  threshold:=0.5 \
  device:=cuda

# Or run directly
ros2 run tarts_ros tarts_segmentation \
  --ros-args \
  -p class_name:=my_object \
  -p threshold:=0.5 \
  -p device:=cuda
```

### 3. View Results

The node publishes the following topics:

- `/tarts/mask` (sensor_msgs/Image): Binary segmentation mask
- `/tarts/visualized` (sensor_msgs/Image): Visualization with overlay
- `/tarts/similarity` (std_msgs/Float32): Average similarity score

View the visualization:
```bash
ros2 run rqt_image_view rqt_image_view /tarts/visualized
```

## Configuration

Edit `config/tarts_params.yaml` to set default parameters:

```yaml
/**:
  ros__parameters:
    class_name: 'corn'
    threshold: 0.5
    device: 'cuda'
    input_size: 480
    prototype_dir: '~/.ros/tarts_prototypes'
    slic_n_segments: 400
    slic_compactness: 30.0
    image_topic: '/camera/color/image_raw'
    visualization_alpha: 0.5
    debug: false
```

## Parameters

- **class_name** (string): Name of the object class to segment
- **threshold** (float): Similarity threshold for binary mask generation [0.0-1.0]
- **device** (string): Device to use ('cuda' or 'cpu')
- **input_size** (int): Input image size (images resized to input_size × input_size)
- **prototype_dir** (string): Directory containing prototype files
- **slic_n_segments** (int): Number of SLIC superpixel segments
- **slic_compactness** (float): SLIC compactness parameter
- **image_topic** (string): Input image topic to subscribe to
- **visualization_alpha** (float): Alpha blending for visualization overlay [0.0-1.0]
- **debug** (bool): Enable debug timing output

## Performance

The pipeline achieves ~20-30ms end-to-end latency on CUDA with:
- Multi-threaded architecture (preprocessing, feature extraction, matching)
- Async CUDA streams for overlapping DINO and SLIC computation
- Pre-computed pixel-to-patch mappings
- Vectorized operations (no Python loops)

## Node Architecture

```
┌─────────────┐
│Image Callback│ (Main Thread)
└──────┬──────┘
       │ Resize
       ▼
  [Queue (1)]
       │
       ▼
┌─────────────┐
│Feature Thread│
└──────┬──────┘
       │ DINO + SLIC + Sparsify
       ▼
  [Queue (10)]
       │
       ▼
┌──────────────┐
│Matching Thread│
└──────┬───────┘
       │ Match + Mask + Visualize
       ▼
   [Publish]
```

## License

MIT

## Maintainer

TARTS Maintainer <user@example.com>
