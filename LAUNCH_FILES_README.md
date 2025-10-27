# TARTS Launch Files Guide

## Overview

The TARTS launch files have been updated to automatically load parameters from YAML configuration files, with support for command-line overrides. This provides a clean separation between default configurations and runtime customization.

## Launch File Architecture

### 1. **tarts_segmentation.launch.py**
Standard segmentation without online prototype updates.
- **Default config**: `tarts_ros/config/tarts_params.yaml`
- **Use case**: Static prototype segmentation

### 2. **tarts_online_update.launch.py**
Segmentation with online prototype adaptation for mobile robots.
- **Default config**: `tarts_ros/config/online_update_params.yaml`
- **Use case**: Dynamic environments with robot navigation
- **Launches**: Both segmentation node and prototype update node

## Usage Patterns

### Pattern 1: Default Configuration

The simplest usage - just launch with default settings from config file:

```bash
# Standard segmentation
ros2 launch tarts_ros tarts_segmentation.launch.py

# Online update mode
ros2 launch tarts_ros tarts_online_update.launch.py
```

**When to use**: When your configuration is already set in the YAML file.

### Pattern 2: Parameter Override

Override specific parameters without modifying config file:

```bash
# Change object class
ros2 launch tarts_ros tarts_segmentation.launch.py \
  class_name:=tomato

# Switch to Otsu thresholding
ros2 launch tarts_ros tarts_segmentation.launch.py \
  threshold_method:=otsu

# Multiple overrides
ros2 launch tarts_ros tarts_segmentation.launch.py \
  class_name:=pepper \
  threshold_method:=otsu \
  device:=cpu
```

**When to use**: Quick testing or one-off parameter changes.

### Pattern 3: Custom Configuration File

Use a completely different configuration file:

```bash
# Create custom config
cp $(ros2 pkg prefix tarts_ros)/share/tarts_ros/config/tarts_params.yaml \
   ~/my_robot_config.yaml

# Edit ~/my_robot_config.yaml as needed
# ...

# Launch with custom config
ros2 launch tarts_ros tarts_segmentation.launch.py \
  config_file:=~/my_robot_config.yaml
```

**When to use**:
- Multiple robot configurations
- Different deployment environments
- Production vs development settings

### Pattern 4: Custom Config + Overrides

Combine custom config file with parameter overrides:

```bash
# Use custom config but override device
ros2 launch tarts_ros tarts_segmentation.launch.py \
  config_file:=~/production_config.yaml \
  device:=cuda
```

**When to use**: Environment-specific configs with runtime flexibility.

## Parameter Override Mechanism

### How It Works

1. **Config file loaded first**: All parameters from YAML loaded into node
2. **Command-line overrides applied**: Any launch arguments override config values
3. **Empty strings ignored**: Only non-empty override values are applied

### Example

**Config file (`tarts_params.yaml`):**
```yaml
/**:
  ros__parameters:
    class_name: 'corn'
    threshold_method: 'fixed'
    threshold: 0.5
    device: 'cuda'
```

**Launch command:**
```bash
ros2 launch tarts_ros tarts_segmentation.launch.py \
  threshold_method:=otsu
```

**Result:**
- `class_name`: 'corn' (from config)
- `threshold_method`: 'otsu' (overridden)
- `threshold`: 0.5 (from config)
- `device`: 'cuda' (from config)

## Available Override Parameters

### Common Parameters (Both Launch Files)

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `config_file` | string | Path to custom config file | `~/my_config.yaml` |
| `class_name` | string | Object class to segment | `tomato` |
| `threshold_method` | string | Thresholding method | `otsu` or `fixed` |
| `device` | string | Computing device | `cuda` or `cpu` |
| `debug` | bool | Enable debug output | `true` |

### Online Update Specific

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `momentum` | float | Prototype update momentum [0-1] | `0.95` |
| `update_interval` | float | Update distance interval (meters) | `0.3` |

## Best Practices

### 1. Version Control Configuration Files

Keep your config files in version control:

```bash
# Development config
git add tarts_ros/config/tarts_params_dev.yaml

# Production config
git add tarts_ros/config/tarts_params_prod.yaml
```

### 2. Environment-Specific Configs

Create configs per environment:

```
configs/
├── lab_robot.yaml          # Lab testing config
├── field_robot.yaml        # Field deployment config
└── simulation.yaml         # Simulation config
```

Launch with environment variable:

```bash
export TARTS_CONFIG=~/configs/lab_robot.yaml
ros2 launch tarts_ros tarts_segmentation.launch.py \
  config_file:=$TARTS_CONFIG
```

### 3. Document Your Parameters

Add comments to custom config files:

```yaml
/**:
  ros__parameters:
    # Corn detection for Experiment #23
    class_name: 'corn'

    # Using Otsu for varying sunlight conditions
    threshold_method: 'otsu'
    otsu_method: 'valley_emphasis'  # Better for outdoor scenes
```

### 4. Test Overrides Before Editing Config

Test parameter changes via overrides before committing to config:

```bash
# Test new threshold method
ros2 launch tarts_ros tarts_segmentation.launch.py \
  threshold_method:=valley_deepness \
  otsu_sigma:=2.0

# If works well, update config file permanently
```

## Troubleshooting

### Issue: Parameters Not Loading

**Check config file path:**
```bash
# Verify default config exists
ros2 pkg prefix tarts_ros
ls $(ros2 pkg prefix tarts_ros)/share/tarts_ros/config/
```

**Check config file syntax:**
```bash
# YAML syntax validator
python3 -c "import yaml; yaml.safe_load(open('tarts_params.yaml'))"
```

### Issue: Override Not Working

**Verify parameter name:**
- Parameter names must match exactly (case-sensitive)
- Check node declares the parameter

**Use `--ros-args` for debugging:**
```bash
ros2 launch tarts_ros tarts_segmentation.launch.py \
  --show-args
```

### Issue: Custom Config Not Found

**Use absolute paths:**
```bash
# Bad
config_file:=configs/my_config.yaml

# Good
config_file:=/home/user/configs/my_config.yaml
config_file:=$(pwd)/configs/my_config.yaml
config_file:=~/configs/my_config.yaml
```

## Migration from Old Launch Files

If you were using the previous launch files with hardcoded parameters:

### Old Way (No Longer Supported)
```bash
ros2 launch tarts_ros tarts_segmentation.launch.py \
  class_name:=corn \
  threshold:=0.5 \
  device:=cuda \
  input_size:=480 \
  slic_n_segments:=400 \
  # ... many parameters
```

### New Way (Recommended)

**Option A: Edit config file once**
```yaml
# tarts_params.yaml
class_name: 'corn'
threshold: 0.5
device: 'cuda'
input_size: 480
slic_n_segments: 400
# ... all parameters
```

```bash
# Launch with defaults
ros2 launch tarts_ros tarts_segmentation.launch.py
```

**Option B: Create environment-specific configs**
```bash
# Launch with specific config
ros2 launch tarts_ros tarts_segmentation.launch.py \
  config_file:=~/corn_detection_config.yaml
```

## Advanced: Programmatic Launch

For integration into larger launch systems:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    tarts_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('tarts_ros'),
                'launch',
                'tarts_segmentation.launch.py'
            )
        ]),
        launch_arguments={
            'config_file': '/path/to/custom_config.yaml',
            'class_name': 'my_object',
        }.items()
    )

    return LaunchDescription([
        tarts_launch,
        # Your other nodes...
    ])
```

## Summary

The new launch file architecture provides:

✅ **Clean configuration management** via YAML files
✅ **Runtime flexibility** via command-line overrides
✅ **Environment separation** via custom config files
✅ **Backward compatibility** with all existing parameters
✅ **Reduced launch command complexity**

For most use cases, simply edit the config file and launch without arguments!
