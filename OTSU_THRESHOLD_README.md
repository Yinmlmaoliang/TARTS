# Otsu Adaptive Thresholding for TARTS

## Overview

TARTS now supports adaptive thresholding using Otsu's method as an alternative to fixed thresholding. This enhancement automatically computes optimal similarity thresholds based on the distribution of similarity scores, eliminating the need for manual threshold tuning.

## Implementation

### Core Components

1. **`tarts_core/tarts/threshold.py`**: Independent Otsu implementation
   - Histogram computation compatible with CUDA and `torch.compile`
   - Four Otsu variants: `standard`, `original`, `valley_emphasis`, `valley_deepness`
   - GPU-accelerated computation

2. **`tarts_core/tarts/engine.py`**: Enhanced SegmentationEngine
   - New parameter `threshold_method`: `'fixed'` or `'otsu'`
   - Automatic threshold computation in `_compute_otsu_threshold()` method
   - Similarity normalization to [0,1] range for Otsu, then mapping back

3. **ROS2 Integration**: Updated segmentation node
   - New parameters: `threshold_method`, `otsu_nbins`, `otsu_method`, `otsu_sigma`
   - Backward compatible: defaults to `threshold_method='fixed'`

## Usage

### Method 1: Configuration File (Recommended)

The launch files now automatically load parameters from configuration files. Simply edit `tarts_params.yaml` or `online_update_params.yaml`:

```yaml
/**:
  ros__parameters:
    # Switch to Otsu adaptive thresholding
    threshold_method: 'otsu'

    # Otsu parameters (optional, defaults shown)
    otsu_nbins: 256
    otsu_method: 'standard'  # or 'original', 'valley_emphasis', 'valley_deepness'
    otsu_sigma: 1.0  # only for 'valley_deepness' method
```

Then launch normally:

```bash
# Standard segmentation (uses tarts_params.yaml)
ros2 launch tarts_ros tarts_segmentation.launch.py

# Online update mode (uses online_update_params.yaml)
ros2 launch tarts_ros tarts_online_update.launch.py
```

### Method 2: Command-Line Override

Override specific parameters from config file:

```bash
# Override threshold method only
ros2 launch tarts_ros tarts_segmentation.launch.py \
  threshold_method:=otsu

# Override multiple parameters
ros2 launch tarts_ros tarts_segmentation.launch.py \
  class_name:=my_object \
  threshold_method:=otsu

# Online update with overrides
ros2 launch tarts_ros tarts_online_update.launch.py \
  threshold_method:=otsu \
  momentum:=0.95
```

### Method 3: Custom Configuration File

Use a completely custom config file:

```bash
# Create your custom config
cp tarts_ros/config/tarts_params.yaml /path/to/my_custom_params.yaml
# Edit /path/to/my_custom_params.yaml as needed

# Launch with custom config
ros2 launch tarts_ros tarts_segmentation.launch.py \
  config_file:=/path/to/my_custom_params.yaml
```

### Method 4: Direct Node Launch (Advanced)

Launch node directly without launch file:

```bash
ros2 run tarts_ros tarts_segmentation \
  --ros-args \
  -p class_name:=corn \
  -p threshold_method:=otsu \
  -p otsu_method:=valley_emphasis
```

## Otsu Method Variants

### 1. **standard** (Recommended)
- Standard Otsu's method maximizing between-class variance
- Numerically stable implementation
- Best for general use cases

### 2. **original**
- Legacy implementation from scikit-image
- Kept for backward compatibility
- May have numerical issues in edge cases

### 3. **valley_emphasis**
- Weights objective function with `(1 - p(t))`
- Favors threshold values at valley points (low probability)
- Good for bimodal distributions with clear valleys

### 4. **valley_deepness**
- Weights by valley depth measure: `W(t) = (1 - p(t)) + D(t)`
- Applies Gaussian smoothing (sigma parameter)
- Best for noisy distributions

**Reference**: Ng, H. F. (2006). "Automatic thresholding for defect detection"

## Algorithm Details

### Similarity Score Processing

When `threshold_method='otsu'`, the engine performs:

1. **Compute similarities**: Cosine similarity between superpixel features and prototype
   ```python
   similarities = F.cosine_similarity(sparse_features, prototype, dim=1)
   ```

2. **Normalize to [0,1]**: Required by Otsu algorithm
   ```python
   min_sim, max_sim = similarities.min(), similarities.max()
   similarities_normalized = (similarities - min_sim) / (max_sim - min_sim)
   ```

3. **Compute Otsu threshold**: On normalized distribution
   ```python
   otsu_thresh_normalized = threshold_otsu(
       similarities_reshaped,
       nbins=otsu_nbins,
       method=otsu_method
   )
   ```

4. **Map back to original range**: For final thresholding
   ```python
   threshold = otsu_thresh_normalized * (max_sim - min_sim) + min_sim
   ```

5. **Generate mask**: Using computed threshold
   ```python
   mask_segments = (similarities > threshold).float()
   ```

### Edge Case Handling

- **All similarities identical**: Returns mean value as threshold
- **Very small range** (< 1e-8): Returns mean value
- **Empty distribution**: Otsu returns 0.5 on normalized scale

## Performance

- **Overhead**: ~0.1-0.2ms for Otsu computation on CUDA
- **Histogram bins**: 256 bins provides good accuracy/speed tradeoff
- **GPU Acceleration**: Full CUDA support via `torch.histc`

## Testing

Run the test script to verify implementation:

```bash
cd /home/maoliang/ROS/TARTS_ws/src/TARTS
python3 test_otsu_threshold.py
```

Expected output:
- Thresholds computed for synthetic bimodal distribution
- All four methods produce reasonable results
- Edge cases handled correctly

## Comparison: Fixed vs Otsu

| Feature | Fixed Threshold | Otsu Adaptive |
|---------|----------------|---------------|
| **Setup** | Requires manual tuning | Automatic computation |
| **Robustness** | Sensitive to lighting/conditions | Adapts to distribution |
| **Performance** | ~0ms overhead | ~0.1-0.2ms overhead |
| **Use Case** | Known environment | Varying conditions |
| **Stability** | Consistent results | May vary per frame |

## Recommendations

### When to Use Fixed Threshold
- Environment is controlled and consistent
- Threshold has been optimized for your use case
- Minimal computational overhead is critical
- You need fully deterministic results

### When to Use Otsu
- Lighting conditions vary
- Unknown similarity score distributions
- Rapid prototyping (no threshold tuning needed)
- Online adaptation with varying prototypes

### Best Practices

1. **Start with Otsu `standard`**: Good default for most cases
2. **Use `valley_emphasis`**: If you notice bimodal distributions
3. **Try `valley_deepness`**: For noisy similarity scores
4. **Fine-tune `nbins`**: Increase to 512 for smoother distributions
5. **Validate results**: Compare Otsu vs fixed threshold on your data

## Configuration Examples

### Example 1: Basic Otsu

```yaml
threshold_method: 'otsu'
otsu_method: 'standard'
otsu_nbins: 256
```

### Example 2: Valley Emphasis for Clear Separation

```yaml
threshold_method: 'otsu'
otsu_method: 'valley_emphasis'
otsu_nbins: 256
```

### Example 3: Noisy Environment

```yaml
threshold_method: 'otsu'
otsu_method: 'valley_deepness'
otsu_nbins: 256
otsu_sigma: 2.0  # Increased smoothing
```

## Backward Compatibility

All existing configurations continue to work without modification:
- Default `threshold_method='fixed'`
- Fixed threshold behavior unchanged
- No breaking changes to API

## References

1. Otsu, N. (1979). "A threshold selection method from gray-level histograms"
2. [Wikipedia: Otsu's Method](https://en.wikipedia.org/wiki/Otsu's_Method)
3. [scikit-image threshold_otsu](https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_otsu)
4. Ng, H. F. (2006). "Automatic thresholding for defect detection"

## Troubleshooting

### Issue: Mask is empty or all foreground
**Solution**: Check similarity distribution. May need different Otsu method or revert to fixed threshold.

### Issue: Threshold varies too much between frames
**Solution**: Consider using fixed threshold for stability, or apply temporal smoothing.

### Issue: Performance degradation
**Solution**: Reduce `otsu_nbins` from 256 to 128, or use fixed threshold if overhead is critical.

## Future Enhancements

Potential improvements for future versions:
- Temporal smoothing of adaptive thresholds
- Multi-level thresholding for hierarchical segmentation
- Percentile-based thresholding alternatives
- Visualization of similarity histograms for debugging
