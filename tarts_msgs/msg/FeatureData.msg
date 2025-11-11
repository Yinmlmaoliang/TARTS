# Feature data message for TARTS online prototype update
# Contains sparse features, segmentation tensor, and metadata

std_msgs/Header header

# Sparse features from DINO (flattened)
float32[] sparse_features
int32[] sparse_features_shape  # Shape: [N_segments, C]

# Segmentation tensor (flattened)
int32[] seg_tensor
int32[] seg_tensor_shape  # Shape: [H, W]

# Image metadata
int32 original_height  # Original image height
int32 original_width   # Original image width
int32 resized_size     # Resized image size (e.g., 480)
