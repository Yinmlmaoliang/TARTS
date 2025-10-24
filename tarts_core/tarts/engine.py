#!/usr/bin/env python3
"""
TARTS Segmentation Engine

Core segmentation logic independent of ROS2:
- SLIC superpixel segmentation
- DINO feature extraction
- Feature sparsification with patch mapping
- Prototype-based matching and mask generation
"""

import torch
import torch.nn.functional as F
from torchvision import transforms as T
import numpy as np
import cv2
from fast_slic import Slic as FastSlic

from tarts_core.models.dino import Dinov3ViT


class SegmentationEngine:
    """
    Template-Assisted Reference-based Target Segmentation Engine.

    This class encapsulates the core segmentation pipeline:
    1. Image preprocessing (resize, normalize)
    2. SLIC superpixel segmentation
    3. DINO feature extraction
    4. Feature sparsification using patch-segment mapping
    5. Cosine similarity matching with prototype
    6. Binary mask generation

    Performance: ~20-30ms end-to-end latency on CUDA
    """

    def __init__(
        self,
        input_size=480,
        device='cuda',
        backbone_type='vits16',
        slic_n_segments=400,
        slic_compactness=30.0,
        dropout_p=0.0
    ):
        """
        Initialize the segmentation engine.

        Args:
            input_size (int): Input image size (images resized to input_size x input_size)
            device (str): Device to use ('cuda' or 'cpu')
            backbone_type (str): DINOv3 backbone type ('vits16', 'vitb16', 'vitl16')
            slic_n_segments (int): Number of SLIC superpixel segments
            slic_compactness (float): SLIC compactness parameter
            dropout_p (float): Dropout probability for DINO features
        """
        self.input_size = input_size
        self.device = device
        self.slic_n_segments = slic_n_segments
        self.slic_compactness = slic_compactness

        # Check device availability
        if self.device == 'cuda' and not torch.cuda.is_available():
            print('Warning: CUDA not available, falling back to CPU')
            self.device = 'cpu'

        # Initialize DINO model
        self.dino_model = Dinov3ViT(
            backbone_type=backbone_type,
            dropout_p=dropout_p,
            device=self.device
        )
        self.dino_model.eval()
        self.feature_dim = self.dino_model.get_output_feat_dim()
        self.patch_size = self.dino_model.patch_size

        # Initialize SLIC segmentation
        self.fast_slic = FastSlic(
            num_components=self.slic_n_segments,
            compactness=self.slic_compactness
        )

        # Image normalization transform (for DINO)
        # ImageNet statistics
        self.normalization = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # Pre-compute pixel-to-patch mapping for sparse feature aggregation
        self.H_feat = self.input_size // self.patch_size
        self.W_feat = self.input_size // self.patch_size
        self.num_patches = self.H_feat * self.W_feat

        # Create mapping from pixel coordinates to patch IDs
        y = torch.arange(self.input_size, device=self.device).view(-1, 1)
        x = torch.arange(self.input_size, device=self.device).view(1, -1)
        self.pixel_to_patch = ((y // self.patch_size) * self.W_feat +
                                (x // self.patch_size)).reshape(-1)  # (input_size^2,)

        # Pre-compute patch pixel counts (constant for fixed input size)
        self.patch_pixel_counts = torch.bincount(
            self.pixel_to_patch,
            minlength=self.num_patches
        ).float()

        # CUDA stream for async execution (if using CUDA)
        if self.device == 'cuda':
            self.cuda_stream = torch.cuda.Stream()
        else:
            self.cuda_stream = None

    def segment(self, image, prototype, threshold=0.5, return_similarity=True):
        """
        Perform segmentation on input image using the given prototype.

        Args:
            image (np.ndarray): RGB image as numpy array (H, W, 3), uint8
            prototype (torch.Tensor): Prototype feature vector (C,), L2-normalized
            threshold (float): Similarity threshold for binary mask generation
            return_similarity (bool): Whether to return average similarity score

        Returns:
            mask (np.ndarray): Binary segmentation mask (H, W), float [0, 1]
            avg_similarity (float): Average similarity score (only if return_similarity=True)
        """
        original_size = (image.shape[0], image.shape[1])  # (H, W)

        # 1. Preprocess: resize image
        image_resized = cv2.resize(
            image,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR
        )

        # 2. Extract features and compute SLIC segments
        dino_features, seg_tensor = self._extract_features_and_segments(image_resized)

        # 3. Sparsify features (aggregate by superpixels)
        sparse_features = self._sparsify_features_ultra_fast(dino_features, seg_tensor)

        # 4. Compute cosine similarity with prototype
        similarities = F.cosine_similarity(
            sparse_features,
            prototype.unsqueeze(0),
            dim=1
        )  # (N_segments,)

        # 5. Create binary mask based on threshold
        mask_segments = (similarities > threshold).float()  # (N_segments,)

        # 6. Map segment mask to pixel mask
        mask = self._segments_to_pixel_mask(mask_segments, seg_tensor)  # (H, W)

        # 7. Interpolate mask to original size
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        mask_original = F.interpolate(
            mask_tensor,
            size=original_size,
            mode='nearest'
        ).squeeze().numpy()  # (H_orig, W_orig)

        if return_similarity:
            avg_similarity = similarities.mean().item()
            return mask_original, avg_similarity
        else:
            return mask_original

    def _extract_features_and_segments(self, image_resized):
        """
        Extract DINO features and compute SLIC segments.
        Uses async execution on CUDA if available.

        Args:
            image_resized (np.ndarray): RGB image (input_size, input_size, 3), uint8

        Returns:
            dino_features (torch.Tensor): DINO features (1, H_feat, W_feat, C)
            seg_tensor (torch.Tensor): SLIC segment labels (H, W), long
        """
        # Convert numpy to torch tensor for DINO
        # numpy (H, W, C) uint8 -> tensor (C, H, W) float [0, 1]
        img_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(self.device)

        # Apply normalization for DINO
        img_tensor = self.normalization(img_tensor)  # (C, H, W)
        img_tensor = img_tensor.unsqueeze(0)  # (1, C, H, W)

        # Extract DINO features (async on CUDA stream if available)
        if self.cuda_stream is not None:
            # Run DINO on separate CUDA stream
            with torch.cuda.stream(self.cuda_stream):
                with torch.no_grad():
                    dino_features = self.dino_model(img_tensor)  # (1, H_feat, W_feat, C)

            # Compute SLIC segmentation on CPU while GPU is busy
            seg_tensor = self._compute_slic_segments(image_resized)

            # Wait for DINO to complete
            torch.cuda.current_stream().wait_stream(self.cuda_stream)
        else:
            # Sequential execution on CPU
            with torch.no_grad():
                dino_features = self.dino_model(img_tensor)  # (1, H_feat, W_feat, C)
            seg_tensor = self._compute_slic_segments(image_resized)

        return dino_features, seg_tensor

    def _compute_slic_segments(self, image):
        """
        Compute SLIC superpixel segmentation.

        Args:
            image (np.ndarray): RGB image (H, W, 3), uint8

        Returns:
            seg_tensor (torch.Tensor): Segment labels (H, W), long, on device
        """
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Ensure C-contiguous array for FastSlic
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)

        # Compute SLIC segmentation
        seg = self.fast_slic.iterate(image)  # (H, W) numpy array

        # Convert to tensor
        seg_tensor = torch.from_numpy(seg).to(self.device).long()

        return seg_tensor

    def _sparsify_features_ultra_fast(self, patch_features, seg_tensor):
        """
        Ultra-fast feature sparsification using pre-computed mappings.

        Aggregates patch features by superpixel segments using intelligent
        selection strategy:
        - If a segment fully covers patches: average all fully-covered patches
        - Otherwise: average the patches with maximum coverage

        Args:
            patch_features (torch.Tensor): DINO features (1, H_feat, W_feat, C)
            seg_tensor (torch.Tensor): Segment labels (H, W), long

        Returns:
            sparse_features (torch.Tensor): Aggregated features (N_segments, C)
        """
        patch_features = patch_features[0]  # (C, H_feat, W_feat)
        C = patch_features.shape[0]

        num_segments = int(seg_tensor.max().item()) + 1

        # Use pre-computed pixel-to-patch mapping
        patch_ids = self.pixel_to_patch  # Pre-computed, zero overhead
        seg_ids = seg_tensor.reshape(-1)

        # Compute patch pixel counts (use pre-computed)
        patch_counts = self.patch_pixel_counts

        # Compute (Patch, Segment) pixel counts
        ps_idx = patch_ids * num_segments + seg_ids
        ps_counts = torch.bincount(
            ps_idx,
            minlength=self.num_patches * num_segments
        ).float()
        coverage = (ps_counts.view(self.num_patches, num_segments) /
                    patch_counts.clamp(min=1).view(-1, 1))

        # Intelligent selection strategy (fully vectorized)
        # Find patches that fully cover segments (coverage >= 0.999)
        full_mask = (coverage >= 0.999)
        has_full = full_mask.any(dim=0)
        max_cov, _ = coverage.max(dim=0)

        # Build selection weights (zero-initialized)
        weights = torch.zeros_like(coverage)

        # Branch 1: Segments with full coverage (vectorized)
        if has_full.any():
            full_counts = full_mask.float().sum(dim=0)  # (num_segments,)
            # Assign weights to all full patches for each segment
            weights[:, has_full] = full_mask[:, has_full].float() / full_counts[has_full].unsqueeze(0)

        # Branch 2: Segments without full coverage (vectorized)
        partial_mask = (~has_full) & (max_cov > 0)
        if partial_mask.any():
            best_mask = (coverage == max_cov.unsqueeze(0)) & partial_mask.unsqueeze(0)
            best_counts = best_mask.float().sum(dim=0)  # (num_segments,)
            # Assign weights to best patches for each segment
            weights[:, partial_mask] = best_mask[:, partial_mask].float() / best_counts[partial_mask].unsqueeze(0)

        # Aggregate features
        patch_feats = patch_features.permute(1, 2, 0).reshape(self.num_patches, C)
        sparse_features = weights.T @ patch_feats  # (num_segments, C)

        return sparse_features

    def _segments_to_pixel_mask(self, mask_segments, seg_tensor):
        """
        Convert segment-level mask to pixel-level mask (vectorized).

        Args:
            mask_segments (torch.Tensor): Binary mask per segment (N_segments,)
            seg_tensor (torch.Tensor): Segment labels (H, W)

        Returns:
            mask (np.ndarray): Pixel-level binary mask (H, W), float
        """
        # Vectorized version: use advanced indexing to map segment masks to pixels
        mask_tensor = mask_segments[seg_tensor]  # Direct indexing, auto-broadcast
        return mask_tensor.cpu().numpy()

    def get_info(self):
        """Get engine configuration information."""
        return {
            'input_size': self.input_size,
            'device': self.device,
            'backbone': self.dino_model.get_backbone_name(),
            'feature_dim': self.feature_dim,
            'patch_size': self.patch_size,
            'num_patches': self.num_patches,
            'slic_n_segments': self.slic_n_segments,
            'slic_compactness': self.slic_compactness,
        }
