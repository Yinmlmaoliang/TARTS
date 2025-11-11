#!/usr/bin/env python3
"""
Visualization Utilities

Functions for visualizing segmentation results, masks, and features.
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def create_mask_overlay(image, mask, color=(0, 255, 0), alpha=0.5):
    """
    Create visualization by overlaying colored mask on image.

    Args:
        image (np.ndarray): RGB image (H, W, 3), uint8
        mask (np.ndarray): Binary mask (H, W), float [0, 1] or uint8
        color (tuple): Overlay color (R, G, B)
        alpha (float): Overlay transparency [0, 1]

    Returns:
        np.ndarray: Visualization image (H, W, 3), uint8
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # Resize mask to match image if needed
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    # Convert mask to binary
    if mask.max() <= 1.0:
        mask_binary = (mask > 0.5).astype(np.uint8)
    else:
        mask_binary = (mask > 127).astype(np.uint8)

    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask_binary > 0] = color

    # Blend with original image
    vis = cv2.addWeighted(image, 1.0 - alpha, colored_mask, alpha, 0)

    return vis.astype(np.uint8)


def visualize_segments(image, segments, show_boundaries=True, boundary_color=(255, 255, 255)):
    """
    Visualize superpixel segments.

    Args:
        image (np.ndarray): RGB image (H, W, 3), uint8
        segments (np.ndarray): Segment labels (H, W), int
        show_boundaries (bool): Whether to draw segment boundaries
        boundary_color (tuple): Boundary color (R, G, B)

    Returns:
        np.ndarray: Visualization image (H, W, 3), uint8
    """
    vis = image.copy()

    if show_boundaries:
        # Find boundaries using morphological operations
        segments_uint8 = segments.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(segments_uint8, kernel, iterations=1)
        boundaries = (dilated != segments_uint8).astype(np.uint8) * 255

        # Draw boundaries
        vis[boundaries > 0] = boundary_color

    return vis


def draw_similarity_map(similarities, image_shape, segments=None, colormap='jet'):
    """
    Visualize similarity scores as a heatmap.

    Args:
        similarities (np.ndarray): Similarity scores per segment (N_segments,)
        image_shape (tuple): Target image shape (H, W)
        segments (np.ndarray): Segment labels (H, W), int (optional)
        colormap (str): Matplotlib colormap name

    Returns:
        np.ndarray: Heatmap visualization (H, W, 3), uint8
    """
    if segments is not None:
        # Map segment similarities to pixel similarities
        similarity_map = similarities[segments]
    else:
        # Resize similarities directly
        similarity_map = cv2.resize(
            similarities.reshape(-1, 1),
            (image_shape[1], image_shape[0]),
            interpolation=cv2.INTER_LINEAR
        ).squeeze()

    # Normalize to [0, 1]
    sim_min = similarity_map.min()
    sim_max = similarity_map.max()
    if sim_max > sim_min:
        similarity_map = (similarity_map - sim_min) / (sim_max - sim_min)
    else:
        similarity_map = np.zeros_like(similarity_map)

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    heatmap = cmap(similarity_map)[:, :, :3]  # Remove alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)

    return heatmap


def create_side_by_side(images, titles=None, padding=10, pad_value=255):
    """
    Create side-by-side comparison of multiple images.

    Args:
        images (list of np.ndarray): List of images (H, W, C)
        titles (list of str): Titles for each image (optional)
        padding (int): Padding between images
        pad_value (int): Padding value

    Returns:
        np.ndarray: Combined visualization
    """
    if len(images) == 0:
        return np.array([])

    # Convert all images to same height
    target_h = max(img.shape[0] for img in images)
    resized_images = []

    for img in images:
        if img.shape[0] != target_h:
            aspect = img.shape[1] / img.shape[0]
            target_w = int(target_h * aspect)
            img = cv2.resize(img, (target_w, target_h))
        resized_images.append(img)

    # Calculate total width
    total_w = sum(img.shape[1] for img in resized_images) + padding * (len(images) + 1)
    channels = resized_images[0].shape[2] if resized_images[0].ndim == 3 else 1

    # Create canvas
    if channels == 1:
        canvas = np.full((target_h, total_w), pad_value, dtype=np.uint8)
    else:
        canvas = np.full((target_h, total_w, channels), pad_value, dtype=np.uint8)

    # Place images
    x_offset = padding
    for img in resized_images:
        w = img.shape[1]
        canvas[:, x_offset:x_offset+w] = img
        x_offset += w + padding

    # Add titles if provided
    if titles is not None:
        canvas = _add_titles(canvas, titles, resized_images)

    return canvas


def _add_titles(image, titles, images):
    """Helper function to add titles above images."""
    title_height = 30
    padded = np.full(
        (image.shape[0] + title_height, image.shape[1], image.shape[2]),
        255,
        dtype=np.uint8
    )
    padded[title_height:] = image

    # Add text
    x_offset = 10
    for title, img in zip(titles, images):
        w = img.shape[1]
        cv2.putText(
            padded,
            title,
            (x_offset, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
        x_offset += w + 10

    return padded


def create_comparison_grid(original, mask, overlay, similarity_map=None):
    """
    Create a comprehensive comparison grid for segmentation results.

    Args:
        original (np.ndarray): Original RGB image (H, W, 3)
        mask (np.ndarray): Binary segmentation mask (H, W)
        overlay (np.ndarray): Mask overlay on original (H, W, 3)
        similarity_map (np.ndarray): Similarity heatmap (H, W, 3), optional

    Returns:
        np.ndarray: Grid visualization
    """
    # Convert mask to 3-channel for display
    mask_vis = np.stack([mask * 255] * 3, axis=-1).astype(np.uint8)

    if similarity_map is not None:
        images = [original, mask_vis, overlay, similarity_map]
        titles = ['Original', 'Mask', 'Overlay', 'Similarity']
    else:
        images = [original, mask_vis, overlay]
        titles = ['Original', 'Mask', 'Overlay']

    return create_side_by_side(images, titles=titles)


def draw_contours(image, mask, color=(0, 255, 0), thickness=2):
    """
    Draw contours of mask on image.

    Args:
        image (np.ndarray): RGB image (H, W, 3), uint8
        mask (np.ndarray): Binary mask (H, W), float or uint8
        color (tuple): Contour color (R, G, B)
        thickness (int): Contour line thickness

    Returns:
        np.ndarray: Image with contours drawn
    """
    vis = image.copy()

    # Convert mask to binary uint8
    if mask.max() <= 1.0:
        mask_binary = (mask > 0.5).astype(np.uint8) * 255
    else:
        mask_binary = mask.astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(
        mask_binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw contours (convert RGB to BGR for cv2, then back)
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    color_bgr = (color[2], color[1], color[0])  # RGB to BGR
    cv2.drawContours(vis_bgr, contours, -1, color_bgr, thickness)
    vis = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

    return vis


def save_visualization(vis, save_path):
    """
    Save visualization to file.

    Args:
        vis (np.ndarray): Visualization image (H, W, 3), uint8
        save_path (str): Path to save image
    """
    # Convert RGB to BGR for cv2
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, vis_bgr)
    print(f'Saved visualization to: {save_path}')


def create_legend(labels, colors, size=(200, 300)):
    """
    Create a legend for class colors.

    Args:
        labels (list of str): Class labels
        colors (list of tuple): RGB colors for each class
        size (tuple): Legend size (H, W)

    Returns:
        np.ndarray: Legend image (H, W, 3), uint8
    """
    legend = np.full((size[0], size[1], 3), 255, dtype=np.uint8)

    box_size = 20
    y_offset = 10
    x_offset = 10

    for label, color in zip(labels, colors):
        # Draw color box
        cv2.rectangle(
            legend,
            (x_offset, y_offset),
            (x_offset + box_size, y_offset + box_size),
            color,
            -1
        )

        # Draw label
        cv2.putText(
            legend,
            label,
            (x_offset + box_size + 10, y_offset + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )

        y_offset += box_size + 10

    return legend
