#!/usr/bin/env python3
"""
Image Processing Utilities

Pure NumPy/PyTorch image processing functions, independent of ROS.
"""

import numpy as np
import cv2
import torch
from PIL import Image as PILImage
from torchvision import transforms


# Standard PyTorch transforms
TO_TENSOR = transforms.ToTensor()
TO_PIL_IMAGE = transforms.ToPILImage()


def resize_image(image, size, interpolation='linear'):
    """
    Resize image to specified size.

    Args:
        image (np.ndarray): Input image (H, W, C) or (H, W)
        size (tuple or int): Target size (H, W) or single int for square
        interpolation (str): Interpolation method ('linear', 'nearest', 'cubic')

    Returns:
        np.ndarray: Resized image
    """
    if isinstance(size, int):
        size = (size, size)

    interp_map = {
        'linear': cv2.INTER_LINEAR,
        'nearest': cv2.INTER_NEAREST,
        'cubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA,
    }

    interp = interp_map.get(interpolation, cv2.INTER_LINEAR)

    # cv2.resize expects (W, H)
    resized = cv2.resize(image, (size[1], size[0]), interpolation=interp)

    return resized


def normalize_image(image, mean, std):
    """
    Normalize image with mean and std (ImageNet-style).

    Args:
        image (np.ndarray or torch.Tensor): Image in range [0, 255] or [0, 1]
        mean (list or tuple): Mean values per channel
        std (list or tuple): Std values per channel

    Returns:
        torch.Tensor: Normalized image tensor (C, H, W)
    """
    # Convert to tensor if numpy
    if isinstance(image, np.ndarray):
        image = numpy_to_torch(image)

    # Ensure in range [0, 1]
    if image.max() > 1.0:
        image = image / 255.0

    # Apply normalization
    normalize = transforms.Normalize(mean, std)
    normalized = normalize(image)

    return normalized


def numpy_to_torch(image, device='cpu'):
    """
    Convert NumPy array to PyTorch tensor.

    Args:
        image (np.ndarray): Image array (H, W, C) or (H, W), uint8 or float
        device (str): Target device ('cpu' or 'cuda')

    Returns:
        torch.Tensor: Image tensor (C, H, W) in range [0, 1]
    """
    # Handle grayscale
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    # Convert to tensor and normalize to [0, 1]
    if image.dtype == np.uint8:
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    else:
        tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        # Ensure in [0, 1]
        if tensor.max() > 1.0:
            tensor = tensor / 255.0

    return tensor.to(device)


def torch_to_numpy(tensor):
    """
    Convert PyTorch tensor to NumPy array.

    Args:
        tensor (torch.Tensor): Image tensor (C, H, W) in range [0, 1] or [0, 255]

    Returns:
        np.ndarray: Image array (H, W, C) as uint8
    """
    # Move to CPU and convert
    tensor = tensor.cpu()

    # Ensure in range [0, 1]
    if tensor.max() > 1.0:
        tensor = tensor / 255.0

    # Use TorchVision's ToPILImage for correct handling
    pil_img = TO_PIL_IMAGE(tensor)
    np_img = np.array(pil_img)

    return np_img


def load_image(image_path, color_space='RGB'):
    """
    Load image from file.

    Args:
        image_path (str): Path to image file
        color_space (str): Color space ('RGB', 'BGR', 'GRAY')

    Returns:
        np.ndarray: Loaded image as uint8
    """
    if color_space == 'GRAY':
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if color_space == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image is None:
        raise FileNotFoundError(f'Failed to load image: {image_path}')

    return image


def save_image(image, save_path, color_space='RGB'):
    """
    Save image to file.

    Args:
        image (np.ndarray): Image array (H, W, C) or (H, W)
        save_path (str): Path to save image
        color_space (str): Color space of input image ('RGB', 'BGR', 'GRAY')
    """
    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # Convert color space if needed
    if color_space == 'RGB' and image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_path, image)


def create_grid(images, nrow=4, padding=2, pad_value=0):
    """
    Create a grid of images.

    Args:
        images (list of np.ndarray): List of images (H, W, C)
        nrow (int): Number of images per row
        padding (int): Padding between images
        pad_value (int): Padding value

    Returns:
        np.ndarray: Grid image
    """
    n = len(images)
    if n == 0:
        return np.array([])

    # Get dimensions
    h, w = images[0].shape[:2]
    channels = images[0].shape[2] if images[0].ndim == 3 else 1

    # Calculate grid dimensions
    ncol = (n + nrow - 1) // nrow
    grid_h = ncol * h + (ncol + 1) * padding
    grid_w = nrow * w + (nrow + 1) * padding

    # Create grid
    if channels == 1:
        grid = np.full((grid_h, grid_w), pad_value, dtype=np.uint8)
    else:
        grid = np.full((grid_h, grid_w, channels), pad_value, dtype=np.uint8)

    # Fill grid
    for idx, img in enumerate(images):
        row = idx // nrow
        col = idx % nrow
        y = row * h + (row + 1) * padding
        x = col * w + (col + 1) * padding

        if img.ndim == 2 and channels == 3:
            img = np.stack([img] * 3, axis=-1)

        grid[y:y+h, x:x+w] = img

    return grid


def pad_to_size(image, target_size, pad_value=0):
    """
    Pad image to target size (centered).

    Args:
        image (np.ndarray): Input image (H, W, C) or (H, W)
        target_size (tuple): Target size (H, W)
        pad_value (int): Padding value

    Returns:
        np.ndarray: Padded image
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size

    if h >= target_h and w >= target_w:
        # Crop if larger
        start_h = (h - target_h) // 2
        start_w = (w - target_w) // 2
        return image[start_h:start_h+target_h, start_w:start_w+target_w]

    # Calculate padding
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Pad
    if image.ndim == 2:
        padded = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=pad_value
        )
    else:
        padded = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='constant',
            constant_values=pad_value
        )

    return padded


def apply_mask(image, mask, alpha=0.5, color=(0, 255, 0)):
    """
    Apply colored mask overlay to image.

    Args:
        image (np.ndarray): RGB image (H, W, 3), uint8
        mask (np.ndarray): Binary mask (H, W), float [0, 1] or uint8 [0, 255]
        alpha (float): Overlay transparency
        color (tuple): Overlay color (R, G, B)

    Returns:
        np.ndarray: Image with mask overlay, uint8
    """
    # Ensure mask is binary
    if mask.max() <= 1.0:
        mask_binary = (mask > 0.5).astype(np.uint8)
    else:
        mask_binary = (mask > 127).astype(np.uint8)

    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask_binary > 0] = color

    # Blend
    result = cv2.addWeighted(image, 1.0 - alpha, colored_mask, alpha, 0)

    return result.astype(np.uint8)
