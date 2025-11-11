#!/usr/bin/env python3
"""
Image Utilities for ROS2 Integration

Provides conversion functions between ROS image messages and common Python formats
(NumPy arrays, PyTorch tensors, PIL images).
"""

import numpy as np
import cv2
import torch
from PIL import Image as PILImage
from torchvision import transforms
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage

# Global CvBridge instance for efficient conversion
CV_BRIDGE = CvBridge()

# Standard PyTorch transforms for image conversion
TO_TENSOR = transforms.ToTensor()
TO_PIL_IMAGE = transforms.ToPILImage()


def ros_image_to_numpy(ros_img, desired_encoding="rgb8"):
    """
    Convert ROS Image or CompressedImage message to NumPy array.

    Args:
        ros_img: ROS Image or CompressedImage message
        desired_encoding (str): Desired OpenCV encoding (e.g., "rgb8", "bgr8", "mono8")

    Returns:
        np.ndarray: Image as NumPy array with shape (H, W, C) or (H, W)

    Raises:
        ValueError: If message type is not supported
    """
    # Handle regular Image message
    if type(ros_img).__name__ == "_sensor_msgs__Image" or isinstance(ros_img, Image):
        np_image = CV_BRIDGE.imgmsg_to_cv2(ros_img, desired_encoding=desired_encoding)

    # Handle CompressedImage message
    elif type(ros_img).__name__ == "_sensor_msgs__CompressedImage" or isinstance(ros_img, CompressedImage):
        np_arr = np.frombuffer(ros_img.data, np.uint8)
        np_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Convert BGR to RGB if needed
        if desired_encoding == "rgb8" and "bgr" in ros_img.format:
            np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        elif desired_encoding == "bgr8" and "rgb" in ros_img.format:
            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

    else:
        raise ValueError(f"Unsupported image message type: {type(ros_img).__name__}")

    return np_image


def ros_image_to_torch(ros_img, desired_encoding="rgb8", device="cpu"):
    """
    Convert ROS Image or CompressedImage message to PyTorch tensor.

    Args:
        ros_img: ROS Image or CompressedImage message
        desired_encoding (str): Desired OpenCV encoding
        device (str): Target device ("cpu" or "cuda")

    Returns:
        torch.Tensor: Image tensor with shape (C, H, W) in range [0, 1]

    Raises:
        ValueError: If message type is not supported
    """
    # Convert to NumPy first
    np_image = ros_image_to_numpy(ros_img, desired_encoding=desired_encoding)

    # Convert NumPy to PyTorch tensor (normalizes to [0, 1])
    torch_image = TO_TENSOR(np_image).to(device)

    return torch_image


def torch_to_ros_image(torch_img, desired_encoding="rgb8"):
    """
    Convert PyTorch tensor to ROS Image message.

    Args:
        torch_img (torch.Tensor): Image tensor with shape (C, H, W)
            - Can be in range [0, 1] (float) or [0, 255] (uint8)
            - Will be automatically converted to uint8 if needed
        desired_encoding (str): ROS image encoding

    Returns:
        sensor_msgs.msg.Image: ROS Image message
    """
    # Convert to PIL Image first (handles normalization)
    pil_img = TO_PIL_IMAGE(torch_img.cpu())

    # Convert PIL to NumPy
    np_img = np.array(pil_img)

    # Convert NumPy to ROS message
    ros_img = CV_BRIDGE.cv2_to_imgmsg(np_img, encoding=desired_encoding)

    return ros_img


def numpy_to_ros_image(np_img, desired_encoding="rgb8"):
    """
    Convert NumPy array to ROS Image message.

    Args:
        np_img (np.ndarray): Image array with shape (H, W, C) or (H, W)
        desired_encoding (str): ROS image encoding

    Returns:
        sensor_msgs.msg.Image: ROS Image message
    """
    ros_image = CV_BRIDGE.cv2_to_imgmsg(np_img, encoding=desired_encoding)
    return ros_image


def torch_to_numpy(torch_img):
    """
    Convert PyTorch tensor to NumPy array.

    Args:
        torch_img (torch.Tensor): Image tensor with shape (C, H, W)

    Returns:
        np.ndarray: Image array with shape (H, W, C) as uint8
    """
    # Convert to PIL first (handles normalization)
    pil_img = TO_PIL_IMAGE(torch_img.cpu())

    # Convert to NumPy
    np_img = np.array(pil_img)

    return np_img


def numpy_to_torch(np_img, device="cpu"):
    """
    Convert NumPy array to PyTorch tensor.

    Args:
        np_img (np.ndarray): Image array with shape (H, W, C) or (H, W)
        device (str): Target device ("cpu" or "cuda")

    Returns:
        torch.Tensor: Image tensor with shape (C, H, W) in range [0, 1]
    """
    torch_img = TO_TENSOR(np_img).to(device)
    return torch_img


def mask_to_ros_image(mask, encoding="mono8"):
    """
    Convert binary mask (NumPy or PyTorch) to ROS Image message.

    Args:
        mask: Binary mask as np.ndarray (H, W) or torch.Tensor (H, W) or (1, H, W)
        encoding (str): ROS image encoding (default "mono8" for grayscale)

    Returns:
        sensor_msgs.msg.Image: ROS Image message
    """
    # Convert PyTorch to NumPy if needed
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Remove channel dimension if present
    if mask.ndim == 3:
        mask = mask.squeeze(0)

    # Ensure uint8 type
    if mask.dtype != np.uint8:
        # Assume binary mask in [0, 1] or [0, 255]
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

    # Convert to ROS message
    ros_msg = CV_BRIDGE.cv2_to_imgmsg(mask, encoding=encoding)

    return ros_msg


def ros_image_to_pil(ros_img, desired_encoding="rgb8"):
    """
    Convert ROS Image message to PIL Image.

    Args:
        ros_img: ROS Image or CompressedImage message
        desired_encoding (str): Desired encoding

    Returns:
        PIL.Image.Image: PIL Image object
    """
    np_img = ros_image_to_numpy(ros_img, desired_encoding=desired_encoding)
    pil_img = PILImage.fromarray(np_img)
    return pil_img


def pil_to_ros_image(pil_img, desired_encoding="rgb8"):
    """
    Convert PIL Image to ROS Image message.

    Args:
        pil_img (PIL.Image.Image): PIL Image object
        desired_encoding (str): ROS image encoding

    Returns:
        sensor_msgs.msg.Image: ROS Image message
    """
    np_img = np.array(pil_img)
    ros_img = CV_BRIDGE.cv2_to_imgmsg(np_img, encoding=desired_encoding)
    return ros_img
