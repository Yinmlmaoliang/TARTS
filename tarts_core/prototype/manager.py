#!/usr/bin/env python3
"""
Prototype Manager

Handles loading, saving, and validation of prototype feature vectors.
"""

import os
import torch
import torch.nn.functional as F


class PrototypeManager:
    """
    Manager for prototype feature vectors.

    Prototypes are pre-computed feature vectors representing target classes.
    They are used for similarity-based matching during segmentation.
    """

    def __init__(self, device='cuda'):
        """
        Initialize prototype manager.

        Args:
            device (str): Device to load prototypes on ('cuda' or 'cpu')
        """
        self.device = device

        # Check device availability
        if self.device == 'cuda' and not torch.cuda.is_available():
            print('Warning: CUDA not available, falling back to CPU')
            self.device = 'cpu'

        # Current prototype (for online update)
        self.prototype = None

    def load(self, prototype_path, normalize=True, verbose=True):
        """
        Load prototype from file.

        Args:
            prototype_path (str): Path to prototype file (.pt)
            normalize (bool): Whether to L2-normalize the prototype
            verbose (bool): Whether to print loading information

        Returns:
            prototype (torch.Tensor): Prototype feature vector (C,)

        Raises:
            FileNotFoundError: If prototype file doesn't exist
            KeyError: If prototype file doesn't contain 'prototype' key
        """
        # Expand user path
        prototype_path = os.path.expanduser(prototype_path)

        if not os.path.exists(prototype_path):
            raise FileNotFoundError(f'Prototype not found: {prototype_path}')

        # Load checkpoint
        # Note: weights_only=False is needed for prototypes that contain metadata dicts
        # This is safe as we trust prototype files from our own system
        checkpoint = torch.load(prototype_path, map_location=self.device, weights_only=False)

        # Extract prototype
        if 'prototype' not in checkpoint:
            raise KeyError(f"Prototype file must contain 'prototype' key, got: {checkpoint.keys()}")

        prototype = checkpoint['prototype'].to(self.device)

        # Ensure prototype is 1D vector
        if prototype.dim() > 1:
            prototype = prototype.squeeze()

        # L2 normalize
        if normalize:
            prototype = F.normalize(prototype, p=2, dim=0)

        if verbose:
            print(f'Loaded prototype from: {prototype_path}')
            print(f'  Shape: {prototype.shape}')
            print(f'  Norm: {torch.norm(prototype).item():.4f}')
            print(f'  Device: {prototype.device}')

        # Store as current prototype
        self.prototype = prototype

        return prototype

    def save(self, prototype, save_path, metadata=None, normalize=True, verbose=True):
        """
        Save prototype to file.

        Args:
            prototype (torch.Tensor): Prototype feature vector (C,)
            save_path (str): Path to save prototype (.pt)
            metadata (dict): Optional metadata to save with prototype
            normalize (bool): Whether to L2-normalize before saving
            verbose (bool): Whether to print save confirmation
        """
        # Expand user path
        save_path = os.path.expanduser(save_path)

        # Create directory if needed
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Ensure prototype is 1D vector
        if prototype.dim() > 1:
            prototype = prototype.squeeze()

        # L2 normalize
        if normalize:
            prototype = F.normalize(prototype, p=2, dim=0)

        # Prepare checkpoint
        checkpoint = {
            'prototype': prototype.cpu(),  # Save on CPU for portability
        }

        # Add metadata if provided
        if metadata is not None:
            checkpoint['metadata'] = metadata

        # Save
        torch.save(checkpoint, save_path)

        if verbose:
            print(f'Saved prototype to: {save_path}')

    def validate(self, prototype, expected_dim=None):
        """
        Validate prototype format and properties.

        Args:
            prototype (torch.Tensor): Prototype to validate
            expected_dim (int): Expected feature dimension (optional)

        Returns:
            bool: True if valid

        Raises:
            ValueError: If prototype is invalid
        """
        # Check type
        if not isinstance(prototype, torch.Tensor):
            raise ValueError(f'Prototype must be torch.Tensor, got {type(prototype)}')

        # Check dimensions
        if prototype.dim() != 1:
            raise ValueError(f'Prototype must be 1D vector, got shape {prototype.shape}')

        # Check expected dimension
        if expected_dim is not None and prototype.shape[0] != expected_dim:
            raise ValueError(
                f'Prototype dimension mismatch: expected {expected_dim}, got {prototype.shape[0]}'
            )

        # Check for NaN or Inf
        if torch.isnan(prototype).any():
            raise ValueError('Prototype contains NaN values')
        if torch.isinf(prototype).any():
            raise ValueError('Prototype contains Inf values')

        # Check norm (should be close to 1 if normalized)
        norm = torch.norm(prototype).item()
        if abs(norm - 1.0) > 0.1:
            print(f'Warning: Prototype norm is {norm:.4f}, expected ~1.0 (not normalized?)')

        return True

    def compute_from_features(self, features, method='mean', normalize=True):
        """
        Compute prototype from a collection of features.

        Args:
            features (torch.Tensor): Feature vectors (N, C) or (N, H, W, C)
            method (str): Aggregation method ('mean', 'median')
            normalize (bool): Whether to L2-normalize the result

        Returns:
            prototype (torch.Tensor): Computed prototype (C,)
        """
        # Flatten spatial dimensions if present
        if features.dim() == 4:  # (N, H, W, C)
            features = features.reshape(-1, features.shape[-1])  # (N*H*W, C)
        elif features.dim() == 3:  # (N, H, C)
            features = features.reshape(-1, features.shape[-1])  # (N*H, C)

        # Compute prototype
        if method == 'mean':
            prototype = features.mean(dim=0)
        elif method == 'median':
            prototype = features.median(dim=0)[0]
        else:
            raise ValueError(f"Unknown method: {method}, choose 'mean' or 'median'")

        # Normalize
        if normalize:
            prototype = F.normalize(prototype, p=2, dim=0)

        return prototype

    def load_multiple(self, prototype_dir, class_names=None):
        """
        Load multiple prototypes from a directory.

        Args:
            prototype_dir (str): Directory containing prototype files
            class_names (list): List of class names to load (default: all .pt files)

        Returns:
            prototypes (dict): Dictionary mapping class names to prototype tensors
        """
        prototype_dir = os.path.expanduser(prototype_dir)

        if not os.path.isdir(prototype_dir):
            raise NotADirectoryError(f'Not a directory: {prototype_dir}')

        prototypes = {}

        # Get list of prototype files
        if class_names is None:
            # Load all .pt files
            files = [f for f in os.listdir(prototype_dir) if f.endswith('.pt')]
            class_names = [os.path.splitext(f)[0] for f in files]
        else:
            files = [f'{name}.pt' for name in class_names]

        # Load each prototype
        for class_name, filename in zip(class_names, files):
            filepath = os.path.join(prototype_dir, filename)
            if os.path.exists(filepath):
                try:
                    prototype = self.load(filepath, verbose=False)
                    prototypes[class_name] = prototype
                except Exception as e:
                    print(f'Warning: Failed to load {class_name}: {e}')
            else:
                print(f'Warning: Prototype file not found: {filepath}')

        print(f'Loaded {len(prototypes)} prototypes from {prototype_dir}')
        return prototypes

    def update_with_momentum(self, pos_features, momentum=0.9):
        """
        Update prototype using exponential moving average with positive sample features.

        This method implements online prototype adaptation using a momentum-based update:
        prototype_new = momentum * prototype_old + (1 - momentum) * prototype_current

        Args:
            pos_features (torch.Tensor): Positive sample features (N, C) from footprint regions
            momentum (float): Momentum coefficient [0, 1]. Higher values = slower adaptation.
                             Default 0.9 means 90% old + 10% new.

        Returns:
            torch.Tensor: Updated prototype (C,)

        Raises:
            ValueError: If prototype hasn't been initialized (call load() first)
            ValueError: If pos_features is empty or invalid
        """
        if self.prototype is None:
            raise ValueError('Prototype not initialized. Call load() first.')

        if not isinstance(pos_features, torch.Tensor):
            raise ValueError(f'pos_features must be torch.Tensor, got {type(pos_features)}')

        if pos_features.dim() != 2:
            raise ValueError(f'pos_features must be 2D (N, C), got shape {pos_features.shape}')

        if pos_features.shape[0] == 0:
            raise ValueError('pos_features is empty (no positive samples)')

        # Check feature dimension matches prototype
        if pos_features.shape[1] != self.prototype.shape[0]:
            raise ValueError(
                f'Feature dimension mismatch: pos_features has {pos_features.shape[1]}, '
                f'prototype has {self.prototype.shape[0]}'
            )

        # Move to same device as prototype
        pos_features = pos_features.to(self.prototype.device)

        # Compute current prototype from positive features (mean aggregation)
        current_prototype = torch.mean(pos_features, dim=0)  # (C,)
        current_prototype = F.normalize(current_prototype, p=2, dim=0)

        # Update with momentum (exponential moving average)
        with torch.no_grad():
            self.prototype = momentum * self.prototype + (1 - momentum) * current_prototype
            self.prototype = F.normalize(self.prototype, p=2, dim=0)

        return self.prototype

    def get_prototype(self):
        """
        Get current prototype.

        Returns:
            torch.Tensor: Current prototype (C,)

        Raises:
            ValueError: If prototype hasn't been initialized
        """
        if self.prototype is None:
            raise ValueError('Prototype not initialized. Call load() first.')
        return self.prototype

    def set_prototype(self, prototype, normalize=True):
        """
        Set prototype directly (for online updates from external source).

        Args:
            prototype (torch.Tensor): Prototype feature vector (C,)
            normalize (bool): Whether to L2-normalize

        Returns:
            torch.Tensor: Set prototype
        """
        # Ensure tensor
        if not isinstance(prototype, torch.Tensor):
            raise ValueError(f'Prototype must be torch.Tensor, got {type(prototype)}')

        # Ensure 1D
        if prototype.dim() > 1:
            prototype = prototype.squeeze()

        if prototype.dim() != 1:
            raise ValueError(f'Prototype must be 1D vector, got shape {prototype.shape}')

        # Move to device
        prototype = prototype.to(self.device)

        # Normalize
        if normalize:
            prototype = F.normalize(prototype, p=2, dim=0)

        self.prototype = prototype
        return self.prototype
