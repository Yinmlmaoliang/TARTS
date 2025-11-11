#!/usr/bin/env python3
"""
TARTS Prototype Registration Tool

Registers object prototypes by extracting DINO features from reference images.
Saves prototypes to ~/.ros/tarts_prototypes/ for use by segmentation node.
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Import TARTS core components
from tarts_core.models.dino import Dinov3ViT
from tarts_core.prototype.manager import PrototypeManager
from tarts_core.utils.image import load_image


def main(args=None):
    """Main function for standalone usage."""
    parser = argparse.ArgumentParser(description='Register object prototype from image')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to reference image')
    parser.add_argument('--class_name', type=str, required=True,
                       help='Name of the object class')
    parser.add_argument('--output_dir', type=str,
                       default=os.path.expanduser('~/.ros/tarts_prototypes'),
                       help='Directory to save prototypes')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for feature extraction')
    parser.add_argument('--backbone', type=str, default='vits16',
                       choices=['vits16', 'vits16plus', 'vitb16', 'vitl16'],
                       help='DINO backbone type')
    parser.add_argument('--input_size', type=int, default=480,
                       help='Input image size')

    parsed_args = parser.parse_args()

    try:
        # Validate image path
        if not os.path.exists(parsed_args.image):
            print(f'Error: Image file not found: {parsed_args.image}')
            return 1

        # Validate class name
        if not parsed_args.class_name or not parsed_args.class_name.replace('_', '').isalnum():
            print(f'Error: Invalid class name: {parsed_args.class_name}')
            print('Class name should only contain letters, numbers, and underscores')
            return 1

        print('=' * 60)
        print('TARTS Prototype Registration')
        print('=' * 60)
        print(f'Image: {parsed_args.image}')
        print(f'Class name: {parsed_args.class_name}')
        print(f'Output directory: {parsed_args.output_dir}')
        print(f'Device: {parsed_args.device}')
        print(f'Backbone: {parsed_args.backbone}')
        print('=' * 60)

        # Check device availability
        device = parsed_args.device
        if device == 'cuda' and not torch.cuda.is_available():
            print('Warning: CUDA not available, falling back to CPU')
            device = 'cpu'

        # Initialize DINO model
        print(f'\nLoading DINOv3 model: {parsed_args.backbone}')
        dino_model = Dinov3ViT(
            backbone_type=parsed_args.backbone,
            dropout_p=0.0,
            device=device
        )
        dino_model.eval()
        print(f'Model loaded: {dino_model.get_backbone_name()}')
        print(f'Feature dimension: {dino_model.get_output_feat_dim()}')

        # Load and preprocess image
        print(f'\nLoading image: {parsed_args.image}')
        img = load_image(parsed_args.image, color_space='RGB')

        # Convert to PIL Image and apply transforms
        pil_img = Image.fromarray(img)
        transform = transforms.Compose([
            transforms.Resize(
                size=(parsed_args.input_size, parsed_args.input_size),
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        img_tensor = transform(pil_img).unsqueeze(0).to(device)

        # Extract DINO features
        print('Extracting prototype features...')
        with torch.no_grad():
            features = dino_model(img_tensor)  # (1, H_feat, W_feat, C)

        # Global average pooling over spatial dimensions
        prototype = features.mean(dim=[1, 2]).squeeze(0)  # (C,)

        # L2 normalize
        prototype = F.normalize(prototype, p=2, dim=0)

        print(f'Extracted prototype shape: {prototype.shape}')
        print(f'Prototype norm: {torch.norm(prototype).item():.4f}')

        # Save prototype using PrototypeManager
        print('\nSaving prototype...')
        prototype_manager = PrototypeManager(device=device)

        output_path = os.path.join(parsed_args.output_dir, f'{parsed_args.class_name}.pt')
        metadata = {
            'class_name': parsed_args.class_name,
            'feature_dim': dino_model.get_output_feat_dim(),
            'backbone': dino_model.get_backbone_name(),
            'patch_size': dino_model.patch_size,
        }

        prototype_manager.save(
            prototype,
            output_path,
            metadata=metadata,
            normalize=False  # Already normalized
        )

        print('\n' + '=' * 60)
        print('Registration completed successfully!')
        print('=' * 60)
        print(f'\nTo use this prototype, run segmentation with:')
        print(f'  class_name:={parsed_args.class_name}')
        print('')

        return 0

    except Exception as e:
        print(f'\nError during registration: {str(e)}')
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
