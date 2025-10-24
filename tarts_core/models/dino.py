#!/usr/bin/env python3
"""
DINOv3 Vision Transformer for feature extraction.
"""

import torch
import torch.nn as nn
import numpy as np


class Dinov3ViT(nn.Module):
    """DINOv3 Vision Transformer backbone."""

    def __init__(self, backbone_type='vits16', dropout_p=0.0, device='cuda'):
        super().__init__()
        self.backbone_type = backbone_type
        self.patch_size = 16
        self.dropout_p = dropout_p
        self.device = device

        URLS = {
            "dinov3_vits16": "https://huggingface.co/huzey/mydv3/resolve/master/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
            "dinov3_vits16plus": "https://huggingface.co/huzey/mydv3/resolve/master/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
            "dinov3_vitb16": "https://huggingface.co/huzey/mydv3/resolve/master/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
            "dinov3_vitl16": "https://huggingface.co/huzey/mydv3/resolve/master/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        }

        if self.backbone_type == "vitb16":
            model_name = "dinov3_vitb16"
            self.n_feats = 768
        elif self.backbone_type == "vits16":
            model_name = "dinov3_vits16"
            self.n_feats = 384
        elif self.backbone_type == "vits16plus":
            model_name = "dinov3_vits16plus"
            self.n_feats = 384
        elif self.backbone_type == "vitl16":
            model_name = "dinov3_vitl16"
            self.n_feats = 1024
        else:
            raise ValueError(f"Model type {backbone_type} unavailable for DINOv3")

        self.model = torch.hub.load("facebookresearch/dinov3", model_name,
                                     weights=URLS[model_name])

        # Move model to device
        self.model = self.model.to(self.device)

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        self.dropout = torch.nn.Dropout2d(p=np.clip(self.dropout_p, 0.0, 1.0))

    def get_output_feat_dim(self):
        return self.n_feats

    @torch.no_grad()
    def forward(self, img):
        self.model.eval()
        assert img.shape[2] % self.patch_size == 0
        assert img.shape[3] % self.patch_size == 0

        feat = self.model.get_intermediate_layers(img)[0]

        feat_h = img.shape[2] // self.patch_size
        feat_w = img.shape[3] // self.patch_size

        image_feat = feat.reshape(feat.shape[0], feat_h, feat_w, -1)

        if self.dropout_p > 0:
            return self.dropout(image_feat)
        else:
            return image_feat

    def get_backbone_name(self):
        return f"DINOv3-{self.backbone_type}-{self.patch_size}"
