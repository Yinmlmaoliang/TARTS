#!/usr/bin/env python3
"""
Footprint projection utilities for TARTS online prototype update.

This package contains tools for projecting robot footprints onto image planes
using camera intrinsics and extrinsics.
"""

from .image_projector import ImageProjector
from .node import BaseNode, DataNode
from .utils import (
    make_box,
    make_plane,
    make_polygon_from_points,
    make_dense_plane,
)

__all__ = [
    'ImageProjector',
    'BaseNode',
    'DataNode',
    'make_box',
    'make_plane',
    'make_polygon_from_points',
    'make_dense_plane',
]
