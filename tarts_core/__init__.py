"""
TARTS Core - Template-Assisted Reference-based Target Segmentation

Core algorithms for reference-based semantic segmentation, independent of ROS2.
"""

__version__ = "0.1.0"

from tarts_core.tarts.engine import SegmentationEngine
from tarts_core.prototype.manager import PrototypeManager

__all__ = [
    'SegmentationEngine',
    'PrototypeManager',
]
