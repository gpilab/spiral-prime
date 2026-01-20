"""
Spiral-PRIME: Physics-based Reconstruction with Iterative Model-based Enhancement

A unified 3D reconstruction pipeline for GPI framework designed to reconstruct
high-resolution, multi-contrast brain volumes from steady-state localized-quadratic
(LQ) spin-echo acquisitions.
"""

__version__ = "0.1.0"
__author__ = "GPI Lab"

from . import core
from . import nodes
from . import utils

__all__ = ["core", "nodes", "utils", "__version__"]
