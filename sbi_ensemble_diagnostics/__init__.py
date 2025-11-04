"""
SBI Ensemble Diagnostics

A library for assessing the reliability of Simulation-Based Inference (SBI)
using ensemble learning and KL divergence metrics.

This package implements the methods described in:
"Simulation-based inference with deep ensembles: Evaluating calibration
uncertainty and detecting model misspecification"
by Alvey, Contaldi, and Pieroni (2025)
"""

from .utils import (
    KLval,
    NPEData,
    build_density_estimator,
    EmbeddingNet,
    setup_scheduler,
    plot_losses,
)

__version__ = "1.0.0"
__all__ = [
    "KLval",
    "NPEData",
    "build_density_estimator",
    "EmbeddingNet",
    "setup_scheduler",
    "plot_losses",
]
