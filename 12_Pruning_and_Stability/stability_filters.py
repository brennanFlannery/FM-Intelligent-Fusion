"""
Stability Filters for CLAM.

This module provides the StabilityFilter class for computing feature masks
and weights based on stability metrics. This is re-exported from
dataset_modules.dataset_stability for convenience.
"""

from dataset_modules.dataset_stability import StabilityFilter

__all__ = ['StabilityFilter']
