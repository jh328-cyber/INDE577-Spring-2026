"""
ml_toolkit: A from-scratch machine learning library
====================================================

This package contains classical supervised and unsupervised learning
algorithms, all implemented from scratch in NumPy for educational purposes.

Modules
-------
supervised : supervised learning algorithms
unsupervised : unsupervised learning algorithms
metrics : evaluation metrics for regression, classification, and clustering
utils : data utilities (train/test split, scaling, etc.)
"""

from ml_toolkit import metrics, supervised, unsupervised, utils

__version__ = "0.1.0"
__all__ = ["supervised", "unsupervised", "metrics", "utils"]
