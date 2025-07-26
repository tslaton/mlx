# Copyright © 2023-2025 Apple Inc.

"""
mlx.metrics
===========

The metrics module provides evaluation functions for machine learning models.

This module includes functions to compute various metrics commonly used in
machine learning tasks, including classification and regression metrics.

Classification Metrics
----------------------
accuracy : Compute accuracy score
precision : Compute precision score
recall : Compute recall score
f1_score : Compute F1 score

Regression Metrics
------------------
mean_squared_error : Compute mean squared error
mean_absolute_error : Compute mean absolute error
r2_score : Compute R-squared score
"""

__version__ = "0.1.0"

# Import from submodules
from .classification import accuracy, precision, recall, f1_score
from .regression import mean_squared_error, mean_absolute_error, r2_score

__all__ = [
    # Classification metrics
    "accuracy",
    "precision", 
    "recall",
    "f1_score",
    # Regression metrics
    "mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    # Module version
    "__version__",
]