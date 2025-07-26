# Copyright © 2023-2025 Apple Inc.

"""Test the metrics module structure and imports."""

import unittest
import sys
import os

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestMetricsStructure(unittest.TestCase):
    """Test that the metrics module is properly structured."""

    def test_module_import(self):
        """Test that mlx.metrics can be imported."""
        import mlx.metrics
        
    def test_has_version(self):
        """Test that the module has a version."""
        import mlx.metrics
        self.assertTrue(hasattr(mlx.metrics, '__version__'))
        self.assertIsInstance(mlx.metrics.__version__, str)
        
    def test_classification_metrics_exist(self):
        """Test that classification metrics are exposed."""
        import mlx.metrics
        self.assertTrue(hasattr(mlx.metrics, 'accuracy'))
        self.assertTrue(hasattr(mlx.metrics, 'precision'))
        self.assertTrue(hasattr(mlx.metrics, 'recall'))
        self.assertTrue(hasattr(mlx.metrics, 'f1_score'))
        
    def test_regression_metrics_exist(self):
        """Test that regression metrics are exposed."""
        import mlx.metrics
        self.assertTrue(hasattr(mlx.metrics, 'mean_squared_error'))
        self.assertTrue(hasattr(mlx.metrics, 'mean_absolute_error'))
        self.assertTrue(hasattr(mlx.metrics, 'r2_score'))
        
    def test_all_exports(self):
        """Test that __all__ is properly defined."""
        import mlx.metrics
        expected = {
            'accuracy', 'precision', 'recall', 'f1_score',
            'mean_squared_error', 'mean_absolute_error', 'r2_score',
            '__version__'
        }
        self.assertEqual(set(mlx.metrics.__all__), expected)


if __name__ == '__main__':
    unittest.main()