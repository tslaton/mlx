# Copyright © 2023 Apple Inc.

import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import (
    LayerInfo,
    collect_layer_info,
    count_parameters,
    format_layer_summary,
    simple_summary,
    summary,
)


class TestModelSummary(unittest.TestCase):
    def test_count_parameters(self):
        """Test parameter counting for various models."""
        # Simple linear layer
        model = nn.Linear(10, 5)
        total, trainable = count_parameters(model)
        self.assertEqual(total, 55)  # 10*5 + 5
        self.assertEqual(trainable, 55)

        # Model with no parameters
        model = nn.ReLU()
        total, trainable = count_parameters(model)
        self.assertEqual(total, 0)
        self.assertEqual(trainable, 0)

    def test_frozen_parameters(self):
        """Test counting with frozen parameters."""
        model = nn.Linear(10, 5)
        model.freeze()
        total, trainable = count_parameters(model)
        self.assertEqual(total, 55)
        self.assertEqual(trainable, 0)

    def test_simple_summary(self):
        """Test simple summary output."""
        model = nn.Linear(10, 5)
        summary_str = simple_summary(model)
        self.assertIn("Total params: 55", summary_str)
        self.assertIn("Trainable params: 55", summary_str)
        self.assertIn("Non-trainable params: 0", summary_str)

    def test_collect_layer_info(self):
        """Test layer information collection."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        layer_infos = collect_layer_info(model)
        
        # Should have 3 layers
        self.assertEqual(len(layer_infos), 3)
        
        # Check first linear layer
        linear_layers = [info for info in layer_infos if info.module_type == "Linear"]
        self.assertEqual(len(linear_layers), 2)
        
        # Check ReLU layer
        relu_layers = [info for info in layer_infos if info.module_type == "ReLU"]
        self.assertEqual(len(relu_layers), 1)
        self.assertEqual(relu_layers[0].num_params, 0)

    def test_layer_info_dataclass(self):
        """Test LayerInfo dataclass."""
        info = LayerInfo(
            name="test_layer",
            module_type="Linear",
            num_params=100,
            trainable_params=100,
            param_bytes=400
        )
        self.assertEqual(info.name, "test_layer")
        self.assertEqual(info.module_type, "Linear")
        self.assertEqual(info.num_params, 100)
        self.assertEqual(info.output_shape, None)

    def test_format_layer_summary(self):
        """Test table formatting."""
        layer_infos = [
            LayerInfo("layer1", "Linear", num_params=100, trainable_params=100),
            LayerInfo("layer2", "ReLU", num_params=0, trainable_params=0),
        ]
        
        table = format_layer_summary(layer_infos)
        self.assertIn("Layer (type)", table)
        self.assertIn("Param #", table)
        self.assertIn("Trainable", table)
        self.assertIn("layer1 (Linear)", table)
        self.assertIn("100", table)

    def test_summary_verbose_levels(self):
        """Test different verbose levels."""
        model = nn.Linear(10, 5)
        
        # Verbose 0 - simple summary
        summary0 = summary(model, verbose=0)
        self.assertIn("Total params:", summary0)
        self.assertNotIn("Layer (type)", summary0)
        
        # Verbose 1 - detailed summary
        summary1 = summary(model, verbose=1)
        self.assertIn("Total params:", summary1)
        self.assertIn("Params size (MB):", summary1)

    def test_sequential_model_summary(self):
        """Test summary of sequential models."""
        model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        summary_str = summary(model)
        self.assertIn("101,770", summary_str)  # Total params
        self.assertIn("Linear", summary_str)
        self.assertIn("ReLU", summary_str)

    def test_conv_model_summary(self):
        """Test summary with convolutional layers."""
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3)
        )
        
        summary_str = summary(model)
        self.assertIn("Conv2d", summary_str)
        
        # Check parameter counts
        # First conv: 3*32*3*3 + 32 = 896
        # Second conv: 32*64*3*3 + 64 = 18496
        # Total: 19392
        self.assertIn("19,392", summary_str)

    def test_model_summary_method(self):
        """Test the model.summary() method."""
        model = nn.Linear(10, 5)
        summary_str = model.summary()
        self.assertIn("Total params: 55", summary_str)

    def test_empty_model(self):
        """Test summary of model with no parameters."""
        model = nn.Sequential()
        summary_str = summary(model)
        self.assertIn("Total params: 0", summary_str)

    def test_nested_modules(self):
        """Test summary with nested module structures."""
        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block1 = nn.Sequential(
                    nn.Linear(10, 20),
                    nn.ReLU()
                )
                self.block2 = nn.Linear(20, 5)
                
            def __call__(self, x):
                x = self.block1(x)
                return self.block2(x)
        
        model = NestedModel()
        summary_str = summary(model)
        
        # Should show all layers including nested ones
        self.assertIn("Linear", summary_str)
        # Total params: 10*20+20 + 20*5+5 = 325
        self.assertIn("325", summary_str)


if __name__ == "__main__":
    unittest.main()