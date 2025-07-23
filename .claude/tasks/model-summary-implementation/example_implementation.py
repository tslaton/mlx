"""
Example implementation of model summary for MLX
This is a simplified version to demonstrate the approach
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import mlx.core as mx
from mlx.nn import Module
from mlx.utils import tree_flatten


@dataclass
class LayerInfo:
    """Information about a single layer"""
    name: str
    module_type: str
    num_params: int = 0
    trainable_params: int = 0
    param_bytes: int = 0
    output_shape: Optional[Tuple[int, ...]] = None


def count_parameters(module: Module) -> Tuple[int, int, int]:
    """
    Count total parameters, trainable parameters, and total bytes.
    
    Returns:
        (total_params, trainable_params, total_bytes)
    """
    total_params = 0
    trainable_params = 0
    total_bytes = 0
    
    # Get all parameters as a flat list
    params = tree_flatten(module.parameters())
    
    for name, param in params:
        if isinstance(param, mx.array):
            param_count = param.size
            param_bytes = param.nbytes
            
            total_params += param_count
            total_bytes += param_bytes
            
            # Check if parameter is trainable (not in no_grad set)
            # Note: This is a simplified check - in real implementation
            # we'd need to track which module the parameter belongs to
            trainable_params += param_count
    
    return total_params, trainable_params, total_bytes


def collect_layer_info(model: Module) -> List[LayerInfo]:
    """Collect information about each layer in the model."""
    layer_infos = []
    
    # Traverse all modules
    for name, module in model.named_modules():
        # Skip the root module
        if module is model:
            continue
        
        # Count only direct parameters of this module
        module_params = 0
        module_trainable = 0
        module_bytes = 0
        
        # Check direct attributes of the module
        for key, value in module.items():
            if isinstance(value, mx.array):
                module_params += value.size
                module_bytes += value.nbytes
                # Simplified: assume all are trainable
                module_trainable += value.size
        
        layer_info = LayerInfo(
            name=name,
            module_type=type(module).__name__,
            num_params=module_params,
            trainable_params=module_trainable,
            param_bytes=module_bytes
        )
        layer_infos.append(layer_info)
    
    return layer_infos


def format_summary(model: Module, layer_infos: List[LayerInfo]) -> str:
    """Format the model summary as a string."""
    lines = []
    
    # Header
    lines.append("=" * 70)
    lines.append(f"{'Layer (type)':<40} {'Param #':<15} {'Trainable':<15}")
    lines.append("=" * 70)
    
    # Layer information
    for info in layer_infos:
        layer_str = f"{info.name} ({info.module_type})"
        if len(layer_str) > 39:
            layer_str = layer_str[:36] + "..."
        
        lines.append(
            f"{layer_str:<40} "
            f"{info.num_params:>14,} "
            f"{info.trainable_params:>14,}"
        )
    
    lines.append("=" * 70)
    
    # Total parameters
    total_params, trainable_params, total_bytes = count_parameters(model)
    non_trainable = total_params - trainable_params
    
    lines.append(f"Total params: {total_params:,}")
    lines.append(f"Trainable params: {trainable_params:,}")
    lines.append(f"Non-trainable params: {non_trainable:,}")
    lines.append("-" * 70)
    
    # Memory usage
    params_mb = total_bytes / (1024 * 1024)
    lines.append(f"Params size (MB): {params_mb:.2f}")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def summary(model: Module, 
           input_shape: Optional[Tuple[int, ...]] = None,
           verbose: int = 1) -> str:
    """
    Generate a summary of the model.
    
    Args:
        model: The MLX model to summarize
        input_shape: Input shape for the model (currently unused)
        verbose: Verbosity level (currently unused)
        
    Returns:
        String containing the model summary
    """
    # Collect layer information
    layer_infos = collect_layer_info(model)
    
    # Format and return summary
    return format_summary(model, layer_infos)


# Example usage
if __name__ == "__main__":
    import mlx.nn as nn
    
    # Example 1: Simple linear model
    print("Example 1: Simple Linear Model")
    model = nn.Linear(10, 5)
    print(summary(model))
    print()
    
    # Example 2: Sequential model
    print("Example 2: Sequential Model")
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    print(summary(model))
    print()
    
    # Example 3: Model with convolutions
    print("Example 3: CNN Model")
    
    class SimpleCNN(Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(128, 10)
            
        def __call__(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = x.reshape(x.shape[0], -1)
            x = self.relu3(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = SimpleCNN()
    print(summary(model))