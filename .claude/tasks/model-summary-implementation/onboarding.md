# Model Summary Implementation Onboarding

## Task Overview
Implement PyTorch-like model summary tools for MLX that show layer information, parameter counts, and memory usage. This feature will help users debug models, understand architecture, and track memory consumption.

## Background Research

### What PyTorch Model Summary Shows
Based on research of PyTorch's torchinfo (successor to torchsummary):
- Layer names and types
- Output shapes for each layer
- Number of parameters (trainable/non-trainable)
- Total parameter count
- Memory usage estimates
- Computational complexity (MACs/FLOPs)

### MLX Architecture Understanding

#### Module System
- `mlx.nn.Module` is a dict subclass that stores parameters
- Key methods:
  - `parameters()` - returns all arrays recursively
  - `trainable_parameters()` - returns non-frozen arrays
  - `named_modules()` - returns (name, module) pairs
  - `modules()` - returns list of all modules
  - `_extra_repr()` - custom string representation for each layer

#### Array Properties
- `shape` - tuple of dimensions
- `dtype` - data type (float32, int32, etc.)
- `itemsize` - bytes per element
- `nbytes` - total bytes (size * itemsize)
- `size` - total number of elements

#### Memory Functions
- `mlx.core.get_active_memory()` - current memory usage
- `mlx.core.get_peak_memory()` - peak memory usage
- `mlx.core.get_cache_memory()` - cached memory

## Design Architecture

### Core Components

1. **Summary Data Structure**
   ```python
   @dataclass
   class LayerInfo:
       name: str
       module_type: str
       output_shape: Optional[Tuple[int, ...]]
       num_params: int
       trainable_params: int
       param_bytes: int
   ```

2. **Summary Generator Class**
   ```python
   class ModelSummary:
       def __init__(self, model: Module, input_shape: Tuple[int, ...])
       def _register_hooks(self)
       def _forward_pass(self)
       def _collect_parameters(self)
       def _format_summary(self) -> str
       def __str__(self) -> str
   ```

3. **Public API Function**
   ```python
   def summary(model: Module, 
              input_shape: Union[Tuple[int, ...], mx.array],
              batch_size: int = -1,
              device: mx.Device = mx.default_device(),
              dtypes: Optional[List[mx.Dtype]] = None) -> ModelSummary
   ```

## Step-by-Step Implementation Guide

### Phase 1: Basic Parameter Counting (2-3 hours)

#### Step 1.1: Create the base module structure
```python
# File: python/mlx/nn/utils/summary.py
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import mlx.core as mx
from mlx.nn import Module
```

#### Step 1.2: Implement parameter counting
```python
def count_parameters(module: Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total_params = 0
    trainable_params = 0
    
    for name, param in tree_flatten(module.parameters()):
        if isinstance(param, mx.array):
            param_count = param.size
            total_params += param_count
            if name not in module._no_grad:
                trainable_params += param_count
    
    return total_params, trainable_params
```

#### Step 1.3: Create simple summary function
```python
def simple_summary(model: Module) -> str:
    """Generate a simple parameter count summary."""
    total, trainable = count_parameters(model)
    non_trainable = total - trainable
    
    return f"""
Total params: {total:,}
Trainable params: {trainable:,}
Non-trainable params: {non_trainable:,}
    """
```

### Phase 2: Layer-by-Layer Summary (3-4 hours)

#### Step 2.1: Create LayerInfo dataclass
```python
@dataclass
class LayerInfo:
    name: str
    module_type: str
    output_shape: Optional[Tuple[int, ...]] = None
    num_params: int = 0
    trainable_params: int = 0
    param_bytes: int = 0
```

#### Step 2.2: Implement layer traversal
```python
def collect_layer_info(model: Module) -> List[LayerInfo]:
    """Collect information about each layer."""
    layer_infos = []
    
    for name, module in model.named_modules():
        if module is model:
            continue  # Skip the root module
            
        # Count parameters for this specific module
        total_params = 0
        trainable_params = 0
        param_bytes = 0
        
        # Only count direct parameters, not nested ones
        for param_name, param in module.items():
            if isinstance(param, mx.array) and not isinstance(param, Module):
                total_params += param.size
                param_bytes += param.nbytes
                if param_name not in module._no_grad:
                    trainable_params += param.size
        
        layer_info = LayerInfo(
            name=name,
            module_type=type(module).__name__,
            num_params=total_params,
            trainable_params=trainable_params,
            param_bytes=param_bytes
        )
        layer_infos.append(layer_info)
    
    return layer_infos
```

#### Step 2.3: Format the output
```python
def format_layer_summary(layer_infos: List[LayerInfo]) -> str:
    """Format layer information as a table."""
    # Table headers
    headers = ["Layer (type)", "Param #", "Trainable"]
    col_widths = [40, 15, 15]
    
    # Build table
    lines = []
    lines.append("=" * sum(col_widths))
    lines.append("".join(h.ljust(w) for h, w in zip(headers, col_widths)))
    lines.append("=" * sum(col_widths))
    
    for info in layer_infos:
        layer_str = f"{info.name} ({info.module_type})"
        param_str = f"{info.num_params:,}"
        trainable_str = f"{info.trainable_params:,}"
        
        lines.append("".join([
            layer_str[:col_widths[0]].ljust(col_widths[0]),
            param_str.ljust(col_widths[1]),
            trainable_str.ljust(col_widths[2])
        ]))
    
    lines.append("=" * sum(col_widths))
    return "\n".join(lines)
```

### Phase 3: Output Shape Tracking (4-5 hours)

#### Step 3.1: Implement forward hooks
```python
class ModelSummary:
    def __init__(self, model: Module):
        self.model = model
        self.layer_infos: Dict[str, LayerInfo] = {}
        self._hooks = []
        
    def register_forward_hook(self, name: str, module: Module):
        """Register a hook to capture output shapes."""
        def hook(module, inputs, outputs):
            # Handle different output types
            if isinstance(outputs, mx.array):
                output_shape = outputs.shape
            elif isinstance(outputs, (tuple, list)):
                output_shape = tuple(o.shape if isinstance(o, mx.array) else None 
                                   for o in outputs)
            else:
                output_shape = None
                
            if name in self.layer_infos:
                self.layer_infos[name].output_shape = output_shape
                
        self._hooks.append(module.register_forward_hook(hook))
```

#### Step 3.2: Execute forward pass
```python
def forward_pass(self, input_data: Union[mx.array, Tuple[int, ...]]):
    """Run a forward pass to collect output shapes."""
    # Create dummy input if shape is provided
    if isinstance(input_data, tuple):
        dummy_input = mx.zeros(input_data)
    else:
        dummy_input = input_data
        
    # Register hooks for all modules
    for name, module in self.model.named_modules():
        self.register_forward_hook(name, module)
    
    # Run forward pass
    try:
        with mx.no_grad():
            _ = self.model(dummy_input)
    except Exception as e:
        print(f"Warning: Forward pass failed: {e}")
    finally:
        # Clean up hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
```

### Phase 4: Memory Usage Calculation (2-3 hours)

#### Step 4.1: Add memory tracking
```python
def calculate_memory_usage(layer_infos: List[LayerInfo], 
                         batch_size: int = 1) -> Dict[str, int]:
    """Calculate memory usage statistics."""
    total_params_mb = sum(info.param_bytes for info in layer_infos) / (1024 * 1024)
    
    # Estimate activation memory (if output shapes are available)
    total_output_size = 0
    for info in layer_infos:
        if info.output_shape:
            # Assume float32 for activations
            elements = batch_size
            for dim in info.output_shape[1:]:  # Skip batch dimension
                elements *= dim
            total_output_size += elements * 4  # 4 bytes for float32
    
    total_activation_mb = total_output_size / (1024 * 1024)
    
    return {
        "params_mb": total_params_mb,
        "activations_mb": total_activation_mb,
        "total_mb": total_params_mb + total_activation_mb
    }
```

### Phase 5: Complete Implementation (3-4 hours)

#### Step 5.1: Create the main summary function
```python
def summary(model: Module,
           input_shape: Optional[Union[Tuple[int, ...], mx.array]] = None,
           batch_size: int = 1,
           device: Optional[mx.Device] = None,
           verbose: int = 1) -> str:
    """
    Generate a summary of the model.
    
    Args:
        model: The MLX model to summarize
        input_shape: Input shape (excluding batch dimension) or sample input
        batch_size: Batch size for memory calculations
        device: Device to run the model on
        verbose: 0=silent, 1=default, 2=detailed
        
    Returns:
        Summary string
    """
    summary_obj = ModelSummary(model)
    
    # Collect basic layer information
    layer_infos = collect_layer_info(model)
    
    # If input shape provided, run forward pass
    if input_shape is not None:
        if isinstance(input_shape, tuple):
            input_shape = (batch_size,) + input_shape
        summary_obj.forward_pass(input_shape)
        
    # Generate summary string
    return summary_obj.generate_summary(layer_infos, batch_size, verbose)
```

#### Step 5.2: Integration with nn module
```python
# In python/mlx/nn/__init__.py
from .utils.summary import summary

# In python/mlx/nn/layers/base.py
def summary(self, input_shape=None, batch_size=1, verbose=1):
    """Generate a summary of this module."""
    from ..utils.summary import summary
    return summary(self, input_shape, batch_size, verbose=verbose)
```

### Phase 6: Testing and Documentation (2-3 hours)

#### Step 6.1: Write unit tests
```python
# File: python/tests/test_nn_summary.py
import unittest
import mlx.core as mx
import mlx.nn as nn

class TestModelSummary(unittest.TestCase):
    def test_simple_linear(self):
        model = nn.Linear(10, 5)
        summary = nn.summary(model)
        self.assertIn("Total params: 55", summary)
        self.assertIn("weight", summary)
        self.assertIn("bias", summary)
        
    def test_sequential_model(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        summary = nn.summary(model, input_shape=(10,))
        self.assertIn("Linear", summary)
        self.assertIn("ReLU", summary)
```

#### Step 6.2: Add documentation
```python
"""
Model Summary Utilities
======================

This module provides utilities for generating model summaries similar to 
PyTorch's torchinfo.

Example usage:
    >>> model = nn.Sequential(
    ...     nn.Linear(784, 128),
    ...     nn.ReLU(),
    ...     nn.Linear(128, 10)
    ... )
    >>> print(nn.summary(model, input_shape=(784,)))
"""
```

## Common Pitfalls and Solutions

### 1. Handling Complex Output Shapes
- Some layers return tuples/lists of arrays
- Solution: Check output type and handle accordingly

### 2. Memory Calculation Accuracy
- Activation memory depends on implementation details
- Solution: Document assumptions clearly

### 3. Nested Module Handling
- Avoid double-counting parameters in nested modules
- Solution: Only count direct parameters, not nested ones

### 4. Forward Pass Failures
- Some models require specific input formats
- Solution: Wrap forward pass in try-except, make it optional

## Testing Strategy

1. **Unit Tests**
   - Test parameter counting
   - Test different layer types
   - Test nested models
   - Test memory calculations

2. **Integration Tests**
   - Test with real models (ResNet, Transformer)
   - Compare parameter counts with known values
   - Test with different input shapes

3. **Edge Cases**
   - Empty models
   - Models with no parameters
   - Models with shared parameters
   - Quantized models

## Example Usage

```python
import mlx.nn as nn

# Simple model
model = nn.Linear(10, 5)
print(nn.summary(model))

# Complex model with input shape
model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Linear(64 * 14 * 14, 10)
)
print(nn.summary(model, input_shape=(3, 28, 28)))
```

## Expected Output Format

```
==============================================================
Layer (type)                    Param #         Trainable
==============================================================
conv2d (Conv2d)                 1,792           1,792
relu (ReLU)                     0               0
maxpool2d (MaxPool2d)           0               0
linear (Linear)                 125,450         125,450
==============================================================
Total params: 127,242
Trainable params: 127,242
Non-trainable params: 0
--------------------------------------------------------------
Input shape: (1, 3, 28, 28)
Forward/backward pass size (MB): 0.31
Params size (MB): 0.49
Estimated Total Size (MB): 0.80
==============================================================
```

## Resources

1. **MLX Documentation**
   - Module API: `python/mlx/nn/layers/base.py`
   - Array API: `python/src/array.cpp`
   - Memory API: `python/src/memory.cpp`

2. **Reference Implementations**
   - torchinfo: https://github.com/TylerYep/torchinfo
   - Original torchsummary: https://github.com/sksq96/pytorch-summary

3. **Key Files to Study**
   - `mlx/nn/layers/base.py` - Module implementation
   - `mlx/nn/utils.py` - Existing utilities
   - `mlx/utils.py` - Tree utilities

## Next Steps

1. Start with Phase 1 - Basic parameter counting
2. Get feedback on initial implementation
3. Incrementally add features
4. Write comprehensive tests
5. Submit PR with documentation

This implementation will provide MLX users with a powerful tool for model inspection and debugging, matching the convenience of PyTorch's model summary tools.