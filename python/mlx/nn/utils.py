# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from functools import reduce, wraps
from typing import Any, Callable, Optional

import mlx.core as mx

from ..utils import tree_flatten, tree_map, tree_unflatten
from .layers.base import Module


def value_and_grad(model: Module, fn: Callable):
    """Transform the passed function ``fn`` to a function that computes the
    gradients of ``fn`` wrt the model's trainable parameters and also its
    value.

    Args:
        model (mlx.nn.Module): The model whose trainable parameters to compute
                               gradients for
        fn (Callable): The scalar function to compute gradients for

    Returns:
        A callable that returns the value of ``fn`` and the gradients wrt the
        trainable parameters of ``model``
    """

    def inner_fn(params, *args, **kwargs):
        model.update(params)
        return fn(*args, **kwargs)

    value_grad_fn = mx.value_and_grad(inner_fn)

    @wraps(fn)
    def wrapped_value_grad_fn(*args, **kwargs):
        value, grad = value_grad_fn(model.trainable_parameters(), *args, **kwargs)
        return value, grad

    return wrapped_value_grad_fn


def checkpoint(module: Module, fn: Optional[Callable] = None):
    """Transform the passed callable to one that performs gradient
    checkpointing with respect to the trainable parameters of the module (and
    the callable's inputs).

    Args:
        module (mlx.nn.Module): The module for whose parameters we will be
            performing gradient checkpointing.
        fn (Callable, optional): The function to checkpoint. If not provided it
            defaults to the provided module.

    Returns:
        A callable that saves the inputs and outputs during the forward pass
        and recomputes all intermediate states during the backward pass.
    """
    if fn is None:
        # Capturing module instead of module.__call__ allows someone to
        # monkey-patch __call__ later on and the correct method will be used
        fn = module

    def inner_fn(params, *args, **kwargs):
        module.update(params)
        return fn(*args, **kwargs)

    checkpointed_fn = mx.checkpoint(inner_fn)

    @wraps(fn)
    def wrapped_checkpointed_fn(*args, **kwargs):
        return checkpointed_fn(module.trainable_parameters(), *args, **kwargs)

    return wrapped_checkpointed_fn


def average_gradients(
    gradients: Any,
    group: Optional[mx.distributed.Group] = None,
    all_reduce_size: int = 32 * 1024**2,
    communication_type: Optional[mx.Dtype] = None,
):
    """Average the gradients across the distributed processes in the passed group.

    This helper enables concatenating several gradients of small arrays to one
    big all reduce call for better networking performance.

    Args:
        gradients (Any): The Python tree containing the gradients (it should
            have the same structure across processes)
        group (Optional[mlx.core.distributed.Group]): The group of processes to
            average the gradients. If set to ``None`` the global group is used.
            Default: ``None``.
        all_reduce_size (int): Group arrays until their size in bytes exceeds
            this number. Perform one communication step per group of arrays. If
            less or equal to 0 array grouping is disabled. Default: ``32MiB``.
        communication_type (Optional[mlx.core.Dtype]): If provided cast to this
            type before performing the communication. Typically cast to a
            smaller float to reduce the communication size. Default: ``None``.
    """
    group = group or mx.distributed.init()
    N = group.size()

    if N == 1:
        return gradients

    def _average(x):
        dt = x.dtype
        x = x.astype(communication_type) if communication_type is not None else x
        return mx.distributed.all_sum(x, stream=mx.cpu).astype(dt) / N

    if all_reduce_size <= 0:
        return tree_map(_average, gradients)

    else:
        flat_grads = tree_flatten(gradients)
        if len(flat_grads) == 0:
            return gradients

        # Extract some info for the gradient
        keys = [k for k, _ in flat_grads]
        shapes = [v.shape for _, v in flat_grads]
        sizes = [v.size for _, v in flat_grads]
        dtypes = [v.dtype for _, v in flat_grads]

        # We can't group them if they have mixed types
        if not all(dt == dtypes[0] for dt in dtypes):
            return average_gradients(gradients, group, 0, communication_type)
        itemsize = (
            communication_type.size
            if communication_type is not None
            else dtypes[0].size
        )

        # Gather the gradients in groups that are just above or equal to all_reduce_size
        grad_groups = []
        grad_group = []
        grad_group_size = 0
        for i in range(len(keys)):
            grad_group.append(i)
            grad_group_size += sizes[i] * itemsize
            if grad_group_size >= all_reduce_size:
                grad_groups.append(grad_group)
                grad_group = []
                grad_group_size = 0
        if grad_group:
            grad_groups.append(grad_group)
            grad_group = []

        # Concatenate-reduce-split
        new_flat_grads = []
        for grad_group in grad_groups:
            indices = reduce(lambda x, y: x + [x[-1] + sizes[y]], grad_group, [0])
            big_grad = mx.concatenate(
                [flat_grads[i][1].reshape(-1) for i in grad_group]
            )
            big_grad = _average(big_grad)
            big_grad = mx.split(big_grad, indices[1:-1])
            new_flat_grads.extend(
                (keys[j], big_grad[i].reshape(shapes[j]))
                for i, j in enumerate(grad_group)
            )

        return tree_unflatten(new_flat_grads)


def count_parameters(module: Module) -> tuple[int, int]:
    """
    Count total and trainable parameters in a module.
    
    Args:
        module: The MLX module to analyze
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = 0
    trainable_params = 0
    
    # Get all parameters
    all_params = tree_flatten(module.parameters())
    trainable = tree_flatten(module.trainable_parameters())
    
    # Create set of trainable parameter ids for fast lookup
    trainable_ids = {id(p) for _, p in trainable}
    
    for name, param in all_params:
        if isinstance(param, mx.array):
            param_count = param.size
            total_params += param_count
            
            # Check if this parameter is trainable
            if id(param) in trainable_ids:
                trainable_params += param_count
    
    return total_params, trainable_params


def simple_summary(model: Module) -> str:
    """
    Generate a simple parameter count summary.
    
    Args:
        model: The MLX model to summarize
        
    Returns:
        String containing parameter counts
    """
    total, trainable = count_parameters(model)
    non_trainable = total - trainable
    
    return f"""Total params: {total:,}
Trainable params: {trainable:,}
Non-trainable params: {non_trainable:,}"""


@dataclass
class LayerInfo:
    """Information about a single layer in the model."""
    name: str
    module_type: str
    output_shape: Optional[tuple[int, ...]] = None
    num_params: int = 0
    trainable_params: int = 0
    param_bytes: int = 0


def collect_layer_info(model: Module) -> list[LayerInfo]:
    """
    Collect information about each layer in the model.
    
    Args:
        model: The MLX model to analyze
        
    Returns:
        List of LayerInfo objects for each layer
    """
    layer_infos = []
    
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
            if isinstance(value, mx.array) and not isinstance(value, Module):
                module_params += value.size
                module_bytes += value.nbytes
                # Check if this parameter is trainable
                if key not in module._no_grad:
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


def format_layer_summary(layer_infos: list[LayerInfo]) -> str:
    """
    Format layer information as a table.
    
    Args:
        layer_infos: List of LayerInfo objects
        
    Returns:
        Formatted table string
    """
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
        if len(layer_str) > col_widths[0] - 1:
            layer_str = layer_str[:col_widths[0] - 4] + "..."
        
        param_str = f"{info.num_params:,}"
        trainable_str = f"{info.trainable_params:,}"
        
        lines.append("".join([
            layer_str.ljust(col_widths[0]),
            param_str.ljust(col_widths[1]),
            trainable_str.ljust(col_widths[2])
        ]))
    
    lines.append("=" * sum(col_widths))
    return "\n".join(lines)


def summary(
    model: Module,
    input_shape: Optional[tuple[int, ...] | mx.array] = None,
    batch_size: int = 1,
    device: Optional[mx.Device] = None,
    verbose: int = 1,
) -> str:
    """
    Generate a summary of the model.
    
    Args:
        model: The MLX model to summarize
        input_shape: Input shape (excluding batch dimension) or sample input
        batch_size: Batch size for memory calculations
        device: Device to run the model on (currently unused)
        verbose: 0=silent, 1=default, 2=detailed
        
    Returns:
        Summary string
    """
    # Phase 2: Layer-by-layer summary
    layer_infos = collect_layer_info(model)
    
    if verbose == 0:
        return simple_summary(model)
    
    # Build the full summary
    lines = []
    
    # Add layer table
    if layer_infos:
        lines.append(format_layer_summary(layer_infos))
    
    # Add total counts
    total, trainable = count_parameters(model)
    non_trainable = total - trainable
    
    lines.append(f"Total params: {total:,}")
    lines.append(f"Trainable params: {trainable:,}")
    lines.append(f"Non-trainable params: {non_trainable:,}")
    lines.append("-" * 70)
    
    # Calculate memory usage (basic for now)
    total_bytes = sum(info.param_bytes for info in layer_infos)
    params_mb = total_bytes / (1024 * 1024)
    lines.append(f"Params size (MB): {params_mb:.2f}")
    lines.append("=" * 70)
    
    return "\n".join(lines)
