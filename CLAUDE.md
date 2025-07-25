# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLX is an array framework for machine learning on Apple silicon, with APIs similar to NumPy/PyTorch. It supports multiple backends (Metal, CUDA, CPU) and provides composable function transformations.

## Essential Commands

### Python Development
```bash
# Install for development (includes all dev dependencies)
pip install -e ".[dev]"

# Fast rebuild during development
python setup.py build_ext --inplace

# Run all Python tests
python -m unittest discover python/tests

# Run a single test file
python -m unittest python/tests/test_array.py

# Run a specific test
python -m unittest python.tests.test_array.TestArray.test_arithmetic

# Generate type stubs after modifying bindings
python setup.py generate_stubs
```

### C++ Development
```bash
# Build C++ library
mkdir -p build && cd build
cmake .. && make -j

# Run C++ tests
make test

# Run specific C++ test
./tests/test_array
```

### Code Formatting
```bash
# Set up pre-commit hooks (required)
pip install pre-commit
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files

# Auto-format specific files
pre-commit run --files path/to/file.cpp
```

## Architecture Overview

### Core Structure
- **mlx/**: Core C++ library implementing array operations, backends, and transformations
- **mlx/backend/**: Backend implementations (metal/, cuda/, cpu/) with device-specific optimizations
- **python/**: Python bindings using nanobind, includes high-level APIs (nn/, optimizers/)
- **tests/**: C++ unit tests
- **python/tests/**: Python unit tests using unittest framework

### Key Architectural Patterns

1. **Backend Abstraction**: Operations are implemented in backend-specific directories. When adding new ops:
   - Define interface in `mlx/primitives.h`
   - Implement in `mlx/backend/{metal,cuda,cpu}/`
   - Add Python bindings in `python/src/`

2. **Lazy Evaluation**: Arrays are lazy-evaluated. Operations build a computation graph that's executed when needed (e.g., when calling `.item()` or printing).

3. **Multi-device Support**: Arrays can move between devices. Check `array.to_device()` and `mlx::Device` for device management.

4. **Function Transformations**: Core transformations (grad, vmap, jit) are in `mlx/transforms.h`. They work by tracing through operations.

### Adding New Features

When implementing new operations:
1. Add C++ implementation in appropriate backend directories
2. Add primitive declaration in `mlx/primitives.h`
3. Add Python bindings in `python/src/ops.cpp` or appropriate binding file
4. Add tests in both `tests/` (C++) and `python/tests/` (Python)
5. Update type stubs with `python setup.py generate_stubs`

When modifying neural network layers:
1. Implementation goes in `python/mlx/nn/layers/`
2. Follow existing patterns for parameter initialization
3. Add tests in `python/tests/test_nn.py`

### Testing Considerations

- Python tests use a custom `MLXTestRunner` that can skip CUDA tests if CUDA is unavailable
- Use `self.assertEqualArray()` for comparing MLX arrays in tests
- C++ tests are in `tests/` and follow Google Test patterns
- Always test multiple backends when applicable

### Performance

- Benchmarks are in `benchmarks/` - run these for performance-sensitive changes
- Metal kernels are JIT-compiled and cached
- Use `mlx.core.eval()` to force evaluation in benchmarks

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md
