# Cloned Code Repositories

This directory contains cloned repositories relevant to PDE solver optimization research.

---

## Repository 1: jaxhps (Primary)

### Overview
- **URL**: https://github.com/meliao/jaxhps
- **Purpose**: JAX implementation of HPS fast direct solver for elliptic PDEs
- **Location**: `code/jaxhps/`
- **License**: MIT

### Key Files for Optimization

**Core Solver Components**:
- `src/jaxhps/_solve.py` - Main solve interface
- `src/jaxhps/_pdeproblem.py` - Problem definition and operators
- `src/jaxhps/_domain.py` - Domain management

**Optimization Targets**:
- `src/jaxhps/_interpolation_methods.py` - Spectral-to-uniform interpolation
- `src/jaxhps/_adaptive_discretization_2D.py` - 2D adaptive grids
- `src/jaxhps/_adaptive_discretization_3D.py` - 3D adaptive grids
- `src/jaxhps/merge/` - Merge operations (kernel fusion candidates)
- `src/jaxhps/local_solve/` - Batched local solvers

**Examples**:
- `examples/hp_convergence_2D_problems.py` - Accuracy testing
- `examples/wavefront_adaptive_discretization_3D.py` - Adaptive grid testing
- `examples/inverse_wave_scattering.py` - Autodiff testing

### Installation

```bash
pip install jaxhps[examples]

# Or for development
cd code/jaxhps
pip install -e .[examples]
```

### Quick Start

```bash
cd code/jaxhps
python examples/hp_convergence_2D_problems.py --DtN --ItI
```

---

## Repository 2: PDEBench

### Overview
- **URL**: https://github.com/pdebench/PDEBench
- **Purpose**: Benchmarking framework for neural PDE solvers
- **Location**: `code/PDEBench/`
- **License**: MIT (code), CC BY (data)

### Key Files

**Neural Solver Baselines**:
- `pdebench/models/fno/` - Fourier Neural Operator
- `pdebench/models/unet/` - U-Net architecture
- `pdebench/models/pinn/` - Physics-Informed Neural Network

**Evaluation**:
- `pdebench/models/metrics.py` - Evaluation metrics
- `pdebench/models/train_models_forward.py` - Training script

**Data Generation**:
- `pdebench/data_gen/` - Scripts for generating benchmark data

### Installation

```bash
cd code/PDEBench
pip install .

# For GPU support
pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Notes
- Dataset must be downloaded separately (see `datasets/README.md`)
- Use for neural solver comparisons only

---

## Repository 3: HPS (Reference Implementation)

### Overview
- **URL**: https://github.com/DamynChipman/HPS
- **Purpose**: C++ implementation of HPS for adaptive meshes
- **Location**: `code/HPS/`
- **License**: See repository

### Key Files
- Reference for algorithm implementation details
- Useful for understanding traditional HPS structure
- Not directly used for experiments (JAX implementation preferred)

---

## Development Notes

### Setting Up Development Environment

```bash
# Create conda environment
conda create -n pde-opt python=3.10
conda activate pde-opt

# Install JAX with GPU support
pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install jaxhps for development
cd code/jaxhps
pip install -e .[examples]

# Install PDEBench (optional)
cd ../PDEBench
pip install -e .
```

### Profiling jaxhps

```bash
# Profile with JAX internal tools
python -c "
import jax
jax.profiler.start_trace('profile_output')
# Run your code
jax.profiler.stop_trace()
"

# View with TensorBoard
tensorboard --logdir=profile_output

# NVIDIA Nsight profiling
nsys profile --stats=true python examples/hp_convergence_2D_problems.py
```

### Running Tests

```bash
cd code/jaxhps
pytest tests/
```

---

## Relationship to Research Hypothesis

The research hypothesis focuses on three optimization areas:

1. **Parallelization for Adaptive Grids**
   - Target: `_adaptive_discretization_2D/3D.py`
   - Current: Sequential tree traversal
   - Opportunity: Parallel leaf processing, load balancing

2. **Kernel Fusion**
   - Target: `merge/`, `local_solve/`
   - Current: Separate kernels for each operation
   - Opportunity: Fuse adjacent operations to reduce memory traffic

3. **Spectral-to-Uniform Interpolation**
   - Target: `_interpolation_methods.py`
   - Current: Direct matrix multiplication
   - Opportunity: Structured matrix algorithms, learned approximations
