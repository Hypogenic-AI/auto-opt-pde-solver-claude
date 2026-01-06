# Datasets for PDE Solver Optimization

This directory contains datasets for the research project on automatic optimization of PDE solvers. Data files are NOT committed to git due to size. Follow the download instructions below.

---

## Dataset 1: jaxhps Built-in Test Problems (Recommended)

### Overview
- **Source**: Built into jaxhps package
- **Size**: Synthetic (generated on-demand)
- **Format**: NumPy arrays / JAX arrays
- **Task**: Elliptic PDE solving (Poisson, Helmholtz, Poisson-Boltzmann)
- **License**: MIT (part of jaxhps)

### Download Instructions

The test problems are built into jaxhps and don't require separate download:

```python
# Install jaxhps
pip install jaxhps[examples]

# Test problems are generated programmatically
from jaxhps import Domain, PDEProblem
```

### Available Test Problems

1. **hp-convergence problems** (`examples/hp_convergence_2D_problems.py`)
   - Problems with known analytic solutions
   - Tests: DtN matrices, ItI matrices
   - Usage: Accuracy validation

2. **Wavefront problem** (`examples/wavefront_adaptive_discretization_3D.py`)
   - 3D problem with sharp gradients
   - Tests: Adaptive discretization
   - Usage: Adaptive grid optimization

3. **Wave scattering** (`examples/wave_scattering_compute_reference_soln.py`)
   - High-wavenumber Helmholtz equation
   - Tests: Highly oscillatory solutions
   - Usage: Performance benchmarking

4. **Inverse scattering** (`examples/inverse_wave_scattering.py`)
   - Tests: Automatic differentiation
   - Usage: Autodiff correctness and performance

5. **Poisson-Boltzmann** (`examples/poisson_boltzmann_example.py`)
   - Electrostatics problem
   - Tests: Variable coefficients
   - Usage: Real-world application testing

### Loading Test Problems

```python
# Example: Generate a 2D test problem
import jaxhps
import jax.numpy as jnp

# Create domain
L = 16  # number of levels
p = 10  # polynomial order
domain = jaxhps.Domain.make_uniform_2D(L, p)

# Define PDE coefficients
def a(x, y):
    return jnp.ones_like(x)

def f(x, y):
    return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)

# Create and solve problem
problem = jaxhps.PDEProblem(domain, a, f)
solution = jaxhps.solve(problem)
```

---

## Dataset 2: PDEBench (Neural Solver Comparison)

### Overview
- **Source**: DaRUS (University of Stuttgart)
- **Size**: ~100GB total (individual files 100MB-10GB)
- **Format**: HDF5
- **Task**: Various PDE benchmarks
- **Splits**: train/valid/test provided
- **License**: CC BY

### Most Relevant Subset: 2D Darcy Flow

The 2D Darcy Flow problem is most relevant for comparison with HPS (elliptic PDE):

```
Size: ~10GB
Samples: 10,000
Resolution: 128x128
Variables: permeability field, solution field
```

### Download Instructions

**Option 1: Direct Download (Recommended)**

```bash
# Create datasets directory
mkdir -p datasets/pdebench

# Download 2D Darcy Flow
cd datasets/pdebench
wget https://darus.uni-stuttgart.de/api/access/datafile/:persistentId?persistentId=doi:10.18419/darus-2986/10 -O 2D_DarcyFlow_beta1.0_Train.hdf5
```

**Option 2: Using PDEBench Download Script**

```bash
cd code/PDEBench
pip install pdebench

# Use download script (see pdebench/data_download/README.md for details)
python pdebench/data_download/download_direct.py --pde darcy --data_folder ../../datasets/pdebench
```

### Loading the Dataset

```python
import h5py
import numpy as np

with h5py.File('datasets/pdebench/2D_DarcyFlow_beta1.0_Train.hdf5', 'r') as f:
    # Data shape: [batch, x, y]
    permeability = f['nu'][:]  # Input field
    solution = f['tensor'][:]  # Solution field

print(f"Permeability shape: {permeability.shape}")
print(f"Solution shape: {solution.shape}")
```

### Sample Data

Example fields from 2D Darcy Flow:
- Input: Random permeability field (log-normal distribution)
- Output: Pressure field solving -div(k * grad(u)) = f

---

## Dataset 3: NIST AMR Benchmarks (Validation)

### Overview
- **Source**: NIST (National Institute of Standards and Technology)
- **Size**: Synthetic (specifications provided)
- **Format**: Problem specifications (not raw data)
- **Task**: Adaptive mesh refinement validation
- **License**: Public domain

### Access Instructions

The NIST AMR benchmarks are problem specifications, not downloadable datasets:

```
URL: https://math.nist.gov/amr-benchmark/
```

### Available Benchmark Problems

1. **L-shaped domain**: Tests corner singularity handling
2. **Reentrant corner**: Tests adaptive refinement near singularities
3. **Peak problem**: Tests refinement for smooth but localized features
4. **Battery problem**: Tests multiple materials
5. **Wave front**: Tests moving sharp gradients

### Using NIST Benchmarks

These benchmarks define problem geometry and boundary conditions. Implementation:

```python
# Example: L-shaped domain setup
import jaxhps

# The L-shape requires custom domain geometry
# See jaxhps documentation for non-rectangular domains
```

---

## Data Generation Notes

For reproducibility, all benchmark data can be regenerated:

### jaxhps Test Problems
- Deterministic with fixed random seeds
- Generated on-the-fly during experiments
- Configurable parameters: domain size, polynomial order, tolerance

### PDEBench Data
- Generated using FEM/finite difference solvers
- Generation scripts in `code/PDEBench/pdebench/data_gen/`
- Fully reproducible with provided configs

---

## .gitignore Configuration

Large data files are excluded from git. The following patterns are ignored:

```
# Data files
datasets/**/*.hdf5
datasets/**/*.h5
datasets/**/*.npy
datasets/**/*.npz
datasets/**/*.pkl
datasets/**/*.pt
datasets/**/data/

# Keep documentation
!datasets/README.md
!datasets/**/README.md
```

---

## Recommended Experiment Setup

For initial experiments, use only jaxhps built-in test problems:

1. **Start**: `hp_convergence_2D_problems.py` - Validate accuracy
2. **Then**: `wavefront_adaptive_discretization_3D.py` - Test adaptive grids
3. **Finally**: `inverse_wave_scattering.py` - Test autodiff

For neural solver comparison, add PDEBench 2D Darcy Flow (~10GB download).
