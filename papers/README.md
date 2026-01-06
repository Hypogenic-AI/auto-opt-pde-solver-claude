# Downloaded Papers

This directory contains papers relevant to the automatic optimization of PDE solvers, specifically focusing on the Hierarchical Poincare-Steklov (HPS) method with GPU acceleration and JAX integration.

## Papers

### 1. Hardware Acceleration for HPS Algorithms in Two and Three Dimensions
- **File**: `2503.17535_hps_hardware_acceleration.pdf`
- **Authors**: Owen Melia, Daniel Fortunato, Jeremy Hoskins, Rebecca Willett
- **Year**: 2025
- **arXiv**: https://arxiv.org/abs/2503.17535
- **Why relevant**: Primary target for optimization - describes the jaxhps package which is the main codebase for our experiments. Introduces GPU acceleration strategies and automatic differentiation integration.

### 2. GPU Optimizations for the Hierarchical Poincare-Steklov Scheme
- **File**: `2211.14969_hps_gpu_optimizations.pdf`
- **Authors**: Anna Yesypenko, Per-Gunnar Martinsson
- **Year**: 2022 (updated 2025)
- **arXiv**: https://arxiv.org/abs/2211.14969
- **Why relevant**: Describes batched linear algebra optimizations for HPS on GPUs. Key reference for understanding parallelization strategies that could be automated.

### 3. A Two-Level Direct Solver for the Hierarchical Poincare-Steklov Method
- **File**: `2503.04033_hps_two_level_solver.pdf`
- **Year**: 2025
- **arXiv**: https://arxiv.org/abs/2503.04033
- **Why relevant**: Novel two-level approach splitting dense GPU operations from sparse hierarchical solver. Provides insights into kernel fusion opportunities.

### 4. Bringing PDEs to JAX with Forward and Reverse Modes Automatic Differentiation
- **File**: `2309.07137_jax_pde_autodiff.pdf`
- **Authors**: Ivan Yashchuk
- **Year**: 2023
- **arXiv**: https://arxiv.org/abs/2309.07137
- **Why relevant**: Demonstrates patterns for integrating PDE solvers with JAX autodiff using tangent-linear and adjoint equations.

### 5. The Hierarchical Poincare-Steklov (HPS) Solver for Elliptic PDEs: A Tutorial
- **File**: `1506.01308_hps_tutorial.pdf`
- **Authors**: P.G. Martinsson
- **Year**: 2015
- **arXiv**: https://arxiv.org/abs/1506.01308
- **Why relevant**: Foundational tutorial explaining the HPS algorithm, complexity analysis, and implementation details. Essential background.

### 6. Towards Modular Hierarchical Poincare-Steklov Solvers
- **File**: `2510.26945_modular_hps.pdf`
- **Authors**: Michal Outrata, Jose Pablo Lucero Lorca
- **Year**: 2025
- **arXiv**: https://arxiv.org/abs/2510.26945
- **Why relevant**: Addresses modular design of HPS solvers, particularly corner handling in merge procedure. Relevant for understanding algorithmic structure.

## Key Concepts Across Papers

### HPS Algorithm Structure
1. **Build phase** (O(N^1.5)): Construct solution operators bottom-up
2. **Solve phase** (O(N log N)): Apply operators top-down

### Optimization Targets
- Local solve operations (batched BLAS)
- Merge operations (Schur complement)
- Interpolation (spectral to uniform grids)
- Memory management (recomputation vs storage tradeoff)

### Technologies
- JAX for autodiff and GPU acceleration
- Batched linear algebra (cuBLAS)
- Spectral collocation (Chebyshev/Legendre)
