# Literature Review: Automatic Optimization of a PDE Solver

## Research Area Overview

This literature review focuses on the optimization of PDE (Partial Differential Equation) solvers, particularly the Hierarchical Poincare-Steklov (HPS) method. The research hypothesis centers on using automatic AI methods to improve PDE solver performance through:
1. Enhanced parallelization for adaptive grids
2. Kernel fusion optimization
3. Optimized interpolation from spectral to uniform grids

The HPS method is a fast direct solver for linear elliptic PDEs that combines multidomain spectral collocation with hierarchical direct solvers, achieving O(N^1.5) complexity for pre-computation and O(N log N) for the solve. Recent work has focused on GPU acceleration using JAX, making these methods amenable to automatic differentiation and machine learning integration.

---

## Key Papers

### Paper 1: Hardware Acceleration for HPS Algorithms in Two and Three Dimensions
- **Authors**: Owen Melia, Daniel Fortunato, Jeremy Hoskins, Rebecca Willett
- **Year**: 2025
- **Source**: arXiv:2503.17535
- **File**: `papers/2503.17535_hps_hardware_acceleration.pdf`
- **Key Contribution**: Developed a flexible, open-source JAX framework for GPU-accelerated HPS algorithms with automatic differentiation support
- **Methodology**:
  - 2D: Novel recomputation strategy minimizing GPU data transfers
  - 3D: Extended adaptive discretization techniques to reduce peak memory usage
  - Uses JAX for hardware acceleration and autodiff integration
- **Datasets Used**: Synthetic test problems (wavefront, scattering problems)
- **Results**: First integration of high-order fast direct solver with automatic differentiation
- **Code Available**: Yes - jaxhps package (https://github.com/meliao/jaxhps)
- **Relevance**: Primary target for optimization - provides the codebase for our experiments

### Paper 2: GPU Optimizations for the Hierarchical Poincare-Steklov Scheme
- **Authors**: Anna Yesypenko, Per-Gunnar Martinsson
- **Year**: 2022 (updated 2025)
- **Source**: arXiv:2211.14969
- **File**: `papers/2211.14969_hps_gpu_optimizations.pdf`
- **Key Contribution**: GPU optimizations for 2D HPS using batched linear algebra
- **Methodology**:
  - Exploits batched linear algebra on hybrid CPU/GPU architectures
  - Focuses on reducing cost of local static condensation for high-order discretizations
  - Uses batched BLAS operations for parallel leaf processing
- **Results**: Significant speedups using GPU optimizations; practical independence from polynomial order for p<=20
- **Code Available**: Not explicitly mentioned
- **Relevance**: Provides optimization strategies for GPU parallelization that could be automated

### Paper 3: A Two-Level Direct Solver for the Hierarchical Poincare-Steklov Method
- **Authors**: (See paper)
- **Year**: 2025
- **Source**: arXiv:2503.04033
- **File**: `papers/2503.04033_hps_two_level_solver.pdf`
- **Key Contribution**: Splits HPS into dense GPU operations and sparse hierarchical solver
- **Methodology**:
  - Two-level approach: dense linear algebra on GPUs + multilevel sparse solvers
  - Batched GPU routines make cost independent of polynomial order
  - Improved numerical stability
- **Results**: O(N p^6) initial reduction in 3D, but practical cost independent of p for p<=20
- **Code Available**: Not explicitly mentioned
- **Relevance**: Architecture insights for parallelization and kernel fusion opportunities

### Paper 4: Bringing PDEs to JAX with Forward and Reverse Modes Automatic Differentiation
- **Authors**: Ivan Yashchuk
- **Year**: 2023
- **Source**: arXiv:2309.07137
- **File**: `papers/2309.07137_jax_pde_autodiff.pdf`
- **Key Contribution**: Extends JAX autodiff with Firedrake finite element library
- **Methodology**:
  - Uses tangent-linear equations (forward mode) and adjoint equations (reverse mode)
  - Bypasses differentiation through iterative solver iterations
  - High-level symbolic PDE representation
- **Results**: Efficient composition of FEM solvers with differentiable programs
- **Code Available**: Yes
- **Relevance**: Demonstrates autodiff integration patterns applicable to HPS solvers

### Paper 5: The Hierarchical Poincare-Steklov (HPS) Solver for Elliptic PDEs: A Tutorial
- **Authors**: P.G. Martinsson
- **Year**: 2015
- **Source**: arXiv:1506.01308
- **File**: `papers/1506.01308_hps_tutorial.pdf`
- **Key Contribution**: Foundational tutorial on HPS methods
- **Methodology**:
  - High-order spectral approximations on 2D domains
  - Direct solver with O(N^1.5) precomputation, O(N log N) solve
  - Particularly suited for oscillatory solutions
- **Results**: Establishes theoretical foundations and complexity analysis
- **Code Available**: MATLAB implementations exist
- **Relevance**: Essential background for understanding HPS algorithm structure

### Paper 6: Towards Modular Hierarchical Poincare-Steklov Solvers
- **Authors**: Michal Outrata, Jose Pablo Lucero Lorca
- **Year**: 2025
- **Source**: arXiv:2510.26945
- **File**: `papers/2510.26945_modular_hps.pdf`
- **Key Contribution**: Clarifies corner handling in HPS merge procedure for Q1 FEM
- **Methodology**:
  - Naturally accommodates corner coupling
  - Connects algebraic Schur-complement methods to operator-based formulations
- **Results**: Enables FEM community adoption while preserving Poincare-Steklov interpretation
- **Relevance**: Provides insights into modular solver design

---

## Common Methodologies

### Spectral Collocation Methods
- Used in Papers 1, 2, 3, 5
- High-order polynomial approximations on each subdomain
- Chebyshev or Legendre polynomials for spectral accuracy
- Exponential convergence for smooth solutions

### Hierarchical/Nested Dissection Solvers
- Used in Papers 1, 2, 3, 5, 6
- Build solution tree from leaves to root
- Dirichlet-to-Neumann (DtN) or Impedance-to-Impedance (ItI) maps
- O(N^1.5) build, O(N log N) solve complexity

### GPU Batched Linear Algebra
- Used in Papers 1, 2, 3
- BLAS3 operations for matrix-matrix multiplications
- Batched operations for processing multiple leaves simultaneously
- Key for achieving GPU utilization

### Automatic Differentiation
- Used in Papers 1, 4
- JAX-based forward (jvp) and reverse (vjp) mode
- Enables gradient-based optimization and ML integration

---

## Standard Baselines

For neural PDE solver comparisons (from PDEBench):
- **FNO (Fourier Neural Operator)**: Best accuracy in most cases
- **U-Net**: Good for spatiotemporal problems
- **PINN (Physics-Informed Neural Networks)**: Physics-constrained but less accurate

For classical solver comparisons:
- **Multigrid methods**: O(N) complexity, iterative
- **FFT-based spectral methods**: Fast for periodic domains
- **Sparse direct solvers (UMFPACK, SuperLU)**: General but not optimized for HPS structure

---

## Evaluation Metrics

### Accuracy Metrics
- Relative L2 error: ||u - u_exact||_2 / ||u_exact||_2
- Maximum pointwise error: max|u - u_exact|
- Solution residual: ||Lu - f||

### Performance Metrics
- Wall-clock time for build phase
- Wall-clock time for solve phase
- Memory usage (peak and average)
- GPU utilization percentage
- FLOPS achieved vs theoretical peak

### Convergence Metrics
- Error vs polynomial order p (spectral convergence)
- Error vs number of DOFs N (hp convergence)
- Error vs wall-clock time (efficiency curves)

---

## Datasets in the Literature

### Synthetic Test Problems (from jaxhps)
- Wavefront problem: 3D problem with sharp gradients
- Wave scattering: High-wavenumber Helmholtz equation
- Poisson-Boltzmann: Electrostatics problems
- hp-convergence problems: Problems with known analytic solutions

### PDEBench Datasets
- 1D/2D/3D Poisson equation (closest to HPS applications)
- 2D Darcy flow
- Diffusion-reaction equations
- Navier-Stokes (compressible/incompressible)

### NIST AMR Benchmarks
- Standard test problems for adaptive mesh refinement
- Designed for hp-adaptive methods

---

## Gaps and Opportunities

### Optimization Opportunities for AI Methods

1. **Adaptive Grid Parallelization**
   - Current: Tree traversal is inherently sequential
   - Opportunity: AI-guided scheduling and load balancing
   - Potential: Predict optimal tree structure from problem characteristics

2. **Kernel Fusion**
   - Current: Separate kernels for local solve, merge, interpolation
   - Opportunity: Automatically identify and fuse adjacent operations
   - Potential: Reduce memory bandwidth and kernel launch overhead

3. **Spectral-to-Uniform Interpolation**
   - Current: Direct matrix multiplication
   - Opportunity: Optimize interpolation operator structure
   - Potential: Fast algorithms or learned approximations

4. **Memory Management**
   - Current: Manual memory allocation strategies
   - Opportunity: AI-guided memory pooling and reuse
   - Potential: Reduce peak memory for 3D problems

5. **Hyperparameter Selection**
   - Current: Manual tuning of p (polynomial order), tree depth
   - Opportunity: Automated parameter selection based on problem
   - Potential: Optimal accuracy-efficiency tradeoffs

---

## Recommendations for Experiments

### Primary Dataset
- **jaxhps synthetic problems**: Directly relevant, controlled, reproducible
- Start with 2D problems, scale to 3D
- Use wavefront and scattering benchmarks from the paper

### Recommended Baselines
1. Unoptimized jaxhps (baseline)
2. GPU-optimized jaxhps (current best)
3. Standard FFT-based spectral solver (for comparison)

### Recommended Metrics
1. Wall-clock time (build + solve)
2. GPU memory usage
3. Relative L2 error
4. Kernel profiling (nvprof/nsight)

### Methodological Considerations
1. Warm up GPU before timing
2. Use multiple runs and report variance
3. Test across problem sizes (N = 1K to 1M DOFs)
4. Test across polynomial orders (p = 4 to 20)
5. Profile kernel-level performance for optimization targets

---

## Key Code Resources

### jaxhps (Primary)
- GitHub: https://github.com/meliao/jaxhps
- PyPI: `pip install jaxhps`
- Key files for optimization:
  - `_interpolation_methods.py`: Spectral-to-uniform interpolation
  - `_adaptive_discretization_2D/3D.py`: Adaptive grid generation
  - `merge/`: Merge operations for hierarchical solver
  - `local_solve/`: Local leaf solvers

### PDEBench (Benchmarking)
- GitHub: https://github.com/pdebench/PDEBench
- Provides baseline neural solver implementations
- Standardized evaluation framework

---

## References

1. Melia et al. "Hardware Acceleration for HPS Algorithms" arXiv:2503.17535 (2025)
2. Yesypenko & Martinsson "GPU Optimizations for HPS" arXiv:2211.14969 (2022)
3. "A Two-Level Direct Solver for HPS" arXiv:2503.04033 (2025)
4. Yashchuk "Bringing PDEs to JAX" arXiv:2309.07137 (2023)
5. Martinsson "HPS Tutorial" arXiv:1506.01308 (2015)
6. Outrata & Lucero Lorca "Modular HPS Solvers" arXiv:2510.26945 (2025)
7. Takamoto et al. "PDEBench" NeurIPS 2022
