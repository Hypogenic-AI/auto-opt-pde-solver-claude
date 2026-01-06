# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project on Automatic Optimization of a PDE Solver. The focus is on the Hierarchical Poincare-Steklov (HPS) method implemented in JAX, with opportunities for AI-driven optimization of parallelization, kernel fusion, and interpolation.

---

## Papers

**Total papers downloaded: 6**

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Hardware Acceleration for HPS Algorithms | Melia et al. | 2025 | papers/2503.17535_hps_hardware_acceleration.pdf | Primary paper - JAX HPS with GPU acceleration |
| GPU Optimizations for the HPS Scheme | Yesypenko, Martinsson | 2022/2025 | papers/2211.14969_hps_gpu_optimizations.pdf | Batched linear algebra optimizations |
| A Two-Level Direct Solver for HPS | - | 2025 | papers/2503.04033_hps_two_level_solver.pdf | Dense GPU + sparse hierarchical approach |
| Bringing PDEs to JAX with Autodiff | Yashchuk | 2023 | papers/2309.07137_jax_pde_autodiff.pdf | JAX + Firedrake autodiff integration |
| HPS Tutorial | Martinsson | 2015 | papers/1506.01308_hps_tutorial.pdf | Foundational HPS algorithm description |
| Modular HPS Solvers | Outrata, Lucero Lorca | 2025 | papers/2510.26945_modular_hps.pdf | Corner handling in HPS merge |

See `papers/README.md` for detailed descriptions.

---

## Datasets

**Total datasets/benchmarks identified: 3**

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| jaxhps Test Problems | Built-in | Synthetic | Elliptic PDEs | code/jaxhps/examples/ | Wavefront, scattering, Poisson-Boltzmann |
| PDEBench | DaRUS | ~100GB total | Various PDEs | External download | 2D Darcy Flow most relevant |
| NIST AMR Benchmarks | NIST | Synthetic | AMR validation | External | Standard test problems |

See `datasets/README.md` for detailed descriptions and download instructions.

---

## Code Repositories

**Total repositories cloned: 3**

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| jaxhps | github.com/meliao/jaxhps | Primary HPS solver | code/jaxhps/ | Main optimization target |
| PDEBench | github.com/pdebench/PDEBench | Neural solver benchmarks | code/PDEBench/ | FNO, U-Net, PINN baselines |
| HPS | github.com/DamynChipman/HPS | C++ HPS implementation | code/HPS/ | Reference for algorithm details |

See `code/README.md` for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy
1. Started with the referenced paper (arXiv:2503.17535) as anchor
2. Traced citations and related work to find GPU optimization papers
3. Searched for JAX-based PDE solvers and autodiff integration
4. Identified PDEBench as standard ML benchmark for comparison
5. Located foundational HPS tutorial paper for algorithm background

### Selection Criteria
- **Relevance**: Direct applicability to HPS solver optimization
- **Recency**: Preference for 2024-2025 papers for state-of-the-art
- **Code availability**: Prioritized papers with open-source implementations
- **JAX ecosystem**: Focus on JAX-compatible approaches

### Challenges Encountered
- Some papers lack open-source code implementations
- PDEBench dataset is large (~100GB); only cloned repo, not data
- Limited papers specifically on AI-driven solver optimization

### Gaps and Workarounds
- **Gap**: No existing work on AI-driven HPS optimization
- **Workaround**: Use general GPU optimization papers as guidance
- **Gap**: Limited adaptive mesh refinement benchmarks
- **Workaround**: Use jaxhps built-in test problems

---

## Recommendations for Experiment Design

### 1. Primary Dataset(s)

**jaxhps synthetic problems (Recommended)**
- Location: `code/jaxhps/examples/`
- Why: Directly relevant, controlled, reproducible
- Problems:
  - `hp_convergence_2D_problems.py` - Convergence testing
  - `wavefront_adaptive_discretization_3D.py` - Adaptive grids
  - `wave_scattering_compute_reference_soln.py` - High-wavenumber
  - `inverse_wave_scattering.py` - Autodiff testing

**PDEBench 2D Darcy Flow (Secondary)**
- Why: Standard ML benchmark, neural solver comparisons
- Download: See datasets/README.md

### 2. Baseline Methods

1. **jaxhps baseline**: Unoptimized JAX implementation
2. **jaxhps GPU-optimized**: Current best implementation
3. **FFT spectral solver**: For periodic domain comparisons
4. **FNO**: Neural operator baseline (from PDEBench)

### 3. Evaluation Metrics

**Accuracy**:
- Relative L2 error
- Maximum pointwise error
- Solution residual

**Performance**:
- Wall-clock time (build + solve separately)
- GPU memory usage (peak)
- Kernel execution time breakdown

**Optimization Targets**:
- Interpolation kernel time
- Merge operation time
- Local solve time
- Memory transfer overhead

### 4. Code to Adapt/Reuse

From jaxhps:
- `src/jaxhps/_interpolation_methods.py` - Target for optimization
- `src/jaxhps/_adaptive_discretization_2D.py` - Adaptive grid target
- `src/jaxhps/merge/` - Kernel fusion candidates
- `src/jaxhps/local_solve/` - Batched operations

From PDEBench:
- `pdebench/models/fno/` - FNO baseline implementation
- `pdebench/models/metrics.py` - Evaluation utilities

---

## Quick Start Guide

### Running jaxhps Examples

```bash
# Install jaxhps
pip install jaxhps[examples]

# Run 2D hp-convergence test
cd code/jaxhps
python examples/hp_convergence_2D_problems.py --DtN --ItI

# Run 3D adaptive discretization
python examples/wavefront_adaptive_discretization_3D.py -p 10 --tol 1e-02 1e-05

# Run inverse scattering (autodiff test)
python examples/inverse_wave_scattering.py --n_iter 25
```

### Profiling for Optimization

```bash
# Profile with JAX
JAX_DISABLE_JIT=0 python -m cProfile -o profile.out examples/hp_convergence_2D_problems.py

# NVIDIA profiling (if on GPU)
nsys profile python examples/hp_convergence_2D_problems.py
```

---

## External Links

### Papers
- arXiv: https://arxiv.org/abs/2503.17535
- arXiv: https://arxiv.org/abs/2211.14969
- arXiv: https://arxiv.org/abs/2503.04033
- arXiv: https://arxiv.org/abs/2309.07137
- arXiv: https://arxiv.org/abs/1506.01308
- arXiv: https://arxiv.org/abs/2510.26945

### Code
- jaxhps: https://github.com/meliao/jaxhps
- jaxhps docs: https://jaxhps.readthedocs.io/
- PDEBench: https://github.com/pdebench/PDEBench

### Data
- PDEBench Data: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986
- NIST AMR Benchmarks: https://math.nist.gov/amr-benchmark/
