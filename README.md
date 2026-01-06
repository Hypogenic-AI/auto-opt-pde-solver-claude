# Automatic Optimization of a PDE Solver

Research project investigating automatic AI scientist methods for optimizing the jaxhps Hierarchical Poincare-Steklov (HPS) PDE solver.

## Key Findings

- **Interpolation Optimization**: 3-8x speedup (mean 5.2x) using JAX JIT compilation
- **Kernel Fusion**: 1.5-41x speedup (mean 16x) on solve phase
- **Combined Optimization**: 3.7-6.7x end-to-end speedup for solve+interpolation

All optimizations maintain numerical accuracy to machine precision.

## Project Structure

```
.
├── REPORT.md                  # Full research report with methodology and results
├── README.md                  # This file
├── planning.md                # Initial research plan
├── src/
│   ├── baseline_profiling.py      # Baseline jaxhps profiling
│   ├── optimized_interpolation_v2.py  # Optimized interpolation implementation
│   ├── kernel_fusion.py           # Fused kernel implementation
│   ├── adaptive_parallelization.py    # Adaptive grid analysis
│   ├── run_experiments.py         # Comprehensive experiment runner
│   └── analyze_results.py         # Statistical analysis
├── results/
│   ├── all_experiments.json       # Raw experimental data
│   ├── statistical_analysis.json  # Statistical summary
│   └── *.json                     # Other result files
├── figures/
│   ├── interpolation_speedup.png
│   ├── kernel_fusion_speedup.png
│   ├── combined_optimization.png
│   └── build_vs_solve.png
├── code/
│   └── jaxhps/                    # jaxhps library (cloned)
├── papers/                        # Reference papers (PDFs)
└── literature_review.md           # Literature review
```

## Quick Start

### Environment Setup

```bash
# Create virtual environment
uv venv

# Activate
source .venv/bin/activate

# Install dependencies
uv add numpy scipy matplotlib jax jaxlib
uv pip install -e code/jaxhps
```

### Run Experiments

```bash
# Run comprehensive experiments
python src/run_experiments.py

# Generate analysis and figures
python src/analyze_results.py
```

### Use Optimized Functions

```python
from src.optimized_interpolation_v2 import interp_hps_optimized
from src.kernel_fusion import fused_solve

# Optimized interpolation (3-8x faster)
result = interp_hps_optimized(f_evals, p, leaf_bounds, n_uniform, target_bounds)

# Optimized solve (1.5-41x faster)
solution = fused_solve(pde_problem, boundary_data)
```

## Results Summary

### Interpolation Speedup

| p | n_uniform | Speedup |
|---|-----------|---------|
| 8 | 50 | 6.75x |
| 12 | 100 | 8.35x |
| 16 | 200 | 3.13x |

### Kernel Fusion Speedup

| p | L | Speedup |
|---|---|---------|
| 8 | 2 | 25.0x |
| 12 | 2 | 40.8x |
| 16 | 4 | 1.49x |

### Statistical Significance

| Optimization | p-value | Cohen's d |
|-------------|---------|-----------|
| Interpolation | 0.009 | 0.86 |
| Kernel Fusion | 1.15×10⁻⁶ | 1.32 |

## Requirements

- Python >= 3.10
- JAX >= 0.6.2
- NumPy >= 2.2.6
- SciPy >= 1.15.3
- Matplotlib >= 3.10.8
- jaxhps >= 0.2

## Limitations

- CPU-only testing (no GPU results)
- Build phase not optimized (dominates total time)
- Limited adaptive discretization testing

## References

1. Melia et al. "Hardware Acceleration for HPS Algorithms" (2025)
2. Yesypenko & Martinsson "GPU Optimizations for HPS" (2022)
3. Martinsson "HPS Tutorial" (2015)

See [REPORT.md](REPORT.md) for full details.

## License

Research code for experimental purposes.
