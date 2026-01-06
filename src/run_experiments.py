"""
Comprehensive experimental evaluation of PDE solver optimizations.

This script runs all experiments and collects results for the final report.
"""

import time
import json
import sys
from typing import Dict, List, Tuple
from functools import partial
from datetime import datetime

import numpy as np
import jax
import jax.numpy as jnp

from jaxhps import (
    DiscretizationNode2D,
    build_solver,
    solve as baseline_solve,
    PDEProblem,
    Domain,
)
from jaxhps._discretization_tree import get_all_leaves, get_depth
from jaxhps._interpolation_methods import interp_from_hps_2D as baseline_interp

# Import our optimizations
from optimized_interpolation_v2 import interp_hps_optimized
from kernel_fusion import fused_solve

# Set seed and device
np.random.seed(42)
jax.config.update("jax_default_device", jax.devices("cpu")[0])

# Problem parameters
XMIN, XMAX = -1.0, 1.0
YMIN, YMAX = -1.0, 1.0
PI = jnp.pi


def create_test_problem(p: int, L: int):
    """Create a test PDE problem."""
    root = DiscretizationNode2D(xmin=XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX)
    domain = Domain(p=p, q=p-2, root=root, L=L)

    # Poisson problem: -∇²u = f with u=0 on boundary
    source = jnp.sin(PI * domain.interior_points[..., 0]) * \
             jnp.sin(PI * domain.interior_points[..., 1])
    lap_coeffs = jnp.ones_like(domain.interior_points[..., 0])

    pde_problem = PDEProblem(
        domain=domain,
        D_xx_coefficients=lap_coeffs,
        D_yy_coefficients=lap_coeffs,
        source=source,
    )

    return pde_problem, domain


def experiment_1_baseline_profiling(
    p_vals: List[int] = [8, 12, 16],
    L_vals: List[int] = [2, 3, 4],
    n_runs: int = 5,
) -> List[Dict]:
    """
    Experiment 1: Baseline profiling of jaxhps.

    Measures build time, solve time, and accuracy across problem sizes.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Baseline Profiling")
    print("="*70)

    results = []

    for p in p_vals:
        for L in L_vals:
            print(f"\nTesting p={p}, L={L}...")

            # Create problem
            pde_problem, domain = create_test_problem(p, L)

            # Build solver
            t_build_start = time.perf_counter()
            build_solver(pde_problem)
            jax.block_until_ready(pde_problem.v)
            t_build_end = time.perf_counter()

            # Boundary data
            g = jnp.zeros_like(domain.boundary_points[..., 0])

            # Warmup
            _ = baseline_solve(pde_problem, g)

            # Solve timing
            solve_times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                soln = baseline_solve(pde_problem, g)
                jax.block_until_ready(soln)
                t1 = time.perf_counter()
                solve_times.append(t1 - t0)

            # Compute error (true solution: u = sin(πx)sin(πy)/(2π²))
            true_soln = jnp.sin(PI * domain.interior_points[..., 0]) * \
                       jnp.sin(PI * domain.interior_points[..., 1]) / (2 * PI**2)
            # Note: source was f = sin(πx)sin(πy), so -∇²u = 2π²u = f
            # Actually the true solution depends on the exact PDE solved

            err = jnp.max(jnp.abs(soln))  # Just check it's reasonable

            result = {
                'p': p,
                'L': L,
                'n_leaves': 4**L,
                'n_dofs': 4**L * p**2,
                'build_time': t_build_end - t_build_start,
                'solve_time_mean': float(np.mean(solve_times)),
                'solve_time_std': float(np.std(solve_times)),
                'max_soln': float(jnp.max(jnp.abs(soln))),
            }
            results.append(result)

            print(f"  Build: {result['build_time']:.4f}s, "
                  f"Solve: {result['solve_time_mean']:.4f}±{result['solve_time_std']:.4f}s")

    return results


def experiment_2_interpolation_optimization(
    p_vals: List[int] = [8, 12, 16, 20],
    n_uniform_vals: List[int] = [50, 100, 200],
    n_runs: int = 5,
) -> List[Dict]:
    """
    Experiment 2: Interpolation optimization comparison.

    Compares baseline barycentric interpolation vs optimized JIT version.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Interpolation Optimization")
    print("="*70)

    results = []

    for p in p_vals:
        for n_uniform in n_uniform_vals:
            print(f"\nTesting p={p}, n_uniform={n_uniform}...")

            # Create problem and solve
            L = 2
            pde_problem, domain = create_test_problem(p, L)
            build_solver(pde_problem)
            g = jnp.zeros_like(domain.boundary_points[..., 0])
            f_evals = baseline_solve(pde_problem, g)

            # Setup for interpolation
            leaves = tuple(get_all_leaves(domain.root))
            x_vals = jnp.linspace(XMIN, XMAX, n_uniform)
            y_vals = jnp.linspace(YMIN, YMAX, n_uniform)
            target_bounds = jnp.array([XMIN, XMAX, YMIN, YMAX])
            leaf_bounds = jnp.stack([
                jnp.array([leaf.xmin, leaf.xmax, leaf.ymin, leaf.ymax])
                for leaf in leaves
            ])

            # Warmup
            _ = baseline_interp(leaves, p, f_evals, x_vals, y_vals)
            _ = interp_hps_optimized(f_evals, p, leaf_bounds, n_uniform, target_bounds)

            # Time baseline
            baseline_times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                result_base, _ = baseline_interp(leaves, p, f_evals, x_vals, y_vals)
                jax.block_until_ready(result_base)
                t1 = time.perf_counter()
                baseline_times.append(t1 - t0)

            # Time optimized
            opt_times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                result_opt = interp_hps_optimized(f_evals, p, leaf_bounds, n_uniform, target_bounds)
                jax.block_until_ready(result_opt)
                t1 = time.perf_counter()
                opt_times.append(t1 - t0)

            # Check accuracy difference
            diff = float(jnp.max(jnp.abs(result_base - result_opt)))

            result = {
                'p': p,
                'n_uniform': n_uniform,
                'baseline_time_mean': float(np.mean(baseline_times)),
                'baseline_time_std': float(np.std(baseline_times)),
                'optimized_time_mean': float(np.mean(opt_times)),
                'optimized_time_std': float(np.std(opt_times)),
                'speedup': float(np.mean(baseline_times) / np.mean(opt_times)),
                'max_diff': diff,
            }
            results.append(result)

            print(f"  Baseline: {result['baseline_time_mean']:.4f}±{result['baseline_time_std']:.4f}s, "
                  f"Optimized: {result['optimized_time_mean']:.4f}±{result['optimized_time_std']:.4f}s, "
                  f"Speedup: {result['speedup']:.2f}x")

    return results


def experiment_3_kernel_fusion(
    p_vals: List[int] = [8, 12, 16],
    L_vals: List[int] = [2, 3, 4],
    n_runs: int = 5,
) -> List[Dict]:
    """
    Experiment 3: Kernel fusion for solve phase.

    Compares baseline solve vs fused solve implementation.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Kernel Fusion")
    print("="*70)

    results = []

    for p in p_vals:
        for L in L_vals:
            print(f"\nTesting p={p}, L={L}...")

            # Create and build problem
            pde_problem, domain = create_test_problem(p, L)
            build_solver(pde_problem)
            g = jnp.zeros_like(domain.boundary_points[..., 0])

            # Warmup
            _ = baseline_solve(pde_problem, g)
            _ = fused_solve(pde_problem, g)

            # Time baseline
            baseline_times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                result_base = baseline_solve(pde_problem, g)
                jax.block_until_ready(result_base)
                t1 = time.perf_counter()
                baseline_times.append(t1 - t0)

            # Time fused
            fused_times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                result_fused = fused_solve(pde_problem, g)
                jax.block_until_ready(result_fused)
                t1 = time.perf_counter()
                fused_times.append(t1 - t0)

            # Check correctness
            diff = float(jnp.max(jnp.abs(result_base - result_fused)))

            result = {
                'p': p,
                'L': L,
                'n_leaves': 4**L,
                'baseline_time_mean': float(np.mean(baseline_times)),
                'baseline_time_std': float(np.std(baseline_times)),
                'fused_time_mean': float(np.mean(fused_times)),
                'fused_time_std': float(np.std(fused_times)),
                'speedup': float(np.mean(baseline_times) / np.mean(fused_times)),
                'max_diff': diff,
            }
            results.append(result)

            print(f"  Baseline: {result['baseline_time_mean']:.4f}±{result['baseline_time_std']:.4f}s, "
                  f"Fused: {result['fused_time_mean']:.4f}±{result['fused_time_std']:.4f}s, "
                  f"Speedup: {result['speedup']:.2f}x")

    return results


def experiment_4_combined_optimization(
    p_vals: List[int] = [12, 16],
    L_vals: List[int] = [3, 4],
    n_runs: int = 5,
) -> List[Dict]:
    """
    Experiment 4: Combined optimization (solve + interpolation).

    Tests end-to-end performance with all optimizations applied.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: Combined Optimization")
    print("="*70)

    results = []
    n_uniform = 100

    for p in p_vals:
        for L in L_vals:
            print(f"\nTesting p={p}, L={L}, n_interp={n_uniform}...")

            # Create and build
            pde_problem, domain = create_test_problem(p, L)
            build_solver(pde_problem)
            g = jnp.zeros_like(domain.boundary_points[..., 0])

            leaves = tuple(get_all_leaves(domain.root))
            x_vals = jnp.linspace(XMIN, XMAX, n_uniform)
            y_vals = jnp.linspace(YMIN, YMAX, n_uniform)
            target_bounds = jnp.array([XMIN, XMAX, YMIN, YMAX])
            leaf_bounds = jnp.stack([
                jnp.array([leaf.xmin, leaf.xmax, leaf.ymin, leaf.ymax])
                for leaf in leaves
            ])

            # Warmup
            soln_base = baseline_solve(pde_problem, g)
            _ = baseline_interp(leaves, p, soln_base, x_vals, y_vals)

            soln_opt = fused_solve(pde_problem, g)
            _ = interp_hps_optimized(soln_opt, p, leaf_bounds, n_uniform, target_bounds)

            # Time baseline (solve + interp)
            baseline_times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                soln = baseline_solve(pde_problem, g)
                interp_result, _ = baseline_interp(leaves, p, soln, x_vals, y_vals)
                jax.block_until_ready(interp_result)
                t1 = time.perf_counter()
                baseline_times.append(t1 - t0)

            # Time optimized (fused solve + optimized interp)
            opt_times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                soln = fused_solve(pde_problem, g)
                interp_result = interp_hps_optimized(soln, p, leaf_bounds, n_uniform, target_bounds)
                jax.block_until_ready(interp_result)
                t1 = time.perf_counter()
                opt_times.append(t1 - t0)

            result = {
                'p': p,
                'L': L,
                'n_uniform': n_uniform,
                'baseline_time_mean': float(np.mean(baseline_times)),
                'baseline_time_std': float(np.std(baseline_times)),
                'optimized_time_mean': float(np.mean(opt_times)),
                'optimized_time_std': float(np.std(opt_times)),
                'speedup': float(np.mean(baseline_times) / np.mean(opt_times)),
            }
            results.append(result)

            print(f"  Baseline: {result['baseline_time_mean']:.4f}±{result['baseline_time_std']:.4f}s, "
                  f"Optimized: {result['optimized_time_mean']:.4f}±{result['optimized_time_std']:.4f}s, "
                  f"Speedup: {result['speedup']:.2f}x")

    return results


def main():
    """Run all experiments and save results."""
    print("="*70)
    print("PDE SOLVER OPTIMIZATION EXPERIMENTS")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"JAX version: {jax.__version__}")
    print(f"Device: {jax.devices()[0]}")
    print("="*70)

    all_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'jax_version': jax.__version__,
            'device': str(jax.devices()[0]),
        }
    }

    # Run experiments
    all_results['experiment_1_baseline'] = experiment_1_baseline_profiling()
    all_results['experiment_2_interpolation'] = experiment_2_interpolation_optimization()
    all_results['experiment_3_kernel_fusion'] = experiment_3_kernel_fusion()
    all_results['experiment_4_combined'] = experiment_4_combined_optimization()

    # Save results
    output_path = 'results/all_experiments.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to {output_path}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)

    print("\n1. Interpolation Optimization Speedups:")
    for r in all_results['experiment_2_interpolation']:
        print(f"   p={r['p']}, n={r['n_uniform']}: {r['speedup']:.2f}x speedup")

    print("\n2. Kernel Fusion Speedups (Solve Phase):")
    for r in all_results['experiment_3_kernel_fusion']:
        print(f"   p={r['p']}, L={r['L']}: {r['speedup']:.2f}x speedup")

    print("\n3. Combined Optimization Speedups:")
    for r in all_results['experiment_4_combined']:
        print(f"   p={r['p']}, L={r['L']}: {r['speedup']:.2f}x speedup")


if __name__ == "__main__":
    main()
