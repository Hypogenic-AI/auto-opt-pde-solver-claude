"""
Baseline profiling of jaxhps PDE solver.

This script profiles the baseline jaxhps implementation to identify
bottlenecks and establish performance baselines.
"""

import time
import json
import sys
from typing import Dict, List, Tuple
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jaxhps import (
    DiscretizationNode2D,
    build_solver,
    solve,
    PDEProblem,
    Domain,
)

# Use CPU for consistent profiling
jax.config.update("jax_default_device", jax.devices("cpu")[0])

# Seed for reproducibility
np.random.seed(42)

# Problem parameters
XMIN, XMAX = -1.0, 1.0
YMIN, YMAX = -1.0, 1.0
K = 5
LAMBDA = 10
PI = jnp.pi


def problem_1_soln(x: jnp.ndarray) -> jnp.ndarray:
    """True solution: sin(pi * lambda * x) * sin(pi * y)"""
    return jnp.sin(PI * LAMBDA * x[..., 0]) * jnp.sin(PI * x[..., 1])


def problem_1_lap_coeffs(x: jnp.ndarray) -> jnp.ndarray:
    """Laplacian coefficient c(x,y) = 1"""
    return jnp.ones_like(x[..., 0])


def problem_1_d_x_coeffs(x: jnp.ndarray) -> jnp.ndarray:
    """D_x coefficient c(x,y) = -cos(ky)"""
    return -1 * jnp.cos(K * x[..., 1])


def problem_1_d_y_coeffs(x: jnp.ndarray) -> jnp.ndarray:
    """D_y coefficient c(x,y) = sin(ky)"""
    return jnp.sin(K * x[..., 1])


def problem_1_source(x: jnp.ndarray) -> jnp.ndarray:
    """Source term for the PDE."""
    term_1 = -1 * (PI**2) * (1 + LAMBDA**2) * problem_1_soln(x)
    term_2 = (
        -1 * PI * LAMBDA
        * jnp.cos(PI * LAMBDA * x[..., 0])
        * jnp.sin(PI * x[..., 1])
        * jnp.cos(K * x[..., 1])
    )
    term_3 = (
        PI
        * jnp.sin(PI * LAMBDA * x[..., 0])
        * jnp.cos(PI * x[..., 1])
        * jnp.sin(K * x[..., 1])
    )
    return term_1 + term_2 + term_3


def profile_single_run(p: int, L: int) -> Dict:
    """
    Profile a single solve with given polynomial order p and tree depth L.

    Returns timing and accuracy metrics.
    """
    results = {}

    # Setup
    root = DiscretizationNode2D(xmin=XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX)
    domain = Domain(p=p, q=p-2, root=root, L=L)

    n_leaves = 4**L
    n_interior_pts = n_leaves * p**2
    n_boundary_pts = 4 * L * p  # Approximate

    results['n_leaves'] = n_leaves
    results['n_interior_pts'] = n_interior_pts
    results['p'] = p
    results['L'] = L

    # Source and coefficients
    source = problem_1_source(domain.interior_points)
    lap_coeffs = problem_1_lap_coeffs(domain.interior_points)
    d_x_coeffs = problem_1_d_x_coeffs(domain.interior_points)
    d_y_coeffs = problem_1_d_y_coeffs(domain.interior_points)

    pde_problem = PDEProblem(
        domain=domain,
        D_xx_coefficients=lap_coeffs,
        D_yy_coefficients=lap_coeffs,
        D_x_coefficients=d_x_coeffs,
        D_y_coefficients=d_y_coeffs,
        source=source,
    )

    # Build phase timing
    t_build_start = time.perf_counter()
    build_solver(pde_problem)
    # Block until computation is done
    jax.block_until_ready(pde_problem.S_lst)
    t_build_end = time.perf_counter()
    results['build_time'] = t_build_end - t_build_start

    # Boundary data
    g = jnp.zeros_like(domain.boundary_points[..., 0])

    # Solve phase timing
    t_solve_start = time.perf_counter()
    computed_soln = solve(pde_problem, g)
    # Block until computation is done
    jax.block_until_ready(computed_soln)
    t_solve_end = time.perf_counter()
    results['solve_time'] = t_solve_end - t_solve_start

    # Total time
    results['total_time'] = results['build_time'] + results['solve_time']

    # Accuracy
    expected_soln = problem_1_soln(domain.interior_points)
    err = jnp.max(jnp.abs(computed_soln - expected_soln))
    nrm = jnp.max(jnp.abs(expected_soln))
    results['rel_error'] = float(err / nrm) if nrm > 0 else float(err)

    return results


def run_baseline_profiling(
    p_vals: List[int] = [8, 12, 16],
    L_vals: List[int] = [2, 3, 4],
    n_runs: int = 3,
) -> List[Dict]:
    """
    Run baseline profiling across parameter ranges.

    Args:
        p_vals: Polynomial orders to test
        L_vals: Tree depths to test
        n_runs: Number of runs per configuration

    Returns:
        List of result dictionaries
    """
    all_results = []

    print(f"Running baseline profiling with {len(p_vals)} p values, {len(L_vals)} L values, {n_runs} runs each")
    print("=" * 60)

    for p in p_vals:
        for L in L_vals:
            print(f"\nProfiling p={p}, L={L}...")

            run_results = []
            for run_idx in range(n_runs):
                try:
                    result = profile_single_run(p, L)
                    result['run_idx'] = run_idx
                    run_results.append(result)
                    print(f"  Run {run_idx+1}/{n_runs}: build={result['build_time']:.4f}s, "
                          f"solve={result['solve_time']:.4f}s, error={result['rel_error']:.2e}")
                except Exception as e:
                    print(f"  Run {run_idx+1}/{n_runs} FAILED: {e}")
                    continue

            if run_results:
                # Compute statistics
                build_times = [r['build_time'] for r in run_results]
                solve_times = [r['solve_time'] for r in run_results]

                summary = {
                    'p': p,
                    'L': L,
                    'n_leaves': run_results[0]['n_leaves'],
                    'n_interior_pts': run_results[0]['n_interior_pts'],
                    'build_time_mean': float(np.mean(build_times)),
                    'build_time_std': float(np.std(build_times)),
                    'solve_time_mean': float(np.mean(solve_times)),
                    'solve_time_std': float(np.std(solve_times)),
                    'total_time_mean': float(np.mean(build_times) + np.mean(solve_times)),
                    'rel_error': run_results[0]['rel_error'],
                    'n_runs': len(run_results),
                }
                all_results.append(summary)

                print(f"  Summary: build={summary['build_time_mean']:.4f}±{summary['build_time_std']:.4f}s, "
                      f"solve={summary['solve_time_mean']:.4f}±{summary['solve_time_std']:.4f}s")

    return all_results


def profile_interpolation(
    p_vals: List[int] = [8, 12, 16, 20],
    n_target_pts: List[int] = [50, 100, 200],
    n_runs: int = 5,
) -> List[Dict]:
    """
    Profile the interpolation operation specifically.

    This tests the interpolation from HPS (Chebyshev) grids to uniform grids.
    """
    from jaxhps._interpolation_methods import interp_from_hps_2D
    from jaxhps._discretization_tree import get_all_leaves

    results = []

    print("\nProfiling interpolation operation...")
    print("=" * 60)

    for p in p_vals:
        for n_pts in n_target_pts:
            print(f"\nInterpolation p={p}, n_target={n_pts}x{n_pts}...")

            # Create a simple domain and solve
            L = 2  # Fixed depth
            root = DiscretizationNode2D(xmin=XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX)
            domain = Domain(p=p, q=p-2, root=root, L=L)

            # Create fake solution data
            f_evals = jnp.ones((4**L, p**2))
            leaves = tuple(get_all_leaves(root))

            # Target grid
            x_vals = jnp.linspace(XMIN, XMAX, n_pts)
            y_vals = jnp.linspace(YMIN, YMAX, n_pts)

            # Warmup
            _ = interp_from_hps_2D(leaves, p, f_evals, x_vals, y_vals)

            # Time the interpolation
            times = []
            for _ in range(n_runs):
                t_start = time.perf_counter()
                result, _ = interp_from_hps_2D(leaves, p, f_evals, x_vals, y_vals)
                jax.block_until_ready(result)
                t_end = time.perf_counter()
                times.append(t_end - t_start)

            result_dict = {
                'p': p,
                'n_target_pts': n_pts,
                'interp_time_mean': float(np.mean(times)),
                'interp_time_std': float(np.std(times)),
                'n_source_pts': 4**L * p**2,
                'n_target_total': n_pts**2,
            }
            results.append(result_dict)

            print(f"  Time: {result_dict['interp_time_mean']:.6f}±{result_dict['interp_time_std']:.6f}s")

    return results


if __name__ == "__main__":
    print("JAX devices:", jax.devices())
    print()

    # Run baseline profiling
    baseline_results = run_baseline_profiling(
        p_vals=[8, 12, 16],
        L_vals=[2, 3, 4],
        n_runs=3
    )

    # Run interpolation profiling
    interp_results = profile_interpolation(
        p_vals=[8, 12, 16, 20],
        n_target_pts=[50, 100, 200],
        n_runs=5
    )

    # Save results
    all_results = {
        'baseline': baseline_results,
        'interpolation': interp_results,
        'metadata': {
            'jax_version': jax.__version__,
            'device': str(jax.devices()[0]),
        }
    }

    output_path = 'results/baseline_profiling.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("BASELINE SUMMARY")
    print("=" * 80)
    print(f"{'p':>4} {'L':>4} {'N_leaves':>10} {'Build(s)':>12} {'Solve(s)':>12} {'Error':>12}")
    print("-" * 80)
    for r in baseline_results:
        print(f"{r['p']:>4} {r['L']:>4} {r['n_leaves']:>10} "
              f"{r['build_time_mean']:>10.4f}±{r['build_time_std']:.2f} "
              f"{r['solve_time_mean']:>10.4f}±{r['solve_time_std']:.2f} "
              f"{r['rel_error']:>12.2e}")
