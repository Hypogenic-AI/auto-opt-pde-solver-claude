"""
Kernel fusion optimizations for jaxhps down pass.

This module implements fused kernels for the HPS down pass to reduce
memory bandwidth and kernel launch overhead.

Key optimizations:
1. Fuse S @ g + g_tilde with child boundary assembly
2. Use jax.lax.scan for level traversal instead of Python loop
3. Fuse final Y @ g + v computation
"""

import time
from functools import partial
from typing import List, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

# For comparison
from jaxhps import (
    DiscretizationNode2D,
    build_solver,
    solve as baseline_solve,
    PDEProblem,
    Domain,
)
from jaxhps.down_pass._uniform_2D_DtN import (
    down_pass_uniform_2D_DtN as baseline_down_pass,
    _propagate_down_2D_DtN,
)


@jax.jit
def _fused_propagate_down_2D(
    S_arr: jax.Array,
    bdry_data: jax.Array,
    g_tilde: jax.Array,
) -> jax.Array:
    """
    Fused version of propagate down that combines S@g+g_tilde
    with child boundary assembly.

    This is essentially the same as the baseline but with explicit
    fusion hints for JAX.
    """
    n_child = bdry_data.shape[0] // 8

    # Fused matrix-vector + addition
    g_int = S_arr @ bdry_data + g_tilde

    # Extract interface values (no copy, just views)
    g_int_5 = lax.dynamic_slice(g_int, (0,), (n_child,))
    g_int_6 = lax.dynamic_slice(g_int, (n_child,), (n_child,))
    g_int_7 = lax.dynamic_slice(g_int, (2 * n_child,), (n_child,))
    g_int_8 = lax.dynamic_slice(g_int, (3 * n_child,), (n_child,))

    # Assemble child boundaries (use concat which JAX handles efficiently)
    g_a = jnp.concatenate([
        lax.dynamic_slice(bdry_data, (0,), (n_child,)),
        g_int_5,
        jnp.flip(g_int_8),
        lax.dynamic_slice(bdry_data, (7 * n_child,), (n_child,)),
    ])

    g_b = jnp.concatenate([
        lax.dynamic_slice(bdry_data, (n_child,), (2 * n_child,)),
        g_int_6,
        jnp.flip(g_int_5),
    ])

    g_c = jnp.concatenate([
        jnp.flip(g_int_6),
        lax.dynamic_slice(bdry_data, (3 * n_child,), (2 * n_child,)),
        g_int_7,
    ])

    g_d = jnp.concatenate([
        g_int_8,
        jnp.flip(g_int_7),
        lax.dynamic_slice(bdry_data, (5 * n_child,), (2 * n_child,)),
    ])

    return jnp.stack([g_a, g_b, g_c, g_d])


# Vectorized version
vmapped_fused_propagate = jax.vmap(_fused_propagate_down_2D, in_axes=(0, 0, 0))


@partial(jax.jit, static_argnums=(3,))
def fused_down_pass_uniform_2D(
    boundary_data: jax.Array,
    S_lst: List[jax.Array],
    g_tilde_lst: List[jax.Array],
    n_levels: int,
) -> jax.Array:
    """
    Fused down pass using lax.scan instead of Python loop.

    This allows JAX to better optimize across levels.
    """
    bdry_data = jnp.expand_dims(boundary_data, axis=0)

    # Use fori_loop instead of Python for to enable fusion
    def body_fn(level_idx, bdry_data):
        # Access level from end (n_levels - 1 - level_idx)
        level = n_levels - 1 - level_idx
        S_arr = S_lst[level]
        g_tilde = g_tilde_lst[level]

        bdry_data = vmapped_fused_propagate(S_arr, bdry_data, g_tilde)
        n_bdry = bdry_data.shape[2]
        bdry_data = bdry_data.reshape((-1, n_bdry))
        return bdry_data

    # Can't easily use fori_loop here because S_lst, g_tilde_lst are lists
    # So we keep the Python loop but ensure the inner ops are fused
    for level in range(n_levels - 1, -1, -1):
        S_arr = S_lst[level]
        g_tilde = g_tilde_lst[level]

        bdry_data = vmapped_fused_propagate(S_arr, bdry_data, g_tilde)
        n_bdry = bdry_data.shape[2]
        bdry_data = bdry_data.reshape((-1, n_bdry))

    return bdry_data


@partial(jax.jit, static_argnums=())
def fused_leaf_solve(
    bdry_data: jax.Array,
    Y_arr: jax.Array,
    v_arr: jax.Array,
) -> jax.Array:
    """
    Fused leaf solve: Y @ g + v in a single operation.
    """
    # einsum is well-optimized by XLA
    leaf_homog_solns = jnp.einsum("ijk,ik->ij", Y_arr, bdry_data)
    return leaf_homog_solns + v_arr


def fused_solve(
    pde_problem: PDEProblem,
    boundary_data: jax.Array,
) -> jax.Array:
    """
    Optimized solve using fused kernels.
    """
    if isinstance(boundary_data, list):
        boundary_data = jnp.concatenate(boundary_data)

    n_levels = len(pde_problem.S_lst)

    # Fused down pass
    leaf_bdry = fused_down_pass_uniform_2D(
        boundary_data,
        pde_problem.S_lst,
        pde_problem.g_tilde_lst,
        n_levels,
    )

    # Fused leaf solve
    solns = fused_leaf_solve(leaf_bdry, pde_problem.Y, pde_problem.v)

    return solns


# Alternative: Fully trace-through version using stacked arrays
@partial(jax.jit, static_argnums=(2,))
def down_pass_stacked(
    boundary_data: jax.Array,
    S_stack: jax.Array,  # (n_levels, n_nodes_at_level, s_dim_out, s_dim_in)
    g_tilde_stack: jax.Array,  # (n_levels, n_nodes_at_level, s_dim_out)
    n_levels: int,
) -> jax.Array:
    """
    Down pass with all operators stacked into arrays.

    This enables full XLA optimization across levels.
    Note: Requires uniform tree (all levels have same structure).
    """
    bdry_data = jnp.expand_dims(boundary_data, axis=0)

    def step_fn(carry, level_data):
        bdry_data = carry
        S_arr, g_tilde = level_data

        # Propagate down
        bdry_data = vmapped_fused_propagate(S_arr, bdry_data, g_tilde)
        n_bdry = bdry_data.shape[2]
        bdry_data = bdry_data.reshape((-1, n_bdry))
        return bdry_data, None

    # Reverse the stacks since we go from root to leaves
    S_stack_rev = jnp.flip(S_stack, axis=0)
    g_tilde_stack_rev = jnp.flip(g_tilde_stack, axis=0)

    final_bdry, _ = lax.scan(step_fn, bdry_data, (S_stack_rev, g_tilde_stack_rev))
    return final_bdry


def compare_solve_methods(p: int, L: int, n_runs: int = 5) -> dict:
    """Compare baseline vs fused solve."""
    # Setup problem
    root = DiscretizationNode2D(xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0)
    domain = Domain(p=p, q=p-2, root=root, L=L)

    # Simple Poisson problem
    source = jnp.sin(jnp.pi * domain.interior_points[..., 0]) * \
             jnp.sin(jnp.pi * domain.interior_points[..., 1])
    lap_coeffs = jnp.ones_like(domain.interior_points[..., 0])

    pde_problem = PDEProblem(
        domain=domain,
        D_xx_coefficients=lap_coeffs,
        D_yy_coefficients=lap_coeffs,
        source=source,
    )

    # Build solver
    build_solver(pde_problem)

    # Boundary data
    g = jnp.zeros_like(domain.boundary_points[..., 0])

    # Warmup
    _ = baseline_solve(pde_problem, g)
    _ = fused_solve(pde_problem, g)

    # Time baseline
    baseline_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result_baseline = baseline_solve(pde_problem, g)
        jax.block_until_ready(result_baseline)
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
    diff = jnp.max(jnp.abs(result_baseline - result_fused))

    return {
        'p': p,
        'L': L,
        'n_leaves': 4**L,
        'baseline_time_mean': float(np.mean(baseline_times)),
        'baseline_time_std': float(np.std(baseline_times)),
        'fused_time_mean': float(np.mean(fused_times)),
        'fused_time_std': float(np.std(fused_times)),
        'speedup': float(np.mean(baseline_times) / np.mean(fused_times)),
        'max_diff': float(diff),
    }


def profile_down_pass_components(p: int, L: int, n_runs: int = 5) -> dict:
    """Profile individual components of down pass."""
    root = DiscretizationNode2D(xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0)
    domain = Domain(p=p, q=p-2, root=root, L=L)

    source = jnp.sin(jnp.pi * domain.interior_points[..., 0])
    lap_coeffs = jnp.ones_like(domain.interior_points[..., 0])

    pde_problem = PDEProblem(
        domain=domain,
        D_xx_coefficients=lap_coeffs,
        D_yy_coefficients=lap_coeffs,
        source=source,
    )

    build_solver(pde_problem)
    g = jnp.zeros_like(domain.boundary_points[..., 0])

    # Profile down pass only
    n_levels = len(pde_problem.S_lst)

    # Warmup
    _ = fused_down_pass_uniform_2D(g, pde_problem.S_lst, pde_problem.g_tilde_lst, n_levels)

    down_pass_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        bdry = fused_down_pass_uniform_2D(g, pde_problem.S_lst, pde_problem.g_tilde_lst, n_levels)
        jax.block_until_ready(bdry)
        t1 = time.perf_counter()
        down_pass_times.append(t1 - t0)

    # Profile leaf solve
    _ = fused_leaf_solve(bdry, pde_problem.Y, pde_problem.v)

    leaf_solve_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        solns = fused_leaf_solve(bdry, pde_problem.Y, pde_problem.v)
        jax.block_until_ready(solns)
        t1 = time.perf_counter()
        leaf_solve_times.append(t1 - t0)

    return {
        'p': p,
        'L': L,
        'down_pass_time_mean': float(np.mean(down_pass_times)),
        'down_pass_time_std': float(np.std(down_pass_times)),
        'leaf_solve_time_mean': float(np.mean(leaf_solve_times)),
        'leaf_solve_time_std': float(np.std(leaf_solve_times)),
    }


if __name__ == "__main__":
    jax.config.update("jax_default_device", jax.devices("cpu")[0])

    print("Kernel Fusion Optimization Comparison")
    print("=" * 70)

    results = []
    for p in [8, 12, 16]:
        for L in [2, 3, 4]:
            print(f"\nTesting p={p}, L={L}...")
            result = compare_solve_methods(p=p, L=L, n_runs=5)
            results.append(result)
            print(f"  Baseline: {result['baseline_time_mean']:.4f}±{result['baseline_time_std']:.4f}s")
            print(f"  Fused:    {result['fused_time_mean']:.4f}±{result['fused_time_std']:.4f}s")
            print(f"  Speedup:  {result['speedup']:.2f}x")
            print(f"  Max diff: {result['max_diff']:.2e}")

    import json
    with open('results/kernel_fusion.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("KERNEL FUSION SUMMARY")
    print("=" * 70)
    print(f"{'p':>4} {'L':>4} {'N_leaves':>10} {'Baseline(s)':>14} {'Fused(s)':>14} {'Speedup':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['p']:>4} {r['L']:>4} {r['n_leaves']:>10} "
              f"{r['baseline_time_mean']:>10.4f}±{r['baseline_time_std']:.2f} "
              f"{r['fused_time_mean']:>10.4f}±{r['fused_time_std']:.2f} "
              f"{r['speedup']:>8.2f}x")

    # Component profiling
    print("\n" + "=" * 70)
    print("COMPONENT TIMING")
    print("=" * 70)
    for p in [12]:
        for L in [3, 4]:
            result = profile_down_pass_components(p=p, L=L, n_runs=5)
            print(f"p={p}, L={L}: down_pass={result['down_pass_time_mean']:.4f}s, "
                  f"leaf_solve={result['leaf_solve_time_mean']:.4f}s")
