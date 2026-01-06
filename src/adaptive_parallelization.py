"""
Parallelization optimization for adaptive grid operations in jaxhps.

The adaptive discretization in jaxhps uses Python loops for mesh refinement
and level restriction checks. This module explores parallel alternatives.

Key insights:
1. The refinement check can be vectorized across all nodes at a level
2. Level restriction can use scatter/gather operations
3. The merge step can be batched across nodes at the same tree level
"""

import time
from functools import partial
from typing import List, Tuple, Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from jaxhps import DiscretizationNode2D, Domain
from jaxhps._discretization_tree import get_all_leaves, get_depth
from jaxhps._adaptive_discretization_2D import (
    generate_adaptive_mesh_level_restriction_2D,
)
from jaxhps._precompute_operators_2D import precompute_L_4f1


def profile_adaptive_mesh_generation(
    f_fn: Callable,
    tol_vals: List[float] = [1e-2, 1e-3, 1e-4],
    p: int = 12,
    q: int = 10,
    n_runs: int = 3,
) -> List[dict]:
    """
    Profile adaptive mesh generation for different tolerance values.

    The mesh generation is inherently sequential due to level restriction,
    but we can profile to understand the bottlenecks.
    """
    results = []

    for tol in tol_vals:
        times = []
        n_leaves_list = []

        for _ in range(n_runs):
            root = DiscretizationNode2D(xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0)

            t0 = time.perf_counter()
            generate_adaptive_mesh_level_restriction_2D(
                root=root,
                f_fn=f_fn,
                tol=tol,
                p=p,
                q=q,
                restrict_bool=True,
                l2_norm=False,
            )
            t1 = time.perf_counter()

            times.append(t1 - t0)
            n_leaves_list.append(len(list(get_all_leaves(root))))

        results.append({
            'tol': tol,
            'time_mean': float(np.mean(times)),
            'time_std': float(np.std(times)),
            'n_leaves': n_leaves_list[0],
            'depth': get_depth(root),
        })

        print(f"tol={tol:.0e}: {np.mean(times):.4f}±{np.std(times):.4f}s, "
              f"n_leaves={n_leaves_list[0]}, depth={get_depth(root)}")

    return results


@partial(jax.jit, static_argnums=(2,))
def batched_refinement_check(
    bounds_arr: jax.Array,
    f_fn: Callable,
    p: int,
    L_4f1: jax.Array,
    tol: float,
    global_nrm: float,
) -> Tuple[jax.Array, jax.Array]:
    """
    Batched refinement check for multiple nodes simultaneously.

    Args:
        bounds_arr: Array of node bounds. Shape (n_nodes, 4) = [xmin, xmax, ymin, ymax]
        f_fn: Function to evaluate
        p: Polynomial order
        L_4f1: Interpolation operator from 1 panel to 4 children
        tol: Tolerance
        global_nrm: Current estimate of global norm

    Returns:
        (needs_refine, new_nrm): Boolean array and updated norm
    """
    from jaxhps._grid_creation_2D import (
        compute_interior_Chebyshev_points_uniform_2D,
    )
    from jaxhps._adaptive_discretization_3D import check_current_discretization_global_linf_norm

    def check_single(bounds):
        """Check refinement for single node."""
        node = DiscretizationNode2D(
            xmin=bounds[0], xmax=bounds[1],
            ymin=bounds[2], ymax=bounds[3],
        )

        # Points at current level
        points_0 = compute_interior_Chebyshev_points_uniform_2D(
            node, L=0, p=p
        ).reshape(-1, 2)

        # Points at refined level
        points_1 = compute_interior_Chebyshev_points_uniform_2D(
            node, L=1, p=p
        ).reshape(-1, 2)

        f_evals = f_fn(points_0)
        f_evals_refined = f_fn(points_1)

        return check_current_discretization_global_linf_norm(
            f_evals, f_evals_refined, L_4f1, tol, global_nrm
        )

    # Vectorize over all nodes
    checks_bool, linf_nrms = jax.vmap(check_single)(bounds_arr)
    new_global_nrm = jnp.max(linf_nrms)

    # needs_refine is where checks_bool is False
    needs_refine = ~checks_bool

    return needs_refine, new_global_nrm


def analyze_adaptive_down_pass(p: int = 12, L: int = 3) -> dict:
    """
    Analyze timing breakdown of adaptive down pass.

    The adaptive down pass in jaxhps processes nodes level by level,
    handling compression lists for each merge interface.
    """
    from jaxhps import PDEProblem, build_solver, solve
    from jaxhps.down_pass._adaptive_2D_DtN import down_pass_adaptive_2D_DtN

    # Create an adaptive mesh
    def test_fn(x):
        """Function with local features requiring adaptation."""
        r = jnp.sqrt(x[..., 0]**2 + x[..., 1]**2)
        return jnp.exp(-50 * r**2) + jnp.sin(20 * x[..., 0])

    root = DiscretizationNode2D(xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0)
    generate_adaptive_mesh_level_restriction_2D(
        root=root,
        f_fn=test_fn,
        tol=1e-3,
        p=p,
        q=p-2,
        restrict_bool=True,
        l2_norm=False,
    )

    n_leaves = len(list(get_all_leaves(root)))
    depth = get_depth(root)

    # Create PDE problem with this mesh
    domain = Domain(p=p, q=p-2, root=root)

    lap_coeffs = jnp.ones_like(domain.interior_points[..., 0])
    source = test_fn(domain.interior_points)

    pde_problem = PDEProblem(
        domain=domain,
        D_xx_coefficients=lap_coeffs,
        D_yy_coefficients=lap_coeffs,
        source=source,
    )

    # Build solver
    t_build_start = time.perf_counter()
    build_solver(pde_problem)
    jax.block_until_ready(pde_problem.v)
    t_build_end = time.perf_counter()

    # Get boundary data
    bdry_data_lst = domain.get_adaptive_boundary_data_lst(
        lambda x: jnp.zeros_like(x[..., 0])
    )

    # Time the solve
    # Warmup
    _ = solve(pde_problem, bdry_data_lst)

    solve_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        result = solve(pde_problem, bdry_data_lst)
        jax.block_until_ready(result)
        t1 = time.perf_counter()
        solve_times.append(t1 - t0)

    return {
        'p': p,
        'n_leaves': n_leaves,
        'depth': depth,
        'build_time': t_build_end - t_build_start,
        'solve_time_mean': float(np.mean(solve_times)),
        'solve_time_std': float(np.std(solve_times)),
    }


def compare_uniform_vs_adaptive(p: int = 12, L: int = 3) -> dict:
    """
    Compare uniform vs adaptive discretization performance.

    This helps quantify the overhead of adaptive handling.
    """
    from jaxhps import PDEProblem, build_solver, solve

    results = {}

    # Uniform discretization
    root_uniform = DiscretizationNode2D(xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0)
    domain_uniform = Domain(p=p, q=p-2, root=root_uniform, L=L)

    source_uniform = jnp.sin(jnp.pi * domain_uniform.interior_points[..., 0])
    lap_coeffs_uniform = jnp.ones_like(domain_uniform.interior_points[..., 0])

    pde_uniform = PDEProblem(
        domain=domain_uniform,
        D_xx_coefficients=lap_coeffs_uniform,
        D_yy_coefficients=lap_coeffs_uniform,
        source=source_uniform,
    )

    t0 = time.perf_counter()
    build_solver(pde_uniform)
    jax.block_until_ready(pde_uniform.v)
    t1 = time.perf_counter()
    results['uniform_build_time'] = t1 - t0

    g_uniform = jnp.zeros_like(domain_uniform.boundary_points[..., 0])

    # Warmup
    _ = solve(pde_uniform, g_uniform)

    uniform_solve_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        result = solve(pde_uniform, g_uniform)
        jax.block_until_ready(result)
        t1 = time.perf_counter()
        uniform_solve_times.append(t1 - t0)

    results['uniform_solve_time_mean'] = float(np.mean(uniform_solve_times))
    results['uniform_solve_time_std'] = float(np.std(uniform_solve_times))
    results['uniform_n_leaves'] = 4**L

    # Adaptive discretization with same depth (for comparison)
    def test_fn(x):
        r = jnp.sqrt(x[..., 0]**2 + x[..., 1]**2)
        return jnp.exp(-50 * r**2)

    root_adaptive = DiscretizationNode2D(xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0)
    generate_adaptive_mesh_level_restriction_2D(
        root=root_adaptive,
        f_fn=test_fn,
        tol=1e-2,
        p=p,
        q=p-2,
        restrict_bool=True,
        l2_norm=False,
    )

    domain_adaptive = Domain(p=p, q=p-2, root=root_adaptive)
    n_leaves_adaptive = len(list(get_all_leaves(root_adaptive)))

    source_adaptive = test_fn(domain_adaptive.interior_points)
    lap_coeffs_adaptive = jnp.ones_like(domain_adaptive.interior_points[..., 0])

    pde_adaptive = PDEProblem(
        domain=domain_adaptive,
        D_xx_coefficients=lap_coeffs_adaptive,
        D_yy_coefficients=lap_coeffs_adaptive,
        source=source_adaptive,
    )

    t0 = time.perf_counter()
    build_solver(pde_adaptive)
    jax.block_until_ready(pde_adaptive.v)
    t1 = time.perf_counter()
    results['adaptive_build_time'] = t1 - t0

    bdry_adaptive = domain_adaptive.get_adaptive_boundary_data_lst(
        lambda x: jnp.zeros_like(x[..., 0])
    )

    # Warmup
    _ = solve(pde_adaptive, bdry_adaptive)

    adaptive_solve_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        result = solve(pde_adaptive, bdry_adaptive)
        jax.block_until_ready(result)
        t1 = time.perf_counter()
        adaptive_solve_times.append(t1 - t0)

    results['adaptive_solve_time_mean'] = float(np.mean(adaptive_solve_times))
    results['adaptive_solve_time_std'] = float(np.std(adaptive_solve_times))
    results['adaptive_n_leaves'] = n_leaves_adaptive
    results['adaptive_depth'] = get_depth(root_adaptive)
    results['p'] = p
    results['L_uniform'] = L

    return results


if __name__ == "__main__":
    jax.config.update("jax_default_device", jax.devices("cpu")[0])

    print("Adaptive Grid Parallelization Analysis")
    print("=" * 70)

    # Test function with localized features
    def localized_fn(x):
        r = jnp.sqrt(x[..., 0]**2 + x[..., 1]**2)
        return jnp.exp(-50 * r**2) + 0.1 * jnp.sin(10 * x[..., 0])

    print("\n1. Adaptive Mesh Generation Profiling")
    print("-" * 70)
    mesh_results = profile_adaptive_mesh_generation(
        f_fn=localized_fn,
        tol_vals=[1e-2, 1e-3, 1e-4],
        p=12,
        q=10,
        n_runs=3,
    )

    print("\n2. Adaptive Down Pass Analysis")
    print("-" * 70)
    adaptive_analysis = analyze_adaptive_down_pass(p=12, L=3)
    print(f"Adaptive solve: n_leaves={adaptive_analysis['n_leaves']}, "
          f"depth={adaptive_analysis['depth']}")
    print(f"Build time: {adaptive_analysis['build_time']:.4f}s")
    print(f"Solve time: {adaptive_analysis['solve_time_mean']:.4f}±"
          f"{adaptive_analysis['solve_time_std']:.4f}s")

    print("\n3. Uniform vs Adaptive Comparison")
    print("-" * 70)
    comparison = compare_uniform_vs_adaptive(p=12, L=3)
    print(f"Uniform (L={comparison['L_uniform']}): "
          f"n_leaves={comparison['uniform_n_leaves']}, "
          f"build={comparison['uniform_build_time']:.4f}s, "
          f"solve={comparison['uniform_solve_time_mean']:.4f}±"
          f"{comparison['uniform_solve_time_std']:.4f}s")
    print(f"Adaptive: "
          f"n_leaves={comparison['adaptive_n_leaves']}, "
          f"depth={comparison['adaptive_depth']}, "
          f"build={comparison['adaptive_build_time']:.4f}s, "
          f"solve={comparison['adaptive_solve_time_mean']:.4f}±"
          f"{comparison['adaptive_solve_time_std']:.4f}s")

    # Save results
    import json
    all_results = {
        'mesh_generation': mesh_results,
        'adaptive_analysis': adaptive_analysis,
        'uniform_vs_adaptive': comparison,
    }
    with open('results/adaptive_parallelization.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\nResults saved to results/adaptive_parallelization.json")
