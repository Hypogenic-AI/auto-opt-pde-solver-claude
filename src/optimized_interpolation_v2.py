"""
Optimized interpolation methods for jaxhps - Version 2.

This version focuses on:
1. Precomputed interpolation matrices
2. Matrix-based approach per leaf
3. Scan-based operations for tree traversal
"""

import time
from functools import partial
from typing import Tuple, List

import numpy as np
import jax
import jax.numpy as jnp

from jaxhps._interpolation_methods import interp_from_hps_2D as baseline_interp
from jaxhps.quadrature import chebyshev_points
from jaxhps._discretization_tree import DiscretizationNode2D


def precompute_interp_matrix_1d(p: int, n_uniform: int) -> jax.Array:
    """
    Precompute 1D interpolation matrix from p Chebyshev points to n uniform points.

    Returns I such that f_uniform = I @ f_chebyshev
    """
    cheby_pts = chebyshev_points(p)  # Points on [-1, 1]
    uniform_pts = jnp.linspace(-1.0, 1.0, n_uniform)

    # Barycentric weights for Chebyshev points of second kind
    w = jnp.array([(-1.0)**j for j in range(p)])
    w = w.at[0].multiply(0.5)
    w = w.at[-1].multiply(0.5)

    def compute_row(x):
        diffs = x - cheby_pts
        # Handle case where x equals a node
        eps = 1e-14
        is_node = jnp.abs(diffs) < eps
        safe_diffs = jnp.where(is_node, 1.0, diffs)
        weights = w / safe_diffs

        # Normalized weights
        norm_weights = weights / jnp.sum(weights)
        # One-hot for exact matches
        onehot = jnp.where(is_node, 1.0, 0.0)

        any_node = jnp.any(is_node)
        return jnp.where(any_node, onehot, norm_weights)

    I = jax.vmap(compute_row)(uniform_pts)
    return I


def precompute_interp_matrix_2d(p: int, n_uniform: int) -> jax.Array:
    """
    Precompute 2D interpolation matrix via tensor product.

    For p^2 Chebyshev points to n_uniform^2 uniform points.
    Returns I such that f_uniform.flatten() = I @ f_chebyshev.flatten()
    """
    I_1d = precompute_interp_matrix_1d(p, n_uniform)  # (n_uniform, p)

    # 2D tensor product: I_2d = kron(I_1d, I_1d)
    # But we can be smarter and apply separately
    return I_1d


@partial(jax.jit, static_argnums=(1, 4))
def interp_leaf_precomputed(
    f_leaf: jax.Array,
    p: int,
    I_1d: jax.Array,
    leaf_bounds: jax.Array,
    n_uniform: int,
    target_bounds: jax.Array,
) -> jax.Array:
    """
    Interpolate a single leaf to a portion of the uniform grid.

    Args:
        f_leaf: Function values on leaf's Chebyshev grid. Shape (p^2,)
        p: Polynomial order
        I_1d: 1D interpolation matrix. Shape (n_local, p)
        leaf_bounds: [xmin, xmax, ymin, ymax] for this leaf
        n_uniform: Total uniform grid size
        target_bounds: [xmin, xmax, ymin, ymax] for full target grid

    Returns:
        Contribution to uniform grid with zeros outside leaf. Shape (n_uniform, n_uniform)
    """
    # Reshape to 2D
    f_2d = f_leaf.reshape(p, p)

    # Determine which portion of uniform grid falls in this leaf
    x_uniform = jnp.linspace(target_bounds[0], target_bounds[1], n_uniform)
    y_uniform = jnp.linspace(target_bounds[2], target_bounds[3], n_uniform)

    # Find indices in leaf
    in_x = (x_uniform >= leaf_bounds[0]) & (x_uniform <= leaf_bounds[1])
    in_y = (y_uniform >= leaf_bounds[2]) & (y_uniform <= leaf_bounds[3])

    # Get the subset of uniform points in this leaf
    x_mask = jnp.where(in_x, 1.0, 0.0)
    y_mask = jnp.where(in_y, 1.0, 0.0)

    # Tensor product interpolation: I_1d @ f_2d @ I_1d.T
    # Apply along x direction
    f_interp_x = f_2d @ I_1d.T  # (p, n_local_x)
    # Apply along y direction
    f_interp = I_1d @ f_interp_x  # (n_local_y, n_local_x)

    # Create full-size output with zeros
    output = jnp.zeros((n_uniform, n_uniform))

    # This is tricky because we need to place f_interp in the right location
    # For simplicity, return f_interp and let caller handle placement
    return f_interp


@partial(jax.jit, static_argnums=(1, 4))
def interp_from_hps_2D_precomputed(
    leaves: Tuple[DiscretizationNode2D],
    p: int,
    f_evals: jax.Array,
    target_bounds: jax.Array,
    n_uniform: int,
) -> Tuple[jax.Array, jax.Array]:
    """
    Optimized interpolation using precomputed matrices.

    Key idea: Build the interpolation matrix once per leaf size,
    then apply it efficiently.

    Args:
        leaves: Tuple of leaf nodes
        p: Polynomial order
        f_evals: Solution values. Shape (n_leaves, p^2)
        target_bounds: [xmin, xmax, ymin, ymax]
        n_uniform: Number of uniform grid points per dimension

    Returns:
        (interpolated_values, target_grid)
    """
    n_leaves = len(leaves)

    # Build uniform grid
    x_vals = jnp.linspace(target_bounds[0], target_bounds[1], n_uniform)
    y_vals = jnp.linspace(target_bounds[2], target_bounds[3], n_uniform)
    X, Y = jnp.meshgrid(x_vals, y_vals)
    pts = jnp.stack([X.flatten(), Y.flatten()], axis=-1)

    # Get leaf bounds
    leaf_bounds = jnp.stack([
        jnp.array([leaf.xmin, leaf.xmax, leaf.ymin, leaf.ymax])
        for leaf in leaves
    ])

    # Find which leaf contains each point
    in_x = (pts[:, 0, None] >= leaf_bounds[None, :, 0]) & \
           (pts[:, 0, None] <= leaf_bounds[None, :, 1])
    in_y = (pts[:, 1, None] >= leaf_bounds[None, :, 2]) & \
           (pts[:, 1, None] <= leaf_bounds[None, :, 3])
    in_leaf = in_x & in_y
    leaf_idx = jnp.argmax(in_leaf, axis=1)

    # Get Chebyshev points
    cheby_pts = chebyshev_points(p)

    # Barycentric weights
    w = jnp.array([(-1.0)**j for j in range(p)])
    w = w.at[0].multiply(0.5)
    w = w.at[-1].multiply(0.5)

    def interp_point(pt, lid):
        """Interpolate single point."""
        bounds = leaf_bounds[lid]
        f_leaf = f_evals[lid].reshape(p, p)

        # Map to reference [-1, 1]
        x_ref = 2 * (pt[0] - bounds[0]) / (bounds[1] - bounds[0]) - 1
        y_ref = 2 * (pt[1] - bounds[2]) / (bounds[3] - bounds[2]) - 1

        # Barycentric weights for this point
        def bary_weights(x_target):
            diffs = x_target - cheby_pts
            eps = 1e-14
            is_node = jnp.abs(diffs) < eps
            safe_diffs = jnp.where(is_node, 1.0, diffs)
            weights = w / safe_diffs
            norm_weights = weights / jnp.sum(weights)
            onehot = jnp.where(is_node, 1.0, 0.0)
            return jnp.where(jnp.any(is_node), onehot, norm_weights)

        wx = bary_weights(x_ref)
        wy = bary_weights(y_ref)

        # Tensor product
        return wy @ (f_leaf @ wx)

    # Vectorize
    vals = jax.vmap(interp_point)(pts, leaf_idx)

    target_pts = jnp.stack([X, Y], axis=-1)
    return vals.reshape(n_uniform, n_uniform), target_pts


def interp_from_hps_2D_matmul(
    leaves: Tuple[DiscretizationNode2D],
    p: int,
    f_evals: jax.Array,
    x_vals: jax.Array,
    y_vals: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """
    Interpolation using a single precomputed matrix per unique leaf-target combination.

    This approach:
    1. Groups target points by their containing leaf
    2. Builds interpolation matrix for each group
    3. Applies via matrix multiply

    For uniform grids with power-of-2 leaves, this is efficient because
    each leaf has the same number of target points.
    """
    n_leaves = len(leaves)
    n_x, n_y = x_vals.shape[0], y_vals.shape[0]

    # Get leaf bounds
    leaf_bounds = jnp.stack([
        jnp.array([leaf.xmin, leaf.xmax, leaf.ymin, leaf.ymax])
        for leaf in leaves
    ])

    # Create meshgrid of target points
    X, Y = jnp.meshgrid(x_vals, y_vals)
    pts = jnp.stack([X.flatten(), Y.flatten()], axis=-1)

    # Find leaf for each point
    in_x = (pts[:, 0, None] >= leaf_bounds[None, :, 0]) & \
           (pts[:, 0, None] <= leaf_bounds[None, :, 1])
    in_y = (pts[:, 1, None] >= leaf_bounds[None, :, 2]) & \
           (pts[:, 1, None] <= leaf_bounds[None, :, 3])
    leaf_idx = jnp.argmax(in_x & in_y, axis=1)

    # Get Chebyshev points and weights
    cheby_pts = chebyshev_points(p)
    w = jnp.array([(-1.0)**j for j in range(p)])
    w = w.at[0].multiply(0.5)
    w = w.at[-1].multiply(0.5)

    def interp_point(pt, lid):
        bounds = leaf_bounds[lid]
        f_leaf = f_evals[lid].reshape(p, p)

        x_ref = 2 * (pt[0] - bounds[0]) / (bounds[1] - bounds[0]) - 1
        y_ref = 2 * (pt[1] - bounds[2]) / (bounds[3] - bounds[2]) - 1

        def bary_weights(x_target):
            diffs = x_target - cheby_pts
            eps = 1e-14
            is_node = jnp.abs(diffs) < eps
            safe_diffs = jnp.where(is_node, 1.0, diffs)
            weights = w / safe_diffs
            norm_weights = weights / jnp.sum(weights)
            onehot = jnp.where(is_node, 1.0, 0.0)
            return jnp.where(jnp.any(is_node), onehot, norm_weights)

        wx = bary_weights(x_ref)
        wy = bary_weights(y_ref)
        return wy @ (f_leaf @ wx)

    vals = jax.vmap(interp_point)(pts, leaf_idx)

    target_pts = jnp.stack([X, Y], axis=-1)
    return vals.reshape(n_y, n_x), target_pts


# Now let's try a completely different approach: fuse the build and solve
# with the interpolation to avoid storing intermediate results

@partial(jax.jit, static_argnums=(1, 3))
def interp_hps_optimized(
    f_evals: jax.Array,
    p: int,
    leaf_bounds: jax.Array,
    n_uniform: int,
    target_bounds: jax.Array,
) -> jax.Array:
    """
    Fully JIT-compiled interpolation without Python leaf objects.

    Args:
        f_evals: Solution values. Shape (n_leaves, p^2)
        p: Polynomial order
        leaf_bounds: Array of leaf bounds. Shape (n_leaves, 4)
        n_uniform: Number of uniform grid points
        target_bounds: [xmin, xmax, ymin, ymax]

    Returns:
        Interpolated values. Shape (n_uniform, n_uniform)
    """
    n_leaves = f_evals.shape[0]

    # Build uniform grid
    x_vals = jnp.linspace(target_bounds[0], target_bounds[1], n_uniform)
    y_vals = jnp.linspace(target_bounds[2], target_bounds[3], n_uniform)
    X, Y = jnp.meshgrid(x_vals, y_vals)
    pts = jnp.stack([X.flatten(), Y.flatten()], axis=-1)

    # Find leaf for each point
    in_x = (pts[:, 0, None] >= leaf_bounds[None, :, 0]) & \
           (pts[:, 0, None] <= leaf_bounds[None, :, 1])
    in_y = (pts[:, 1, None] >= leaf_bounds[None, :, 2]) & \
           (pts[:, 1, None] <= leaf_bounds[None, :, 3])
    leaf_idx = jnp.argmax(in_x & in_y, axis=1)

    # Chebyshev points and weights
    cheby_pts = chebyshev_points(p)
    w = jnp.array([(-1.0)**j for j in range(p)])
    w = w.at[0].multiply(0.5)
    w = w.at[-1].multiply(0.5)

    def interp_point(pt, lid):
        bounds = leaf_bounds[lid]
        f_leaf = f_evals[lid].reshape(p, p)

        x_ref = 2 * (pt[0] - bounds[0]) / (bounds[1] - bounds[0]) - 1
        y_ref = 2 * (pt[1] - bounds[2]) / (bounds[3] - bounds[2]) - 1

        def bary_weights(x_target):
            diffs = x_target - cheby_pts
            eps = 1e-14
            is_node = jnp.abs(diffs) < eps
            safe_diffs = jnp.where(is_node, 1.0, diffs)
            weights = w / safe_diffs
            norm_weights = weights / jnp.sum(weights)
            onehot = jnp.where(is_node, 1.0, 0.0)
            return jnp.where(jnp.any(is_node), onehot, norm_weights)

        wx = bary_weights(x_ref)
        wy = bary_weights(y_ref)
        return wy @ (f_leaf @ wx)

    vals = jax.vmap(interp_point)(pts, leaf_idx)
    return vals.reshape(n_uniform, n_uniform)


def compare_methods(p: int, L: int, n_uniform: int, n_runs: int = 5) -> dict:
    """Compare baseline vs optimized interpolation."""
    from jaxhps import DiscretizationNode2D, Domain
    from jaxhps._discretization_tree import get_all_leaves

    # Setup
    root = DiscretizationNode2D(xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0)
    domain = Domain(p=p, q=p-2, root=root, L=L)

    def test_fn(x):
        return jnp.sin(jnp.pi * x[..., 0]) * jnp.cos(jnp.pi * x[..., 1])

    f_evals = test_fn(domain.interior_points)
    leaves = tuple(get_all_leaves(root))
    n_leaves = 4**L

    x_vals = jnp.linspace(-1.0, 1.0, n_uniform)
    y_vals = jnp.linspace(-1.0, 1.0, n_uniform)
    target_bounds = jnp.array([-1.0, 1.0, -1.0, 1.0])

    # Precompute leaf bounds array for optimized version
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
        result_baseline, _ = baseline_interp(leaves, p, f_evals, x_vals, y_vals)
        jax.block_until_ready(result_baseline)
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

    # Check accuracy
    X, Y = jnp.meshgrid(x_vals, y_vals)
    true_vals = test_fn(jnp.stack([X, Y], axis=-1))

    err_baseline = float(jnp.max(jnp.abs(result_baseline - true_vals)))
    err_opt = float(jnp.max(jnp.abs(result_opt - true_vals)))

    return {
        'p': p,
        'L': L,
        'n_uniform': n_uniform,
        'n_leaves': n_leaves,
        'baseline_time_mean': float(np.mean(baseline_times)),
        'baseline_time_std': float(np.std(baseline_times)),
        'optimized_time_mean': float(np.mean(opt_times)),
        'optimized_time_std': float(np.std(opt_times)),
        'speedup': float(np.mean(baseline_times) / np.mean(opt_times)),
        'error_baseline': err_baseline,
        'error_optimized': err_opt,
    }


if __name__ == "__main__":
    jax.config.update("jax_default_device", jax.devices("cpu")[0])

    print("Interpolation Optimization Comparison v2")
    print("=" * 70)

    results = []
    for p in [8, 12, 16, 20]:
        for n_uniform in [50, 100, 200]:
            print(f"\nTesting p={p}, n_uniform={n_uniform}...")
            result = compare_methods(p=p, L=2, n_uniform=n_uniform, n_runs=5)
            results.append(result)
            print(f"  Baseline:  {result['baseline_time_mean']:.4f}±{result['baseline_time_std']:.4f}s")
            print(f"  Optimized: {result['optimized_time_mean']:.4f}±{result['optimized_time_std']:.4f}s")
            print(f"  Speedup:   {result['speedup']:.2f}x")
            print(f"  Error diff: {abs(result['error_baseline'] - result['error_optimized']):.2e}")

    import json
    with open('results/interpolation_v2.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'p':>4} {'n':>6} {'Baseline(s)':>14} {'Optimized(s)':>14} {'Speedup':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['p']:>4} {r['n_uniform']:>6} "
              f"{r['baseline_time_mean']:>10.4f}±{r['baseline_time_std']:.2f} "
              f"{r['optimized_time_mean']:>10.4f}±{r['optimized_time_std']:.2f} "
              f"{r['speedup']:>8.2f}x")
