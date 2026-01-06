"""
Optimized interpolation methods for jaxhps.

This module implements FFT-based interpolation from Chebyshev (spectral)
grids to uniform grids, potentially faster than the baseline dense
matrix multiplication approach.

Key insight: Chebyshev polynomials T_n(cos(θ)) = cos(nθ), so interpolation
from Chebyshev nodes can be done via DCT (Discrete Cosine Transform).
"""

import time
from functools import partial
from typing import Tuple, List

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.fft import dct, idct

# For comparison with baseline
from jaxhps._interpolation_methods import (
    interp_from_hps_2D as baseline_interp_from_hps_2D,
    _interp_to_point_2D,
)
from jaxhps.quadrature import (
    chebyshev_points,
    affine_transform,
)
from jaxhps._discretization_tree import DiscretizationNode2D


@partial(jax.jit, static_argnums=(1,))
def chebyshev_to_uniform_1D_dct(
    f_cheby: jax.Array,
    n_uniform: int,
) -> jax.Array:
    """
    Interpolate from Chebyshev nodes to uniform grid using DCT.

    Given function values at Chebyshev nodes, compute values on uniform grid.
    Uses the fact that Chebyshev polynomials are cosines in angle space.

    Args:
        f_cheby: Function values at Chebyshev points of order p. Shape (p,)
        n_uniform: Number of uniform grid points.

    Returns:
        Function values at uniform grid points. Shape (n_uniform,)
    """
    p = f_cheby.shape[0]

    # Step 1: Compute Chebyshev coefficients via DCT-I
    # The Chebyshev nodes are x_k = cos(pi*k/(p-1)) for k=0,...,p-1
    # DCT-I transforms f_k to coefficients c_n where f(x) = sum c_n T_n(x)

    # Normalize for DCT-I (type 1)
    f_normalized = f_cheby.copy()
    f_normalized = f_normalized.at[0].multiply(0.5)
    f_normalized = f_normalized.at[-1].multiply(0.5)

    # DCT-I gives us scaled Chebyshev coefficients
    coeffs = dct(f_normalized, type=1) / (p - 1)

    # Step 2: Evaluate on uniform grid using Clenshaw algorithm or direct DCT
    # For uniform grid x_j = -1 + 2j/(n-1), we have theta_j = pi * (n-1-j)/(n-1)
    # Direct evaluation via iDCT on padded coefficients

    # Pad coefficients to desired output size
    if n_uniform > p:
        coeffs_padded = jnp.pad(coeffs, (0, n_uniform - p))
    else:
        coeffs_padded = coeffs[:n_uniform]

    # Scale for inverse DCT
    coeffs_padded = coeffs_padded.at[0].multiply(2.0)
    if n_uniform > 1:
        coeffs_padded = coeffs_padded.at[-1].multiply(2.0 if n_uniform <= p else 1.0)

    # Inverse DCT-I to get values on uniform grid in angle space
    f_uniform = dct(coeffs_padded * (n_uniform - 1), type=1) / 2

    return f_uniform


@partial(jax.jit, static_argnums=(2,))
def chebyshev_to_uniform_2D_dct(
    f_cheby_2d: jax.Array,
    bounds: jax.Array,
    n_uniform: int,
) -> jax.Array:
    """
    Interpolate 2D Chebyshev data to uniform grid using 2D DCT.

    Args:
        f_cheby_2d: Function values on Chebyshev grid. Shape (p, p)
        bounds: [xmin, xmax, ymin, ymax]
        n_uniform: Number of points in each direction for uniform grid

    Returns:
        Function values on uniform n_uniform x n_uniform grid. Shape (n_uniform, n_uniform)
    """
    p = f_cheby_2d.shape[0]

    # 2D Chebyshev coefficients via separable 1D DCTs
    # First along rows (x direction)
    f_temp = f_cheby_2d.copy()
    f_temp = f_temp.at[:, 0].multiply(0.5)
    f_temp = f_temp.at[:, -1].multiply(0.5)
    coeffs_x = dct(f_temp, type=1, axis=1) / (p - 1)

    # Then along columns (y direction)
    coeffs_x = coeffs_x.at[0, :].multiply(0.5)
    coeffs_x = coeffs_x.at[-1, :].multiply(0.5)
    coeffs_2d = dct(coeffs_x, type=1, axis=0) / (p - 1)

    # Pad coefficients
    if n_uniform > p:
        coeffs_padded = jnp.pad(coeffs_2d, ((0, n_uniform - p), (0, n_uniform - p)))
    else:
        coeffs_padded = coeffs_2d[:n_uniform, :n_uniform]

    # Inverse 2D DCT
    # Scale for inverse
    coeffs_padded = coeffs_padded.at[0, :].multiply(2.0)
    coeffs_padded = coeffs_padded.at[:, 0].multiply(2.0)

    # Row-wise inverse DCT
    f_temp = dct(coeffs_padded * (n_uniform - 1), type=1, axis=1) / 2

    # Column-wise inverse DCT
    f_uniform = dct(f_temp * (n_uniform - 1), type=1, axis=0) / 2

    return f_uniform


@partial(jax.jit, static_argnums=(0,))
def _compute_barycentric_weights(p: int) -> jax.Array:
    """
    Precompute barycentric weights for Chebyshev points.

    For Chebyshev points of the first kind:
    w_j = (-1)^j * sin((2j+1)*pi/(2p))

    For Chebyshev points of the second kind (used in jaxhps):
    w_j = (-1)^j, with w_0 = w_{p-1} = 0.5
    """
    w = jnp.array([(-1.0)**j for j in range(p)])
    w = w.at[0].multiply(0.5)
    w = w.at[-1].multiply(0.5)
    return w


@partial(jax.jit, static_argnums=(2,))
def _barycentric_interp_weights(
    x_target: jax.Array,
    nodes: jax.Array,
    p: int,
) -> jax.Array:
    """
    Compute barycentric Lagrange interpolation weights.

    Uses the numerically stable barycentric formula:
    L(x) = sum_j w_j/(x - x_j) * f_j / sum_j w_j/(x - x_j)

    This version handles the case where x_target equals a node.
    """
    w = _compute_barycentric_weights(p)

    # Compute differences
    diffs = x_target - nodes  # (p,)

    # Add small epsilon to avoid division by zero, but track where it was zero
    eps = 1e-14
    abs_diffs = jnp.abs(diffs)
    is_node = abs_diffs < eps

    # Safe division: use where to handle zeros
    safe_diffs = jnp.where(is_node, 1.0, diffs)
    weights = w / safe_diffs

    # If x_target is a node, return one-hot
    any_node = jnp.any(is_node)

    # Normalized weights for interpolation case
    norm_weights = weights / jnp.sum(weights)

    # One-hot weights for exact node case
    onehot_weights = jnp.where(is_node, 1.0, 0.0)

    # Select based on whether x_target matches a node
    return jnp.where(any_node, onehot_weights, norm_weights)


def interp_from_hps_2D_batched(
    leaves: Tuple[DiscretizationNode2D],
    p: int,
    f_evals: jax.Array,
    target_bounds: jax.Array,
    n_uniform: int,
) -> jax.Array:
    """
    Optimized interpolation using batched operations.

    Instead of per-point interpolation, we:
    1. For each leaf, identify which target points fall in it
    2. Use vectorized barycentric Lagrange interpolation

    Args:
        leaves: Tuple of leaf nodes
        p: Polynomial order
        f_evals: Solution values at Chebyshev nodes. Shape (n_leaves, p^2)
        target_bounds: [xmin, xmax, ymin, ymax] for target grid
        n_uniform: Number of uniform grid points per dimension

    Returns:
        Interpolated values on uniform grid. Shape (n_uniform, n_uniform)
    """
    n_leaves = len(leaves)

    # Create uniform target grid
    x_vals = jnp.linspace(target_bounds[0], target_bounds[1], n_uniform)
    y_vals = jnp.linspace(target_bounds[2], target_bounds[3], n_uniform)

    # Get leaf bounds as array
    corners_lst = [
        jnp.array([node.xmin, node.xmax, node.ymin, node.ymax])
        for node in leaves
    ]
    leaf_bounds = jnp.stack(corners_lst)  # (n_leaves, 4)

    # For each target point, find which leaf it belongs to
    X, Y = jnp.meshgrid(x_vals, y_vals, indexing='xy')
    pts = jnp.stack([X.flatten(), Y.flatten()], axis=-1)  # (n_pts, 2)

    # Check containment for all points and all leaves
    in_x = (pts[:, 0, None] >= leaf_bounds[None, :, 0]) & \
           (pts[:, 0, None] <= leaf_bounds[None, :, 1])
    in_y = (pts[:, 1, None] >= leaf_bounds[None, :, 2]) & \
           (pts[:, 1, None] <= leaf_bounds[None, :, 3])
    in_leaf = in_x & in_y  # (n_pts, n_leaves)

    # For each point, get the leaf index
    leaf_idx = jnp.argmax(in_leaf, axis=1)  # (n_pts,)

    # Get Chebyshev points for interpolation
    cheby_pts = chebyshev_points(p)

    # JIT-compiled inner function
    @partial(jax.jit, static_argnums=())
    def _interp_single(xy, leaf_id, leaf_bounds, f_evals, cheby_pts):
        """Interpolate to a single point given its leaf."""
        x, y = xy[0], xy[1]
        bounds = leaf_bounds[leaf_id]
        f_leaf = f_evals[leaf_id].reshape(p, p)

        # Map target point to [-1, 1] reference element
        x_ref = 2 * (x - bounds[0]) / (bounds[1] - bounds[0]) - 1
        y_ref = 2 * (y - bounds[2]) / (bounds[3] - bounds[2]) - 1

        # Get interpolation weights
        wx = _barycentric_interp_weights(x_ref, cheby_pts, p)
        wy = _barycentric_interp_weights(y_ref, cheby_pts, p)

        # f_leaf has shape (p, p)
        # Tensor product interpolation
        # First interpolate along x (axis 1): f_leaf @ wx gives (p,)
        f_y = f_leaf @ wx
        # Then interpolate along y
        result = wy @ f_y

        return result

    # Vectorize over all points using vmap
    interp_fn = partial(_interp_single, leaf_bounds=leaf_bounds,
                       f_evals=f_evals, cheby_pts=cheby_pts)

    vals = jax.vmap(lambda xy, lid: interp_fn(xy, lid))(pts, leaf_idx)

    return vals.reshape(n_uniform, n_uniform)


def barycentric_lagrange_weights(nodes: jax.Array) -> jax.Array:
    """
    Compute barycentric weights for Lagrange interpolation.

    For Chebyshev nodes, these are:
    w_j = (-1)^j * sin((2j+1)*pi/(2n)) for j=0,...,n-1
    """
    n = nodes.shape[0]

    # General formula
    w = jnp.ones(n)
    for j in range(n):
        for k in range(n):
            if k != j:
                w = w.at[j].multiply(1.0 / (nodes[j] - nodes[k]))

    return w


@partial(jax.jit, static_argnums=(0,))
def precompute_interp_matrix_cheby_to_uniform(
    p: int,
    n_uniform: int,
) -> jax.Array:
    """
    Precompute interpolation matrix from Chebyshev points to uniform grid.

    For 1D: maps p Chebyshev points to n_uniform uniform points on [-1, 1].

    Returns matrix I where f_uniform = I @ f_cheby
    """
    cheby_pts = chebyshev_points(p)
    uniform_pts = jnp.linspace(-1, 1, n_uniform)

    # Barycentric weights for Chebyshev nodes
    # For Chebyshev points, w_j = (-1)^j * (1 if j in {0, n-1} else 2)
    w = jnp.array([(-1)**j for j in range(p)], dtype=jnp.float64)
    w = w.at[0].multiply(0.5)
    w = w.at[-1].multiply(0.5)

    # Build interpolation matrix
    def interp_row(x_target):
        diffs = x_target - cheby_pts
        # Check if x_target is a node
        is_node = jnp.abs(diffs) < 1e-14
        exact_match = jnp.any(is_node)

        def exact_case(_):
            return jnp.where(is_node, 1.0, 0.0)

        def interp_case(_):
            weights = w / diffs
            return weights / jnp.sum(weights)

        return jax.lax.cond(exact_match, exact_case, interp_case, None)

    I = jax.vmap(interp_row)(uniform_pts)
    return I


def compare_interpolation_methods(
    p: int = 16,
    L: int = 2,
    n_uniform: int = 100,
    n_runs: int = 5,
) -> dict:
    """
    Compare baseline vs optimized interpolation methods.

    Returns timing comparison and accuracy metrics.
    """
    from jaxhps import DiscretizationNode2D, Domain
    from jaxhps._discretization_tree import get_all_leaves

    # Setup domain
    root = DiscretizationNode2D(xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0)
    domain = Domain(p=p, q=p-2, root=root, L=L)

    # Create test function values (smooth function)
    def test_fn(x):
        return jnp.sin(jnp.pi * x[..., 0]) * jnp.cos(jnp.pi * x[..., 1])

    f_evals = test_fn(domain.interior_points)  # (n_leaves, p^2)
    leaves = tuple(get_all_leaves(root))
    n_leaves = 4**L

    x_vals = jnp.linspace(-1.0, 1.0, n_uniform)
    y_vals = jnp.linspace(-1.0, 1.0, n_uniform)

    # Warmup
    _ = baseline_interp_from_hps_2D(leaves, p, f_evals, x_vals, y_vals)
    target_bounds = jnp.array([-1.0, 1.0, -1.0, 1.0])
    _ = interp_from_hps_2D_batched(leaves, p, f_evals, target_bounds, n_uniform)

    # Time baseline
    baseline_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result_baseline, _ = baseline_interp_from_hps_2D(
            leaves, p, f_evals, x_vals, y_vals
        )
        jax.block_until_ready(result_baseline)
        t1 = time.perf_counter()
        baseline_times.append(t1 - t0)

    # Time optimized (batched)
    batched_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result_batched = interp_from_hps_2D_batched(
            leaves, p, f_evals, target_bounds, n_uniform
        )
        jax.block_until_ready(result_batched)
        t1 = time.perf_counter()
        batched_times.append(t1 - t0)

    # Compare accuracy
    # True values on uniform grid
    X, Y = jnp.meshgrid(x_vals, y_vals)
    true_vals = test_fn(jnp.stack([X, Y], axis=-1))

    error_baseline = jnp.max(jnp.abs(result_baseline - true_vals))
    error_batched = jnp.max(jnp.abs(result_batched - true_vals))

    return {
        'p': p,
        'L': L,
        'n_uniform': n_uniform,
        'n_leaves': n_leaves,
        'baseline_time_mean': float(np.mean(baseline_times)),
        'baseline_time_std': float(np.std(baseline_times)),
        'batched_time_mean': float(np.mean(batched_times)),
        'batched_time_std': float(np.std(batched_times)),
        'speedup': float(np.mean(baseline_times) / np.mean(batched_times)),
        'error_baseline': float(error_baseline),
        'error_batched': float(error_batched),
    }


if __name__ == "__main__":
    jax.config.update("jax_default_device", jax.devices("cpu")[0])

    print("Comparing interpolation methods...")
    print("=" * 70)

    results = []
    for p in [8, 12, 16, 20]:
        for n_uniform in [50, 100, 200]:
            print(f"\nTesting p={p}, n_uniform={n_uniform}...")
            result = compare_interpolation_methods(
                p=p, L=2, n_uniform=n_uniform, n_runs=5
            )
            results.append(result)
            print(f"  Baseline: {result['baseline_time_mean']:.4f}±{result['baseline_time_std']:.4f}s")
            print(f"  Batched:  {result['batched_time_mean']:.4f}±{result['batched_time_std']:.4f}s")
            print(f"  Speedup:  {result['speedup']:.2f}x")
            print(f"  Error baseline: {result['error_baseline']:.2e}")
            print(f"  Error batched:  {result['error_batched']:.2e}")

    # Save results
    import json
    with open('results/interpolation_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("INTERPOLATION COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'p':>4} {'n_unif':>8} {'Baseline(s)':>14} {'Batched(s)':>14} {'Speedup':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['p']:>4} {r['n_uniform']:>8} "
              f"{r['baseline_time_mean']:>10.4f}±{r['baseline_time_std']:.2f} "
              f"{r['batched_time_mean']:>10.4f}±{r['batched_time_std']:.2f} "
              f"{r['speedup']:>8.2f}x")
