"""
Statistical analysis of experimental results.

This script performs statistical analysis and generates visualizations.
"""

import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend

# Load results
with open('results/all_experiments.json', 'r') as f:
    results = json.load(f)


def compute_statistics():
    """Compute statistical summary of results."""
    print("="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)

    # 1. Interpolation optimization statistics
    print("\n1. INTERPOLATION OPTIMIZATION")
    print("-"*70)

    interp_results = results['experiment_2_interpolation']
    speedups = [r['speedup'] for r in interp_results]

    print(f"Mean speedup: {np.mean(speedups):.2f}x")
    print(f"Std speedup: {np.std(speedups):.2f}")
    print(f"Min speedup: {np.min(speedups):.2f}x")
    print(f"Max speedup: {np.max(speedups):.2f}x")
    print(f"Median speedup: {np.median(speedups):.2f}x")

    # Effect size (Cohen's d)
    baseline_times = [r['baseline_time_mean'] for r in interp_results]
    opt_times = [r['optimized_time_mean'] for r in interp_results]
    pooled_std = np.sqrt((np.std(baseline_times)**2 + np.std(opt_times)**2) / 2)
    cohens_d_interp = (np.mean(baseline_times) - np.mean(opt_times)) / pooled_std if pooled_std > 0 else 0
    print(f"Cohen's d: {cohens_d_interp:.2f}")

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(baseline_times, opt_times)
    print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.2e}")

    # 2. Kernel fusion statistics
    print("\n2. KERNEL FUSION OPTIMIZATION")
    print("-"*70)

    fusion_results = results['experiment_3_kernel_fusion']
    speedups_fusion = [r['speedup'] for r in fusion_results]

    print(f"Mean speedup: {np.mean(speedups_fusion):.2f}x")
    print(f"Std speedup: {np.std(speedups_fusion):.2f}")
    print(f"Min speedup: {np.min(speedups_fusion):.2f}x")
    print(f"Max speedup: {np.max(speedups_fusion):.2f}x")
    print(f"Median speedup: {np.median(speedups_fusion):.2f}x")

    baseline_times_fusion = [r['baseline_time_mean'] for r in fusion_results]
    fused_times = [r['fused_time_mean'] for r in fusion_results]
    pooled_std_fusion = np.sqrt((np.std(baseline_times_fusion)**2 + np.std(fused_times)**2) / 2)
    cohens_d_fusion = (np.mean(baseline_times_fusion) - np.mean(fused_times)) / pooled_std_fusion if pooled_std_fusion > 0 else 0
    print(f"Cohen's d: {cohens_d_fusion:.2f}")

    t_stat_fusion, p_value_fusion = stats.ttest_rel(baseline_times_fusion, fused_times)
    print(f"Paired t-test: t={t_stat_fusion:.4f}, p={p_value_fusion:.2e}")

    # 3. Combined optimization statistics
    print("\n3. COMBINED OPTIMIZATION")
    print("-"*70)

    combined_results = results['experiment_4_combined']
    speedups_combined = [r['speedup'] for r in combined_results]

    print(f"Mean speedup: {np.mean(speedups_combined):.2f}x")
    print(f"Std speedup: {np.std(speedups_combined):.2f}")
    print(f"Min speedup: {np.min(speedups_combined):.2f}x")
    print(f"Max speedup: {np.max(speedups_combined):.2f}x")

    return {
        'interpolation': {
            'mean_speedup': float(np.mean(speedups)),
            'std_speedup': float(np.std(speedups)),
            'cohens_d': float(cohens_d_interp),
            'p_value': float(p_value),
        },
        'kernel_fusion': {
            'mean_speedup': float(np.mean(speedups_fusion)),
            'std_speedup': float(np.std(speedups_fusion)),
            'cohens_d': float(cohens_d_fusion),
            'p_value': float(p_value_fusion),
        },
        'combined': {
            'mean_speedup': float(np.mean(speedups_combined)),
            'std_speedup': float(np.std(speedups_combined)),
        }
    }


def create_visualizations():
    """Generate visualization plots."""
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    # 1. Interpolation speedup by parameters
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    interp_results = results['experiment_2_interpolation']

    # Group by p
    p_vals = sorted(set(r['p'] for r in interp_results))
    n_vals = sorted(set(r['n_uniform'] for r in interp_results))

    speedup_by_p = {p: [] for p in p_vals}
    speedup_by_n = {n: [] for n in n_vals}

    for r in interp_results:
        speedup_by_p[r['p']].append(r['speedup'])
        speedup_by_n[r['n_uniform']].append(r['speedup'])

    # Plot by p
    ax1 = axes[0]
    x = np.arange(len(p_vals))
    means = [np.mean(speedup_by_p[p]) for p in p_vals]
    stds = [np.std(speedup_by_p[p]) for p in p_vals]
    ax1.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'p={p}' for p in p_vals])
    ax1.set_ylabel('Speedup')
    ax1.set_title('Interpolation Speedup by Polynomial Order')
    ax1.axhline(y=1, color='r', linestyle='--', label='Baseline')
    ax1.legend()

    # Plot by n
    ax2 = axes[1]
    x = np.arange(len(n_vals))
    means = [np.mean(speedup_by_n[n]) for n in n_vals]
    stds = [np.std(speedup_by_n[n]) for n in n_vals]
    ax2.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='darkorange')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'n={n}' for n in n_vals])
    ax2.set_ylabel('Speedup')
    ax2.set_title('Interpolation Speedup by Target Grid Size')
    ax2.axhline(y=1, color='r', linestyle='--', label='Baseline')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('figures/interpolation_speedup.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved figures/interpolation_speedup.png")

    # 2. Kernel fusion speedup by problem size
    fig, ax = plt.subplots(figsize=(10, 6))

    fusion_results = results['experiment_3_kernel_fusion']

    # Create labels
    labels = [f"p={r['p']}, L={r['L']}\n({r['n_leaves']} leaves)" for r in fusion_results]
    speedups = [r['speedup'] for r in fusion_results]

    x = np.arange(len(labels))
    colors = ['steelblue' if s > 5 else 'coral' for s in speedups]

    ax.bar(x, speedups, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Speedup')
    ax.set_title('Kernel Fusion Speedup for Solve Phase')
    ax.axhline(y=1, color='r', linestyle='--', label='Baseline')

    # Add value labels
    for i, v in enumerate(speedups):
        ax.text(i, v + 0.5, f'{v:.1f}x', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('figures/kernel_fusion_speedup.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved figures/kernel_fusion_speedup.png")

    # 3. Combined optimization comparison
    fig, ax = plt.subplots(figsize=(8, 6))

    combined_results = results['experiment_4_combined']
    labels = [f"p={r['p']}, L={r['L']}" for r in combined_results]

    x = np.arange(len(labels))
    width = 0.35

    baseline_times = [r['baseline_time_mean'] * 1000 for r in combined_results]  # Convert to ms
    opt_times = [r['optimized_time_mean'] * 1000 for r in combined_results]

    ax.bar(x - width/2, baseline_times, width, label='Baseline', color='coral', alpha=0.7)
    ax.bar(x + width/2, opt_times, width, label='Optimized', color='steelblue', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Time (ms)')
    ax.set_title('Combined Optimization: Solve + Interpolation')
    ax.legend()

    # Add speedup annotations
    for i, r in enumerate(combined_results):
        ax.annotate(f'{r["speedup"]:.1f}x',
                   xy=(i, max(baseline_times[i], opt_times[i]) + 2),
                   ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('figures/combined_optimization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved figures/combined_optimization.png")

    # 4. Build vs Solve time breakdown
    fig, ax = plt.subplots(figsize=(10, 6))

    baseline_results = results['experiment_1_baseline']
    labels = [f"p={r['p']}, L={r['L']}" for r in baseline_results]

    build_times = [r['build_time'] for r in baseline_results]
    solve_times = [r['solve_time_mean'] for r in baseline_results]

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width/2, build_times, width, label='Build', color='steelblue', alpha=0.7)
    ax.bar(x + width/2, [t * 100 for t in solve_times], width, label='Solve (x100)', color='darkorange', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Time (s)')
    ax.set_title('Build vs Solve Time (Solve scaled x100 for visibility)')
    ax.legend()

    plt.tight_layout()
    plt.savefig('figures/build_vs_solve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved figures/build_vs_solve.png")


def main():
    """Main analysis function."""
    stats_results = compute_statistics()
    create_visualizations()

    # Save statistical summary
    with open('results/statistical_analysis.json', 'w') as f:
        json.dump(stats_results, f, indent=2)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("Generated files:")
    print("  - results/statistical_analysis.json")
    print("  - figures/interpolation_speedup.png")
    print("  - figures/kernel_fusion_speedup.png")
    print("  - figures/combined_optimization.png")
    print("  - figures/build_vs_solve.png")


if __name__ == "__main__":
    main()
