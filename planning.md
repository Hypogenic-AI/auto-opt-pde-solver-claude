# Research Plan: Automatic Optimization of a PDE Solver

## Research Question

Can automatic AI scientist methods improve the performance and efficiency of the jaxhps PDE solver by:
1. Enhancing parallelization for adaptive grids during merge operations
2. Enabling kernel fusion between local solve and merge stages
3. Optimizing interpolation from spectral (Chebyshev) to uniform grids using FFT-based methods

## Background and Motivation

### Problem Context
The Hierarchical Poincare-Steklov (HPS) method is a fast direct solver for linear elliptic PDEs that achieves O(N^1.5) complexity for pre-computation and O(N log N) for the solve phase. The jaxhps library provides a JAX-based GPU-accelerated implementation with automatic differentiation support.

### Identified Optimization Opportunities
From analyzing the jaxhps codebase and literature:

1. **Adaptive Grid Parallelization**: The current adaptive discretization (`_adaptive_discretization_2D.py`) processes nodes sequentially in a while loop during level restriction checks. The merge step in the down pass also processes nodes level-by-level but could benefit from better batching.

2. **Kernel Fusion**: The solve step involves separate operations:
   - Down pass: `S_arr @ bdry_data + g_tilde` followed by interface decompression
   - Leaf solve: `Y @ g + v`
   These could potentially be fused to reduce memory transfers.

3. **Interpolation**: The current interpolation (`_interpolation_methods.py`) uses barycentric Lagrange interpolation matrices via dense matrix-vector products (`I @ f`). For uniform target grids, FFT-based methods could be more efficient.

### Why This Matters
- HPS solvers are crucial for high-accuracy PDE solutions
- GPU acceleration is increasingly important for large-scale scientific computing
- Automatic optimization could reduce the need for manual tuning
- Findings could generalize to other spectral solvers

## Hypothesis Decomposition

### H1: Parallelization Optimization
**Sub-hypothesis**: Using JAX's vectorization (vmap) and parallel primitives (pmap) on the adaptive merge operations will improve throughput.

**Measurable criteria**:
- Speedup in build time for adaptive grids
- Improved GPU utilization during merge phase

### H2: Kernel Fusion
**Sub-hypothesis**: Combining the interface computation (`S @ g + g_tilde`) with the decompression step into a single fused kernel will reduce memory bandwidth and improve performance.

**Measurable criteria**:
- Reduced wall-clock time for down pass
- Lower peak memory usage
- Fewer memory transfers

### H3: FFT-based Interpolation
**Sub-hypothesis**: For interpolation from Chebyshev grids to uniform grids, using discrete cosine transforms (DCT) will be faster than dense matrix multiplication for sufficiently large grids.

**Measurable criteria**:
- Speedup in interpolation time
- Crossover point where FFT beats dense multiplication
- Maintained numerical accuracy (< 1e-10 relative error)

## Proposed Methodology

### Approach
We will implement and benchmark three optimization strategies, comparing against the baseline jaxhps implementation:

1. **Baseline Profiling**: Profile the existing jaxhps code to identify actual bottlenecks
2. **Systematic Optimization**: Implement each optimization independently
3. **Combined Testing**: Test optimizations in combination
4. **Statistical Analysis**: Multiple runs with different problem sizes and configurations

### Experimental Steps

#### Step 1: Environment Setup and Baseline Profiling
- Install jaxhps and dependencies
- Run existing benchmarks to establish baseline performance
- Profile using JAX's profiling tools to identify bottlenecks
- Measure: build time, solve time, memory usage across problem sizes

**Rationale**: Before optimizing, we need to understand where time is actually spent.

#### Step 2: Implement Parallelized Merge for Adaptive Grids
- Analyze the current merge implementation in `down_pass/_adaptive_2D_DtN.py`
- Implement batched version using `jax.vmap` over nodes at each level
- Handle variable-size boundary data using padding or scan operations
- Test correctness against baseline

**Rationale**: The merge step processes nodes sequentially; batching could improve GPU utilization.

#### Step 3: Implement Kernel Fusion
- Identify fusion opportunities in the down pass
- Use `jax.lax.scan` or custom JIT compilation to fuse operations
- Create unified kernel for `S @ g + g_tilde` and decompression
- Verify numerical equivalence

**Rationale**: Reducing memory round-trips is a key GPU optimization strategy.

#### Step 4: Implement FFT-based Interpolation
- Implement DCT-based interpolation from Chebyshev to uniform grids
- Use `jax.scipy.fftpack.dct` for efficient transforms
- Handle boundary conditions and grid alignment
- Test accuracy against barycentric interpolation

**Rationale**: FFT is O(n log n) vs O(n²) for dense matrix multiply.

#### Step 5: Comprehensive Benchmarking
- Test each optimization independently and combined
- Vary: problem size (N = 1K to 100K DOFs), polynomial order (p = 8 to 20), tree depth
- Run multiple trials (5+ runs) for statistical significance
- Profile kernel-level performance

#### Step 6: Analysis and Documentation
- Statistical comparison of baseline vs optimized versions
- Effect size calculation
- Document failure modes and limitations

### Baselines

1. **jaxhps baseline**: Original implementation, CPU
2. **jaxhps GPU**: Original implementation on GPU
3. **FFT spectral solver**: For comparison on uniform grids (using `jax.numpy.fft`)
4. **NumPy reference**: Dense operations without JIT (sanity check)

### Evaluation Metrics

#### Performance Metrics
- **Wall-clock time**: Build phase and solve phase separately
- **Memory usage**: Peak GPU/CPU memory
- **GPU utilization**: Percentage of compute capacity used
- **FLOPS efficiency**: Achieved vs theoretical peak

#### Accuracy Metrics
- **Relative L2 error**: ||u_computed - u_exact||_2 / ||u_exact||_2
- **Maximum pointwise error**: max|u_computed - u_exact|
- **Residual norm**: ||Lu - f|| (verifies PDE is satisfied)

#### Why These Metrics
- Wall-clock time is the primary user-facing metric
- Memory usage limits problem sizes that can be solved
- Accuracy ensures optimizations don't degrade solution quality
- GPU utilization reveals if we're compute or memory bound

### Statistical Analysis Plan

- **Multiple runs**: 5 runs per configuration, report mean ± std
- **Statistical tests**: Paired t-tests for comparing baseline vs optimized
- **Significance level**: α = 0.05
- **Effect size**: Cohen's d for practical significance
- **Confidence intervals**: 95% CIs on speedup ratios

## Expected Outcomes

### If Hypothesis Supported
- H1 (Parallelization): 2-5x speedup on adaptive grid construction
- H2 (Kernel Fusion): 1.3-2x speedup on down pass, 20-50% memory reduction
- H3 (FFT Interpolation): 2-10x speedup for interpolation on large grids (p > 16)

### If Hypothesis Refuted
- H1: Overhead of batching/padding outweighs benefits for small adaptive grids
- H2: JIT compiler already performs sufficient fusion
- H3: Dense multiplication is fast enough for typical polynomial orders (p ≤ 20)

### Either Way
- We'll have quantitative data on where jaxhps spends time
- Insights into optimization-accuracy tradeoffs
- Guidance for future optimization efforts

## Timeline and Milestones

1. **Environment Setup** (~15 min): Install dependencies, verify jaxhps works
2. **Baseline Profiling** (~30 min): Run benchmarks, analyze profiles
3. **Optimization Implementation** (~90 min):
   - Parallelization: 30 min
   - Kernel fusion: 30 min
   - FFT interpolation: 30 min
4. **Benchmarking** (~45 min): Run all experiments
5. **Analysis** (~30 min): Statistical analysis, visualization
6. **Documentation** (~30 min): Write REPORT.md

Total estimated time: ~4 hours

## Potential Challenges

### Technical Challenges
1. **JAX tracing**: Dynamic shapes in adaptive grids may prevent JIT compilation
   - Mitigation: Use padding or static maximum sizes

2. **Memory constraints**: Large 3D problems may exceed GPU memory
   - Mitigation: Focus on 2D, document scaling limits

3. **Numerical stability**: FFT methods may have different numerical properties
   - Mitigation: Compare errors carefully, ensure < 1e-10 relative error

4. **Compilation overhead**: JIT compilation could dominate for small problems
   - Mitigation: Report compilation time separately, focus on warm runs

### Methodological Challenges
1. **Variability**: GPU timing can be noisy
   - Mitigation: Multiple runs, warm-up iterations

2. **Hardware dependence**: Results may vary by GPU/CPU
   - Mitigation: Document hardware, focus on relative comparisons

## Success Criteria

The research will be considered successful if:

1. **Quantitative**: We achieve statistically significant (p < 0.05) speedup on at least one optimization with effect size d > 0.5

2. **Scientific**: We provide clear, reproducible evidence for or against each hypothesis with proper statistical analysis

3. **Practical**: We document concrete guidance on when each optimization is beneficial

4. **Reproducible**: All experiments can be reproduced with provided code and documentation

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| JAX compilation issues | Medium | High | Use simpler test cases, debug incrementally |
| No significant speedup | Medium | Medium | Document findings as negative result |
| GPU memory limits | Low | Medium | Focus on 2D problems |
| Time overrun | Medium | Medium | Prioritize most promising optimizations |
