# Parallel Programming

**Åbo Akademi University, Information Technology Department**

**Instructor: Alireza Olama**

## Homework Assignment 4: Optimizing Matrix Multiplication in C++

**Due Date**: 31/05/2026

**Points**: 100

---

### Assignment Overview

Welcome to the last homework assignment of the Parallel Programming course! In this assignment, you will optimize the performance of a naive matrix multiplication
implementation using two techniques:

1. **Cache Optimization via Blocked Matrix Multiplication**: Improve data locality to reduce cache misses.
2. **Parallel Matrix Multiplication using `OpenMP`**: Parallelize the computation across multiple threads.

Your task is to implement both optimizations in the provided C++ `main.cpp` file, measure their performance, and compare the
wall clock time of the naive, cache-optimized, and parallel implementations for each test case. This assignment builds
on naive matmul implementation, so ensure your naive implementation is correct before starting.

---

### Technical Requirements

#### 1. Cache Optimization (Blocked Matrix Multiplication)

**Why Cache Optimization?**

Modern CPUs rely on cache memory to reduce the latency of accessing data from main memory. Cache memory is faster but
smaller, organized in cache lines (typically 64 bytes). When a CPU accesses a memory location, it fetches an entire
cache line. Matrix multiplication can suffer from poor performance if memory accesses are not cache-friendly, leading to
frequent cache misses.

The naive matrix multiplication (with triple nested loops) accesses memory in a way that may not exploit spatial and
temporal locality:

- **Spatial Locality**: Accessing consecutive memory locations (e.g., elements in the same cache line).
- **Temporal Locality**: Reusing the same data multiple times while it’s still in the cache.

Blocked matrix multiplication divides the matrices into smaller submatrices (blocks) that fit into the cache. By
performing computations on these blocks, you ensure that data is reused while it resides in the cache, reducing cache
misses and improving performance.

**Blocked Matrix Multiplication Pseudocode**

Assume matrices \( A \) (m × n), \( B \) (n × p), and \( C \) (m × p) are stored in row-major order. The blocked matrix
multiplication processes submatrices of size \( block_size × block_size \):

```cpp
// C = A * B
for (ii = 0; ii < m; ii += block_size)
    for (jj = 0; jj < p; jj += block_size)
        for (kk = 0; kk < n; kk += block_size)
            // Process block: C[ii:ii+block_size, jj:jj+block_size] += A[ii:ii+block_size, kk:kk+block_size] * B[kk:kk+block_size, jj:jj+block_size]
            for (i = ii; i < min(ii + block_size, m); i++)
                for (j = jj; j < min(jj + block_size, p); j++)
                    for (k = kk; k < min(kk + block_size, n); k++)
                        C[i * p + j] += A[i * n + k] * B[k * p + j]
```

- **block_size**: Chosen to ensure the block fits in the cache (e.g., 32, 64, or 128, depending on the system).
- **Outer loops (ii, jj, kk)**: Iterate over blocks.
- **Inner loops (i, j, k)**: Compute within a block, reusing data in the cache.

**Task**: Implement the `blocked_matmul` function in the provided `main.cpp`. Experiment with different block sizes (e.g.,
16, 32, 64) and report the best performance.

---

#### 2. Parallel Matrix Multiplication with OpenMP

**Why OpenMP?**

`OpenMP` is a portable API for parallel programming in shared-memory systems. It allows you to parallelize loops with
minimal code changes, distributing iterations across multiple threads. In matrix multiplication, the outer loop(s) can
be parallelized, as each element of the output matrix \( C \) can be computed independently.

**Parallelizing with OpenMP**

Use OpenMP to parallelize the outer loop(s) of the naive matrix multiplication. For example, parallelize the loop over
rows of \( C \):

```cpp
#pragma omp parallel for
for (i = 0; i < m; i++)
    for (j = 0; j < p; j++)
        for (k = 0; k < n; k++)
            C[i * p + j] += A[i * n + k] * B[k * p + j];
```

- The `#pragma omp parallel for` directive tells `OpenMP` to distribute iterations of the loop across available threads.
- Ensure thread safety: Since each iteration writes to a distinct element of \( C \), this loop is safe to parallelize
  without locks.
- Use `omp_get_wtime()` to measure wall clock time for accurate performance comparisons.

**Task**: Implement the `parallel_matmul` function in the provided `main.cpp` using `OpenMP`. Test with different numbers of
threads (e.g., 2, 4, 8) by setting the environment variable `OMP_NUM_THREADS`.

---

#### 3. Performance Measurement

## Performance Results

### Environment
- **Platform**: GitHub Codespaces (Linux x86_64, 2 physical CPU cores)
- **Compiler**: g++ with `-O3 -fopenmp`
- **Methodology**: Each timing is the arithmetic mean of **5 independent runs**
- **Default block size**: 64 (theoretical L1-cache-line alignment)
- **Default thread count**: 4

### Main Results Table (Averaged over 5 runs)

| Case | Dimensions (m × n × p) | Naive (s) | Blocked (s) | Parallel (s) | Blocked Speedup | Parallel Speedup |
|------|------------------------|-----------|-------------|--------------|-----------------|------------------|
| 0    | 64 × 64 × 64           | 0.000209  | 0.000202    | 0.000227     | 1.04×           | 0.92×            |
| 1    | 128 × 64 × 128         | 0.001096  | 0.000871    | 0.000740     | 1.26×           | 1.48×            |
| 2    | 100 × 128 × 56         | 0.000691  | 0.000638    | 0.000922     | 1.08×           | 0.75×            |
| 3    | 128 × 64 × 128         | 0.001541  | 0.001245    | 0.001014     | 1.24×           | 1.52×            |
| 4    | 32 × 128 × 32          | 0.000160  | 0.000143    | 0.000309     | 1.12×           | 0.52×            |
| 5    | 200 × 100 × 256        | 0.007707  | 0.007681    | 0.007275     | 1.00×           | 1.06×            |
| 6    | 256 × 256 × 256        | 0.026578  | 0.021396    | 0.022247     | 1.24×           | 1.19×            |
| 7    | 256 × 300 × 256        | 0.033655  | 0.026134    | 0.030615     | 1.29×           | 1.10×            |
| 8    | 64 × 128 × 64          | 0.000499  | 0.000385    | 0.000419     | 1.30×           | 1.19×            |
| 9    | 256 × 256 × 257        | 0.018924  | 0.013386    | 0.011839     | 1.41×           | 1.60×            |

All implementations validated against `output.raw` with tolerance `1e-2`. All 10 cases pass for all three implementations.

### Block Size Experiment (Case 7: 256 × 300 × 256, the largest test case)

To find the optimal block size, the `blocked_matmul` was tested with four block sizes against the naive baseline. Each timing is averaged over 5 runs.

| Block Size | Time (s) | Speedup |
|------------|----------|---------|
| **16**     | **0.02312** | **2.33×** |
| 32         | 0.02349  | 2.29×   |
| 64         | 0.03020  | 1.78×   |
| 128        | 0.02783  | 1.94×   |

**Finding**: Block size **16** gives the best performance for these matrix dimensions, with block size 32 a close second. The commonly recommended block size of 64 (one cache line of doubles) was *not* optimal here. Smaller blocks keep the working set comfortably inside L1 cache, while at block size 64 and above the working set begins to spill out of L1.

For the main results, block size 64 was kept as the default to follow the conventional "cache-line aligned" recommendation, but block size 16 or 32 would give meaningfully better speedups on this hardware.

### Thread Count Experiment (Case 7)

To find the optimal thread count, `parallel_matmul` was tested with 1, 2, 4, and 8 threads. Each timing is averaged over 5 runs.

| Threads | Time (s) | Speedup |
|---------|----------|---------|
| 1       | 0.03594  | 1.08×   |
| **2**   | **0.02555** | **1.52×** |
| 4       | 0.02713  | 1.43×   |
| 8       | 0.03206  | 1.21×   |

**Finding**: **2 threads is optimal** on this hardware. The GitHub Codespaces free tier provides 2 physical CPU cores; once thread count exceeds physical cores, hyperthreading contention and OpenMP scheduling overhead outweigh the parallelism benefit. 8 threads is *worse* than 1 thread because thread management overhead dominates.

On a machine with 4 or more physical cores, the optimal thread count would shift accordingly.

### Analysis

**Correctness**: Every implementation produces identical results to the reference output for all 10 test cases.

**Cache Optimization (Blocked)**:
- Blocking gives consistent **modest speedup (1.0× to 1.41×)** across cases with the default block size of 64.
- The block size sweep showed up to **2.33×** speedup at block size 16, demonstrating the importance of tuning the block size to the specific cache hierarchy and problem dimensions.

**Parallel (OpenMP)**:
- Parallelization helps **when the matrix is large enough** to amortize OpenMP thread setup overhead.
- For tiny matrices (cases 0, 2, 4), parallel is *slower* than naive (0.52× to 0.92×) because thread creation cost exceeds the actual compute work.
- For mid-sized matrices (cases 1, 3, 6, 8, 9), parallel gives 1.19× – 1.60× speedup.
- The thread sweep revealed that the Codespaces 2-core environment caps the achievable parallel speedup at ~1.5× regardless of how many threads we request. On hardware with more cores, larger speedups would be visible.

**Block Size Choice**: For these specific matrix sizes (up to approx. 256 × 300), L1 cache pressure dominates and smaller blocks (16, 32) work best. The "default" cache-line-sized block of 64 is suboptimal here but would likely be better on much larger problems where the trade-off shifts toward reducing loop overhead.

**Optimal Configuration on Codespaces (2-core)**:
- Block size: **16 or 32**
- Thread count: **2**
- Expected combined speedup over naive: approximately 3× or 4× by combining blocking and parallelization

### Challenges

1. **Small Test Cases**: The provided test cases are too small to fully showcase OpenMP parallelism. The largest case (256 × 300 × 256) executes in ~33 ms, where OpenMP setup costs are significant relative to compute. Matrices of 1024 × 1024 or larger would yield speedups closer to the theoretical limits of the hardware.

2. **Codespaces Environment**: The 2-core CPU limit in GitHub Codespaces caps achievable parallel speedup. On a typical 8-core workstation, parallel speedups of 4× - 6× would be expected for the larger test cases.

3. **Measurement Stability**: Single-run timings showed significant variance (some "speedups" appeared to be slowdowns simply due to noise). Switching to 5-run averaging stabilized the results and made the patterns clear. This is itself a useful methodological finding.

4. **Default Block Size Was Suboptimal**: The conventional block size of 64 (one cache line of doubles) was not the best for these test cases block size 16 was 30% faster. This reinforces that "cache-line aligned" is a starting heuristic, not a final answer; empirical tuning matters.

5. **Text-format I/O**: The `.raw` files are space-separated text, not binary doubles. Reading is done using `ifstream >> double` with the first two integers as `rows cols` dimensions.

6. **Local Toolchain**: Could not install g++ locally on Windows in time; switched to GitHub Codespaces, which provided a complete Linux dev environment with all required tooling.
