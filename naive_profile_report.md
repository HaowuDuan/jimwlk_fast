# Naive CuPy JIMWLK Profile Report

GPU: NVIDIA GeForce RTX 3090 (GA102, SM 8.6)
Lattice: N=1024, L=64, Ny=50, dY=0.01

## Table 1: Wall-clock time per section

All times in milliseconds. Per-call is median over repeated calls (excludes first-call JIT overhead).

### Initial Condition (50 layers)

| Section | Per-call (ms) | Calls | Total (ms) | % of IC |
|---------|-------------:|------:|-----------:|--------:|
| Noise gen (8 fields) | 0.32 | 50 | 16.6 | 0.2% |
| FFT + propagate (8 fields) | 1.28 | 50 | 64.3 | 0.7% |
| Algebra assembly (8 einsum) | 3.10 | 50 | 155.4 | 1.7% |
| **matexp (Cayley-Hamilton)** | **172.23** | **50** | **8649.9** | **92.1%** |
| Path ordering (matmul) | 10.33 | 50 | 506.2 | 5.4% |
| **IC TOTAL** | | | **9395.7** | |

### Evolution (per step, median of 5 steps)

| Section | Per-call (ms) | % of step |
|---------|-------------:|----------:|
| Noise gen (einsum) | 1.52 | 0.1% |
| Left FFT (fft2) | 2.30 | 0.2% |
| Left conv + IFFT | 3.26 | 0.2% |
| **Left matexp** | **287.00** | **19.6%** |
| **V†ξV (2× matmul)** | **571.73** | **39.0%** |
| Right FFT (fft2) | 9.01 | 0.6% |
| Right conv + IFFT | 7.77 | 0.5% |
| **Right matexp** | **286.71** | **19.6%** |
| **V update (2× matmul)** | **278.91** | **19.0%** |
| Reunitarize | 17.70 | 1.2% |
| **Step TOTAL** | **1466.1** | |

### Observables (per call, median of 3)

| Section | Per-call (ms) |
|---------|-------------:|
| Dipole (Wiener-Khinchin) | 24.30 |
| xG / xh (gauge field TMDs) | 321.85 |

## Table 2: CUDA kernel launch breakdown

Captured via `ncu --set basic` over IC (50 layers) + 1 evolution step.

**Total kernel launches: 16,487** across 84 unique kernel types.

| Category | Launches | Examples |
|----------|--------:|---------|
| CuPy elementwise ops | 10,799 | `cupy_multiply`, `cupy_add`, `cupy_subtract`, `cupy_exp`, `cupy_cos`, `cupy_where`, `cupy_copy` |
| cuBLAS batched GEMM | 3,571 | `ampere_cgemm_32x64_nn` (complex64), `cutlass_z884gemm` (complex128) |
| cuFFT | 1,610 | `regular_fft`, `vector_fft` |
| RNG | 401 | `execute_dist<standard_normal_float_functor>` |
| Reductions | 106 | `cupy_scan_naive`, `cupy_bsum_shfl` |

### Top kernel types by launch count

| Kernel | Launches | What generates it |
|--------|--------:|----|
| `ampere_cgemm_32x64_nn` | 2,448 | `cp.matmul` on complex64 3×3 — path ordering, V†ξV, V update |
| `cupy_multiply__complex128` | 1,100 | matexp eigenvalue/coefficient arithmetic in float64 |
| `cutlass_z884gemm` | 969 | `cp.matmul` on complex128 3×3 — matexp M² in double precision |
| `regular_fft` / `vector_fft` | 805 each | cuFFT plans for fft2/ifft2 |
| `cupy_scan_naive` | 780 | internal reductions (argmax, sum, etc.) |
| `cupy_subtract__complex128` | 628 | matexp Lagrange interpolation |
| `cupy_add__complex128` | 522 | matexp coefficient accumulation |
| `cupy_add__complex64` | 502 | algebra assembly (A += field * t^a) |
| `cupy_multiply__float64` | 411 | matexp eigenvalue scaling |
| `cupy_multiply__float32_complex64` | 400 | IC propagation (field × propagator) |

## Table 3: Kernel-level roofline data (1 evolution step)

Profiled via `ncu --set full` on 1 evolution step (499 kernel launches, 34 replay passes each).

### Top kernels by total GPU time

| Kernel | Count | Total (ms) | Avg (ms) | SM % | DRAM % | Occ % | Regs | Bound |
|--------|------:|-----------:|---------:|-----:|-------:|------:|-----:|-------|
| `cutlass::Kernel` (complex128 3×3 GEMM) | 102 | 1176.4 | 11.53 | 87.9 | 0.2 | 8.0 | 110 | Compute |
| `ampere_cgemm_32x64_nn` (complex64 3×3 GEMM) | 64 | 59.9 | 0.94 | 70.4 | 1.6 | 40.3 | 94 | Compute |
| `cub::DeviceSegmentedReduceKernel` | 1 | 16.0 | 15.97 | 86.3 | 0.5 | 77.3 | 40 | Compute |
| `cupy_copy__complex64` | 8 | 5.1 | 0.64 | 14.1 | 60.0 | 78.4 | 23 | Memory |
| `regular_fft` (cuFFT) | 4 | 4.9 | 1.22 | 3.2 | 54.1 | 31.2 | 56 | Memory |
| `_norm_ord2_complex` (reunitarize) | 2 | 4.4 | 2.19 | 83.9 | 29.9 | 54.2 | 38 | Compute |
| `cupy_multiply__complex128` | 50 | 4.0 | 0.08 | 26.4 | 91.1 | 82.9 | 28 | Memory |
| `cupy_copy__complex128` | 11 | 2.5 | 0.22 | 6.3 | 85.7 | 82.0 | 20 | Memory |
| `cupy_add__complex128` | 22 | 2.4 | 0.11 | 14.8 | 90.5 | 83.6 | 26 | Memory |
| `cupy_subtract__complex128` | 28 | 1.8 | 0.07 | 15.7 | 89.7 | 84.4 | 26 | Memory |
| `vector_fft` (cuFFT) | 4 | 1.4 | 0.36 | 29.8 | 91.7 | 56.1 | 70 | Memory |
| `cupy_exp__complex128` | 6 | 1.1 | 0.18 | 85.7 | 19.7 | 92.5 | 34 | Compute |
| **TOTAL (499 kernels)** | | **1293.7** | | | | | | |

### Key observations

**1. GEMM dominates: 95.5% of GPU time**

- `cutlass::Kernel` (complex128): 102 calls, 1176 ms — matexp's M² computation in double precision. 8% occupancy, 110 registers. Catastrophically inefficient for 3×3 matrices.
- `ampere_cgemm_32x64_nn` (complex64): 64 calls, 60 ms — V†ξV, V·exp_R, exp_L·V. 40% occupancy.
- Combined: 1236 ms out of 1294 ms total = **95.5%** of all GPU time.
- These are cuBLAS batched GEMM kernels designed for large matrices. For 3×3 matrices, each warp processes a single tiny matmul — the 32×64 or 32×16 tile is massive overkill. The hardware is doing 3×3 work with 2048-element tiles.

**2. complex128 overhead**

- matexp computes eigenvalues in float64 for numerical stability, but this promotes all intermediates to complex128 (16 bytes/element vs 8 for complex64).
- The cutlass complex128 GEMM uses 110 registers → 8% occupancy (vs 94 registers / 40% for complex64).
- 50 elementwise `cupy_multiply__complex128` ops + 28 subtracts + 22 adds = 100 extra kernel launches just for the double-precision arithmetic.

**3. Elementwise kernels: individually fast, collectively expensive**

- Each elementwise op (multiply, add, exp, cos) takes 0.05–0.18 ms — the kernel itself runs at 80-92% DRAM bandwidth.
- But there are ~200 of them per step, each reading/writing the full N×N×3×3 array. Total memory traffic: ~200 × 2 × 1024² × 9 × 16 bytes ≈ 60 GB of redundant data movement.
- A fused RawKernel eliminates all intermediate reads/writes.
