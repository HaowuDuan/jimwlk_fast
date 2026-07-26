# Nsight Compute Profile Report — jimwlk_cuda N=1024

GPU: NVIDIA GeForce RTX 3090 (GA102, SM 8.6)
Profile script: `profile_kernels.py` (one full evolution step + IC)
Report: `profile_kernels_optimized.ncu-rep`

## Custom CUDA Kernels

| ID | Kernel | Duration (μs) | SM % | Mem % | Occupancy % | Registers | Grid | Block |
|---:|--------|-------------:|-----:|------:|------------:|----------:|-----:|------:|
| 22 | assemble_algebra (IC) | 13630.0 | 5.67 | 80.95 | 91.41 | 26 | 204,800 | 256 |
| 24 | matexp_su3 (IC, unscaled) | 98030.0 | 86.88 | 8.57 | 28.18 | 90 | 204,800 | 256 |
| 25 | path_order (IC) | 4530.0 | 14.12 | 95.22 | 63.49 | 55 | 4,096 | 256 |
| 56 | assemble_su3_noise | 551.52 | 5.58 | 79.65 | 89.92 | 26 | 8,192 | 256 |
| 68 | assemble_algebra (evolution) | 282.21 | 5.53 | 79.03 | 88.56 | 26 | 4,096 | 256 |
| 69 | matexp_su3_scaled (left) | 1950.0 | 86.43 | 8.50 | 31.61 | 90 | 4,096 | 256 |
| 70 | matmul_adgba_dual | 632.16 | 8.09 | 79.05 | 61.36 | 56 | 4,096 | 256 |
| 75 | fused_conv_c2c | 288.32 | 6.02 | 92.34 | 87.51 | 40 | 4,096 | 256 |
| 80 | matexp_su3_scaled (right) | 2380.0 | 86.32 | 6.88 | 29.82 | 90 | 4,096 | 256 |
| 81 | matmul_abc | 418.62 | 6.51 | 90.13 | 60.90 | 54 | 4,096 | 256 |
| 82 | reunitarize_su3 | 309.86 | 5.00 | 79.79 | 87.50 | 28 | 4,096 | 256 |
| 95 | matexp_su3 (IC large) | 1980.0 | 85.87 | 8.34 | 28.58 | 90 | 4,096 | 256 |
| 146 | path_order (Ny=50) | 4510.0 | 14.14 | 95.23 | 63.39 | 55 | 4,096 | 256 |

### Key Kernel Details

**assemble_algebra (IC)** (ID 22, `assemble_algebra`)
- Duration: 13.63 msecond
- SM throughput: 5.67%, DRAM: 44.23%, L1: 93.73%, L2: 80.95%
- Memory bandwidth: 403.08 GB/s, max: 80.95%
- L1 hit rate: 79.47%, L2 hit rate: 89.81%
- IPC active: 0.06, issue slots busy: 1.52%
- Occupancy: 91.41% (theoretical: 100%)
- Registers: 26, waves/SM: 416.26
- Occupancy limiters: registers=8, warps=6, SM=16
- Warp cycles/inst: 720.79, avg threads/warp: 32

**matexp_su3 (IC, unscaled)** (ID 24, `matexp_su3`)
- Duration: 98.03 msecond
- SM throughput: 86.88%, DRAM: 8.57%, L1: 3.85%, L2: 4.38%
- Memory bandwidth: 78.07 GB/s, max: 8.57%
- L1 hit rate: 82.84%, L2 hit rate: 58.66%
- IPC active: 0.16, issue slots busy: 4.01%
- Occupancy: 28.18% (theoretical: 33.33%)
- Registers: 90, waves/SM: 1,248.78
- Occupancy limiters: registers=2, warps=6, SM=16
- Warp cycles/inst: 84.28, avg threads/warp: 31.14

**path_order (IC)** (ID 25, `path_order`)
- Duration: 4.53 msecond
- SM throughput: 14.12%, DRAM: 95.22%, L1: 23.55%, L2: 50.23%
- Memory bandwidth: 866.78 GB/s, max: 95.22%
- L1 hit rate: 74.22%, L2 hit rate: 6.79%
- IPC active: 0.57, issue slots busy: 14.19%
- Occupancy: 63.49% (theoretical: 66.67%)
- Registers: 55, waves/SM: 12.49
- Occupancy limiters: registers=4, warps=6, SM=16
- Warp cycles/inst: 53.74, avg threads/warp: 32

**assemble_su3_noise** (ID 56, `assemble_su3_noise`)
- Duration: 551.52 usecond
- SM throughput: 5.58%, DRAM: 43.33%, L1: 92.20%, L2: 79.65%
- Memory bandwidth: 394.33 GB/s, max: 79.65%
- L1 hit rate: 79.13%, L2 hit rate: 89.78%
- IPC active: 0.06, issue slots busy: 1.56%
- Occupancy: 89.92% (theoretical: 100%)
- Registers: 26, waves/SM: 16.65
- Occupancy limiters: registers=8, warps=6, SM=16
- Warp cycles/inst: 700.59, avg threads/warp: 32

**assemble_algebra (evolution)** (ID 68, `assemble_algebra`)
- Duration: 282.21 usecond
- SM throughput: 5.53%, DRAM: 42.80%, L1: 91.48%, L2: 79.03%
- Memory bandwidth: 386.42 GB/s, max: 79.03%
- L1 hit rate: 79.27%, L2 hit rate: 89.81%
- IPC active: 0.06, issue slots busy: 1.52%
- Occupancy: 88.56% (theoretical: 100%)
- Registers: 26, waves/SM: 8.33
- Occupancy limiters: registers=8, warps=6, SM=16
- Warp cycles/inst: 705.52, avg threads/warp: 32

**matexp_su3_scaled (left)** (ID 69, `matexp_su3_scaled`)
- Duration: 1.95 msecond
- SM throughput: 86.43%, DRAM: 8.50%, L1: 4.07%, L2: 4.53%
- Memory bandwidth: 77.40 GB/s, max: 8.50%
- L1 hit rate: 82.80%, L2 hit rate: 60.07%
- IPC active: 0.15, issue slots busy: 3.77%
- Occupancy: 31.61% (theoretical: 33.33%)
- Registers: 90, waves/SM: 24.98
- Occupancy limiters: registers=2, warps=6, SM=16
- Warp cycles/inst: 100.32, avg threads/warp: 30.72

**matmul_adgba_dual** (ID 70, `matmul_adgba_dual`)
- Duration: 632.16 usecond
- SM throughput: 8.09%, DRAM: 72.51%, L1: 89.73%, L2: 79.05%
- Memory bandwidth: 658.88 GB/s, max: 79.05%
- L1 hit rate: 75.67%, L2 hit rate: 69.72%
- IPC active: 0.33, issue slots busy: 8.20%
- Occupancy: 61.36% (theoretical: 66.67%)
- Registers: 56, waves/SM: 12.49
- Occupancy limiters: registers=4, warps=6, SM=16
- Warp cycles/inst: 89.77, avg threads/warp: 32

**fused_conv_c2c** (ID 75, `fused_conv_c2c`)
- Duration: 288.32 usecond
- SM throughput: 6.02%, DRAM: 92.34%, L1: 19.41%, L2: 40.70%
- Memory bandwidth: 834.34 GB/s, max: 92.34%
- L1 hit rate: 0%, L2 hit rate: 31.10%
- IPC active: 0.24, issue slots busy: 5.89%
- Occupancy: 87.51% (theoretical: 100%)
- Registers: 40, waves/SM: 8.33
- Occupancy limiters: registers=6, warps=6, SM=16
- Warp cycles/inst: 177.99, avg threads/warp: 32

**matexp_su3_scaled (right)** (ID 80, `matexp_su3_scaled`)
- Duration: 2.38 msecond
- SM throughput: 86.32%, DRAM: 6.88%, L1: 3.15%, L2: 3.64%
- Memory bandwidth: 62.68 GB/s, max: 6.88%
- L1 hit rate: 83.34%, L2 hit rate: 59.31%
- IPC active: 0.15, issue slots busy: 3.78%
- Occupancy: 29.82% (theoretical: 33.33%)
- Registers: 90, waves/SM: 24.98
- Occupancy limiters: registers=2, warps=6, SM=16
- Warp cycles/inst: 94.99, avg threads/warp: 24.82

**matmul_abc** (ID 81, `matmul_abc`)
- Duration: 418.62 usecond
- SM throughput: 6.51%, DRAM: 90.13%, L1: 78.30%, L2: 70.61%
- Memory bandwidth: 812.20 GB/s, max: 90.13%
- L1 hit rate: 74.13%, L2 hit rate: 54.75%
- IPC active: 0.26, issue slots busy: 6.62%
- Occupancy: 60.90% (theoretical: 66.67%)
- Registers: 54, waves/SM: 12.49
- Occupancy limiters: registers=4, warps=6, SM=16
- Warp cycles/inst: 110.40, avg threads/warp: 32

**reunitarize_su3** (ID 82, `reunitarize_su3`)
- Duration: 309.86 usecond
- SM throughput: 5.00%, DRAM: 52.97%, L1: 90.27%, L2: 79.79%
- Memory bandwidth: 480.12 GB/s, max: 79.79%
- L1 hit rate: 84.36%, L2 hit rate: 79.89%
- IPC active: 0.11, issue slots busy: 2.88%
- Occupancy: 87.50% (theoretical: 100%)
- Registers: 28, waves/SM: 8.33
- Occupancy limiters: registers=8, warps=6, SM=16
- Warp cycles/inst: 365.26, avg threads/warp: 32

**matexp_su3 (IC large)** (ID 95, `matexp_su3`)
- Duration: 1.98 msecond
- SM throughput: 85.87%, DRAM: 8.34%, L1: 3.85%, L2: 4.35%
- Memory bandwidth: 75.88 GB/s, max: 8.34%
- L1 hit rate: 82.77%, L2 hit rate: 58.95%
- IPC active: 0.16, issue slots busy: 4.01%
- Occupancy: 28.58% (theoretical: 33.33%)
- Registers: 90, waves/SM: 24.98
- Occupancy limiters: registers=2, warps=6, SM=16
- Warp cycles/inst: 86.74, avg threads/warp: 31.02

**path_order (Ny=50)** (ID 146, `path_order`)
- Duration: 4.51 msecond
- SM throughput: 14.14%, DRAM: 95.23%, L1: 23.56%, L2: 50.27%
- Memory bandwidth: 870.72 GB/s, max: 95.23%
- L1 hit rate: 74.28%, L2 hit rate: 6.80%
- IPC active: 0.57, issue slots busy: 14.22%
- Occupancy: 63.39% (theoretical: 66.67%)
- Registers: 55, waves/SM: 12.49
- Occupancy limiters: registers=4, warps=6, SM=16
- Warp cycles/inst: 53.67, avg threads/warp: 32

## cuFFT Kernels

| ID | Label | Duration (μs) | SM % | Mem % | Occupancy % |
|---:|-------|-------------:|-----:|------:|------------:|
| 14 | rfft2 r2c (IC, 8 fields) | 3980.0 | 35.75 | 92.67 | 32.92 |
| 15 | fft regular (IC) | 4890.0 | 17.26 | 76.36 | 31.15 |
| 18 | ifft regular (IC) | 4910.0 | 17.21 | 76.15 | 30.75 |
| 19 | irfft2 c2r (IC) | 3980.0 | 35.81 | 92.85 | 33.01 |
| 59 | rfft2 r2c (evol left) | 162.91 | 35.59 | 89.73 | 32.17 |
| 60 | fft regular (evol left) | 217.47 | 15.66 | 67.77 | 30.93 |
| 65 | ifft regular (evol left) | 116.80 | 14.71 | 62.50 | 30.79 |
| 66 | irfft2 c2r (evol left) | 84.19 | 34.85 | 85.55 | 31.59 |
| 73 | fft regular (evol right 1) | 416.61 | 18.39 | 79.54 | 31.66 |
| 74 | fft vector (evol right 2) | 362.66 | 29.99 | 91.96 | 56.01 |
| 76 | ifft regular (evol right 1) | 218.05 | 17.62 | 75.48 | 31.08 |
| 77 | ifft vector (evol right 2) | 182.75 | 30.01 | 90.92 | 55.34 |

## Bottleneck Classification

A kernel is **compute-bound** if SM% > Mem%, **memory-bound** if Mem% > SM%.

| Kernel | Bound | SM % | Mem % | Limiting Factor |
|--------|-------|-----:|------:|-----------------|
| assemble_algebra (IC) | Memory | 5.67 | 80.95 | DRAM bandwidth |
| matexp_su3 (IC, unscaled) | Compute | 86.88 | 8.57 | Low occupancy (28.18%), 90 regs |
| path_order (IC) | Memory | 14.12 | 95.22 | DRAM bandwidth |
| assemble_su3_noise | Memory | 5.58 | 79.65 | DRAM bandwidth |
| assemble_algebra (evolution) | Memory | 5.53 | 79.03 | DRAM bandwidth |
| matexp_su3_scaled (left) | Compute | 86.43 | 8.50 | Low occupancy (31.61%), 90 regs |
| matmul_adgba_dual | Memory | 8.09 | 79.05 | DRAM bandwidth |
| fused_conv_c2c | Memory | 6.02 | 92.34 | DRAM bandwidth |
| matexp_su3_scaled (right) | Compute | 86.32 | 6.88 | Low occupancy (29.82%), 90 regs |
| matmul_abc | Memory | 6.51 | 90.13 | DRAM bandwidth |
| reunitarize_su3 | Memory | 5.00 | 79.79 | DRAM bandwidth |
| matexp_su3 (IC large) | Compute | 85.87 | 8.34 | Low occupancy (28.58%), 90 regs |
| path_order (Ny=50) | Memory | 14.14 | 95.23 | DRAM bandwidth |