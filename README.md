# jimwlk_cuda

GPU-accelerated JIMWLK evolution of SU(3) Wilson lines on a 2D transverse lattice, implemented with CuPy and CUDA RawKernels.

Evolves the Wilson line V(x) via the Langevin equation:

    V(x) → exp(-ε ξ_L·K) · V(x) · exp(+ε ξ_R·K)

where ξ are SU(3) color noise fields, K is the Weizsäcker-Williams kernel, and ε = √(α_s dY)/π.

## Performance and optimizations

The implementation is optimized for the many independent 3×3 SU(3) operations in each lattice update. The original CuPy implementation spent 95.5% of its profiled GPU time in generic batched GEMM kernels and launched thousands of elementwise kernels. The optimized path instead uses workload-specific CUDA kernels to accelerate the dominant SU(3) operations.

### Optimization techniques

- **Specialized 3×3 CUDA kernels:** per-site, register-resident kernels replace `cupy.matmul` for `V†ξV`, `exp_L · V · exp_R`, and initial-condition path ordering. Both polarizations are rotated together, and the three-matrix Wilson-line update is completed in one launch.
- **Fused SU(3) operations:** Cayley-Hamilton matrix exponentiation, scaling, algebra assembly, and reunitarization run as fused `RawKernel`s. Matrix data stays in `complex64`; only the numerically sensitive eigenvalue and coefficient calculations use `float64`.
- **Batched initial conditions:** all 50 longitudinal layers are generated, transformed, exponentiated, and path-ordered in batches instead of repeating the pipeline layer by layer.
- **Lower FFT and memory overhead:** real-to-complex FFTs store only the half-spectrum where possible; the right-side convolution is fused; plane-major contiguous layouts improve coalescing; and the propagator and Weizsäcker-Williams kernel are cached.
- **Fused observables:** gauge-field extraction, adjoint Wilson-line construction, WW-field extraction, and power-spectrum accumulation avoid the large temporary arrays produced by `einsum` chains.

### Profiled hot paths

The table compares the naive CuPy profile with the optimized Nsight Compute profile on an NVIDIA GeForce RTX 3090, using N=1024, L=64, N_y=50, and `dY=0.01`.

| Hot path | Naive CuPy | Optimized | Speedup |
|----------|-----------:|----------:|--------:|
| Initial-condition matrix exponentials (50 layers) | 8.650 s | 98.03 ms | 88× |
| Initial-condition path ordering | 506.2 ms | 4.53 ms | 112× |
| Adjoint rotation `V†ξV` (two polarizations) | 571.7 ms | 0.632 ms | 904× |
| Wilson-line update `exp_L · V · exp_R` | 278.9 ms | 0.419 ms | 666× |

### Evolution benchmark

For the same N=512, 100-step workload, excluding observable evaluation, the measured average evolution time per configuration is:

| Implementation | Evolution time/config | Speedup vs CPU | Speedup vs pure CuPy |
|----------------|----------------------:|---------------:|---------------------:|
| CPU | 173.6400 s | 1.00× | — |
| Pure CuPy | 38.1720 s | 4.55× | 1.00× |
| **Optimized `jimwlk_cuda`** | **0.2455 s** | **707.3×** | **155.5×** |

The optimized implementation is therefore **155.5× faster than pure CuPy** and **707.3× faster than the CPU version** for this evolution-only benchmark. Observable costs must be measured separately and depend strongly on which quantities are enabled and how frequently they are sampled. Wall-clock results vary with hardware, CUDA/CuPy versions, and memory pressure. See [`naive_profile_report.md`](naive_profile_report.md) and [`ncu_profile_report.md`](ncu_profile_report.md) for the detailed kernel measurements.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.x
- Dependencies: `pip install -r requirements.txt`

```
numpy>=1.24
cupy-cuda12x>=13.0
scipy>=1.10
matplotlib>=3.7
```

## Quick start

Run an ensemble of 1000 MV initial condition configs on a 512×512 lattice:

```bash
python -m jimwlk_cuda.evolution_ensemble \
    --N 512 --l 32 --ic mv \
    --Yf 1.0 --dY 0.01 \
    --n_configs 1000 --measure_every 10 \
    --seed 42 --outdir results_mv/
```

## Command-line options

| Flag | Default | Description |
|------|---------|-------------|
| `--N` | 512 | Lattice sites per dimension (N×N grid) |
| `--l` | 32 | Physical lattice size L (lattice spacing a = L/N) |
| `--ic` | `mv` | Initial condition: `mv` (McLerran-Venugopalan) or `cn` (color-neutralized) |
| `--mu2` | 1.0 (MV) / 0.4974 (CN) | Color charge variance μ² |
| `--m2` | (πk_IR)² | IR regulator mass squared |
| `--Yf` | 1.0 | Final rapidity |
| `--dY` | 0.01 | Rapidity step size |
| `--n_configs` | 1000 | Number of independent configurations |
| `--measure_every` | 10 | Measure observables every N steps |
| `--seed` | 42 | Base random seed (config i uses seed + i) |
| `--outdir` | `results_gpu` | Output directory |
| `--derivative` | `forward` | Gauge field derivative: `forward`, `centered`, or `spectral` |

## Example runs

**N=1024, L=64, MV then CN:**

```bash
PY=.venv/bin/python3

# MV initial conditions (μ²=1.0)
$PY -m jimwlk_cuda.evolution_ensemble \
    --N 1024 --l 64 --ic mv --mu2 1.0 \
    --m2 0.0024095713869847065 \
    --Yf 1.0 --dY 0.01 \
    --n_configs 1000 --measure_every 10 \
    --seed 42 --outdir results_gpu_mv_xG_L64_N1024

# CN initial conditions (μ²=3.0)
$PY -m jimwlk_cuda.evolution_ensemble \
    --N 1024 --l 64 --ic cn --mu2 3.0 \
    --m2 0.0024095713869847065 \
    --Yf 1.0 --dY 0.01 \
    --n_configs 1000 --measure_every 10 \
    --seed 42 --outdir results_gpu_cn_xG_L64_N1024
```

## Observables

The code measures these observables at each measurement step:

- **Dipole amplitude S(r)**: the Fourier correlator of Wilson lines projected onto Gell-Mann generators, radially binned in coordinate space. S(r=0) = 1 by definition; the saturation scale Q_s is extracted from S(r_s) = e^{-1/2}.

- **Gluon TMDs xG(r), xh(r)**: gauge field correlators from lattice derivatives of V. The gauge field is A^a_i(x) = (2/ig) tr(V†∂_iV · t^a). xG is the unpolarized gluon TMD, xh is the linearly polarized gluon TMD.

- **k-space TMDs xG(k), xh(k)**: power spectra of the gauge field components. The ratio xh/xG measures the degree of linear polarization.

## Output files

Each run produces CSV files in the output directory:

| File | Contents |
|------|----------|
| `S_ensemble.csv` | Dipole S(r) at each measured Y. Columns: Y, r=0.125, r=0.250, ... |
| `xG_ensemble.csv` | xG(r) at each measured Y |
| `xh_ensemble.csv` | xh(r) at each measured Y |
| `xG_k_ensemble.csv` | xG(k) at each measured Y |
| `xh_k_ensemble.csv` | xh(k) at each measured Y |
| `Qs.csv` | Saturation scale Q_s(Y) for ensemble and each config |
| `params.csv` | Run parameters for reproducibility |

## Initial conditions

**MV (McLerran-Venugopalan):** Gaussian random color charges with variance μ², uncorrelated between lattice sites. The Wilson line is path-ordered over N_y = 50 longitudinal layers.

**CN (color-neutralized):** Uses a modified propagator with parameters (Q, β, γ) fitted to match the NN dipole at x = 0.01. Defaults: μ² = 0.4974, Q = 0.317, β = 0.675, γ = 1.681. The `--mu2` flag overrides μ² while keeping the other CN parameters fixed.

## Module structure

| File | Role |
|------|------|
| `evolution_ensemble.py` | CLI entry point, ensemble averaging, CSV output |
| `evolution_single.py` | Single-config evolution loop, WW kernel, convolution |
| `observables.py` | Dipole S(r), TMDs xG/xh measurement |
| `config.py` | Lattice parameters, Gell-Mann generators |
| `initial_conditions_mv.py` | MV initial condition generator |
| `initial_conditions_cn.py` | CN initial condition generator |
| `noise_kernel.py` | SU(3) noise generation |
| `su3_kernel.py` | Matrix exponential, reunitarization |
| `matmul3x3_kernel.py` | 3×3 complex matrix products |
| `ic_fast.py` | Batched initial condition generation |
| `observables_fast.py` | Fused gauge field extraction and power spectra |

## Plotting

Use `plot_L64.py` for MV vs CN comparison (k-space TMDs with k/Q_s scaling):

```bash
python plot_L64.py
```

This reads from `results_gpu_mv_xG_L64/` and `results_gpu_cn_xG_L64/`. Edit the directory names in the script to point at your output.
