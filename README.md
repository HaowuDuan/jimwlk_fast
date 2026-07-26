# jimwlk_cuda

GPU-accelerated JIMWLK evolution of SU(3) Wilson lines on a 2D transverse lattice, implemented with CuPy and CUDA RawKernels.

Evolves the Wilson line V(x) via the Langevin equation:

    V(x) → exp(-ε ξ_L·K) · V(x) · exp(+ε ξ_R·K)

where ξ are SU(3) color noise fields, K is the Weizsäcker-Williams kernel, and ε = √(α_s dY)/π.

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

Typical timing on an RTX 3090: ~1.2 s/config at N=1024, ~0.3 s/config at N=512.

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
