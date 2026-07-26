"""Compare WW TMD (adjoint, spectral deriv) vs fundamental xG_k after fix."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from jimwlk_cuda import config as cfg
from jimwlk_cuda.initial_conditions_mv import compute_path_ordered_fund_Wilson_line
from jimwlk_cuda.observables import measure_xG_xh, measure_ww_tmd

cfg.N = 512
cfg.l = 32
cfg.a = cfg.l / cfg.N
cfg.a2 = cfg.a ** 2
cfg.Ny = 50
cfg.variance_of_mv_noise = float(np.sqrt(cfg.mu2 / (cfg.Ny * cfg.a2)))

rng = cp.random.Generator(cp.random.XORWOW(seed=42))
V = compute_path_ordered_fund_Wilson_line(rng)
cp.cuda.Stream.null.synchronize()

# Fundamental: forward derivative
r_vals, xG_r, xh_r, k_fft, xG_k_fwd, xh_k_fwd = measure_xG_xh(V, derivative='forward')

# Fundamental: spectral derivative
_, _, _, _, xG_k_spec, xh_k_spec = measure_xG_xh(V, derivative='spectral')

# WW TMD (adjoint, now spectral derivative)
k_ww, ww_G, ww_H = measure_ww_tmd(V)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.plot(k_fft[1:], xG_k_fwd[1:], 'o-', ms=2, label='Fundamental (forward)', alpha=0.7)
ax.plot(k_fft[1:], xG_k_spec[1:], 's-', ms=2, label='Fundamental (spectral)', alpha=0.7)
ax.plot(k_ww[1:], np.abs(ww_G[1:]), '^-', ms=2, label='WW TMD G (adjoint, spectral)', alpha=0.7)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('k')
ax.set_ylabel('xG(k) / WW G')
ax.set_title('xG(k): Fundamental vs WW TMD')
ax.legend()
ax.set_xlim(0.15, 60)

ax = axes[1]
ax.plot(k_fft[1:], np.abs(xh_k_fwd[1:]), 'o-', ms=2, label='Fundamental xh (forward)', alpha=0.7)
ax.plot(k_fft[1:], np.abs(xh_k_spec[1:]), 's-', ms=2, label='Fundamental xh (spectral)', alpha=0.7)
ax.plot(k_ww[1:], np.abs(ww_H[1:]), '^-', ms=2, label='WW TMD H (adjoint, spectral)', alpha=0.7)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('k')
ax.set_ylabel('|xh(k)| / |WW H|')
ax.set_title('|xh(k)|: Fundamental vs WW TMD')
ax.legend()
ax.set_xlim(0.15, 60)

plt.suptitle('WW TMD fix: spectral derivatives, raw FFT momentum', fontsize=13)
plt.tight_layout()
plt.savefig('ww_tmd_fix_comparison.png', dpi=150)
print("Saved ww_tmd_fix_comparison.png")

# Print ratio at a few k values
print("\nRatio WW_G / xG_k (spectral) at select k:")
for idx in [2, 8, 32, 64, 128, 200]:
    if idx < len(k_fft) and idx < len(k_ww):
        ratio = ww_G[idx] / xG_k_spec[idx] if abs(xG_k_spec[idx]) > 1e-20 else float('nan')
        print(f"  k={k_fft[idx]:.3f}  ratio={ratio:.4f}")
