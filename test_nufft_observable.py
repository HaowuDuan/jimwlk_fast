"""Compare NUFFT vs FFT observable extraction."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cupy as cp
import numpy as np
from jimwlk_cuda import config as cfg
from jimwlk_cuda.ic_fast import compute_path_ordered_Wilson_line

cfg.N = 512
cfg.l = 32
cfg.a = cfg.l / cfg.N
cfg.a2 = cfg.a ** 2

print(f"N={cfg.N}, a={cfg.a}, L={cfg.l}")

# Generate a Wilson line
rng = cp.random.Generator(cp.random.XORWOW(seed=42))
from jimwlk_cupy.initial_condition import build_propagator
propagator = build_propagator()
sigma = cfg.variance_of_mv_noise
V = compute_path_ordered_Wilson_line(rng, propagator, sigma, cfg.Ny)
cp.cuda.Device().synchronize()
print(f"V shape: {V.shape}")

# ── FFT-based (existing) ────────────────────────────────────────────────────
from jimwlk_cuda.observables import measure_xG_xh, measure_xG_xh_nufft

cp.cuda.Device().synchronize()
s = cp.cuda.Event(); e = cp.cuda.Event()
s.record()
r_vals, xG_r, xh_r, k_vals_fft, xG_k_fft, xh_k_fft = measure_xG_xh(V)
e.record(); e.synchronize()
t_fft = cp.cuda.get_elapsed_time(s, e)
print(f"\nFFT observable:  {t_fft:.1f} ms")
print(f"  k range: [{k_vals_fft[1]:.4f}, {k_vals_fft[-1]:.4f}], {len(k_vals_fft)} points")

# ── NUFFT at same uniform k-points (validation) ─────────────────────────────
k_uniform = k_vals_fft[1:]  # skip k=0

cp.cuda.Device().synchronize()
s = cp.cuda.Event(); e = cp.cuda.Event()
s.record()
k_nu, xG_k_nu, xh_k_nu = measure_xG_xh_nufft(V, k_points=k_uniform)
e.record(); e.synchronize()
t_nufft_uniform = cp.cuda.get_elapsed_time(s, e)
print(f"\nNUFFT at uniform k:  {t_nufft_uniform:.1f} ms")

# Compare
xG_fft_sub = xG_k_fft[1:]
xh_fft_sub = xh_k_fft[1:]

mask = np.abs(xG_fft_sub) > 1e-15
rel_err_xG = np.abs(xG_k_nu[mask] - xG_fft_sub[mask]) / np.abs(xG_fft_sub[mask])
mask_h = np.abs(xh_fft_sub) > 1e-15
rel_err_xh = np.abs(xh_k_nu[mask_h] - xh_fft_sub[mask_h]) / np.abs(xh_fft_sub[mask_h])

print(f"  xG max rel error: {rel_err_xG.max():.2e}  (median: {np.median(rel_err_xG):.2e})")
print(f"  xh max rel error: {rel_err_xh.max():.2e}  (median: {np.median(rel_err_xh):.2e})")

# ── NUFFT at log-spaced k-points ────────────────────────────────────────────
cp.cuda.Device().synchronize()
s = cp.cuda.Event(); e = cp.cuda.Event()
s.record()
k_log, xG_k_log, xh_k_log = measure_xG_xh_nufft(V, n_k=200)
e.record(); e.synchronize()
t_nufft_log = cp.cuda.get_elapsed_time(s, e)
print(f"\nNUFFT at 200 log-spaced k:  {t_nufft_log:.1f} ms")
print(f"  k range: [{k_log[0]:.4f}, {k_log[-1]:.4f}]")
print(f"  xG range: [{xG_k_log.min():.4e}, {xG_k_log.max():.4e}]")

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"FFT time:              {t_fft:.1f} ms")
print(f"NUFFT (uniform, {len(k_uniform)} k): {t_nufft_uniform:.1f} ms")
print(f"NUFFT (log, 200 k):    {t_nufft_log:.1f} ms")
