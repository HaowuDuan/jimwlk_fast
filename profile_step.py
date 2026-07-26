"""Profile one JIMWLK evolution config — per-component GPU timings.

Usage:
    python -m jimwlk_cuda.profile_step [--N 512] [--steps 20]
"""

import argparse
import numpy as np
import cupy as cp
from . import config as cfg
from .su3_kernel import matexp_su3
from .matmul3x3_kernel import matmul_abc, matmul_adgba_dual
from .noise_kernel import generate_noise_raw_batch
from .ic_fast import _assemble_algebra_kernel, _block as _ic_block
from .evolution_single import precompute_ww_kernel, _fused_conv_right, compute_exp
from .initial_conditions_mv import compute_path_ordered_fund_Wilson_line
from .observables import measure_dipole, Qs_of_S


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=512)
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()

    cfg.N = args.N
    cfg.a = cfg.l / cfg.N
    cfg.a2 = cfg.a**2
    cfg.variance_of_mv_noise = float(np.sqrt(cfg.mu2 / (cfg.Ny * cfg.a2)))
    N = cfg.N
    W = N // 2 + 1

    dev = cp.cuda.Device()
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(dev.id)['name'].decode()}")
    print(f"N={N}, profiling {args.steps} steps\n")

    rng = cp.random.Generator(cp.random.XORWOW(seed=42))

    # --- Profile MV init ---
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    V = compute_path_ordered_fund_Wilson_line(rng)
    end.record()
    end.synchronize()
    t_init = cp.cuda.get_elapsed_time(start, end)
    print(f"MV init (50 layers):  {t_init:.1f} ms\n")

    K_of_k = precompute_ww_kernel()
    K_r2c = cp.ascontiguousarray(K_of_k[:, :, :W])
    dY = cfg.dY
    pref = np.sqrt(cfg.alpha_fc * dY) / np.pi

    xi_R = cp.zeros((2, N, N, 3, 3), dtype=cp.complex64)

    t_noise = []
    t_rfft_noise = []
    t_conv_L = []
    t_irfft_L = []
    t_assemble_L = []
    t_matexp_L = []
    t_assemble_xi = []
    t_rotate = []
    t_fft_R = []
    t_conv_R = []
    t_ifft_R = []
    t_transpose_R = []
    t_matexp_R = []
    t_vupdate = []
    t_measure = []
    t_full_step = []

    for step in range(args.steps):
        s_full = cp.cuda.Event(); e_full = cp.cuda.Event()
        s = cp.cuda.Event(); e = cp.cuda.Event()

        s_full.record()

        # --- LEFT SIDE (R2C path) ---

        # noise generation (raw float32)
        s.record()
        raw = generate_noise_raw_batch(rng, N, 1)
        e.record(); e.synchronize()
        t_noise.append(cp.cuda.get_elapsed_time(s, e))

        # rfft2 on 8 real fields
        s.record()
        raw_k = cp.fft.rfft2(raw, axes=(-2, -1))
        e.record(); e.synchronize()
        t_rfft_noise.append(cp.cuda.get_elapsed_time(s, e))

        # convolution (broadcast)
        s.record()
        conv_k = raw_k[:, :, 0] * K_r2c[0] + raw_k[:, :, 1] * K_r2c[1]
        e.record(); e.synchronize()
        t_conv_L.append(cp.cuda.get_elapsed_time(s, e))

        # irfft2
        s.record()
        conv = cp.fft.irfft2(conv_k, s=(N, N), axes=(-2, -1))
        e.record(); e.synchronize()
        t_irfft_L.append(cp.cuda.get_elapsed_time(s, e))

        # assemble convolved coefficients
        s.record()
        n_sites = N * N
        conv_flat = cp.ascontiguousarray(conv.reshape(8, n_sites))
        A_mat = cp.empty((n_sites, 9), dtype=cp.complex64)
        grid = (n_sites + _ic_block - 1) // _ic_block
        _assemble_algebra_kernel((grid,), (_ic_block,), (conv_flat, A_mat, n_sites))
        e.record(); e.synchronize()
        t_assemble_L.append(cp.cuda.get_elapsed_time(s, e))

        # matexp (exp_L)
        s.record()
        exp_L = matexp_su3(A_mat.reshape(N, N, 3, 3), scale=float(-pref))
        e.record(); e.synchronize()
        t_matexp_L.append(cp.cuda.get_elapsed_time(s, e))

        # assemble raw noise for right side
        s.record()
        n_sites_all = 2 * N * N
        raw_flat = cp.ascontiguousarray(raw.reshape(8, n_sites_all))
        xi_mat = cp.empty((n_sites_all, 9), dtype=cp.complex64)
        grid2 = (n_sites_all + _ic_block - 1) // _ic_block
        _assemble_algebra_kernel((grid2,), (_ic_block,), (raw_flat, xi_mat, n_sites_all))
        xi = xi_mat.reshape(2, N, N, 3, 3)
        e.record(); e.synchronize()
        t_assemble_xi.append(cp.cuda.get_elapsed_time(s, e))

        # --- RIGHT SIDE (C2C + fused kernel) ---

        # rotation V† @ xi @ V
        s.record()
        matmul_adgba_dual(V, xi[0], xi[1], out0=xi_R[0], out1=xi_R[1])
        e.record(); e.synchronize()
        t_rotate.append(cp.cuda.get_elapsed_time(s, e))

        # fft2 plane-major
        s.record()
        xi_R_flat = cp.ascontiguousarray(xi_R.reshape(2, 9, N, N))
        xi_R_k = cp.fft.fft2(xi_R_flat, axes=(-2, -1))
        e.record(); e.synchronize()
        t_fft_R.append(cp.cuda.get_elapsed_time(s, e))

        # fused convolution
        s.record()
        conv_R_k = _fused_conv_right(K_of_k, xi_R_k)
        e.record(); e.synchronize()
        t_conv_R.append(cp.cuda.get_elapsed_time(s, e))

        # ifft2
        s.record()
        conv_R = cp.fft.ifft2(conv_R_k, axes=(-2, -1))
        e.record(); e.synchronize()
        t_ifft_R.append(cp.cuda.get_elapsed_time(s, e))

        # transpose to site-major
        s.record()
        xi_conv_R = cp.ascontiguousarray(conv_R.transpose(1, 2, 0)).reshape(N, N, 3, 3)
        e.record(); e.synchronize()
        t_transpose_R.append(cp.cuda.get_elapsed_time(s, e))

        # matexp (exp_R)
        s.record()
        exp_R = matexp_su3(xi_conv_R, scale=float(pref))
        e.record(); e.synchronize()
        t_matexp_R.append(cp.cuda.get_elapsed_time(s, e))

        # --- V update ---
        s.record()
        matmul_abc(exp_L, V, exp_R, out=V)
        e.record(); e.synchronize()
        t_vupdate.append(cp.cuda.get_elapsed_time(s, e))

        e_full.record(); e_full.synchronize()
        t_full_step.append(cp.cuda.get_elapsed_time(s_full, e_full))

    # --- measure_dipole ---
    for _ in range(3):
        s = cp.cuda.Event(); e = cp.cuda.Event()
        s.record()
        r_bins, Sb = measure_dipole(V)
        Qs = Qs_of_S(r_bins, Sb)
        e.record(); e.synchronize()
        t_measure.append(cp.cuda.get_elapsed_time(s, e))

    # --- Print results ---
    def stats(name, arr):
        a = np.array(arr)
        if len(a) > 2:
            a = a[2:]
        return f"{name:30s}  {np.mean(a):8.2f} ms  (std {np.std(a):6.2f})"

    print("=" * 60)
    print("Per-step breakdown (ms, excluding first 2 warmup steps):")
    print("=" * 60)
    print("--- LEFT (R2C) ---")
    print(stats("noise gen (raw f32)", t_noise))
    print(stats("rfft2 (8 real fields)", t_rfft_noise))
    print(stats("conv L (broadcast)", t_conv_L))
    print(stats("irfft2", t_irfft_L))
    print(stats("assemble conv -> matrix", t_assemble_L))
    print(stats("matexp (exp_L)", t_matexp_L))
    print(stats("assemble raw -> xi", t_assemble_xi))
    print("--- RIGHT (C2C + fused) ---")
    print(stats("rotation V†·ξ·V (dual)", t_rotate))
    print(stats("fft2 plane-major", t_fft_R))
    print(stats("fused conv R", t_conv_R))
    print(stats("ifft2 R", t_ifft_R))
    print(stats("transpose -> site-major", t_transpose_R))
    print(stats("matexp (exp_R)", t_matexp_R))
    print("--- UPDATE ---")
    print(stats("V update (exp_L·V·exp_R)", t_vupdate))
    print("-" * 60)
    print(stats("FULL STEP", t_full_step))
    print()
    print(stats("measure_dipole + Qs", t_measure))

    full = np.array(t_full_step[2:])
    print(f"\nExtrapolated 500 steps:  {np.mean(full) * 500 / 1000:.1f} s")
    print(f"Extrapolated 1000 configs:  {np.mean(full) * 500 * 1000 / 1000 / 3600:.1f} h")
    print(f"MV init per config:  {t_init / 1000:.1f} s")
    print(f"Total per config (est):  {(np.mean(full) * 500 + t_init) / 1000:.1f} s")


if __name__ == '__main__':
    main()
