"""Cross-check the fused WW TMD path and print the xh/xG ratios."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cupy as cp
import numpy as np

from jimwlk_cuda import config as cfg
from jimwlk_cuda.initial_conditions_mv import (
    compute_path_ordered_fund_Wilson_line,
)
from jimwlk_cuda.observables import measure_ww_tmd, measure_xG_xh
from jimwlk_cuda.observables_fast import (
    extract_adjoint_wilson_line_fast,
    extract_ww_field_fast,
)


def _standard_structure_constants():
    """Return the standard f^{cab}, defined by [t^a,t^b] = i f^{abc}t^c."""
    t = cfg.generators[:8].get()
    f_abc = np.zeros((8, 8, 8), dtype=np.float32)
    for c in range(8):
        for a in range(8):
            for b in range(8):
                comm = t[a] @ t[b] - t[b] @ t[a]
                f_abc[c, a, b] = (
                    -2.0j * np.trace(comm @ t[c])
                ).real
    return cp.asarray(f_abc)


def _spectral_derivatives(U):
    N = cfg.N
    momentum = (
        2.0 * cp.pi * cp.fft.fftfreq(N, d=cfg.a)
    ).astype(cp.float64)
    Uk = cp.fft.fft2(U, axes=(0, 1))
    dU_x = cp.fft.ifft2(
        1j * momentum.reshape(N, 1, 1, 1) * Uk,
        axes=(0, 1),
    ).real.astype(cp.float32)
    dU_y = cp.fft.ifft2(
        1j * momentum.reshape(1, N, 1, 1) * Uk,
        axes=(0, 1),
    ).real.astype(cp.float32)
    return dU_x, dU_y


def _slow_ww_fields(V):
    """Original CuPy einsum chain, retained as a test reference."""
    t = cfg.generators[:8]
    Vd = V.conj().transpose(0, 1, 3, 2)
    tV = cp.einsum("acd,ijde->aijce", t, V)
    tVd = cp.einsum("bcd,ijde->bijce", t, Vd)
    U = 2.0 * cp.einsum(
        "aijck,bijkc->ijab", tV, tVd
    ).real.astype(cp.float32)

    dU_x, dU_y = _spectral_derivatives(U)
    Ut = U.transpose(0, 1, 3, 2)
    A_x = cp.einsum("ijab,ijbc->ijac", Ut, dU_x)
    A_y = cp.einsum("ijab,ijbc->ijac", Ut, dU_y)

    f_abc = _standard_structure_constants()
    alpha_x = (-1.0 / 3.0) * cp.einsum(
        "cab,ijba->ijc", f_abc, A_x
    )
    alpha_y = (-1.0 / 3.0) * cp.einsum(
        "cab,ijba->ijc", f_abc, A_y
    )
    return U, alpha_x.transpose(2, 0, 1), alpha_y.transpose(2, 0, 1)


def _tmd_from_fields(alpha_x, alpha_y):
    N = cfg.N
    a = cfg.a
    norm = cp.float32(a**2 / (4.0 * np.pi**2))
    ak_x = norm * cp.fft.fft2(alpha_x, axes=(1, 2))
    ak_y = norm * cp.fft.fft2(alpha_y, axes=(1, 2))
    prefactor = 8.0 * np.pi / (N * a) ** 2
    G_xx = prefactor * cp.sum(cp.abs(ak_x)**2, axis=0).real
    G_xy = prefactor * cp.sum((ak_x * ak_y.conj()).real, axis=0)
    G_yy = prefactor * cp.sum(cp.abs(ak_y)**2, axis=0).real

    k_raw = 2.0 * cp.pi * cp.fft.fftfreq(N, d=a).astype(cp.float64)
    KX, KY = cp.meshgrid(k_raw, k_raw, indexing="ij")
    K2 = KX**2 + KY**2
    tmd_G = (G_xx + G_yy).astype(cp.float64)
    projected = (
        KX**2 * G_xx
        + 2.0 * KX * KY * G_xy
        + KY**2 * G_yy
    ).astype(cp.float64)
    tmd_H = cp.zeros_like(tmd_G)
    mask = K2 > 0
    tmd_H[mask] = 2.0 * projected[mask] / K2[mask] - tmd_G[mask]

    N2 = N // 2
    return (
        cp.asnumpy(k_raw[:N2]),
        cp.asnumpy(tmd_G[:N2, 0]),
        cp.asnumpy(tmd_H[:N2, 0]),
    )


def cross_check_fast_against_slow(V):
    """Check both fused kernels and the public end-to-end WW result."""
    U_slow, alpha_x_slow, alpha_y_slow = _slow_ww_fields(V)

    U_fast = extract_adjoint_wilson_line_fast(V)
    cp.testing.assert_allclose(U_fast, U_slow, rtol=2e-5, atol=2e-6)

    dU_x, dU_y = _spectral_derivatives(U_fast)
    alpha_x_fast, alpha_y_fast = extract_ww_field_fast(
        U_fast, dU_x, dU_y
    )
    cp.testing.assert_allclose(
        alpha_x_fast, alpha_x_slow, rtol=2e-4, atol=2e-5
    )
    cp.testing.assert_allclose(
        alpha_y_fast, alpha_y_slow, rtol=2e-4, atol=2e-5
    )

    k_slow, G_slow, H_slow = _tmd_from_fields(
        alpha_x_slow, alpha_y_slow
    )
    k_fast, G_fast, H_fast = measure_ww_tmd(V)
    np.testing.assert_allclose(k_fast, k_slow, rtol=0.0, atol=0.0)
    spectrum_atol = max(1e-12, 5e-6 * np.max(np.abs(G_slow)))
    np.testing.assert_allclose(
        G_fast, G_slow, rtol=5e-4, atol=spectrum_atol
    )
    np.testing.assert_allclose(
        H_fast, H_slow, rtol=5e-4, atol=spectrum_atol
    )
    print("Fast WW kernels agree with the slow CuPy einsum reference.")
    return k_fast, G_fast, H_fast


def _configure_lattice(N, Ny):
    cfg.N = N
    cfg.l = 32
    cfg.a = cfg.l / cfg.N
    cfg.a2 = cfg.a**2
    cfg.Ny = Ny
    cfg.variance_of_mv_noise = float(
        np.sqrt(cfg.mu2 / (cfg.Ny * cfg.a2))
    )


def _make_mv_wilson_line(seed):
    rng = cp.random.Generator(cp.random.XORWOW(seed=seed))
    V = compute_path_ordered_fund_Wilson_line(rng)
    cp.cuda.Stream.null.synchronize()
    return V


def main():
    # Keep the slow reference check small, then run the original ratio setup
    # entirely through the new public fast path.
    _configure_lattice(N=32, Ny=6)
    V_check = _make_mv_wilson_line(seed=42)
    cross_check_fast_against_slow(V_check)
    del V_check
    cp.get_default_memory_pool().free_all_blocks()

    _configure_lattice(N=512, Ny=50)
    V = _make_mv_wilson_line(seed=42)
    k_ww, ww_G, ww_H = measure_ww_tmd(V)
    _, _, _, k_fft, xG_k, xh_k = measure_xG_xh(
        V, derivative="spectral"
    )

    print("\nFundamental xh/xG vs WW H/G along k_x (k_y=0):")
    print(
        f"{'k':>8s}  {'fund_xh/xG':>12s}  "
        f"{'WW_H/G':>12s}  {'diff':>10s}"
    )
    for idx in [1, 2, 4, 8, 16, 32, 64, 128, 200, 250]:
        if idx >= len(k_fft):
            continue
        fund_ratio = (
            xh_k[idx] / xG_k[idx]
            if abs(xG_k[idx]) > 1e-20
            else float("nan")
        )
        ww_ratio = (
            ww_H[idx] / ww_G[idx]
            if abs(ww_G[idx]) > 1e-20
            else float("nan")
        )
        print(
            f"{k_fft[idx]:8.3f}  {fund_ratio:12.6f}  "
            f"{ww_ratio:12.6f}  {fund_ratio - ww_ratio:10.6f}"
        )


if __name__ == "__main__":
    main()
