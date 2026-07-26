import numpy as np
import cupy as cp
from . import config as cfg
from .su3_kernel import matexp_su3


def inv_propagator():
    i = cp.arange(cfg.N, dtype=cp.float64)
    si = cp.sin(cp.pi * i / cfg.N) ** 2
    denom = cfg.a2 * cfg.m2 + 4.0 * (si[:, None] + si[None, :])
    return (cfg.a2 / denom).astype(cp.float32)


_cached_D = None
_cached_D_key = None


def _get_inv_propagator():
    global _cached_D, _cached_D_key
    key = (cfg.N, cfg.a, cfg.m2)
    if _cached_D_key != key:
        _cached_D = inv_propagator()
        _cached_D_key = key
    return _cached_D


def compute_local_fund_Wilson_line(rng, D=None, A_arr=None):
    if rng is None:
        rng = cp.random.Generator(cp.random.XORWOW())

    N, Nc = cfg.N, cfg.Nc
    t = cfg.generators
    if D is None:
        D = _get_inv_propagator()

    if A_arr is None:
        A_arr = cp.zeros((N, N, Nc, Nc), dtype=cp.complex64)
    else:
        A_arr[:] = 0

    for b in range(Nc**2 - 1):
        rho = cfg.variance_of_mv_noise * rng.standard_normal((N, N), dtype=cp.float32)
        rhok = cp.fft.fft2(rho)
        rhok *= D
        rhok[0, 0] = 0.0
        A_field = cp.real(cp.fft.ifft2(rhok))
        A_arr += A_field[:, :, None, None] * t[b][None, None, :, :]

    V = matexp_su3(1j * A_arr)
    return V


def compute_path_ordered_fund_Wilson_line(rng=None, fast=True):
    if rng is None:
        rng = cp.random.Generator(cp.random.XORWOW())

    D = _get_inv_propagator()

    if fast:
        from .ic_fast import compute_path_ordered_Wilson_line
        D_zeromode = D.copy()
        D_zeromode[0, 0] = 0.0
        sigma = float(np.sqrt(cfg.mu2 / (cfg.Ny * cfg.a2)))
        return compute_path_ordered_Wilson_line(rng, D_zeromode, sigma, cfg.Ny)

    A_arr = cp.zeros((cfg.N, cfg.N, cfg.Nc, cfg.Nc), dtype=cp.complex64)
    V = compute_local_fund_Wilson_line(rng, D=D, A_arr=A_arr)
    for step in range(cfg.Ny - 1):
        tmp = compute_local_fund_Wilson_line(rng, D=D, A_arr=A_arr)
        V = V @ tmp
    return V
