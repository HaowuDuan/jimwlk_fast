import math
import cupy as cp
from . import config as cfg
from .su3_kernel import matexp_su3

_t8_flat = cfg.generators[:8].reshape(8, 9)

# fitted parameters matching NN dipole (x=0.01)
mu2 = 0.4974
Q = 0.3170
beta = 0.6749
gamma = 1.6805


def _build_propagator(Q_val, beta_val, gamma_val):
    N, a = cfg.N, cfg.a
    ki = (2.0 / a) * cp.sin(cp.pi * cp.arange(N, dtype=cp.float64) / N)
    k2 = ki[:, None]**2 + ki[None, :]**2
    ln_v = cp.log(cp.maximum(cp.sqrt(k2) / Q_val, 1e-30))
    log_sqrt_f = beta_val * ln_v - 0.5 * cp.logaddexp(
        0, 2 * (beta_val - 1 + gamma_val) * ln_v)
    G = cp.zeros((N, N), dtype=cp.float64)
    mask = k2 > 0
    G[mask] = cp.exp(log_sqrt_f[mask]) / k2[mask]
    return G.astype(cp.float32)


_cached_G = None
_cached_G_key = None


def _get_propagator():
    global _cached_G, _cached_G_key
    key = (cfg.N, cfg.a, Q, beta, gamma)
    if _cached_G_key != key:
        _cached_G = _build_propagator(Q, beta, gamma)
        _cached_G_key = key
    return _cached_G


def set_params(mu2_new, Q_new, beta_new, gamma_new):
    global mu2, Q, beta, gamma, _cached_G_key
    mu2, Q, beta, gamma = mu2_new, Q_new, beta_new, gamma_new
    _cached_G_key = None


def compute_local_fund_Wilson_line(rng, G=None, A_arr=None):
    if rng is None:
        rng = cp.random.Generator(cp.random.XORWOW())

    N, Nc = cfg.N, cfg.Nc
    if G is None:
        G = _get_propagator()

    sigma = math.sqrt(mu2 / (cfg.Ny * cfg.a2))

    if A_arr is None:
        A_arr = cp.zeros((N, N, Nc, Nc), dtype=cp.complex64)
    else:
        A_arr[:] = 0

    for b in range(Nc**2 - 1):
        rho = sigma * rng.standard_normal((N, N), dtype=cp.float32)
        rhok = cp.fft.fft2(rho)
        rhok *= G
        A_field = cp.real(cp.fft.ifft2(rhok))
        A_arr += A_field[:, :, None, None] * cfg.generators[b][None, None, :, :]

    return matexp_su3(1j * A_arr)


def compute_path_ordered_fund_Wilson_line(rng=None, fast=True):
    if rng is None:
        rng = cp.random.Generator(cp.random.XORWOW())

    G = _get_propagator()

    if fast:
        from .ic_fast import compute_path_ordered_Wilson_line
        sigma = math.sqrt(mu2 / (cfg.Ny * cfg.a2))
        return compute_path_ordered_Wilson_line(rng, G, sigma, cfg.Ny)

    A_arr = cp.zeros((cfg.N, cfg.N, cfg.Nc, cfg.Nc), dtype=cp.complex64)
    V = compute_local_fund_Wilson_line(rng, G=G, A_arr=A_arr)
    for step in range(cfg.Ny - 1):
        tmp = compute_local_fund_Wilson_line(rng, G=G, A_arr=A_arr)
        V = V @ tmp
    return V
