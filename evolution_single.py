"""JIMWLK evolution on a 2D transverse lattice via GPU (CuPy + CUDA RawKernels).

Each Langevin step updates the Wilson line V(x) as:
    V(x) → exp(-ε ξ_L·K) · V(x) · exp(+ε ξ_R·K)
where ξ are SU(3) color noise fields, K is the Weizsäcker-Williams kernel,
and ε = √(α_s dY)/π is the step prefactor.

Left side uses R2C FFTs (noise is real → half-spectrum), right side uses
full C2C FFTs (ξ_R = V†ξV is complex). Both sides use fused convolution
RawKernels to reduce GPU memory traffic.
"""
import cupy as cp
import numpy as np
from . import config as cfg
from .su3_kernel import matexp_su3, reunitarize
from .matmul3x3_kernel import matmul_abc, matmul_adgba, matmul_adgba_dual
from .noise_kernel import generate_noise, generate_noise_batch, generate_noise_raw_batch
from .ic_fast import _assemble_algebra_kernel, _block as _ic_block
from .observables import measure_dipole, measure_xG_xh, measure_ww_tmd, Qs_of_S


# ── Weizsäcker-Williams kernel ──────────────────────────────────────────────

def _K_x(x, y):
    r2 = (cp.sin(x * cp.pi / cfg.N)**2 + cp.sin(y * cp.pi / cfg.N)**2) * (cfg.l / cp.pi)**2
    kx = cp.sin(x * 2.0 * cp.pi / cfg.N) * cfg.l / (2.0 * cp.pi)
    return kx / (r2 + 1e-16)


def precompute_ww_kernel():
    """Precompute K̃_i(k) = a · FFT[K_i(x)] with lattice spacing absorbed.

    Returns (2, N, N) complex64: index 0 = x-polarization, 1 = y (by transpose).
    The factor 'a' is absorbed here so convolution is just multiply-in-k-space.
    """
    x = cp.arange(cfg.N)
    y = cp.arange(cfg.N)
    xx, yy = cp.meshgrid(x, y, indexing='ij')
    K = _K_x(xx, yy)

    K_of_k = cp.zeros((2, cfg.N, cfg.N), dtype=cp.complex64)
    K_of_k[0] = (cfg.a * cp.fft.fft2(K)).astype(cp.complex64)
    K_of_k[1] = K_of_k[0].T

    return K_of_k


_cached_K = None
_cached_K_key = None


def _get_ww_kernel():
    global _cached_K, _cached_K_key
    key = (cfg.N, cfg.a, cfg.m2)
    if _cached_K_key != key:
        _cached_K = precompute_ww_kernel()
        _cached_K_key = key
    return _cached_K


# ── FFT helpers for arbitrary spatial axes ──────────────────────────────────

def _fft2_spatial(x, spatial_axes):
    perm = [i for i in range(x.ndim) if i not in spatial_axes] + list(spatial_axes)
    inv_perm = [0] * x.ndim
    for i, p in enumerate(perm):
        inv_perm[p] = i
    xt = cp.ascontiguousarray(x.transpose(perm))
    shape = xt.shape
    flat = xt.reshape(-1, shape[-2], shape[-1])
    out = cp.fft.fft2(flat, axes=(1, 2))
    return out.reshape(shape).transpose(inv_perm)


def _ifft2_spatial(x, spatial_axes):
    perm = [i for i in range(x.ndim) if i not in spatial_axes] + list(spatial_axes)
    inv_perm = [0] * x.ndim
    for i, p in enumerate(perm):
        inv_perm[p] = i
    xt = cp.ascontiguousarray(x.transpose(perm))
    shape = xt.shape
    flat = xt.reshape(-1, shape[-2], shape[-1])
    out = cp.fft.ifft2(flat, axes=(1, 2))
    return out.reshape(shape).transpose(inv_perm)


# ── Fused convolution CUDA RawKernel (right side) ──────────────────────────
# Replaces 3 separate CuPy elementwise ops (load K0, load K1, add) with
# a single kernel that keeps K0, K1 in registers and loops over elements.
# Reduces memory traffic from ~664 B/site (3 round-trips) to ~232 B/site.

_fused_conv_c2c_kernel = cp.RawKernel(r'''
extern "C" __global__
void fused_conv_c2c(const float2* __restrict__ xi_k,
                    const float2* __restrict__ K,
                    float2* __restrict__ out,
                    int n_elem, int n_spatial) {
    // Same fusion as R2C but for full complex spectrum (right-side, 9 matrix elements).
    // xi_k: (2, 9, N*N) - VdagXiV in k-space, 2 polarizations x 9 matrix elements
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_spatial) return;

    float2 k0 = K[idx];
    float2 k1 = K[n_spatial + idx];

    for (int e = 0; e < n_elem; e++) {
        float2 x0 = xi_k[e * n_spatial + idx];
        float2 x1 = xi_k[(n_elem + e) * n_spatial + idx];
        float re = k0.x * x0.x - k0.y * x0.y + k1.x * x1.x - k1.y * x1.y;
        float im = k0.x * x0.y + k0.y * x0.x + k1.x * x1.y + k1.y * x1.x;
        out[e * n_spatial + idx] = make_float2(re, im);
    }
}
''', 'fused_conv_c2c')


# ── Legacy convolution (unfused, kept for profiling comparison) ─────────────

def convolution(K_of_k, xi_k):
    """K̃_x · ξ̃_x + K̃_y · ξ̃_y in k-space, then IFFT. Uses CuPy broadcast (3 GPU ops)."""
    result = (xi_k[0] * K_of_k[0, :, :, None, None]
            + xi_k[1] * K_of_k[1, :, :, None, None])
    return _ifft2_spatial(result, (0, 1))


def compute_exp(xi_conv, sign, pref):
    """exp(sign · pref · ξ·K) via Cayley-Hamilton SU(3) matexp on GPU."""
    return matexp_su3(xi_conv, scale=float(sign * pref))


# ── Left-side precomputation (R2C path) ─────────────────────────────────────

def _precompute_left_chunk(rng, K_of_k, K_r2c, pref, N, chunk_size):
    """Batch-precompute left exponentials and cache noise for right side.

    Pipeline: raw noise (8 real fields, color-major layout)
        → rfft2 (half-spectrum, ~2x less data than fft2)
        → convolve with K̃ in k-space (broadcast, both polarizations summed)
        → irfft2 → assemble su(3) algebra element → matexp → exp_L

    Also assembles the raw noise into 3×3 matrices (xi_all) for right-side
    reuse: the same ξ appears in both exp_L and ξ_R = V†ξV.
    """
    # (8, batch, 2, N, N) float32 — color-major for zero-copy reshape into assemble_algebra
    raw = generate_noise_raw_batch(rng, N, chunk_size)

    # R2C FFT: real input → half-spectrum (N, N//2+1), ~2x fewer complex values
    raw_k = cp.fft.rfft2(raw, axes=(-2, -1))
    W = N // 2 + 1

    # Convolution: K̃_x·ξ̃_x + K̃_y·ξ̃_y via CuPy broadcast (strided slicing, no copy needed)
    # Fused kernel not used here: polarization axis is interior (8,batch,2,N,W),
    # so extracting it for the kernel requires a transpose+copy that negates the fusion benefit.
    # With only 8 coefficients (vs 9 on right side), fusion amortizes less.
    conv_k = (raw_k[:, :, 0] * K_r2c[0]
            + raw_k[:, :, 1] * K_r2c[1])
    del raw_k

    conv = cp.fft.irfft2(conv_k, s=(N, N), axes=(-2, -1))
    del conv_k

    # Assemble 8 real coefficients → 3×3 su(3) matrix via Gell-Mann generators t^a
    n_sites = chunk_size * N * N
    conv_flat = cp.ascontiguousarray(conv.reshape(8, n_sites))
    A_mat = cp.empty((n_sites, 9), dtype=cp.complex64)
    grid = (n_sites + _ic_block - 1) // _ic_block
    _assemble_algebra_kernel((grid,), (_ic_block,), (conv_flat, A_mat, n_sites))
    del conv, conv_flat

    # exp(-ε · A) via Cayley-Hamilton matexp (one GPU kernel, one thread per site)
    exp_L_all = matexp_su3(A_mat.reshape(chunk_size, N, N, 3, 3), scale=float(-pref))
    del A_mat

    # Also assemble raw noise → 3×3 matrices for right-side V†ξV computation
    n_sites_all = chunk_size * 2 * N * N
    raw_flat = cp.ascontiguousarray(raw.reshape(8, n_sites_all))
    xi_mat = cp.empty((n_sites_all, 9), dtype=cp.complex64)
    grid = (n_sites_all + _ic_block - 1) // _ic_block
    _assemble_algebra_kernel((grid,), (_ic_block,), (raw_flat, xi_mat, n_sites_all))
    del raw, raw_flat
    xi_all = xi_mat.reshape(chunk_size, 2, N, N, 3, 3)

    return xi_all, exp_L_all


# ── Right-side fused convolution ─────────────────────────────────────────────

def _fused_conv_right(K_of_k, xi_R_k):
    """Fused K̃·ξ̃_R convolution for right side (full C2C spectrum, 9 matrix elements)."""
    N = xi_R_k.shape[2]
    n_spatial = N * N
    # Plane-major layout (2, 9, N*N): contiguous for coalesced GPU memory access
    xi_flat = cp.ascontiguousarray(xi_R_k.reshape(2, 9, n_spatial))
    K_flat = cp.ascontiguousarray(K_of_k.reshape(2, n_spatial))
    out = cp.empty((9, n_spatial), dtype=cp.complex64)
    block = 256
    grid = (n_spatial + block - 1) // block
    _fused_conv_c2c_kernel((grid,), (block,), (xi_flat, K_flat, out, 9, n_spatial))
    return out.reshape(9, N, N)


# ── Main evolution loop ─────────────────────────────────────────────────────

def jimwlk_evolution(V, Y_f=None, dY=None, rng=None, callback=None,
                     measure_interval=1, save_V_at=None, K_of_k=None,
                     derivative='forward', measure_ww=True):
    """Evolve Wilson line V(x) from Y=0 to Y=Y_f in steps of dY.

    Left exponentials are batch-precomputed (chunk_size steps at once) using
    R2C FFTs. Right exponentials are computed per-step (depend on current V).
    """
    if Y_f is None:
        Y_f = cfg.Y_f
    if dY is None:
        dY = cfg.dY
    if rng is None:
        rng = cp.random.Generator(cp.random.XORWOW(seed=42))
    if K_of_k is None:
        K_of_k = _get_ww_kernel()
    # Half-spectrum slice of K for R2C convolution (left side)
    W = cfg.N // 2 + 1
    K_r2c = cp.ascontiguousarray(K_of_k[:, :, :W])

    N = cfg.N
    pref = np.sqrt(cfg.alpha_fc * dY) / np.pi
    chunk_size = 10

    xi_R = cp.zeros((2, N, N, 3, 3), dtype=cp.complex64)

    Y = 0.0
    step = 0

    Y_list = []
    Qs_list = []
    S_list = []
    xG_list = []
    xh_list = []
    xG_k_list = []
    xh_k_list = []
    ww_G_list = []
    ww_H_list = []
    V_snapshots = {}
    r_bins = None
    r_xG = None
    k_vals = None
    k_ww = None

    xi_cache = None
    exp_L_cache = None
    cache_offset = 0
    cache_len = 0

    while Y < Y_f + 0.5 * dY:
        Y_round = round(Y, 6)

        if step % measure_interval == 0:
            r_bins, Sb = measure_dipole(V)
            Qs = Qs_of_S(r_bins, Sb)
            r_xG, xG, xh, k_vals, xG_k, xh_k = measure_xG_xh(V, derivative=derivative)
            Y_list.append(Y_round)
            Qs_list.append(Qs)
            S_list.append(Sb.copy())
            xG_list.append(xG.copy())
            xh_list.append(xh.copy())
            xG_k_list.append(xG_k.copy())
            xh_k_list.append(xh_k.copy())
            if measure_ww:
                k_ww, ww_G, ww_H = measure_ww_tmd(V)
                ww_G_list.append(ww_G.copy())
                ww_H_list.append(ww_H.copy())
            if callback:
                callback(step, Y_round, Qs, r_bins, Sb)

        if save_V_at is not None:
            for Ys in save_V_at:
                if abs(Y_round - Ys) < 0.5 * dY and Ys not in V_snapshots:
                    V_snapshots[Ys] = V.get().copy()

        if cache_offset >= cache_len:
            remaining = max(1, int(round((Y_f - Y) / dY)) + 1)
            cache_len = min(chunk_size, remaining)
            xi_cache, exp_L_cache = _precompute_left_chunk(
                rng, K_of_k, K_r2c, pref, N, cache_len)
            cache_offset = 0

        xi = xi_cache[cache_offset]
        exp_L = exp_L_cache[cache_offset]
        cache_offset += 1

        # Right side: ξ_R = V†·ξ·V for both polarizations (fused dual kernel)
        matmul_adgba_dual(V, xi[0], xi[1], out0=xi_R[0], out1=xi_R[1])
        # Reshape to plane-major (2, 9, N, N) for batch FFT, then fused convolution
        xi_R_flat = cp.ascontiguousarray(xi_R.transpose(0, 3, 4, 1, 2).reshape(2, 9, N, N))
        xi_R_k = cp.fft.fft2(xi_R_flat, axes=(-2, -1))

        conv_R_k = _fused_conv_right(K_of_k, xi_R_k)
        conv_R = cp.fft.ifft2(conv_R_k, axes=(-2, -1))
        # Transpose from (9, N, N) → (N, N, 9) → (N, N, 3, 3) site-major for matexp
        xi_conv_R = cp.ascontiguousarray(conv_R.transpose(1, 2, 0)).reshape(N, N, 3, 3)
        exp_R = compute_exp(xi_conv_R, 1.0, pref)

        # V(x) ← exp_L · V · exp_R  (single fused 3-matrix product kernel)
        matmul_abc(exp_L, V, exp_R, out=V)

        # Periodic reunitarization to compensate floating-point drift from SU(3)
        if step % 5 == 4:
            V = reunitarize(V)

        Y += dY
        step += 1

    result = {
        'Y': np.array(Y_list),
        'Qs': np.array(Qs_list),
        'S': np.array(S_list),
        'r': r_bins,
        'r_xG': r_xG,
        'xG': np.array(xG_list),
        'xh': np.array(xh_list),
        'k_vals': k_vals,
        'xG_k': np.array(xG_k_list),
        'xh_k': np.array(xh_k_list),
    }
    if measure_ww:
        result['k_ww'] = k_ww
        result['ww_G'] = np.array(ww_G_list)
        result['ww_H'] = np.array(ww_H_list)
    if save_V_at is not None:
        result['V_snapshots'] = V_snapshots
    return result
