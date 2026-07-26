"""Observable measurements from Wilson lines on GPU.

Dipole amplitude S(r): Fourier correlator of V projected onto Gell-Mann generators.
  Pipeline: V → tr(V·t^a) → FFT → |Ṽ^a(k)|² → IFFT → radial binning → S(r)

Gluon TMDs xG(r), xh(r): gauge field correlators from lattice derivatives of V.
  A^a_i(x) = (2/ig) tr(V†∂_iV · t^a), then power spectrum and Fourier transform.
  Supports forward, centered, and spectral (exact Fourier) derivatives.

GPU operations: all FFTs and linear algebra run on GPU via CuPy; radial binning
and Qs extraction run on CPU (small arrays after Fourier compression).
"""
import cupy as cp
import numpy as np
from . import config as cfg


def compute_field_of_V_components(V):
    t = cfg.generators
    Vc = 2.0 * cp.einsum('ijac,bca->bij', V, t)
    return Vc.astype(cp.complex64)


def FFT_Wilson_components(Vc):
    Vk = cfg.a**2 * cp.fft.fft2(Vc, axes=(-2, -1))
    return Vk.astype(cp.complex64)


def dipole(Vk):
    Sk = cp.sum(cp.abs(Vk)**2, axis=0) * 0.5 / cfg.Nc
    return cp.real(cp.fft.ifft2(Sk)) / (cfg.a2 * cfg.l**2)


def bin_x(S):
    S_np = S.get() if hasattr(S, 'get') else S
    N2 = cfg.N // 2
    step = 2.0 * cfg.a
    Nbins = N2

    i = np.arange(N2)
    j = np.arange(N2)
    ii, jj = np.meshgrid(i, j, indexing='ij')
    r = np.sqrt((ii * cfg.a)**2 + (jj * cfg.a)**2)
    idx_r = np.floor(r / step).astype(int)

    Sb = np.zeros(Nbins, dtype=np.float64)
    Nb = np.zeros(Nbins, dtype=np.float64)

    mask = idx_r < Nbins
    S_quad = np.real(S_np[:N2, :N2])

    np.add.at(Sb, idx_r[mask], S_quad[mask])
    np.add.at(Nb, idx_r[mask], 1.0)

    r_bins = (np.arange(1, Nbins + 1)) * step
    return r_bins, (Sb / (1e-16 + Nb)).astype(np.float32)


def Qs_of_S(r, Sb):
    Ssat = np.exp(-0.5)
    j = -1
    for i in range(1, len(Sb)):
        if (Sb[i - 1] - Ssat) * (Sb[i] - Ssat) < 0:
            j = i
            break
    if j < 0:
        if Sb[0] < Ssat:
            return np.sqrt(2.0) / r[0]
        return np.sqrt(2.0) / r[-1]
    dS = Sb[j - 1] - Sb[j]
    if abs(dS) < 1e-12:
        return np.sqrt(2.0) / r[j]
    r_cross = r[j] + (r[j - 1] - r[j]) / dS * (Ssat - Sb[j])
    if r_cross <= 0:
        return np.sqrt(2.0) / r[0]
    return np.sqrt(2.0) / r_cross


def measure_xG_xh(V, derivative='forward'):
    N = cfg.N
    a = cfg.a

    if derivative == 'forward':
        from .observables_fast import extract_gauge_field_fast, compute_power_spectra_fast
        Aa_x, Aa_y = extract_gauge_field_fast(V)
        Ak_x = cp.fft.fft2(Aa_x, axes=(1, 2))
        del Aa_x
        Ak_y = cp.fft.fft2(Aa_y, axes=(1, 2))
        del Aa_y
        Pxx, Pyy, _ = compute_power_spectra_fast(Ak_x, Ak_y)
        del Ak_x, Ak_y
        Px_sum = Pxx
        Py_sum = Pyy
    else:
        t = cfg.generators[:8]
        if derivative == 'centered':
            dxV = (cp.roll(V, -1, axis=0) - cp.roll(V, 1, axis=0)) / (2.0 * a)
            dyV = (cp.roll(V, -1, axis=1) - cp.roll(V, 1, axis=1)) / (2.0 * a)
        elif derivative == 'spectral':
            Vk = cp.fft.fft2(V, axes=(0, 1))
            kx = (2.0 * cp.pi * cp.fft.fftfreq(N, d=a)).astype(cp.float32).reshape(N, 1, 1, 1)
            ky = (2.0 * cp.pi * cp.fft.fftfreq(N, d=a)).astype(cp.float32).reshape(1, N, 1, 1)
            Vk = Vk.astype(cp.complex64)
            dxV = cp.fft.ifft2(cp.complex64(1j) * kx * Vk, axes=(0, 1))
            dyV = cp.fft.ifft2(cp.complex64(1j) * ky * Vk, axes=(0, 1))
            del Vk
        else:
            raise ValueError(f"Unknown derivative method: {derivative!r}")
        Vd = V.conj().transpose(0, 1, 3, 2)
        igAx = Vd @ dxV
        del dxV
        igAy = Vd @ dyV
        del dyV, Vd
        Aa_x = 2.0 * cp.einsum('ijcd,adc->ija', igAx, t).imag
        del igAx
        Aa_y = 2.0 * cp.einsum('ijcd,adc->ija', igAy, t).imag
        del igAy
        Ak_x = cp.fft.fft2(Aa_x, axes=(0, 1))
        del Aa_x
        Ak_y = cp.fft.fft2(Aa_y, axes=(0, 1))
        del Aa_y
        pow_x = cp.abs(Ak_x)**2
        del Ak_x
        pow_y = cp.abs(Ak_y)**2
        del Ak_y
        Px_sum = pow_x.sum(axis=2)
        del pow_x
        Py_sum = pow_y.sum(axis=2)
        del pow_y

    Cx_2d = cp.fft.ifft2(Px_sum) / (N * N)
    Cy_2d = cp.fft.ifft2(Py_sum) / (N * N)

    N2 = N // 2
    Cx = cp.asnumpy(cp.real(Cx_2d[:N2, 0])).astype(np.float64)
    Cy = cp.asnumpy(cp.real(Cy_2d[:N2, 0])).astype(np.float64)

    r_vals = np.arange(N2) * a
    prefactor_r = 2.0 / np.pi
    xG = prefactor_r * (Cx + Cy)
    xh = prefactor_r * (Cx - Cy)

    Pk_x = cp.asnumpy(cp.real(Px_sum[:N2, 0])).astype(np.float64)
    Pk_y = cp.asnumpy(cp.real(Py_sum[:N2, 0])).astype(np.float64)
    k_vals = 2.0 * np.pi * np.arange(N2) / (N * a)
    prefactor_k = 8.0 * np.pi * a**2 / ((2.0 * np.pi)**4 * N * N)
    xG_k = prefactor_k * (Pk_x + Pk_y)
    xh_k = prefactor_k * (Pk_x - Pk_y)

    return r_vals, xG, xh, k_vals, xG_k, xh_k


def measure_dipole(V):
    Vc = compute_field_of_V_components(V)
    Vk = FFT_Wilson_components(Vc)
    S = dipole(Vk)
    return bin_x(S)


def measure_observables(V):
    r, Sb = measure_dipole(V)
    Qs = Qs_of_S(r, Sb)
    return Qs, r, Sb


def measure_ww_tmd(V):
    """Measure the WW gluon TMD with fused adjoint CUDA kernels."""
    from .observables_fast import measure_ww_tmd_fast

    return measure_ww_tmd_fast(V)


def measure_xG_xh_nufft(V, k_points=None, n_k=100, derivative='forward'):
    """Compute xG(k), xh(k) at non-uniform k-points using NUFFT.

    Same gauge field extraction as measure_xG_xh, but evaluates the Fourier
    transform at arbitrary (e.g. log-spaced) k-points along the k_x axis
    (k_y = 0) via cufinufft Type 3.

    Since k_y = 0, the 2D DFT reduces to a 1D problem:
        A_tilde(k_x, 0) = sum_{j1} [ sum_{j2} A(x_{j1}, y_{j2}) ] exp(-i k_x x_{j1})
    We sum over y first, then 1D NUFFT at non-uniform k_x.

    Parameters
    ----------
    V : cupy array (N, N, 3, 3)
    k_points : array-like or None
        Non-uniform k values. If None, log-spaced from k_min to k_max.
    n_k : int
        Number of log-spaced k-points if k_points is None.
    derivative : str
        'forward' (default) uses fast CUDA kernel.

    Returns
    -------
    k_points, xG_k, xh_k : numpy arrays
    """
    import cufinufft

    N = cfg.N
    a = cfg.a

    if k_points is None:
        k_min = 2.0 * np.pi / (N * a)
        k_max = np.pi / a
        k_points = np.logspace(np.log10(k_min), np.log10(k_max), n_k)

    k_points = np.asarray(k_points, dtype=np.float64)

    if derivative == 'forward':
        from .observables_fast import extract_gauge_field_fast
        Aa_x, Aa_y = extract_gauge_field_fast(V)
    else:
        t = cfg.generators[:8]
        if derivative == 'centered':
            dxV = (cp.roll(V, -1, axis=0) - cp.roll(V, 1, axis=0)) / (2.0 * a)
            dyV = (cp.roll(V, -1, axis=1) - cp.roll(V, 1, axis=1)) / (2.0 * a)
        elif derivative == 'spectral':
            Vk = cp.fft.fft2(V, axes=(0, 1))
            kx = (2.0 * cp.pi * cp.fft.fftfreq(N, d=a)).astype(cp.float32).reshape(N, 1, 1, 1)
            ky = (2.0 * cp.pi * cp.fft.fftfreq(N, d=a)).astype(cp.float32).reshape(1, N, 1, 1)
            Vk = Vk.astype(cp.complex64)
            dxV = cp.fft.ifft2(cp.complex64(1j) * kx * Vk, axes=(0, 1))
            dyV = cp.fft.ifft2(cp.complex64(1j) * ky * Vk, axes=(0, 1))
            del Vk
        else:
            raise ValueError(f"Unknown derivative method: {derivative!r}")
        Vd = V.conj().transpose(0, 1, 3, 2)
        igAx = Vd @ dxV
        del dxV
        igAy = Vd @ dyV
        del dyV, Vd
        Aa_x = 2.0 * cp.einsum('ijcd,adc->aij', igAx, t).imag.astype(cp.float32)
        del igAx
        Aa_y = 2.0 * cp.einsum('ijcd,adc->aij', igAy, t).imag.astype(cp.float32)
        del igAy

    # Aa_x, Aa_y: (8, N, N) float32
    # Sum over y (axis 2) since k_y = 0:  f_a(x_j) = sum_{j2} A^a(x_{j1}, y_{j2})
    # Then 1D NUFFT Type 3: uniform x → non-uniform k
    fx_summed = Aa_x.sum(axis=2)  # (8, N) float32
    fy_summed = Aa_y.sum(axis=2)  # (8, N) float32
    del Aa_x, Aa_y

    # Source locations: uniform x_j = j * a, scaled to cufinufft convention
    # cufinufft Type 3 expects source coords in arbitrary units;
    # transform is: F(s) = sum_j c_j exp(i * s * x_j)
    # Our DFT convention: F(k) = sum_j f(x_j) exp(-i k x_j)
    # So pass s = -k (or equivalently, conjugate the output)
    x_src = cp.arange(N, dtype=cp.float64) * a
    s_target = cp.asarray(-k_points)  # negative k for exp(-ikx) convention

    # Batch all 8 color components × 2 polarizations into one NUFFT plan
    # Stack (8, N) x-components and y-components into (16, N)
    all_summed = cp.concatenate([fx_summed, fy_summed], axis=0).astype(cp.complex128)  # (16, N)
    del fx_summed, fy_summed

    M = len(k_points)
    plan = cufinufft.Plan(3, 1, n_trans=16, dtype=np.complex128)
    plan.setpts(x_src, s=s_target)
    Fk_all = plan.execute(all_summed)  # (16, M)

    Pk_all = cp.asnumpy(cp.abs(Fk_all) ** 2)  # (16, M)
    Pk_x = Pk_all[:8].sum(axis=0)
    Pk_y = Pk_all[8:].sum(axis=0)

    prefactor_k = 8.0 * np.pi * a**2 / ((2.0 * np.pi)**4 * N * N)
    xG_k = prefactor_k * (Pk_x + Pk_y)
    xh_k = prefactor_k * (Pk_x - Pk_y)

    return k_points, xG_k, xh_k
