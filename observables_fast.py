"""Fast observable kernels for fundamental/adjoint fields and power spectra.

Fused CUDA kernels that avoid intermediate allocations.
"""

import cupy as cp
import numpy as np
from . import config as cfg

# Kernel: extract A^a_x and A^a_y from Wilson line V
# For each site (i,j):
#   dxV = (V(i+1,j) - V(i,j)) / a
#   dyV = (V(i,j+1) - V(i,j)) / a
#   igAx = V^dag @ dxV
#   igAy = V^dag @ dyV
#   A^a_x = 2 * Im(tr(igAx * t^a))  for a=0..7
#   A^a_y = 2 * Im(tr(igAy * t^a))
# Output: Aa_x(a, i, j), Aa_y(a, i, j) as float32
_extract_gauge_kernel = cp.RawKernel(r'''
extern "C" __global__
void extract_gauge_field(const float2* __restrict__ V,
                         float* __restrict__ Ax,
                         float* __restrict__ Ay,
                         int N, float inv_a) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N * N) return;

    int i = idx / N;
    int j = idx % N;
    int ip = ((i + 1) % N) * N + j;
    int jp = i * N + ((j + 1) % N);

    // Load V(i,j), V(i+1,j), V(i,j+1) as 3x3 complex
    float vr[9], vi[9];
    float vxr[9], vxi[9];
    float vyr[9], vyi[9];
    for (int m = 0; m < 9; m++) {
        float2 v0 = V[idx * 9 + m];
        float2 vx = V[ip * 9 + m];
        float2 vy = V[jp * 9 + m];
        vr[m] = v0.x; vi[m] = v0.y;
        vxr[m] = vx.x; vxi[m] = vx.y;
        vyr[m] = vy.x; vyi[m] = vy.y;
    }

    // dxV = (V(i+1,j) - V(i,j)) * inv_a
    // dyV = (V(i,j+1) - V(i,j)) * inv_a
    float dxr[9], dxi[9], dyr[9], dyi[9];
    for (int m = 0; m < 9; m++) {
        dxr[m] = (vxr[m] - vr[m]) * inv_a;
        dxi[m] = (vxi[m] - vi[m]) * inv_a;
        dyr[m] = (vyr[m] - vr[m]) * inv_a;
        dyi[m] = (vyi[m] - vi[m]) * inv_a;
    }

    // V^dag: conjugate transpose
    float vdr[9], vdi[9];
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) {
            vdr[r*3+c] =  vr[c*3+r];
            vdi[r*3+c] = -vi[c*3+r];
        }

    // igAx = V^dag @ dxV, igAy = V^dag @ dyV
    float gxr[9], gxi[9], gyr[9], gyi[9];
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) {
            float sr_x = 0.f, si_x = 0.f;
            float sr_y = 0.f, si_y = 0.f;
            for (int k = 0; k < 3; k++) {
                int rk = r*3+k, kc = k*3+c;
                sr_x += vdr[rk]*dxr[kc] - vdi[rk]*dxi[kc];
                si_x += vdr[rk]*dxi[kc] + vdi[rk]*dxr[kc];
                sr_y += vdr[rk]*dyr[kc] - vdi[rk]*dyi[kc];
                si_y += vdr[rk]*dyi[kc] + vdi[rk]*dyr[kc];
            }
            gxr[r*3+c] = sr_x; gxi[r*3+c] = si_x;
            gyr[r*3+c] = sr_y; gyi[r*3+c] = si_y;
        }

    // A^a = 2 * Im(tr(igA * t^a))
    // tr(M * t^a) for Gell-Mann t^a = lambda^a/2:
    // a=0 (t1): M[0,1]/2 + M[1,0]/2            -> Re part
    // a=1 (t2): -Im(M[0,1])/2 + Im(M[1,0])/2   -> i*(M[1,0]-M[0,1])/2
    // a=2 (t3): (M[0,0] - M[1,1])/2
    // a=3 (t4): (M[0,2] + M[2,0])/2
    // a=4 (t5): i*(M[2,0] - M[0,2])/2
    // a=5 (t6): (M[1,2] + M[2,1])/2
    // a=6 (t7): i*(M[2,1] - M[1,2])/2
    // a=7 (t8): (M[0,0] + M[1,1])/(2*sqrt3) - M[2,2]/sqrt3
    //
    // tr(M * t^a) is complex. A^a = 2 * Im(tr(igA * t^a))
    // For M = igA (complex matrix with real part gxr, imag part gxi):
    // tr(M * t^a) = sum over matrix elements
    // We need Im of that trace.

    float inv2sqrt3 = 0.28867513459f;
    float inv_sqrt3 = 0.57735026919f;
    int n = N * N;

    // For igAx:
    // t1: tr = (M01 + M10)/2, where Mij = gxr[i*3+j] + I*gxi[i*3+j]
    // Re(tr) = (gxr[1] + gxr[3])/2, Im(tr) = (gxi[1] + gxi[3])/2
    // A^0_x = 2 * Im(tr) = gxi[1] + gxi[3]
    Ax[0*n + idx] = gxi[0*3+1] + gxi[1*3+0];
    Ay[0*n + idx] = gyi[0*3+1] + gyi[1*3+0];

    // t2: tr = i*(M10 - M01)/2
    // = i*((gxr[3]-gxr[1]) + i*(gxi[3]-gxi[1]))/2
    // = (-(gxi[3]-gxi[1]) + i*(gxr[3]-gxr[1]))/2
    // Im(tr) = (gxr[3] - gxr[1])/2
    // A^1 = 2*Im = gxr[3] - gxr[1]
    Ax[1*n + idx] = gxr[1*3+0] - gxr[0*3+1];
    Ay[1*n + idx] = gyr[1*3+0] - gyr[0*3+1];

    // t3: tr = (M00 - M11)/2
    // Im(tr) = (gxi[0] - gxi[4])/2
    // A^2 = gxi[0] - gxi[4]
    Ax[2*n + idx] = gxi[0*3+0] - gxi[1*3+1];
    Ay[2*n + idx] = gyi[0*3+0] - gyi[1*3+1];

    // t4: tr = (M02 + M20)/2
    // Im(tr) = (gxi[2] + gxi[6])/2
    // A^3 = gxi[2] + gxi[6]
    Ax[3*n + idx] = gxi[0*3+2] + gxi[2*3+0];
    Ay[3*n + idx] = gyi[0*3+2] + gyi[2*3+0];

    // t5: tr = i*(M20 - M02)/2
    // Im(tr) = (gxr[6] - gxr[2])/2
    // A^4 = gxr[6] - gxr[2]
    Ax[4*n + idx] = gxr[2*3+0] - gxr[0*3+2];
    Ay[4*n + idx] = gyr[2*3+0] - gyr[0*3+2];

    // t6: tr = (M12 + M21)/2
    // Im(tr) = (gxi[5] + gxi[7])/2
    // A^5 = gxi[5] + gxi[7]
    Ax[5*n + idx] = gxi[1*3+2] + gxi[2*3+1];
    Ay[5*n + idx] = gyi[1*3+2] + gyi[2*3+1];

    // t7: tr = i*(M21 - M12)/2
    // Im(tr) = (gxr[7] - gxr[5])/2
    // A^6 = gxr[7] - gxr[5]
    Ax[6*n + idx] = gxr[2*3+1] - gxr[1*3+2];
    Ay[6*n + idx] = gyr[2*3+1] - gyr[1*3+2];

    // t8: tr = M00/(2*sqrt3) + M11/(2*sqrt3) - M22/sqrt3
    // Im(tr) = gxi[0]/(2*sqrt3) + gxi[4]/(2*sqrt3) - gxi[8]/sqrt3
    // A^7 = 2*Im(tr)
    Ax[7*n + idx] = 2.f * (inv2sqrt3*gxi[0*3+0] + inv2sqrt3*gxi[1*3+1] - inv_sqrt3*gxi[2*3+2]);
    Ay[7*n + idx] = 2.f * (inv2sqrt3*gyi[0*3+0] + inv2sqrt3*gyi[1*3+1] - inv_sqrt3*gyi[2*3+2]);
}
''', 'extract_gauge_field')


# Kernel: build the adjoint Wilson line
#   U^{ab} = 2 Re tr(t^a V t^b V^dag)
# from the fundamental 3x3 Wilson line.  V is site-major complex64 and U is
# site-major float32.  V t^b V^dag is Hermitian, so only its three diagonal
# and three upper-triangular entries are needed for all eight projections.
_extract_adjoint_wilson_line_kernel = cp.RawKernel(r'''
__device__ __forceinline__
float2 outer_element(const float2* __restrict__ v,
                     int row, int col, int p, int q) {
    // v[row,p] * conj(v[col,q])
    float2 x = v[row * 3 + p];
    float2 y = v[col * 3 + q];
    return make_float2(x.x * y.x + x.y * y.y,
                       x.y * y.x - x.x * y.y);
}

__device__ __forceinline__
float2 adjoint_matrix_element(const float2* __restrict__ v,
                              int b, int row, int col) {
    float2 x, y, z;

    if (b == 0) {
        // (|0><1| + |1><0|) / 2
        x = outer_element(v, row, col, 0, 1);
        y = outer_element(v, row, col, 1, 0);
        return make_float2(0.5f * (x.x + y.x), 0.5f * (x.y + y.y));
    }
    if (b == 1) {
        // (-i |0><1| + i |1><0|) / 2
        x = outer_element(v, row, col, 0, 1);
        y = outer_element(v, row, col, 1, 0);
        return make_float2(0.5f * (x.y - y.y),
                           0.5f * (-x.x + y.x));
    }
    if (b == 2) {
        x = outer_element(v, row, col, 0, 0);
        y = outer_element(v, row, col, 1, 1);
        return make_float2(0.5f * (x.x - y.x), 0.5f * (x.y - y.y));
    }
    if (b == 3) {
        // (|0><2| + |2><0|) / 2
        x = outer_element(v, row, col, 0, 2);
        y = outer_element(v, row, col, 2, 0);
        return make_float2(0.5f * (x.x + y.x), 0.5f * (x.y + y.y));
    }
    if (b == 4) {
        // (-i |0><2| + i |2><0|) / 2
        x = outer_element(v, row, col, 0, 2);
        y = outer_element(v, row, col, 2, 0);
        return make_float2(0.5f * (x.y - y.y),
                           0.5f * (-x.x + y.x));
    }
    if (b == 5) {
        // (|1><2| + |2><1|) / 2
        x = outer_element(v, row, col, 1, 2);
        y = outer_element(v, row, col, 2, 1);
        return make_float2(0.5f * (x.x + y.x), 0.5f * (x.y + y.y));
    }
    if (b == 6) {
        // (-i |1><2| + i |2><1|) / 2
        x = outer_element(v, row, col, 1, 2);
        y = outer_element(v, row, col, 2, 1);
        return make_float2(0.5f * (x.y - y.y),
                           0.5f * (-x.x + y.x));
    }

    // diag(1, 1, -2) / (2 sqrt(3))
    x = outer_element(v, row, col, 0, 0);
    y = outer_element(v, row, col, 1, 1);
    z = outer_element(v, row, col, 2, 2);
    const float inv_2sqrt3 = 0.2886751345948129f;
    return make_float2(inv_2sqrt3 * (x.x + y.x - 2.0f * z.x),
                       inv_2sqrt3 * (x.y + y.y - 2.0f * z.y));
}

extern "C" __global__
void extract_adjoint_wilson_line(const float2* __restrict__ V,
                                 float* __restrict__ U,
                                 int n_sites) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_sites) return;

    const float2* v = V + idx * 9;
    float* u = U + idx * 64;
    const float inv_sqrt3 = 0.5773502691896258f;

    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        float2 m00 = adjoint_matrix_element(v, b, 0, 0);
        float2 m11 = adjoint_matrix_element(v, b, 1, 1);
        float2 m22 = adjoint_matrix_element(v, b, 2, 2);
        float2 m01 = adjoint_matrix_element(v, b, 0, 1);
        float2 m02 = adjoint_matrix_element(v, b, 0, 2);
        float2 m12 = adjoint_matrix_element(v, b, 1, 2);

        // 2 tr(t^a M), with M = V t^b V^dag.
        u[0 * 8 + b] = 2.0f * m01.x;
        u[1 * 8 + b] = -2.0f * m01.y;
        u[2 * 8 + b] = m00.x - m11.x;
        u[3 * 8 + b] = 2.0f * m02.x;
        u[4 * 8 + b] = -2.0f * m02.y;
        u[5 * 8 + b] = 2.0f * m12.x;
        u[6 * 8 + b] = -2.0f * m12.y;
        u[7 * 8 + b] =
            inv_sqrt3 * (m00.x + m11.x - 2.0f * m22.x);
    }
}
''', 'extract_adjoint_wilson_line')


# Kernel: fuse U^T dU and contraction with the standard SU(3) f^{cab}.
# For p < q, B_pq = A_qp - A_pq.  The sparse standard structure constants
# reduce the contraction sum_ab f^{cab} A_ba to 27 B_pq evaluations instead
# of materializing either A matrix.
_extract_ww_field_kernel = cp.RawKernel(r'''
__device__ __forceinline__
float2 antisymmetric_product(const float* __restrict__ U,
                             const float* __restrict__ dU_x,
                             const float* __restrict__ dU_y,
                             int p, int q) {
    // A_qp - A_pq, where A = U^T dU.
    float bx = 0.0f;
    float by = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        float ukp = U[k * 8 + p];
        float ukq = U[k * 8 + q];
        bx += ukq * dU_x[k * 8 + p] - ukp * dU_x[k * 8 + q];
        by += ukq * dU_y[k * 8 + p] - ukp * dU_y[k * 8 + q];
    }
    return make_float2(bx, by);
}

extern "C" __global__
void extract_ww_field(const float* __restrict__ U,
                      const float* __restrict__ dU_x,
                      const float* __restrict__ dU_y,
                      float* __restrict__ alpha_x,
                      float* __restrict__ alpha_y,
                      int n_sites) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_sites) return;

    const float* u = U + idx * 64;
    const float* dx = dU_x + idx * 64;
    const float* dy = dU_y + idx * 64;
    const float half = 0.5f;
    const float sqrt3_over_2 = 0.8660254037844386f;
    const float minus_one_third = -0.3333333333333333f;
    float2 b0, b1, b2, b3;
    float sx, sy;

    // f^{0ab} A^{ba} = B_12 + B_36/2 - B_45/2
    b0 = antisymmetric_product(u, dx, dy, 1, 2);
    b1 = antisymmetric_product(u, dx, dy, 3, 6);
    b2 = antisymmetric_product(u, dx, dy, 4, 5);
    sx = b0.x + half * (b1.x - b2.x);
    sy = b0.y + half * (b1.y - b2.y);
    alpha_x[0 * n_sites + idx] = minus_one_third * sx;
    alpha_y[0 * n_sites + idx] = minus_one_third * sy;

    // f^{1ab} A^{ba} = -B_02 + B_35/2 + B_46/2
    b0 = antisymmetric_product(u, dx, dy, 0, 2);
    b1 = antisymmetric_product(u, dx, dy, 3, 5);
    b2 = antisymmetric_product(u, dx, dy, 4, 6);
    sx = -b0.x + half * (b1.x + b2.x);
    sy = -b0.y + half * (b1.y + b2.y);
    alpha_x[1 * n_sites + idx] = minus_one_third * sx;
    alpha_y[1 * n_sites + idx] = minus_one_third * sy;

    // f^{2ab} A^{ba} = B_01 + B_34/2 - B_56/2
    b0 = antisymmetric_product(u, dx, dy, 0, 1);
    b1 = antisymmetric_product(u, dx, dy, 3, 4);
    b2 = antisymmetric_product(u, dx, dy, 5, 6);
    sx = b0.x + half * (b1.x - b2.x);
    sy = b0.y + half * (b1.y - b2.y);
    alpha_x[2 * n_sites + idx] = minus_one_third * sx;
    alpha_y[2 * n_sites + idx] = minus_one_third * sy;

    // f^{3ab} A^{ba} = -(B_06 + B_15 + B_24)/2
    //                       + sqrt(3) B_47/2
    b0 = antisymmetric_product(u, dx, dy, 0, 6);
    b1 = antisymmetric_product(u, dx, dy, 1, 5);
    b2 = antisymmetric_product(u, dx, dy, 2, 4);
    b3 = antisymmetric_product(u, dx, dy, 4, 7);
    sx = -half * (b0.x + b1.x + b2.x) + sqrt3_over_2 * b3.x;
    sy = -half * (b0.y + b1.y + b2.y) + sqrt3_over_2 * b3.y;
    alpha_x[3 * n_sites + idx] = minus_one_third * sx;
    alpha_y[3 * n_sites + idx] = minus_one_third * sy;

    // f^{4ab} A^{ba} = (B_05 - B_16 + B_23)/2
    //                       - sqrt(3) B_37/2
    b0 = antisymmetric_product(u, dx, dy, 0, 5);
    b1 = antisymmetric_product(u, dx, dy, 1, 6);
    b2 = antisymmetric_product(u, dx, dy, 2, 3);
    b3 = antisymmetric_product(u, dx, dy, 3, 7);
    sx = half * (b0.x - b1.x + b2.x) - sqrt3_over_2 * b3.x;
    sy = half * (b0.y - b1.y + b2.y) - sqrt3_over_2 * b3.y;
    alpha_x[4 * n_sites + idx] = minus_one_third * sx;
    alpha_y[4 * n_sites + idx] = minus_one_third * sy;

    // f^{5ab} A^{ba} = (-B_04 + B_13 + B_26)/2
    //                       + sqrt(3) B_67/2
    b0 = antisymmetric_product(u, dx, dy, 0, 4);
    b1 = antisymmetric_product(u, dx, dy, 1, 3);
    b2 = antisymmetric_product(u, dx, dy, 2, 6);
    b3 = antisymmetric_product(u, dx, dy, 6, 7);
    sx = half * (-b0.x + b1.x + b2.x) + sqrt3_over_2 * b3.x;
    sy = half * (-b0.y + b1.y + b2.y) + sqrt3_over_2 * b3.y;
    alpha_x[5 * n_sites + idx] = minus_one_third * sx;
    alpha_y[5 * n_sites + idx] = minus_one_third * sy;

    // f^{6ab} A^{ba} = (B_03 + B_14 - B_25)/2
    //                       - sqrt(3) B_57/2
    b0 = antisymmetric_product(u, dx, dy, 0, 3);
    b1 = antisymmetric_product(u, dx, dy, 1, 4);
    b2 = antisymmetric_product(u, dx, dy, 2, 5);
    b3 = antisymmetric_product(u, dx, dy, 5, 7);
    sx = half * (b0.x + b1.x - b2.x) - sqrt3_over_2 * b3.x;
    sy = half * (b0.y + b1.y - b2.y) - sqrt3_over_2 * b3.y;
    alpha_x[6 * n_sites + idx] = minus_one_third * sx;
    alpha_y[6 * n_sites + idx] = minus_one_third * sy;

    // f^{7ab} A^{ba} = sqrt(3) (B_34 + B_56)/2
    b0 = antisymmetric_product(u, dx, dy, 3, 4);
    b1 = antisymmetric_product(u, dx, dy, 5, 6);
    sx = sqrt3_over_2 * (b0.x + b1.x);
    sy = sqrt3_over_2 * (b0.y + b1.y);
    alpha_x[7 * n_sites + idx] = minus_one_third * sx;
    alpha_y[7 * n_sites + idx] = minus_one_third * sy;
}
''', 'extract_ww_field')


# Kernel: compute Pxx, Pyy, Pxy from Ak_x and Ak_y (already FFT'd)
# Sums |Ak_x|^2, |Ak_y|^2, Re(Ak_x * conj(Ak_y)) over 8 color components
_power_spectrum_kernel = cp.RawKernel(r'''
extern "C" __global__
void power_spectrum(const float2* __restrict__ Akx,
                    const float2* __restrict__ Aky,
                    float* __restrict__ Pxx,
                    float* __restrict__ Pyy,
                    float* __restrict__ Pxy,
                    int n_sites, int n_color) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_sites) return;

    float sxx = 0.f, syy = 0.f, sxy = 0.f;
    for (int a = 0; a < n_color; a++) {
        float2 ax = Akx[a * n_sites + idx];
        float2 ay = Aky[a * n_sites + idx];
        sxx += ax.x * ax.x + ax.y * ax.y;
        syy += ay.x * ay.x + ay.y * ay.y;
        sxy += ax.x * ay.x + ax.y * ay.y;
    }
    Pxx[idx] = sxx;
    Pyy[idx] = syy;
    Pxy[idx] = sxy;
}
''', 'power_spectrum')

_block = 256


def extract_gauge_field_fast(V):
    """Extract A^a_x, A^a_y from V using fused CUDA kernel.
    Returns Aa_x, Aa_y each of shape (8, N, N) as float32."""
    N = cfg.N
    n_sites = N * N
    inv_a = 1.0 / cfg.a

    V_flat = cp.ascontiguousarray(V.reshape(n_sites, 9))
    Ax = cp.empty((8, n_sites), dtype=cp.float32)
    Ay = cp.empty((8, n_sites), dtype=cp.float32)

    grid = (n_sites + _block - 1) // _block
    _extract_gauge_kernel((grid,), (_block,), (V_flat, Ax, Ay, N, cp.float32(inv_a)))

    return Ax.reshape(8, N, N), Ay.reshape(8, N, N)


def extract_adjoint_wilson_line_fast(V):
    """Build ``U^{ab} = 2 Re tr(t^a V t^b V†)`` with one CUDA kernel.

    Parameters
    ----------
    V : cupy.ndarray
        Fundamental Wilson lines with shape ``(N, N, 3, 3)`` and complex64
        elements.

    Returns
    -------
    cupy.ndarray
        Real adjoint Wilson lines with shape ``(N, N, 8, 8)`` and dtype
        float32.
    """
    N = cfg.N
    n_sites = N * N
    V_flat = cp.ascontiguousarray(V.reshape(n_sites, 9), dtype=cp.complex64)
    U = cp.empty((n_sites, 8, 8), dtype=cp.float32)

    grid = (n_sites + _block - 1) // _block
    _extract_adjoint_wilson_line_kernel(
        (grid,), (_block,), (V_flat, U, n_sites)
    )
    return U.reshape(N, N, 8, 8)


def extract_ww_field_fast(U, dU_x, dU_y):
    """Compute the two WW fields from adjoint Wilson lines and derivatives.

    This fuses ``A_i = U.T @ d_i U`` with
    ``alpha_i^c = -(1/3) f^{cab} A_i^{ba}``.  Inputs have shape
    ``(N, N, 8, 8)``; outputs are color-major ``(8, N, N)`` float32 arrays.
    """
    N = cfg.N
    n_sites = N * N
    U_flat = cp.ascontiguousarray(U.reshape(n_sites, 64), dtype=cp.float32)
    dUx_flat = cp.ascontiguousarray(
        dU_x.reshape(n_sites, 64), dtype=cp.float32
    )
    dUy_flat = cp.ascontiguousarray(
        dU_y.reshape(n_sites, 64), dtype=cp.float32
    )
    alpha_x = cp.empty((8, n_sites), dtype=cp.float32)
    alpha_y = cp.empty((8, n_sites), dtype=cp.float32)

    grid = (n_sites + _block - 1) // _block
    _extract_ww_field_kernel(
        (grid,),
        (_block,),
        (U_flat, dUx_flat, dUy_flat, alpha_x, alpha_y, n_sites),
    )
    return alpha_x.reshape(8, N, N), alpha_y.reshape(8, N, N)


def compute_power_spectra_fast(Ak_x, Ak_y):
    """Compute Pxx, Pyy, Pxy from FFT'd gauge fields using fused kernel.
    Ak_x, Ak_y: (8, N, N) complex64
    Returns Pxx, Pyy, Pxy each (N, N) float32."""
    N = cfg.N
    n_sites = N * N
    n_color = 8

    Akx_flat = cp.ascontiguousarray(
        Ak_x.reshape(n_color, n_sites), dtype=cp.complex64
    )
    Aky_flat = cp.ascontiguousarray(
        Ak_y.reshape(n_color, n_sites), dtype=cp.complex64
    )
    Pxx = cp.empty(n_sites, dtype=cp.float32)
    Pyy = cp.empty(n_sites, dtype=cp.float32)
    Pxy = cp.empty(n_sites, dtype=cp.float32)

    grid = (n_sites + _block - 1) // _block
    _power_spectrum_kernel((grid,), (_block,),
                           (Akx_flat, Aky_flat, Pxx, Pyy, Pxy, n_sites, n_color))

    return Pxx.reshape(N, N), Pyy.reshape(N, N), Pxy.reshape(N, N)


def measure_ww_tmd_fast(V):
    """Measure the WW gluon TMD using fused adjoint-representation kernels.

    The spatial derivatives remain spectral: U is transformed with CuPy FFTs,
    multiplied by the exact Fourier momenta, and transformed back before the
    fused ``U.T @ dU``/structure-constant kernel.
    """
    N = cfg.N
    a = cfg.a

    U = extract_adjoint_wilson_line_fast(V)
    Uk = cp.fft.fft2(U, axes=(0, 1))
    k = (2.0 * cp.pi * cp.fft.fftfreq(N, d=a)).astype(cp.float64)
    kx = k.reshape(N, 1, 1, 1)
    ky = k.reshape(1, N, 1, 1)
    dU_x = cp.fft.ifft2(1j * kx * Uk, axes=(0, 1)).real
    dU_x = dU_x.astype(cp.float32, copy=False)
    dU_y = cp.fft.ifft2(1j * ky * Uk, axes=(0, 1)).real
    dU_y = dU_y.astype(cp.float32, copy=False)
    del Uk

    alpha_x, alpha_y = extract_ww_field_fast(U, dU_x, dU_y)
    del U, dU_x, dU_y

    norm = cp.float32(a**2 / (4.0 * np.pi**2))
    ak_x = norm * cp.fft.fft2(alpha_x, axes=(1, 2))
    ak_y = norm * cp.fft.fft2(alpha_y, axes=(1, 2))
    del alpha_x, alpha_y
    Pxx, Pyy, Pxy = compute_power_spectra_fast(ak_x, ak_y)
    del ak_x, ak_y

    prefactor = cp.float32(8.0 * np.pi / (N * a) ** 2)
    G_xx = prefactor * Pxx
    G_xy = prefactor * Pxy
    G_yy = prefactor * Pyy
    del Pxx, Pxy, Pyy

    k_raw = 2.0 * cp.pi * cp.fft.fftfreq(N, d=a).astype(cp.float64)
    KX, KY = cp.meshgrid(k_raw, k_raw, indexing='ij')
    K2 = KX**2 + KY**2

    TMD_G = (G_xx + G_yy).astype(cp.float64)
    projected = (
        KX**2 * G_xx
        + 2.0 * KX * KY * G_xy
        + KY**2 * G_yy
    ).astype(cp.float64)
    TMD_H = cp.zeros_like(TMD_G)
    mask = K2 > 0
    TMD_H[mask] = 2.0 * projected[mask] / K2[mask] - TMD_G[mask]

    N2 = N // 2
    k_vals_np = cp.asnumpy(k_raw[:N2])
    G_np = cp.asnumpy(TMD_G[:N2, 0])
    H_np = cp.asnumpy(TMD_H[:N2, 0])
    return k_vals_np, G_np, H_np
