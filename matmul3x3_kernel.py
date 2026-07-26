"""Per-site 3×3 complex matrix products via CUDA RawKernels.

Each kernel processes one lattice site per thread -- all 9 complex matrix elements
live in registers (no shared memory needed for 3×3).

Kernels:
  matmul_abc:        out = A · B · C          (evolution update: exp_L · V · exp_R)
  matmul_adgba:      out = A† · B · A         (adjoint rotation, single polarization)
  matmul_adgba_dual: out0,out1 = A†·B0·A, A†·B1·A  (both polarizations, A† computed once)
"""
import cupy as cp

# ── Shared device code: load/store/multiply/dagger for 3×3 complex matrices ─
_mat3_code = r'''
__device__ __forceinline__ void load3x3(const float2* src, int idx,
                                        float* re, float* im) {
    for (int i = 0; i < 9; i++) {
        float2 v = src[idx*9+i];
        re[i] = v.x; im[i] = v.y;
    }
}

__device__ __forceinline__ void store3x3(float2* dst, int idx,
                                         const float* re, const float* im) {
    for (int i = 0; i < 9; i++)
        dst[idx*9+i] = make_float2(re[i], im[i]);
}

__device__ __forceinline__ void mul3x3(const float* ar, const float* ai,
                                       const float* br, const float* bi,
                                       float* cr, float* ci) {
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            float sr = 0.f, si = 0.f;
            for (int k = 0; k < 3; k++) {
                int rk = r*3+k, kc = k*3+c;
                sr += ar[rk]*br[kc] - ai[rk]*bi[kc];
                si += ar[rk]*bi[kc] + ai[rk]*br[kc];
            }
            cr[r*3+c] = sr;
            ci[r*3+c] = si;
        }
    }
}

__device__ __forceinline__ void dagger3x3(const float* ar, const float* ai,
                                          float* dr, float* di) {
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) {
            dr[r*3+c] =  ar[c*3+r];
            di[r*3+c] = -ai[c*3+r];
        }
}
'''

_abc_kernel = cp.RawKernel(_mat3_code + r'''
extern "C" __global__
void matmul_abc(const float2* __restrict__ A,
                const float2* __restrict__ B,
                const float2* __restrict__ C,
                float2* __restrict__ out,
                int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    float ar[9], ai[9], br[9], bi[9], cr[9], ci[9];
    float t1r[9], t1i[9], t2r[9], t2i[9];

    load3x3(A, idx, ar, ai);
    load3x3(B, idx, br, bi);
    load3x3(C, idx, cr, ci);

    mul3x3(ar, ai, br, bi, t1r, t1i);
    mul3x3(t1r, t1i, cr, ci, t2r, t2i);

    store3x3(out, idx, t2r, t2i);
}
''', 'matmul_abc')


_adgba_kernel = cp.RawKernel(_mat3_code + r'''
extern "C" __global__
void matmul_adgba(const float2* __restrict__ A,
                  const float2* __restrict__ B,
                  float2* __restrict__ out,
                  int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    float ar[9], ai[9], br[9], bi[9];
    float dr[9], di[9], t1r[9], t1i[9], t2r[9], t2i[9];

    load3x3(A, idx, ar, ai);
    load3x3(B, idx, br, bi);

    dagger3x3(ar, ai, dr, di);
    mul3x3(dr, di, br, bi, t1r, t1i);
    mul3x3(t1r, t1i, ar, ai, t2r, t2i);

    store3x3(out, idx, t2r, t2i);
}
''', 'matmul_adgba')

_adgba_dual_kernel = cp.RawKernel(_mat3_code + r'''
extern "C" __global__
void matmul_adgba_dual(const float2* __restrict__ A,
                       const float2* __restrict__ B0,
                       const float2* __restrict__ B1,
                       float2* __restrict__ out0,
                       float2* __restrict__ out1,
                       int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    float ar[9], ai[9], dr[9], di[9];
    float br[9], bi[9], t1r[9], t1i[9], t2r[9], t2i[9];

    load3x3(A, idx, ar, ai);
    dagger3x3(ar, ai, dr, di);

    // polarization 0
    load3x3(B0, idx, br, bi);
    mul3x3(dr, di, br, bi, t1r, t1i);
    mul3x3(t1r, t1i, ar, ai, t2r, t2i);
    store3x3(out0, idx, t2r, t2i);

    // polarization 1
    load3x3(B1, idx, br, bi);
    mul3x3(dr, di, br, bi, t1r, t1i);
    mul3x3(t1r, t1i, ar, ai, t2r, t2i);
    store3x3(out1, idx, t2r, t2i);
}
''', 'matmul_adgba_dual')

_block = 256


def matmul_abc(A, B, C, out=None):
    shape = A.shape
    n = 1
    for d in shape[:-2]:
        n *= d
    a = cp.ascontiguousarray(A.reshape(n, 9))
    b = cp.ascontiguousarray(B.reshape(n, 9))
    c = cp.ascontiguousarray(C.reshape(n, 9))
    if out is None:
        out_flat = cp.empty_like(a)
    else:
        out_flat = out.reshape(n, 9)
    grid = (n + _block - 1) // _block
    _abc_kernel((grid,), (_block,), (a, b, c, out_flat, n))
    if out is None:
        return out_flat.reshape(shape)
    return out


def matmul_adgba_dual(A, B0, B1, out0=None, out1=None):
    shape = B0.shape
    n = 1
    for d in shape[:-2]:
        n *= d
    a = cp.ascontiguousarray(A.reshape(n, 9))
    b0 = cp.ascontiguousarray(B0.reshape(n, 9))
    b1 = cp.ascontiguousarray(B1.reshape(n, 9))
    if out0 is None:
        out0_flat = cp.empty_like(b0)
    else:
        out0_flat = out0.reshape(n, 9)
    if out1 is None:
        out1_flat = cp.empty_like(b1)
    else:
        out1_flat = out1.reshape(n, 9)
    grid = (n + _block - 1) // _block
    _adgba_dual_kernel((grid,), (_block,), (a, b0, b1, out0_flat, out1_flat, n))
    if out0 is None:
        return out0_flat.reshape(shape), out1_flat.reshape(shape)
    return out0, out1


def matmul_adgba(A, B, out=None):
    shape = A.shape
    n_B = 1
    for d in B.shape[:-2]:
        n_B *= d
    n_A = 1
    for d in A.shape[:-2]:
        n_A *= d
    a = cp.ascontiguousarray(A.reshape(n_A, 9))
    b = cp.ascontiguousarray(B.reshape(n_B, 9))
    if out is None:
        out_flat = cp.empty_like(b)
    else:
        out_flat = out.reshape(n_B, 9)
    grid = (n_B + _block - 1) // _block
    _adgba_kernel((grid,), (_block,), (a, b, out_flat, n_B))
    if out is None:
        return out_flat.reshape(B.shape)
    return out
