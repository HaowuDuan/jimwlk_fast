"""Fast initial condition generation for MV and CN models.

Batches all Ny layers' noise/FFT/assembly/matexp, then path-orders in a single kernel.

Usage:
    from jimwlk_cuda.ic_fast import compute_path_ordered_Wilson_line
    V = compute_path_ordered_Wilson_line(rng, propagator, sigma, Ny)
"""

import cupy as cp
from . import config as cfg
from .su3_kernel import matexp_su3


# ── Path-ordering kernel: V = V_1 · V_2 · ... · V_Ny ───────────────────────
# One thread per spatial site; sequentially multiplies Ny layers in registers.
_path_order_kernel = cp.RawKernel(r'''
extern "C" __global__
void path_order(const float2* __restrict__ layers,
                float2* __restrict__ out,
                int n_sites, int Ny) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_sites) return;

    float ar[9], ai[9];
    for (int i = 0; i < 9; i++) {
        float2 v = layers[idx * 9 + i];
        ar[i] = v.x; ai[i] = v.y;
    }

    for (int layer = 1; layer < Ny; layer++) {
        float br[9], bi[9], cr[9], ci[9];
        int base = layer * n_sites * 9 + idx * 9;
        for (int i = 0; i < 9; i++) {
            float2 v = layers[base + i];
            br[i] = v.x; bi[i] = v.y;
        }
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
        for (int i = 0; i < 9; i++) { ar[i] = cr[i]; ai[i] = ci[i]; }
    }

    for (int i = 0; i < 9; i++)
        out[idx * 9 + i] = make_float2(ar[i], ai[i]);
}
''', 'path_order')


# ── Gell-Mann assembly: 8 real coefficients → 3×3 su(3) matrix ──────────────
# A = sum_a fields[a] · t^a where t^a = λ^a/2 are Gell-Mann generators.
# Used by both IC generation and the R2C evolution path.
_assemble_algebra_kernel = cp.RawKernel(r'''
extern "C" __global__
void assemble_algebra(const float* __restrict__ fields,
                      float2* __restrict__ A,
                      int n_sites) {
    // fields: (8, N, N) flattened, 8 real scalar fields A^a(x) per site
    // A: (N*N, 9) as float2, output su(3) algebra element sum_a A^a * t^a
    //
    // Gell-Mann t^a = lambda^a / 2:
    //   t1: (0,1)=1/2, (1,0)=1/2
    //   t2: (0,1)=-i/2, (1,0)=i/2
    //   t3: (0,0)=1/2, (1,1)=-1/2
    //   t4: (0,2)=1/2, (2,0)=1/2
    //   t5: (0,2)=-i/2, (2,0)=i/2
    //   t6: (1,2)=1/2, (2,1)=1/2
    //   t7: (1,2)=-i/2, (2,1)=i/2
    //   t8: (0,0)=1/(2*sqrt3), (1,1)=1/(2*sqrt3), (2,2)=-1/sqrt3

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_sites) return;

    int n = n_sites;
    float a1 = fields[0*n + i];
    float a2 = fields[1*n + i];
    float a3 = fields[2*n + i];
    float a4 = fields[3*n + i];
    float a5 = fields[4*n + i];
    float a6 = fields[5*n + i];
    float a7 = fields[6*n + i];
    float a8 = fields[7*n + i];

    float inv2sqrt3 = 0.28867513459f;  // 1/(2*sqrt(3))
    float inv_sqrt3 = 0.57735026919f;  // 1/sqrt(3)

    // A[0,0] = a3/2 + a8/(2*sqrt3)
    A[i*9 + 0] = make_float2(0.5f*a3 + inv2sqrt3*a8, 0.f);
    // A[0,1] = (a1 - i*a2)/2
    A[i*9 + 1] = make_float2(0.5f*a1, -0.5f*a2);
    // A[0,2] = (a4 - i*a5)/2
    A[i*9 + 2] = make_float2(0.5f*a4, -0.5f*a5);
    // A[1,0] = (a1 + i*a2)/2
    A[i*9 + 3] = make_float2(0.5f*a1, 0.5f*a2);
    // A[1,1] = -a3/2 + a8/(2*sqrt3)
    A[i*9 + 4] = make_float2(-0.5f*a3 + inv2sqrt3*a8, 0.f);
    // A[1,2] = (a6 - i*a7)/2
    A[i*9 + 5] = make_float2(0.5f*a6, -0.5f*a7);
    // A[2,0] = (a4 + i*a5)/2
    A[i*9 + 6] = make_float2(0.5f*a4, 0.5f*a5);
    // A[2,1] = (a6 + i*a7)/2
    A[i*9 + 7] = make_float2(0.5f*a6, 0.5f*a7);
    // A[2,2] = -a8/sqrt3
    A[i*9 + 8] = make_float2(-inv_sqrt3*a8, 0.f);
}
''', 'assemble_algebra')

_block = 256


def compute_single_layer(rng, propagator, sigma, N):
    """Generate one layer: 8 noise fields → FFT → propagator → IFFT → assemble algebra → matexp."""
    n_color = 8
    n_sites = N * N

    rho = sigma * rng.standard_normal((n_color, N, N), dtype=cp.float32)

    rhok = cp.fft.fft2(rho, axes=(1, 2))
    rhok *= propagator[None, :, :]

    A_fields = cp.real(cp.fft.ifft2(rhok, axes=(1, 2)))

    A_flat = cp.ascontiguousarray(A_fields.reshape(n_color, n_sites))
    A_mat = cp.empty((n_sites, 9), dtype=cp.complex64)
    grid = (n_sites + _block - 1) // _block
    _assemble_algebra_kernel((grid,), (_block,), (A_flat, A_mat, n_sites))

    V = matexp_su3(1j * A_mat.reshape(N, N, 3, 3))
    return V


def compute_path_ordered_Wilson_line(rng, propagator, sigma, Ny):
    """Generate path-ordered Wilson line V = prod_{y=1}^{Ny} exp(i·A_y).

    Pipeline (all on GPU):
      1. Generate Ny×8 Gaussian noise fields ρ^a_y(x)
      2. rfft2 all fields (R2C: half-spectrum for real input)
      3. Multiply by 1/(-k²+m²) propagator in k-space
      4. irfft2 back to position space → A^a_y(x)
      5. Assemble A^a·t^a into 3×3 su(3) matrices (Gell-Mann kernel)
      6. matexp each layer: V_y = exp(i·A_y)
      7. Path-order: V = V_1 · V_2 · ... · V_Ny (path_order kernel)
    """
    N = cfg.N
    n_color = 8
    n_sites = N * N

    all_rho = sigma * rng.standard_normal((Ny * n_color, N, N), dtype=cp.float32)

    prop_rfft = propagator[:, :N // 2 + 1]
    all_rhok = cp.fft.rfft2(all_rho, axes=(1, 2))
    all_rhok *= prop_rfft[None, :, :]
    del all_rho

    all_A = cp.fft.irfft2(all_rhok, axes=(1, 2), s=(N, N))
    del all_rhok

    total_sites = Ny * n_sites
    # Transpose to color-major (8, Ny*N*N) for assemble_algebra kernel
    all_A = all_A.reshape(Ny, n_color, N, N).transpose(1, 0, 2, 3)
    A_flat = cp.ascontiguousarray(all_A.reshape(n_color, total_sites))
    del all_A

    A_mat = cp.empty((total_sites, 9), dtype=cp.complex64)
    grid = (total_sites + _block - 1) // _block
    _assemble_algebra_kernel((grid,), (_block,), (A_flat, A_mat, total_sites))
    del A_flat

    V_all = matexp_su3(1j * A_mat.reshape(Ny, N, N, 3, 3))
    del A_mat

    # Path-order all layers in a single kernel (one thread per spatial site)
    V_flat = cp.ascontiguousarray(V_all.reshape(Ny, n_sites, 9))
    del V_all
    out = cp.empty((n_sites, 9), dtype=cp.complex64)
    grid = (n_sites + _block - 1) // _block
    _path_order_kernel((grid,), (_block,), (V_flat, out, n_sites, Ny))
    return out.reshape(N, N, 3, 3)
