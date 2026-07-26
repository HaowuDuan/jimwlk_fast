"""SU(3) noise generation for JIMWLK evolution.

Two paths for generating noise fields ξ^a(x) t^a:
  1. assemble_su3_noise: generates random numbers and directly assembles into
     anti-hermitian traceless 3×3 matrices (used by right side / legacy path).
  2. generate_noise_raw_batch: returns raw real coefficients in color-major
     layout (8, batch, 2, N, N) for the R2C FFT path (left side). These are
     later assembled into algebra elements via assemble_algebra (in ic_fast.py).

The √2 factor in raw batch compensates for the difference between the two
assembly kernels: assemble_su3_noise divides off-diag by √2 while
assemble_algebra uses Gell-Mann t^a = λ^a/2 which does not.
"""
import cupy as cp

_assemble_noise_kernel = cp.RawKernel(r'''
extern "C" __global__
void assemble_su3_noise(const float* __restrict__ rands,
                        float2* __restrict__ xi,
                        int n_sites) {
    // rands: (8, n_sites) - 8 real random numbers per site (one polarization)
    // xi: (n_sites, 9) as float2 (complex64) - output anti-hermitian traceless matrix
    //
    // Random number layout per site:
    //   rands[0*n + i] = re_01    rands[1*n + i] = im_01
    //   rands[2*n + i] = re_02    rands[3*n + i] = im_02
    //   rands[4*n + i] = re_12    rands[5*n + i] = im_12
    //   rands[6*n + i] = d11_raw  rands[7*n + i] = d33_raw
    //
    // Off-diagonal (ic < jc):
    //   xi[ic,jc] = (re + i*im) / sqrt(2)
    //   xi[jc,ic] = (re - i*im) / sqrt(2)    (anti-hermitian)
    //
    // Diagonal:
    //   d11 = d11_raw / sqrt(2)
    //   d33 = -sqrt(2/3) * d33_raw
    //   xi[0,0] = d11 - d33/2
    //   xi[1,1] = -(d11 + d33/2)
    //   xi[2,2] = d33

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_sites) return;

    int n = n_sites;
    float inv_sqrt2 = 0.70710678118f;
    float sqrt_2_3 = 0.81649658093f;  // sqrt(2/3)

    // off-diagonal (0,1)
    float re01 = rands[0*n + i];
    float im01 = rands[1*n + i];
    // off-diagonal (0,2)
    float re02 = rands[2*n + i];
    float im02 = rands[3*n + i];
    // off-diagonal (1,2)
    float re12 = rands[4*n + i];
    float im12 = rands[5*n + i];

    // diagonal
    float d11 = rands[6*n + i] * inv_sqrt2;
    float d33 = -sqrt_2_3 * rands[7*n + i];

    // xi[0,0] = d11 - d33/2  (real)
    xi[i*9 + 0] = make_float2(d11 - d33*0.5f, 0.f);
    // xi[0,1] = (re01 + i*im01)/sqrt(2)
    xi[i*9 + 1] = make_float2(re01*inv_sqrt2, im01*inv_sqrt2);
    // xi[0,2] = (re02 + i*im02)/sqrt(2)
    xi[i*9 + 2] = make_float2(re02*inv_sqrt2, im02*inv_sqrt2);
    // xi[1,0] = (re01 - i*im01)/sqrt(2)
    xi[i*9 + 3] = make_float2(re01*inv_sqrt2, -im01*inv_sqrt2);
    // xi[1,1] = -(d11 + d33/2)  (real)
    xi[i*9 + 4] = make_float2(-(d11 + d33*0.5f), 0.f);
    // xi[1,2] = (re12 + i*im12)/sqrt(2)
    xi[i*9 + 5] = make_float2(re12*inv_sqrt2, im12*inv_sqrt2);
    // xi[2,0] = (re02 - i*im02)/sqrt(2)
    xi[i*9 + 6] = make_float2(re02*inv_sqrt2, -im02*inv_sqrt2);
    // xi[2,1] = (re12 - i*im12)/sqrt(2)
    xi[i*9 + 7] = make_float2(re12*inv_sqrt2, -im12*inv_sqrt2);
    // xi[2,2] = d33  (real)
    xi[i*9 + 8] = make_float2(d33, 0.f);
}
''', 'assemble_su3_noise')

_block = 256


def generate_noise(rng, N):
    n_sites = 2 * N * N
    rands = rng.standard_normal((8, n_sites), dtype=cp.float32)
    xi = cp.empty((n_sites, 9), dtype=cp.complex64)
    grid = (n_sites + _block - 1) // _block
    _assemble_noise_kernel((grid,), (_block,), (rands, xi, n_sites))
    return xi.reshape(2, N, N, 3, 3)


def generate_noise_batch(rng, N, batch):
    n_sites = batch * 2 * N * N
    rands = rng.standard_normal((8, n_sites), dtype=cp.float32)
    xi = cp.empty((n_sites, 9), dtype=cp.complex64)
    grid = (n_sites + _block - 1) // _block
    _assemble_noise_kernel((grid,), (_block,), (rands, xi, n_sites))
    return xi.reshape(batch, 2, N, N, 3, 3)


def generate_noise_raw_batch(rng, N, batch):
    """Raw noise for R2C path: 8 real Gell-Mann coefficients, color-major layout.

    Shape (8, batch, 2, N, N): 8 color components × batch × 2 polarizations × N×N.
    Multiplied by √2 to match the variance of assemble_su3_noise when later
    passed through assemble_algebra (Gell-Mann t^a = λ^a/2 normalization).
    """
    raw = rng.standard_normal((8, batch, 2, N, N), dtype=cp.float32)
    raw *= cp.float32(1.4142135623730951)
    return raw


