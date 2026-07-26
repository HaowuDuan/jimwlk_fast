"""SU(3) matrix exponential and reunitarization via CUDA RawKernels.

matexp_su3: computes exp(i·scale·H) for each 3×3 Hermitian matrix H on the lattice.
  - Uses Cayley-Hamilton: exp(iH) = c0·I + c1·(iH) + c2·(iH)² where c_k are
    determined by the 3 eigenvalues of iH (found via tr(H²) and det(H)).
  - Eigenvalue computation in float64 for numerical stability; final result in float32.
  - Repeated squaring for large ||H||: scale down, exponentiate, square back up.
  - Handles degenerate eigenvalues (2-fold and 3-fold) via L'Hôpital limits.

reunitarize: projects back onto SU(3) via Gram-Schmidt on rows + cross product
  for the third row (ensures det=1). Called every ~5 steps to fix FP drift.
"""
import cupy as cp

# ── matexp_su3: unscaled version (input is already i·H) ────────────────────
_matexp_kernel = cp.RawKernel(r'''
#include <cuComplex.h>

__device__ __forceinline__ cuFloatComplex cmul(cuFloatComplex a, cuFloatComplex b) {
    return make_cuFloatComplex(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

__device__ __forceinline__ cuFloatComplex cadd(cuFloatComplex a, cuFloatComplex b) {
    return make_cuFloatComplex(a.x+b.x, a.y+b.y);
}

__device__ __forceinline__ cuFloatComplex csub(cuFloatComplex a, cuFloatComplex b) {
    return make_cuFloatComplex(a.x-b.x, a.y-b.y);
}

__device__ __forceinline__ cuFloatComplex cscale(float s, cuFloatComplex a) {
    return make_cuFloatComplex(s*a.x, s*a.y);
}

extern "C" __global__
void matexp_su3(const float2* __restrict__ in,
                float2* __restrict__ out,
                int n_sites) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_sites) return;

    // load iH (3x3 complex64)
    cuFloatComplex H[9];
    for (int i = 0; i < 9; i++)
        H[i] = make_cuFloatComplex(in[idx*9+i].x, in[idx*9+i].y);

    // iH^2 = iH @ iH
    cuFloatComplex H2[9];
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            cuFloatComplex s = make_cuFloatComplex(0.f, 0.f);
            for (int k = 0; k < 3; k++)
                s = cadd(s, cmul(H[r*3+k], H[k*3+c]));
            H2[r*3+c] = s;
        }
    }

    // tr(iH^2) and det(iH) in float64 for stability
    double tr_iH2_re = (double)H2[0].x + (double)H2[4].x + (double)H2[8].x;
    double tr_iH2_im = (double)H2[0].y + (double)H2[4].y + (double)H2[8].y;

    // det(iH) = H[0,0]*(H[1,1]*H[2,2]-H[1,2]*H[2,1])
    //          -H[0,1]*(H[1,0]*H[2,2]-H[1,2]*H[2,0])
    //          +H[0,2]*(H[1,0]*H[2,1]-H[1,1]*H[2,0])
    double d_re = 0.0, d_im = 0.0;
    {
        // cofactor 00: H[1,1]*H[2,2] - H[1,2]*H[2,1]
        double c00r = (double)H[4].x*(double)H[8].x - (double)H[4].y*(double)H[8].y
                     -(double)H[5].x*(double)H[7].x + (double)H[5].y*(double)H[7].y;
        double c00i = (double)H[4].x*(double)H[8].y + (double)H[4].y*(double)H[8].x
                     -(double)H[5].x*(double)H[7].y - (double)H[5].y*(double)H[7].x;
        // cofactor 01: H[1,0]*H[2,2] - H[1,2]*H[2,0]
        double c01r = (double)H[3].x*(double)H[8].x - (double)H[3].y*(double)H[8].y
                     -(double)H[5].x*(double)H[6].x + (double)H[5].y*(double)H[6].y;
        double c01i = (double)H[3].x*(double)H[8].y + (double)H[3].y*(double)H[8].x
                     -(double)H[5].x*(double)H[6].y - (double)H[5].y*(double)H[6].x;
        // cofactor 02: H[1,0]*H[2,1] - H[1,1]*H[2,0]
        double c02r = (double)H[3].x*(double)H[7].x - (double)H[3].y*(double)H[7].y
                     -(double)H[4].x*(double)H[6].x + (double)H[4].y*(double)H[6].y;
        double c02i = (double)H[3].x*(double)H[7].y + (double)H[3].y*(double)H[7].x
                     -(double)H[4].x*(double)H[6].y - (double)H[4].y*(double)H[6].x;

        d_re = (double)H[0].x*c00r - (double)H[0].y*c00i
              -(double)H[1].x*c01r + (double)H[1].y*c01i
              +(double)H[2].x*c02r - (double)H[2].y*c02i;
        d_im = (double)H[0].x*c00i + (double)H[0].y*c00r
              -(double)H[1].x*c01i - (double)H[1].y*c01r
              +(double)H[2].x*c02i + (double)H[2].y*c02r;
    }

    // tr_H2 = -Re(tr_iH2),  det_H = Re(i * det_iH) = -det_iH.im
    double tr_H2 = -tr_iH2_re;
    double det_H = -d_im;

    // repeated squaring: scale iH so Frobenius norm < 1
    int n_sqr = 0;
    double norm2 = tr_H2 > 0.0 ? tr_H2 : 0.0;
    if (norm2 > 1.0) {
        double norm = sqrt(norm2);
        n_sqr = (int)ceil(log2(norm));
        if (n_sqr > 20) n_sqr = 20;
        double s = 1.0 / (double)(1 << n_sqr);
        float fs = (float)s;
        float fs2 = (float)(s * s);
        for (int i = 0; i < 9; i++) {
            H[i] = cscale(fs, H[i]);
            H2[i] = cscale(fs2, H2[i]);
        }
        tr_H2 *= s * s;
        det_H *= s * s * s;
    }

    double u = tr_H2 / 6.0;
    if (u < 0.0) u = 0.0;
    double sqrt_u = sqrt(u);
    double u3 = u * u * u;

    double cos_arg = 0.0;
    if (u3 > 1e-30)
        cos_arg = det_H / (2.0 * sqrt(u3 > 1e-60 ? u3 : 1e-60));
    if (cos_arg > 1.0) cos_arg = 1.0;
    if (cos_arg < -1.0) cos_arg = -1.0;
    double phi = acos(cos_arg) / 3.0;

    double theta0 = 2.0 * sqrt_u * cos(phi);
    double theta1 = 2.0 * sqrt_u * cos(phi - 2.0943951023931953);  // 2*pi/3
    double theta2 = 2.0 * sqrt_u * cos(phi + 2.0943951023931953);

    // e_k = exp(i * theta_k)
    double e0r, e0i, e1r, e1i, e2r, e2i;
    sincos(theta0, &e0i, &e0r);
    sincos(theta1, &e1i, &e1r);
    sincos(theta2, &e2i, &e2r);

    // lam_k = i * theta_k  (purely imaginary)
    double l0 = theta0, l1 = theta1, l2 = theta2;

    bool is_small = (tr_H2 < 1e-12);

    // c0, c1, c2 in complex128
    double c0r, c0i, c1r, c1i, c2r, c2i;

    if (is_small) {
        c0r = 1.0; c0i = 0.0;
        c1r = 1.0; c1i = 0.0;
        c2r = 0.5; c2i = 0.0;
    } else {
        double d01 = l0 - l1;
        double d02 = l0 - l2;
        double d12 = l1 - l2;

        // Relative degeneracy test: compare eigenvalue differences to scale
        double scale = fmax(fmax(fabs(l0), fabs(l1)), fmax(fabs(l2), 1e-14));
        double rd01 = fabs(d01) / scale;
        double rd02 = fabs(d02) / scale;
        double rd12 = fabs(d12) / scale;
        double degen_eps = 1e-6;

        bool degen_01 = (rd01 < degen_eps);
        bool degen_02 = (rd02 < degen_eps);
        bool degen_12 = (rd12 < degen_eps);
        int n_degen = (int)degen_01 + (int)degen_02 + (int)degen_12;

        if (n_degen >= 2) {
            // All three eigenvalues nearly equal. Polynomial interpolation
            // with derivative conditions at the triple root.
            double lam = (l0 + l1 + l2) / 3.0;
            double lam2 = lam * lam;
            double elr, eli;
            sincos(lam, &eli, &elr);
            c2r = elr * 0.5;
            c2i = eli * 0.5;
            c1r = elr + eli * lam;
            c1i = eli - elr * lam;
            c0r = elr * (1.0 - 0.5 * lam2) + eli * lam;
            c0i = eli * (1.0 - 0.5 * lam2) - elr * lam;
        } else if (n_degen == 1) {
            // One pair nearly degenerate (la ~ lb), lc distinct.
            // 2-eigenvalue Cayley-Hamilton with derivative condition at lm.
            double la, lb, lc;
            double ecr, eci;
            if (degen_01) {
                la = l0; lb = l1; lc = l2;
                ecr = e2r; eci = e2i;
            } else if (degen_02) {
                la = l0; lb = l2; lc = l1;
                ecr = e1r; eci = e1i;
            } else {
                la = l1; lb = l2; lc = l0;
                ecr = e0r; eci = e0i;
            }
            double lm = 0.5 * (la + lb);
            double dac = lm - lc;
            double invdac = 1.0 / dac;
            double invdac2 = invdac * invdac;
            double emr, emi;
            sincos(lm, &emi, &emr);

            // c2 = [exp(i*lm)(1 - i*dac) - exp(i*lc)] / dac^2
            double t2r = (emr + emi * dac) - ecr;
            double t2i = (emi - emr * dac) - eci;
            c2r = t2r * invdac2;
            c2i = t2i * invdac2;

            c1r = emr - 2.0 * (-c2i * lm);
            c1i = emi - 2.0 * (c2r * lm);

            c0r = (emr + emi * lm) + c2r * (-lm * lm);
            c0i = (emi - emr * lm) + c2i * (-lm * lm);
        } else {
            // Non-degenerate case: original formula
            double D = d01 * d02 * d12;
            double invD = 1.0 / D;

            double p0 = l1*l2*d12;
            double p1 = l0*l2*d02;
            double p2 = l0*l1*d01;

            double num0r = e0i*p0 - e1i*p1 + e2i*p2;
            double num0i = -e0r*p0 + e1r*p1 - e2r*p2;

            double q0 = -(l1*l1 - l2*l2);
            double q1 = -(l0*l0 - l2*l2);
            double q2 = -(l0*l0 - l1*l1);

            double num1r = -e0r*q0 + e1r*q1 - e2r*q2;
            double num1i = -e0i*q0 + e1i*q1 - e2i*q2;

            double num2r = -e0i*d12 + e1i*d02 - e2i*d01;
            double num2i =  e0r*d12 - e1r*d02 + e2r*d01;

            c0r = -num0i * invD;  c0i = num0r * invD;
            c1r = -num1i * invD;  c1i = num1r * invD;
            c2r = -num2i * invD;  c2i = num2r * invD;
        }
    }

    // result = c0*I + c1*iH + c2*iH^2  (in float32)
    float fc0r = (float)c0r, fc0i = (float)c0i;
    float fc1r = (float)c1r, fc1i = (float)c1i;
    float fc2r = (float)c2r, fc2i = (float)c2i;

    cuFloatComplex R[9];
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            int j = r*3+c;
            float rr = fc1r*H[j].x - fc1i*H[j].y + fc2r*H2[j].x - fc2i*H2[j].y;
            float ri = fc1r*H[j].y + fc1i*H[j].x + fc2r*H2[j].y + fc2i*H2[j].x;
            if (r == c) { rr += fc0r; ri += fc0i; }
            R[j] = make_cuFloatComplex(rr, ri);
        }
    }

    // repeated squaring: exp(iH) = exp(iH/2^n)^{2^n}
    for (int s = 0; s < n_sqr; s++) {
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                cuFloatComplex sum = make_cuFloatComplex(0.f, 0.f);
                for (int k = 0; k < 3; k++)
                    sum = cadd(sum, cmul(R[r*3+k], R[k*3+c]));
                H2[r*3+c] = sum;
            }
        }
        for (int i = 0; i < 9; i++) R[i] = H2[i];
    }

    for (int i = 0; i < 9; i++)
        out[idx*9+i] = make_float2(R[i].x, R[i].y);
}
''', 'matexp_su3')


# ── matexp_su3_scaled: fuses scale*i multiplication into the load ────────────
# Input is a Hermitian matrix H; kernel computes exp(scale · i · H).
# Avoids a separate CuPy elementwise op to form scale*i*H before matexp.
_matexp_scaled_kernel = cp.RawKernel(r'''
#include <cuComplex.h>

__device__ __forceinline__ cuFloatComplex cmul_s(cuFloatComplex a, cuFloatComplex b) {
    return make_cuFloatComplex(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
__device__ __forceinline__ cuFloatComplex cadd_s(cuFloatComplex a, cuFloatComplex b) {
    return make_cuFloatComplex(a.x+b.x, a.y+b.y);
}
__device__ __forceinline__ cuFloatComplex cscale_s(float s, cuFloatComplex a) {
    return make_cuFloatComplex(s*a.x, s*a.y);
}

extern "C" __global__
void matexp_su3_scaled(const float2* __restrict__ in,
                       float2* __restrict__ out,
                       int n_sites, float scale) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_sites) return;

    // load and apply scale * i: scale*i*(a+bi) = scale*(-b + ai)
    cuFloatComplex H[9];
    for (int i = 0; i < 9; i++) {
        float2 v = in[idx*9+i];
        H[i] = make_cuFloatComplex(-scale * v.y, scale * v.x);
    }

    cuFloatComplex H2[9];
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            cuFloatComplex s = make_cuFloatComplex(0.f, 0.f);
            for (int k = 0; k < 3; k++)
                s = cadd_s(s, cmul_s(H[r*3+k], H[k*3+c]));
            H2[r*3+c] = s;
        }
    }

    double tr_iH2_re = (double)H2[0].x + (double)H2[4].x + (double)H2[8].x;

    double d_re = 0.0, d_im = 0.0;
    {
        double c00r = (double)H[4].x*(double)H[8].x - (double)H[4].y*(double)H[8].y
                     -(double)H[5].x*(double)H[7].x + (double)H[5].y*(double)H[7].y;
        double c00i = (double)H[4].x*(double)H[8].y + (double)H[4].y*(double)H[8].x
                     -(double)H[5].x*(double)H[7].y - (double)H[5].y*(double)H[7].x;
        double c01r = (double)H[3].x*(double)H[8].x - (double)H[3].y*(double)H[8].y
                     -(double)H[5].x*(double)H[6].x + (double)H[5].y*(double)H[6].y;
        double c01i = (double)H[3].x*(double)H[8].y + (double)H[3].y*(double)H[8].x
                     -(double)H[5].x*(double)H[6].y - (double)H[5].y*(double)H[6].x;
        double c02r = (double)H[3].x*(double)H[7].x - (double)H[3].y*(double)H[7].y
                     -(double)H[4].x*(double)H[6].x + (double)H[4].y*(double)H[6].y;
        double c02i = (double)H[3].x*(double)H[7].y + (double)H[3].y*(double)H[7].x
                     -(double)H[4].x*(double)H[6].y - (double)H[4].y*(double)H[6].x;
        d_re = (double)H[0].x*c00r - (double)H[0].y*c00i
              -(double)H[1].x*c01r + (double)H[1].y*c01i
              +(double)H[2].x*c02r - (double)H[2].y*c02i;
        d_im = (double)H[0].x*c00i + (double)H[0].y*c00r
              -(double)H[1].x*c01i - (double)H[1].y*c01r
              +(double)H[2].x*c02i + (double)H[2].y*c02r;
    }

    double tr_H2 = -tr_iH2_re;
    double det_H = -d_im;

    int n_sqr = 0;
    double norm2 = tr_H2 > 0.0 ? tr_H2 : 0.0;
    if (norm2 > 1.0) {
        double norm = sqrt(norm2);
        n_sqr = (int)ceil(log2(norm));
        if (n_sqr > 20) n_sqr = 20;
        double s = 1.0 / (double)(1 << n_sqr);
        float fs = (float)s;
        float fs2 = (float)(s * s);
        for (int i = 0; i < 9; i++) {
            H[i] = cscale_s(fs, H[i]);
            H2[i] = cscale_s(fs2, H2[i]);
        }
        tr_H2 *= s * s;
        det_H *= s * s * s;
    }

    double u = tr_H2 / 6.0;
    if (u < 0.0) u = 0.0;
    double sqrt_u = sqrt(u);
    double u3 = u * u * u;

    double cos_arg = 0.0;
    if (u3 > 1e-30)
        cos_arg = det_H / (2.0 * sqrt(u3 > 1e-60 ? u3 : 1e-60));
    if (cos_arg > 1.0) cos_arg = 1.0;
    if (cos_arg < -1.0) cos_arg = -1.0;
    double phi = acos(cos_arg) / 3.0;

    double theta0 = 2.0 * sqrt_u * cos(phi);
    double theta1 = 2.0 * sqrt_u * cos(phi - 2.0943951023931953);
    double theta2 = 2.0 * sqrt_u * cos(phi + 2.0943951023931953);

    double e0r, e0i, e1r, e1i, e2r, e2i;
    sincos(theta0, &e0i, &e0r);
    sincos(theta1, &e1i, &e1r);
    sincos(theta2, &e2i, &e2r);

    double l0 = theta0, l1 = theta1, l2 = theta2;

    bool is_small = (tr_H2 < 1e-12);
    double c0r, c0i, c1r, c1i, c2r, c2i;

    if (is_small) {
        c0r = 1.0; c0i = 0.0;
        c1r = 1.0; c1i = 0.0;
        c2r = 0.5; c2i = 0.0;
    } else {
        double d01 = l0 - l1;
        double d02 = l0 - l2;
        double d12 = l1 - l2;

        double sc = fmax(fmax(fabs(l0), fabs(l1)), fmax(fabs(l2), 1e-14));
        double rd01 = fabs(d01) / sc;
        double rd02 = fabs(d02) / sc;
        double rd12 = fabs(d12) / sc;
        double degen_eps = 1e-6;
        bool degen_01 = (rd01 < degen_eps);
        bool degen_02 = (rd02 < degen_eps);
        bool degen_12 = (rd12 < degen_eps);
        int n_degen = (int)degen_01 + (int)degen_02 + (int)degen_12;

        if (n_degen >= 2) {
            double lam = (l0 + l1 + l2) / 3.0;
            double lam2 = lam * lam;
            double elr2, eli2;
            sincos(lam, &eli2, &elr2);
            c2r = elr2 * 0.5;
            c2i = eli2 * 0.5;
            c1r = elr2 + eli2 * lam;
            c1i = eli2 - elr2 * lam;
            c0r = elr2 * (1.0 - 0.5 * lam2) + eli2 * lam;
            c0i = eli2 * (1.0 - 0.5 * lam2) - elr2 * lam;
        } else if (n_degen == 1) {
            double la, lb, lc;
            double ear2, eai2, ebr2, ebi2, ecr2, eci2;
            if (degen_01) {
                la = l0; lb = l1; lc = l2;
                ear2 = e0r; eai2 = e0i; ebr2 = e1r; ebi2 = e1i; ecr2 = e2r; eci2 = e2i;
            } else if (degen_02) {
                la = l0; lb = l2; lc = l1;
                ear2 = e0r; eai2 = e0i; ebr2 = e2r; ebi2 = e2i; ecr2 = e1r; eci2 = e1i;
            } else {
                la = l1; lb = l2; lc = l0;
                ear2 = e1r; eai2 = e1i; ebr2 = e2r; ebi2 = e2i; ecr2 = e0r; eci2 = e0i;
            }
            double lm = 0.5 * (la + lb);
            double dac = lm - lc;
            double invdac = 1.0 / dac;
            double invdac2 = invdac * invdac;
            double emr2, emi2;
            sincos(lm, &emi2, &emr2);
            double t2r = (emr2 + emi2 * dac) - ecr2;
            double t2i = (emi2 - emr2 * dac) - eci2;
            c2r = t2r * invdac2;
            c2i = t2i * invdac2;
            c1r = emr2 - 2.0 * (-c2i * lm);
            c1i = emi2 - 2.0 * (c2r * lm);
            c0r = (emr2 + emi2 * lm) + c2r * (-lm * lm);
            c0i = (emi2 - emr2 * lm) + c2i * (-lm * lm);
        } else {
            double D = d01 * d02 * d12;
            double invD = 1.0 / D;
            double p0 = l1*l2*d12;
            double p1 = l0*l2*d02;
            double p2 = l0*l1*d01;
            double num0r = e0i*p0 - e1i*p1 + e2i*p2;
            double num0i = -e0r*p0 + e1r*p1 - e2r*p2;
            double q0 = -(l1*l1 - l2*l2);
            double q1 = -(l0*l0 - l2*l2);
            double q2 = -(l0*l0 - l1*l1);
            double num1r = -e0r*q0 + e1r*q1 - e2r*q2;
            double num1i = -e0i*q0 + e1i*q1 - e2i*q2;
            double num2r = -e0i*d12 + e1i*d02 - e2i*d01;
            double num2i =  e0r*d12 - e1r*d02 + e2r*d01;
            c0r = -num0i * invD;  c0i = num0r * invD;
            c1r = -num1i * invD;  c1i = num1r * invD;
            c2r = -num2i * invD;  c2i = num2r * invD;
        }
    }

    float fc0r = (float)c0r, fc0i = (float)c0i;
    float fc1r = (float)c1r, fc1i = (float)c1i;
    float fc2r = (float)c2r, fc2i = (float)c2i;

    cuFloatComplex R[9];
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            int j = r*3+c;
            float rr = fc1r*H[j].x - fc1i*H[j].y + fc2r*H2[j].x - fc2i*H2[j].y;
            float ri = fc1r*H[j].y + fc1i*H[j].x + fc2r*H2[j].y + fc2i*H2[j].x;
            if (r == c) { rr += fc0r; ri += fc0i; }
            R[j] = make_cuFloatComplex(rr, ri);
        }
    }

    for (int s = 0; s < n_sqr; s++) {
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                cuFloatComplex sum = make_cuFloatComplex(0.f, 0.f);
                for (int k = 0; k < 3; k++)
                    sum = cadd_s(sum, cmul_s(R[r*3+k], R[k*3+c]));
                H2[r*3+c] = sum;
            }
        }
        for (int i = 0; i < 9; i++) R[i] = H2[i];
    }

    for (int i = 0; i < 9; i++)
        out[idx*9+i] = make_float2(R[i].x, R[i].y);
}
''', 'matexp_su3_scaled')


# ── Reunitarization: Gram-Schmidt + cross product → SU(3) ───────────────────
_reunit_kernel = cp.RawKernel(r'''
extern "C" __global__
void reunitarize_su3(float2* __restrict__ V, int n_sites) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_sites) return;

    // load 3x3 matrix: row-major V[row][col]
    float vr[9], vi[9];
    for (int i = 0; i < 9; i++) {
        float2 z = V[idx*9+i];
        vr[i] = z.x; vi[i] = z.y;
    }

    // Gram-Schmidt on rows
    // row0: normalize
    float n0 = 0.f;
    for (int c = 0; c < 3; c++) {
        n0 += vr[c]*vr[c] + vi[c]*vi[c];
    }
    float inv0 = rsqrtf(n0);
    for (int c = 0; c < 3; c++) {
        vr[c] *= inv0; vi[c] *= inv0;
    }

    // row1: subtract projection onto row0, normalize
    // <row0|row1> = sum_c conj(row0[c]) * row1[c]
    float pr = 0.f, pi = 0.f;
    for (int c = 0; c < 3; c++) {
        pr += vr[c]*vr[3+c] + vi[c]*vi[3+c];
        pi += vr[c]*vi[3+c] - vi[c]*vr[3+c];
    }
    for (int c = 0; c < 3; c++) {
        // row1 -= proj * row0
        vr[3+c] -= pr*vr[c] - pi*vi[c];
        vi[3+c] -= pr*vi[c] + pi*vr[c];
    }
    float n1 = 0.f;
    for (int c = 0; c < 3; c++) {
        n1 += vr[3+c]*vr[3+c] + vi[3+c]*vi[3+c];
    }
    float inv1 = rsqrtf(n1);
    for (int c = 0; c < 3; c++) {
        vr[3+c] *= inv1; vi[3+c] *= inv1;
    }

    // row2 = conj(row0 x row1) for det=1
    // (row0 x row1)[c] = row0[a]*row1[b] - row0[b]*row1[a]  (cyclic)
    for (int c = 0; c < 3; c++) {
        int a = (c+1)%3, b = (c+2)%3;
        float xr = vr[a]*vr[3+b] - vi[a]*vi[3+b] - vr[b]*vr[3+a] + vi[b]*vi[3+a];
        float xi = vr[a]*vi[3+b] + vi[a]*vr[3+b] - vr[b]*vi[3+a] - vi[b]*vr[3+a];
        vr[6+c] = xr;
        vi[6+c] = -xi;
    }

    for (int i = 0; i < 9; i++)
        V[idx*9+i] = make_float2(vr[i], vi[i]);
}
''', 'reunitarize_su3')


def reunitarize(V):
    shape = V.shape
    n_sites = 1
    for d in shape[:-2]:
        n_sites *= d
    V_flat = cp.ascontiguousarray(V.reshape(n_sites, 9))
    block = 256
    grid = (n_sites + block - 1) // block
    _reunit_kernel((grid,), (block,), (V_flat, n_sites))
    return V_flat.reshape(shape)


def matexp_su3(iH, scale=None):
    """Compute exp(scale·i·H) for each 3×3 matrix on the lattice (one GPU thread per site).

    Without scale: input is already i·H. With scale: kernel fuses the scale*i multiply.
    """
    shape = iH.shape
    n_sites = 1
    for d in shape[:-2]:
        n_sites *= d
    iH_flat = cp.ascontiguousarray(iH.reshape(n_sites, 9))
    out = cp.empty_like(iH_flat)
    block = 256
    grid = (n_sites + block - 1) // block
    if scale is not None:
        _matexp_scaled_kernel((grid,), (block,),
                              (iH_flat, out, n_sites, cp.float32(scale)))
    else:
        _matexp_kernel((grid,), (block,), (iH_flat, out, n_sites))
    return out.reshape(shape)
