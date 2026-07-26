"""JIMWLK fixed-coupling evolution -- single configuration (GPU).

Usage:
    python -m jimwlk_cuda.main [--N 64] [--Yf 1.0] [--dY 0.01] [--seed 42] [--output qs_gpu.dat]
"""

import argparse
import time
import numpy as np
import cupy as cp
from . import config as cfg
from .config import validate_orthonormality
from .initial_conditions_mv import compute_path_ordered_fund_Wilson_line
from .evolution_single import jimwlk_evolution


def check_unitarity(V):
    Vdag = cp.conj(cp.swapaxes(V, -2, -1))
    VVd = V @ Vdag
    I3 = cp.eye(3, dtype=cp.complex64)
    diff = VVd - I3[None, None, :, :]
    return float(cp.max(cp.sqrt(cp.sum(cp.abs(diff)**2, axis=(-2, -1)))))


def main():
    parser = argparse.ArgumentParser(description='JIMWLK GPU — single config')
    parser.add_argument('--N', type=int, default=None)
    parser.add_argument('--Yf', type=float, default=None)
    parser.add_argument('--dY', type=float, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='qs_gpu.dat')
    args = parser.parse_args()

    if args.N is not None:
        cfg.N = args.N
        cfg.a = cfg.l / cfg.N
        cfg.a2 = cfg.a**2
        cfg.variance_of_mv_noise = float(np.sqrt(cfg.mu2 / (cfg.Ny * cfg.a2)))
    if args.Yf is not None:
        cfg.Y_f = args.Yf
    if args.dY is not None:
        cfg.dY = args.dY

    dev = cp.cuda.Device()
    print(f"GPU: {dev.id} ({cp.cuda.runtime.getDeviceProperties(dev.id)['name'].decode()})")

    err = validate_orthonormality()
    print(f"Generator orthonormality max error: {err:.2e}")

    rng = cp.random.Generator(cp.random.XORWOW(seed=args.seed))

    print(f"Computing MV initial conditions (N={cfg.N}, Ny={cfg.Ny})...")
    t0 = time.time()
    V = compute_path_ordered_fund_Wilson_line(rng)
    cp.cuda.Stream.null.synchronize()
    print(f"  done in {time.time() - t0:.1f}s")
    print(f"  unitarity error: {check_unitarity(V):.2e}")

    print(f"Running JIMWLK evolution (Y_f={cfg.Y_f}, dY={cfg.dY})...")
    t0 = time.time()

    def print_step(step, Y, Qs, r, Sb):
        print(f"  step {step:4d}  Y={Y:.4f}  Qs={Qs:.6f}")

    result = jimwlk_evolution(V, rng=rng, callback=print_step)
    cp.cuda.Stream.null.synchronize()
    t_evol = time.time() - t0
    n_steps = len(result['Y'])
    print(f"  done in {t_evol:.1f}s ({n_steps} measurements)")

    print(f"  final unitarity error: {check_unitarity(V):.2e}")

    data = np.column_stack([result['Y'], result['Qs']])
    np.savetxt(args.output, data, fmt='%.6f', header='Y  Qs')
    print(f"Saved {args.output}")


if __name__ == '__main__':
    main()
