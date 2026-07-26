"""Diagnose NaN in JIMWLK evolution.

Run a few configs and check where NaN first appears.

Usage:
    python -m jimwlk_cuda.diagnose_nan --ic mv --mu2 3.2 --l 256
"""

import argparse
import numpy as np
import cupy as cp
from . import config as cfg
from .evolution_single import jimwlk_evolution, _get_ww_kernel
from .observables import measure_dipole


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ic', type=str, default='mv', choices=['mv', 'cn'])
    parser.add_argument('--mu2', type=float, default=3.2)
    parser.add_argument('--l', type=float, default=256.0)
    parser.add_argument('--N', type=int, default=512)
    parser.add_argument('--n_configs', type=int, default=20)
    args = parser.parse_args()

    cfg.N = args.N
    cfg.l = args.l
    cfg.a = cfg.l / cfg.N
    cfg.a2 = cfg.a ** 2

    if args.ic == 'mv':
        cfg.mu2 = args.mu2
        cfg.variance_of_mv_noise = float(np.sqrt(args.mu2 / (cfg.Ny * cfg.a2)))
        from .initial_conditions_mv import compute_path_ordered_fund_Wilson_line as init_V
    else:
        from .initial_conditions_cn import compute_path_ordered_fund_Wilson_line as init_V
        from .initial_conditions_cn import set_params, Q, beta, gamma
        set_params(args.mu2, Q, beta, gamma)

    for ic in range(args.n_configs):
        rng = cp.random.Generator(cp.random.XORWOW(seed=42 + ic))
        V = init_V(rng)

        V_np = V.get()
        if np.any(np.isnan(V_np)):
            print(f"Config {ic}: NaN in initial V!")
            continue

        # Check unitarity
        Vd = np.conj(np.swapaxes(V_np, -2, -1))
        err = np.max(np.abs(V_np @ Vd - np.eye(3)))

        r, Sb = measure_dipole(V)
        has_nan = np.any(np.isnan(Sb))
        print(f"Config {ic}: unitarity err={err:.2e}, S(r=1)={Sb[0]:.4f}, "
              f"NaN in S={has_nan}")

        # Now evolve step by step and check
        result = jimwlk_evolution(V, Y_f=0.5, dY=0.01, rng=rng, measure_interval=10)

        for iy, Y in enumerate(result['Y']):
            S = result['S'][iy]
            n_nan = np.sum(np.isnan(S))
            if n_nan > 0:
                first_nan_idx = np.where(np.isnan(S))[0][0]
                print(f"  Y={Y:.2f}: {n_nan} NaN bins, first at r={result['r'][first_nan_idx]:.1f}")
            else:
                print(f"  Y={Y:.2f}: OK, S(r=1)={S[0]:.4f}")

        # Check final V
        V_np = V.get()
        n_nan_V = np.sum(np.isnan(V_np))
        if n_nan_V > 0:
            print(f"  Final V has {n_nan_V} NaN elements!")
        print()


if __name__ == '__main__':
    main()
