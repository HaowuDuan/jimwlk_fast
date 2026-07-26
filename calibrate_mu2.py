"""Calibrate mu2 to achieve a target initial Q_s(0).

Usage:
    python -m jimwlk_cuda.calibrate_mu2 --ic mv --l 256 --target_qs 0.75
    python -m jimwlk_cuda.calibrate_mu2 --ic cn --l 256 --target_qs 0.75
"""

import argparse
import numpy as np
import cupy as cp
from . import config as cfg
from .observables import measure_observables


def measure_qs0(ic_func, n_configs, seed_base=0):
    qs_list = []
    for s in range(n_configs):
        rng = cp.random.Generator(cp.random.XORWOW(seed=seed_base + s))
        V = ic_func(rng)
        Qs, _, _ = measure_observables(V)
        qs_list.append(float(Qs))
    return np.mean(qs_list), np.std(qs_list) / np.sqrt(n_configs)


def main():
    parser = argparse.ArgumentParser(description='Calibrate mu2 for target Q_s(0)')
    parser.add_argument('--ic', type=str, default='mv', choices=['mv', 'cn'])
    parser.add_argument('--l', type=float, default=256.0)
    parser.add_argument('--N', type=int, default=512)
    parser.add_argument('--target_qs', type=float, default=0.75)
    parser.add_argument('--n_configs', type=int, default=100)
    parser.add_argument('--mu2_values', type=float, nargs='+',
                        default=None, help='mu2 values to scan')
    args = parser.parse_args()

    cfg.N = args.N
    cfg.l = args.l
    cfg.a = cfg.l / cfg.N
    cfg.a2 = cfg.a ** 2

    if args.ic == 'mv':
        from .initial_conditions_mv import compute_path_ordered_fund_Wilson_line
        ic_func = compute_path_ordered_fund_Wilson_line
        if args.mu2_values is None:
            args.mu2_values = [2.0, 2.5, 3.0, 3.2, 3.5, 4.0]
    else:
        from .initial_conditions_cn import compute_path_ordered_fund_Wilson_line, set_params, Q, beta, gamma
        ic_func = compute_path_ordered_fund_Wilson_line
        if args.mu2_values is None:
            args.mu2_values = [3.0, 5.0, 7.0, 8.0, 9.0, 10.0]

    print(f"Calibrating {args.ic.upper()} at l={args.l}, N={args.N}, "
          f"target Q_s(0)={args.target_qs}, {args.n_configs} configs per point")
    print(f"{'mu2':>8s}  {'Qs(0)':>10s}  {'err':>8s}")
    print("-" * 30)

    for mu2_test in args.mu2_values:
        if args.ic == 'mv':
            cfg.mu2 = mu2_test
            cfg.variance_of_mv_noise = float(np.sqrt(mu2_test / (cfg.Ny * cfg.a2)))
        else:
            set_params(mu2_test, Q, beta, gamma)

        qs_mean, qs_err = measure_qs0(ic_func, args.n_configs)
        print(f"{mu2_test:8.2f}  {qs_mean:10.4f}  {qs_err:8.4f}")

    print(f"\nPick mu2 closest to Q_s(0) = {args.target_qs}")


if __name__ == '__main__':
    main()
