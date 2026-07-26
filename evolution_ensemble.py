"""Ensemble-averaged JIMWLK evolution -- GPU version.

Usage:
    python -m jimwlk_cuda.ensemble \
        --N 512 --Yf 1.0 --dY 0.01 --n_configs 1000 \
        --measure_every 10 --outdir results_gpu/
"""

import argparse
import os
import time
import numpy as np
import cupy as cp
from . import config as cfg
from .evolution_single import jimwlk_evolution
from .observables import Qs_of_S

_init_V = None


def set_initial_condition(ic_type='mv'):
    global _init_V
    if ic_type == 'mv':
        from .initial_conditions_mv import compute_path_ordered_fund_Wilson_line
    elif ic_type == 'cn':
        from .initial_conditions_cn import compute_path_ordered_fund_Wilson_line
    else:
        raise ValueError(f"Unknown initial condition: {ic_type!r} (use 'mv' or 'cn')")
    _init_V = compute_path_ordered_fund_Wilson_line


def run_single_config(config_id, rng, Y_f, dY, measure_interval, derivative='forward', measure_ww=True):
    V = _init_V(rng)
    result = jimwlk_evolution(V, Y_f=Y_f, dY=dY, rng=rng,
                              measure_interval=measure_interval,
                              derivative=derivative, measure_ww=measure_ww)
    return result


def run_ensemble(n_configs, Y_f, dY, measure_interval, seed_base=None,
                 progress_callback=None, derivative='forward', measure_ww=True):
    S_sum = None
    xG_sum = None
    xh_sum = None
    xG_k_sum = None
    xh_k_sum = None
    ww_G_sum = None
    ww_H_sum = None
    Qs_configs = []
    r_bins = None
    r_xG = None
    k_vals = None
    k_ww = None
    Y_arr = None
    n_valid = 0

    for ic in range(n_configs):
        seed = (seed_base + ic) if seed_base is not None else ic
        rng = cp.random.Generator(cp.random.XORWOW(seed=seed))

        t0 = time.time()
        result = run_single_config(ic, rng, Y_f, dY, measure_interval, derivative=derivative, measure_ww=measure_ww)
        cp.cuda.Stream.null.synchronize()
        dt = time.time() - t0

        if S_sum is None:
            S_sum = np.zeros_like(result['S'], dtype=np.float64)
            xG_sum = np.zeros_like(result['xG'], dtype=np.float64)
            xh_sum = np.zeros_like(result['xh'], dtype=np.float64)
            xG_k_sum = np.zeros_like(result['xG_k'], dtype=np.float64)
            xh_k_sum = np.zeros_like(result['xh_k'], dtype=np.float64)
            if measure_ww:
                ww_G_sum = np.zeros_like(result['ww_G'], dtype=np.float64)
                ww_H_sum = np.zeros_like(result['ww_H'], dtype=np.float64)
                k_ww = result['k_ww']
            r_bins = result['r']
            r_xG = result['r_xG']
            k_vals = result['k_vals']
            Y_arr = result['Y']

        if np.any(np.isnan(result['S'])):
            nan_ys = [result['Y'][i] for i in range(len(result['Y']))
                      if np.any(np.isnan(result['S'][i]))]
            print(f"  WARNING: config {ic} (seed={seed}) has NaN at Y={nan_ys}, skipping")
            continue

        S_sum += result['S']
        xG_sum += result['xG']
        xh_sum += result['xh']
        xG_k_sum += result['xG_k']
        xh_k_sum += result['xh_k']
        if measure_ww:
            ww_G_sum += result['ww_G']
            ww_H_sum += result['ww_H']
        n_valid += 1
        Qs_configs.append(result['Qs'])

        if progress_callback:
            progress_callback(ic, n_configs, dt, result['Qs'][-1])

    if n_valid < n_configs:
        print(f"  {n_configs - n_valid} configs skipped due to NaN, {n_valid} valid")

    S_ensemble = (S_sum / n_valid).astype(np.float32)
    xG_ensemble = (xG_sum / n_valid).astype(np.float32)
    xh_ensemble = (xh_sum / n_valid).astype(np.float32)
    xG_k_ensemble = (xG_k_sum / n_valid).astype(np.float32)
    xh_k_ensemble = (xh_k_sum / n_valid).astype(np.float32)
    if measure_ww:
        ww_G_ensemble = (ww_G_sum / n_valid).astype(np.float32)
        ww_H_ensemble = (ww_H_sum / n_valid).astype(np.float32)
    Qs_configs = np.array(Qs_configs)

    Qs_ensemble = np.zeros(len(Y_arr))
    for iy in range(len(Y_arr)):
        Qs_ensemble[iy] = Qs_of_S(r_bins, S_ensemble[iy])

    out = {
        'r': r_bins,
        'r_xG': r_xG,
        'k_vals': k_vals,
        'Y': Y_arr,
        'S_ensemble': S_ensemble,
        'xG_ensemble': xG_ensemble,
        'xh_ensemble': xh_ensemble,
        'xG_k_ensemble': xG_k_ensemble,
        'xh_k_ensemble': xh_k_ensemble,
        'Qs_ensemble': Qs_ensemble,
        'Qs_configs': Qs_configs,
    }
    if measure_ww:
        out['k_ww'] = k_ww
        out['ww_G_ensemble'] = ww_G_ensemble
        out['ww_H_ensemble'] = ww_H_ensemble
    return out


def main():
    parser = argparse.ArgumentParser(
        description='Ensemble-averaged JIMWLK dipole evolution (GPU)')
    parser.add_argument('--N', type=int, default=512)
    parser.add_argument('--l', type=float, default=None,
                        help='Lattice size (default: 32)')
    parser.add_argument('--Ny', type=int, default=None,
                        help='Number of longitudinal color charge layers (overrides config default)')
    parser.add_argument('--mu2', type=float, default=None,
                        help='Color charge variance mu^2 (overrides default for chosen IC)')
    parser.add_argument('--m2', type=float, default=None,
                        help='IR regulator mass squared (overrides config default)')
    parser.add_argument('--Yf', type=float, default=1.0)
    parser.add_argument('--dY', type=float, default=0.01)
    parser.add_argument('--n_configs', type=int, default=1000)
    parser.add_argument('--measure_every', type=int, default=10,
                        help='Measure observables every N steps')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outdir', type=str, default='results_gpu')
    parser.add_argument('--ic', type=str, default='mv', choices=['mv', 'cn'],
                        help='Initial condition: mv (McLerran-Venugopalan) or cn (color-neutralized)')
    parser.add_argument('--derivative', type=str, default='forward',
                        choices=['forward', 'centered', 'spectral'],
                        help='Derivative method for gauge field extraction')
    parser.add_argument('--no-ww', action='store_true',
                        help='Skip WW TMD measurement (fundamental only)')
    args = parser.parse_args()
    measure_ww = not args.no_ww

    set_initial_condition(args.ic)

    if args.l is not None:
        cfg.l = args.l

    if args.mu2 is not None:
        cfg.mu2 = args.mu2
        if args.ic == 'cn':
            from .initial_conditions_cn import set_params, Q, beta, gamma
            set_params(args.mu2, Q, beta, gamma)

    cfg.N = args.N
    cfg.a = cfg.l / cfg.N
    cfg.a2 = cfg.a**2
    if args.Ny is not None:
        cfg.Ny = args.Ny
    if args.m2 is not None:
        cfg.m2 = args.m2
    cfg.variance_of_mv_noise = float(np.sqrt(cfg.mu2 / (cfg.Ny * cfg.a2)))
    cfg.Y_f = args.Yf
    cfg.dY = args.dY

    n_steps = int(args.Yf / args.dY)
    n_meas = n_steps // args.measure_every + 1

    dev = cp.cuda.Device()
    print("=" * 60)
    print("JIMWLK Ensemble — GPU")
    print("=" * 60)
    print(f"  GPU:           {dev.id} ({cp.cuda.runtime.getDeviceProperties(dev.id)['name'].decode()})")
    ic_label = 'McLerran-Venugopalan' if args.ic == 'mv' else 'Color-neutralized'
    print(f"  Initial cond:  {ic_label} ({args.ic})")
    print(f"  Lattice:       {cfg.N} x {cfg.N}  (a = {cfg.a:.4f}, l = {cfg.l})")
    print(f"  MV layers:     {cfg.Ny}")
    print(f"  Evolution:     Y = 0 -> {args.Yf}, dY = {args.dY} ({n_steps} steps)")
    print(f"  Measure every: {args.measure_every} steps ({n_meas} measurements)")
    print(f"  Configs:       {args.n_configs}")
    print(f"  Seed base:     {args.seed}")
    print(f"  Derivative:    {args.derivative}")
    print(f"  Output:        {args.outdir}/")
    print("=" * 60)

    print("\nRunning first config to estimate total time...")
    rng_test = cp.random.Generator(cp.random.XORWOW(seed=args.seed))
    t0 = time.time()
    _ = run_single_config(0, rng_test, args.Yf, args.dY, args.measure_every, derivative=args.derivative, measure_ww=measure_ww)
    cp.cuda.Stream.null.synchronize()
    t_one = time.time() - t0
    print(f"  1 config: {t_one:.1f}s")
    print(f"  Estimated total: {t_one * args.n_configs:.0f}s "
          f"({t_one * args.n_configs / 3600:.1f}h)")
    print()

    def progress(ic, total, dt, final_Qs):
        elapsed = time.time() - t_start
        eta = elapsed / (ic + 1) * (total - ic - 1)
        print(f"  config {ic+1:4d}/{total}  "
              f"({dt:.1f}s)  "
              f"Qs(Y_f)={final_Qs:.4f}  "
              f"ETA {eta/60:.0f}min")

    t_start = time.time()
    result = run_ensemble(
        n_configs=args.n_configs,
        Y_f=args.Yf,
        dY=args.dY,
        measure_interval=args.measure_every,
        seed_base=args.seed,
        progress_callback=progress,
        derivative=args.derivative,
        measure_ww=measure_ww,
    )
    t_total = time.time() - t_start

    print(f"\nDone in {t_total:.0f}s ({t_total/3600:.1f}h)")

    Qs_final = result['Qs_configs'][:, -1]
    print(f"\nQ_s(Y={args.Yf}):")
    print(f"  ensemble-averaged S(r): {result['Qs_ensemble'][-1]:.4f}")
    print(f"  per-config mean:        {np.mean(Qs_final):.4f} +/- {np.std(Qs_final)/np.sqrt(args.n_configs):.4f}")

    os.makedirs(args.outdir, exist_ok=True)

    import csv

    r = result['r']
    r_xG = result['r_xG']
    k_vals = result['k_vals']
    Y = result['Y']
    S = result['S_ensemble']
    xG = result['xG_ensemble']
    xh = result['xh_ensemble']
    xG_k = result['xG_k_ensemble']
    xh_k = result['xh_k_ensemble']
    Qs_ens = result['Qs_ensemble']
    Qs_cfg = result['Qs_configs']

    with open(os.path.join(args.outdir, 'S_ensemble.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        header = ['Y'] + [f'r={r[j]:.4f}' for j in range(len(r))]
        w.writerow(header)
        for iy in range(len(Y)):
            w.writerow([f'{Y[iy]:.4f}'] + [f'{S[iy, j]:.8f}' for j in range(len(r))])

    with open(os.path.join(args.outdir, 'Qs.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Y', 'Qs_ensemble'] + [f'config_{i}' for i in range(len(Qs_cfg))])
        for iy in range(len(Y)):
            w.writerow([f'{Y[iy]:.4f}', f'{Qs_ens[iy]:.6f}']
                       + [f'{Qs_cfg[ic, iy]:.6f}' for ic in range(len(Qs_cfg))])

    with open(os.path.join(args.outdir, 'xG_ensemble.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        header = ['Y'] + [f'r={r_xG[j]:.4f}' for j in range(len(r_xG))]
        w.writerow(header)
        for iy in range(len(Y)):
            w.writerow([f'{Y[iy]:.4f}'] + [f'{xG[iy, j]:.8f}' for j in range(len(r_xG))])

    with open(os.path.join(args.outdir, 'xh_ensemble.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        header = ['Y'] + [f'r={r_xG[j]:.4f}' for j in range(len(r_xG))]
        w.writerow(header)
        for iy in range(len(Y)):
            w.writerow([f'{Y[iy]:.4f}'] + [f'{xh[iy, j]:.8f}' for j in range(len(r_xG))])

    with open(os.path.join(args.outdir, 'xG_k_ensemble.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        header = ['Y'] + [f'k={k_vals[j]:.6f}' for j in range(len(k_vals))]
        w.writerow(header)
        for iy in range(len(Y)):
            w.writerow([f'{Y[iy]:.4f}'] + [f'{xG_k[iy, j]:.8f}' for j in range(len(k_vals))])

    with open(os.path.join(args.outdir, 'xh_k_ensemble.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        header = ['Y'] + [f'k={k_vals[j]:.6f}' for j in range(len(k_vals))]
        w.writerow(header)
        for iy in range(len(Y)):
            w.writerow([f'{Y[iy]:.4f}'] + [f'{xh_k[iy, j]:.8f}' for j in range(len(k_vals))])

    if measure_ww:
        k_ww = result['k_ww']
        ww_G = result['ww_G_ensemble']
        ww_H = result['ww_H_ensemble']

        with open(os.path.join(args.outdir, 'ww_G_ensemble.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            header = ['Y'] + [f'k={k_ww[j]:.6f}' for j in range(len(k_ww))]
            w.writerow(header)
            for iy in range(len(Y)):
                w.writerow([f'{Y[iy]:.4f}'] + [f'{ww_G[iy, j]:.8f}' for j in range(len(k_ww))])

        with open(os.path.join(args.outdir, 'ww_H_ensemble.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            header = ['Y'] + [f'k={k_ww[j]:.6f}' for j in range(len(k_ww))]
            w.writerow(header)
            for iy in range(len(Y)):
                w.writerow([f'{Y[iy]:.4f}'] + [f'{ww_H[iy, j]:.8f}' for j in range(len(k_ww))])

    with open(os.path.join(args.outdir, 'params.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['parameter', 'value'])
        for k, v in [('ic', args.ic), ('N', cfg.N), ('l', cfg.l), ('a', cfg.a),
                     ('Ny', cfg.Ny), ('mu2', cfg.mu2), ('m2', cfg.m2),
                     ('alpha_fc', cfg.alpha_fc), ('Y_f', args.Yf),
                     ('dY', args.dY), ('n_configs', args.n_configs),
                     ('measure_every', args.measure_every),
                     ('seed_base', args.seed), ('derivative', args.derivative),
                     ('backend', 'CuPy (GPU)'),
                     ('total_time_s', f'{t_total:.1f}')]:
            w.writerow([k, v])

    print(f"\nSaved to {args.outdir}/")
    for fname in ['S_ensemble.csv', 'xG_ensemble.csv', 'xh_ensemble.csv', 'Qs.csv', 'params.csv']:
        fpath = os.path.join(args.outdir, fname)
        size = os.path.getsize(fpath)
        print(f"  {fname:25s} {size/1024:.1f} KB")


if __name__ == '__main__':
    main()
