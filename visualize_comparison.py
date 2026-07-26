"""Compare MV vs CN ensemble-averaged JIMWLK results side by side.

Usage:
    python -m jimwlk_cuda.visualize_comparison \
        --mv results_gpu_mv --cn results_gpu_cn --alpha_s 0.04 --save
"""

import argparse
import csv
import os
import numpy as np
import matplotlib.pyplot as plt


def load_S_ensemble(datadir):
    path = os.path.join(datadir, 'S_ensemble.csv')
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        r_bins = np.array([float(h.split('=')[1]) for h in header[1:]])
        Y_vals = []
        S_data = []
        for row in reader:
            Y_vals.append(float(row[0]))
            S_data.append([float(x) for x in row[1:]])
    return np.array(Y_vals), r_bins, np.array(S_data)


def load_Qs(datadir):
    path = os.path.join(datadir, 'Qs.csv')
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        Y_vals = []
        Qs_ens = []
        Qs_cfg = []
        for row in reader:
            Y_vals.append(float(row[0]))
            Qs_ens.append(float(row[1]))
            Qs_cfg.append([float(x) for x in row[2:]])
    return np.array(Y_vals), np.array(Qs_ens), np.array(Qs_cfg)


def main():
    parser = argparse.ArgumentParser(description='Compare MV vs CN JIMWLK results')
    parser.add_argument('--mv', type=str, default='results_gpu_mv')
    parser.add_argument('--cn', type=str, default='results_gpu_cn')
    parser.add_argument('--alpha_s', type=float, default=0.2)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--outdir', type=str, default='.')
    args = parser.parse_args()
    alpha_s = args.alpha_s

    Y_mv, r_mv, S_mv = load_S_ensemble(args.mv)
    Y_cn, r_cn, S_cn = load_S_ensemble(args.cn)
    Y_qs_mv, Qs_ens_mv, Qs_cfg_mv = load_Qs(args.mv)
    Y_qs_cn, Qs_ens_cn, Qs_cfg_cn = load_Qs(args.cn)

    Y_mv_phys = Y_mv / alpha_s
    Y_cn_phys = Y_cn / alpha_s
    Y_qs_mv_phys = Y_qs_mv / alpha_s
    Y_qs_cn_phys = Y_qs_cn / alpha_s

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: -ln S(r) vs r^2 at all Y for both ICs
    ax = axes[0]
    r2_max = 25.0
    ln_max = 5.0
    mask_mv = (r_mv**2 < r2_max) & (r_mv > 0)
    mask_cn = (r_cn**2 < r2_max) & (r_cn > 0)
    colors_mv = plt.cm.Blues(np.linspace(0.35, 1.0, len(Y_mv)))
    colors_cn = plt.cm.Reds(np.linspace(0.35, 1.0, len(Y_cn)))
    for iy in range(len(Y_mv)):
        neg_ln = -np.log(np.maximum(S_mv[iy][mask_mv], 1e-6))
        ax.plot(r_mv[mask_mv]**2, neg_ln, '-', color=colors_mv[iy], linewidth=1.5,
                label=f'MV Y={Y_mv_phys[iy]:.1f}')
    for iy in range(len(Y_cn)):
        neg_ln = -np.log(np.maximum(S_cn[iy][mask_cn], 1e-6))
        ax.plot(r_cn[mask_cn]**2, neg_ln, '--', color=colors_cn[iy], linewidth=1.5,
                label=f'CN Y={Y_cn_phys[iy]:.1f}')
    ax.axhline(0.5, color='gray', ls=':', alpha=0.6, label=r'$-\ln S = 1/2$')
    ax.set_xlabel(r'$r^2$')
    ax.set_ylabel(r'$-\ln S(r)$')
    ax.set_title(r'$-\ln S$ vs $r^2$: MV (solid) vs CN (dashed)')
    ax.set_ylim(-0.2, ln_max)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 2: Q_s(Y) comparison
    ax = axes[1]
    Qs_mean_mv = np.mean(Qs_cfg_mv, axis=1)
    Qs_err_mv = np.std(Qs_cfg_mv, axis=1) / np.sqrt(Qs_cfg_mv.shape[1])
    Qs_mean_cn = np.mean(Qs_cfg_cn, axis=1)
    Qs_err_cn = np.std(Qs_cfg_cn, axis=1) / np.sqrt(Qs_cfg_cn.shape[1])

    ax.errorbar(Y_qs_mv_phys, Qs_mean_mv, yerr=Qs_err_mv, fmt='bo-', markersize=4,
                capsize=3, label='MV')
    ax.errorbar(Y_qs_cn_phys, Qs_mean_cn, yerr=Qs_err_cn, fmt='rs-', markersize=4,
                capsize=3, label='CN')
    ax.set_xlabel(r'$Y_{\rm phys}$')
    ax.set_ylabel(r'$Q_s$')
    ax.set_title(r'$Q_s(Y)$: MV vs CN ($\alpha_s = ' + f'{alpha_s}$)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Q_s(Y)/Q_s(0) growth rate
    ax = axes[2]
    ax.plot(Y_qs_mv_phys, Qs_mean_mv / Qs_mean_mv[0], 'bo-', markersize=4,
            label='MV')
    ax.plot(Y_qs_cn_phys, Qs_mean_cn / Qs_mean_cn[0], 'rs-', markersize=4,
            label='CN')
    ax.set_xlabel(r'$Y_{\rm phys}$')
    ax.set_ylabel(r'$Q_s(Y) / Q_s(0)$')
    ax.set_title(r'Normalized $Q_s$ growth')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if args.save:
        out = os.path.join(args.outdir, 'mv_vs_cn_comparison.png')
        plt.savefig(out, dpi=150)
        print(f"Saved {out}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
