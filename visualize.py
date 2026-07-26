"""Visualize ensemble-averaged JIMWLK dipole S(r) and Q_s(Y).

Usage:
    python -m jimwlk_cuda.visualize --datadir results_gpu/
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
        n_configs = len(header) - 2
        Y_vals = []
        Qs_ens = []
        Qs_cfg = []
        for row in reader:
            Y_vals.append(float(row[0]))
            Qs_ens.append(float(row[1]))
            Qs_cfg.append([float(x) for x in row[2:]])
    return np.array(Y_vals), np.array(Qs_ens), np.array(Qs_cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='results_gpu')
    parser.add_argument('--save', action='store_true', help='Save figures instead of showing')
    parser.add_argument('--alpha_s', type=float, default=0.2,
                        help='Physical alpha_s for rescaling Y (Y_phys = Y_code / alpha_s)')
    args = parser.parse_args()
    alpha_s = args.alpha_s

    Y_vals, r_bins, S = load_S_ensemble(args.datadir)
    Y_qs, Qs_ens, Qs_cfg = load_Qs(args.datadir)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    Y_phys = Y_vals / alpha_s
    Y_qs_phys = Y_qs / alpha_s

    ax = axes[0]
    r_max = 15.0
    mask = (r_bins > 0) & (r_bins < r_max)
    for iy in range(len(Y_vals)):
        ax.plot(r_bins[mask], S[iy][mask], label=f'Y = {Y_phys[iy]:.1f}')
    ax.set_xlabel('r')
    ax.set_ylabel('S(r)')
    ax.set_title(r'Ensemble-averaged dipole S(r), $\alpha_s = $' + f'{alpha_s}')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if Qs_cfg.shape[1] > 0:
        Qs_mean = np.mean(Qs_cfg, axis=1)
        Qs_std = np.std(Qs_cfg, axis=1)
        Qs_err = Qs_std / np.sqrt(Qs_cfg.shape[1])
        ax.errorbar(Y_qs_phys, Qs_mean, yerr=Qs_err, fmt='ko-', markersize=5,
                    capsize=3, label=r'$\langle Q_s \rangle$ per-config')
        ax.fill_between(Y_qs_phys, Qs_mean - Qs_std, Qs_mean + Qs_std,
                         alpha=0.15, color='blue', label=r'$\pm 1\sigma$')
    else:
        ax.plot(Y_qs_phys, Qs_ens, 'ko-', markersize=5, label='ensemble avg')
    ax.set_xlabel('Y')
    ax.set_ylabel(r'$Q_s$')
    ax.set_title(r'$Q_s(Y)$, $\alpha_s = $' + f'{alpha_s}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if args.save:
        out = os.path.join(args.datadir, 'dipole_results.png')
        plt.savefig(out, dpi=150)
        print(f"Saved {out}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
