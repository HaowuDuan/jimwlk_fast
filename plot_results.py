"""Plot ensemble-averaged JIMWLK results (works for both CPU and GPU output).

Usage:
    python -m jimwlk_cuda.plot_results --indir results_gpu/
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=False)
rc('font', size=12)


def plot_dipole_evolution(r, Y, S, outpath):
    fig, ax = plt.subplots(figsize=(8, 6))

    n_meas = len(Y)
    indices = np.linspace(0, n_meas - 1, min(8, n_meas)).astype(int)
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))

    for idx, color in zip(indices, colors):
        mask = r < 12.0
        ax.plot(r[mask], S[idx][mask], color=color,
                label=f'Y = {Y[idx]:.2f}', linewidth=1.5)

    ax.axhline(np.exp(-0.5), color='gray', linestyle='--', alpha=0.5,
               label=r'$S = e^{-1/2}$')

    ax.set_xlabel(r'$r$')
    ax.set_ylabel(r'$S(r)$')
    ax.set_title('Dipole amplitude — JIMWLK evolution (GPU)')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_Qs_evolution(Y, Qs_ensemble, Qs_configs, outpath):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    Qs_mean = np.mean(Qs_configs, axis=0)
    Qs_std = np.std(Qs_configs, axis=0)
    n_configs = Qs_configs.shape[0]
    Qs_err = Qs_std / np.sqrt(n_configs)

    ax1.plot(Y, Qs_ensemble, 'b-', linewidth=2, label=r'$Q_s$ from $\langle S(r) \rangle$')
    ax1.plot(Y, Qs_mean, 'r--', linewidth=1.5, label=r'$\langle Q_s \rangle$ per-config')
    ax1.fill_between(Y, Qs_mean - Qs_err, Qs_mean + Qs_err,
                     color='red', alpha=0.2, label=f'stat. error ({n_configs} configs)')

    ax1.set_xlabel(r'$Y$')
    ax1.set_ylabel(r'$Q_s$')
    ax1.set_title(r'Saturation scale $Q_s(Y)$')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    Qs0 = Qs_ensemble[0]
    ax2.plot(Y, Qs_ensemble / Qs0, 'b-', linewidth=2,
             label=r'$Q_s(Y)/Q_s(0)$ from $\langle S \rangle$')
    ax2.plot(Y, Qs_mean / Qs_mean[0], 'r--', linewidth=1.5,
             label=r'$\langle Q_s(Y) \rangle / \langle Q_s(0) \rangle$')

    ax2.set_xlabel(r'$Y$')
    ax2.set_ylabel(r'$Q_s(Y) / Q_s(0)$')
    ax2.set_title(r'Normalized $Q_s$ growth')
    ax2.set_yscale('log')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")


def main():
    parser = argparse.ArgumentParser(description='Plot JIMWLK ensemble results')
    parser.add_argument('--indir', type=str, default='results_gpu')
    args = parser.parse_args()

    d = args.indir
    r = np.load(os.path.join(d, 'r_bins.npy'))
    Y = np.load(os.path.join(d, 'Y_values.npy'))
    S = np.load(os.path.join(d, 'S_ensemble.npy'))
    Qs_ens = np.load(os.path.join(d, 'Qs_ensemble.npy'))
    Qs_cfg = np.load(os.path.join(d, 'Qs_per_config.npy'))

    print(f"Loaded from {d}/")
    print(f"  {len(Y)} rapidity points, {Qs_cfg.shape[0]} configs, {len(r)} r-bins")

    plot_dipole_evolution(r, Y, S, os.path.join(d, 'dipole_evolution.pdf'))
    plot_Qs_evolution(Y, Qs_ens, Qs_cfg, os.path.join(d, 'Qs_evolution.pdf'))


if __name__ == '__main__':
    main()
