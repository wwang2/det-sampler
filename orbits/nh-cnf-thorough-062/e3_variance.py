"""E3.1 Variance scaling: variance of log p(x) estimate vs dimension.

For each (d, target family, method), run 10 seeds. For each seed, integrate
the NH-CNF on 100 test points, repeat 10 times. Report the across-trajectory
std of log p(x). Plot vs d with seed-variability error bars.

Methods:
  - NH exact divergence (zero variance)
  - Hutchinson(1)
  - Hutchinson(5)
  - Hutchinson(20)

Target families:
  - isotropic Gaussian
  - anisotropic Gaussian (kappas log-spaced [1, 100])
  - bimodal (+/- e_1)
"""

import os, time, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

from _nh_core import (
    run_nh_cnf_batch, make_iso_gaussian, make_aniso_gaussian, make_bimodal,
)

mpl.rcParams.update({
    'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 130, 'savefig.dpi': 220, 'savefig.pad_inches': 0.2,
})

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGDIR, exist_ok=True)
RESDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESDIR, exist_ok=True)

DIMS = [2, 5, 10, 20, 50, 100, 200]  # stopping at 200 for time budget
N_TEST = 32        # test points
N_REPEATS = 8      # independent ODE trajectories per test point (noise source)
N_SEEDS = 5        # for error bars
N_STEPS = 200      # integration steps
DT = 0.02

METHODS = [
    ('NH exact',       'exact', 0),
    ('Hutch(1)',       'hutch', 1),
    ('Hutch(5)',       'hutch', 5),
    ('Hutch(20)',      'hutch', 20),
]
COLORS = {
    'NH exact':  '#1f77b4',
    'Hutch(1)':  '#ff7f0e',
    'Hutch(5)':  '#9467bd',
    'Hutch(20)': '#8c564b',
}

TARGETS = {
    'iso':   (lambda d: make_iso_gaussian(d),       'Isotropic Gaussian'),
    'aniso': (lambda d: make_aniso_gaussian(d),     'Anisotropic Gaussian'),
    'bimod': (lambda d: make_bimodal(d),            'Bimodal'),
}


def measure_variance(d, target_maker, method_mode, hutch_k, seed):
    """Return the mean across-trajectory std of log p over test points."""
    V_fn, grad_V_fn = target_maker(d)
    # Use same initial points across repeats to measure method-induced variance
    torch.manual_seed(seed)
    x0 = torch.randn(N_TEST, d)  # test points (we reuse their init but noise differs)

    logp_matrix = np.zeros((N_REPEATS, N_TEST))
    for r in range(N_REPEATS):
        # Distinct seed per repeat to randomize any stochastic parts (hutch)
        _, logp = run_nh_cnf_batch(
            grad_V_fn, n_samples=N_TEST, d=d, n_steps=N_STEPS, dt=DT,
            seed=seed * 1000 + r, mode=method_mode, hutch_k=hutch_k,
        )
        logp_matrix[r] = logp
    # across-trajectory std for each test point, then averaged
    per_point_std = logp_matrix.std(axis=0, ddof=1)
    return float(per_point_std.mean())


def main():
    results = {}
    t0 = time.time()
    for target_key, (maker, _label) in TARGETS.items():
        results[target_key] = {}
        for method_name, mode, k in METHODS:
            results[target_key][method_name] = {}
            for d in DIMS:
                seed_vars = []
                for s in range(N_SEEDS):
                    v = measure_variance(d, maker, mode, k, s)
                    seed_vars.append(v)
                mean = float(np.mean(seed_vars))
                std = float(np.std(seed_vars, ddof=1))
                results[target_key][method_name][d] = (mean, std)
                print(f"  {target_key:5s} d={d:4d} {method_name:10s} var={mean:.3e} +/- {std:.1e}  ({time.time()-t0:.1f}s)")
    with open(os.path.join(RESDIR, 'e3_variance.json'), 'w') as f:
        json.dump({k: {m: {str(d): v for d, v in md.items()} for m, md in tm.items()}
                   for k, tm in results.items()}, f, indent=2)

    # --- figure: 1x3 panels, one per target family ---
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (target_key, (_maker, label)) in zip(axes, TARGETS.items()):
        for method_name, _mode, _k in METHODS:
            md = results[target_key][method_name]
            ds = np.array(DIMS)
            means = np.array([md[d][0] for d in DIMS])
            stds  = np.array([md[d][1] for d in DIMS])
            lo = np.maximum(means - stds, 1e-12)
            hi = means + stds
            ax.plot(ds, np.maximum(means, 1e-12), '-o', color=COLORS[method_name],
                    lw=1.8, ms=4.5, label=method_name)
            ax.fill_between(ds, lo, hi, alpha=0.18, color=COLORS[method_name])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('dimension d')
        ax.set_title(label)
        ax.grid(True, alpha=0.3, which='both')
    axes[0].set_ylabel('std[ log p(x) ]  across ODE trajectories')
    axes[-1].legend(loc='upper left', frameon=True, framealpha=0.95)
    fig.suptitle('E3.1  NH-CNF exact divergence vs Hutchinson — variance scaling', y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig_variance_scaling.png'), bbox_inches='tight')
    fig.savefig(os.path.join(FIGDIR, 'fig_variance_scaling.pdf'), bbox_inches='tight')
    print('saved fig_variance_scaling')


if __name__ == '__main__':
    main()
