"""E3.4 Wall-clock comparison: per-step cost vs dimension.

Compare RK4 NH step + exact div (free), vs RK4 NH step + Hutchinson(k) trace
estimator (requires k backward passes through the RHS).

Plot: wall-clock per step vs d, log-log.
"""

import os, time, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

from _nh_core import (
    nh_tanh_f, rk4_step_nh, div_exact_step, trace_jac_hutch_step,
    make_iso_gaussian,
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

DIMS = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
BATCH = 256
N_STEPS_TIMED = 30
N_WARM = 5


def time_one(d, method, k=1):
    """Return mean seconds per step over N_STEPS_TIMED (median across 3 trials)."""
    V, grad_V = make_iso_gaussian(d)
    dt = 0.01
    trials = []
    for trial in range(3):
        torch.manual_seed(trial)
        q = torch.randn(BATCH, d)
        p = torch.randn(BATCH, d)
        xi = torch.zeros(BATCH, 1)
        # warm-up
        for _ in range(N_WARM):
            if method == 'exact':
                q_new, p_new, xi_new = rk4_step_nh(q, p, xi, grad_V, dt)
                _ = div_exact_step(xi, xi_new, d, dt)
                q, p, xi = q_new, p_new, xi_new
            else:
                _ = trace_jac_hutch_step(q, p, xi, grad_V, 1.0, 1.0, k)
                q, p, xi = rk4_step_nh(q, p, xi, grad_V, dt)
        # timed
        t0 = time.perf_counter()
        for _ in range(N_STEPS_TIMED):
            if method == 'exact':
                q_new, p_new, xi_new = rk4_step_nh(q, p, xi, grad_V, dt)
                _ = div_exact_step(xi, xi_new, d, dt)
                q, p, xi = q_new, p_new, xi_new
            else:
                _ = trace_jac_hutch_step(q, p, xi, grad_V, 1.0, 1.0, k)
                q, p, xi = rk4_step_nh(q, p, xi, grad_V, dt)
        t1 = time.perf_counter()
        trials.append((t1 - t0) / N_STEPS_TIMED)
    return float(np.median(trials))


def main():
    results = {'exact': {}, 'hutch1': {}, 'hutch5': {}}
    for d in DIMS:
        te = time_one(d, 'exact')
        t1 = time_one(d, 'hutch', 1)
        t5 = time_one(d, 'hutch', 5)
        results['exact'][d] = te
        results['hutch1'][d] = t1
        results['hutch5'][d] = t5
        print(f"  d={d:5d} exact={te*1e3:8.3f} ms  h1={t1*1e3:8.3f} ms  h5={t5*1e3:8.3f} ms")

    with open(os.path.join(RESDIR, 'e3_walltime.json'), 'w') as f:
        json.dump({k: {str(d): v for d, v in md.items()} for k, md in results.items()}, f, indent=2)

    # figure
    ds = np.array(DIMS)
    te = np.array([results['exact'][d] for d in DIMS]) * 1e3
    t1 = np.array([results['hutch1'][d] for d in DIMS]) * 1e3
    t5 = np.array([results['hutch5'][d] for d in DIMS]) * 1e3

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))
    ax.plot(ds, te, '-o', color='#1f77b4', lw=2, ms=6, label='NH-CNF (exact div)')
    ax.plot(ds, t1, '-s', color='#ff7f0e', lw=2, ms=6, label='FFJORD-style Hutchinson(1)')
    ax.plot(ds, t5, '-^', color='#9467bd', lw=2, ms=6, label='FFJORD-style Hutchinson(5)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('dimension d')
    ax.set_ylabel('wall-clock per step (ms)  batch=%d' % BATCH)
    ax.set_title('E3.4  Per-step cost: exact divergence vs Hutchinson trace')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left', frameon=True, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig_walltime.png'), bbox_inches='tight')
    fig.savefig(os.path.join(FIGDIR, 'fig_walltime.pdf'), bbox_inches='tight')
    print('saved fig_walltime')


if __name__ == '__main__':
    main()
