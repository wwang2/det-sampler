#!/usr/bin/env python3
"""Figure 1: The Problem -- Why Nose-Hoover Chain (M=3) Fails.

2x3 panel layout:
  Row 1: NHC phase portraits on 1D HO at Q=0.1, 0.5, 1.0 (KAM tori persist)
  Row 2: NHC sampled density vs true density on DW, GMM, Rosenbrock

Uses 1M force evals per run, NHC(M=3) as primary baseline.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from shared import (
    COLOR_NHC, FONTSIZE_LABEL, FONTSIZE_TICK, FONTSIZE_TITLE, FONTSIZE_ANNOT, DPI,
    run_trajectory, get_potential, _kl_1d, _kl_2d,
)
from research.eval.baselines import NoseHooverChain
from research.eval.integrators import VelocityVerletThermostat

N_EVALS = 1_000_000
SEED = 42


def run_nhc_ho(Q, n_evals=N_EVALS, seed=SEED):
    pot = get_potential('HO')
    return run_trajectory(
        NoseHooverChain, VelocityVerletThermostat, pot,
        dt=0.005, n_force_evals=n_evals, seed=seed,
        q0=np.array([1.0]), chain_length=3, Q=Q
    )


def run_nhc_2d(pot_name, n_evals=N_EVALS, seed=SEED):
    pot = get_potential(pot_name)
    return run_trajectory(
        NoseHooverChain, VelocityVerletThermostat, pot,
        dt=0.01, n_force_evals=n_evals, seed=seed,
        chain_length=3, Q=1.0
    )


def make_figure():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    # ── Row 1: Phase portraits on 1D HO at different Q ──
    Q_vals = [0.1, 0.5, 1.0]
    axis_lim = 4.0  # 4-sigma range

    for col, Q in enumerate(Q_vals):
        ax = axes[0, col]
        print(f"Running NHC on HO with Q={Q}...")
        result = run_nhc_ho(Q)
        qs = result['q'].ravel()
        ps = result['p'].ravel()

        # Thin for plotting
        thin = max(1, len(qs) // 50000)
        ax.scatter(qs[::thin], ps[::thin], s=0.15, alpha=0.3, color=COLOR_NHC,
                   rasterized=True)

        # Gaussian contours (1,2,3 sigma)
        theta = np.linspace(0, 2 * np.pi, 200)
        for sigma in [1, 2, 3]:
            ax.plot(sigma * np.cos(theta), sigma * np.sin(theta),
                    '--', color='gray', alpha=0.5, lw=0.8)

        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_aspect('equal')
        ax.set_xlabel(r'$q$', fontsize=FONTSIZE_LABEL)
        if col == 0:
            ax.set_ylabel(r'$p$', fontsize=FONTSIZE_LABEL)
        ax.set_title(f'Q = {Q}', fontsize=FONTSIZE_TITLE - 2)
        ax.tick_params(labelsize=FONTSIZE_TICK)
        ax.text(0.03, 0.95, panel_labels[col], transform=ax.transAxes,
                fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

        # Compute KL for annotation
        burn = len(qs) // 10
        kl = _kl_1d(qs[burn:], get_potential('HO'), 1.0)
        ax.text(0.97, 0.05, f'KL={kl:.3f}', transform=ax.transAxes,
                fontsize=FONTSIZE_ANNOT, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # ── Row 2: Density comparison on DW, GMM, RB ──
    pot_names = ['DW', 'GMM', 'RB']
    pot_titles = ['Double Well', 'GMM (5-mode)', 'Rosenbrock']

    for col, (pname, ptitle) in enumerate(zip(pot_names, pot_titles)):
        ax = axes[1, col]
        pot = get_potential(pname)
        print(f"Running NHC on {pname}...")
        result = run_nhc_2d(pname)
        qs = result['q']
        burn = len(qs) // 10
        qs_post = qs[burn:]

        # Compute KL
        kl = _kl_2d(qs_post, pot, 1.0)

        # True density as filled contour
        if pname == 'DW':
            xr = np.linspace(-2.5, 2.5, 150)
            yr = np.linspace(-3, 3, 150)
        elif pname == 'GMM':
            xr = np.linspace(-5.5, 5.5, 150)
            yr = np.linspace(-5.5, 5.5, 150)
        else:  # RB
            xr = np.linspace(-3, 3, 150)
            yr = np.linspace(-2, 6, 150)

        XX, YY = np.meshgrid(xr, yr)
        ZZ = np.zeros_like(XX)
        for i in range(len(xr)):
            for j in range(len(yr)):
                ZZ[j, i] = -pot.energy(np.array([xr[i], yr[j]]))
        ZZ -= np.max(ZZ)
        P_true = np.exp(ZZ)

        ax.contour(XX, YY, P_true, levels=8, colors='gray', linewidths=0.6, alpha=0.6)

        # Sampled density as scatter
        thin = max(1, len(qs_post) // 30000)
        ax.scatter(qs_post[::thin, 0], qs_post[::thin, 1], s=0.3, alpha=0.25,
                   color=COLOR_NHC, rasterized=True)

        ax.set_xlabel(r'$x$', fontsize=FONTSIZE_LABEL)
        if col == 0:
            ax.set_ylabel(r'$y$', fontsize=FONTSIZE_LABEL)
        ax.set_title(ptitle, fontsize=FONTSIZE_TITLE - 2)
        ax.tick_params(labelsize=FONTSIZE_TICK)
        ax.set_xlim(xr[0], xr[-1])
        ax.set_ylim(yr[0], yr[-1])

        ax.text(0.03, 0.95, panel_labels[3 + col], transform=ax.transAxes,
                fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')
        ax.text(0.97, 0.05, f'KL={kl:.3f}', transform=ax.transAxes,
                fontsize=FONTSIZE_ANNOT, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    fig.suptitle('Nose-Hoover Chain (M=3) fails on multi-modal and stiff systems',
                 fontsize=FONTSIZE_TITLE, fontweight='bold', y=0.98)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fig1_problem.png')
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == '__main__':
    make_figure()
