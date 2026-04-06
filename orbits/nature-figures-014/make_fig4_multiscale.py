#!/usr/bin/env python3
"""Figure 4: Multi-Scale Dynamics -- Broadband Thermostat.

2x3 panel layout:
  (a) Schematic: 3 thermostats at different Q coupled to system
  (b) Power spectrum of friction signal: single-Q vs multi-scale (broadband)
  (c) GMM trajectory overlay: single Log-Osc (stuck in 1-2 modes)
  (d) GMM trajectory overlay: MultiScale (visits all 5 modes)
  (e) KL convergence traces: NHC, single-LogOsc, MultiScale, NHCTail on GMM
  (f) Mode visitation histogram (fraction of samples in each of 5 modes)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch

from shared import (
    COLOR_NHC, COLOR_LO, COLOR_MS, COLOR_NHCT,
    FONTSIZE_LABEL, FONTSIZE_TICK, FONTSIZE_TITLE, FONTSIZE_ANNOT, DPI,
    g_func, SEEDS,
    LogOscThermostat, LogOscVelocityVerlet,
    MultiScaleLogOsc, MultiScaleLogOscVerlet,
    NHCTailThermostat, NHCTailVerlet,
    run_trajectory, compute_kl_trace, get_potential,
    SAMPLER_LABELS, SAMPLER_COLORS,
)
from research.eval.baselines import NoseHooverChain
from research.eval.integrators import VelocityVerletThermostat

N_EVALS = 1_000_000
SEED = 42
KT = 1.0


def mode_visitation(qs, centers, sigma=0.5):
    """Compute fraction of samples assigned to each mode."""
    n_modes = len(centers)
    counts = np.zeros(n_modes)
    for i, c in enumerate(centers):
        dist2 = np.sum((qs - c)**2, axis=1)
        counts[i] = np.sum(dist2 < (3 * sigma)**2)
    total = np.sum(counts)
    if total == 0:
        return np.zeros(n_modes)
    return counts / total


def make_figure():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    pot_gmm = get_potential('GMM')
    Qs = [0.1, 0.7, 10.0]

    # ── Run simulations ──
    print("Running single Log-Osc on GMM...")
    lo_result = run_trajectory(
        LogOscThermostat, LogOscVelocityVerlet, pot_gmm,
        dt=0.03, n_force_evals=N_EVALS, seed=SEED, collect_xi=True, Q=0.5
    )

    print("Running MultiScale on GMM...")
    ms_result = run_trajectory(
        MultiScaleLogOsc, MultiScaleLogOscVerlet, pot_gmm,
        dt=0.03, n_force_evals=N_EVALS, seed=SEED, collect_xi=True,
        Qs=Qs
    )

    print("Running NHC on GMM...")
    nhc_result = run_trajectory(
        NoseHooverChain, VelocityVerletThermostat, pot_gmm,
        dt=0.01, n_force_evals=N_EVALS, seed=SEED, collect_xi=True,
        chain_length=3, Q=1.0
    )

    print("Running NHCTail on GMM...")
    nhct_result = run_trajectory(
        NHCTailThermostat, NHCTailVerlet, pot_gmm,
        dt=0.03, n_force_evals=N_EVALS, seed=SEED, collect_xi=True,
        Qs=Qs, chain_length=2, Q_chain=1.0
    )

    # ── Panel (a): Schematic -- friction at multiple timescales ──
    ax = axes[0, 0]
    # Show g(xi_k) for each thermostat and their sum
    ms_xi = ms_result['xi']
    t_window = slice(5000, 12000)
    dt_ms = 0.03
    t = np.arange(ms_xi.shape[0])[t_window] * dt_ms

    colors_q = ['#e41a1c', '#377eb8', '#4daf4a']
    labels_q = [f'Q={Q}' for Q in Qs]

    g_total = np.zeros(len(t))
    for k in range(3):
        g_k = g_func(ms_xi[t_window, k])
        g_total += g_k
        ax.plot(t, g_k, color=colors_q[k], lw=0.8, alpha=0.7, label=labels_q[k])
    ax.plot(t, g_total, 'k-', lw=1.5, alpha=0.9, label='Total')

    ax.set_xlabel('Time', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'Friction $g(\xi_k)$', fontsize=FONTSIZE_LABEL)
    ax.legend(fontsize=9, loc='upper right', ncol=2)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.text(0.03, 0.95, panel_labels[0], transform=ax.transAxes,
            fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    # ── Panel (b): Power spectrum ──
    ax = axes[0, 1]

    # Single Log-Osc friction PSD
    lo_xi = lo_result['xi'][:, 0]
    burn = len(lo_xi) // 10
    g_single = g_func(lo_xi[burn:])
    f_s, psd_s = welch(g_single, fs=1.0/0.03, nperseg=4096)

    # Multi-scale total friction PSD
    ms_xi_all = ms_result['xi'][burn:]
    g_multi = sum(g_func(ms_xi_all[:, k]) for k in range(3))
    f_m, psd_m = welch(g_multi, fs=1.0/0.03, nperseg=4096)

    ax.loglog(f_s, psd_s, color=COLOR_LO, lw=2, alpha=0.8, label='Single Log-Osc')
    ax.loglog(f_m, psd_m, color=COLOR_MS, lw=2, alpha=0.8, label='MultiScale')

    ax.set_xlabel('Frequency', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('PSD of friction', fontsize=FONTSIZE_LABEL)
    ax.legend(fontsize=FONTSIZE_ANNOT)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.text(0.03, 0.93, panel_labels[1], transform=ax.transAxes,
            fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    # ── Panel (c): Single Log-Osc GMM trajectory ──
    ax = axes[0, 2]
    lo_q = lo_result['q']
    burn = len(lo_q) // 10
    lo_post = lo_q[burn:]
    thin = max(1, len(lo_post) // 30000)

    # True density contours
    _plot_gmm_contours(ax, pot_gmm)
    ax.scatter(lo_post[::thin, 0], lo_post[::thin, 1], s=0.5, alpha=0.2,
               color=COLOR_LO, rasterized=True)
    for c in pot_gmm.centers:
        ax.plot(c[0], c[1], 'k*', markersize=8, markeredgecolor='white',
                markeredgewidth=0.5)
    ax.set_xlim(-5.5, 5.5); ax.set_ylim(-5.5, 5.5)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$', fontsize=FONTSIZE_LABEL)
    ax.set_title('Single Log-Osc', fontsize=FONTSIZE_TITLE - 2, color=COLOR_LO)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.text(0.03, 0.95, panel_labels[2], transform=ax.transAxes,
            fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    # ── Panel (d): MultiScale GMM trajectory ──
    ax = axes[1, 0]
    ms_q = ms_result['q']
    ms_post = ms_q[burn:]
    thin = max(1, len(ms_post) // 30000)

    _plot_gmm_contours(ax, pot_gmm)
    ax.scatter(ms_post[::thin, 0], ms_post[::thin, 1], s=0.5, alpha=0.2,
               color=COLOR_MS, rasterized=True)
    for c in pot_gmm.centers:
        ax.plot(c[0], c[1], 'k*', markersize=8, markeredgecolor='white',
                markeredgewidth=0.5)
    ax.set_xlim(-5.5, 5.5); ax.set_ylim(-5.5, 5.5)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'$y$', fontsize=FONTSIZE_LABEL)
    ax.set_title('MultiScale', fontsize=FONTSIZE_TITLE - 2, color=COLOR_MS)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.text(0.03, 0.95, panel_labels[3], transform=ax.transAxes,
            fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    # ── Panel (e): KL convergence traces on GMM ──
    ax = axes[1, 1]
    configs = [
        ('NHC', nhc_result),
        ('LogOsc', lo_result),
        ('MultiScale', ms_result),
        ('NHCTail', nhct_result),
    ]
    for name, result in configs:
        evals, kl = compute_kl_trace(result['q'], pot_gmm)
        color = SAMPLER_COLORS.get(name, 'gray')
        label = SAMPLER_LABELS.get(name, name)
        ax.loglog(evals, kl, color=color, lw=2, label=label)

    ax.axhline(0.01, color='gray', ls='--', lw=1, alpha=0.7, label='KL=0.01')
    ax.set_xlabel('Force evaluations', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('KL divergence', fontsize=FONTSIZE_LABEL)
    ax.legend(fontsize=9, loc='upper right')
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.text(0.03, 0.95, panel_labels[4], transform=ax.transAxes,
            fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    # ── Panel (f): Mode visitation histogram ──
    ax = axes[1, 2]
    centers = pot_gmm.centers
    mode_labels = [f'Mode {i+1}' for i in range(len(centers))]

    mv_nhc = mode_visitation(nhc_result['q'][len(nhc_result['q'])//10:], centers)
    mv_lo = mode_visitation(lo_post, centers)
    mv_ms = mode_visitation(ms_post, centers)
    mv_nhct = mode_visitation(nhct_result['q'][len(nhct_result['q'])//10:], centers)

    x = np.arange(len(centers))
    width = 0.2
    ax.bar(x - 1.5*width, mv_nhc, width, color=COLOR_NHC, label='NHC', alpha=0.8)
    ax.bar(x - 0.5*width, mv_lo, width, color=COLOR_LO, label='Log-Osc', alpha=0.8)
    ax.bar(x + 0.5*width, mv_ms, width, color=COLOR_MS, label='MultiScale', alpha=0.8)
    ax.bar(x + 1.5*width, mv_nhct, width, color=COLOR_NHCT, label='NHCTail', alpha=0.8)

    ax.axhline(0.2, color='gray', ls='--', lw=1, alpha=0.5, label='Uniform')
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels, fontsize=FONTSIZE_TICK - 1)
    ax.set_ylabel('Fraction of samples', fontsize=FONTSIZE_LABEL)
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.text(0.03, 0.95, panel_labels[5], transform=ax.transAxes,
            fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    fig.suptitle('Multi-Scale Dynamics: Broadband Friction Enables Complete Mode Coverage',
                 fontsize=FONTSIZE_TITLE, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fig4_multiscale.png')
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def _plot_gmm_contours(ax, pot, n_grid=100):
    """Plot GMM true density contours."""
    xr = np.linspace(-5.5, 5.5, n_grid)
    yr = np.linspace(-5.5, 5.5, n_grid)
    XX, YY = np.meshgrid(xr, yr)
    ZZ = np.zeros_like(XX)
    for i in range(n_grid):
        for j in range(n_grid):
            ZZ[j, i] = -pot.energy(np.array([xr[i], yr[j]]))
    ZZ -= np.max(ZZ)
    ax.contour(XX, YY, np.exp(ZZ), levels=6, colors='gray', linewidths=0.5, alpha=0.5)


if __name__ == '__main__':
    make_figure()
