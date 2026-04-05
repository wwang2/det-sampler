#!/usr/bin/env python3
"""Figure 5: 'Multi-Scale Mechanism' -- Why multiple timescales help.

4-panel (2x2) layout:
  (a) Schematic: 3 thermostat oscillators summing to create complex friction
  (b) Time series of xi_fast, xi_medium, xi_slow
  (c) Power spectral density: single-Q vs multi-scale
  (d) GMM samples: single Log-Osc vs Multi-Scale (mode coverage)
"""

import sys, os, importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from research.eval.potentials import GaussianMixture2D
from research.eval.integrators import ThermostatState

# Import Log-Osc
_base = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "log_osc_001", os.path.join(_base, '..', 'log-osc-001', 'solution.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LogOscThermostat = _mod.LogOscThermostat
LogOscVelocityVerlet = _mod.LogOscVelocityVerlet

# Import Multi-Scale Log-Osc
_spec2 = importlib.util.spec_from_file_location(
    "log_osc_multiT_005", os.path.join(_base, '..', 'log-osc-multiT-005', 'solution.py'))
_mod2 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_mod2)
MultiScaleLogOsc = _mod2.MultiScaleLogOsc
MultiScaleLogOscVerlet = _mod2.MultiScaleLogOscVerlet

# ── Style constants ──
tab10 = plt.cm.tab10
COLOR_LOGOSC = tab10(2)
COLOR_MSLO = tab10(4)
COLOR_FAST = '#e41a1c'
COLOR_MED = '#377eb8'
COLOR_SLOW = '#4daf4a'
FONTSIZE_LABEL = 14
FONTSIZE_TICK = 12
FONTSIZE_TITLE = 16
DPI = 300

SEED = 42
KT = 1.0


def g_func(xi):
    return 2.0 * xi / (1.0 + xi**2)


def run_multiscale_with_xi(potential, Qs, dt, n_steps, seed=SEED):
    """Run Multi-Scale Log-Osc and return (q, p, xi_all) trajectories."""
    rng = np.random.default_rng(seed)
    dyn = MultiScaleLogOsc(dim=2, kT=KT, mass=1.0, Qs=Qs)
    q0 = rng.normal(0, 0.5, size=2)
    state = dyn.initial_state(q0, rng=rng)
    integrator = MultiScaleLogOscVerlet(dyn, potential, dt=dt, kT=KT, mass=1.0)

    qs = np.empty((n_steps, 2))
    xis = np.empty((n_steps, len(Qs)))

    for i in range(n_steps):
        qs[i] = state.q[:2]
        xis[i] = state.xi
        state = integrator.step(state)

    return qs, xis


def run_single_logosc(potential, Q, dt, n_steps, seed=SEED):
    """Run single Log-Osc and return (q, xi) trajectories."""
    rng = np.random.default_rng(seed)
    dyn = LogOscThermostat(dim=2, kT=KT, mass=1.0, Q=Q)
    q0 = rng.normal(0, 0.5, size=2)
    state = dyn.initial_state(q0, rng=rng)
    integrator = LogOscVelocityVerlet(dyn, potential, dt=dt, kT=KT, mass=1.0)

    qs = np.empty((n_steps, 2))
    xis = np.empty(n_steps)

    for i in range(n_steps):
        qs[i] = state.q[:2]
        xis[i] = state.xi[0]
        state = integrator.step(state)

    return qs, xis


def make_figure():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    pot_gmm = GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5)
    Qs = [0.1, 1.0, 10.0]
    n_steps = 200000

    print("Running Multi-Scale Log-Osc...")
    qs_ms, xis_ms = run_multiscale_with_xi(pot_gmm, Qs=Qs, dt=0.03, n_steps=n_steps)
    print("Running single Log-Osc...")
    qs_single, xi_single = run_single_logosc(pot_gmm, Q=1.0, dt=0.01, n_steps=n_steps)

    # ── Panel (a): Schematic -- friction signals at different timescales ──
    ax = axes[0, 0]
    # Show g(xi_k) for each thermostat and their sum over a short window
    t_window = slice(10000, 14000)
    t = np.arange(n_steps)[t_window] * 0.03

    g_fast = g_func(xis_ms[t_window, 0])
    g_med = g_func(xis_ms[t_window, 1])
    g_slow = g_func(xis_ms[t_window, 2])
    g_total = g_fast + g_med + g_slow

    ax.plot(t, g_fast, color=COLOR_FAST, lw=1.0, alpha=0.8, label=f'$g(\\xi_{{fast}})$, Q={Qs[0]}')
    ax.plot(t, g_med, color=COLOR_MED, lw=1.0, alpha=0.8, label=f'$g(\\xi_{{med}})$, Q={Qs[1]}')
    ax.plot(t, g_slow, color=COLOR_SLOW, lw=1.0, alpha=0.8, label=f'$g(\\xi_{{slow}})$, Q={Qs[2]}')
    ax.plot(t, g_total, 'k-', lw=1.8, alpha=0.9, label='Total friction')
    ax.set_xlabel('Time $t$', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Friction $g(\\xi)$', fontsize=FONTSIZE_LABEL)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.legend(fontsize=9, loc='upper right', ncol=2)
    ax.text(0.03, 0.93, '(a)', transform=ax.transAxes, fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    # ── Panel (b): xi time series showing different frequencies ──
    ax = axes[0, 1]
    t_window2 = slice(10000, 30000)
    t2 = np.arange(n_steps)[t_window2] * 0.03

    ax.plot(t2, xis_ms[t_window2, 0], color=COLOR_FAST, lw=0.5, alpha=0.7,
            label=f'$\\xi_{{fast}}$ (Q={Qs[0]})')
    ax.plot(t2, xis_ms[t_window2, 1], color=COLOR_MED, lw=0.5, alpha=0.7,
            label=f'$\\xi_{{med}}$ (Q={Qs[1]})')
    ax.plot(t2, xis_ms[t_window2, 2], color=COLOR_SLOW, lw=0.8, alpha=0.8,
            label=f'$\\xi_{{slow}}$ (Q={Qs[2]})')
    ax.set_xlabel('Time $t$', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'$\xi_k(t)$', fontsize=FONTSIZE_LABEL)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.legend(fontsize=9, loc='upper right')
    ax.text(0.03, 0.93, '(b)', transform=ax.transAxes, fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    # ── Panel (c): Power spectral density ──
    ax = axes[1, 0]
    # Compute PSD of friction signal for single vs multi-scale
    from scipy.signal import welch

    # Single Log-Osc friction
    g_single = g_func(xi_single[10000:])  # skip burn-in
    dt_single = 0.01
    f_s, psd_s = welch(g_single, fs=1.0/dt_single, nperseg=8192)

    # Multi-scale total friction
    g_multi = g_func(xis_ms[10000:, 0]) + g_func(xis_ms[10000:, 1]) + g_func(xis_ms[10000:, 2])
    dt_multi = 0.03
    f_m, psd_m = welch(g_multi, fs=1.0/dt_multi, nperseg=8192)

    ax.loglog(f_s, psd_s, color=COLOR_LOGOSC, lw=2, alpha=0.8, label='Single Log-Osc (Q=1)')
    ax.loglog(f_m, psd_m, color=COLOR_MSLO, lw=2, alpha=0.8, label='Multi-Scale')

    # Reference 1/f line
    f_ref = np.logspace(-1, 1, 50)
    ax.loglog(f_ref, 0.5 * f_ref**(-1), 'k:', lw=1, alpha=0.4, label='$1/f$ reference')

    ax.set_xlabel('Frequency', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('PSD of friction signal', fontsize=FONTSIZE_LABEL)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.legend(fontsize=10, loc='best')
    ax.text(0.03, 0.93, '(c)', transform=ax.transAxes, fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    # ── Panel (d): GMM samples comparison ──
    ax = axes[1, 1]
    burn = n_steps // 10
    # Single Log-Osc samples
    ax.scatter(qs_single[burn::10, 0], qs_single[burn::10, 1],
               s=1, alpha=0.25, color=COLOR_LOGOSC, label='Single Log-Osc', rasterized=True)
    # Multi-Scale samples
    ax.scatter(qs_ms[burn::10, 0], qs_ms[burn::10, 1],
               s=1, alpha=0.25, color=COLOR_MSLO, label='Multi-Scale', rasterized=True)
    # Mark mode centers
    for c in pot_gmm.centers:
        ax.plot(c[0], c[1], 'k*', markersize=10, markeredgecolor='white', markeredgewidth=0.5)

    ax.set_xlabel(r'$x$', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'$y$', fontsize=FONTSIZE_LABEL)
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.legend(fontsize=10, loc='upper right', markerscale=5)
    ax.text(0.03, 0.93, '(d)', transform=ax.transAxes, fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    fig.suptitle('Multi-Scale Mechanism: Broadband Friction Enables Mode Hopping',
                 fontsize=FONTSIZE_TITLE, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(os.path.dirname(__file__), 'figures', 'fig5_multiscale.png')
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    make_figure()
