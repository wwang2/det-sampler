#!/usr/bin/env python3
"""Figure 1: 'The Problem' -- Why Nose-Hoover fails on the harmonic oscillator.

4-panel (2x2) layout:
  (a) Energy landscape of 1D HO with NH trajectory overlay (colored by time)
  (b) Phase space (q,p) of NH showing torus trapping
  (c) Position marginal P(q) of NH vs analytical N(0,1)
  (d) xi(t) trajectory of NH showing quasi-periodic oscillation
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from research.eval.potentials import HarmonicOscillator1D
from research.eval.baselines import NoseHoover
from research.eval.integrators import VelocityVerletThermostat

# ── Style constants (from research/style.md) ──
COLOR_NH = '#1f77b4'
FONTSIZE_LABEL = 14
FONTSIZE_TICK = 12
FONTSIZE_TITLE = 16
DPI = 300

# ── Simulation parameters ──
SEED = 42
N_STEPS = 500_000   # enough to show torus structure clearly
DT = 0.01
Q = 1.0
KT = 1.0

def run_nh_trajectory():
    """Run NH on 1D HO and collect full trajectory."""
    rng = np.random.default_rng(SEED)
    pot = HarmonicOscillator1D(omega=1.0)
    dyn = NoseHoover(dim=1, kT=KT, mass=1.0, Q=Q)
    q0 = np.array([1.0])  # start off-center
    state = dyn.initial_state(q0, rng=rng)
    integrator = VelocityVerletThermostat(dyn, pot, dt=DT, kT=KT, mass=1.0)

    qs = np.empty(N_STEPS)
    ps = np.empty(N_STEPS)
    xis = np.empty(N_STEPS)

    for i in range(N_STEPS):
        qs[i] = state.q[0]
        ps[i] = state.p[0]
        xis[i] = state.xi[0]
        state = integrator.step(state)

    return qs, ps, xis


def make_figure():
    qs, ps, xis = run_nh_trajectory()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Nose-Hoover Fails on the Harmonic Oscillator', fontsize=FONTSIZE_TITLE, fontweight='bold', y=0.97)

    # ── Panel (a): Energy landscape with trajectory ──
    ax = axes[0, 0]
    q_range = np.linspace(-3.5, 3.5, 300)
    U = 0.5 * q_range**2
    ax.fill_between(q_range, U, alpha=0.15, color='gray')
    ax.plot(q_range, U, 'k-', lw=1.5, label=r'$U(q) = \frac{1}{2}q^2$')

    # Overlay first 5000 steps colored by time
    n_show = 5000
    t_colors = np.linspace(0, 1, n_show)
    cmap = plt.cm.coolwarm
    for i in range(n_show - 1):
        ax.plot([qs[i], qs[i+1]], [0.5*qs[i]**2, 0.5*qs[i+1]**2],
                color=cmap(t_colors[i]), lw=0.5, alpha=0.7)
    ax.set_xlabel(r'$q$', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'$U(q)$', fontsize=FONTSIZE_LABEL)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-0.2, 6)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.legend(fontsize=11, loc='upper right')
    ax.text(0.03, 0.93, '(a)', transform=ax.transAxes, fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    # ── Panel (b): Phase space (q, p) showing torus ──
    ax = axes[0, 1]
    # Thin to avoid overplotting -- show every 5th point
    thin = 5
    n_phase = min(len(qs), N_STEPS)
    idx = np.arange(0, n_phase, thin)
    scatter = ax.scatter(qs[idx], ps[idx], c=np.arange(len(idx)), cmap='coolwarm',
                         s=0.3, alpha=0.4, rasterized=True)

    # Overlay expected Gaussian contours
    theta = np.linspace(0, 2*np.pi, 200)
    for sigma_mult in [1, 2, 3]:
        ax.plot(sigma_mult * np.cos(theta), sigma_mult * np.sin(theta),
                '--', color='gray', alpha=0.4, lw=0.8)
    ax.set_xlabel(r'$q$', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'$p$', fontsize=FONTSIZE_LABEL)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.text(0.03, 0.93, '(b)', transform=ax.transAxes, fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    # ── Panel (c): Position marginal P(q) ──
    ax = axes[1, 0]
    # Burn in
    burn = N_STEPS // 10
    q_post = qs[burn:]
    ax.hist(q_post, bins=100, density=True, alpha=0.6, color=COLOR_NH, label='NH samples')
    q_theory = np.linspace(-4, 4, 300)
    pdf_theory = (1.0 / np.sqrt(2*np.pi)) * np.exp(-0.5 * q_theory**2)
    ax.plot(q_theory, pdf_theory, 'k--', lw=2, label=r'Analytical $\mathcal{N}(0,1)$')
    ax.set_xlabel(r'$q$', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'$P(q)$', fontsize=FONTSIZE_LABEL)
    ax.set_xlim(-4, 4)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.legend(fontsize=11, loc='upper right')
    ax.text(0.03, 0.93, '(c)', transform=ax.transAxes, fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    # ── Panel (d): xi(t) trajectory ──
    ax = axes[1, 1]
    t = np.arange(N_STEPS) * DT
    # Show a representative window
    t_start, t_end = 0, 50000
    ax.plot(t[t_start:t_end], xis[t_start:t_end], color=COLOR_NH, lw=0.4, alpha=0.8)
    ax.set_xlabel(r'Time $t$', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'$\xi(t)$', fontsize=FONTSIZE_LABEL)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.text(0.03, 0.93, '(d)', transform=ax.transAxes, fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(os.path.dirname(__file__), 'figures', 'fig1_problem.png')
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    make_figure()
