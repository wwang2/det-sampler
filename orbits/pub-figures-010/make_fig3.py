#!/usr/bin/env python3
"""Figure 3: 'Trajectory on Landscape' -- Samplers exploring energy surfaces.

4-panel (2x2) layout:
  (a) 2D double-well contour + NH trajectory
  (b) 2D double-well contour + Log-Osc trajectory
  (c) 5-mode GMM contour + Multi-Scale Log-Osc trajectory
  (d) Rosenbrock banana contour + Log-Osc trajectory
"""

import sys, os, importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from research.eval.potentials import DoubleWell2D, GaussianMixture2D, Rosenbrock2D
from research.eval.baselines import NoseHoover
from research.eval.integrators import VelocityVerletThermostat

# Import Log-Osc
_spec = importlib.util.spec_from_file_location(
    "log_osc_001",
    os.path.join(os.path.dirname(__file__), '..', 'log-osc-001', 'solution.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LogOscThermostat = _mod.LogOscThermostat
LogOscVelocityVerlet = _mod.LogOscVelocityVerlet

# Import Multi-Scale Log-Osc
_spec2 = importlib.util.spec_from_file_location(
    "log_osc_multiT_005",
    os.path.join(os.path.dirname(__file__), '..', 'log-osc-multiT-005', 'solution.py'))
_mod2 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_mod2)
MultiScaleLogOsc = _mod2.MultiScaleLogOsc
MultiScaleLogOscVerlet = _mod2.MultiScaleLogOscVerlet

# ── Style constants ──
COLOR_NH = '#1f77b4'
tab10 = plt.cm.tab10
COLOR_LOGOSC = tab10(2)
COLOR_MSLO = tab10(4)
FONTSIZE_LABEL = 14
FONTSIZE_TICK = 12
FONTSIZE_TITLE = 16
DPI = 300

SEED = 42
KT = 1.0


def run_2d_sampler(SamplerClass, IntegratorClass, potential, dim=2, Q=1.0,
                   dt=0.01, n_steps=80000, seed=SEED, **kwargs):
    """Run a 2D sampler and return trajectory arrays."""
    rng = np.random.default_rng(seed)
    dyn = SamplerClass(dim=dim, kT=KT, mass=1.0, Q=Q, **kwargs)
    q0 = rng.normal(0, 0.5, size=dim)
    state = dyn.initial_state(q0, rng=rng)
    integrator = IntegratorClass(dyn, potential, dt=dt, kT=KT, mass=1.0)

    qs = np.empty((n_steps, dim))
    for i in range(n_steps):
        qs[i] = state.q[:dim]
        state = integrator.step(state)
    return qs


def run_2d_multiscale(potential, Qs=None, dt=0.03, n_steps=80000, seed=SEED):
    """Run MultiScale Log-Osc on a 2D potential."""
    rng = np.random.default_rng(seed)
    if Qs is None:
        Qs = [0.1, 1.0, 10.0]
    dyn = MultiScaleLogOsc(dim=2, kT=KT, mass=1.0, Qs=Qs)
    q0 = rng.normal(0, 0.5, size=2)
    state = dyn.initial_state(q0, rng=rng)
    integrator = MultiScaleLogOscVerlet(dyn, potential, dt=dt, kT=KT, mass=1.0)

    qs = np.empty((n_steps, 2))
    for i in range(n_steps):
        qs[i] = state.q[:2]
        state = integrator.step(state)
    return qs


def make_landscape(ax, potential, xlim, ylim, n_grid=150):
    """Create contourf landscape on an axis."""
    x = np.linspace(xlim[0], xlim[1], n_grid)
    y = np.linspace(ylim[0], ylim[1], n_grid)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(n_grid):
        for j in range(n_grid):
            Z[i, j] = potential.energy(np.array([X[i, j], Y[i, j]]))
    # Clip for visualization
    Z = np.clip(Z, Z.min(), np.percentile(Z, 95))
    cf = ax.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.55)
    ax.contour(X, Y, Z, levels=10, colors='k', alpha=0.15, linewidths=0.3)
    return cf


def overlay_trajectory(ax, qs, n_show=None, lw=0.3, alpha=0.5):
    """Overlay trajectory colored by time."""
    if n_show is None:
        n_show = len(qs)
    qs_show = qs[:n_show]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(qs_show)))
    for i in range(len(qs_show) - 1):
        ax.plot(qs_show[i:i+2, 0], qs_show[i:i+2, 1],
                color=colors[i], lw=lw, alpha=alpha)


def make_figure():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ── Panel (a): Double-well + NH ──
    ax = axes[0, 0]
    pot_dw = DoubleWell2D(barrier_height=1.0, y_stiffness=0.5)
    xlim, ylim = (-2.5, 2.5), (-3.5, 3.5)
    make_landscape(ax, pot_dw, xlim, ylim)
    print("Running NH on double-well...")
    qs_nh_dw = run_2d_sampler(NoseHoover, VelocityVerletThermostat, pot_dw,
                               Q=1.0, dt=0.01, n_steps=60000)
    overlay_trajectory(ax, qs_nh_dw, n_show=60000, lw=0.25)
    ax.set_xlabel(r'$x$', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'$y$', fontsize=FONTSIZE_LABEL)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.set_title('NH on Double-Well', fontsize=13, color=COLOR_NH)
    ax.text(0.03, 0.95, '(a)', transform=ax.transAxes, fontsize=FONTSIZE_LABEL,
            fontweight='bold', va='top', color='white')

    # ── Panel (b): Double-well + Log-Osc ──
    ax = axes[0, 1]
    make_landscape(ax, pot_dw, xlim, ylim)
    print("Running Log-Osc on double-well...")
    qs_lo_dw = run_2d_sampler(LogOscThermostat, LogOscVelocityVerlet, pot_dw,
                               Q=0.5, dt=0.01, n_steps=60000)
    overlay_trajectory(ax, qs_lo_dw, n_show=60000, lw=0.25)
    ax.set_xlabel(r'$x$', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'$y$', fontsize=FONTSIZE_LABEL)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.set_title('Log-Osc on Double-Well', fontsize=13, color=COLOR_LOGOSC)
    ax.text(0.03, 0.95, '(b)', transform=ax.transAxes, fontsize=FONTSIZE_LABEL,
            fontweight='bold', va='top', color='white')

    # ── Panel (c): GMM + Multi-Scale Log-Osc ──
    ax = axes[1, 0]
    pot_gmm = GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5)
    xlim_g, ylim_g = (-5, 5), (-5, 5)
    make_landscape(ax, pot_gmm, xlim_g, ylim_g)
    print("Running Multi-Scale Log-Osc on GMM...")
    qs_ms_gmm = run_2d_multiscale(pot_gmm, Qs=[0.1, 1.0, 10.0], dt=0.03, n_steps=80000)
    overlay_trajectory(ax, qs_ms_gmm, n_show=80000, lw=0.2, alpha=0.4)
    # Mark mode centers
    for c in pot_gmm.centers:
        ax.plot(c[0], c[1], 'w*', markersize=6, markeredgecolor='k', markeredgewidth=0.5)
    ax.set_xlabel(r'$x$', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'$y$', fontsize=FONTSIZE_LABEL)
    ax.set_xlim(*xlim_g); ax.set_ylim(*ylim_g)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.set_title('Multi-Scale Log-Osc on GMM', fontsize=13, color=COLOR_MSLO)
    ax.text(0.03, 0.95, '(c)', transform=ax.transAxes, fontsize=FONTSIZE_LABEL,
            fontweight='bold', va='top', color='white')

    # ── Panel (d): Rosenbrock + Log-Osc ──
    ax = axes[1, 1]
    pot_rb = Rosenbrock2D(a=0.0, b=5.0)
    xlim_r, ylim_r = (-2.5, 2.5), (-1, 4)
    make_landscape(ax, pot_rb, xlim_r, ylim_r)
    print("Running Log-Osc on Rosenbrock...")
    qs_lo_rb = run_2d_sampler(LogOscThermostat, LogOscVelocityVerlet, pot_rb,
                               Q=0.5, dt=0.008, n_steps=80000)
    overlay_trajectory(ax, qs_lo_rb, n_show=80000, lw=0.2, alpha=0.4)
    ax.set_xlabel(r'$x$', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'$y$', fontsize=FONTSIZE_LABEL)
    ax.set_xlim(*xlim_r); ax.set_ylim(*ylim_r)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.set_title('Log-Osc on Rosenbrock', fontsize=13, color=COLOR_LOGOSC)
    ax.text(0.03, 0.95, '(d)', transform=ax.transAxes, fontsize=FONTSIZE_LABEL,
            fontweight='bold', va='top', color='white')

    fig.suptitle('Trajectories on Energy Landscapes', fontsize=FONTSIZE_TITLE, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(os.path.dirname(__file__), 'figures', 'fig3_trajectories.png')
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    make_figure()
