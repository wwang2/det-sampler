"""Figure 1: Landscape + Trajectory Overlays.

For each 2D system (DW, GMM, Rosenbrock): 1x3 panel showing
  (a) Energy landscape contour plot
  (b) NHC trajectory overlaid on landscape contours
  (c) NHCTail trajectory overlaid on landscape contours

For 1D HO: (q,p) phase space with trajectory overlay.
Total: 4 rows x 3 cols = 12-panel figure.
"""

import sys
import os
import importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORKTREE = '/Users/wujiewang/code/det-sampler/.worktrees/visual-validation-013'
sys.path.insert(0, WORKTREE)

from research.eval.potentials import (
    HarmonicOscillator1D, DoubleWell2D, GaussianMixture2D, Rosenbrock2D
)
from research.eval.integrators import ThermostatState, VelocityVerletThermostat
from research.eval.baselines import NoseHooverChain

# Load solution modules with hyphens in path
def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(WORKTREE, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

sol009 = _load_mod('sol009', 'orbits/multiscale-chain-009/solution.py')
MultiScaleNHCTail = sol009.MultiScaleNHCTail
MultiScaleNHCTailVerlet = sol009.MultiScaleNHCTailVerlet
MultiScaleLogOsc = sol009.MultiScaleLogOsc
MultiScaleLogOscVerlet = sol009.MultiScaleLogOscVerlet

FIGDIR = os.path.join(WORKTREE, 'orbits/visual-validation-013/figures')
os.makedirs(FIGDIR, exist_ok=True)

SEED = 42
N_FORCE_EVALS = 2_000_000

# (potential_cls, kwargs, nhc_dt, nhctail_dt, xlim, ylim, label)
SYSTEMS_2D = [
    (DoubleWell2D, {}, 0.01, 0.055, (-2.5, 2.5), (-3, 3), 'Double Well'),
    (GaussianMixture2D, {}, 0.01, 0.03, (-5.5, 5.5), (-5.5, 5.5), 'Gaussian Mixture'),
    (Rosenbrock2D, {}, 0.01, 0.03, (-3, 3), (-2, 6), 'Rosenbrock'),
]


def collect_trajectory(dynamics, potential, dt, n_force_evals, seed, integrator_cls=None):
    """Run sampler and collect trajectory points subsampled for plotting."""
    rng = np.random.default_rng(seed)
    q0 = rng.normal(0, 0.5, size=potential.dim)
    state = dynamics.initial_state(q0, rng=rng)

    if integrator_cls is None:
        integrator_cls = VelocityVerletThermostat
    integrator = integrator_cls(dynamics, potential, dt, kT=1.0, mass=1.0)

    qs = []
    ps = []
    store_every = 10  # ~100k points from 1M steps
    step = 0
    while state.n_force_evals < n_force_evals:
        state = integrator.step(state)
        step += 1
        if np.any(np.isnan(state.q)):
            break
        if step % store_every == 0:
            qs.append(state.q.copy())
            ps.append(state.p.copy())

    return np.array(qs), np.array(ps)


def compute_landscape(potential, xlim, ylim, n_grid=200):
    """Compute energy landscape on a grid."""
    x = np.linspace(xlim[0], xlim[1], n_grid)
    y = np.linspace(ylim[0], ylim[1], n_grid)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(n_grid):
        for j in range(n_grid):
            Z[i, j] = potential.energy(np.array([X[i, j], Y[i, j]]))
    return X, Y, Z


def compute_density(potential, xlim, ylim, kT=1.0, n_grid=200):
    """Compute Boltzmann density on a grid."""
    x = np.linspace(xlim[0], xlim[1], n_grid)
    y = np.linspace(ylim[0], ylim[1], n_grid)
    X, Y = np.meshgrid(x, y)
    logp = np.zeros_like(X)
    for i in range(n_grid):
        for j in range(n_grid):
            logp[i, j] = -potential.energy(np.array([X[i, j], Y[i, j]])) / kT
    logp -= logp.max()
    return X, Y, np.exp(logp)


def plot_2d_row(fig, axes, potential, label, xlim, ylim,
                traj_nhc_q, traj_nhctail_q):
    """Plot one row: landscape, NHC traj, NHCTail traj."""
    X, Y, Z = compute_landscape(potential, xlim, ylim, n_grid=250)

    # Clip for viz
    z_clip = np.percentile(Z, 92)
    Z_clip = np.clip(Z, Z.min(), z_clip)
    levels = np.linspace(Z.min(), z_clip, 30)

    for idx, ax in enumerate(axes):
        # Full opacity landscape for panel 0, lighter for trajectory panels
        alpha_fill = 0.9 if idx == 0 else 0.35
        cf = ax.contourf(X, Y, Z_clip, levels=levels, cmap='RdYlBu_r', alpha=alpha_fill)
        ax.contour(X, Y, Z_clip, levels=levels[::3], colors='gray',
                   linewidths=0.3, alpha=0.4 if idx == 0 else 0.25)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=10)

    axes[0].set_ylabel(label, fontsize=14, fontweight='bold')

    # NHC trajectory overlay -- use blue, higher alpha, thicker line
    if len(traj_nhc_q) > 0:
        axes[1].plot(traj_nhc_q[:, 0], traj_nhc_q[:, 1],
                     color='#1f77b4', alpha=0.25, linewidth=0.2, rasterized=True)

    # NHCTail trajectory overlay -- use green, higher alpha
    if len(traj_nhctail_q) > 0:
        axes[2].plot(traj_nhctail_q[:, 0], traj_nhctail_q[:, 1],
                     color='#2ca02c', alpha=0.25, linewidth=0.2, rasterized=True)


def plot_ho_row(fig, axes, traj_nhc_q, traj_nhc_p, traj_nhctail_q, traj_nhctail_p):
    """Plot 1D HO row in (q,p) phase space."""
    sigma_q = 1.0
    sigma_p = 1.0
    qlim = (-4.5, 4.5)
    plim = (-4.5, 4.5)

    q_grid = np.linspace(qlim[0], qlim[1], 200)
    p_grid = np.linspace(plim[0], plim[1], 200)
    Q, P = np.meshgrid(q_grid, p_grid)
    Z = Q**2 / (2 * sigma_q**2) + P**2 / (2 * sigma_p**2)
    levels = np.linspace(0, 8, 25)

    for idx, ax in enumerate(axes):
        alpha_fill = 0.9 if idx == 0 else 0.3
        ax.contourf(Q, P, Z, levels=levels, cmap='RdYlBu_r', alpha=alpha_fill)
        ax.contour(Q, P, Z, levels=[0.5, 2.0, 4.5], colors='gray',
                   linewidths=0.6, alpha=0.4, linestyles='--')
        ax.set_xlim(qlim)
        ax.set_ylim(plim)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=10)
        ax.set_xlabel('q', fontsize=12)

    axes[0].set_ylabel('Harmonic Osc.\n(q, p) phase space', fontsize=13, fontweight='bold')

    if len(traj_nhc_q) > 0:
        axes[1].plot(traj_nhc_q[:, 0], traj_nhc_p[:, 0],
                     color='#1f77b4', alpha=0.20, linewidth=0.2, rasterized=True)

    if len(traj_nhctail_q) > 0:
        axes[2].plot(traj_nhctail_q[:, 0], traj_nhctail_p[:, 0],
                     color='#2ca02c', alpha=0.20, linewidth=0.2, rasterized=True)


def main():
    print("=== Figure 1: Landscape + Trajectory Overlays ===")

    fig, axes = plt.subplots(4, 3, figsize=(14, 19), dpi=300)

    axes[0, 0].set_title('Energy Landscape', fontsize=14, fontweight='bold', pad=10)
    axes[0, 1].set_title('NHC (M=3, Q=1.0)', fontsize=14, fontweight='bold', pad=10)
    axes[0, 2].set_title('NHCTail (Qs=[0.1, 0.7, 10])', fontsize=14, fontweight='bold', pad=10)

    # --- 2D systems ---
    for row_idx, (pot_cls, pot_kwargs, nhc_dt, nhctail_dt, xlim, ylim, label) in enumerate(SYSTEMS_2D):
        print(f"\n--- {label} ---")
        potential = pot_cls(**pot_kwargs)

        print(f"  Running NHC (dt={nhc_dt})...")
        nhc = NoseHooverChain(dim=2, chain_length=3, kT=1.0, Q=1.0)
        traj_nhc_q, traj_nhc_p = collect_trajectory(
            nhc, potential, nhc_dt, N_FORCE_EVALS, SEED)
        print(f"  NHC: {len(traj_nhc_q)} points collected")

        print(f"  Running NHCTail (dt={nhctail_dt})...")
        nhctail = MultiScaleNHCTail(dim=2, Qs=[0.1, 0.7, 10.0], chain_length=2)
        traj_nhctail_q, traj_nhctail_p = collect_trajectory(
            nhctail, potential, nhctail_dt, N_FORCE_EVALS, SEED,
            integrator_cls=MultiScaleNHCTailVerlet)
        print(f"  NHCTail: {len(traj_nhctail_q)} points collected")

        plot_2d_row(fig, axes[row_idx], potential, label, xlim, ylim,
                    traj_nhc_q, traj_nhctail_q)

    # --- 1D HO ---
    print(f"\n--- Harmonic Oscillator ---")
    ho = HarmonicOscillator1D()
    ho_dt = 0.005

    print(f"  Running NHC (dt={ho_dt})...")
    nhc_ho = NoseHooverChain(dim=1, chain_length=3, kT=1.0, Q=1.0)
    traj_nhc_q, traj_nhc_p = collect_trajectory(nhc_ho, ho, ho_dt, N_FORCE_EVALS, SEED)
    print(f"  NHC: {len(traj_nhc_q)} points")

    print(f"  Running NHCTail (dt={ho_dt})...")
    nhctail_ho = MultiScaleNHCTail(dim=1, Qs=[0.1, 0.7, 10.0], chain_length=2)
    traj_nhctail_q, traj_nhctail_p = collect_trajectory(
        nhctail_ho, ho, ho_dt, N_FORCE_EVALS, SEED,
        integrator_cls=MultiScaleNHCTailVerlet)
    print(f"  NHCTail: {len(traj_nhctail_q)} points")

    plot_ho_row(fig, axes[3], traj_nhc_q, traj_nhc_p, traj_nhctail_q, traj_nhctail_p)

    # Label bottom row x-axes for 2D systems too
    for row in range(3):
        for col in range(3):
            axes[row, col].set_xlabel('x', fontsize=11)
        axes[row, 0].set_xlabel('')  # ylabel already present

    plt.tight_layout(h_pad=1.5, w_pad=0.8)

    outpath = os.path.join(FIGDIR, 'fig1_trajectory_overlays.png')
    fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nSaved: {outpath}")


if __name__ == '__main__':
    main()
