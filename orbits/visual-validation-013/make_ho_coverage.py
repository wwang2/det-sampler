"""Figure 5: HO Phase Space Coverage Over Time.

2x4 panel showing phase space (q,p) snapshots at different force eval counts.
Top row: NHC at 10k, 50k, 200k, 1M force evals
Bottom row: NHCTail at same counts
Color points by time (early=blue, late=red).
Overlay true Gaussian contours.
"""

import sys
import os
import importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

WORKTREE = '/Users/wujiewang/code/det-sampler/.worktrees/visual-validation-013'
sys.path.insert(0, WORKTREE)

from research.eval.potentials import HarmonicOscillator1D
from research.eval.integrators import ThermostatState, VelocityVerletThermostat
from research.eval.baselines import NoseHooverChain

def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(WORKTREE, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

sol009 = _load_mod('sol009', 'orbits/multiscale-chain-009/solution.py')
MultiScaleNHCTail = sol009.MultiScaleNHCTail
MultiScaleNHCTailVerlet = sol009.MultiScaleNHCTailVerlet

FIGDIR = os.path.join(WORKTREE, 'orbits/visual-validation-013/figures')
os.makedirs(FIGDIR, exist_ok=True)

SEED = 42
CHECKPOINTS = [10_000, 50_000, 200_000, 1_000_000]
CHECKPOINT_LABELS = ['10k', '50k', '200k', '1M']


def collect_trajectory_checkpoints(dynamics, potential, dt, max_evals, seed,
                                    checkpoints, integrator_cls=None):
    """Collect (q,p) points and split by checkpoint force eval counts."""
    rng = np.random.default_rng(seed)
    q0 = rng.normal(0, 0.5, size=potential.dim)
    state = dynamics.initial_state(q0, rng=rng)

    if integrator_cls is None:
        integrator_cls = VelocityVerletThermostat
    integrator = integrator_cls(dynamics, potential, dt, kT=1.0, mass=1.0)

    all_q = []
    all_p = []
    all_evals = []
    step = 0
    store_every = 3

    while state.n_force_evals < max_evals:
        state = integrator.step(state)
        step += 1
        if np.any(np.isnan(state.q)):
            break
        if step % store_every == 0:
            all_q.append(state.q[0])
            all_p.append(state.p[0])
            all_evals.append(state.n_force_evals)

    all_q = np.array(all_q)
    all_p = np.array(all_p)
    all_evals = np.array(all_evals)

    # Split by checkpoint
    results = []
    for cp in checkpoints:
        mask = all_evals <= cp
        results.append((all_q[mask], all_p[mask], all_evals[mask]))

    return results


def main():
    print("=== Figure 5: HO Phase Space Coverage Over Time ===")

    ho = HarmonicOscillator1D()
    dt = 0.005

    print("  Running NHC...")
    nhc = NoseHooverChain(dim=1, chain_length=3, kT=1.0, Q=1.0)
    nhc_data = collect_trajectory_checkpoints(
        nhc, ho, dt, max(CHECKPOINTS), SEED, CHECKPOINTS)

    print("  Running NHCTail...")
    nhctail = MultiScaleNHCTail(dim=1, Qs=[0.1, 0.7, 10.0], chain_length=2)
    tail_data = collect_trajectory_checkpoints(
        nhctail, ho, dt, max(CHECKPOINTS), SEED, CHECKPOINTS,
        integrator_cls=MultiScaleNHCTailVerlet)

    # --- Plot ---
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=300)

    qlim = (-4.5, 4.5)
    plim = (-4.5, 4.5)

    # Gaussian contours
    q_grid = np.linspace(qlim[0], qlim[1], 100)
    p_grid = np.linspace(plim[0], plim[1], 100)
    Q, P = np.meshgrid(q_grid, p_grid)
    Z = Q**2 / 2 + P**2 / 2  # sigma_q = sigma_p = 1

    cmap_time = cm.coolwarm

    row_data = [nhc_data, tail_data]
    row_labels = ['NHC (M=3, Q=1.0)', 'NHCTail (Qs=[0.1, 0.7, 10])']

    for row_idx, (data_list, row_label) in enumerate(zip(row_data, row_labels)):
        for col_idx, (q_vals, p_vals, eval_vals) in enumerate(data_list):
            ax = axes[row_idx, col_idx]

            # Background: true Gaussian contours
            ax.contour(Q, P, Z, levels=[0.5, 2.0, 4.5, 8.0],
                       colors='gray', linewidths=0.8, alpha=0.5, linestyles='--')

            # Scatter points colored by time
            if len(q_vals) > 0:
                # Subsample for scatter (max 5000 points for readability)
                n_pts = len(q_vals)
                if n_pts > 5000:
                    idx = np.linspace(0, n_pts - 1, 5000, dtype=int)
                else:
                    idx = np.arange(n_pts)

                colors = np.linspace(0, 1, len(idx))
                ax.scatter(q_vals[idx], p_vals[idx], c=colors, cmap=cmap_time,
                          s=1.2, alpha=0.6, rasterized=True, edgecolors='none')

            ax.set_xlim(qlim)
            ax.set_ylim(plim)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=9)

            if row_idx == 0:
                ax.set_title(f'{CHECKPOINT_LABELS[col_idx]} force evals',
                            fontsize=13, fontweight='bold')
            if row_idx == 1:
                ax.set_xlabel('q', fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(f'{row_label}\np', fontsize=12, fontweight='bold')

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap_time, norm=Normalize(0, 1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.93, 0.25, 0.015, 0.5])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Relative time (early -> late)', fontsize=11)

    plt.subplots_adjust(left=0.06, right=0.91, top=0.94, bottom=0.06,
                        wspace=0.25, hspace=0.25)

    outpath = os.path.join(FIGDIR, 'fig5_ho_coverage.png')
    fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nSaved: {outpath}")


if __name__ == '__main__':
    main()
