"""Figure 2: Sampled Density vs True Density.

For each system: 1x3 panel: (a) True density, (b) NHC sampled density, (c) NHCTail sampled density.
Uses 2D histogram (100x100 bins) for 2D systems, 1D histogram for HO.
Same colormap and color scale across all 3 panels per row.
Annotated with KL divergence in corner of sampled panels.
4 rows x 3 cols = 12-panel figure.
"""

import sys
import os
import importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

WORKTREE = '/Users/wujiewang/code/det-sampler/.worktrees/visual-validation-013'
sys.path.insert(0, WORKTREE)

from research.eval.potentials import (
    HarmonicOscillator1D, DoubleWell2D, GaussianMixture2D, Rosenbrock2D
)
from research.eval.integrators import ThermostatState, VelocityVerletThermostat
from research.eval.baselines import NoseHooverChain
from research.eval.evaluator import kl_divergence_histogram

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
N_FORCE_EVALS = 2_000_000

SYSTEMS_2D = [
    (DoubleWell2D, {}, 0.01, 0.055, (-2.5, 2.5), (-3, 3), 'Double Well'),
    (GaussianMixture2D, {}, 0.01, 0.03, (-5.5, 5.5), (-5.5, 5.5), 'Gaussian Mixture'),
    (Rosenbrock2D, {}, 0.01, 0.03, (-3, 3), (-2, 6), 'Rosenbrock'),
]


def run_and_collect(dynamics, potential, dt, n_force_evals, seed, integrator_cls=None):
    """Run sampler and collect post-burnin samples."""
    rng = np.random.default_rng(seed)
    q0 = rng.normal(0, 0.5, size=potential.dim)
    state = dynamics.initial_state(q0, rng=rng)

    if integrator_cls is None:
        integrator_cls = VelocityVerletThermostat
    integrator = integrator_cls(dynamics, potential, dt, kT=1.0, mass=1.0)

    burnin_evals = int(n_force_evals * 0.1)
    qs = []
    ps = []
    step = 0
    store_every = 5  # denser sampling for histograms
    while state.n_force_evals < n_force_evals:
        state = integrator.step(state)
        step += 1
        if np.any(np.isnan(state.q)):
            break
        if state.n_force_evals >= burnin_evals and step % store_every == 0:
            qs.append(state.q.copy())
            ps.append(state.p.copy())

    return np.array(qs), np.array(ps)


def true_density_2d(potential, xlim, ylim, kT=1.0, n_grid=100):
    """Compute normalized true Boltzmann density on grid."""
    x = np.linspace(xlim[0], xlim[1], n_grid)
    y = np.linspace(ylim[0], ylim[1], n_grid)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing='ij')
    logp = np.zeros((n_grid, n_grid))
    for i in range(n_grid):
        for j in range(n_grid):
            logp[i, j] = -potential.energy(np.array([x[i], y[j]])) / kT
    logp -= logp.max()
    p = np.exp(logp)
    p /= (p.sum() * dx * dy)
    return x, y, p


def sampled_density_2d(samples, xlim, ylim, n_bins=100):
    """Compute 2D histogram density from samples."""
    xedges = np.linspace(xlim[0], xlim[1], n_bins + 1)
    yedges = np.linspace(ylim[0], ylim[1], n_bins + 1)
    hist, _, _ = np.histogram2d(samples[:, 0], samples[:, 1],
                                 bins=[xedges, yedges], density=True)
    return xedges, yedges, hist


def plot_density_row_2d(axes, potential, label, xlim, ylim,
                        samples_nhc, samples_nhctail, n_bins=100):
    """Plot one row of density comparisons for a 2D system."""
    # True density
    x, y, p_true = true_density_2d(potential, xlim, ylim, n_grid=n_bins)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Sampled densities
    xe_nhc, ye_nhc, h_nhc = sampled_density_2d(samples_nhc, xlim, ylim, n_bins)
    xe_tail, ye_tail, h_tail = sampled_density_2d(samples_nhctail, xlim, ylim, n_bins)

    # Common color scale
    vmax = max(p_true.max(), h_nhc.max(), h_tail.max())
    vmin = 0

    # KL divergences
    kl_nhc = kl_divergence_histogram(samples_nhc, potential, kT=1.0, n_bins=n_bins)
    kl_tail = kl_divergence_histogram(samples_nhctail, potential, kT=1.0, n_bins=n_bins)

    X, Y = np.meshgrid(x, y, indexing='ij')
    datasets = [p_true, h_nhc, h_tail]
    kl_vals = [None, kl_nhc, kl_tail]
    panel_labels = ['True Density', 'NHC Sampled', 'NHCTail Sampled']

    for idx, (ax, data, kl_val) in enumerate(zip(axes, datasets, kl_vals)):
        im = ax.pcolormesh(X, Y, data, cmap='inferno', vmin=vmin, vmax=vmax,
                          rasterized=True)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=10)

        if kl_val is not None:
            ax.text(0.03, 0.97, f'KL={kl_val:.4f}',
                   transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    axes[0].set_ylabel(label, fontsize=14, fontweight='bold')
    return kl_nhc, kl_tail


def plot_density_row_1d(axes, samples_nhc_q, samples_nhc_p,
                        samples_nhctail_q, samples_nhctail_p):
    """Plot 1D HO density comparison -- marginal q and p distributions."""
    sigma_q = 1.0
    sigma_p = 1.0

    # True joint density in (q,p) space
    qlim = (-4.5, 4.5)
    plim = (-4.5, 4.5)
    n_bins = 80
    q_edges = np.linspace(qlim[0], qlim[1], n_bins + 1)
    p_edges = np.linspace(plim[0], plim[1], n_bins + 1)
    qc = 0.5 * (q_edges[:-1] + q_edges[1:])
    pc = 0.5 * (p_edges[:-1] + p_edges[1:])
    Q, P = np.meshgrid(qc, pc, indexing='ij')
    p_true = np.exp(-Q**2 / (2 * sigma_q**2) - P**2 / (2 * sigma_p**2))
    p_true /= (p_true.sum() * (qc[1] - qc[0]) * (pc[1] - pc[0]))

    # Sampled (q,p) joint
    h_nhc, _, _ = np.histogram2d(samples_nhc_q[:, 0], samples_nhc_p[:, 0],
                                  bins=[q_edges, p_edges], density=True)
    h_tail, _, _ = np.histogram2d(samples_nhctail_q[:, 0], samples_nhctail_p[:, 0],
                                   bins=[q_edges, p_edges], density=True)

    vmax = max(p_true.max(), h_nhc.max(), h_tail.max())

    # Compute KL for q marginal using the HO potential
    ho = HarmonicOscillator1D()
    kl_nhc = kl_divergence_histogram(samples_nhc_q, ho, kT=1.0, n_bins=100)
    kl_tail = kl_divergence_histogram(samples_nhctail_q, ho, kT=1.0, n_bins=100)

    datasets = [p_true, h_nhc, h_tail]
    kl_vals = [None, kl_nhc, kl_tail]

    for idx, (ax, data, kl_val) in enumerate(zip(axes, datasets, kl_vals)):
        ax.pcolormesh(Q, P, data, cmap='inferno', vmin=0, vmax=vmax, rasterized=True)
        ax.set_xlim(qlim)
        ax.set_ylim(plim)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=10)
        ax.set_xlabel('q', fontsize=12)

        if kl_val is not None:
            ax.text(0.03, 0.97, f'KL={kl_val:.4f}',
                   transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    axes[0].set_ylabel('Harmonic Osc.\n(q, p)', fontsize=13, fontweight='bold')
    return kl_nhc, kl_tail


def main():
    print("=== Figure 2: Sampled Density vs True Density ===")

    fig, axes = plt.subplots(4, 3, figsize=(14, 19), dpi=300)

    axes[0, 0].set_title('True Density', fontsize=14, fontweight='bold', pad=10)
    axes[0, 1].set_title('NHC (M=3, Q=1.0)', fontsize=14, fontweight='bold', pad=10)
    axes[0, 2].set_title('NHCTail (Qs=[0.1, 0.7, 10])', fontsize=14, fontweight='bold', pad=10)

    all_kl = {}

    # 2D systems
    for row_idx, (pot_cls, pot_kwargs, nhc_dt, nhctail_dt, xlim, ylim, label) in enumerate(SYSTEMS_2D):
        print(f"\n--- {label} ---")
        potential = pot_cls(**pot_kwargs)

        print(f"  Running NHC (dt={nhc_dt})...")
        nhc = NoseHooverChain(dim=2, chain_length=3, kT=1.0, Q=1.0)
        samples_nhc, _ = run_and_collect(nhc, potential, nhc_dt, N_FORCE_EVALS, SEED)
        print(f"  NHC: {len(samples_nhc)} samples")

        print(f"  Running NHCTail (dt={nhctail_dt})...")
        nhctail = MultiScaleNHCTail(dim=2, Qs=[0.1, 0.7, 10.0], chain_length=2)
        samples_tail, _ = run_and_collect(nhctail, potential, nhctail_dt, N_FORCE_EVALS, SEED,
                                          integrator_cls=MultiScaleNHCTailVerlet)
        print(f"  NHCTail: {len(samples_tail)} samples")

        kl_nhc, kl_tail = plot_density_row_2d(
            axes[row_idx], potential, label, xlim, ylim, samples_nhc, samples_tail)
        all_kl[label] = {'nhc': kl_nhc, 'nhctail': kl_tail}
        print(f"  KL: NHC={kl_nhc:.4f}, NHCTail={kl_tail:.4f}")

    # 1D HO
    print(f"\n--- Harmonic Oscillator ---")
    ho = HarmonicOscillator1D()
    ho_dt = 0.005

    print(f"  Running NHC...")
    nhc_ho = NoseHooverChain(dim=1, chain_length=3, kT=1.0, Q=1.0)
    sq_nhc, sp_nhc = run_and_collect(nhc_ho, ho, ho_dt, N_FORCE_EVALS, SEED)
    print(f"  NHC: {len(sq_nhc)} samples")

    print(f"  Running NHCTail...")
    nhctail_ho = MultiScaleNHCTail(dim=1, Qs=[0.1, 0.7, 10.0], chain_length=2)
    sq_tail, sp_tail = run_and_collect(nhctail_ho, ho, ho_dt, N_FORCE_EVALS, SEED,
                                       integrator_cls=MultiScaleNHCTailVerlet)
    print(f"  NHCTail: {len(sq_tail)} samples")

    kl_nhc, kl_tail = plot_density_row_1d(axes[3], sq_nhc, sp_nhc, sq_tail, sp_tail)
    all_kl['HO'] = {'nhc': kl_nhc, 'nhctail': kl_tail}
    print(f"  KL: NHC={kl_nhc:.4f}, NHCTail={kl_tail:.4f}")

    plt.tight_layout(h_pad=1.5, w_pad=0.8)

    outpath = os.path.join(FIGDIR, 'fig2_density_comparison.png')
    fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nSaved: {outpath}")
    print("\nKL Summary:")
    for name, kls in all_kl.items():
        print(f"  {name}: NHC={kls['nhc']:.4f}, NHCTail={kls['nhctail']:.4f}")


if __name__ == '__main__':
    main()
