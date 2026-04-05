"""Generate diagnostic plots for Multi-Scale NHCTail sampler."""

import sys
import os
import json
import importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORKTREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WORKTREE)

from research.eval.evaluator import run_sampler
from research.eval.potentials import (
    HarmonicOscillator1D, DoubleWell2D, GaussianMixture2D, Rosenbrock2D
)

_spec = importlib.util.spec_from_file_location(
    "solution",
    os.path.join(WORKTREE, "orbits", "multiscale-chain-009", "solution.py"),
)
_sol = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sol)

FIG_DIR = os.path.join(WORKTREE, "orbits", "multiscale-chain-009", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

N_FORCE_EVALS = 1_000_000
KT = 1.0
QS = [0.1, 0.7, 10.0]


def run_and_collect(pot, dt, seed=42):
    """Run sampler and return full trajectory data."""
    dim = pot.dim
    dyn = _sol.MultiScaleNHCTail(dim=dim, kT=KT, Qs=QS, chain_length=2)
    rng = np.random.default_rng(seed)
    q0 = rng.normal(0, 0.5, size=dim)

    # We need raw trajectory, so run manually
    state = dyn.initial_state(q0, rng=rng)
    integrator = _sol.MultiScaleNHCTailVerlet(dyn, pot, dt=dt, kT=KT, mass=1.0)

    all_q = []
    all_p = []
    burnin = int(N_FORCE_EVALS * 0.1)

    while state.n_force_evals < N_FORCE_EVALS:
        state = integrator.step(state)
        if np.any(np.isnan(state.q)):
            break
        if state.n_force_evals >= burnin:
            all_q.append(state.q.copy())
            all_p.append(state.p.copy())

    return np.array(all_q), np.array(all_p)


def plot_ho_phase_space():
    """Phase space scatter for 1D HO with Gaussian contours."""
    print("Plotting HO phase space...")
    q, p = run_and_collect(HarmonicOscillator1D(), dt=0.005)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    # Subsample for clarity
    n = min(len(q), 50000)
    idx = np.random.default_rng(0).choice(len(q), n, replace=False)
    ax.scatter(q[idx, 0], p[idx, 0], s=0.5, alpha=0.3, c='#2ca02c', rasterized=True)

    # Expected Gaussian contours
    sigma_q = np.sqrt(KT)  # omega=1
    sigma_p = np.sqrt(KT)  # mass=1
    theta = np.linspace(0, 2*np.pi, 200)
    for nsig in [1, 2, 3]:
        ax.plot(nsig*sigma_q*np.cos(theta), nsig*sigma_p*np.sin(theta),
                'k--', alpha=0.5, linewidth=1)

    ax.set_xlabel("q", fontsize=14)
    ax.set_ylabel("p", fontsize=14)
    ax.set_title("NHCTail: 1D HO Phase Space", fontsize=16)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "ho_phase_space.png"), dpi=150)
    plt.close(fig)
    print("  Saved ho_phase_space.png")


def plot_dw_density():
    """2D density plot for double well."""
    print("Plotting DW density...")
    q, p = run_and_collect(DoubleWell2D(), dt=0.055)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Sampled density
    ax = axes[0]
    h, xedges, yedges = np.histogram2d(q[:, 0], q[:, 1], bins=80, density=True)
    ax.imshow(h.T, origin='lower', aspect='auto',
              extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
              cmap='viridis')
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    ax.set_title("Sampled density", fontsize=16)
    ax.tick_params(labelsize=12)

    # True density
    ax = axes[1]
    pot = DoubleWell2D()
    xc = 0.5*(xedges[:-1] + xedges[1:])
    yc = 0.5*(yedges[:-1] + yedges[1:])
    XX, YY = np.meshgrid(xc, yc, indexing='ij')
    Z = np.zeros_like(XX)
    for i in range(len(xc)):
        for j in range(len(yc)):
            Z[i, j] = np.exp(-pot.energy(np.array([XX[i, j], YY[i, j]])) / KT)
    Z /= Z.sum() * (xc[1]-xc[0]) * (yc[1]-yc[0])
    ax.imshow(Z.T, origin='lower', aspect='auto',
              extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
              cmap='viridis')
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    ax.set_title("True Boltzmann density", fontsize=16)
    ax.tick_params(labelsize=12)

    fig.suptitle("NHCTail: 2D Double Well", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "dw_density.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved dw_density.png")


def plot_gmm_samples():
    """Scatter plot of GMM samples showing mode coverage."""
    print("Plotting GMM samples...")
    q, p = run_and_collect(GaussianMixture2D(), dt=0.03)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    n = min(len(q), 50000)
    idx = np.random.default_rng(0).choice(len(q), n, replace=False)
    ax.scatter(q[idx, 0], q[idx, 1], s=0.5, alpha=0.3, c='#d62728', rasterized=True)

    # Plot mode centers
    pot = GaussianMixture2D()
    for center in pot.centers:
        ax.plot(center[0], center[1], 'k+', markersize=15, markeredgewidth=2)

    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    ax.set_title("NHCTail: 5-mode GMM Samples", fontsize=16)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "gmm_samples.png"), dpi=150)
    plt.close(fig)
    print("  Saved gmm_samples.png")


def plot_comparison_bar():
    """Bar chart comparing NHCTail with parent and baselines."""
    print("Plotting comparison bar chart...")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=150)

    samplers = ['NH', 'NHC(M=3)', 'MultiScale\n(parent)', 'NHCTail\n(this work)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # HO Ergodicity
    ax = axes[0]
    ho_erg = [0.54, 0.92, 0.927, 0.932]
    bars = ax.bar(samplers, ho_erg, color=colors, alpha=0.8)
    ax.axhline(y=0.85, color='gray', linestyle='--', linewidth=1, label='Target (0.85)')
    ax.set_ylabel("Ergodicity Score", fontsize=14)
    ax.set_title("1D HO Ergodicity", fontsize=16)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)

    # DW KL
    ax = axes[1]
    dw_kl = [0.037, 0.029, 0.010, 0.008]
    bars = ax.bar(samplers, dw_kl, color=colors, alpha=0.8)
    ax.axhline(y=0.007, color='gray', linestyle='--', linewidth=1, label='Target (0.007)')
    ax.set_ylabel("KL Divergence", fontsize=14)
    ax.set_title("2D Double Well KL", fontsize=16)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)

    # GMM KL
    ax = axes[2]
    gmm_kl = [0.383, 0.544, 0.148, 0.054]
    bars = ax.bar(samplers, gmm_kl, color=colors, alpha=0.8)
    ax.axhline(y=0.15, color='gray', linestyle='--', linewidth=1, label='Target (0.15)')
    ax.set_ylabel("KL Divergence", fontsize=14)
    ax.set_title("5-mode GMM KL", fontsize=16)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)

    fig.suptitle("Multi-Scale NHCTail vs Baselines", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "comparison.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved comparison.png")


if __name__ == "__main__":
    plot_comparison_bar()  # No simulation needed
    plot_ho_phase_space()
    plot_dw_density()
    plot_gmm_samples()
    print("\nAll plots saved to", FIG_DIR)
