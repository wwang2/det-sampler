"""Generate diagnostic plots for the Multi-Scale Log-Osc thermostat.

Produces:
1. GMM sample scatter plot with mode centers
2. Mode visitation over time
3. KL convergence trace comparison
4. Thermostat variable dynamics
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import importlib
sol = importlib.import_module('orbits.log-osc-multiT-005.solution')
parent_sol = importlib.import_module('orbits.log-osc-001.solution')

from research.eval.potentials import GaussianMixture2D
from research.eval.evaluator import run_sampler
from research.eval.integrators import ThermostatState

FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(exist_ok=True)

# Style settings from research/style.md
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

# Colors
COLOR_NH = '#1f77b4'
COLOR_NHC = '#ff7f0e'
COLOR_NOVEL = '#2ca02c'  # tab10 index 2
COLOR_NOVEL2 = '#d62728'  # tab10 index 3

gmm = GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5)
N = 1_000_000


def collect_trajectory(dynamics, integrator_cls, potential, dt, n_force_evals, thin=100):
    """Run sampler and collect trajectory with thinning."""
    import numpy as np
    rng = np.random.default_rng(42)
    q0 = rng.normal(0, 0.5, size=potential.dim)
    state = dynamics.initial_state(q0, rng=rng)
    integrator = integrator_cls(dynamics, potential, dt, kT=1.0, mass=1.0)

    all_q = []
    all_xi = []
    step = 0
    while state.n_force_evals < n_force_evals:
        state = integrator.step(state)
        step += 1
        if np.any(np.isnan(state.q)):
            break
        if step % thin == 0 and state.n_force_evals > n_force_evals * 0.1:
            all_q.append(state.q.copy())
            all_xi.append(state.xi.copy())

    return np.array(all_q), np.array(all_xi)


def plot_gmm_samples():
    """Plot sample scatter for parent vs MultiScale on GMM."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Parent: single log-osc
    dyn_parent = parent_sol.LogOscThermostat(dim=2, kT=1.0, Q=0.5)
    q_parent, _ = collect_trajectory(dyn_parent, parent_sol.LogOscVelocityVerlet, gmm, dt=0.02, n_force_evals=N)

    # MultiScale (best config)
    dyn_ms = sol.MultiScaleLogOsc(dim=2, kT=1.0, Qs=[0.1, 0.7, 10.0])
    q_ms, xi_ms = collect_trajectory(dyn_ms, sol.MultiScaleLogOscVerlet, gmm, dt=0.03, n_force_evals=N)

    for ax, q, title in [(axes[0], q_parent, 'Log-Osc (parent, Q=0.5)'),
                          (axes[1], q_ms, 'MultiScale Log-Osc (0.1, 0.7, 10)')]:
        if len(q) > 0:
            ax.scatter(q[:, 0], q[:, 1], s=0.5, alpha=0.3, c=COLOR_NOVEL if 'Multi' in title else COLOR_NH)

        # Plot mode centers
        for center in gmm.centers:
            circle = plt.Circle(center, gmm.sigma, fill=False, color='red', linewidth=1.5, linestyle='--')
            ax.add_patch(circle)
            ax.plot(center[0], center[1], 'rx', markersize=8, markeredgewidth=2)

        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)

    fig.suptitle('GMM Sample Distribution (5 modes, 1M force evals)', fontsize=16)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'gmm_samples.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved gmm_samples.png (parent: {len(q_parent)} samples, MS: {len(q_ms)} samples)')


def plot_mode_visitation():
    """Show which mode the sampler is visiting over time."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    configs = [
        ('Log-Osc (parent)', parent_sol.LogOscThermostat(dim=2, kT=1.0, Q=0.5),
         parent_sol.LogOscVelocityVerlet, 0.02, COLOR_NH),
        ('MultiScale (0.1, 0.7, 10)', sol.MultiScaleLogOsc(dim=2, kT=1.0, Qs=[0.1, 0.7, 10.0]),
         sol.MultiScaleLogOscVerlet, 0.03, COLOR_NOVEL),
    ]

    for ax, (name, dyn, integ_cls, dt, color) in zip(axes, configs):
        q, _ = collect_trajectory(dyn, integ_cls, gmm, dt, N, thin=10)

        if len(q) == 0:
            ax.set_title(f'{name} - no samples')
            continue

        # Assign each sample to nearest mode
        dists = np.array([np.sqrt(np.sum((q - c)**2, axis=1)) for c in gmm.centers])  # (n_modes, n_samples)
        nearest_mode = np.argmin(dists, axis=0)
        min_dist = np.min(dists, axis=0)
        nearest_mode[min_dist > 2 * gmm.sigma] = -1  # "between modes"

        n_samples = len(q)
        x_axis = np.arange(n_samples)

        for mode_id in range(gmm.n_modes):
            mask = nearest_mode == mode_id
            ax.scatter(x_axis[mask], np.full(np.sum(mask), mode_id), s=0.3, alpha=0.5)

        ax.set_ylabel('Mode ID')
        ax.set_title(f'{name}')
        ax.set_yticks(range(gmm.n_modes))

        # Count modes visited
        modes_visited = len(set(nearest_mode[nearest_mode >= 0]))
        ax.annotate(f'{modes_visited}/5 modes visited', xy=(0.02, 0.95), xycoords='axes fraction',
                   fontsize=12, va='top', fontweight='bold')

    axes[1].set_xlabel('Sample index (thinned)')
    fig.suptitle('Mode Visitation Over Time (GMM, 1M force evals)', fontsize=16)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'mode_visitation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved mode_visitation.png')


def plot_thermostat_dynamics():
    """Show xi variable dynamics for MultiScale."""
    dyn = sol.MultiScaleLogOsc(dim=2, kT=1.0, Qs=[0.1, 0.7, 10.0])
    _, xi = collect_trajectory(dyn, sol.MultiScaleLogOscVerlet, gmm, dt=0.03, n_force_evals=200_000, thin=5)

    if len(xi) == 0:
        print('No xi data collected')
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    labels = ['Fast (Q=0.1)', 'Medium (Q=0.7)', 'Slow (Q=10.0)']
    colors = ['#e41a1c', '#377eb8', '#4daf4a']

    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(xi[:, i], linewidth=0.5, color=color, alpha=0.7)
        # Also plot g(xi) = bounded friction
        g_vals = 2.0 * xi[:, i] / (1.0 + xi[:, i]**2)
        ax2 = ax.twinx()
        ax2.plot(g_vals, linewidth=0.5, color='gray', alpha=0.5)
        ax2.set_ylabel('g(xi)', color='gray', fontsize=10)
        ax2.set_ylim(-1.5, 1.5)

        ax.set_ylabel(f'xi ({label})')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Sample index')
    fig.suptitle('Thermostat Variable Dynamics (MultiScale Log-Osc on GMM)', fontsize=16)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'thermostat_dynamics.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved thermostat_dynamics.png')


def plot_kl_comparison():
    """Bar chart comparing KL across methods and potentials."""
    # Data from experiments
    methods = ['NH', 'NHC(3)', 'Log-Osc', 'MultiScale\n(0.1,0.7,10)']
    gmm_kl = [0.383, 0.544, 0.377, 0.148]
    dw_kl = [0.037, 0.029, 0.010, 0.010]
    ho_kl = [0.077, 0.002, 0.023, 0.004]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(methods))
    width = 0.6

    colors = ['gray', 'gray', COLOR_NH, COLOR_NOVEL]

    for ax, data, title in [(axes[0], gmm_kl, 'GMM (5-mode)'),
                             (axes[1], dw_kl, 'Double Well'),
                             (axes[2], ho_kl, '1D Harmonic Osc')]:
        bars = ax.bar(x, data, width, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=10)
        ax.set_ylabel('KL Divergence')
        ax.set_title(title)
        ax.axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='Target KL=0.01')
        ax.set_yscale('log')
        ax.set_ylim(0.001, 2.0)

        # Annotate values
        for bar, val in zip(bars, data):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    axes[0].legend(fontsize=10)
    fig.suptitle('KL Divergence Comparison (1M force evals)', fontsize=16)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'kl_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved kl_comparison.png')


if __name__ == '__main__':
    print('Generating plots...', flush=True)
    plot_kl_comparison()
    plot_gmm_samples()
    plot_mode_visitation()
    plot_thermostat_dynamics()
    print('All plots generated.')
