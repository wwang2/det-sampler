"""Generate diagnostic figures for the Log-Osc thermostat."""

import sys
import importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/wujiewang/code/det-sampler/.worktrees/log-osc-001')

from research.eval.evaluator import run_sampler
from research.eval.potentials import HarmonicOscillator1D, DoubleWell2D
from research.eval.baselines import NoseHoover, NoseHooverChain
from research.eval.integrators import VelocityVerletThermostat

spec = importlib.util.spec_from_file_location(
    'solution',
    '/Users/wujiewang/code/det-sampler/.worktrees/log-osc-001/orbits/log-osc-001/solution.py',
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

FIGDIR = '/Users/wujiewang/code/det-sampler/.worktrees/log-osc-001/orbits/log-osc-001/figures'

# Style constants from research/style.md
COLOR_NH = '#1f77b4'
COLOR_NHC = '#ff7f0e'
COLOR_LOGOSC = '#2ca02c'  # tab10 index 2
COLOR_LOGOSC_CHAIN = '#d62728'  # tab10 index 3


def collect_trajectory(dynamics, potential, integrator_cls, dt, n_force_evals, kT=1.0):
    """Run sampler and collect full trajectory (including burnin)."""
    rng = np.random.default_rng(42)
    q0 = rng.normal(0, 0.5, size=potential.dim)
    state = dynamics.initial_state(q0, rng=rng)
    integrator = integrator_cls(dynamics, potential, dt, kT=kT, mass=1.0)

    all_q, all_p, all_xi = [], [], []
    while state.n_force_evals < n_force_evals:
        state = integrator.step(state)
        if np.any(np.isnan(state.q)):
            break
        all_q.append(state.q.copy())
        all_p.append(state.p.copy())
        all_xi.append(state.xi.copy())

    return np.array(all_q), np.array(all_p), np.array(all_xi)


def plot_phase_space_ho():
    """Phase space (q, p) for 1D HO -- the key ergodicity diagnostic."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

    pot = HarmonicOscillator1D()
    n_evals = 500_000

    configs = [
        ("Nose-Hoover (Q=1)", NoseHoover(dim=1, kT=1.0, Q=1.0),
         VelocityVerletThermostat, 0.005, COLOR_NH),
        ("Log-Osc (Q=0.8)", mod.LogOscThermostat(dim=1, kT=1.0, Q=0.8),
         mod.LogOscVelocityVerlet, 0.005, COLOR_LOGOSC),
        ("NHC M=3 (Q=1)", NoseHooverChain(dim=1, chain_length=3, kT=1.0, Q=1.0),
         VelocityVerletThermostat, 0.005, COLOR_NHC),
    ]

    for ax, (title, dyn, intg, dt, color) in zip(axes, configs):
        q_traj, p_traj, _ = collect_trajectory(dyn, pot, intg, dt, n_evals)

        # Subsample for plotting
        step = max(1, len(q_traj) // 5000)
        q_plot = q_traj[::step, 0]
        p_plot = p_traj[::step, 0]

        ax.scatter(q_plot, p_plot, s=0.5, alpha=0.3, c=color)

        # Overlay expected Gaussian contours
        sigma_q = 1.0  # sqrt(kT/omega^2) = 1
        sigma_p = 1.0  # sqrt(m*kT) = 1
        theta = np.linspace(0, 2*np.pi, 200)
        for nsig in [1, 2, 3]:
            ax.plot(nsig*sigma_q*np.cos(theta), nsig*sigma_p*np.sin(theta),
                    'k--', alpha=0.3, linewidth=0.8)

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.set_xlabel('q', fontsize=14)
        ax.set_ylabel('p', fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.tick_params(labelsize=12)

    fig.suptitle('Phase Space Coverage: 1D Harmonic Oscillator', fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/phase_space_ho.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIGDIR}/phase_space_ho.png')


def plot_g_function():
    """Plot the friction function g(xi) = 2xi/(1+xi^2) vs xi (NH)."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    xi_vals = np.linspace(-5, 5, 500)
    g_vals = 2 * xi_vals / (1 + xi_vals**2)
    nh_vals = xi_vals  # Standard NH: alpha = xi

    ax.plot(xi_vals, g_vals, color=COLOR_LOGOSC, linewidth=2.5, label=r'Log-Osc: $g(\xi) = 2\xi/(1+\xi^2)$')
    ax.plot(xi_vals, nh_vals, color=COLOR_NH, linewidth=2.5, linestyle='--', label=r'Nose-Hoover: $\alpha(\xi) = \xi$')
    ax.axhline(y=1, color='gray', linewidth=0.8, linestyle=':')
    ax.axhline(y=-1, color='gray', linewidth=0.8, linestyle=':')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)

    ax.set_xlabel(r'$\xi$', fontsize=14)
    ax.set_ylabel(r'Friction coupling $\alpha(\xi)$', fontsize=14)
    ax.set_title('Friction Coupling: Log-Osc vs Nose-Hoover', fontsize=16)
    ax.legend(fontsize=12, loc='upper left')
    ax.set_ylim(-5, 5)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/g_function.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIGDIR}/g_function.png')


def plot_xi_trajectory():
    """Plot thermostat variable xi over time for NH vs Log-Osc."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=150, sharex=True)

    pot = HarmonicOscillator1D()
    n_evals = 200_000

    # NH
    dyn_nh = NoseHoover(dim=1, kT=1.0, Q=1.0)
    _, _, xi_nh = collect_trajectory(dyn_nh, pot, VelocityVerletThermostat, 0.005, n_evals)

    # Log-Osc
    dyn_lo = mod.LogOscThermostat(dim=1, kT=1.0, Q=0.8)
    _, _, xi_lo = collect_trajectory(dyn_lo, pot, mod.LogOscVelocityVerlet, 0.005, n_evals)

    # Time axis (in force evals, approximately)
    t_nh = np.arange(len(xi_nh))
    t_lo = np.arange(len(xi_lo))

    axes[0].plot(t_nh[:20000], xi_nh[:20000, 0], color=COLOR_NH, linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel(r'$\xi$', fontsize=14)
    axes[0].set_title('Nose-Hoover: thermostat variable', fontsize=16)
    axes[0].tick_params(labelsize=12)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_lo[:20000], xi_lo[:20000, 0], color=COLOR_LOGOSC, linewidth=0.5, alpha=0.7)
    axes[1].set_ylabel(r'$\xi$', fontsize=14)
    axes[1].set_xlabel('Integration step', fontsize=14)
    axes[1].set_title('Log-Osc (Q=0.8): thermostat variable', fontsize=16)
    axes[1].tick_params(labelsize=12)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/xi_trajectory.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIGDIR}/xi_trajectory.png')


def plot_kl_convergence():
    """KL convergence trace for all methods on double-well."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    pot = DoubleWell2D()
    n_evals = 1_000_000

    configs = [
        ("NH (Q=1, dt=0.01)", NoseHoover(dim=2, kT=1.0, Q=1.0),
         VelocityVerletThermostat, 0.01, COLOR_NH, '--'),
        ("NHC M=3 (Q=1, dt=0.01)", NoseHooverChain(dim=2, chain_length=3, kT=1.0, Q=1.0),
         VelocityVerletThermostat, 0.01, COLOR_NHC, '--'),
        ("Log-Osc (Q=1, dt=0.03)", mod.LogOscThermostat(dim=2, kT=1.0, Q=1.0),
         mod.LogOscVelocityVerlet, 0.03, COLOR_LOGOSC, '-'),
    ]

    for label, dyn, intg, dt, color, ls in configs:
        r = run_sampler(dyn, pot, dt=dt, n_force_evals=n_evals, kT=1.0,
                        integrator_cls=intg, kl_checkpoints=50)
        trace = r['kl_trace']
        if trace:
            xs = [t[0] for t in trace]
            ys = [max(t[1], 1e-5) for t in trace]
            ax.plot(xs, ys, color=color, linewidth=2, linestyle=ls, label=f'{label} (final={r["kl_divergence"]:.3f})')

    ax.axhline(y=0.01, color='gray', linewidth=1, linestyle=':', label='KL=0.01 threshold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Force evaluations', fontsize=14)
    ax.set_ylabel('KL divergence', fontsize=14)
    ax.set_title('KL Convergence: 2D Double-Well', fontsize=16)
    ax.legend(fontsize=11, loc='upper right')
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/kl_convergence_dw.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIGDIR}/kl_convergence_dw.png')


def plot_q_scan_summary():
    """Bar chart of Q-scan results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # Data from Q-scan
    Q_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 2.0, 5.0, 10.0]
    # HO results
    ho_ergo = [0.745, 0.814, 0.860, 0.855, 0.863, 0.855, 0.944, 0.591, 0.543, 0.536, 0.542]
    ho_kl = [0.049, 0.020, 0.007, 0.006, 0.002, 0.005, 0.023, 0.036, 0.075, 0.090, 0.098]

    # DW results (dt=0.01)
    Q_dw = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    dw_kl = [0.386, 0.055, 0.033, 0.041, 0.057, 0.098]

    x = np.arange(len(Q_values))
    ax = axes[0]
    ax.bar(x, ho_ergo, color=COLOR_LOGOSC, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0.54, color=COLOR_NH, linewidth=2, linestyle='--', label='NH baseline (0.54)')
    ax.axhline(y=0.92, color=COLOR_NHC, linewidth=2, linestyle='--', label='NHC(M=3) baseline (0.92)')
    ax.axhline(y=0.85, color='gray', linewidth=1, linestyle=':', label='Ergodic threshold (0.85)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(q) for q in Q_values], fontsize=10)
    ax.set_xlabel('Q', fontsize=14)
    ax.set_ylabel('Ergodicity Score', fontsize=14)
    ax.set_title('1D HO: Ergodicity vs Q', fontsize=16)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=12)
    ax.set_ylim(0, 1.05)

    x2 = np.arange(len(Q_dw))
    ax2 = axes[1]
    ax2.bar(x2, dw_kl, color=COLOR_LOGOSC, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0.037, color=COLOR_NH, linewidth=2, linestyle='--', label='NH baseline (0.037)')
    ax2.axhline(y=0.029, color=COLOR_NHC, linewidth=2, linestyle='--', label='NHC(M=3) baseline (0.029)')
    ax2.axhline(y=0.01, color='gray', linewidth=1, linestyle=':', label='KL=0.01 threshold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels([str(q) for q in Q_dw], fontsize=10)
    ax2.set_xlabel('Q', fontsize=14)
    ax2.set_ylabel('KL divergence', fontsize=14)
    ax2.set_title('2D Double-Well: KL vs Q (dt=0.01)', fontsize=16)
    ax2.legend(fontsize=10)
    ax2.tick_params(labelsize=12)

    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/q_scan_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIGDIR}/q_scan_summary.png')


if __name__ == '__main__':
    print('Generating diagnostic figures...')
    plot_g_function()
    plot_phase_space_ho()
    plot_xi_trajectory()
    plot_q_scan_summary()
    plot_kl_convergence()
    print('Done.')
