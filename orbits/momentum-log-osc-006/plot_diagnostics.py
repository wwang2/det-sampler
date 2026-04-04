"""Generate diagnostic plots for MLOSC variants."""

import sys
import json
import importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORKTREE = '/Users/wujiewang/code/det-sampler/.worktrees/momentum-log-osc-006'
sys.path.insert(0, WORKTREE)

from research.eval.evaluator import run_sampler
from research.eval.potentials import (
    DoubleWell2D, HarmonicOscillator1D,
    GaussianMixture2D, Rosenbrock2D,
)

# Import solution
_spec = importlib.util.spec_from_file_location(
    "solution",
    f"{WORKTREE}/orbits/momentum-log-osc-006/solution.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MomentumLogOsc = _mod.MomentumLogOsc
MomentumLogOscVerlet = _mod.MomentumLogOscVerlet
RippledLogOsc = _mod.RippledLogOsc
RippledLogOscVerlet = _mod.RippledLogOscVerlet

FIGURES_DIR = f"{WORKTREE}/orbits/momentum-log-osc-006/figures"


def plot_kl_convergence(results_dict, title, filename):
    """Plot KL convergence traces for multiple configurations."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    for (label, result), color in zip(results_dict.items(), colors):
        kl_trace = result.get('kl_trace', [])
        if kl_trace:
            evals, kls = zip(*kl_trace)
            ax.plot(evals, kls, '-o', label=label, color=color, markersize=3)

    ax.axhline(y=0.01, color='gray', linestyle='--', alpha=0.7, label='KL=0.01 threshold')
    ax.set_xlabel('Force evaluations', fontsize=14)
    ax.set_ylabel('KL divergence', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=10, loc='upper right')
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=150)
    plt.close()
    print(f"Saved {filename}")


def plot_phase_space(dynamics, integrator_cls, potential, dt, Q, label, filename,
                     n_force_evals=1_000_000, **kwargs):
    """Plot phase space for 1D HO."""
    result = run_sampler(
        dynamics, potential, dt=dt, n_force_evals=n_force_evals,
        kT=1.0, integrator_cls=integrator_cls,
    )

    if result.get('nan_detected') or result['n_samples'] == 0:
        print(f"Skipping {label}: NaN or no samples")
        return result

    # Collect samples from a fresh run for phase space
    state = dynamics.initial_state(np.array([0.0]), rng=np.random.default_rng(42))
    integrator = integrator_cls(dynamics, potential, dt, kT=1.0, mass=1.0)

    qs, ps = [], []
    while state.n_force_evals < n_force_evals:
        state = integrator.step(state)
        if state.n_force_evals > n_force_evals * 0.1:
            qs.append(state.q[0])
            ps.append(state.p[0])
        if np.any(np.isnan(state.q)):
            break

    if len(qs) < 100:
        print(f"Skipping {label}: too few samples ({len(qs)})")
        return result

    qs = np.array(qs)
    ps = np.array(ps)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot
    ax.scatter(qs[::10], ps[::10], s=1, alpha=0.3, color='#2ca02c')

    # Expected Gaussian contours
    sigma_q = 1.0  # sqrt(kT/omega^2) = 1 for omega=1, kT=1
    sigma_p = 1.0  # sqrt(m*kT) = 1 for m=1, kT=1
    theta = np.linspace(0, 2*np.pi, 100)
    for n_sigma in [1, 2, 3]:
        ax.plot(n_sigma*sigma_q*np.cos(theta), n_sigma*sigma_p*np.sin(theta),
                'k-', alpha=0.3, linewidth=1)

    ax.set_xlabel('q', fontsize=14)
    ax.set_ylabel('p', fontsize=14)
    ax.set_title(f'{label} - Phase Space (1D HO)', fontsize=16)
    ax.set_aspect('equal')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=150)
    plt.close()
    print(f"Saved {filename}")

    return result


def plot_alpha_comparison():
    """Generate comparison plot of alpha values for MLOSC-A."""
    alphas = [0.0, 0.2, 0.5, 1.0]
    pot = DoubleWell2D()

    results = {}
    for alpha in alphas:
        dyn = MomentumLogOsc(dim=2, kT=1.0, Q=1.0, alpha=alpha)
        r = run_sampler(dyn, pot, dt=0.035, n_force_evals=1_000_000, kT=1.0,
                        integrator_cls=MomentumLogOscVerlet)
        results[f"alpha={alpha}"] = r

    plot_kl_convergence(results, "MLOSC-A: Alpha Scan on 2D Double-Well",
                        "alpha_scan_dw_kl.png")


def plot_rippled_comparison():
    """Generate comparison plot of epsilon values for MLOSC-B."""
    configs = [(0.0, 2.0), (0.1, 2.0), (0.3, 2.0), (0.5, 2.0)]
    pot = DoubleWell2D()

    results = {}
    for eps, w in configs:
        if eps == 0.0:
            label = "Log-Osc (baseline)"
        else:
            label = f"eps={eps}, w={w}"
        dyn = RippledLogOsc(dim=2, kT=1.0, Q=1.0, epsilon=eps, omega_xi=w)
        r = run_sampler(dyn, pot, dt=0.035, n_force_evals=1_000_000, kT=1.0,
                        integrator_cls=RippledLogOscVerlet)
        results[label] = r

    plot_kl_convergence(results, "MLOSC-B: Rippled Scan on 2D Double-Well",
                        "rippled_scan_dw_kl.png")


def plot_best_phase_space():
    """Phase space plots for best MLOSC-A and MLOSC-B on 1D HO."""
    pot = HarmonicOscillator1D()

    # MLOSC-A best
    dyn_a = MomentumLogOsc(dim=1, kT=1.0, Q=0.8, alpha=0.5)
    plot_phase_space(dyn_a, MomentumLogOscVerlet, pot, dt=0.005, Q=0.8,
                     label="MLOSC-A (alpha=0.5)", filename="phase_space_mlosc_a.png")

    # MLOSC-B best
    dyn_b = RippledLogOsc(dim=1, kT=1.0, Q=0.8, epsilon=0.3, omega_xi=2.0)
    plot_phase_space(dyn_b, RippledLogOscVerlet, pot, dt=0.005, Q=0.8,
                     label="MLOSC-B (eps=0.3,w=2)", filename="phase_space_mlosc_b.png")


def plot_thermostat_potentials():
    """Plot V(xi) for different thermostat potentials."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    xi_vals = np.linspace(-6, 6, 500)

    # Left: V(xi) landscape
    ax = axes[0]
    # Standard log-osc
    V_log = np.log(1 + xi_vals**2)
    ax.plot(xi_vals, V_log, 'k-', linewidth=2, label='Log-Osc: log(1+xi^2)')

    # Rippled variants
    for eps, w, ls in [(0.1, 2.0, '--'), (0.3, 2.0, '-.'), (0.5, 2.0, ':')]:
        V_rip = np.log(1 + xi_vals**2) + eps * np.cos(w * xi_vals)
        ax.plot(xi_vals, V_rip, ls, linewidth=1.5, label=f'Rippled: eps={eps}, w={w}')

    ax.set_xlabel('xi', fontsize=14)
    ax.set_ylabel('V(xi)', fontsize=14)
    ax.set_title('Thermostat Potential', fontsize=16)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)

    # Right: V'(xi) friction function
    ax = axes[1]
    g_log = 2*xi_vals / (1 + xi_vals**2)
    ax.plot(xi_vals, g_log, 'k-', linewidth=2, label='Log-Osc: g(xi)')

    for eps, w, ls in [(0.1, 2.0, '--'), (0.3, 2.0, '-.'), (0.5, 2.0, ':')]:
        vp = 2*xi_vals / (1 + xi_vals**2) - eps*w*np.sin(w*xi_vals)
        ax.plot(xi_vals, vp, ls, linewidth=1.5, label=f'Rippled: eps={eps}, w={w}')

    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.set_xlabel('xi', fontsize=14)
    ax.set_ylabel("V'(xi)", fontsize=14)
    ax.set_title('Friction Function', fontsize=16)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/thermostat_potentials.png", dpi=150)
    plt.close()
    print("Saved thermostat_potentials.png")


def plot_benchmark_comparison():
    """Bar chart comparing baselines and our best variants."""
    # Baseline data from config
    data = {
        'NH': {'DW KL': 0.037, 'HO Erg': 0.54},
        'NHC(3)': {'DW KL': 0.029, 'HO Erg': 0.92},
        'Log-Osc': {'DW KL': 0.010, 'HO Erg': 0.944},
    }

    # Run our best configs
    pot_dw = DoubleWell2D()
    pot_ho = HarmonicOscillator1D()

    # MLOSC-A best
    dyn_a = MomentumLogOsc(dim=2, kT=1.0, Q=1.0, alpha=0.5)
    r_a_dw = run_sampler(dyn_a, pot_dw, dt=0.035, n_force_evals=1_000_000,
                          kT=1.0, integrator_cls=MomentumLogOscVerlet)
    dyn_a_ho = MomentumLogOsc(dim=1, kT=1.0, Q=0.8, alpha=0.5)
    r_a_ho = run_sampler(dyn_a_ho, pot_ho, dt=0.005, n_force_evals=1_000_000,
                          kT=1.0, integrator_cls=MomentumLogOscVerlet)
    data['MLOSC-A'] = {
        'DW KL': r_a_dw['kl_divergence'],
        'HO Erg': r_a_ho['ergodicity']['score'] if r_a_ho['ergodicity'] else 0,
    }

    # MLOSC-B best
    dyn_b = RippledLogOsc(dim=2, kT=1.0, Q=1.0, epsilon=0.3, omega_xi=2.0)
    r_b_dw = run_sampler(dyn_b, pot_dw, dt=0.035, n_force_evals=1_000_000,
                          kT=1.0, integrator_cls=RippledLogOscVerlet)
    dyn_b_ho = RippledLogOsc(dim=1, kT=1.0, Q=0.8, epsilon=0.3, omega_xi=2.0)
    r_b_ho = run_sampler(dyn_b_ho, pot_ho, dt=0.005, n_force_evals=1_000_000,
                          kT=1.0, integrator_cls=RippledLogOscVerlet)
    data['MLOSC-B'] = {
        'DW KL': r_b_dw['kl_divergence'],
        'HO Erg': r_b_ho['ergodicity']['score'] if r_b_ho['ergodicity'] else 0,
    }

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    methods = list(data.keys())
    x = np.arange(len(methods))
    bar_width = 0.5

    # Colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # DW KL
    ax = axes[0]
    vals = [data[m]['DW KL'] for m in methods]
    bars = ax.bar(x, vals, bar_width, color=colors[:len(methods)])
    ax.axhline(y=0.01, color='gray', linestyle='--', alpha=0.7, label='Target')
    ax.set_ylabel('KL Divergence', fontsize=14)
    ax.set_title('2D Double-Well KL', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11, rotation=15)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=10)
    for i, v in enumerate(vals):
        if v is not None and v != float('inf'):
            ax.text(i, v + 0.001, f'{v:.3f}', ha='center', fontsize=10)

    # HO Ergodicity
    ax = axes[1]
    vals = [data[m]['HO Erg'] for m in methods]
    bars = ax.bar(x, vals, bar_width, color=colors[:len(methods)])
    ax.axhline(y=0.85, color='gray', linestyle='--', alpha=0.7, label='Ergodic threshold')
    ax.set_ylabel('Ergodicity Score', fontsize=14)
    ax.set_title('1D Harmonic Oscillator Ergodicity', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11, rotation=15)
    ax.tick_params(labelsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    for i, v in enumerate(vals):
        if v is not None:
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/benchmark_comparison.png", dpi=150)
    plt.close()
    print("Saved benchmark_comparison.png")

    return data


if __name__ == "__main__":
    print("Generating thermostat potential plot...")
    plot_thermostat_potentials()

    print("\nGenerating alpha comparison plot...")
    plot_alpha_comparison()

    print("\nGenerating rippled comparison plot...")
    plot_rippled_comparison()

    print("\nGenerating phase space plots...")
    plot_best_phase_space()

    print("\nGenerating benchmark comparison plot...")
    comparison_data = plot_benchmark_comparison()

    print("\n=== FINAL COMPARISON ===")
    for method, vals in comparison_data.items():
        print(f"  {method}: DW KL={vals['DW KL']:.4f}, HO Erg={vals['HO Erg']:.3f}")
