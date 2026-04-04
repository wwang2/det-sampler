"""Log-Osc Stage 2: Q-Optimization and Stage 2 Benchmarks.

Builds on orbits/log-osc-001/solution.py (LogOscThermostat, LogOscChain).
Tasks:
  1. Fine-grained Q scan to find a config winning on BOTH ergodicity AND HO KL
  2. Stage 2 benchmarks: GMM 5-mode, Rosenbrock banana
  3. Comprehensive diagnostic plots
  4. Q-adaptation study across all potentials
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Add worktree root to path so we can import research.eval and parent orbit
WORKTREE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKTREE))

from research.eval.evaluator import run_sampler, kl_divergence_histogram
from research.eval.potentials import (
    HarmonicOscillator1D, DoubleWell2D, GaussianMixture2D, Rosenbrock2D,
)
from research.eval.integrators import ThermostatState
import importlib.util
_parent_solution_path = str(WORKTREE / "orbits" / "log-osc-001" / "solution.py")
_spec = importlib.util.spec_from_file_location("log_osc_001_solution", _parent_solution_path)
_parent_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_parent_mod)
LogOscThermostat = _parent_mod.LogOscThermostat
LogOscVelocityVerlet = _parent_mod.LogOscVelocityVerlet
LogOscChain = _parent_mod.LogOscChain
LogOscChainVerlet = _parent_mod.LogOscChainVerlet
g_func = _parent_mod.g_func

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Style constants from research/style.md
COLOR_NH = '#1f77b4'
COLOR_NHC = '#ff7f0e'
COLOR_LOSC = plt.cm.tab10(2)  # novel sampler starts at index 2
COLOR_LOSC_CHAIN = plt.cm.tab10(3)
FIGSIZE_SINGLE = (8, 6)
FIGSIZE_WIDE = (12, 5)
DPI = 150
FONT_LABEL = 14
FONT_TICK = 12
FONT_TITLE = 16

SEED = 42
N_FORCE_EVALS = 1_000_000


def set_plot_style():
    plt.rcParams.update({
        'font.size': FONT_TICK,
        'axes.labelsize': FONT_LABEL,
        'axes.titlesize': FONT_TITLE,
        'xtick.labelsize': FONT_TICK,
        'ytick.labelsize': FONT_TICK,
        'legend.fontsize': 11,
    })


# ============================================================================
# Task 1: Fine-grained Q scan on 1D HO
# ============================================================================

def task1_q_scan_ho():
    """Fine-grained Q scan between 0.4-0.9 with step 0.05, multiple dt values."""
    print("=" * 60)
    print("TASK 1: Fine-grained Q scan on 1D Harmonic Oscillator")
    print("=" * 60)

    pot = HarmonicOscillator1D(omega=1.0)
    q_values = np.arange(0.40, 0.91, 0.05)
    dt_values = [0.003, 0.005, 0.007, 0.01, 0.015]

    results = []
    for Q in q_values:
        for dt in dt_values:
            print(f"\n  Q={Q:.2f}, dt={dt:.3f} ... ", end="", flush=True)
            dyn = LogOscThermostat(dim=1, kT=1.0, mass=1.0, Q=Q)
            try:
                r = run_sampler(
                    dyn, pot, dt=dt, n_force_evals=N_FORCE_EVALS,
                    kT=1.0, mass=1.0,
                    rng=np.random.default_rng(SEED),
                    integrator_cls=LogOscVelocityVerlet,
                )
                kl = r['kl_divergence']
                erg = r['ergodicity']['score'] if r['ergodicity'] else 0.0
                ess = r['ess_metrics']['ess_per_force_eval'] if r['ess_metrics'] else 0.0
                print(f"KL={kl:.4f}, erg={erg:.3f}, ESS/fe={ess:.5f}")
                results.append({
                    'Q': float(Q), 'dt': dt,
                    'kl': kl, 'ergodicity': erg, 'ess_per_fe': ess,
                    'nan': r.get('nan_detected', False),
                })
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    'Q': float(Q), 'dt': dt,
                    'kl': float('inf'), 'ergodicity': 0.0, 'ess_per_fe': 0.0,
                    'nan': True,
                })

    with open(RESULTS_DIR / "task1_q_scan_ho.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


def task1_chain_scan():
    """Test LogOscChain (M=2,3) on 1D HO with promising Q/dt combos."""
    print("\n" + "=" * 60)
    print("TASK 1b: Chain variant scan on 1D HO")
    print("=" * 60)

    pot = HarmonicOscillator1D(omega=1.0)
    configs = [
        {'M': 2, 'Q': 0.6, 'dt': 0.005},
        {'M': 2, 'Q': 0.7, 'dt': 0.005},
        {'M': 2, 'Q': 0.8, 'dt': 0.005},
        {'M': 3, 'Q': 0.6, 'dt': 0.005},
        {'M': 3, 'Q': 0.7, 'dt': 0.005},
        {'M': 3, 'Q': 0.8, 'dt': 0.005},
        {'M': 2, 'Q': 0.6, 'dt': 0.007},
        {'M': 2, 'Q': 0.7, 'dt': 0.007},
        {'M': 3, 'Q': 0.6, 'dt': 0.007},
        {'M': 3, 'Q': 0.7, 'dt': 0.007},
    ]

    results = []
    for cfg in configs:
        M, Q, dt = cfg['M'], cfg['Q'], cfg['dt']
        print(f"\n  M={M}, Q={Q:.2f}, dt={dt:.3f} ... ", end="", flush=True)
        dyn = LogOscChain(dim=1, chain_length=M, kT=1.0, mass=1.0, Q=Q)
        try:
            r = run_sampler(
                dyn, pot, dt=dt, n_force_evals=N_FORCE_EVALS,
                kT=1.0, mass=1.0,
                rng=np.random.default_rng(SEED),
                integrator_cls=LogOscChainVerlet,
            )
            kl = r['kl_divergence']
            erg = r['ergodicity']['score'] if r['ergodicity'] else 0.0
            ess = r['ess_metrics']['ess_per_force_eval'] if r['ess_metrics'] else 0.0
            print(f"KL={kl:.4f}, erg={erg:.3f}, ESS/fe={ess:.5f}")
            results.append({
                'M': M, 'Q': float(Q), 'dt': dt,
                'kl': kl, 'ergodicity': erg, 'ess_per_fe': ess,
                'nan': r.get('nan_detected', False),
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'M': M, 'Q': float(Q), 'dt': dt,
                'kl': float('inf'), 'ergodicity': 0.0, 'ess_per_fe': 0.0,
                'nan': True,
            })

    with open(RESULTS_DIR / "task1_chain_scan_ho.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ============================================================================
# Task 2: Stage 2 Benchmarks (GMM + Rosenbrock)
# ============================================================================

def task2_stage2_benchmarks(best_Q=0.7, best_dt=0.005):
    """Run the best log-osc config on Stage 2 potentials."""
    print("\n" + "=" * 60)
    print("TASK 2: Stage 2 Benchmarks")
    print("=" * 60)

    potentials = {
        'gmm': GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5),
        'rosenbrock': Rosenbrock2D(a=0.0, b=5.0),
    }

    # Also test on Stage 1 for comparison
    potentials['ho'] = HarmonicOscillator1D(omega=1.0)
    potentials['dw'] = DoubleWell2D(barrier_height=1.0, y_stiffness=0.5)

    # Test multiple configs for stage 2 potentials
    configs = [
        {'Q': best_Q, 'dt': best_dt, 'label': f'Q={best_Q},dt={best_dt}'},
        {'Q': 0.8, 'dt': 0.005, 'label': 'Q=0.8,dt=0.005'},
        {'Q': 1.0, 'dt': 0.01, 'label': 'Q=1.0,dt=0.01'},
        {'Q': 1.0, 'dt': 0.035, 'label': 'Q=1.0,dt=0.035'},
        {'Q': 0.6, 'dt': 0.005, 'label': 'Q=0.6,dt=0.005'},
        {'Q': 0.5, 'dt': 0.007, 'label': 'Q=0.5,dt=0.007'},
    ]

    all_results = {}
    for pot_key, pot in potentials.items():
        all_results[pot_key] = []
        for cfg in configs:
            Q, dt, label = cfg['Q'], cfg['dt'], cfg['label']
            print(f"\n  {pot_key}: {label} ... ", end="", flush=True)
            dim = pot.dim
            dyn = LogOscThermostat(dim=dim, kT=1.0, mass=1.0, Q=Q)

            # For DW, use larger dt if configured
            try:
                r = run_sampler(
                    dyn, pot, dt=dt, n_force_evals=N_FORCE_EVALS,
                    kT=1.0, mass=1.0,
                    rng=np.random.default_rng(SEED),
                    integrator_cls=LogOscVelocityVerlet,
                )
                kl = r['kl_divergence'] if r['kl_divergence'] is not None else float('inf')
                erg = r['ergodicity']['score'] if r['ergodicity'] else None
                ess = r['ess_metrics']['ess_per_force_eval'] if r['ess_metrics'] else 0.0
                ttt = r['time_to_threshold_force_evals']
                print(f"KL={kl:.4f}, ESS/fe={ess:.5f}" +
                      (f", erg={erg:.3f}" if erg is not None else "") +
                      (f", TTT={ttt}" if ttt else ""))
                all_results[pot_key].append({
                    'Q': Q, 'dt': dt, 'label': label,
                    'kl': kl, 'ergodicity': erg, 'ess_per_fe': ess,
                    'ttt': ttt,
                    'kl_trace': r['kl_trace'],
                    'wall_seconds': r['wall_seconds'],
                    'nan': r.get('nan_detected', False),
                    'n_samples': r['n_samples'],
                    'energy_distribution': r.get('energy_distribution'),
                })
            except Exception as e:
                print(f"ERROR: {e}")
                all_results[pot_key].append({
                    'Q': Q, 'dt': dt, 'label': label,
                    'kl': float('inf'), 'ergodicity': None, 'ess_per_fe': 0.0,
                    'ttt': None, 'kl_trace': [], 'wall_seconds': 0,
                    'nan': True, 'n_samples': 0, 'energy_distribution': None,
                })

    # Save results (convert for JSON)
    save_results = {}
    for k, v in all_results.items():
        save_results[k] = []
        for item in v:
            item_copy = dict(item)
            item_copy['kl_trace'] = [(int(a), float(b)) for a, b in item_copy['kl_trace']]
            save_results[k].append(item_copy)

    with open(RESULTS_DIR / "task2_stage2.json", 'w') as f:
        json.dump(save_results, f, indent=2)

    return all_results


# ============================================================================
# Task 3: Diagnostic Plots
# ============================================================================

def collect_samples(pot, Q, dt, n_force_evals=N_FORCE_EVALS, chain_length=None):
    """Run sampler and return (q_samples, p_samples, result_dict)."""
    dim = pot.dim
    if chain_length is not None:
        dyn = LogOscChain(dim=dim, chain_length=chain_length, kT=1.0, mass=1.0, Q=Q)
        integ = LogOscChainVerlet
    else:
        dyn = LogOscThermostat(dim=dim, kT=1.0, mass=1.0, Q=Q)
        integ = LogOscVelocityVerlet

    rng = np.random.default_rng(SEED)
    q0 = rng.normal(0, 0.5, size=dim)
    state = dyn.initial_state(q0, rng=rng)
    integrator = integ(dyn, pot, dt, kT=1.0, mass=1.0)

    all_q, all_p = [], []
    burnin = int(n_force_evals * 0.1)
    kl_trace = []
    checkpoint_interval = max(n_force_evals // 20, 1)

    while state.n_force_evals < n_force_evals:
        state = integrator.step(state)
        if np.any(np.isnan(state.q)) or np.any(np.isnan(state.p)):
            break
        if state.n_force_evals >= burnin:
            all_q.append(state.q.copy())
            all_p.append(state.p.copy())
        if state.n_force_evals > 0 and state.n_force_evals % checkpoint_interval < 3:
            if len(all_q) > 100 and dim <= 2:
                q_arr = np.array(all_q)
                kl = kl_divergence_histogram(q_arr, pot, 1.0, n_bins=50)
                kl_trace.append((state.n_force_evals, kl))

    q_samples = np.array(all_q) if all_q else np.zeros((0, dim))
    p_samples = np.array(all_p) if all_p else np.zeros((0, dim))
    return q_samples, p_samples, kl_trace


def plot_density_comparison(pot, q_samples, title, filename, kT=1.0):
    """Plot sample density vs true density (2D heatmap or 1D histogram)."""
    set_plot_style()
    dim = q_samples.shape[1] if len(q_samples) > 0 else pot.dim

    if dim == 1:
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

        # Sample histogram
        ax = axes[0]
        ax.hist(q_samples[:, 0], bins=100, density=True, alpha=0.7, color=COLOR_LOSC, label='Samples')
        q_grid = np.linspace(q_samples[:, 0].min() - 0.5, q_samples[:, 0].max() + 0.5, 500)
        log_p = np.array([-pot.energy(np.array([q])) / kT for q in q_grid])
        log_p -= np.max(log_p)
        p_true = np.exp(log_p)
        p_true /= np.trapz(p_true, q_grid)
        ax.plot(q_grid, p_true, 'k-', lw=2, label='True density')
        ax.set_xlabel('q')
        ax.set_ylabel('Density')
        ax.set_title(f'{title} - Position marginal')
        ax.legend()

        # Phase space
        ax = axes[1]
        ax.scatter(q_samples[::10, 0], q_samples[::10, 0] if q_samples.shape[1] == 1 else q_samples[::10, 1],
                   s=0.1, alpha=0.3, c=COLOR_LOSC)
        ax.set_xlabel('q')
        ax.set_ylabel('q (repeated)' if dim == 1 else 'q2')
        ax.set_title(f'{title} - Samples')

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / filename, dpi=DPI, bbox_inches='tight')
        plt.close()

    elif dim == 2:
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

        # Sample density
        ax = axes[0]
        if len(q_samples) > 0:
            xmin, xmax = q_samples[:, 0].min() - 0.5, q_samples[:, 0].max() + 0.5
            ymin, ymax = q_samples[:, 1].min() - 0.5, q_samples[:, 1].max() + 0.5
            h, xe, ye = np.histogram2d(q_samples[:, 0], q_samples[:, 1], bins=80, density=True)
            ax.pcolormesh(xe, ye, h.T, cmap='viridis', shading='auto')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{title} - Sample density')
        ax.set_aspect('equal')

        # True density
        ax = axes[1]
        n_grid = 100
        if len(q_samples) > 0:
            x_grid = np.linspace(xmin, xmax, n_grid)
            y_grid = np.linspace(ymin, ymax, n_grid)
        else:
            x_grid = np.linspace(-4, 4, n_grid)
            y_grid = np.linspace(-4, 4, n_grid)
        XX, YY = np.meshgrid(x_grid, y_grid)
        Z = np.zeros_like(XX)
        for i in range(n_grid):
            for j in range(n_grid):
                Z[i, j] = -pot.energy(np.array([XX[i, j], YY[i, j]])) / kT
        Z -= np.max(Z)
        Z = np.exp(Z)
        ax.pcolormesh(x_grid, y_grid, Z, cmap='viridis', shading='auto')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{title} - True density')
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / filename, dpi=DPI, bbox_inches='tight')
        plt.close()


def plot_kl_convergence(kl_traces, labels, colors, title, filename):
    """Plot KL convergence curves."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    for trace, label, color in zip(kl_traces, labels, colors):
        if len(trace) > 0:
            evals = [t[0] for t in trace]
            kls = [max(t[1], 1e-6) for t in trace]  # floor for log scale
            ax.plot(evals, kls, '-o', markersize=3, label=label, color=color)

    ax.axhline(y=0.01, color='gray', linestyle='--', alpha=0.7, label='KL=0.01 threshold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Force evaluations')
    ax.set_ylabel('KL divergence')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=DPI, bbox_inches='tight')
    plt.close()


def plot_trajectory_trace(q_samples, title, filename, centers=None):
    """Plot trajectory trace showing mode-hopping (for GMM)."""
    set_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Trajectory in x coordinate over time
    n_show = min(50000, len(q_samples))
    step_idx = np.arange(n_show)
    ax = axes[0]
    ax.plot(step_idx, q_samples[:n_show, 0], lw=0.3, color=COLOR_LOSC, alpha=0.7)
    ax.set_xlabel('Sample index')
    ax.set_ylabel('x')
    ax.set_title(f'{title} - x trajectory')
    if centers is not None:
        for c in centers:
            ax.axhline(y=c[0], color='red', linestyle=':', alpha=0.5)

    ax = axes[1]
    ax.plot(step_idx, q_samples[:n_show, 1], lw=0.3, color=COLOR_LOSC, alpha=0.7)
    ax.set_xlabel('Sample index')
    ax.set_ylabel('y')
    ax.set_title(f'{title} - y trajectory')
    if centers is not None:
        for c in centers:
            ax.axhline(y=c[1], color='red', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=DPI, bbox_inches='tight')
    plt.close()


def plot_energy_distribution(q_samples, p_samples, pot, title, filename, kT=1.0, mass=1.0):
    """Plot energy distribution histogram."""
    set_plot_style()
    energies = np.array([
        0.5 * np.sum(p_samples[i]**2) / mass + pot.energy(q_samples[i])
        for i in range(len(q_samples))
    ])

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.hist(energies, bins=100, density=True, alpha=0.7, color=COLOR_LOSC,
            label=f'Log-Osc (mean={np.mean(energies):.2f}, std={np.std(energies):.2f})')
    ax.set_xlabel('Total Energy')
    ax.set_ylabel('Density')
    ax.set_title(f'{title} - Energy distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=DPI, bbox_inches='tight')
    plt.close()


def plot_phase_space_ho(q_samples, p_samples, title, filename, kT=1.0, omega=1.0, mass=1.0):
    """Phase space scatter for 1D HO with Gaussian contours."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    sigma_q = np.sqrt(kT / omega**2)
    sigma_p = np.sqrt(mass * kT)

    ax.scatter(q_samples[::5], p_samples[::5], s=0.2, alpha=0.2, c=COLOR_LOSC)

    # Gaussian contours
    theta = np.linspace(0, 2 * np.pi, 200)
    for n_sigma in [1, 2, 3]:
        ax.plot(n_sigma * sigma_q * np.cos(theta), n_sigma * sigma_p * np.sin(theta),
                'k-', lw=1, alpha=0.5)
    ax.set_xlabel('q')
    ax.set_ylabel('p')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=DPI, bbox_inches='tight')
    plt.close()


def plot_q_scan_heatmap(results, title, filename):
    """Plot Q vs dt heatmap for KL and ergodicity."""
    set_plot_style()
    q_vals = sorted(set(r['Q'] for r in results))
    dt_vals = sorted(set(r['dt'] for r in results))

    kl_grid = np.full((len(q_vals), len(dt_vals)), np.nan)
    erg_grid = np.full((len(q_vals), len(dt_vals)), np.nan)

    for r in results:
        i = q_vals.index(r['Q'])
        j = dt_vals.index(r['dt'])
        kl_grid[i, j] = r['kl'] if r['kl'] < float('inf') else np.nan
        erg_grid[i, j] = r['ergodicity']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # KL heatmap
    ax = axes[0]
    im = ax.imshow(kl_grid, aspect='auto', origin='lower',
                   extent=[0, len(dt_vals), 0, len(q_vals)],
                   cmap='RdYlGn_r', vmin=0, vmax=0.05)
    ax.set_xticks(np.arange(len(dt_vals)) + 0.5)
    ax.set_xticklabels([f'{d:.3f}' for d in dt_vals], rotation=45)
    ax.set_yticks(np.arange(len(q_vals)) + 0.5)
    ax.set_yticklabels([f'{q:.2f}' for q in q_vals])
    ax.set_xlabel('dt')
    ax.set_ylabel('Q')
    ax.set_title(f'{title} - KL Divergence')
    plt.colorbar(im, ax=ax)

    # Add text annotations
    for i in range(len(q_vals)):
        for j in range(len(dt_vals)):
            if not np.isnan(kl_grid[i, j]):
                ax.text(j + 0.5, i + 0.5, f'{kl_grid[i, j]:.3f}',
                        ha='center', va='center', fontsize=8,
                        color='white' if kl_grid[i, j] > 0.025 else 'black')

    # Ergodicity heatmap
    ax = axes[1]
    im = ax.imshow(erg_grid, aspect='auto', origin='lower',
                   extent=[0, len(dt_vals), 0, len(q_vals)],
                   cmap='RdYlGn', vmin=0.7, vmax=1.0)
    ax.set_xticks(np.arange(len(dt_vals)) + 0.5)
    ax.set_xticklabels([f'{d:.3f}' for d in dt_vals], rotation=45)
    ax.set_yticks(np.arange(len(q_vals)) + 0.5)
    ax.set_yticklabels([f'{q:.2f}' for q in q_vals])
    ax.set_xlabel('dt')
    ax.set_ylabel('Q')
    ax.set_title(f'{title} - Ergodicity Score')
    plt.colorbar(im, ax=ax)

    for i in range(len(q_vals)):
        for j in range(len(dt_vals)):
            if not np.isnan(erg_grid[i, j]):
                ax.text(j + 0.5, i + 0.5, f'{erg_grid[i, j]:.3f}',
                        ha='center', va='center', fontsize=8,
                        color='white' if erg_grid[i, j] < 0.85 else 'black')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=DPI, bbox_inches='tight')
    plt.close()


# ============================================================================
# Task 4: Q-adaptation study
# ============================================================================

def task4_q_adaptation():
    """Study Q-sensitivity across all potentials."""
    print("\n" + "=" * 60)
    print("TASK 4: Q-adaptation study across potentials")
    print("=" * 60)

    potentials = {
        'HO': (HarmonicOscillator1D(omega=1.0), 0.005),
        'DW': (DoubleWell2D(), 0.035),
        'GMM': (GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5), 0.01),
        'Rosenbrock': (Rosenbrock2D(a=0.0, b=5.0), 0.01),
    }

    q_values = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
    results = {}

    for pot_name, (pot, dt) in potentials.items():
        results[pot_name] = []
        for Q in q_values:
            print(f"\n  {pot_name}: Q={Q:.1f}, dt={dt} ... ", end="", flush=True)
            dyn = LogOscThermostat(dim=pot.dim, kT=1.0, mass=1.0, Q=Q)
            try:
                r = run_sampler(
                    dyn, pot, dt=dt, n_force_evals=N_FORCE_EVALS,
                    kT=1.0, mass=1.0,
                    rng=np.random.default_rng(SEED),
                    integrator_cls=LogOscVelocityVerlet,
                )
                kl = r['kl_divergence'] if r['kl_divergence'] is not None else float('inf')
                ess = r['ess_metrics']['ess_per_force_eval'] if r['ess_metrics'] else 0.0
                erg = r['ergodicity']['score'] if r['ergodicity'] else None
                print(f"KL={kl:.4f}, ESS/fe={ess:.5f}")
                results[pot_name].append({'Q': Q, 'kl': kl, 'ess_per_fe': ess, 'ergodicity': erg})
            except Exception as e:
                print(f"ERROR: {e}")
                results[pot_name].append({'Q': Q, 'kl': float('inf'), 'ess_per_fe': 0.0, 'ergodicity': None})

    with open(RESULTS_DIR / "task4_q_adaptation.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Plot Q-sensitivity
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    ax = axes[0]
    for pot_name, res_list in results.items():
        qs = [r['Q'] for r in res_list]
        kls = [r['kl'] if r['kl'] < 10 else np.nan for r in res_list]
        ax.plot(qs, kls, '-o', markersize=5, label=pot_name)
    ax.axhline(y=0.01, color='gray', linestyle='--', alpha=0.7, label='KL=0.01')
    ax.set_xlabel('Q')
    ax.set_ylabel('KL divergence')
    ax.set_title('Q-sensitivity: KL divergence')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for pot_name, res_list in results.items():
        qs = [r['Q'] for r in res_list]
        ess = [r['ess_per_fe'] for r in res_list]
        ax.plot(qs, ess, '-o', markersize=5, label=pot_name)
    ax.set_xlabel('Q')
    ax.set_ylabel('ESS / force eval')
    ax.set_title('Q-sensitivity: Sampling efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "q_sensitivity_all_potentials.png", dpi=DPI, bbox_inches='tight')
    plt.close()

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    print("Log-Osc Stage 2 + Q-Optimization")
    print("Seed:", SEED)
    print()

    # Task 1: Q scan
    q_scan_results = task1_q_scan_ho()
    plot_q_scan_heatmap(q_scan_results, "1D HO Q-scan", "task1_q_scan_heatmap.png")

    # Find best combined config (maximize ergodicity, minimize KL)
    # Score = ergodicity * (1 - min(kl, 0.1)/0.1), want both high ergodicity and low KL
    valid = [r for r in q_scan_results if r['kl'] < float('inf') and r['ergodicity'] > 0.8]
    if valid:
        for r in valid:
            r['combined_score'] = r['ergodicity'] * (1.0 - min(r['kl'], 0.1) / 0.1)
        best = max(valid, key=lambda r: r['combined_score'])
        print(f"\n  Best combined: Q={best['Q']:.2f}, dt={best['dt']:.3f}, "
              f"KL={best['kl']:.4f}, erg={best['ergodicity']:.3f}, "
              f"score={best['combined_score']:.3f}")
        best_Q = best['Q']
        best_dt = best['dt']
    else:
        best_Q, best_dt = 0.7, 0.005
        print(f"\n  No valid combined config found, using Q={best_Q}, dt={best_dt}")

    # Task 1b: Chain scan
    chain_results = task1_chain_scan()

    # Task 2: Stage 2 benchmarks
    stage2_results = task2_stage2_benchmarks(best_Q=best_Q, best_dt=best_dt)

    # Task 3: Diagnostic plots for each potential
    print("\n" + "=" * 60)
    print("TASK 3: Diagnostic Plots")
    print("=" * 60)

    # Find the best config for each potential from task 2 results
    for pot_key in ['ho', 'dw', 'gmm', 'rosenbrock']:
        res_list = stage2_results.get(pot_key, [])
        if not res_list:
            continue
        valid_res = [r for r in res_list if r['kl'] < float('inf') and not r['nan']]
        if not valid_res:
            print(f"  Skipping {pot_key}: no valid results")
            continue
        best_r = min(valid_res, key=lambda r: r['kl'])
        Q, dt = best_r['Q'], best_r['dt']
        print(f"\n  Plotting {pot_key}: Q={Q}, dt={dt}, KL={best_r['kl']:.4f}")

        if pot_key == 'ho':
            pot = HarmonicOscillator1D(omega=1.0)
        elif pot_key == 'dw':
            pot = DoubleWell2D()
        elif pot_key == 'gmm':
            pot = GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5)
        elif pot_key == 'rosenbrock':
            pot = Rosenbrock2D(a=0.0, b=5.0)

        q_samples, p_samples, kl_trace = collect_samples(pot, Q, dt)

        if len(q_samples) == 0:
            print(f"  Skipping {pot_key}: no samples collected")
            continue

        # Density comparison
        plot_density_comparison(pot, q_samples, f"Log-Osc ({pot_key})",
                               f"density_{pot_key}.png")

        # KL convergence
        if kl_trace:
            plot_kl_convergence([kl_trace], [f'Q={Q},dt={dt}'], [COLOR_LOSC],
                                f"KL convergence ({pot_key})", f"kl_convergence_{pot_key}.png")

        # Energy distribution
        plot_energy_distribution(q_samples, p_samples, pot,
                                f"Log-Osc ({pot_key})", f"energy_{pot_key}.png")

        # Phase space for HO
        if pot_key == 'ho':
            plot_phase_space_ho(q_samples[:, 0], p_samples[:, 0],
                               "Log-Osc Phase Space (1D HO)", "phase_space_ho.png")

        # Trajectory trace for GMM
        if pot_key == 'gmm':
            centers = pot.centers
            plot_trajectory_trace(q_samples, "Log-Osc (GMM)", "trajectory_gmm.png",
                                 centers=centers)

    # Task 4: Q-adaptation
    q_adapt_results = task4_q_adaptation()

    print("\n" + "=" * 60)
    print("ALL TASKS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
