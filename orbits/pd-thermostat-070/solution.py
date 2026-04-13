"""PD Thermostat: Nose-Hoover with derivative feedback term.

Phase 0 result: the D-term breaks canonical measure preservation.
This code numerically confirms the analytical finding by measuring
KL divergence as a function of K_d.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from research.eval.integrators import ThermostatState, VelocityVerletThermostat
from research.eval.evaluator import run_sampler, kl_divergence_histogram
from research.eval.potentials import DoubleWell2D, HarmonicOscillator1D, GaussianMixture2D
from research.eval.baselines import NoseHoover, NoseHooverChain


class PDThermostat:
    """Nose-Hoover with proportional-derivative feedback.

    dq/dt = p/m
    dp/dt = -nabla_U - tanh(xi)*p
    dxi/dt = (|p|^2/m - D*kT)/Q + K_d * 2*p.(-nabla_U - tanh(xi)*p)/m

    Phase 0 proved this does NOT preserve the canonical measure for K_d != 0.
    """

    def __init__(self, dim: int, kT: float = 1.0, mass: float = 1.0,
                 Q: float = 1.0, K_d: float = 0.0):
        self.dim = dim
        self.kT = kT
        self.mass = mass
        self.Q = Q
        self.K_d = K_d
        self.name = f"pd_thermostat_Kd={K_d}"

    def initial_state(self, q0: np.ndarray, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        xi0 = np.array([0.0])
        return ThermostatState(q0.copy(), p0, xi0, 0)

    def dqdt(self, state, grad_U):
        return state.p / self.mass

    def dpdt(self, state, grad_U):
        return -grad_U - np.tanh(state.xi[0]) * state.p

    def dxidt(self, state, grad_U):
        p = state.p
        xi = state.xi[0]
        kinetic = np.sum(p**2) / self.mass

        # Standard NH term (proportional/integral)
        nh_term = (kinetic - self.dim * self.kT) / self.Q

        # Derivative term: K_d * d(|p|^2/m)/dt = K_d * 2*p.dp/dt / m
        # dp/dt = -grad_U - tanh(xi)*p
        dpdt = -grad_U - np.tanh(xi) * p
        d_kinetic_dt = 2.0 * np.dot(p, dpdt) / self.mass
        d_term = self.K_d * d_kinetic_dt

        return np.array([nh_term + d_term])


def run_kd_sweep(seed=42, n_force_evals=500_000):
    """Sweep K_d values and measure KL divergence on double-well."""
    pot = DoubleWell2D()
    kd_values = [0.0, 0.01, 0.05, 0.1, 0.5]
    results = {}

    rng = np.random.default_rng(seed)
    q0 = rng.normal(0, 0.5, size=pot.dim)

    for kd in kd_values:
        dyn = PDThermostat(dim=2, kT=1.0, Q=1.0, K_d=kd)
        r = run_sampler(
            dyn, pot, dt=0.01, n_force_evals=n_force_evals,
            kT=1.0, q0=q0.copy(),
            rng=np.random.default_rng(seed)
        )
        results[kd] = {
            'kl': r['kl_divergence'],
            'ess': r['ess_metrics']['ess_per_force_eval'] if r['ess_metrics'] else 0.0,
            'tau': r['ess_metrics']['tau'] if r['ess_metrics'] else float('inf'),
            'wall_seconds': r['wall_seconds'],
            'nan': r.get('nan_detected', False),
            'n_samples': r['n_samples'],
        }

    return results


def _run_one_seed(args):
    """Top-level function for multiprocessing pickling."""
    seed, n_force_evals = args
    return (seed, run_kd_sweep(seed=seed, n_force_evals=n_force_evals))


def run_parallel_seeds(seeds=[42, 123, 7], n_force_evals=500_000):
    """Run K_d sweep across multiple seeds in parallel."""
    from multiprocessing import Pool

    args = [(s, n_force_evals) for s in seeds]
    with Pool(min(len(seeds), os.cpu_count() or 1)) as pool:
        all_results = pool.map(_run_one_seed, args)

    return dict(all_results)


def make_figure(all_seed_results, output_path):
    """Create 3-panel figure: (a) KL vs K_d, (b) ESS vs K_d, (c) tau vs K_d."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.pad_inches': 0.2,
    })

    kd_values = [0.0, 0.01, 0.05, 0.1, 0.5]
    seeds = sorted(all_seed_results.keys())

    # Collect per-Kd statistics across seeds
    kl_means, kl_stds = [], []
    ess_means, ess_stds = [], []
    tau_means, tau_stds = [], []

    for kd in kd_values:
        kls = [all_seed_results[s][kd]['kl'] for s in seeds
               if all_seed_results[s][kd]['kl'] is not None
               and not np.isinf(all_seed_results[s][kd]['kl'])]
        ess_vals = [all_seed_results[s][kd]['ess'] for s in seeds]
        tau_vals = [all_seed_results[s][kd]['tau'] for s in seeds
                    if not np.isinf(all_seed_results[s][kd]['tau'])]

        kl_means.append(np.mean(kls) if kls else float('nan'))
        kl_stds.append(np.std(kls) if len(kls) > 1 else 0.0)
        ess_means.append(np.mean(ess_vals))
        ess_stds.append(np.std(ess_vals) if len(ess_vals) > 1 else 0.0)
        tau_means.append(np.mean(tau_vals) if tau_vals else float('nan'))
        tau_stds.append(np.std(tau_vals) if len(tau_vals) > 1 else 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    # NHC baseline KL for reference
    nhc_kl = 0.029  # from config.yaml baselines

    # (a) KL vs K_d
    ax = axes[0]
    ax.errorbar(kd_values, kl_means, yerr=kl_stds, fmt='o-', color='#2ca02c',
                capsize=4, linewidth=2, markersize=8, label='PD thermostat')
    ax.axhline(nhc_kl, color='#ff7f0e', linestyle='--', linewidth=1.5, label='NHC(M=3)')
    ax.axhline(0.01, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='KL=0.01 target')
    ax.set_xlabel('$K_d$ (derivative gain)')
    ax.set_ylabel('KL divergence')
    ax.set_title('(a) KL vs derivative gain', fontweight='bold')
    ax.legend(frameon=False, fontsize=10)
    ax.set_yscale('log')

    # (b) ESS/force_eval vs K_d
    ax = axes[1]
    ax.errorbar(kd_values, ess_means, yerr=ess_stds, fmt='s-', color='#2ca02c',
                capsize=4, linewidth=2, markersize=8)
    ax.axhline(0.00261, color='#ff7f0e', linestyle='--', linewidth=1.5, label='NHC(M=3)')
    ax.set_xlabel('$K_d$ (derivative gain)')
    ax.set_ylabel('ESS / force eval')
    ax.set_title('(b) Sampling efficiency', fontweight='bold')
    ax.legend(frameon=False, fontsize=10)

    # (c) Autocorrelation time vs K_d
    ax = axes[2]
    ax.errorbar(kd_values, tau_means, yerr=tau_stds, fmt='D-', color='#2ca02c',
                capsize=4, linewidth=2, markersize=8)
    ax.set_xlabel('$K_d$ (derivative gain)')
    ax.set_ylabel('Autocorrelation time $\\tau_{int}$')
    ax.set_title('(c) Mixing time', fontweight='bold')

    fig.suptitle('PD Thermostat: derivative feedback breaks canonical measure',
                 fontsize=14, fontweight='bold', y=1.02)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {output_path}")


if __name__ == '__main__':
    import json

    print("=" * 60)
    print("PD Thermostat: Numerical verification of measure violation")
    print("=" * 60)

    # Run parallel seeds
    seeds = [42, 123, 7]
    print(f"\nRunning K_d sweep with seeds {seeds} in parallel...")
    all_results = run_parallel_seeds(seeds=seeds, n_force_evals=500_000)

    # Print results table
    kd_values = [0.0, 0.01, 0.05, 0.1, 0.5]
    print(f"\n{'K_d':>6} | {'KL (mean +/- std)':>25} | {'ESS/eval (mean)':>15} | {'tau (mean)':>12}")
    print("-" * 70)
    for kd in kd_values:
        kls = [all_results[s][kd]['kl'] for s in seeds
               if all_results[s][kd]['kl'] is not None
               and not np.isinf(all_results[s][kd]['kl'])]
        ess_vals = [all_results[s][kd]['ess'] for s in seeds]
        tau_vals = [all_results[s][kd]['tau'] for s in seeds
                    if not np.isinf(all_results[s][kd]['tau'])]

        kl_m = np.mean(kls) if kls else float('nan')
        kl_s = np.std(kls) if len(kls) > 1 else 0.0
        ess_m = np.mean(ess_vals)
        tau_m = np.mean(tau_vals) if tau_vals else float('nan')

        print(f"{kd:>6.2f} | {kl_m:>10.4f} +/- {kl_s:<10.4f} | {ess_m:>15.6f} | {tau_m:>12.1f}")

    # Save raw results
    orbit_dir = os.path.dirname(os.path.abspath(__file__))

    # Convert to JSON-safe format
    json_results = {}
    for seed, res in all_results.items():
        json_results[str(seed)] = {}
        for kd, vals in res.items():
            json_results[str(seed)][str(kd)] = {
                k: (None if v is not None and (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else v)
                for k, v in vals.items()
            }

    results_path = os.path.join(orbit_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Make figure
    fig_path = os.path.join(orbit_dir, 'figures', 'pd_comparison.png')
    make_figure(all_results, fig_path)

    print("\nDone.")
