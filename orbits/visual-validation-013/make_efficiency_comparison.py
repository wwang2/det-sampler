"""Figure 4: ESS and Efficiency Comparison.

1x3 panel:
  (a) Bar chart: ESS/force-eval by system (grouped bars for NHC, LogOsc, MultiScale, NHCTail)
  (b) Bar chart: Time-to-threshold (force evals to reach KL<0.01)
  (c) Ergodicity score comparison on 1D HO (bar chart with error bars from 5 seeds)
Error bars from 5 seeds.
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
from research.eval.baselines import NoseHooverChain
from research.eval.evaluator import run_sampler

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

sol001 = _load_mod('sol001', 'orbits/log-osc-001/solution.py')
LogOscThermostat = sol001.LogOscThermostat
LogOscVelocityVerlet = sol001.LogOscVelocityVerlet

FIGDIR = os.path.join(WORKTREE, 'orbits/visual-validation-013/figures')
os.makedirs(FIGDIR, exist_ok=True)

SEEDS = [42, 123, 7, 999, 314]
N_FORCE_EVALS = 2_000_000

COLORS = {
    'NHC': '#ff7f0e',
    'LogOsc': '#2ca02c',
    'MultiScale': '#d62728',
    'NHCTail': '#9467bd',
}

SAMPLER_NAMES = ['NHC', 'LogOsc', 'MultiScale', 'NHCTail']

SYSTEMS = [
    ('HO', HarmonicOscillator1D, {}, 1,
     {'NHC': 0.005, 'LogOsc': 0.005, 'MultiScale': 0.005, 'NHCTail': 0.005}),
    ('DW', DoubleWell2D, {}, 2,
     {'NHC': 0.01, 'LogOsc': 0.01, 'MultiScale': 0.035, 'NHCTail': 0.055}),
    ('GMM', GaussianMixture2D, {}, 2,
     {'NHC': 0.01, 'LogOsc': 0.01, 'MultiScale': 0.03, 'NHCTail': 0.03}),
    ('RB', Rosenbrock2D, {}, 2,
     {'NHC': 0.01, 'LogOsc': 0.01, 'MultiScale': 0.03, 'NHCTail': 0.03}),
]


def make_sampler(name, dim):
    if name == 'NHC':
        return NoseHooverChain(dim=dim, chain_length=3, kT=1.0, Q=1.0), None
    elif name == 'LogOsc':
        return LogOscThermostat(dim=dim, kT=1.0, Q=1.0), LogOscVelocityVerlet
    elif name == 'MultiScale':
        return MultiScaleLogOsc(dim=dim, Qs=[0.1, 0.7, 10.0]), MultiScaleLogOscVerlet
    elif name == 'NHCTail':
        return MultiScaleNHCTail(dim=dim, Qs=[0.1, 0.7, 10.0], chain_length=2), MultiScaleNHCTailVerlet


def main():
    print("=== Figure 4: ESS and Efficiency Comparison ===")

    # Collect data: ess_per_fe[system][sampler] = [values over seeds]
    ess_data = {s: {n: [] for n in SAMPLER_NAMES} for s, *_ in SYSTEMS}
    ttt_data = {s: {n: [] for n in SAMPLER_NAMES} for s, *_ in SYSTEMS}
    erg_data = {n: [] for n in SAMPLER_NAMES}

    for sys_label, pot_cls, pot_kwargs, dim, dt_map in SYSTEMS:
        potential = pot_cls(**pot_kwargs)
        print(f"\n--- {sys_label} ---")

        for sname in SAMPLER_NAMES:
            dt = dt_map[sname]
            for seed in SEEDS:
                dynamics, integ_cls = make_sampler(sname, dim)
                rng = np.random.default_rng(seed)
                result = run_sampler(
                    dynamics, potential, dt=dt, n_force_evals=N_FORCE_EVALS,
                    kT=1.0, rng=rng, integrator_cls=integ_cls,
                    kl_checkpoints=40
                )

                # ESS per force eval
                if result['ess_metrics']:
                    ess_data[sys_label][sname].append(result['ess_metrics']['ess_per_force_eval'])
                else:
                    ess_data[sys_label][sname].append(0.0)

                # Time to threshold
                ttt = result['time_to_threshold_force_evals']
                ttt_data[sys_label][sname].append(ttt if ttt is not None else N_FORCE_EVALS * 2)

                # Ergodicity (HO only)
                if sys_label == 'HO' and result['ergodicity']:
                    erg_data[sname].append(result['ergodicity']['score'])

                print(f"  {sname} seed={seed}: ESS/fe={result['ess_metrics']['ess_per_force_eval']:.6f}" if result['ess_metrics'] else f"  {sname} seed={seed}: no ESS")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)

    # Panel (a): ESS/force-eval grouped bar chart
    ax = axes[0]
    sys_labels = [s for s, *_ in SYSTEMS]
    n_systems = len(sys_labels)
    n_samplers = len(SAMPLER_NAMES)
    x = np.arange(n_systems)
    width = 0.18

    for i, sname in enumerate(SAMPLER_NAMES):
        means = [np.mean(ess_data[s][sname]) for s in sys_labels]
        stds = [np.std(ess_data[s][sname]) for s in sys_labels]
        offset = (i - n_samplers / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=sname,
               color=COLORS[sname], alpha=0.85, capsize=3, edgecolor='white', linewidth=0.5)

    ax.set_xlabel('System', fontsize=12)
    ax.set_ylabel('ESS / Force Evaluation', fontsize=12)
    ax.set_title('(a) Sampling Efficiency', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sys_labels, fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.tick_params(labelsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    # Panel (b): Time-to-threshold
    ax = axes[1]
    for i, sname in enumerate(SAMPLER_NAMES):
        means = []
        stds = []
        for s in sys_labels:
            vals = ttt_data[s][sname]
            # Filter out "never reached" (set to 2*budget)
            vals_arr = np.array(vals, dtype=float)
            means.append(np.mean(vals_arr))
            stds.append(np.std(vals_arr))
        offset = (i - n_samplers / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=sname,
               color=COLORS[sname], alpha=0.85, capsize=3, edgecolor='white', linewidth=0.5)

    ax.set_xlabel('System', fontsize=12)
    ax.set_ylabel('Force Evals to KL < 0.01', fontsize=12)
    ax.set_title('(b) Time to Convergence', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sys_labels, fontsize=11)
    ax.set_yscale('log')
    ax.axhline(y=N_FORCE_EVALS, color='gray', linestyle='--', linewidth=1, alpha=0.5,
               label='Budget (2M)')
    ax.legend(fontsize=9, loc='upper right')
    ax.tick_params(labelsize=10)
    ax.grid(True, axis='y', alpha=0.3, which='both')

    # Panel (c): Ergodicity on HO
    ax = axes[2]
    x_erg = np.arange(n_samplers)
    means_erg = [np.mean(erg_data[s]) if erg_data[s] else 0 for s in SAMPLER_NAMES]
    stds_erg = [np.std(erg_data[s]) if erg_data[s] else 0 for s in SAMPLER_NAMES]
    bars = ax.bar(x_erg, means_erg, 0.5, yerr=stds_erg,
                  color=[COLORS[s] for s in SAMPLER_NAMES],
                  alpha=0.85, capsize=5, edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Sampler', fontsize=12)
    ax.set_ylabel('Ergodicity Score', fontsize=12)
    ax.set_title('(c) HO Ergodicity (5 seeds)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_erg)
    ax.set_xticklabels(SAMPLER_NAMES, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.85, color='gray', linestyle='--', linewidth=1, alpha=0.5,
               label='Ergodic threshold')
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    # Annotate bars with values
    for bar, mean in zip(bars, means_erg):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout(w_pad=2.0)

    outpath = os.path.join(FIGDIR, 'fig4_efficiency_comparison.png')
    fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nSaved: {outpath}")


if __name__ == '__main__':
    main()
