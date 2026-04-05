"""Figure 3: Convergence Studies with Error Bands.

For each system: KL divergence vs force evaluations.
Mean line + shaded std band over 5 seeds for: NHC, LogOsc, MultiScale, NHCTail.
2x2 grid: (a) HO KL, (b) DW KL, (c) GMM KL, (d) Rosenbrock KL.
Log-log axes, threshold line at KL=0.01.
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
from research.eval.integrators import VelocityVerletThermostat
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
KL_CHECKPOINTS = 40  # more checkpoints for smoother curves

# Style colors per style.md
COLORS = {
    'NHC': '#ff7f0e',       # orange
    'LogOsc': '#2ca02c',    # tab10 index 2 (green)
    'MultiScale': '#d62728', # tab10 index 3 (red)
    'NHCTail': '#9467bd',   # tab10 index 4 (purple)
}

# System configs: (potential_cls, kwargs, dim, dt_map, label)
SYSTEMS = [
    (HarmonicOscillator1D, {}, 1, {
        'NHC': 0.005, 'LogOsc': 0.005, 'MultiScale': 0.005, 'NHCTail': 0.005
    }, 'Harmonic Oscillator'),
    (DoubleWell2D, {}, 2, {
        'NHC': 0.01, 'LogOsc': 0.01, 'MultiScale': 0.035, 'NHCTail': 0.055
    }, 'Double Well'),
    (GaussianMixture2D, {}, 2, {
        'NHC': 0.01, 'LogOsc': 0.01, 'MultiScale': 0.03, 'NHCTail': 0.03
    }, 'Gaussian Mixture'),
    (Rosenbrock2D, {}, 2, {
        'NHC': 0.01, 'LogOsc': 0.01, 'MultiScale': 0.03, 'NHCTail': 0.03
    }, 'Rosenbrock'),
]


def make_sampler(name, dim):
    """Create sampler dynamics and integrator class."""
    if name == 'NHC':
        return NoseHooverChain(dim=dim, chain_length=3, kT=1.0, Q=1.0), None
    elif name == 'LogOsc':
        return LogOscThermostat(dim=dim, kT=1.0, Q=1.0), LogOscVelocityVerlet
    elif name == 'MultiScale':
        return MultiScaleLogOsc(dim=dim, Qs=[0.1, 0.7, 10.0]), MultiScaleLogOscVerlet
    elif name == 'NHCTail':
        return MultiScaleNHCTail(dim=dim, Qs=[0.1, 0.7, 10.0], chain_length=2), MultiScaleNHCTailVerlet
    else:
        raise ValueError(f"Unknown sampler: {name}")


def interpolate_kl_trace(kl_trace, eval_points):
    """Interpolate KL trace to common eval points."""
    if len(kl_trace) < 2:
        return np.full(len(eval_points), np.nan)
    evals = np.array([e for e, k in kl_trace])
    kls = np.array([k for e, k in kl_trace])
    # Clip inf/nan
    mask = np.isfinite(kls)
    if mask.sum() < 2:
        return np.full(len(eval_points), np.nan)
    evals = evals[mask]
    kls = kls[mask]
    return np.interp(eval_points, evals, kls, left=np.nan, right=kls[-1])


def main():
    print("=== Figure 3: Convergence Studies ===")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axes = axes.flatten()

    sampler_names = ['NHC', 'LogOsc', 'MultiScale', 'NHCTail']
    # Common eval points for interpolation
    eval_points = np.logspace(np.log10(50000), np.log10(N_FORCE_EVALS), 30)

    for sys_idx, (pot_cls, pot_kwargs, dim, dt_map, label) in enumerate(SYSTEMS):
        ax = axes[sys_idx]
        print(f"\n--- {label} ---")
        potential = pot_cls(**pot_kwargs)

        for sname in sampler_names:
            dt = dt_map[sname]
            all_traces = []

            for seed in SEEDS:
                dynamics, integ_cls = make_sampler(sname, dim)
                rng = np.random.default_rng(seed)
                result = run_sampler(
                    dynamics, potential, dt=dt, n_force_evals=N_FORCE_EVALS,
                    kT=1.0, rng=rng, integrator_cls=integ_cls,
                    kl_checkpoints=KL_CHECKPOINTS
                )
                trace = result['kl_trace']
                interp_kl = interpolate_kl_trace(trace, eval_points)
                all_traces.append(interp_kl)
                print(f"  {sname} seed={seed}: final KL={result['kl_divergence']:.4f}" if result['kl_divergence'] is not None else f"  {sname} seed={seed}: KL=None")

            traces = np.array(all_traces)
            mean_kl = np.nanmean(traces, axis=0)
            std_kl = np.nanstd(traces, axis=0)

            ax.plot(eval_points, mean_kl, color=COLORS[sname], linewidth=2.0,
                    label=sname, alpha=0.9)
            ax.fill_between(eval_points, mean_kl - std_kl, mean_kl + std_kl,
                           color=COLORS[sname], alpha=0.15)

        # Threshold line
        ax.axhline(y=0.01, color='gray', linestyle='--', linewidth=1.0, alpha=0.7,
                   label='KL=0.01 threshold')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Force Evaluations', fontsize=12)
        ax.set_ylabel('KL Divergence', fontsize=12)
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.tick_params(labelsize=10)
        ax.set_ylim(bottom=1e-4, top=10)
        ax.grid(True, alpha=0.3, which='both')

        if sys_idx == 0:
            ax.legend(fontsize=10, loc='upper right', framealpha=0.9)

    plt.tight_layout(h_pad=2.0, w_pad=2.0)

    outpath = os.path.join(FIGDIR, 'fig3_convergence_studies.png')
    fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nSaved: {outpath}")


if __name__ == '__main__':
    main()
