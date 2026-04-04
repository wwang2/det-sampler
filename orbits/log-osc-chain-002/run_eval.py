"""Full evaluation of LOCR thermostat + diagnostic plots.

Best configuration: M=3, Q=1.0, alpha=0.0
  - HO: dt=0.015
  - DW: dt=0.06
  - GM2D: dt=0.04
  - Rosenbrock: dt=0.04

Usage:
    cd /Users/wujiewang/code/det-sampler/.worktrees/log-osc-chain-002
    orbits/log-osc-chain-002/.venv/bin/python orbits/log-osc-chain-002/run_eval.py
"""

import sys
import json
import importlib.util
import numpy as np

sys.path.insert(0, '/Users/wujiewang/code/det-sampler/.worktrees/log-osc-chain-002')

from research.eval.evaluator import run_sampler
from research.eval.potentials import (
    HarmonicOscillator1D, DoubleWell2D, GaussianMixture2D, Rosenbrock2D
)

# Import solution
_spec = importlib.util.spec_from_file_location(
    "solution",
    "/Users/wujiewang/code/det-sampler/.worktrees/log-osc-chain-002/orbits/log-osc-chain-002/solution.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LogOscChainRotation = _mod.LogOscChainRotation
LOCRIntegrator = _mod.LOCRIntegrator

SEED = 42
FIGDIR = "/Users/wujiewang/code/det-sampler/.worktrees/log-osc-chain-002/orbits/log-osc-chain-002/figures"
OUTDIR = "/Users/wujiewang/code/det-sampler/.worktrees/log-osc-chain-002/orbits/log-osc-chain-002"


def make_dynamics(dim, M=3, Q=1.0, alpha=0.0, kT=1.0):
    return LogOscChainRotation(dim=dim, chain_length=M, kT=kT, Q=Q, alpha=alpha)


def run_stage1():
    """Run Stage 1 evaluation with multi-seed."""
    results = {}

    # HO
    print("\n=== Stage 1: HO (dt=0.015) ===")
    ho_results = []
    for seed in [42, 123, 456]:
        pot = HarmonicOscillator1D()
        dyn = make_dynamics(dim=1)
        r = run_sampler(dyn, pot, dt=0.015, n_force_evals=1_000_000, kT=1.0,
                        integrator_cls=LOCRIntegrator, rng=np.random.default_rng(seed))
        ho_results.append(r)
        print(f"  seed={seed}: KL={r['kl_divergence']:.4f} erg={r['ergodicity']['score']:.4f} "
              f"ESS/fe={r['ess_metrics']['ess_per_force_eval']:.6f}")

    results['harmonic_1d'] = {
        'kl_mean': float(np.mean([r['kl_divergence'] for r in ho_results])),
        'kl_std': float(np.std([r['kl_divergence'] for r in ho_results])),
        'erg_mean': float(np.mean([r['ergodicity']['score'] for r in ho_results])),
        'erg_std': float(np.std([r['ergodicity']['score'] for r in ho_results])),
        'ess_mean': float(np.mean([r['ess_metrics']['ess_per_force_eval'] for r in ho_results])),
        'dt': 0.015,
        'kl_trace': ho_results[0]['kl_trace'],
    }

    # DW
    print("\n=== Stage 1: DW (dt=0.06) ===")
    dw_results = []
    for seed in [42, 123, 456]:
        pot = DoubleWell2D()
        dyn = make_dynamics(dim=2)
        r = run_sampler(dyn, pot, dt=0.06, n_force_evals=1_000_000, kT=1.0,
                        integrator_cls=LOCRIntegrator, rng=np.random.default_rng(seed))
        dw_results.append(r)
        print(f"  seed={seed}: KL={r['kl_divergence']:.4f} "
              f"ESS/fe={r['ess_metrics']['ess_per_force_eval']:.6f} "
              f"TTT={r['time_to_threshold_force_evals']}")

    results['double_well_2d'] = {
        'kl_mean': float(np.mean([r['kl_divergence'] for r in dw_results])),
        'kl_std': float(np.std([r['kl_divergence'] for r in dw_results])),
        'ess_mean': float(np.mean([r['ess_metrics']['ess_per_force_eval'] for r in dw_results])),
        'ttt_mean': float(np.mean([r['time_to_threshold_force_evals'] or 1e7 for r in dw_results])),
        'dt': 0.06,
        'kl_trace': dw_results[0]['kl_trace'],
    }

    return results, ho_results[0], dw_results[0]


def run_stage2():
    """Run Stage 2 evaluation."""
    results = {}

    # GM2D
    print("\n=== Stage 2: GM2D (dt=0.04) ===")
    pot = GaussianMixture2D()
    dyn = make_dynamics(dim=2)
    r = run_sampler(dyn, pot, dt=0.04, n_force_evals=1_000_000, kT=1.0,
                    integrator_cls=LOCRIntegrator, rng=np.random.default_rng(SEED))
    print(f"  KL={r['kl_divergence']:.4f} ESS/fe={r['ess_metrics']['ess_per_force_eval']:.6f}")
    results['gaussian_mixture_2d'] = {
        'kl': float(r['kl_divergence']),
        'ess': float(r['ess_metrics']['ess_per_force_eval']),
        'dt': 0.04,
        'kl_trace': r['kl_trace'],
    }

    # Rosenbrock
    print("\n=== Stage 2: Rosenbrock (dt=0.04) ===")
    pot = Rosenbrock2D()
    dyn = make_dynamics(dim=2)
    r = run_sampler(dyn, pot, dt=0.04, n_force_evals=1_000_000, kT=1.0,
                    integrator_cls=LOCRIntegrator, rng=np.random.default_rng(SEED))
    print(f"  KL={r['kl_divergence']:.4f} ESS/fe={r['ess_metrics']['ess_per_force_eval']:.6f} "
          f"TTT={r['time_to_threshold_force_evals']}")
    results['rosenbrock_2d'] = {
        'kl': float(r['kl_divergence']),
        'ess': float(r['ess_metrics']['ess_per_force_eval']),
        'ttt': r['time_to_threshold_force_evals'],
        'dt': 0.04,
        'kl_trace': r['kl_trace'],
    }

    return results


def make_plots(ho_result, dw_result):
    """Generate diagnostic plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats

    # Style settings from research/style.md
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })

    # ---- Plot 1: Phase space coverage (HO) ----
    print("\nGenerating phase space plot...")
    # Rerun to get raw samples
    pot = HarmonicOscillator1D()
    dyn = make_dynamics(dim=1)
    rng = np.random.default_rng(42)
    state = dyn.initial_state(np.array([0.5]), rng=rng)
    integrator = LOCRIntegrator(dyn, pot, dt=0.015, kT=1.0, mass=1.0)

    qs, ps = [], []
    for i in range(1_000_000):
        state = integrator.step(state)
        if state.n_force_evals > 1_000_000:
            break
        if i > 10000 and i % 5 == 0:
            qs.append(state.q[0])
            ps.append(state.p[0])

    qs = np.array(qs)
    ps = np.array(ps)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
    # Subsample for plotting
    n_plot = min(20000, len(qs))
    idx = np.random.default_rng(0).choice(len(qs), n_plot, replace=False)
    ax.scatter(qs[idx], ps[idx], s=0.5, alpha=0.3, c='#2ca02c', rasterized=True)

    # Gaussian contours
    sigma_q = 1.0
    sigma_p = 1.0
    theta = np.linspace(0, 2 * np.pi, 200)
    for n_sigma in [1, 2, 3]:
        ax.plot(n_sigma * sigma_q * np.cos(theta), n_sigma * sigma_p * np.sin(theta),
                '--', color='gray', linewidth=1, alpha=0.7,
                label=f'{n_sigma}$\\sigma$' if n_sigma == 1 else f'{n_sigma}$\\sigma$')

    ax.set_xlabel('q')
    ax.set_ylabel('p')
    ax.set_title('LOCR Phase Space Coverage (1D HO)')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/phase_space_ho.png', dpi=150)
    plt.close(fig)
    print(f"  Saved {FIGDIR}/phase_space_ho.png")

    # ---- Plot 2: DW density comparison ----
    print("Generating DW density comparison...")
    pot = DoubleWell2D()
    dyn = make_dynamics(dim=2)
    rng = np.random.default_rng(42)
    state = dyn.initial_state(np.array([0.5, 0.0]), rng=rng)
    integrator = LOCRIntegrator(dyn, pot, dt=0.06, kT=1.0, mass=1.0)

    dw_qs = []
    for i in range(1_000_000):
        state = integrator.step(state)
        if state.n_force_evals > 1_000_000:
            break
        if i > 10000 and i % 5 == 0:
            dw_qs.append(state.q.copy())

    dw_qs = np.array(dw_qs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Sampled density
    ax = axes[0]
    h, xedges, yedges = np.histogram2d(dw_qs[:, 0], dw_qs[:, 1], bins=60,
                                         range=[[-3, 3], [-3, 3]], density=True)
    im = ax.imshow(h.T, origin='lower', extent=[-3, 3, -3, 3], cmap='viridis', aspect='equal')
    ax.set_xlabel('$q_1$')
    ax.set_ylabel('$q_2$')
    ax.set_title('LOCR Sampled Density')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # True density
    ax = axes[1]
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    XX, YY = np.meshgrid(x, y)
    Z = np.zeros_like(XX)
    for i in range(len(x)):
        for j in range(len(y)):
            Z[j, i] = np.exp(-pot.energy(np.array([x[i], y[j]])))
    Z /= Z.sum() * (x[1] - x[0]) * (y[1] - y[0])
    im = ax.imshow(Z, origin='lower', extent=[-3, 3, -3, 3], cmap='viridis', aspect='equal')
    ax.set_xlabel('$q_1$')
    ax.set_ylabel('$q_2$')
    ax.set_title('True Boltzmann Density')
    plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Double-Well 2D: LOCR vs True', fontsize=16)
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/density_dw.png', dpi=150)
    plt.close(fig)
    print(f"  Saved {FIGDIR}/density_dw.png")

    # ---- Plot 3: KL convergence traces ----
    print("Generating KL convergence plot...")
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)

    if ho_result['kl_trace']:
        evals_ho, kls_ho = zip(*ho_result['kl_trace'])
        ax.plot(evals_ho, kls_ho, '-', color='#2ca02c', linewidth=2, label='LOCR - HO')

    if dw_result['kl_trace']:
        evals_dw, kls_dw = zip(*dw_result['kl_trace'])
        ax.plot(evals_dw, kls_dw, '-', color='#d62728', linewidth=2, label='LOCR - DW')

    # Baselines
    ax.axhline(y=0.01, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='KL=0.01 threshold')
    ax.axhline(y=0.029, color='#ff7f0e', linestyle=':', linewidth=1, alpha=0.5, label='NHC DW baseline')
    ax.axhline(y=0.002, color='#1f77b4', linestyle=':', linewidth=1, alpha=0.5, label='NHC HO baseline')

    ax.set_xlabel('Force evaluations')
    ax.set_ylabel('KL divergence')
    ax.set_title('LOCR KL Convergence (Stage 1)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax.set_ylim(1e-4, 10)
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/kl_convergence.png', dpi=150)
    plt.close(fig)
    print(f"  Saved {FIGDIR}/kl_convergence.png")


def main():
    import os
    os.makedirs(FIGDIR, exist_ok=True)

    print("=" * 60)
    print("LOCR Thermostat Evaluation")
    print("Config: M=3, Q=1.0, alpha=0.0")
    print("=" * 60)

    # Stage 1
    s1_results, ho_full, dw_full = run_stage1()

    # Stage 2
    s2_results = run_stage2()

    # Plots
    make_plots(ho_full, dw_full)

    # Save all results
    all_results = {**s1_results, **s2_results}
    # Remove non-serializable traces for JSON
    for k, v in all_results.items():
        if 'kl_trace' in v:
            v['kl_trace'] = [(int(e), float(kl)) for e, kl in v['kl_trace']]

    with open(f'{OUTDIR}/eval_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Potential':<20} {'KL':>10} {'Ergodicity':>12} {'ESS/fe':>10} {'dt':>6}")
    print("-" * 60)
    for name, r in all_results.items():
        kl = r.get('kl_mean', r.get('kl', 0))
        erg = r.get('erg_mean', '')
        ess = r.get('ess_mean', r.get('ess', 0))
        dt = r.get('dt', '')
        erg_str = f'{erg:.4f}' if isinstance(erg, float) else 'N/A'
        print(f"{name:<20} {kl:>10.4f} {erg_str:>12} {ess:>10.6f} {dt:>6}")

    print("\n--- vs Baselines ---")
    ho = all_results['harmonic_1d']
    dw = all_results['double_well_2d']
    print(f"        | erg     | HO KL   | DW KL   | HO ESS/fe | DW ESS/fe")
    print(f"NHC     | 0.920   | 0.002   | 0.029   | 0.00431   | 0.00261  ")
    print(f"LOC-001 | 0.944   | 0.023   | 0.010   | ---       | ---      ")
    print(f"LOCR    | {ho['erg_mean']:.3f}   | {ho['kl_mean']:.3f}   | {dw['kl_mean']:.3f}   | {ho['ess_mean']:.5f} | {dw['ess_mean']:.5f}")


if __name__ == "__main__":
    main()
