#!/usr/bin/env python3
"""Figure 4: 'Quantitative Comparison' -- Performance across benchmarks.

3-panel (1x3) layout:
  (a) Ergodicity score by sampler (1D HO)
  (b) KL divergence on 2D double-well
  (c) KL divergence on 5-mode GMM
"""

import sys, os, importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import kstest

from research.eval.potentials import HarmonicOscillator1D, DoubleWell2D, GaussianMixture2D
from research.eval.baselines import NoseHoover, NoseHooverChain
from research.eval.integrators import VelocityVerletThermostat

# Import Log-Osc
_base = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "log_osc_001", os.path.join(_base, '..', 'log-osc-001', 'solution.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LogOscThermostat = _mod.LogOscThermostat
LogOscVelocityVerlet = _mod.LogOscVelocityVerlet
LogOscChain = _mod.LogOscChain
LogOscChainVerlet = _mod.LogOscChainVerlet

# Import Multi-Scale Log-Osc
_spec2 = importlib.util.spec_from_file_location(
    "log_osc_multiT_005", os.path.join(_base, '..', 'log-osc-multiT-005', 'solution.py'))
_mod2 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_mod2)
MultiScaleLogOsc = _mod2.MultiScaleLogOsc
MultiScaleLogOscVerlet = _mod2.MultiScaleLogOscVerlet

# ── Style constants ──
COLOR_NH = '#1f77b4'
COLOR_NHC = '#ff7f0e'
tab10 = plt.cm.tab10
COLOR_LOGOSC = tab10(2)
COLOR_LOCR = tab10(3)
COLOR_MSLO = tab10(4)
COLOR_NHCTAIL = tab10(5)
FONTSIZE_LABEL = 14
FONTSIZE_TICK = 12
FONTSIZE_TITLE = 16
DPI = 300

KT = 1.0
N_FORCE_EVALS = 1_000_000
SEEDS = [42, 123, 7]


# ── Sampler configs ──
SAMPLERS = {
    'NH': {
        'cls': NoseHoover, 'integ': VelocityVerletThermostat,
        'color': COLOR_NH, 'params': {'Q': 1.0}, 'dt': 0.01,
    },
    'NHC': {
        'cls': NoseHooverChain, 'integ': VelocityVerletThermostat,
        'color': COLOR_NHC, 'params': {'Q': 1.0, 'chain_length': 3}, 'dt': 0.01,
    },
    'Log-Osc': {
        'cls': LogOscThermostat, 'integ': LogOscVelocityVerlet,
        'color': COLOR_LOGOSC, 'params': {'Q': 0.5}, 'dt': 0.01,
    },
    'LOCR': {
        'cls': LogOscChain, 'integ': LogOscChainVerlet,
        'color': COLOR_LOCR, 'params': {'Q': 1.0, 'chain_length': 2}, 'dt': 0.008,
    },
    'MSLO': {
        'cls': MultiScaleLogOsc, 'integ': MultiScaleLogOscVerlet,
        'color': COLOR_MSLO, 'params': {'Qs': [0.1, 1.0, 10.0]}, 'dt': 0.03,
    },
}


def run_sampler(config, potential, dim, n_evals, seed):
    """Run a sampler and collect trajectory."""
    rng = np.random.default_rng(seed)
    dyn = config['cls'](dim=dim, kT=KT, mass=1.0, **config['params'])
    q0 = rng.normal(0, 0.3, size=dim)
    state = dyn.initial_state(q0, rng=rng)
    dt = config['dt']
    integrator = config['integ'](dyn, potential, dt=dt, kT=KT, mass=1.0)

    # Estimate steps needed (1 force eval per step for Verlet after first)
    n_steps = n_evals
    qs = []
    ps = []
    for i in range(n_steps):
        if state.n_force_evals >= n_evals:
            break
        qs.append(state.q.copy())
        ps.append(state.p.copy())
        state = integrator.step(state)

    return np.array(qs), np.array(ps)


def compute_ergodicity_score(qs, ps, kT=1.0):
    """Compute ergodicity score for 1D HO samples."""
    burn = len(qs) // 10
    q_post = qs[burn:, 0]
    p_post = ps[burn:, 0]

    ks_q = kstest(q_post, 'norm', args=(0, np.sqrt(kT))).statistic
    ks_p = kstest(p_post, 'norm', args=(0, np.sqrt(kT))).statistic
    ks_comp = max(1.0 - max(ks_q, ks_p), 0.01)

    var_q_err = abs(np.var(q_post) / kT - 1.0)
    var_p_err = abs(np.var(p_post) / kT - 1.0)
    var_comp = max(1.0 - max(var_q_err, var_p_err), 0.01)

    q_edges = np.linspace(-4, 4, 21)
    p_edges = np.linspace(-4, 4, 21)
    H, _, _ = np.histogram2d(q_post, p_post, bins=[q_edges, p_edges])
    coverage = max(np.sum(H > 0) / 400.0, 0.01)

    return (ks_comp * var_comp * coverage) ** (1.0/3.0)


def compute_kl_2d(qs, potential, kT=1.0, n_bins=50):
    """Compute KL divergence for 2D potential via histogram."""
    burn = len(qs) // 10
    q_post = qs[burn:]

    # Determine range
    q_range = 4.0
    edges_x = np.linspace(-q_range, q_range, n_bins + 1)
    edges_y = np.linspace(-q_range, q_range, n_bins + 1)

    # Empirical histogram
    H_emp, _, _ = np.histogram2d(q_post[:, 0], q_post[:, 1], bins=[edges_x, edges_y], density=True)

    # Analytical (unnormalized Boltzmann)
    cx = 0.5 * (edges_x[:-1] + edges_x[1:])
    cy = 0.5 * (edges_y[:-1] + edges_y[1:])
    CX, CY = np.meshgrid(cx, cy, indexing='ij')
    H_theory = np.zeros_like(CX)
    for i in range(n_bins):
        for j in range(n_bins):
            H_theory[i, j] = np.exp(-potential.energy(np.array([cx[i], cy[j]])) / kT)
    H_theory /= (H_theory.sum() * (edges_x[1] - edges_x[0]) * (edges_y[1] - edges_y[0]))

    # KL divergence: sum p * log(p/q) where p = empirical
    eps = 1e-10
    H_emp_safe = H_emp + eps
    H_theory_safe = H_theory + eps
    kl = np.sum(H_emp_safe * np.log(H_emp_safe / H_theory_safe)) * \
         (edges_x[1] - edges_x[0]) * (edges_y[1] - edges_y[0])
    return max(kl, 0.0)


def make_figure():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    sampler_names = list(SAMPLERS.keys())
    colors = [SAMPLERS[s]['color'] for s in sampler_names]
    n_samplers = len(sampler_names)
    x = np.arange(n_samplers)
    width = 0.6

    # ── Panel (a): Ergodicity score (1D HO) ──
    ax = axes[0]
    print("Computing ergodicity scores...")
    pot_ho = HarmonicOscillator1D(omega=1.0)
    ergo_means = []
    ergo_stds = []
    for name in sampler_names:
        scores = []
        for seed in SEEDS:
            qs, ps = run_sampler(SAMPLERS[name], pot_ho, dim=1, n_evals=N_FORCE_EVALS, seed=seed)
            scores.append(compute_ergodicity_score(qs, ps))
        ergo_means.append(np.mean(scores))
        ergo_stds.append(np.std(scores))
        print(f"  {name}: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")

    bars = ax.bar(x, ergo_means, width, yerr=ergo_stds, capsize=4,
                  color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.axhline(0.85, color='gray', ls='--', lw=1.2, label='Ergodic threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(sampler_names, fontsize=FONTSIZE_TICK, rotation=15)
    ax.set_ylabel('Ergodicity Score', fontsize=FONTSIZE_LABEL)
    ax.set_ylim(0, 1.1)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_title('1D Harmonic Oscillator', fontsize=13)
    ax.text(0.03, 0.95, '(a)', transform=ax.transAxes, fontsize=FONTSIZE_LABEL,
            fontweight='bold', va='top')

    # ── Panel (b): KL on double-well ──
    ax = axes[1]
    print("Computing KL on double-well...")
    pot_dw = DoubleWell2D(barrier_height=1.0, y_stiffness=0.5)
    kl_dw_means = []
    kl_dw_stds = []
    for name in sampler_names:
        kls = []
        for seed in SEEDS:
            qs, _ = run_sampler(SAMPLERS[name], pot_dw, dim=2, n_evals=N_FORCE_EVALS, seed=seed)
            kls.append(compute_kl_2d(qs, pot_dw))
        kl_dw_means.append(np.mean(kls))
        kl_dw_stds.append(np.std(kls))
        print(f"  {name}: {np.mean(kls):.4f} +/- {np.std(kls):.4f}")

    bars = ax.bar(x, kl_dw_means, width, color=colors, edgecolor='black',
                  linewidth=0.5, alpha=0.85)
    # Add value labels on top of bars
    for i, (m, s) in enumerate(zip(kl_dw_means, kl_dw_stds)):
        ax.text(i, m * 1.15 + s, f'{m:.3f}', ha='center', va='bottom', fontsize=8)
    ax.axhline(0.01, color='gray', ls='--', lw=1.2, label='Target KL=0.01')
    ax.set_xticks(x)
    ax.set_xticklabels(sampler_names, fontsize=FONTSIZE_TICK, rotation=15)
    ax.set_ylabel('KL Divergence', fontsize=FONTSIZE_LABEL)
    ax.set_yscale('log')
    ax.set_ylim(0.001, 20)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_title('2D Double-Well', fontsize=13)
    ax.text(0.03, 0.95, '(b)', transform=ax.transAxes, fontsize=FONTSIZE_LABEL,
            fontweight='bold', va='top')

    # ── Panel (c): KL on GMM ──
    ax = axes[2]
    print("Computing KL on GMM...")
    pot_gmm = GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5)
    kl_gmm_means = []
    kl_gmm_stds = []
    for name in sampler_names:
        kls = []
        for seed in SEEDS:
            qs, _ = run_sampler(SAMPLERS[name], pot_gmm, dim=2, n_evals=N_FORCE_EVALS, seed=seed)
            kls.append(compute_kl_2d(qs, pot_gmm, n_bins=40))
        kl_gmm_means.append(np.mean(kls))
        kl_gmm_stds.append(np.std(kls))
        print(f"  {name}: {np.mean(kls):.4f} +/- {np.std(kls):.4f}")

    bars = ax.bar(x, kl_gmm_means, width, color=colors, edgecolor='black',
                  linewidth=0.5, alpha=0.85)
    for i, (m, s) in enumerate(zip(kl_gmm_means, kl_gmm_stds)):
        ax.text(i, m * 1.15 + s, f'{m:.3f}', ha='center', va='bottom', fontsize=8)
    ax.axhline(0.01, color='gray', ls='--', lw=1.2, label='Target KL=0.01')
    ax.set_xticks(x)
    ax.set_xticklabels(sampler_names, fontsize=FONTSIZE_TICK, rotation=15)
    ax.set_ylabel('KL Divergence', fontsize=FONTSIZE_LABEL)
    ax.set_yscale('log')
    ax.set_ylim(0.001, 50)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_title('5-Mode GMM', fontsize=13)
    ax.text(0.03, 0.95, '(c)', transform=ax.transAxes, fontsize=FONTSIZE_LABEL,
            fontweight='bold', va='top')

    fig.suptitle('Quantitative Comparison Across Benchmarks',
                 fontsize=FONTSIZE_TITLE, fontweight='bold', y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(os.path.dirname(__file__), 'figures', 'fig4_comparison.png')
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    make_figure()
