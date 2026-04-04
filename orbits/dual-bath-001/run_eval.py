"""Evaluate dual-bath thermostat on Stage 1 benchmarks and generate figures."""

import sys
import json
import importlib.util
import numpy as np

sys.path.insert(0, '/Users/wujiewang/code/det-sampler/.worktrees/dual-bath-001')

from research.eval.evaluator import run_sampler
from research.eval.potentials import DoubleWell2D, HarmonicOscillator1D

# Load solution module
_spec = importlib.util.spec_from_file_location(
    "solution",
    "/Users/wujiewang/code/det-sampler/.worktrees/dual-bath-001/orbits/dual-bath-001/solution.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
DualBathThermostat = _mod.DualBathThermostat
DualBathVelocityVerlet = _mod.DualBathVelocityVerlet

SEED = 42
# Best parameters found through scanning
Q_XI = 1.0
Q_ETA = 1.0
ALPHA = 0.5
FIGURES_DIR = "/Users/wujiewang/code/det-sampler/.worktrees/dual-bath-001/orbits/dual-bath-001/figures"


def evaluate():
    """Run full Stage 1 evaluation."""
    import os
    os.makedirs(FIGURES_DIR, exist_ok=True)

    results = {}

    # 1. Double Well 2D
    print(f"\n=== Double Well 2D (Q_xi={Q_XI}, Q_eta={Q_ETA}, alpha={ALPHA}) ===")
    pot_dw = DoubleWell2D()
    dyn_dw = DualBathThermostat(dim=2, kT=1.0, Q_xi=Q_XI, Q_eta=Q_ETA, alpha=ALPHA)
    r = run_sampler(
        dyn_dw, pot_dw, dt=0.01, n_force_evals=1_000_000, kT=1.0,
        rng=np.random.default_rng(SEED),
        integrator_cls=DualBathVelocityVerlet,
    )
    print(f"  KL divergence: {r['kl_divergence']:.4f}")
    if r['ess_metrics']:
        print(f"  ESS/force_eval: {r['ess_metrics']['ess_per_force_eval']:.6f}")
        print(f"  Autocorrelation time: {r['ess_metrics']['tau']:.1f}")
    print(f"  Time to KL<0.01: {r['time_to_threshold_force_evals']}")
    print(f"  Wall time: {r['wall_seconds']:.2f}s")
    results['double_well_2d'] = r

    # 2. Harmonic Oscillator 1D
    print(f"\n=== Harmonic Oscillator 1D ===")
    pot_ho = HarmonicOscillator1D()
    dyn_ho = DualBathThermostat(dim=1, kT=1.0, Q_xi=Q_XI, Q_eta=Q_ETA, alpha=ALPHA)
    r = run_sampler(
        dyn_ho, pot_ho, dt=0.005, n_force_evals=1_000_000, kT=1.0,
        rng=np.random.default_rng(SEED),
        integrator_cls=DualBathVelocityVerlet,
    )
    print(f"  KL divergence: {r['kl_divergence']:.4f}")
    if r['ess_metrics']:
        print(f"  ESS/force_eval: {r['ess_metrics']['ess_per_force_eval']:.6f}")
    if r['ergodicity']:
        erg = r['ergodicity']
        print(f"  Ergodicity score: {erg['score']:.4f} ({'ERGODIC' if erg['ergodic'] else 'NOT ergodic'})")
        print(f"    KS component: {erg['ks_component']:.4f}")
        print(f"    Var component: {erg['var_component']:.4f}")
        print(f"    Coverage: {erg['coverage']:.4f}")
    print(f"  Wall time: {r['wall_seconds']:.2f}s")
    results['harmonic_1d'] = r

    # Print comparison
    print("\n\n=== COMPARISON WITH BASELINES ===")
    print(f"{'Metric':<25} {'Dual-Bath':>12} {'NH':>12} {'NHC(M=3)':>12}")
    print("-" * 65)
    dw = results['double_well_2d']
    ho = results['harmonic_1d']
    print(f"{'DW KL':<25} {dw['kl_divergence']:>12.4f} {'0.037':>12} {'0.029':>12}")
    if dw['ess_metrics']:
        print(f"{'DW ESS/force':<25} {dw['ess_metrics']['ess_per_force_eval']:>12.6f} {'0.00310':>12} {'0.00261':>12}")
    print(f"{'HO KL':<25} {ho['kl_divergence']:>12.4f} {'0.077':>12} {'0.002':>12}")
    if ho['ergodicity']:
        print(f"{'HO Ergodicity':<25} {ho['ergodicity']['score']:>12.4f} {'0.54':>12} {'0.92':>12}")

    return results


def generate_figures(results):
    """Generate diagnostic figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Style settings from research/style.md
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })

    # Color for novel sampler (tab10 index 2 = green)
    color_novel = plt.cm.tab10(2)
    color_nh = '#1f77b4'
    color_nhc = '#ff7f0e'

    # --- Figure 1: KL convergence traces ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (pot_name, title) in zip(axes, [('double_well_2d', '2D Double Well'),
                                              ('harmonic_1d', '1D Harmonic Oscillator')]):
        r = results[pot_name]
        if r['kl_trace']:
            evals, kls = zip(*r['kl_trace'])
            ax.semilogy(evals, kls, '-', color=color_novel, linewidth=2, label='Dual-Bath (ours)')
        ax.axhline(y=0.01, color='gray', linestyle='--', alpha=0.7, label='KL=0.01 threshold')
        # Baseline reference points
        if pot_name == 'double_well_2d':
            ax.axhline(y=0.029, color=color_nhc, linestyle=':', alpha=0.7, label='NHC(M=3) baseline')
            ax.axhline(y=0.037, color=color_nh, linestyle=':', alpha=0.7, label='NH baseline')
        else:
            ax.axhline(y=0.002, color=color_nhc, linestyle=':', alpha=0.7, label='NHC(M=3) baseline')
            ax.axhline(y=0.077, color=color_nh, linestyle=':', alpha=0.7, label='NH baseline')
        ax.set_xlabel('Force evaluations')
        ax.set_ylabel('KL divergence')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_xscale('log')

    fig.tight_layout()
    fig.savefig(f'{FIGURES_DIR}/kl_convergence.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {FIGURES_DIR}/kl_convergence.png")

    # --- Figure 2: Phase space coverage (HO 1D) ---
    # Re-run a short trajectory to collect phase space data
    pot_ho = HarmonicOscillator1D()
    dyn = DualBathThermostat(dim=1, kT=1.0, Q_xi=Q_XI, Q_eta=Q_ETA, alpha=ALPHA)
    state = dyn.initial_state(np.array([0.5]), rng=np.random.default_rng(SEED))
    integrator = DualBathVelocityVerlet(dyn, pot_ho, dt=0.005, kT=1.0, mass=1.0)

    qs, ps, xis, etas = [], [], [], []
    for i in range(200000):
        state = integrator.step(state)
        if i > 20000 and i % 5 == 0:
            qs.append(state.q[0])
            ps.append(state.p[0])
            xis.append(state.xi[0])
            etas.append(state.xi[1])

    qs, ps = np.array(qs), np.array(ps)
    xis, etas = np.array(xis), np.array(etas)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Phase space (q, p)
    ax = axes[0]
    ax.scatter(qs[::10], ps[::10], s=1, alpha=0.3, color=color_novel)
    # Gaussian contours
    theta = np.linspace(0, 2*np.pi, 100)
    for n_sigma in [1, 2, 3]:
        ax.plot(n_sigma*np.cos(theta), n_sigma*np.sin(theta), 'k-', alpha=0.3, linewidth=1)
    ax.set_xlabel('q')
    ax.set_ylabel('p')
    ax.set_title('Phase Space (q, p)')
    ax.set_aspect('equal')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    # Thermostat space (xi, eta)
    ax = axes[1]
    ax.scatter(xis[::10], etas[::10], s=1, alpha=0.3, color=color_novel)
    ax.set_xlabel(r'$\xi$')
    ax.set_ylabel(r'$\eta$')
    ax.set_title(r'Thermostat Space ($\xi$, $\eta$)')
    ax.set_aspect('equal')

    # Position marginal histogram vs analytical
    ax = axes[2]
    ax.hist(qs, bins=80, density=True, alpha=0.7, color=color_novel, label='Dual-Bath')
    q_range = np.linspace(-4, 4, 200)
    p_true = np.exp(-0.5 * q_range**2) / np.sqrt(2*np.pi)
    ax.plot(q_range, p_true, 'k-', linewidth=2, label='Analytical N(0,1)')
    ax.set_xlabel('q')
    ax.set_ylabel('Density')
    ax.set_title('Position Marginal (1D HO)')
    ax.legend()

    fig.tight_layout()
    fig.savefig(f'{FIGURES_DIR}/phase_space_ho.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {FIGURES_DIR}/phase_space_ho.png")

    # --- Figure 3: Double-well density comparison ---
    pot_dw = DoubleWell2D()
    dyn = DualBathThermostat(dim=2, kT=1.0, Q_xi=Q_XI, Q_eta=Q_ETA, alpha=ALPHA)
    state = dyn.initial_state(np.array([0.5, 0.0]), rng=np.random.default_rng(SEED))
    integrator = DualBathVelocityVerlet(dyn, pot_dw, dt=0.01, kT=1.0, mass=1.0)

    qs_dw = []
    for i in range(500000):
        state = integrator.step(state)
        if i > 50000 and i % 5 == 0:
            qs_dw.append(state.q.copy())
    qs_dw = np.array(qs_dw)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Sampled density
    ax = axes[0]
    h = ax.hist2d(qs_dw[:, 0], qs_dw[:, 1], bins=60, density=True, cmap='viridis',
                  range=[[-3, 3], [-3, 3]])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Sampled Density (Dual-Bath)')
    ax.set_aspect('equal')
    plt.colorbar(h[3], ax=ax)

    # True density
    ax = axes[1]
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-((X**2 - 1)**2 + 0.5*Y**2))
    Z /= Z.sum() * (x[1]-x[0]) * (y[1]-y[0])
    im = ax.pcolormesh(X, Y, Z, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('True Boltzmann Density')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(f'{FIGURES_DIR}/density_double_well.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {FIGURES_DIR}/density_double_well.png")


if __name__ == "__main__":
    results = evaluate()
    generate_figures(results)
