"""Evaluate MLOSC-B (Rippled Log-Osc) on Stage 1 + Stage 2 benchmarks.

Best configuration: eps=0.3, omega_xi=5.0
"""

import sys
import importlib.util
import numpy as np

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
RippledLogOsc = _mod.RippledLogOsc
RippledLogOscVerlet = _mod.RippledLogOscVerlet


def evaluate_single(dynamics, potential, integrator_cls, dt=0.01,
                     n_force_evals=1_000_000, label=""):
    """Run evaluation and print results."""
    print(f"\n{'='*60}")
    print(f"{label}: {dynamics.name} on {potential.name}")
    print(f"  dt={dt}, n_force_evals={n_force_evals}")
    print(f"{'='*60}")

    result = run_sampler(
        dynamics, potential, dt=dt, n_force_evals=n_force_evals,
        kT=1.0, integrator_cls=integrator_cls,
    )

    kl = result['kl_divergence']
    print(f"  KL divergence: {kl}")
    if result['ess_metrics']:
        print(f"  ESS/force_eval: {result['ess_metrics']['ess_per_force_eval']:.6f}")
        print(f"  Autocorrelation time: {result['ess_metrics']['tau']:.1f}")
    if result['ergodicity']:
        erg = result['ergodicity']
        print(f"  Ergodicity score: {erg['score']:.4f} "
              f"({'ergodic' if erg['ergodic'] else 'NOT ergodic'})")
        print(f"    KS component: {erg['ks_component']:.4f}")
        print(f"    Var component: {erg['var_component']:.4f}")
        print(f"    Coverage: {erg['coverage']:.4f}")
    print(f"  Wall time: {result['wall_seconds']:.2f}s")
    print(f"  Time to KL<0.01: {result['time_to_threshold_force_evals']}")
    if result.get('nan_detected'):
        print(f"  *** NaN DETECTED ***")
    print(f"  N samples: {result['n_samples']}")

    return result


if __name__ == "__main__":
    print("=" * 60)
    print("MLOSC-B (Rippled Log-Osc) Final Evaluation")
    print("Best config: eps=0.3, omega_xi=5.0")
    print("=" * 60)

    # Stage 1: 1D Harmonic Oscillator
    pot_ho = HarmonicOscillator1D()
    dyn_ho = RippledLogOsc(dim=1, kT=1.0, Q=0.8, epsilon=0.3, omega_xi=5.0)
    r_ho = evaluate_single(dyn_ho, pot_ho, RippledLogOscVerlet,
                           dt=0.005, n_force_evals=1_000_000,
                           label="Stage 1")

    # Stage 1: 2D Double-Well
    pot_dw = DoubleWell2D()
    dyn_dw = RippledLogOsc(dim=2, kT=1.0, Q=1.0, epsilon=0.3, omega_xi=5.0)
    r_dw = evaluate_single(dyn_dw, pot_dw, RippledLogOscVerlet,
                           dt=0.04, n_force_evals=1_000_000,
                           label="Stage 1")

    # Stage 2: Gaussian Mixture
    pot_gmm = GaussianMixture2D()
    dyn_gmm = RippledLogOsc(dim=2, kT=1.0, Q=1.0, epsilon=0.3, omega_xi=5.0)
    r_gmm = evaluate_single(dyn_gmm, pot_gmm, RippledLogOscVerlet,
                            dt=0.03, n_force_evals=1_000_000,
                            label="Stage 2")

    # Stage 2: Rosenbrock
    pot_rb = Rosenbrock2D()
    dyn_rb = RippledLogOsc(dim=2, kT=1.0, Q=1.0, epsilon=0.3, omega_xi=5.0)
    r_rb = evaluate_single(dyn_rb, pot_rb, RippledLogOscVerlet,
                           dt=0.02, n_force_evals=1_000_000,
                           label="Stage 2")

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY: MLOSC-B (eps=0.3, w=5.0)")
    print("=" * 60)
    print(f"  1D HO:  KL={r_ho['kl_divergence']:.4f}, "
          f"Erg={r_ho['ergodicity']['score']:.4f}")
    print(f"  2D DW:  KL={r_dw['kl_divergence']:.4f}, "
          f"TTT={r_dw['time_to_threshold_force_evals']}")
    print(f"  2D GMM: KL={r_gmm['kl_divergence']:.4f}")
    print(f"  2D RB:  KL={r_rb['kl_divergence']:.4f}")
