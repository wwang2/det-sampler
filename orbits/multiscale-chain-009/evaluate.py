"""Evaluate Multi-Scale LOCR architectures on all benchmarks."""

import sys
import os
import json
import importlib.util
import numpy as np

# Add the worktree root to path
WORKTREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WORKTREE)

from research.eval.evaluator import run_sampler
from research.eval.potentials import (
    HarmonicOscillator1D, DoubleWell2D, GaussianMixture2D, Rosenbrock2D
)

# Import solution module (hyphenated directory name)
_spec = importlib.util.spec_from_file_location(
    "solution",
    os.path.join(WORKTREE, "orbits", "multiscale-chain-009", "solution.py"),
)
_sol = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sol)

MultiScaleLogOsc = _sol.MultiScaleLogOsc
MultiScaleLogOscVerlet = _sol.MultiScaleLogOscVerlet
MultiScaleNHCTail = _sol.MultiScaleNHCTail
MultiScaleNHCTailVerlet = _sol.MultiScaleNHCTailVerlet
HierarchicalLOCR = _sol.HierarchicalLOCR
HierarchicalLOCRVerlet = _sol.HierarchicalLOCRVerlet
HybridMSLOCR = _sol.HybridMSLOCR
HybridMSLOCRVerlet = _sol.HybridMSLOCRVerlet
MultiScale4LogOsc = _sol.MultiScale4LogOsc

SEED = 42
N_FORCE_EVALS = 1_000_000
KT = 1.0

POTENTIALS = {
    "ho": (HarmonicOscillator1D, 1),
    "dw": (DoubleWell2D, 2),
    "gmm": (GaussianMixture2D, 2),
    "rb": (Rosenbrock2D, 2),
}


def eval_sampler(name, dynamics, integrator_cls, potential, dt, seed=SEED):
    """Run one evaluation and return results dict."""
    rng = np.random.default_rng(seed)
    q0 = rng.normal(0, 0.5, size=potential.dim)
    result = run_sampler(
        dynamics, potential, dt=dt, n_force_evals=N_FORCE_EVALS,
        kT=KT, q0=q0, rng=rng, integrator_cls=integrator_cls,
    )
    kl = result.get("kl_divergence", float("inf"))
    if kl is None:
        kl = float("inf")
    erg = result.get("ergodicity", {})
    erg_score = erg.get("score", None) if erg else None
    ess = result.get("ess_metrics", {})
    ess_fe = ess.get("ess_per_force_eval", None) if ess else None
    ttt = result.get("time_to_threshold_force_evals", None)

    print(f"  {name} on {potential.name}: KL={kl:.4f}" +
          (f" ergo={erg_score:.3f}" if erg_score is not None else "") +
          (f" ESS/fe={ess_fe:.5f}" if ess_fe is not None else "") +
          (f" TTT={ttt}" if ttt is not None else ""))
    return {
        "sampler": name, "potential": potential.name, "dt": dt,
        "kl": kl, "ergodicity": erg_score, "ess_per_fe": ess_fe,
        "ttt": ttt, "seed": seed,
        "wall_seconds": result.get("wall_seconds", None),
        "n_samples": result.get("n_samples", 0),
    }


def run_all_potentials(name, make_dyn, integrator_cls, dt_ho=0.005, dt_2d=0.03):
    """Run sampler on all 4 potentials."""
    results = []
    for key, (PotCls, dim) in POTENTIALS.items():
        pot = PotCls()
        dt = dt_ho if key == "ho" else dt_2d
        dyn = make_dyn(dim)
        results.append(eval_sampler(name, dyn, integrator_cls, pot, dt))
    return results


def main():
    all_results = []

    # 1. Baseline: Multi-Scale Log-Osc (parent, no chain)
    print("\n=== Parent baseline: MultiScaleLogOsc Qs=[0.1, 0.7, 10.0] ===")
    all_results.extend(run_all_potentials(
        "ms_parent",
        lambda dim: MultiScaleLogOsc(dim=dim, kT=KT, Qs=[0.1, 0.7, 10.0]),
        MultiScaleLogOscVerlet,
        dt_ho=0.005, dt_2d=0.03,
    ))

    # 2. Architecture B: Multi-Scale NHC-Tail (chain on Q >= 0.7 scales)
    print("\n=== Arch B: MultiScaleNHCTail Qs=[0.1, 0.7, 10.0], M=2 ===")
    all_results.extend(run_all_potentials(
        "nhctail_B",
        lambda dim: MultiScaleNHCTail(dim=dim, kT=KT, Qs=[0.1, 0.7, 10.0],
                                        chain_length=2, chain_Q_multiplier=1.0),
        MultiScaleNHCTailVerlet,
        dt_ho=0.005, dt_2d=0.03,
    ))

    # 3. Architecture C: Hierarchical LOCR Qs=[0.7, 1.0, 10.0]
    print("\n=== Arch C: HierarchicalLOCR Qs=[0.7, 1.0, 10.0] ===")
    all_results.extend(run_all_potentials(
        "hier_C",
        lambda dim: HierarchicalLOCR(dim=dim, kT=KT, Qs=[0.7, 1.0, 10.0]),
        HierarchicalLOCRVerlet,
        dt_ho=0.005, dt_2d=0.03,
    ))

    # 4. Hybrid: Chain on medium Q only
    print("\n=== Arch D: HybridMSLOCR Q_f=0.1, Q_m=0.7+chain, Q_s=10 ===")
    all_results.extend(run_all_potentials(
        "hybrid_D",
        lambda dim: HybridMSLOCR(dim=dim, kT=KT, Q_fast=0.1, Q_med=0.7,
                                   Q_med_chain=0.7, Q_slow=10.0),
        HybridMSLOCRVerlet,
        dt_ho=0.005, dt_2d=0.03,
    ))

    # 5. 4-thermostat wider range (no chain)
    print("\n=== Arch E: MultiScale4 Qs=[0.05, 0.3, 1.5, 10.0] ===")
    all_results.extend(run_all_potentials(
        "ms4_E",
        lambda dim: MultiScale4LogOsc(dim=dim, kT=KT, Qs=[0.05, 0.3, 1.5, 10.0]),
        MultiScaleLogOscVerlet,
        dt_ho=0.005, dt_2d=0.03,
    ))

    # Save results
    out_dir = os.path.join(WORKTREE, "orbits", "multiscale-chain-009")
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n\n=== SUMMARY ===")
    print(f"{'Sampler':<15} {'Potential':<20} {'KL':>8} {'Ergo':>8} {'ESS/fe':>10}")
    print("-" * 65)
    for r in all_results:
        erg_str = f"{r['ergodicity']:.3f}" if r['ergodicity'] is not None else "N/A"
        ess_str = f"{r['ess_per_fe']:.5f}" if r['ess_per_fe'] is not None else "N/A"
        kl_str = f"{r['kl']:.4f}" if r['kl'] != float('inf') else "inf"
        print(f"{r['sampler']:<15} {r['potential']:<20} {kl_str:>8} {erg_str:>8} {ess_str:>10}")


if __name__ == "__main__":
    main()
