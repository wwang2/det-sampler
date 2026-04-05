"""Final evaluation of best Multi-Scale NHCTail configuration.

Best config: Qs=[0.05, 0.7, 10.0], chain_length=2, chain_Q_multiplier=1.0
Per-potential dt: HO=0.005, DW=0.055, GMM=0.03, RB=0.03
"""

import sys
import os
import json
import importlib.util
import numpy as np

WORKTREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WORKTREE)

from research.eval.evaluator import run_sampler
from research.eval.potentials import (
    HarmonicOscillator1D, DoubleWell2D, GaussianMixture2D, Rosenbrock2D
)

_spec = importlib.util.spec_from_file_location(
    "solution",
    os.path.join(WORKTREE, "orbits", "multiscale-chain-009", "solution.py"),
)
_sol = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sol)

N_FORCE_EVALS = 1_000_000
KT = 1.0
QS = [0.05, 0.7, 10.0]
CHAIN_LENGTH = 2
CHAIN_Q_MULT = 1.0

BEST_DT = {
    "harmonic_1d": 0.005,
    "double_well_2d": 0.055,
    "gaussian_mixture_2d": 0.03,
    "rosenbrock_2d": 0.03,
}


def full_eval(pot, seed=42):
    """Full evaluation on one potential."""
    dim = pot.dim
    dt = BEST_DT[pot.name]
    dyn = _sol.MultiScaleNHCTail(
        dim=dim, kT=KT, Qs=QS,
        chain_length=CHAIN_LENGTH, chain_Q_multiplier=CHAIN_Q_MULT,
    )
    rng = np.random.default_rng(seed)
    q0 = rng.normal(0, 0.5, size=dim)
    result = run_sampler(
        dyn, pot, dt=dt, n_force_evals=N_FORCE_EVALS,
        kT=KT, q0=q0, rng=rng, integrator_cls=_sol.MultiScaleNHCTailVerlet,
    )
    return result


def main():
    potentials = [
        HarmonicOscillator1D(),
        DoubleWell2D(),
        GaussianMixture2D(),
        Rosenbrock2D(),
    ]

    all_results = {}

    # Single seed evaluation
    print("=== Final Evaluation: NHCTail Qs=[0.05, 0.7, 10.0], M=2 ===\n")
    for pot in potentials:
        r = full_eval(pot, seed=42)
        kl = r.get("kl_divergence", float("inf"))
        if kl is None:
            kl = float("inf")
        erg = (r.get("ergodicity") or {}).get("score")
        ess = (r.get("ess_metrics") or {}).get("ess_per_force_eval")
        ttt = r.get("time_to_threshold_force_evals")

        all_results[pot.name] = {
            "kl": kl,
            "ergodicity": erg,
            "ess_per_fe": ess,
            "ttt": ttt,
            "dt": BEST_DT[pot.name],
            "kl_trace": r.get("kl_trace", []),
            "wall_seconds": r.get("wall_seconds"),
            "n_samples": r.get("n_samples", 0),
        }

        print(f"{pot.name}:")
        print(f"  dt={BEST_DT[pot.name]}, KL={kl:.4f}", end="")
        if erg is not None:
            print(f", ergo={erg:.3f}", end="")
        if ess is not None:
            print(f", ESS/fe={ess:.5f}", end="")
        if ttt is not None:
            print(f", TTT={ttt}", end="")
        print()

    # GMM robustness (5 seeds)
    print("\n=== GMM Robustness (5 seeds) ===")
    gmm_kls = []
    for seed in [42, 123, 7, 999, 314]:
        r = full_eval(GaussianMixture2D(), seed=seed)
        kl = r.get("kl_divergence", float("inf")) or float("inf")
        gmm_kls.append(kl)
        print(f"  seed={seed}: KL={kl:.4f}")
    print(f"  Mean={np.mean(gmm_kls):.4f}, Std={np.std(gmm_kls):.4f}")
    all_results["gmm_robustness"] = {
        "seeds": [42, 123, 7, 999, 314],
        "kls": gmm_kls,
        "mean": float(np.mean(gmm_kls)),
        "std": float(np.std(gmm_kls)),
    }

    # DW robustness (5 seeds)
    print("\n=== DW Robustness (5 seeds) ===")
    dw_kls = []
    for seed in [42, 123, 7, 999, 314]:
        r = full_eval(DoubleWell2D(), seed=seed)
        kl = r.get("kl_divergence", float("inf")) or float("inf")
        dw_kls.append(kl)
        print(f"  seed={seed}: KL={kl:.4f}")
    print(f"  Mean={np.mean(dw_kls):.4f}, Std={np.std(dw_kls):.4f}")
    all_results["dw_robustness"] = {
        "seeds": [42, 123, 7, 999, 314],
        "kls": dw_kls,
        "mean": float(np.mean(dw_kls)),
        "std": float(np.std(dw_kls)),
    }

    # Save
    out_dir = os.path.join(WORKTREE, "orbits", "multiscale-chain-009")
    with open(os.path.join(out_dir, "final_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {out_dir}/final_results.json")

    # Print comparison table
    print("\n=== Comparison with Targets ===")
    print(f"{'Metric':<20} {'NHCTail':>10} {'Parent':>10} {'Target':>10} {'Status':>10}")
    print("-" * 65)

    ho = all_results.get("harmonic_1d", {})
    dw = all_results.get("double_well_2d", {})
    gmm = all_results.get("gaussian_mixture_2d", {})
    gmm_rob = all_results.get("gmm_robustness", {})
    rb = all_results.get("rosenbrock_2d", {})

    rows = [
        ("HO Ergodicity", ho.get("ergodicity", 0), 0.927, ">0.93", ho.get("ergodicity", 0) > 0.93 if ho.get("ergodicity") else False),
        ("HO KL", ho.get("kl", 999), 0.004, "<0.005", ho.get("kl", 999) < 0.005),
        ("DW KL", dw.get("kl", 999), 0.010, "<0.007", dw.get("kl", 999) < 0.007),
        ("DW KL (mean)", all_results.get("dw_robustness", {}).get("mean", 999), 0.010, "<0.007", all_results.get("dw_robustness", {}).get("mean", 999) < 0.007),
        ("GMM KL", gmm.get("kl", 999), 0.148, "<0.15", gmm.get("kl", 999) < 0.15),
        ("GMM KL (mean)", gmm_rob.get("mean", 999), 0.148, "<0.15", gmm_rob.get("mean", 999) < 0.15),
        ("RB KL", rb.get("kl", 999), 0.006, "<0.01", rb.get("kl", 999) < 0.01),
    ]
    for name, val, parent, target, passed in rows:
        status = "PASS" if passed else "MISS"
        print(f"{name:<20} {val:>10.4f} {parent:>10.3f} {target:>10} {status:>10}")


if __name__ == "__main__":
    main()
