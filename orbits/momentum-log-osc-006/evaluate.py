"""Evaluate MLOSC variants on Stage 1 + Stage 2 benchmarks."""

import sys
import json
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
MomentumLogOsc = _mod.MomentumLogOsc
MomentumLogOscVerlet = _mod.MomentumLogOscVerlet
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


def scan_alpha():
    """Scan alpha parameter for MLOSC-A on Stage 1."""
    print("\n" + "#"*60)
    print("# ALPHA SCAN: MomentumLogOsc on Stage 1")
    print("#"*60)

    alphas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    results = {}

    for alpha in alphas:
        for pot_cls, pot_name, dt, Q in [
            (HarmonicOscillator1D, "harmonic_1d", 0.005, 0.8),
            (DoubleWell2D, "double_well_2d", 0.035, 1.0),
        ]:
            pot = pot_cls()
            dynamics = MomentumLogOsc(dim=pot.dim, kT=1.0, Q=Q, alpha=alpha)
            result = evaluate_single(
                dynamics, pot, MomentumLogOscVerlet, dt=dt,
                n_force_evals=1_000_000,
                label=f"MLOSC-A alpha={alpha}, Q={Q}",
            )
            results[(alpha, pot_name)] = result

    print("\n" + "="*90)
    print("ALPHA SCAN SUMMARY (MLOSC-A)")
    print("="*90)
    print(f"{'alpha':>6} | {'Potential':<20} | {'KL':>10} | {'ESS/fe':>10} | {'Ergo':>8} | {'TTT':>10}")
    print("-" * 90)

    for alpha in alphas:
        for pot_name in ["harmonic_1d", "double_well_2d"]:
            r = results[(alpha, pot_name)]
            kl = r['kl_divergence']
            ess = r['ess_metrics']['ess_per_force_eval'] if r['ess_metrics'] else 0
            ergo = r['ergodicity']['score'] if r['ergodicity'] else None
            ttt = r['time_to_threshold_force_evals']
            kl_str = f"{kl:.4f}" if kl is not None and kl != float('inf') else "inf"
            ess_str = f"{ess:.6f}"
            ergo_str = f"{ergo:.4f}" if ergo is not None else "N/A"
            ttt_str = str(ttt) if ttt is not None else "never"
            print(f"{alpha:>6.1f} | {pot_name:<20} | {kl_str:>10} | {ess_str:>10} | {ergo_str:>8} | {ttt_str:>10}")

    return results


def scan_rippled():
    """Scan epsilon and omega_xi for MLOSC-B on Stage 1."""
    print("\n" + "#"*60)
    print("# RIPPLED SCAN: RippledLogOsc on Stage 1")
    print("#"*60)

    configs = [
        (0.1, 2.0), (0.2, 2.0), (0.3, 2.0), (0.5, 2.0),
        (0.3, 1.0), (0.3, 3.0), (0.3, 5.0),
    ]
    results = {}

    for eps, w in configs:
        for pot_cls, pot_name, dt, Q in [
            (HarmonicOscillator1D, "harmonic_1d", 0.005, 0.8),
            (DoubleWell2D, "double_well_2d", 0.035, 1.0),
        ]:
            pot = pot_cls()
            dynamics = RippledLogOsc(dim=pot.dim, kT=1.0, Q=Q, epsilon=eps, omega_xi=w)
            result = evaluate_single(
                dynamics, pot, RippledLogOscVerlet, dt=dt,
                n_force_evals=1_000_000,
                label=f"MLOSC-B eps={eps},w={w}, Q={Q}",
            )
            results[(eps, w, pot_name)] = result

    print("\n" + "="*100)
    print("RIPPLED SCAN SUMMARY (MLOSC-B)")
    print("="*100)
    print(f"{'eps':>5} | {'omega':>5} | {'Potential':<20} | {'KL':>10} | {'ESS/fe':>10} | {'Ergo':>8} | {'TTT':>10}")
    print("-" * 100)

    for eps, w in configs:
        for pot_name in ["harmonic_1d", "double_well_2d"]:
            r = results[(eps, w, pot_name)]
            kl = r['kl_divergence']
            ess = r['ess_metrics']['ess_per_force_eval'] if r['ess_metrics'] else 0
            ergo = r['ergodicity']['score'] if r['ergodicity'] else None
            ttt = r['time_to_threshold_force_evals']
            kl_str = f"{kl:.4f}" if kl is not None and kl != float('inf') else "inf"
            ess_str = f"{ess:.6f}"
            ergo_str = f"{ergo:.4f}" if ergo is not None else "N/A"
            ttt_str = str(ttt) if ttt is not None else "never"
            print(f"{eps:>5.1f} | {w:>5.1f} | {pot_name:<20} | {kl_str:>10} | {ess_str:>10} | {ergo_str:>8} | {ttt_str:>10}")

    return results


def stage2_eval(best_configs):
    """Evaluate best configs on Stage 2."""
    print("\n" + "#"*60)
    print("# STAGE 2 EVALUATION")
    print("#"*60)

    results = {}
    for name, dynamics, integrator_cls in best_configs:
        for pot_cls, dt in [
            (GaussianMixture2D, 0.02),
            (Rosenbrock2D, 0.01),
        ]:
            pot = pot_cls()
            result = evaluate_single(
                dynamics, pot, integrator_cls, dt=dt,
                n_force_evals=1_000_000,
                label=f"Stage2: {name}",
            )
            results[(name, pot.name)] = result

    return results


if __name__ == "__main__":
    # Quick test first
    print("=" * 60)
    print("QUICK TEST: MLOSC-A alpha=0.5 on 1D HO")
    print("=" * 60)
    pot = HarmonicOscillator1D()
    dyn = MomentumLogOsc(dim=1, kT=1.0, Q=0.8, alpha=0.5)
    r = evaluate_single(dyn, pot, MomentumLogOscVerlet, dt=0.005,
                        n_force_evals=200_000, label="Quick test A")

    if r.get('nan_detected') or r['kl_divergence'] == float('inf'):
        print("\n*** QUICK TEST A FAILED ***")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("QUICK TEST: MLOSC-B eps=0.3,w=2 on 1D HO")
    print("=" * 60)
    dyn_b = RippledLogOsc(dim=1, kT=1.0, Q=0.8, epsilon=0.3, omega_xi=2.0)
    r_b = evaluate_single(dyn_b, pot, RippledLogOscVerlet, dt=0.005,
                          n_force_evals=200_000, label="Quick test B")

    if r_b.get('nan_detected') or r_b['kl_divergence'] == float('inf'):
        print("\n*** QUICK TEST B FAILED ***")
        sys.exit(1)

    print("\nQuick tests passed. Running full scans...")

    # Full alpha scan
    alpha_results = scan_alpha()

    # Full rippled scan
    rippled_results = scan_rippled()

    # Find best configs for Stage 2
    # Best MLOSC-A: pick alpha with best DW KL
    best_alpha_dw = min(
        [a for a in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]],
        key=lambda a: alpha_results.get((a, "double_well_2d"), {}).get("kl_divergence", float('inf'))
    )
    best_alpha_ho = min(
        [a for a in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]],
        key=lambda a: -(alpha_results.get((a, "harmonic_1d"), {}).get("ergodicity", {}) or {}).get("score", 0)
    )
    print(f"\nBest MLOSC-A alpha for DW: {best_alpha_dw}")
    print(f"Best MLOSC-A alpha for HO ergodicity: {best_alpha_ho}")

    # Best MLOSC-B: pick eps/w with best DW KL
    configs_b = [
        (0.1, 2.0), (0.2, 2.0), (0.3, 2.0), (0.5, 2.0),
        (0.3, 1.0), (0.3, 3.0), (0.3, 5.0),
    ]
    best_b_dw = min(
        configs_b,
        key=lambda c: rippled_results.get((c[0], c[1], "double_well_2d"), {}).get("kl_divergence", float('inf'))
    )
    print(f"Best MLOSC-B (eps,w) for DW: {best_b_dw}")

    # Stage 2 with best configs
    best_configs_s2 = [
        ("MLOSC-A", MomentumLogOsc(dim=2, kT=1.0, Q=1.0, alpha=best_alpha_dw), MomentumLogOscVerlet),
        ("MLOSC-B", RippledLogOsc(dim=2, kT=1.0, Q=1.0, epsilon=best_b_dw[0], omega_xi=best_b_dw[1]), RippledLogOscVerlet),
    ]
    stage2_results = stage2_eval(best_configs_s2)

    # Save all results
    all_results = {
        "alpha_scan": {
            str(k): {
                "kl": v["kl_divergence"],
                "ess": v["ess_metrics"]["ess_per_force_eval"] if v["ess_metrics"] else None,
                "ergodicity": v["ergodicity"]["score"] if v["ergodicity"] else None,
                "ttt": v["time_to_threshold_force_evals"],
            }
            for k, v in alpha_results.items()
        },
        "rippled_scan": {
            str(k): {
                "kl": v["kl_divergence"],
                "ess": v["ess_metrics"]["ess_per_force_eval"] if v["ess_metrics"] else None,
                "ergodicity": v["ergodicity"]["score"] if v["ergodicity"] else None,
                "ttt": v["time_to_threshold_force_evals"],
            }
            for k, v in rippled_results.items()
        },
        "stage2": {
            str(k): {
                "kl": v["kl_divergence"],
                "ess": v["ess_metrics"]["ess_per_force_eval"] if v["ess_metrics"] else None,
                "ttt": v["time_to_threshold_force_evals"],
            }
            for k, v in stage2_results.items()
        },
        "best_alpha_dw": best_alpha_dw,
        "best_alpha_ho": best_alpha_ho,
        "best_rippled_dw": best_b_dw,
    }

    output_path = f"{WORKTREE}/orbits/momentum-log-osc-006/results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
