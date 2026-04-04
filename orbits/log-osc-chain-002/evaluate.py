"""Evaluate the LOCR thermostat on Stage 1 benchmarks."""

import sys
import json
import importlib.util
import numpy as np

sys.path.insert(0, '/Users/wujiewang/code/det-sampler/.worktrees/log-osc-chain-002')

from research.eval.evaluator import run_sampler
from research.eval.potentials import DoubleWell2D, HarmonicOscillator1D

# Import from hyphenated directory
_spec = importlib.util.spec_from_file_location(
    "solution",
    "/Users/wujiewang/code/det-sampler/.worktrees/log-osc-chain-002/orbits/log-osc-chain-002/solution.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LogOscChainRotation = _mod.LogOscChainRotation
LOCRIntegrator = _mod.LOCRIntegrator


def evaluate_single(dynamics, potential, integrator_cls, dt=0.01, n_force_evals=1_000_000,
                     label=""):
    """Run evaluation and print results."""
    print(f"\n{'='*60}")
    print(f"{label}: {dynamics.name} on {potential.name}")
    print(f"  M={dynamics.M}, Q={dynamics.Q}, alpha={dynamics.alpha}, dt={dt}")
    print(f"{'='*60}")

    result = run_sampler(
        dynamics, potential, dt=dt, n_force_evals=n_force_evals,
        kT=1.0, integrator_cls=integrator_cls,
    )

    print(f"  KL divergence: {result['kl_divergence']}")
    if result['ess_metrics']:
        print(f"  ESS/force_eval: {result['ess_metrics']['ess_per_force_eval']:.6f}")
        print(f"  Autocorrelation time: {result['ess_metrics']['tau']:.1f}")
    if result['ergodicity']:
        print(f"  Ergodicity score: {result['ergodicity']['score']:.4f} "
              f"({'ergodic' if result['ergodicity']['ergodic'] else 'NOT ergodic'})")
        print(f"    KS component: {result['ergodicity']['ks_component']:.4f}")
        print(f"    Var component: {result['ergodicity']['var_component']:.4f}")
        print(f"    Coverage: {result['ergodicity']['coverage']:.4f}")
    print(f"  Wall time: {result['wall_seconds']:.2f}s")
    print(f"  Time to KL<0.01: {result['time_to_threshold_force_evals']}")
    if result.get('nan_detected'):
        print(f"  *** NaN DETECTED ***")
    print(f"  N samples: {result['n_samples']}")

    return result


def quick_test():
    """Quick sanity check on 1D HO."""
    print("QUICK TEST: LOCR M=2, Q=0.8, alpha=0.1 on 1D HO")
    pot = HarmonicOscillator1D()
    dyn = LogOscChainRotation(dim=1, chain_length=2, kT=1.0, Q=0.8, alpha=0.1)
    r = evaluate_single(dyn, pot, LOCRIntegrator, dt=0.005, n_force_evals=200_000,
                        label="Quick test")
    return r


def scan_params():
    """Scan over key parameters: M, Q, alpha, dt."""
    print("\n" + "#" * 60)
    print("# PARAMETER SCAN: LOCR on Stage 1")
    print("#" * 60)

    results = []

    # Scan configurations
    configs = [
        # (M, Q, alpha, dt_ho, dt_dw)
        # --- Baseline: M=2, no rotation ---
        (2, 0.5, 0.0, 0.005, 0.02),
        (2, 0.8, 0.0, 0.005, 0.02),
        (2, 1.0, 0.0, 0.005, 0.02),
        # --- M=2, with rotation ---
        (2, 0.5, 0.1, 0.005, 0.02),
        (2, 0.5, 0.3, 0.005, 0.02),
        (2, 0.5, 0.5, 0.005, 0.02),
        (2, 0.8, 0.1, 0.005, 0.02),
        (2, 0.8, 0.3, 0.005, 0.02),
        (2, 0.8, 0.5, 0.005, 0.02),
        (2, 1.0, 0.1, 0.005, 0.02),
        (2, 1.0, 0.3, 0.005, 0.02),
        # --- M=3 ---
        (3, 0.5, 0.0, 0.005, 0.02),
        (3, 0.5, 0.1, 0.005, 0.02),
        (3, 0.5, 0.3, 0.005, 0.02),
        (3, 0.8, 0.0, 0.005, 0.02),
        (3, 0.8, 0.1, 0.005, 0.02),
        (3, 0.8, 0.3, 0.005, 0.02),
        (3, 1.0, 0.0, 0.005, 0.02),
        (3, 1.0, 0.1, 0.005, 0.02),
    ]

    for M, Q, alpha, dt_ho, dt_dw in configs:
        for pot_cls, pot_name, dt in [
            (HarmonicOscillator1D, "harmonic_1d", dt_ho),
            (DoubleWell2D, "double_well_2d", dt_dw),
        ]:
            pot = pot_cls()
            dynamics = LogOscChainRotation(
                dim=pot.dim, chain_length=M, kT=1.0, Q=Q, alpha=alpha
            )
            result = evaluate_single(
                dynamics, pot, LOCRIntegrator, dt=dt,
                n_force_evals=1_000_000,
                label=f"M={M},Q={Q},a={alpha}",
            )
            results.append({
                'M': M, 'Q': Q, 'alpha': alpha, 'dt': dt,
                'potential': pot_name,
                'kl': result['kl_divergence'],
                'ess_fe': result['ess_metrics']['ess_per_force_eval'] if result['ess_metrics'] else 0,
                'ergo': result['ergodicity']['score'] if result['ergodicity'] else None,
                'ttt': result['time_to_threshold_force_evals'],
                'nan': result.get('nan_detected', False),
            })

    # Summary
    print("\n" + "=" * 100)
    print("PARAMETER SCAN SUMMARY")
    print("=" * 100)
    print(f"{'M':>3} | {'Q':>5} | {'alpha':>6} | {'dt':>6} | {'Potential':<20} | {'KL':>10} | {'ESS/fe':>10} | {'Ergo':>8} | {'TTT':>10}")
    print("-" * 100)

    for r in results:
        kl_str = f"{r['kl']:.4f}" if r['kl'] is not None and r['kl'] != float('inf') else "inf"
        ess_str = f"{r['ess_fe']:.6f}"
        ergo_str = f"{r['ergo']:.4f}" if r['ergo'] is not None else "N/A"
        ttt_str = str(r['ttt']) if r['ttt'] is not None else "never"
        nan_str = " NaN!" if r['nan'] else ""
        print(f"{r['M']:>3} | {r['Q']:>5.1f} | {r['alpha']:>6.2f} | {r['dt']:>6.3f} | {r['potential']:<20} | {kl_str:>10} | {ess_str:>10} | {ergo_str:>8} | {ttt_str:>10}{nan_str}")

    # Save results
    with open('/Users/wujiewang/code/det-sampler/.worktrees/log-osc-chain-002/orbits/log-osc-chain-002/scan_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def dt_scan_best():
    """Fine-tune dt for the best configuration found."""
    print("\n" + "#" * 60)
    print("# DT SCAN for best config")
    print("#" * 60)

    # Will be filled in after initial scan
    # Placeholder: scan dt for a reasonable config
    best_configs = [
        # (M, Q, alpha)
        (2, 0.8, 0.3),
        (2, 0.5, 0.3),
        (3, 0.8, 0.1),
    ]

    results = []
    for M, Q, alpha in best_configs:
        # DW dt scan
        for dt in [0.015, 0.020, 0.025, 0.030, 0.035]:
            pot = DoubleWell2D()
            dyn = LogOscChainRotation(dim=2, chain_length=M, kT=1.0, Q=Q, alpha=alpha)
            r = evaluate_single(dyn, pot, LOCRIntegrator, dt=dt,
                                n_force_evals=1_000_000, label=f"DT-scan M={M},Q={Q},a={alpha}")
            results.append({
                'M': M, 'Q': Q, 'alpha': alpha, 'dt': dt,
                'potential': 'double_well_2d',
                'kl': r['kl_divergence'],
                'ess_fe': r['ess_metrics']['ess_per_force_eval'] if r['ess_metrics'] else 0,
                'ttt': r['time_to_threshold_force_evals'],
                'nan': r.get('nan_detected', False),
            })

        # HO dt scan
        for dt in [0.003, 0.005, 0.007, 0.010]:
            pot = HarmonicOscillator1D()
            dyn = LogOscChainRotation(dim=1, chain_length=M, kT=1.0, Q=Q, alpha=alpha)
            r = evaluate_single(dyn, pot, LOCRIntegrator, dt=dt,
                                n_force_evals=1_000_000, label=f"DT-scan M={M},Q={Q},a={alpha}")
            results.append({
                'M': M, 'Q': Q, 'alpha': alpha, 'dt': dt,
                'potential': 'harmonic_1d',
                'kl': r['kl_divergence'],
                'ess_fe': r['ess_metrics']['ess_per_force_eval'] if r['ess_metrics'] else 0,
                'ergo': r['ergodicity']['score'] if r['ergodicity'] else None,
                'ttt': r['time_to_threshold_force_evals'],
                'nan': r.get('nan_detected', False),
            })

    # Summary
    print("\n" + "=" * 100)
    print("DT SCAN SUMMARY")
    print("=" * 100)
    for r in results:
        kl_str = f"{r['kl']:.4f}" if r['kl'] is not None and r['kl'] != float('inf') else "inf"
        ergo_str = f"{r.get('ergo', 0):.4f}" if r.get('ergo') is not None else "N/A"
        ttt_str = str(r['ttt']) if r['ttt'] is not None else "never"
        print(f"M={r['M']} Q={r['Q']:.1f} a={r['alpha']:.2f} dt={r['dt']:.3f} {r['potential']:<20} KL={kl_str} ESS={r['ess_fe']:.6f} Ergo={ergo_str} TTT={ttt_str}")

    return results


if __name__ == "__main__":
    # Step 1: Quick test
    r = quick_test()
    if r.get('nan_detected') or r['kl_divergence'] == float('inf'):
        print("\n*** QUICK TEST FAILED ***")
        sys.exit(1)

    # Step 2: Parameter scan
    scan_results = scan_params()

    # Step 3: dt fine-tuning
    dt_results = dt_scan_best()
