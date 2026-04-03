"""Evaluate the Log-Osc thermostat on Stage 1 benchmarks."""

import sys
import json
import importlib.util
import numpy as np

sys.path.insert(0, '/Users/wujiewang/code/det-sampler/.worktrees/log-osc-001')

from research.eval.evaluator import run_sampler
from research.eval.potentials import DoubleWell2D, HarmonicOscillator1D

# Import from hyphenated directory using importlib
_spec = importlib.util.spec_from_file_location(
    "solution",
    "/Users/wujiewang/code/det-sampler/.worktrees/log-osc-001/orbits/log-osc-001/solution.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LogOscThermostat = _mod.LogOscThermostat
LogOscVelocityVerlet = _mod.LogOscVelocityVerlet
LogOscChain = _mod.LogOscChain
LogOscChainVerlet = _mod.LogOscChainVerlet


def evaluate_single(dynamics, potential, integrator_cls, dt=0.01, n_force_evals=1_000_000,
                     label=""):
    """Run evaluation and print results."""
    print(f"\n{'='*60}")
    print(f"{label}: {dynamics.name} on {potential.name}")
    print(f"  Q={getattr(dynamics, 'Q', '?')}, dt={dt}, n_force_evals={n_force_evals}")
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


def q_scan():
    """Scan over Q values for the single-variable Log-Osc thermostat."""
    print("\n" + "#"*60)
    print("# Q-SCAN: LogOscThermostat on Stage 1")
    print("#"*60)

    Q_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    results = {}

    for Q in Q_values:
        for pot_cls, pot_name, dt in [
            (DoubleWell2D, "double_well_2d", 0.01),
            (HarmonicOscillator1D, "harmonic_1d", 0.005),
        ]:
            pot = pot_cls()
            dynamics = LogOscThermostat(dim=pot.dim, kT=1.0, Q=Q)
            result = evaluate_single(
                dynamics, pot, LogOscVelocityVerlet, dt=dt,
                n_force_evals=1_000_000,
                label=f"Q={Q}",
            )
            results[(Q, pot_name)] = result

    # Summary table
    print("\n" + "="*80)
    print("Q-SCAN SUMMARY")
    print("="*80)
    print(f"{'Q':>6} | {'Potential':<20} | {'KL':>10} | {'ESS/fe':>10} | {'Ergo':>8} | {'TTT':>10}")
    print("-" * 80)

    for Q in Q_values:
        for pot_name in ["double_well_2d", "harmonic_1d"]:
            r = results[(Q, pot_name)]
            kl = r['kl_divergence']
            ess = r['ess_metrics']['ess_per_force_eval'] if r['ess_metrics'] else 0
            ergo = r['ergodicity']['score'] if r['ergodicity'] else None
            ttt = r['time_to_threshold_force_evals']
            kl_str = f"{kl:.4f}" if kl is not None and kl != float('inf') else "inf"
            ess_str = f"{ess:.6f}"
            ergo_str = f"{ergo:.4f}" if ergo is not None else "N/A"
            ttt_str = str(ttt) if ttt is not None else "never"
            print(f"{Q:>6.1f} | {pot_name:<20} | {kl_str:>10} | {ess_str:>10} | {ergo_str:>8} | {ttt_str:>10}")

    return results


def chain_eval():
    """Evaluate Log-Osc Chain thermostat."""
    print("\n" + "#"*60)
    print("# LOG-OSC CHAIN (M=3) on Stage 1")
    print("#"*60)

    results = {}
    for pot_cls, pot_name, dt in [
        (DoubleWell2D, "double_well_2d", 0.01),
        (HarmonicOscillator1D, "harmonic_1d", 0.005),
    ]:
        for Q in [0.5, 1.0, 2.0]:
            pot = pot_cls()
            dynamics = LogOscChain(dim=pot.dim, chain_length=3, kT=1.0, Q=Q)
            result = evaluate_single(
                dynamics, pot, LogOscChainVerlet, dt=dt,
                n_force_evals=1_000_000,
                label=f"LogOscChain(M=3,Q={Q})",
            )
            results[(Q, pot_name)] = result

    return results


if __name__ == "__main__":
    # First: quick test with single Q
    print("QUICK TEST: LogOsc Q=1.0 on 1D HO")
    pot = HarmonicOscillator1D()
    dyn = LogOscThermostat(dim=1, kT=1.0, Q=1.0)
    r = evaluate_single(dyn, pot, LogOscVelocityVerlet, dt=0.005, n_force_evals=200_000,
                        label="Quick test")

    if r.get('nan_detected') or r['kl_divergence'] == float('inf'):
        print("\n*** QUICK TEST FAILED - NaN or inf KL ***")
        print("Trying with RK4 integrator instead...")
        from research.eval.integrators import AdaptiveRK45
        r2 = evaluate_single(dyn, pot, AdaptiveRK45, dt=0.005, n_force_evals=200_000,
                             label="Quick test (RK4)")
        if r2.get('nan_detected') or r2['kl_divergence'] == float('inf'):
            print("*** RK4 ALSO FAILED ***")
            sys.exit(1)

    # Full Q scan
    q_results = q_scan()

    # Chain evaluation
    chain_results = chain_eval()
