"""Evaluate Log-Osc Multi-Thermostat variants on all benchmarks.

Usage:
    cd /Users/wujiewang/code/det-sampler/.worktrees/log-osc-multiT-005
    python -m orbits.log-osc-multiT-005.evaluate
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add worktree root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from research.eval.evaluator import run_sampler
from research.eval.potentials import (
    HarmonicOscillator1D, DoubleWell2D, GaussianMixture2D, Rosenbrock2D
)

# Import our samplers
from orbits import __path__ as orbits_path

# Direct import workaround for hyphenated module name
import importlib
sol = importlib.import_module("orbits.log-osc-multiT-005.solution")

DualLogOsc = sol.DualLogOsc
DualLogOscVerlet = sol.DualLogOscVerlet
DualLogOscCross = sol.DualLogOscCross
LogOscTempPulse = sol.LogOscTempPulse
LogOscTempPulseVerlet = sol.LogOscTempPulseVerlet
MultiScaleLogOsc = sol.MultiScaleLogOsc
MultiScaleLogOscVerlet = sol.MultiScaleLogOscVerlet

# Also import parent log-osc for comparison
parent_sol = importlib.import_module("orbits.log-osc-001.solution")
LogOscThermostat = parent_sol.LogOscThermostat
LogOscVelocityVerlet = parent_sol.LogOscVelocityVerlet


def evaluate_variant(name, dynamics, integrator_cls, potential, dt, n_force_evals=1_000_000):
    """Run evaluation and print results."""
    print(f"\n{'='*60}")
    print(f"  {name} on {potential.name} (dt={dt})")
    print(f"{'='*60}")

    result = run_sampler(
        dynamics, potential, dt=dt,
        n_force_evals=n_force_evals,
        kT=1.0, mass=1.0,
        integrator_cls=integrator_cls,
    )

    kl = result["kl_divergence"]
    print(f"  KL divergence: {kl}")

    if result.get("nan_detected"):
        print(f"  ** NaN detected, run aborted **")
        return result

    if result["ess_metrics"]:
        print(f"  ESS/force_eval: {result['ess_metrics']['ess_per_force_eval']:.6f}")
        print(f"  Autocorr time:  {result['ess_metrics']['tau']:.1f}")

    if result["ergodicity"]:
        erg = result["ergodicity"]
        print(f"  Ergodicity:     {erg['score']:.4f} ({'ergodic' if erg['ergodic'] else 'NOT ergodic'})")
        print(f"    KS component: {erg['ks_component']:.4f}")
        print(f"    Var component:{erg['var_component']:.4f}")
        print(f"    Coverage:     {erg['coverage']:.4f}")

    ttt = result["time_to_threshold_force_evals"]
    print(f"  TTT (KL<0.01): {ttt}")
    print(f"  Wall time:      {result['wall_seconds']:.1f}s")
    print(f"  N samples:      {result['n_samples']}")

    return result


def main():
    results = {}

    # ---- Stage 1: Quick iteration ----
    print("\n" + "#"*60)
    print("# STAGE 1: Quick iteration")
    print("#"*60)

    ho = HarmonicOscillator1D(omega=1.0)
    dw = DoubleWell2D(barrier_height=1.0, y_stiffness=0.5)

    # Parent baseline: single Log-Osc
    for pot, dt_val in [(ho, 0.005), (dw, 0.035)]:
        dyn = LogOscThermostat(dim=pot.dim, kT=1.0, Q=0.8 if pot.dim == 1 else 1.0)
        r = evaluate_variant("Log-Osc (parent)", dyn, LogOscVelocityVerlet, pot, dt=dt_val)
        results[f"parent_{pot.name}"] = r

    # Variant A: Dual Log-Osc
    for pot, dt_val in [(ho, 0.005), (dw, 0.02)]:
        dyn = DualLogOsc(dim=pot.dim, kT=1.0, Q1=0.3, Q2=3.0)
        r = evaluate_variant("DualLogOsc(Q1=0.3,Q2=3)", dyn, DualLogOscVerlet, pot, dt=dt_val)
        results[f"dual_A_{pot.name}"] = r

    # Variant B: Dual Log-Osc with Cross-Coupling
    for pot, dt_val in [(ho, 0.005), (dw, 0.02)]:
        dyn = DualLogOscCross(dim=pot.dim, kT=1.0, Q1=0.3, Q2=3.0, epsilon=0.5)
        r = evaluate_variant("DualLogOscCross(Q1=0.3,Q2=3,eps=0.5)", dyn, DualLogOscVerlet, pot, dt=dt_val)
        results[f"dual_cross_{pot.name}"] = r

    # Variant C: Temp Pulse
    for pot, dt_val in [(ho, 0.005), (dw, 0.02)]:
        dyn = LogOscTempPulse(dim=pot.dim, kT=1.0, Q=0.8 if pot.dim == 1 else 1.0,
                              omega_z=0.1, amplitude=0.5)
        r = evaluate_variant("TempPulse(Q=1,wz=0.1,A=0.5)", dyn, LogOscTempPulseVerlet, pot, dt=dt_val)
        results[f"temp_pulse_{pot.name}"] = r

    # Variant E: Multi-Scale
    for pot, dt_val in [(ho, 0.005), (dw, 0.02)]:
        dyn = MultiScaleLogOsc(dim=pot.dim, kT=1.0, Q_fast=0.1, Q_med=1.0, Q_slow=10.0)
        r = evaluate_variant("MultiScale(0.1,1,10)", dyn, MultiScaleLogOscVerlet, pot, dt=dt_val)
        results[f"multi_scale_{pot.name}"] = r

    # ---- Stage 2: GMM (the target!) ----
    print("\n" + "#"*60)
    print("# STAGE 2: Gaussian Mixture 2D (the target)")
    print("#"*60)

    gmm = GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5)

    # Parent baseline
    dyn = LogOscThermostat(dim=2, kT=1.0, Q=0.5)
    r = evaluate_variant("Log-Osc parent (Q=0.5)", dyn, LogOscVelocityVerlet, gmm, dt=0.02)
    results["parent_gmm"] = r

    # Dual Log-Osc with various Q combinations
    for Q1, Q2 in [(0.1, 2.0), (0.3, 3.0), (0.5, 5.0), (0.1, 10.0)]:
        dyn = DualLogOsc(dim=2, kT=1.0, Q1=Q1, Q2=Q2)
        r = evaluate_variant(f"DualLogOsc(Q1={Q1},Q2={Q2})", dyn, DualLogOscVerlet, gmm, dt=0.02)
        results[f"dual_{Q1}_{Q2}_gmm"] = r

    # Dual Cross with various settings
    for Q1, Q2, eps in [(0.3, 3.0, 0.5), (0.1, 5.0, 1.0), (0.3, 5.0, 0.3)]:
        dyn = DualLogOscCross(dim=2, kT=1.0, Q1=Q1, Q2=Q2, epsilon=eps)
        r = evaluate_variant(f"DualCross(Q1={Q1},Q2={Q2},eps={eps})", dyn, DualLogOscVerlet, gmm, dt=0.02)
        results[f"cross_{Q1}_{Q2}_{eps}_gmm"] = r

    # Temp Pulse with various settings
    for Q, wz, A in [(1.0, 0.05, 0.7), (0.5, 0.1, 0.5), (1.0, 0.2, 0.5)]:
        dyn = LogOscTempPulse(dim=2, kT=1.0, Q=Q, omega_z=wz, amplitude=A)
        r = evaluate_variant(f"TempPulse(Q={Q},wz={wz},A={A})", dyn, LogOscTempPulseVerlet, gmm, dt=0.02)
        results[f"pulse_{Q}_{wz}_{A}_gmm"] = r

    # Multi-Scale
    for Qf, Qm, Qs in [(0.1, 1.0, 10.0), (0.05, 0.5, 5.0), (0.2, 2.0, 20.0)]:
        dyn = MultiScaleLogOsc(dim=2, kT=1.0, Q_fast=Qf, Q_med=Qm, Q_slow=Qs)
        r = evaluate_variant(f"MultiScale({Qf},{Qm},{Qs})", dyn, MultiScaleLogOscVerlet, gmm, dt=0.02)
        results[f"multi_{Qf}_{Qm}_{Qs}_gmm"] = r

    # ---- Summary ----
    print("\n" + "="*70)
    print("SUMMARY: GMM KL divergences")
    print("="*70)
    for key, r in results.items():
        if "gmm" in key:
            kl = r["kl_divergence"]
            ess = r["ess_metrics"]["ess_per_force_eval"] if r.get("ess_metrics") else "N/A"
            print(f"  {key:40s}  KL={kl:.4f}" if kl != float('inf') else f"  {key:40s}  KL=inf (NaN)")

    # Save results
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "results.json"

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if obj == float('inf'):
            return "inf"
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
