#!/usr/bin/env python3
"""Orbit tensor-nh-078: Tensor NH thermostat experiments.

Compares tensor NH vs scalar NH and NHC(M=3) baselines.
Optimized for ~10 min total runtime.
"""

import sys
import json
import numpy as np
import time as _time
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from research.eval.evaluator import run_sampler
from research.eval.integrators import VelocityVerletThermostat
from research.eval.baselines import NoseHoover, NoseHooverChain
from research.eval.potentials import DoubleWell2D, HarmonicOscillator1D

from tensor_nh import (
    TensorNH, TensorNHIntegrator, AnisotropicGaussian2D,
    friction_linear, friction_log_osc,
)

import functools
print = functools.partial(print, flush=True)

SEEDS = (42, 123, 456)


def run_one(sampler, sname, potential, dt, n_evals, seed):
    rng = np.random.default_rng(seed)
    icls = TensorNHIntegrator if sname.startswith("Tensor") else VelocityVerletThermostat
    try:
        return run_sampler(sampler, potential, dt=dt, n_force_evals=n_evals,
                           kT=1.0, mass=1.0, rng=rng, integrator_cls=icls)
    except Exception as e:
        return {"sampler": sname, "potential": getattr(potential, 'name', '?'),
                "error": str(e), "kl_divergence": float('inf'),
                "ess_metrics": None, "ergodicity": None}


def run_multi_seed(sampler, sname, potential, dt, n_evals, seeds=SEEDS):
    results = [run_one(sampler, sname, potential, dt, n_evals, s) for s in seeds]
    kls = [r["kl_divergence"] for r in results if r.get("kl_divergence") is not None and r["kl_divergence"] < 1e10]
    taus = [r["ess_metrics"]["tau"] for r in results if r.get("ess_metrics")]
    esses = [r["ess_metrics"]["ess_per_force_eval"] for r in results if r.get("ess_metrics")]
    ergs = [r["ergodicity"]["score"] for r in results if r.get("ergodicity")]
    walls = [r.get("wall_seconds", 0) for r in results]
    return {
        "sampler": sname,
        "kl_mean": float(np.mean(kls)) if kls else None,
        "kl_std": float(np.std(kls)) if len(kls) > 1 else 0.0,
        "tau_mean": float(np.mean(taus)) if taus else None,
        "tau_std": float(np.std(taus)) if len(taus) > 1 else 0.0,
        "ess_mean": float(np.mean(esses)) if esses else None,
        "erg_mean": float(np.mean(ergs)) if ergs else None,
        "erg_std": float(np.std(ergs)) if len(ergs) > 1 else 0.0,
        "wall_mean": float(np.mean(walls)),
        "n_valid": len(kls),
        "errors": [r.get("error") for r in results if r.get("error")],
    }


def fmt(r):
    kl = r["kl_mean"]
    kl_s = f"{kl:.4f}" if kl is not None and kl < 100 else "  inf "
    tau_s = f"{r['tau_mean']:.1f}" if r["tau_mean"] else "  ---"
    ess_s = f"{r['ess_mean']:.5f}" if r["ess_mean"] else "  ---"
    erg_s = f"{r['erg_mean']:.3f}" if r["erg_mean"] is not None else " ---"
    return f"  {r['sampler']:28s} KL={kl_s:>8s}  tau={tau_s:>8s}  ESS/fe={ess_s:>8s}  erg={erg_s:>6s}  wall={r['wall_mean']:.1f}s"


def make_2d_samplers(Q, Q_offdiag=None):
    return [
        ("NH_scalar", NoseHoover(dim=2, kT=1.0, Q=Q)),
        ("NHC_M3", NoseHooverChain(dim=2, chain_length=3, kT=1.0, Q=Q)),
        ("TensorNH_linear", TensorNH(dim=2, kT=1.0, Q=Q, Q_offdiag=Q_offdiag,
            friction_fn=friction_linear, friction_name="linear")),
        ("TensorNH_logosc", TensorNH(dim=2, kT=1.0, Q=Q, Q_offdiag=Q_offdiag,
            friction_fn=friction_log_osc, friction_name="logosc")),
    ]


def main():
    t_global = _time.time()
    all_results = {}

    # =========================================================================
    # EXP 1: Anisotropic Gaussian — Q sweep (200k evals, fast)
    # =========================================================================
    print("=" * 80)
    print("EXP 1: Anisotropic Gaussian 2D — Q sweep")
    print("=" * 80)

    for kappa in [10, 100]:
        print(f"\n--- kappa = {kappa} ---")
        pot = AnisotropicGaussian2D(kappa_x=float(kappa), kappa_y=1.0)

        for Q in [0.3, 1.0, 3.0]:
            print(f"\n  Q = {Q}")
            for sname, sampler in make_2d_samplers(Q):
                r = run_multi_seed(sampler, sname, pot, 0.005, 200_000)
                print(fmt(r))
                all_results[f"aniso_k{kappa}_Q{Q}_{sname}"] = {
                    "exp": "aniso", "kappa": kappa, "Q": Q, **r}

    # =========================================================================
    # EXP 2: Best-Q deep run (500k evals) at kappa=100
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXP 2: Deep run at kappa=100, Q=1.0 (500k evals)")
    print("=" * 80)

    pot100 = AnisotropicGaussian2D(kappa_x=100.0, kappa_y=1.0)
    for sname, sampler in make_2d_samplers(1.0):
        r = run_multi_seed(sampler, sname, pot100, 0.005, 500_000)
        print(fmt(r))
        all_results[f"deep_k100_{sname}"] = {"exp": "deep_k100", **r}

    # =========================================================================
    # EXP 3: Q_offdiag sweep (kappa=100)
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXP 3: Q_offdiag sweep (kappa=100, Q_diag=1)")
    print("=" * 80)

    for Q_off in [0.1, 1.0, 5.0]:
        print(f"\n  Q_offdiag = {Q_off}")
        for fname, ffn in [("linear", friction_linear), ("logosc", friction_log_osc)]:
            sname = f"TensorNH_{fname}"
            sampler = TensorNH(dim=2, kT=1.0, Q=1.0, Q_offdiag=Q_off,
                               friction_fn=ffn, friction_name=fname)
            r = run_multi_seed(sampler, sname, pot100, 0.005, 200_000)
            print(fmt(r))
            all_results[f"qoff_{Q_off}_{fname}"] = {
                "exp": "q_offdiag", "Q_diag": 1.0, "Q_offdiag": Q_off, **r}

    # =========================================================================
    # EXP 4: Double-well 2D
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXP 4: Double-well 2D (Q=1, 500k evals)")
    print("=" * 80)

    pot_dw = DoubleWell2D()
    for sname, sampler in make_2d_samplers(1.0):
        r = run_multi_seed(sampler, sname, pot_dw, 0.01, 500_000)
        print(fmt(r))
        all_results[f"dw_{sname}"] = {"exp": "double_well", **r}

    # =========================================================================
    # EXP 5: 1D Harmonic Oscillator (ergodicity)
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXP 5: 1D Harmonic Oscillator (ergodicity, 1M evals)")
    print("=" * 80)

    pot_ho = HarmonicOscillator1D()
    samplers_1d = [
        ("NH_scalar", NoseHoover(dim=1, kT=1.0, Q=1.0)),
        ("NHC_M3", NoseHooverChain(dim=1, chain_length=3, kT=1.0, Q=1.0)),
        ("TensorNH_linear_1D", TensorNH(dim=1, kT=1.0, Q=1.0,
            friction_fn=friction_linear, friction_name="linear")),
    ]
    for sname, sampler in samplers_1d:
        r = run_multi_seed(sampler, sname, pot_ho, 0.005, 1_000_000)
        print(fmt(r))
        all_results[f"ho_{sname}"] = {"exp": "harmonic_1d", **r}

    # =========================================================================
    # SUMMARY
    # =========================================================================
    elapsed = _time.time() - t_global
    print(f"\n{'=' * 80}")
    print(f"SUMMARY (total wall time: {elapsed:.0f}s)")
    print(f"{'=' * 80}")

    # Key comparison: kappa=100 deep run
    print("\n  [Anisotropic Gaussian, kappa=100, Q=1.0, 500k evals]")
    for key in sorted(all_results):
        if key.startswith("deep_k100"):
            r = all_results[key]
            print(f"    {r['sampler']:28s}  KL={r['kl_mean']:.4f} +/- {r['kl_std']:.4f}  tau={r['tau_mean']:.1f}")

    # Compute metric
    deep = {all_results[k]["sampler"]: all_results[k] for k in all_results if k.startswith("deep_k100")}
    tau_nh = deep.get("NH_scalar", {}).get("tau_mean")
    tau_tl = deep.get("TensorNH_linear", {}).get("tau_mean")
    tau_to = deep.get("TensorNH_logosc", {}).get("tau_mean")
    tau_nhc = deep.get("NHC_M3", {}).get("tau_mean")

    if tau_nh and tau_tl and tau_tl > 0:
        print(f"\n  METRIC: tau_NH / tau_TensorNH_linear = {tau_nh/tau_tl:.2f}")
    if tau_nh and tau_to and tau_to > 0:
        print(f"  METRIC: tau_NH / tau_TensorNH_logosc = {tau_nh/tau_to:.2f}")
    if tau_nhc and tau_tl and tau_tl > 0:
        print(f"  METRIC: tau_NHC / tau_TensorNH_linear = {tau_nhc/tau_tl:.2f}")

    # Double-well
    print("\n  [Double-well 2D, Q=1.0]")
    for key in sorted(all_results):
        if key.startswith("dw_"):
            r = all_results[key]
            kl_s = f"{r['kl_mean']:.4f}" if r['kl_mean'] is not None else "N/A"
            tau_s = f"{r['tau_mean']:.1f}" if r['tau_mean'] is not None else "N/A"
            print(f"    {r['sampler']:28s}  KL={kl_s}  tau={tau_s}")

    # Ergodicity
    print("\n  [1D Harmonic Oscillator, ergodicity]")
    for key in sorted(all_results):
        if key.startswith("ho_"):
            r = all_results[key]
            erg_s = f"{r['erg_mean']:.3f}" if r['erg_mean'] is not None else "N/A"
            print(f"    {r['sampler']:28s}  erg={erg_s}")

    # Save
    out_path = Path(__file__).parent / "results.json"
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
