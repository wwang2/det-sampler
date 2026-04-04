"""Systematic evaluation of all nonlinear friction functions.

Self-contained evaluation script that works on Python 3.9+.
Embeds metric functions from research.eval.evaluator to avoid import issues
with Python 3.10+ type syntax in evaluator.py.

Usage:
    cd /path/to/worktree
    python3 -m orbits.general_nonlinear_004.evaluate_all --mode default
"""
from __future__ import annotations

import sys
import json
import time
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Optional

# These modules use compatible syntax
from research.eval.integrators import ThermostatState
from research.eval.potentials import (
    Potential, HarmonicOscillator1D, DoubleWell2D
)

from orbits.general_nonlinear_004.solution import (
    NonlinearFrictionThermostat, NonlinearFrictionVerlet, FRICTION_CATALOG
)


# ---------------------------------------------------------------------------
# Metric functions (adapted from research.eval.evaluator for Python 3.9)
# ---------------------------------------------------------------------------

def kl_divergence_histogram(samples: np.ndarray, potential: Potential,
                            kT: float, n_bins: int = 100) -> float:
    """KL divergence via histogram for low-D systems."""
    dim = samples.shape[1]

    if dim == 1:
        hist, edges = np.histogram(samples[:, 0], bins=n_bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        log_p = np.array([-potential.energy(np.array([c])) / kT for c in centers])
        log_p -= np.max(log_p)
        p_true = np.exp(log_p)
        p_true /= np.sum(p_true) * (centers[1] - centers[0])
        mask = (hist > 0) & (p_true > 0)
        if np.sum(mask) == 0:
            return float('inf')
        return float(np.sum(hist[mask] * np.log(hist[mask] / p_true[mask]) * (centers[1] - centers[0])))

    elif dim == 2:
        hist, xedges, yedges = np.histogram2d(
            samples[:, 0], samples[:, 1], bins=n_bins, density=True
        )
        dx = xedges[1] - xedges[0]
        dy = yedges[1] - yedges[0]
        xc = 0.5 * (xedges[:-1] + xedges[1:])
        yc = 0.5 * (yedges[:-1] + yedges[1:])
        XX, YY = np.meshgrid(xc, yc, indexing='ij')
        log_p = np.zeros_like(XX)
        for i in range(len(xc)):
            for j in range(len(yc)):
                log_p[i, j] = -potential.energy(np.array([XX[i, j], YY[i, j]])) / kT
        log_p -= np.max(log_p)
        p_true = np.exp(log_p)
        p_true /= np.sum(p_true) * dx * dy
        mask = (hist > 0) & (p_true > 0)
        if np.sum(mask) == 0:
            return float('inf')
        return float(np.sum(hist[mask] * np.log(hist[mask] / p_true[mask]) * dx * dy))
    else:
        raise ValueError(f"Histogram KL only supports dim 1 or 2, got {dim}")


def autocorrelation_time(samples: np.ndarray, max_lag: int = 5000) -> float:
    """Integrated autocorrelation time of the first coordinate."""
    x = samples[:, 0]
    x = x - np.mean(x)
    n = len(x)
    if n < 10:
        return float('inf')
    var = np.var(x)
    if var < 1e-15:
        return float('inf')
    fft_x = np.fft.fft(x, n=2 * n)
    acf = np.real(np.fft.ifft(fft_x * np.conj(fft_x)))[:n] / (var * n)
    tau = 1.0
    for lag in range(1, min(max_lag, n // 2)):
        if acf[lag] < 0.05:
            break
        tau += 2.0 * acf[lag]
    return float(tau)


def effective_sample_size(samples: np.ndarray, n_force_evals: int) -> dict:
    """Compute ESS and ESS per force evaluation."""
    n = len(samples)
    tau = autocorrelation_time(samples)
    ess = n / tau
    return {
        "ess": float(ess),
        "tau": float(tau),
        "ess_per_force_eval": float(ess / max(n_force_evals, 1)),
    }


def ergodicity_score_harmonic(q_samples: np.ndarray, p_samples: np.ndarray,
                               kT: float = 1.0, omega: float = 1.0, mass: float = 1.0) -> dict:
    """Ergodicity score for 1D harmonic oscillator."""
    sigma_q = np.sqrt(kT / omega**2)
    sigma_p = np.sqrt(mass * kT)
    ks_q_stat, ks_q_pval = stats.kstest(q_samples, 'norm', args=(0, sigma_q))
    ks_p_stat, ks_p_pval = stats.kstest(p_samples, 'norm', args=(0, sigma_p))
    var_q_err = abs(np.var(q_samples) - sigma_q**2) / sigma_q**2
    var_p_err = abs(np.var(p_samples) - sigma_p**2) / sigma_p**2
    q_range = 4 * sigma_q
    p_range = 4 * sigma_p
    n_grid = 20
    q_bins = np.linspace(-q_range, q_range, n_grid + 1)
    p_bins = np.linspace(-p_range, p_range, n_grid + 1)
    hist, _, _ = np.histogram2d(q_samples, p_samples, bins=[q_bins, p_bins])
    coverage = float(np.sum(hist > 0)) / (n_grid * n_grid)
    ks_component = max(0.0, 1.0 - max(ks_q_stat, ks_p_stat))
    var_component = max(0.0, 1.0 - max(var_q_err, var_p_err))
    score = (ks_component * var_component * coverage) ** (1.0 / 3.0)
    return {
        "ks_q_stat": float(ks_q_stat),
        "ks_p_stat": float(ks_p_stat),
        "var_q_rel_err": float(var_q_err),
        "var_p_rel_err": float(var_p_err),
        "coverage": coverage,
        "ks_component": float(ks_component),
        "var_component": float(var_component),
        "score": float(score),
        "ergodic": score > 0.85,
    }


def time_to_threshold(kl_trace, threshold=0.01):
    """Force evals needed to reach KL < threshold."""
    for n_evals, kl in kl_trace:
        if kl < threshold:
            return n_evals
    return None


# ---------------------------------------------------------------------------
# Runner (adapted from research.eval.evaluator.run_sampler)
# ---------------------------------------------------------------------------

def run_sampler(dynamics, potential: Potential, dt: float, n_force_evals: int,
                kT: float = 1.0, mass: float = 1.0,
                q0: Optional[np.ndarray] = None,
                burnin_frac: float = 0.1, kl_checkpoints: int = 20,
                rng: Optional[np.random.Generator] = None,
                integrator_cls=None) -> dict:
    """Run a thermostat sampler and compute all metrics."""
    if rng is None:
        rng = np.random.default_rng(42)
    if q0 is None:
        q0 = rng.normal(0, 0.5, size=potential.dim)

    state = dynamics.initial_state(q0, rng=rng)
    if integrator_cls is None:
        from research.eval.integrators import VelocityVerletThermostat
        integrator_cls = VelocityVerletThermostat
    integrator = integrator_cls(dynamics, potential, dt, kT=kT, mass=mass)

    all_q = []
    all_p = []
    kl_trace = []
    checkpoint_interval = max(n_force_evals // kl_checkpoints, 1)
    burnin_evals = int(n_force_evals * burnin_frac)
    nan_detected = False

    t_start = time.time()
    step_count = 0

    while state.n_force_evals < n_force_evals:
        state = integrator.step(state)
        step_count += 1

        if np.any(np.isnan(state.q)) or np.any(np.isnan(state.p)) or \
           np.any(np.isinf(state.q)) or np.any(np.isinf(state.p)):
            nan_detected = True
            break

        if state.n_force_evals >= burnin_evals:
            all_q.append(state.q.copy())
            all_p.append(state.p.copy())

        if state.n_force_evals > 0 and state.n_force_evals % checkpoint_interval < 3:
            if len(all_q) > 100 and potential.dim <= 2:
                q_arr = np.array(all_q)
                kl = kl_divergence_histogram(q_arr, potential, kT, n_bins=50)
                kl_trace.append((state.n_force_evals, kl))

    wall_time = time.time() - t_start

    if nan_detected or len(all_q) == 0:
        return {
            "sampler": dynamics.name,
            "potential": potential.name,
            "kl_divergence": float('inf'),
            "kl_trace": kl_trace,
            "ess_metrics": None,
            "ergodicity": None,
            "time_to_threshold_force_evals": None,
            "wall_seconds": wall_time,
            "n_samples": 0,
            "nan_detected": True,
        }

    q_samples = np.array(all_q)
    p_samples = np.array(all_p)
    actual_force_evals = state.n_force_evals

    if potential.dim <= 2 and len(q_samples) > 0:
        kl_final = kl_divergence_histogram(q_samples, potential, kT)
        kl_final = max(0.0, kl_final)
    else:
        kl_final = None

    ess_metrics = effective_sample_size(q_samples, actual_force_evals) if len(q_samples) > 10 else None

    ergodicity = None
    if isinstance(potential, HarmonicOscillator1D) and len(q_samples) > 100:
        ergodicity = ergodicity_score_harmonic(
            q_samples[:, 0], p_samples[:, 0],
            kT=kT, omega=potential.omega, mass=mass,
        )

    ttt = time_to_threshold(kl_trace, threshold=0.01)

    return {
        "sampler": dynamics.name,
        "potential": potential.name,
        "kl_divergence": kl_final,
        "kl_trace": kl_trace,
        "ess_metrics": ess_metrics,
        "ergodicity": ergodicity,
        "time_to_threshold_force_evals": ttt,
        "wall_seconds": wall_time,
        "n_samples": len(q_samples),
        "nan_detected": False,
    }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
N_FORCE_EVALS = 1_000_000
KT = 1.0
MASS = 1.0
DEFAULT_Q = 1.0
DEFAULT_DT_HO = 0.01
DEFAULT_DT_DW = 0.01

FRICTION_KEYS = list(FRICTION_CATALOG.keys())


def run_single(friction_key, potential, Q, dt):
    """Run a single friction function on a single potential."""
    dim = potential.dim
    dynamics = NonlinearFrictionThermostat(friction_key, dim=dim, kT=KT, mass=MASS, Q=Q)
    rng = np.random.default_rng(SEED)
    result = run_sampler(
        dynamics, potential, dt=dt, n_force_evals=N_FORCE_EVALS,
        kT=KT, mass=MASS, rng=rng, integrator_cls=NonlinearFrictionVerlet
    )
    return result


def run_default_comparison():
    """Run all friction functions with default Q=1.0 and dt=0.01."""
    ho = HarmonicOscillator1D(omega=1.0)
    dw = DoubleWell2D(barrier_height=1.0, y_stiffness=0.5)

    results = {}
    for key in FRICTION_KEYS:
        label = FRICTION_CATALOG[key]["label"]
        print(f"\n{'='*60}")
        print(f"  {label} (Q={DEFAULT_Q}, dt_ho={DEFAULT_DT_HO}, dt_dw={DEFAULT_DT_DW})")
        print(f"{'='*60}")

        print(f"  Running on 1D HO...")
        r_ho = run_single(key, ho, Q=DEFAULT_Q, dt=DEFAULT_DT_HO)
        ho_kl = r_ho["kl_divergence"]
        ho_erg = r_ho["ergodicity"]["score"] if r_ho["ergodicity"] else 0.0
        ho_ess = r_ho["ess_metrics"]["ess_per_force_eval"] if r_ho["ess_metrics"] else 0.0
        print(f"    KL={ho_kl:.4f}, Erg={ho_erg:.4f}, ESS/fe={ho_ess:.5f}")

        print(f"  Running on 2D DW...")
        r_dw = run_single(key, dw, Q=DEFAULT_Q, dt=DEFAULT_DT_DW)
        dw_kl = r_dw["kl_divergence"]
        dw_ess = r_dw["ess_metrics"]["ess_per_force_eval"] if r_dw["ess_metrics"] else 0.0
        dw_ttt = r_dw["time_to_threshold_force_evals"]
        print(f"    KL={dw_kl:.4f}, ESS/fe={dw_ess:.5f}, TTT={dw_ttt}")

        results[key] = {
            "label": label,
            "ho_kl": ho_kl if ho_kl != float('inf') else 999.0,
            "ho_ergodicity": ho_erg,
            "ho_ess_per_fe": ho_ess,
            "dw_kl": dw_kl if dw_kl != float('inf') else 999.0,
            "dw_ess_per_fe": dw_ess,
            "dw_ttt": dw_ttt,
            "ho_nan": r_ho.get("nan_detected", False),
            "dw_nan": r_dw.get("nan_detected", False),
        }

    return results


def run_q_scan(friction_key, potential, dt, q_values):
    """Scan Q values for a single friction function."""
    results = []
    for Q in q_values:
        r = run_single(friction_key, potential, Q=Q, dt=dt)
        kl = r["kl_divergence"]
        erg = r["ergodicity"]["score"] if r["ergodicity"] else None
        ess = r["ess_metrics"]["ess_per_force_eval"] if r["ess_metrics"] else 0.0
        ttt = r["time_to_threshold_force_evals"]
        results.append({
            "Q": Q, "kl": kl if kl != float('inf') else 999.0,
            "ergodicity": erg, "ess_per_fe": ess, "ttt": ttt,
            "nan": r.get("nan_detected", False),
        })
        erg_str = f"{erg:.4f}" if erg is not None else "N/A"
        print(f"    Q={Q:.1f}: KL={kl:.4f}, Erg={erg_str}, ESS/fe={ess:.5f}")
    return results


def run_dt_scan(friction_key, potential, Q, dt_values):
    """Scan dt values for a single friction function."""
    results = []
    for dt in dt_values:
        r = run_single(friction_key, potential, Q=Q, dt=dt)
        kl = r["kl_divergence"]
        erg = r["ergodicity"]["score"] if r["ergodicity"] else None
        ess = r["ess_metrics"]["ess_per_force_eval"] if r["ess_metrics"] else 0.0
        ttt = r["time_to_threshold_force_evals"]
        nan = r.get("nan_detected", False)
        results.append({
            "dt": dt, "kl": kl if kl != float('inf') else 999.0,
            "ergodicity": erg, "ess_per_fe": ess, "ttt": ttt, "nan": nan,
        })
        erg_str = f"{erg:.4f}" if erg is not None else "N/A"
        nan_str = " [NaN!]" if nan else ""
        print(f"    dt={dt:.4f}: KL={kl:.4f}, Erg={erg_str}, ESS/fe={ess:.5f}{nan_str}")
    return results


def print_summary_table(results):
    """Print a formatted comparison table."""
    print(f"\n{'='*100}")
    print(f"  SUMMARY: Default Parameters (Q=1.0, dt=0.01)")
    print(f"{'='*100}")
    header = f"{'Function':<20} {'HO KL':>8} {'HO Erg':>8} {'HO ESS/fe':>10} {'DW KL':>8} {'DW ESS/fe':>10} {'DW TTT':>10}"
    print(header)
    print("-" * 100)

    sorted_keys = sorted(results.keys(), key=lambda k: results[k]["ho_ergodicity"], reverse=True)
    for key in sorted_keys:
        r = results[key]
        ttt_str = str(r["dw_ttt"]) if r["dw_ttt"] is not None else "never"
        nan_flag = ""
        if r["ho_nan"] or r["dw_nan"]:
            nan_flag = " [NaN]"
        print(f"{r['label']:<20} {r['ho_kl']:>8.4f} {r['ho_ergodicity']:>8.4f} {r['ho_ess_per_fe']:>10.5f} "
              f"{r['dw_kl']:>8.4f} {r['dw_ess_per_fe']:>10.5f} {ttt_str:>10}{nan_flag}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["default", "q-scan", "dt-scan", "optimized"],
                        default="default")
    parser.add_argument("--friction", type=str, default=None,
                        help="Single friction key to test (default: all)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent

    if args.mode == "default":
        results = run_default_comparison()
        print_summary_table(results)
        out_path = out_dir / "results_default.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")

    elif args.mode == "q-scan":
        ho = HarmonicOscillator1D(omega=1.0)
        q_values = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]
        keys = [args.friction] if args.friction else FRICTION_KEYS
        all_results = {}
        for key in keys:
            label = FRICTION_CATALOG[key]["label"]
            print(f"\n--- Q-scan: {label} on 1D HO (dt=0.01) ---")
            all_results[key] = run_q_scan(key, ho, dt=0.01, q_values=q_values)
        out_path = out_dir / "results_qscan.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")

    elif args.mode == "dt-scan":
        dw = DoubleWell2D(barrier_height=1.0, y_stiffness=0.5)
        dt_values = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
        keys = [args.friction] if args.friction else FRICTION_KEYS
        all_results = {}
        for key in keys:
            label = FRICTION_CATALOG[key]["label"]
            print(f"\n--- dt-scan: {label} on 2D DW (Q=1.0) ---")
            all_results[key] = run_dt_scan(key, dw, Q=1.0, dt_values=dt_values)
        out_path = out_dir / "results_dtscan.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")

    elif args.mode == "optimized":
        # Will be updated with optimized params after scans
        ho = HarmonicOscillator1D(omega=1.0)
        dw = DoubleWell2D(barrier_height=1.0, y_stiffness=0.5)
        keys = [args.friction] if args.friction else FRICTION_KEYS

        # Use best Q from q-scan results if available
        OPTIMIZED_PARAMS = {}
        qscan_path = out_dir / "results_qscan.json"
        if qscan_path.exists():
            with open(qscan_path) as f:
                qscan = json.load(f)
            for key in keys:
                if key in qscan:
                    entries = qscan[key]
                    # Find Q with best ergodicity for HO
                    valid = [e for e in entries if e.get("ergodicity") is not None and not e.get("nan")]
                    if valid:
                        best = max(valid, key=lambda e: e["ergodicity"])
                        OPTIMIZED_PARAMS[key] = {"Q_ho": best["Q"]}

        dtscan_path = out_dir / "results_dtscan.json"
        if dtscan_path.exists():
            with open(dtscan_path) as f:
                dtscan = json.load(f)
            for key in keys:
                if key in dtscan:
                    entries = dtscan[key]
                    valid = [e for e in entries if not e.get("nan") and e["kl"] < 900]
                    if valid:
                        best = min(valid, key=lambda e: e["kl"])
                        if key not in OPTIMIZED_PARAMS:
                            OPTIMIZED_PARAMS[key] = {}
                        OPTIMIZED_PARAMS[key]["dt_dw"] = best["dt"]

        results = {}
        for key in keys:
            params = OPTIMIZED_PARAMS.get(key, {})
            Q_ho = params.get("Q_ho", 1.0)
            dt_ho = 0.01
            Q_dw = 1.0
            dt_dw = params.get("dt_dw", 0.01)
            label = FRICTION_CATALOG[key]["label"]

            print(f"\n--- Optimized: {label} ---")
            print(f"  HO: Q={Q_ho}, dt={dt_ho}")
            r_ho = run_single(key, ho, Q=Q_ho, dt=dt_ho)
            ho_kl = r_ho["kl_divergence"]
            ho_erg = r_ho["ergodicity"]["score"] if r_ho["ergodicity"] else 0.0
            ho_ess = r_ho["ess_metrics"]["ess_per_force_eval"] if r_ho["ess_metrics"] else 0.0
            print(f"    KL={ho_kl:.4f}, Erg={ho_erg:.4f}")

            print(f"  DW: Q={Q_dw}, dt={dt_dw}")
            r_dw = run_single(key, dw, Q=Q_dw, dt=dt_dw)
            dw_kl = r_dw["kl_divergence"]
            dw_ess = r_dw["ess_metrics"]["ess_per_force_eval"] if r_dw["ess_metrics"] else 0.0
            dw_ttt = r_dw["time_to_threshold_force_evals"]
            print(f"    KL={dw_kl:.4f}, ESS/fe={dw_ess:.5f}, TTT={dw_ttt}")

            results[key] = {
                "label": label,
                "Q_ho": Q_ho, "dt_ho": dt_ho,
                "Q_dw": Q_dw, "dt_dw": dt_dw,
                "ho_kl": ho_kl if ho_kl != float('inf') else 999.0,
                "ho_ergodicity": ho_erg,
                "ho_ess_per_fe": ho_ess,
                "dw_kl": dw_kl if dw_kl != float('inf') else 999.0,
                "dw_ess_per_fe": dw_ess,
                "dw_ttt": dw_ttt,
            }

        out_path = out_dir / "results_optimized.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")
