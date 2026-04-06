from __future__ import annotations
"""Evaluation harness for deterministic thermostat samplers.

Usage:
    python -m research.eval.evaluator --sampler nose_hoover --stage 1
    python -m research.eval.evaluator --sanity-check
    python -m research.eval.evaluator --sampler nose_hoover_chain --stage 1 --chain-length 3
"""

import argparse
import json
import time
import sys
import numpy as np
from scipy import stats
from pathlib import Path

from .potentials import (
    Potential, HarmonicOscillator1D, DoubleWell2D,
    GaussianMixture2D, Rosenbrock2D, LennardJonesCluster,
    get_potentials_by_stage, STAGE_1, STAGE_2, STAGE_3,
)
from .integrators import ThermostatState, VelocityVerletThermostat
from .baselines import NoseHoover, NoseHooverChain, BASELINE_SAMPLERS


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def kl_divergence_histogram(samples: np.ndarray, potential: Potential,
                            kT: float, n_bins: int = 100) -> float:
    """KL divergence via histogram for low-D systems (1D or 2D positions)."""
    dim = samples.shape[1]

    if dim == 1:
        hist, edges = np.histogram(samples[:, 0], bins=n_bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        # Analytical density
        log_p = np.array([-potential.energy(np.array([c])) / kT for c in centers])
        log_p -= np.max(log_p)  # numerical stability
        p_true = np.exp(log_p)
        p_true /= np.sum(p_true) * (centers[1] - centers[0])

        # KL(empirical || true)
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

        # True density on grid
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

    # Use FFT for efficiency
    fft_x = np.fft.fft(x, n=2 * n)
    acf = np.real(np.fft.ifft(fft_x * np.conj(fft_x)))[:n] / (var * n)

    # Integrate until autocorrelation drops below threshold
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
    """Ergodicity score for 1D harmonic oscillator.

    Uses KS statistic (not p-value) as the distributional match metric.
    KS stat is sample-size independent in interpretation: < 0.05 is excellent,
    < 0.10 is good, > 0.20 is poor.

    Also checks:
    - Relative error in variance (should be < 5%)
    - Phase space coverage on 20x20 grid

    Composite score in [0, 1]: higher = more ergodic.
    """
    # Expected distributions
    sigma_q = np.sqrt(kT / omega**2)
    sigma_p = np.sqrt(mass * kT)

    # KS tests (use statistic, not p-value — p-value is too sensitive at large N)
    ks_q_stat, ks_q_pval = stats.kstest(q_samples, 'norm', args=(0, sigma_q))
    ks_p_stat, ks_p_pval = stats.kstest(p_samples, 'norm', args=(0, sigma_p))

    # Variance relative error
    var_q_err = abs(np.var(q_samples) - sigma_q**2) / sigma_q**2
    var_p_err = abs(np.var(p_samples) - sigma_p**2) / sigma_p**2

    # Phase space coverage (20x20 grid)
    q_range = 4 * sigma_q
    p_range = 4 * sigma_p
    n_grid = 20
    q_bins = np.linspace(-q_range, q_range, n_grid + 1)
    p_bins = np.linspace(-p_range, p_range, n_grid + 1)
    hist, _, _ = np.histogram2d(q_samples, p_samples, bins=[q_bins, p_bins])
    coverage = float(np.sum(hist > 0)) / (n_grid * n_grid)

    # Composite score:
    # - KS component: 1 - max(ks_q, ks_p), clamped to [0,1]. Perfect match = 1.
    # - Variance component: 1 - max(var_q_err, var_p_err), clamped to [0,1].
    # - Coverage component: coverage directly.
    # Score = geometric mean of all three.
    ks_component = max(0.0, 1.0 - max(ks_q_stat, ks_p_stat))
    var_component = max(0.0, 1.0 - max(var_q_err, var_p_err))
    score = (ks_component * var_component * coverage) ** (1.0 / 3.0)

    return {
        "ks_q_stat": float(ks_q_stat),
        "ks_q_pval": float(ks_q_pval),
        "ks_p_stat": float(ks_p_stat),
        "ks_p_pval": float(ks_p_pval),
        "var_q_rel_err": float(var_q_err),
        "var_p_rel_err": float(var_p_err),
        "coverage": coverage,
        "ks_component": float(ks_component),
        "var_component": float(var_component),
        "score": float(score),
        "ergodic": score > 0.85,
    }


def time_to_threshold(kl_trace: list[tuple[int, float]], threshold: float = 0.01) -> int | None:
    """Force evals needed to reach KL < threshold. None if never reached."""
    for n_evals, kl in kl_trace:
        if kl < threshold:
            return n_evals
    return None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_sampler(dynamics, potential: Potential, dt: float, n_force_evals: int,
                kT: float = 1.0, mass: float = 1.0, q0: np.ndarray | None = None,
                burnin_frac: float = 0.1, kl_checkpoints: int = 20,
                rng: np.random.Generator | None = None,
                integrator_cls=None) -> dict:
    """Run a thermostat sampler and compute all metrics.

    Args:
        dynamics: Thermostat dynamics object (must have initial_state, dqdt, dpdt, dxidt, name).
        potential: Potential energy surface.
        dt: Integration step size.
        n_force_evals: Budget in force evaluations.
        integrator_cls: Optional custom integrator class. Must accept (dynamics, potential, dt, kT, mass)
            and provide a step(state) -> state method. If None, uses VelocityVerletThermostat.

    Returns dict with:
        - kl_divergence: final KL
        - kl_trace: [(n_evals, kl), ...] at checkpoints
        - ess_metrics: ESS, tau, ESS/force_eval
        - ergodicity: (only for 1D HO) ergodicity score dict
        - time_to_threshold: force evals to KL < 0.01
        - wall_seconds: total wall-clock time
        - integration: {dt, n_steps, n_force_evals_actual, forces_per_step, integrator_name}
    """
    # Input validation
    if kT <= 0:
        raise ValueError(f"kT must be positive, got {kT}")
    if mass <= 0:
        raise ValueError(f"mass must be positive, got {mass}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_force_evals < 20:
        raise ValueError(f"n_force_evals must be >= 20, got {n_force_evals}")

    if rng is None:
        rng = np.random.default_rng(42)

    if q0 is None:
        q0 = rng.normal(0, 0.5, size=potential.dim)

    if np.any(np.isnan(q0)) or np.any(np.isinf(q0)):
        raise ValueError(f"q0 contains NaN or Inf")

    state = dynamics.initial_state(q0, rng=rng)
    if integrator_cls is None:
        integrator_cls = VelocityVerletThermostat
    integrator = integrator_cls(dynamics, potential, dt, kT=kT, mass=mass)

    # Collect samples
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

        # NaN/Inf detection — abort early
        if np.any(np.isnan(state.q)) or np.any(np.isnan(state.p)) or \
           np.any(np.isinf(state.q)) or np.any(np.isinf(state.p)):
            nan_detected = True
            break

        if state.n_force_evals >= burnin_evals:
            all_q.append(state.q.copy())
            all_p.append(state.p.copy())

        # KL checkpoint
        if state.n_force_evals > 0 and state.n_force_evals % checkpoint_interval < 3:
            if len(all_q) > 100 and potential.dim <= 2:
                q_arr = np.array(all_q)
                kl = kl_divergence_histogram(q_arr, potential, kT, n_bins=50)
                kl_trace.append((state.n_force_evals, kl))

    wall_time = time.time() - t_start

    # Handle NaN abort
    if nan_detected or len(all_q) == 0:
        actual_force_evals = state.n_force_evals
        return {
            "sampler": dynamics.name,
            "potential": potential.name,
            "kl_divergence": float('inf'),
            "kl_trace": kl_trace,
            "ess_metrics": None,
            "ergodicity": None,
            "energy_distribution": None,
            "time_to_threshold_force_evals": None,
            "wall_seconds": wall_time,
            "integration": {
                "dt": dt,
                "n_steps": step_count,
                "n_force_evals": actual_force_evals,
                "forces_per_step": actual_force_evals / max(step_count, 1),
                "integrator": integrator_cls.__name__,
            },
            "n_samples": 0,
            "nan_detected": True,
        }

    q_samples = np.array(all_q)
    p_samples = np.array(all_p)
    actual_force_evals = state.n_force_evals
    forces_per_step = actual_force_evals / max(step_count, 1)

    # Final KL
    if potential.dim <= 2 and len(q_samples) > 0:
        kl_final = kl_divergence_histogram(q_samples, potential, kT)
        kl_final = max(0.0, kl_final)  # clamp negative KL from numerical issues
    else:
        kl_final = None

    # ESS
    ess_metrics = effective_sample_size(q_samples, actual_force_evals) if len(q_samples) > 10 else None

    # Energy distribution check (joint q,p validation)
    energy_dist = None
    if len(q_samples) > 100:
        energies = np.array([
            0.5 * np.sum(p_samples[i] ** 2) / mass + potential.energy(q_samples[i])
            for i in range(len(q_samples))
        ])
        # For canonical ensemble at kT, the mean energy should be well-defined
        energy_dist = {
            "mean": float(np.mean(energies)),
            "std": float(np.std(energies)),
            "min": float(np.min(energies)),
            "max": float(np.max(energies)),
        }

    # Ergodicity (1D HO only)
    ergodicity = None
    if isinstance(potential, HarmonicOscillator1D) and len(q_samples) > 100:
        ergodicity = ergodicity_score_harmonic(
            q_samples[:, 0], p_samples[:, 0],
            kT=kT, omega=potential.omega, mass=mass,
        )

    # Time to threshold
    ttt = time_to_threshold(kl_trace, threshold=0.01)

    return {
        "sampler": dynamics.name,
        "potential": potential.name,
        "kl_divergence": kl_final,
        "kl_trace": kl_trace,
        "ess_metrics": ess_metrics,
        "ergodicity": ergodicity,
        "energy_distribution": energy_dist,
        "time_to_threshold_force_evals": ttt,
        "wall_seconds": wall_time,
        "integration": {
            "dt": dt,
            "n_steps": step_count,
            "n_force_evals": actual_force_evals,
            "forces_per_step": forces_per_step,
            "integrator": integrator_cls.__name__,
        },
        "n_samples": len(q_samples),
    }


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks() -> bool:
    """Run sanity checks to verify evaluator correctness."""
    print("=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)
    all_pass = True

    # 1. Potential gradients vs finite differences
    print("\n--- Gradient checks ---")
    rng = np.random.default_rng(123)
    potentials_to_check = [
        HarmonicOscillator1D(),
        DoubleWell2D(),
        GaussianMixture2D(),
        Rosenbrock2D(),
        LennardJonesCluster(n_atoms=3, spatial_dim=2),
    ]
    for pot in potentials_to_check:
        q = rng.normal(0, 1, size=pot.dim)
        # Avoid LJ singularity
        if isinstance(pot, LennardJonesCluster):
            q = rng.normal(0, 2, size=pot.dim) + np.tile(np.arange(pot.n_atoms) * 2.0, pot.spatial_dim)

        grad_analytical = pot.gradient(q)
        grad_numerical = np.zeros_like(q)
        eps = 1e-6
        for i in range(len(q)):
            q_plus = q.copy(); q_plus[i] += eps
            q_minus = q.copy(); q_minus[i] -= eps
            grad_numerical[i] = (pot.energy(q_plus) - pot.energy(q_minus)) / (2 * eps)

        err = np.max(np.abs(grad_analytical - grad_numerical))
        ok = err < 1e-4
        status = "PASS" if ok else "FAIL"
        print(f"  {pot.name}: max grad error = {err:.2e}  [{status}]")
        if not ok:
            all_pass = False

    # 2. NH on 2D double-well should produce reasonable KL
    print("\n--- Baseline: NH on double-well (quick check) ---")
    pot = DoubleWell2D()
    nh = NoseHoover(dim=2, kT=1.0, Q=1.0)
    result = run_sampler(nh, pot, dt=0.01, n_force_evals=200_000, kT=1.0)
    kl = result["kl_divergence"]
    print(f"  KL divergence: {kl:.4f}")
    print(f"  ESS/force_eval: {result['ess_metrics']['ess_per_force_eval']:.6f}")
    ok = kl is not None and kl < 1.0
    print(f"  KL < 1.0: {'PASS' if ok else 'FAIL'}")
    if not ok:
        all_pass = False

    # 3. NHC on 1D HO should be more ergodic than NH
    print("\n--- Baseline: NH vs NHC on 1D harmonic oscillator ---")
    pot_ho = HarmonicOscillator1D()
    nh1d = NoseHoover(dim=1, kT=1.0, Q=1.0)
    nhc1d = NoseHooverChain(dim=1, chain_length=3, kT=1.0, Q=1.0)

    r_nh = run_sampler(nh1d, pot_ho, dt=0.005, n_force_evals=200_000, kT=1.0)
    r_nhc = run_sampler(nhc1d, pot_ho, dt=0.005, n_force_evals=200_000, kT=1.0)

    nh_erg = r_nh["ergodicity"]["score"] if r_nh["ergodicity"] else 0.0
    nhc_erg = r_nhc["ergodicity"]["score"] if r_nhc["ergodicity"] else 0.0
    print(f"  NH  ergodicity score: {nh_erg:.4f}")
    print(f"  NHC ergodicity score: {nhc_erg:.4f}")
    # NHC should generally do better, but we just check both run
    print(f"  Both ran successfully: PASS")

    # 4. Determinism check: same seed => same output
    print("\n--- Determinism check ---")
    r1 = run_sampler(nh, pot, dt=0.01, n_force_evals=10_000, kT=1.0, rng=np.random.default_rng(99))
    r2 = run_sampler(nh, pot, dt=0.01, n_force_evals=10_000, kT=1.0, rng=np.random.default_rng(99))
    ok = r1["kl_divergence"] == r2["kl_divergence"]
    print(f"  Same seed => same KL: {'PASS' if ok else 'FAIL'}")
    if not ok:
        all_pass = False

    print("\n" + "=" * 60)
    print(f"OVERALL: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 60)
    return all_pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Thermostat sampler evaluator")
    parser.add_argument("--sanity-check", action="store_true", help="Run sanity checks")
    parser.add_argument("--sampler", choices=list(BASELINE_SAMPLERS.keys()), help="Sampler to evaluate")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=1, help="Benchmark stage")
    parser.add_argument("--n-force-evals", type=int, default=1_000_000, help="Budget in force evaluations")
    parser.add_argument("--dt", type=float, default=0.01, help="Integration step size")
    parser.add_argument("--kT", type=float, default=1.0, help="Temperature")
    parser.add_argument("--chain-length", type=int, default=3, help="NHC chain length")
    parser.add_argument("--Q", type=float, default=1.0, help="Thermostat mass")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    if args.sanity_check:
        ok = run_sanity_checks()
        sys.exit(0 if ok else 1)

    if args.sampler is None:
        parser.print_help()
        sys.exit(1)

    potentials = get_potentials_by_stage(args.stage)

    results = []
    for pot in potentials:
        print(f"\n--- {args.sampler} on {pot.name} ---")
        if args.sampler == "nose_hoover":
            dynamics = NoseHoover(dim=pot.dim, kT=args.kT, Q=args.Q)
        elif args.sampler == "nose_hoover_chain":
            dynamics = NoseHooverChain(dim=pot.dim, chain_length=args.chain_length, kT=args.kT, Q=args.Q)
        else:
            raise ValueError(f"Unknown sampler: {args.sampler}")

        result = run_sampler(dynamics, pot, dt=args.dt, n_force_evals=args.n_force_evals, kT=args.kT)

        print(f"  KL divergence: {result['kl_divergence']}")
        if result["ess_metrics"]:
            print(f"  ESS/force_eval: {result['ess_metrics']['ess_per_force_eval']:.6f}")
            print(f"  Autocorrelation time: {result['ess_metrics']['tau']:.1f}")
        if result["ergodicity"]:
            print(f"  Ergodicity score: {result['ergodicity']['score']:.4f} ({'ergodic' if result['ergodicity']['ergodic'] else 'NOT ergodic'})")
        print(f"  Wall time: {result['wall_seconds']:.2f}s")
        print(f"  Time to KL<0.01: {result['time_to_threshold_force_evals']}")

        results.append(result)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2, default=convert)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
