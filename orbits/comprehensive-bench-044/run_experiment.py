"""comprehensive-bench-044: All-method tuned benchmark comparison.

Methods:
  1. Log-osc N=3 (our multi-scale parallel thermostat)
  2. Log-osc N=5
  3. NHC M=3 (Nose-Hoover Chain)
  4. NHC M=5
  5. Underdamped Langevin (stochastic baseline)
  6. NH single (plain Nose-Hoover)

Targets:
  1. 1D harmonic (omega=1)
  2. 2D double-well
  3. 2D GMM 5-mode
  4. 5D anisotropic Gaussian (kappa_ratio=100)
  5. 10D anisotropic Gaussian (kappa_ratio=100)
  6. 10D GMM 5-mode

Protocol:
  - Tuning: 3 seeds x 100k force evals per (method, param, target)
  - Final: 10 seeds x 400k force evals with best param
"""

import json
import os
import sys
import time
import numpy as np
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from research.eval.potentials import (
    HarmonicOscillator1D, DoubleWell2D, GaussianMixture2D
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================================
# Potentials
# ============================================================================

class AnisotropicGaussian:
    """U = 0.5 * sum_i kappa_i * q_i^2."""
    name = "anisotropic_gaussian"

    def __init__(self, kappas):
        self.kappas = np.asarray(kappas, dtype=float)
        self.dim = len(self.kappas)

    def energy(self, q):
        return 0.5 * float(np.sum(self.kappas * q * q))

    def gradient(self, q):
        return self.kappas * q


class GaussianMixtureND:
    """N-dimensional Gaussian mixture with modes on random unit vectors."""
    name = "gaussian_mixture_nd"

    def __init__(self, dim, n_modes=5, radius=3.0, sigma=0.5, seed=0):
        self.dim = dim
        self.n_modes = n_modes
        self.sigma = sigma
        self.weights = np.ones(n_modes) / n_modes
        rng = np.random.default_rng(seed)
        raw = rng.normal(0, 1, size=(n_modes, dim))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        self.centers = radius * raw / norms

    def _component_densities(self, q):
        diffs = self.centers - q[np.newaxis, :]
        exponents = -0.5 * np.sum(diffs**2, axis=1) / self.sigma**2
        return self.weights * np.exp(exponents)

    def energy(self, q):
        densities = self._component_densities(q)
        total = np.sum(densities)
        if total < 1e-300:
            return 700.0
        return -np.log(total)

    def gradient(self, q):
        densities = self._component_densities(q)
        total = np.sum(densities)
        if total < 1e-300:
            return np.zeros(self.dim)
        diffs = self.centers - q[np.newaxis, :]
        weighted = densities[:, np.newaxis] * diffs / self.sigma**2
        return -np.sum(weighted, axis=0) / total


# ============================================================================
# Simulators
# ============================================================================

def g_func(xi):
    """Log-oscillator friction: g(xi) = 2*xi / (1 + xi^2)."""
    return 2.0 * xi / (1.0 + xi * xi)


def simulate_multiscale(potential, Qs, dt, n_steps, kT=1.0, mass=1.0, seed=0,
                        record_every=1):
    """Parallel multi-scale log-osc thermostat with BAOAB splitting."""
    rng = np.random.default_rng(seed)
    dim = potential.dim
    Qs = np.asarray(Qs, dtype=float)
    N = len(Qs)

    if hasattr(potential, "kappas"):
        q = rng.normal(0, 1.0, size=dim) / np.sqrt(np.maximum(potential.kappas, 1e-6))
    else:
        q = rng.normal(0, 1.0, size=dim)
    p = rng.normal(0, np.sqrt(mass * kT), size=dim)
    xi = np.zeros(N)

    half = 0.5 * dt
    grad_U = potential.gradient(q)
    n_rec = n_steps // record_every
    qs_rec = np.empty((n_rec, dim))
    rec_i = 0

    for step in range(n_steps):
        K = float(np.sum(p * p)) / mass
        drive = (K - dim * kT) / Qs
        xi = xi + half * drive

        gtot = float(np.sum(g_func(xi)))
        p = p * np.exp(-gtot * half)
        p = p - half * grad_U

        q = q + dt * p / mass
        grad_U = potential.gradient(q)

        p = p - half * grad_U
        gtot = float(np.sum(g_func(xi)))
        p = p * np.exp(-gtot * half)

        K = float(np.sum(p * p)) / mass
        drive = (K - dim * kT) / Qs
        xi = xi + half * drive

        if (step + 1) % record_every == 0 and rec_i < n_rec:
            qs_rec[rec_i] = q
            rec_i += 1

        if not np.isfinite(p).all() or not np.isfinite(q).all():
            break

    return qs_rec[:rec_i]


def simulate_nhc(potential, Qs, dt, n_steps, kT=1.0, mass=1.0, seed=0,
                 record_every=1):
    """Nose-Hoover chain length M with BAOAB-like splitting."""
    rng = np.random.default_rng(seed)
    dim = potential.dim
    Qs = np.asarray(Qs, dtype=float)
    M = len(Qs)

    if hasattr(potential, "kappas"):
        q = rng.normal(0, 1.0, size=dim) / np.sqrt(np.maximum(potential.kappas, 1e-6))
    else:
        q = rng.normal(0, 1.0, size=dim)
    p = rng.normal(0, np.sqrt(mass * kT), size=dim)
    xi = np.zeros(M)

    half = 0.5 * dt
    grad_U = potential.gradient(q)
    n_rec = n_steps // record_every
    qs_rec = np.empty((n_rec, dim))
    rec_i = 0

    def chain_dxi(p_val, xi_val):
        d = np.zeros(M)
        K = float(np.sum(p_val * p_val)) / mass
        d[0] = (K - dim * kT) / Qs[0]
        if M > 1:
            d[0] -= xi_val[1] * xi_val[0]
        for i in range(1, M):
            G = Qs[i - 1] * xi_val[i - 1] ** 2 - kT
            d[i] = G / Qs[i]
            if i < M - 1:
                d[i] -= xi_val[i + 1] * xi_val[i]
        return d

    for step in range(n_steps):
        xi = xi + half * chain_dxi(p, xi)
        p = p * np.exp(-xi[0] * half)
        p = p - half * grad_U
        q = q + dt * p / mass
        grad_U = potential.gradient(q)
        p = p - half * grad_U
        p = p * np.exp(-xi[0] * half)
        xi = xi + half * chain_dxi(p, xi)

        if (step + 1) % record_every == 0 and rec_i < n_rec:
            qs_rec[rec_i] = q
            rec_i += 1

        if not np.isfinite(p).all() or not np.isfinite(q).all():
            break

    return qs_rec[:rec_i]


def simulate_nh_single(potential, Q, dt, n_steps, kT=1.0, mass=1.0, seed=0,
                       record_every=1):
    """Plain single Nose-Hoover thermostat."""
    rng = np.random.default_rng(seed)
    dim = potential.dim

    if hasattr(potential, "kappas"):
        q = rng.normal(0, 1.0, size=dim) / np.sqrt(np.maximum(potential.kappas, 1e-6))
    else:
        q = rng.normal(0, 1.0, size=dim)
    p = rng.normal(0, np.sqrt(mass * kT), size=dim)
    xi = 0.0

    half = 0.5 * dt
    grad_U = potential.gradient(q)
    n_rec = n_steps // record_every
    qs_rec = np.empty((n_rec, dim))
    rec_i = 0

    for step in range(n_steps):
        K = float(np.sum(p * p)) / mass
        xi = xi + half * (K - dim * kT) / Q

        p = p * np.exp(-xi * half)
        p = p - half * grad_U

        q = q + dt * p / mass
        grad_U = potential.gradient(q)

        p = p - half * grad_U
        p = p * np.exp(-xi * half)

        K = float(np.sum(p * p)) / mass
        xi = xi + half * (K - dim * kT) / Q

        if (step + 1) % record_every == 0 and rec_i < n_rec:
            qs_rec[rec_i] = q
            rec_i += 1

        if not np.isfinite(p).all() or not np.isfinite(q).all():
            break

    return qs_rec[:rec_i]


def simulate_langevin(potential, gamma, dt, n_steps, kT=1.0, mass=1.0, seed=0,
                      record_every=1):
    """Underdamped Langevin dynamics (BAOAB splitting, stochastic)."""
    rng = np.random.default_rng(seed)
    dim = potential.dim

    if hasattr(potential, "kappas"):
        q = rng.normal(0, 1.0, size=dim) / np.sqrt(np.maximum(potential.kappas, 1e-6))
    else:
        q = rng.normal(0, 1.0, size=dim)
    p = rng.normal(0, np.sqrt(mass * kT), size=dim)

    half = 0.5 * dt
    grad_U = potential.gradient(q)
    n_rec = n_steps // record_every
    qs_rec = np.empty((n_rec, dim))
    rec_i = 0

    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt(mass * kT * (1.0 - c1 * c1))

    for step in range(n_steps):
        # BAOAB
        p = p - half * grad_U
        q = q + half * p / mass
        p = c1 * p + c2 * rng.normal(0, 1.0, size=dim)
        q = q + half * p / mass
        grad_U = potential.gradient(q)
        p = p - half * grad_U

        if (step + 1) % record_every == 0 and rec_i < n_rec:
            qs_rec[rec_i] = q
            rec_i += 1

        if not np.isfinite(p).all() or not np.isfinite(q).all():
            break

    return qs_rec[:rec_i]


# ============================================================================
# Diagnostics
# ============================================================================

def autocorr_time(x, c=5.0):
    """Integrated autocorrelation time of 1D series x."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    if n < 16 or np.std(x) < 1e-12:
        return float(n)
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f))[:n].real
    acf /= acf[0]
    tau = 1.0
    for k in range(1, n // 4):
        tau += 2.0 * acf[k]
        if k >= c * tau:
            break
    return float(max(tau, 1.0))


def tau_q2_mean(traj):
    """Mean autocorrelation time of q^2 across dimensions."""
    if len(traj) < 64:
        return 1e6
    taus = []
    for d in range(traj.shape[1]):
        x = traj[:, d] ** 2
        if not np.isfinite(x).all():
            return 1e6
        taus.append(autocorr_time(x))
    return float(np.mean(taus))


def tau_q_mean(traj):
    """Mean autocorrelation time of q across dimensions."""
    if len(traj) < 64:
        return 1e6
    taus = []
    for d in range(traj.shape[1]):
        x = traj[:, d]
        if not np.isfinite(x).all():
            return 1e6
        taus.append(autocorr_time(x))
    return float(np.mean(taus))


def count_mode_crossings(traj, potential):
    """Count transitions between nearest-mode assignments."""
    if len(traj) == 0:
        return 0
    centers = potential.centers
    d2 = np.sum((traj[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    assign = np.argmin(d2, axis=1)
    return int(np.sum(assign[1:] != assign[:-1]))


def fraction_modes_visited(traj, potential, threshold=2.0):
    """Fraction of modes visited (within threshold * sigma)."""
    if len(traj) == 0:
        return 0.0
    centers = potential.centers
    sigma = potential.sigma
    visited = set()
    for i, c in enumerate(centers):
        dists = np.sqrt(np.sum((traj - c[np.newaxis, :]) ** 2, axis=1))
        if np.any(dists < threshold * sigma):
            visited.add(i)
    return len(visited) / len(centers)


def divergence_rate(traj):
    """Fraction of trajectory that diverged."""
    if len(traj) == 0:
        return 1.0
    bad = ~np.isfinite(traj).all(axis=1) | (np.abs(traj) > 1e6).any(axis=1)
    return float(np.mean(bad))


# ============================================================================
# Target definitions
# ============================================================================

def make_targets():
    targets = {}

    targets["1d_harmonic"] = {
        "potential": HarmonicOscillator1D(omega=1.0),
        "dt": 0.03,
        "metric": "tau_q2",
        "multimodal": False,
        "kappas_for_Q": np.array([1.0]),
    }

    targets["2d_double_well"] = {
        "potential": DoubleWell2D(barrier_height=1.0, y_stiffness=0.5),
        "dt": 0.03,
        "metric": "tau_q",
        "multimodal": False,
        "kappas_for_Q": np.array([4.0, 1.0]),
    }

    targets["2d_gmm"] = {
        "potential": GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5),
        "dt": 0.03,
        "metric": "mode_crossings",
        "multimodal": True,
        "kappas_for_Q": np.array([4.0, 4.0]),
    }

    kappas_5d = np.array([100.0 ** (i / 4.0) for i in range(5)])
    targets["5d_aniso_gauss"] = {
        "potential": AnisotropicGaussian(kappas_5d),
        "dt": 0.005,
        "metric": "tau_q2",
        "multimodal": False,
        "kappas_for_Q": kappas_5d,
    }

    kappas_10d = np.array([100.0 ** (i / 9.0) for i in range(10)])
    targets["10d_aniso_gauss"] = {
        "potential": AnisotropicGaussian(kappas_10d),
        "dt": 0.005,
        "metric": "tau_q2",
        "multimodal": False,
        "kappas_for_Q": kappas_10d,
    }

    targets["10d_gmm"] = {
        "potential": GaussianMixtureND(dim=10, n_modes=5, radius=3.0, sigma=0.5, seed=0),
        "dt": 0.03,
        "metric": "mode_crossings",
        "multimodal": True,
        "kappas_for_Q": np.ones(10) * 4.0,
    }

    return targets


# ============================================================================
# Method definitions
# ============================================================================

def get_methods_for_target(target_info):
    """Return list of method dicts with param grids for a target."""
    kappas = target_info["kappas_for_Q"]
    kappa_max = float(np.max(kappas))
    kappa_min = float(np.min(kappas))

    methods = []

    # Log-osc N=3
    Q_min = 1.0 / kappa_max
    Q_max = 1.0 / max(kappa_min, 0.01)
    Qs_3 = np.exp(np.linspace(np.log(max(Q_min, 1e-4)), np.log(Q_max), 3))
    methods.append({
        "name": "LogOsc-3",
        "grid": [{"label": "auto", "Qs": Qs_3}],
    })

    # Log-osc N=5
    Qs_5 = np.exp(np.linspace(np.log(max(Q_min, 1e-4)), np.log(Q_max), 5))
    methods.append({
        "name": "LogOsc-5",
        "grid": [{"label": "auto", "Qs": Qs_5}],
    })

    # NHC M=3
    nhc3_grid = []
    for Q_ref in [0.1, 1.0, 10.0, 31.6, 100.0]:
        nhc3_grid.append({"label": f"Q={Q_ref}", "Qs": np.ones(3) * Q_ref})
    methods.append({"name": "NHC-3", "grid": nhc3_grid})

    # NHC M=5
    nhc5_grid = []
    for Q_ref in [0.1, 1.0, 10.0, 31.6, 100.0]:
        nhc5_grid.append({"label": f"Q={Q_ref}", "Qs": np.ones(5) * Q_ref})
    methods.append({"name": "NHC-5", "grid": nhc5_grid})

    # Langevin
    lang_grid = []
    for gamma in [0.1, 1.0, 3.0, 10.0]:
        lang_grid.append({"label": f"g={gamma}", "gamma": gamma})
    methods.append({"name": "Langevin", "grid": lang_grid})

    # NH single
    nh_grid = []
    for Q in [0.1, 1.0, 10.0, 100.0]:
        nh_grid.append({"label": f"Q={Q}", "Q": Q})
    methods.append({"name": "NH-1", "grid": nh_grid})

    return methods


# ============================================================================
# Single simulation task (for multiprocessing)
# ============================================================================

def _run_single(args):
    """Run a single simulation and return metrics."""
    (target_name, method_name, param_entry, seed,
     n_steps, record_every, phase, target_serialized) = args

    # Reconstruct target from serialized info
    tinfo = _deserialize_target(target_serialized)
    pot = tinfo["potential"]
    dt = tinfo["dt"]
    kT = 1.0
    metric_type = tinfo["metric"]
    multimodal = tinfo["multimodal"]

    try:
        if method_name in ("LogOsc-3", "LogOsc-5"):
            Qs = np.array(param_entry["Qs"])
            traj = simulate_multiscale(pot, Qs, dt, n_steps, kT=kT,
                                       seed=seed, record_every=record_every)
        elif method_name in ("NHC-3", "NHC-5"):
            Qs = np.array(param_entry["Qs"])
            traj = simulate_nhc(pot, Qs, dt, n_steps, kT=kT,
                                seed=seed, record_every=record_every)
        elif method_name == "Langevin":
            gamma = param_entry["gamma"]
            traj = simulate_langevin(pot, gamma, dt, n_steps, kT=kT,
                                     seed=seed, record_every=record_every)
        elif method_name == "NH-1":
            Q = param_entry["Q"]
            traj = simulate_nh_single(pot, Q, dt, n_steps, kT=kT,
                                      seed=seed, record_every=record_every)
        else:
            return None

        result = {
            "target": target_name,
            "method": method_name,
            "param_label": param_entry["label"],
            "seed": seed,
            "phase": phase,
            "n_samples": len(traj),
            "diverged": divergence_rate(traj),
        }

        if len(traj) < 64:
            result["tau"] = 1e6
            result["mode_crossings"] = 0
            result["frac_modes"] = 0.0
            return result

        if metric_type == "tau_q2":
            result["tau"] = tau_q2_mean(traj)
        elif metric_type == "tau_q":
            result["tau"] = tau_q_mean(traj)
        elif metric_type == "mode_crossings":
            result["tau"] = tau_q_mean(traj)

        if multimodal and hasattr(pot, "centers"):
            result["mode_crossings"] = count_mode_crossings(traj, pot)
            result["frac_modes"] = fraction_modes_visited(traj, pot)
        else:
            result["mode_crossings"] = 0
            result["frac_modes"] = 1.0

        if phase == "final":
            n_keep = min(5000, len(traj))
            step = max(1, len(traj) // n_keep)
            result["traj_sample"] = traj[::step][:n_keep].tolist()

        return result

    except Exception as e:
        return {
            "target": target_name, "method": method_name,
            "param_label": param_entry["label"], "seed": seed,
            "phase": phase, "error": str(e), "tau": 1e6,
            "mode_crossings": 0, "frac_modes": 0.0, "diverged": 1.0,
        }


def _serialize_target(tname, tinfo):
    """Serialize target info for multiprocessing (avoid pickling complex objects)."""
    s = {
        "name": tname,
        "dt": tinfo["dt"],
        "metric": tinfo["metric"],
        "multimodal": tinfo["multimodal"],
    }
    pot = tinfo["potential"]
    if isinstance(pot, HarmonicOscillator1D):
        s["pot_type"] = "harmonic_1d"
        s["pot_params"] = {"omega": pot.omega}
    elif isinstance(pot, DoubleWell2D):
        s["pot_type"] = "double_well_2d"
        s["pot_params"] = {"barrier_height": pot.a, "y_stiffness": pot.b}
    elif isinstance(pot, GaussianMixture2D):
        s["pot_type"] = "gmm_2d"
        s["pot_params"] = {"n_modes": pot.n_modes, "radius": 3.0, "sigma": pot.sigma}
    elif isinstance(pot, AnisotropicGaussian):
        s["pot_type"] = "aniso_gauss"
        s["pot_params"] = {"kappas": pot.kappas.tolist()}
    elif isinstance(pot, GaussianMixtureND):
        s["pot_type"] = "gmm_nd"
        s["pot_params"] = {"dim": pot.dim, "n_modes": pot.n_modes,
                           "radius": 3.0, "sigma": pot.sigma, "seed": 0}
    return s


def _deserialize_target(s):
    """Reconstruct target from serialized dict."""
    pt = s["pot_type"]
    pp = s["pot_params"]
    if pt == "harmonic_1d":
        pot = HarmonicOscillator1D(**pp)
    elif pt == "double_well_2d":
        pot = DoubleWell2D(**pp)
    elif pt == "gmm_2d":
        pot = GaussianMixture2D(**pp)
    elif pt == "aniso_gauss":
        pot = AnisotropicGaussian(pp["kappas"])
    elif pt == "gmm_nd":
        pot = GaussianMixtureND(**pp)
    else:
        raise ValueError(f"Unknown pot_type: {pt}")
    return {
        "potential": pot,
        "dt": s["dt"],
        "metric": s["metric"],
        "multimodal": s["multimodal"],
    }


def _serialize_param_entry(pe):
    """Make param entry JSON-safe for multiprocessing."""
    out = {"label": pe["label"]}
    if "Qs" in pe:
        out["Qs"] = pe["Qs"].tolist() if isinstance(pe["Qs"], np.ndarray) else pe["Qs"]
    if "gamma" in pe:
        out["gamma"] = pe["gamma"]
    if "Q" in pe:
        out["Q"] = pe["Q"]
    return out


# ============================================================================
# Tuning phase
# ============================================================================

def run_tuning(targets, n_seeds=3, n_force_evals=100_000, n_workers=10):
    print("=" * 70)
    print("TUNING PHASE")
    print("=" * 70)

    record_every = 4
    tasks = []

    for tname, tinfo in targets.items():
        methods = get_methods_for_target(tinfo)
        n_steps = n_force_evals
        ts = _serialize_target(tname, tinfo)
        for method in methods:
            for param_entry in method["grid"]:
                pe_ser = _serialize_param_entry(param_entry)
                for s in range(n_seeds):
                    tasks.append((tname, method["name"], pe_ser,
                                  1000 + s, n_steps, record_every, "tune", ts))

    print(f"  Total tuning tasks: {len(tasks)}")
    t0 = time.time()

    with Pool(processes=min(n_workers, len(tasks), cpu_count())) as pool:
        results = pool.map(_run_single, tasks)

    results = [r for r in results if r is not None]
    elapsed = time.time() - t0
    print(f"  Tuning completed in {elapsed:.1f}s ({len(results)} results)")

    # Find best param for each (target, method)
    best_params = {}
    for tname in targets:
        metric_type = targets[tname]["metric"]
        multimodal = targets[tname]["multimodal"]
        methods = get_methods_for_target(targets[tname])

        for method in methods:
            mname = method["name"]
            method_results = [r for r in results
                              if r["target"] == tname and r["method"] == mname]

            param_scores = {}
            for r in method_results:
                label = r["param_label"]
                if label not in param_scores:
                    param_scores[label] = []
                if multimodal and metric_type == "mode_crossings":
                    param_scores[label].append(r.get("mode_crossings", 0))
                else:
                    param_scores[label].append(r.get("tau", 1e6))

            best_label = None
            best_score = None
            for label, scores in param_scores.items():
                avg = float(np.mean(scores))
                if multimodal and metric_type == "mode_crossings":
                    if best_score is None or avg > best_score:
                        best_score = avg
                        best_label = label
                else:
                    if best_score is None or avg < best_score:
                        best_score = avg
                        best_label = label

            for pe in method["grid"]:
                if pe["label"] == best_label:
                    best_params[(tname, mname)] = {
                        "param_entry": _serialize_param_entry(pe),
                        "tuning_score": best_score,
                        "label": best_label,
                    }
                    break

            if best_score is not None:
                print(f"  {tname:20s} {mname:10s} -> best={best_label} score={best_score:.2f}")

    return best_params, results


# ============================================================================
# Final phase
# ============================================================================

def run_final(targets, best_params, n_seeds=10, n_force_evals=400_000, n_workers=10):
    print("\n" + "=" * 70)
    print("FINAL PHASE")
    print("=" * 70)

    record_every = 4
    tasks = []

    for tname, tinfo in targets.items():
        methods = get_methods_for_target(tinfo)
        n_steps = n_force_evals
        ts = _serialize_target(tname, tinfo)
        for method in methods:
            mname = method["name"]
            key = (tname, mname)
            if key not in best_params:
                continue
            param_entry = best_params[key]["param_entry"]
            for s in range(n_seeds):
                tasks.append((tname, mname, param_entry,
                              5000 + s, n_steps, record_every, "final", ts))

    print(f"  Total final tasks: {len(tasks)}")
    t0 = time.time()

    with Pool(processes=min(n_workers, len(tasks), cpu_count())) as pool:
        results = pool.map(_run_single, tasks)

    results = [r for r in results if r is not None]
    elapsed = time.time() - t0
    print(f"  Final completed in {elapsed:.1f}s ({len(results)} results)")
    return results


# ============================================================================
# Aggregation
# ============================================================================

def aggregate_results(final_results, targets, best_params):
    summary = {}
    for tname in targets:
        metric_type = targets[tname]["metric"]
        multimodal = targets[tname]["multimodal"]
        summary[tname] = {}

        for mname in ["LogOsc-3", "LogOsc-5", "NHC-3", "NHC-5", "Langevin", "NH-1"]:
            key = (tname, mname)
            if key not in best_params:
                continue

            method_results = [r for r in final_results
                              if r["target"] == tname and r["method"] == mname
                              and r["phase"] == "final"]
            if not method_results:
                continue

            taus = [r["tau"] for r in method_results if r["tau"] < 1e5]
            mc = [r["mode_crossings"] for r in method_results]
            fm = [r["frac_modes"] for r in method_results]
            divr = [r["diverged"] for r in method_results]

            entry = {
                "best_param": best_params[key]["label"],
                "tau_mean": float(np.mean(taus)) if taus else 1e6,
                "tau_std": float(np.std(taus)) if taus else 0,
                "mode_crossings_mean": float(np.mean(mc)),
                "mode_crossings_std": float(np.std(mc)),
                "frac_modes_mean": float(np.mean(fm)),
                "diverged_mean": float(np.mean(divr)),
                "n_seeds": len(method_results),
            }
            summary[tname][mname] = entry

    return summary


# ============================================================================
# Figures
# ============================================================================

def make_figures(summary, final_results, targets, best_params):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    METHOD_COLORS = {
        "LogOsc-3": "#2ca02c",
        "LogOsc-5": "#d62728",
        "NHC-3": "#ff7f0e",
        "NHC-5": "#e377c2",
        "Langevin": "#7f7f7f",
        "NH-1": "#1f77b4",
    }
    METHOD_ORDER = ["LogOsc-3", "LogOsc-5", "NHC-3", "NHC-5", "Langevin", "NH-1"]

    # ---- Per-target bar charts ----
    for tname in targets:
        if tname not in summary:
            continue
        tdata = summary[tname]
        multimodal = targets[tname]["multimodal"]

        present = [m for m in METHOD_ORDER if m in tdata]
        if not present:
            continue

        n_panels = 2 if multimodal else 1
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
        if n_panels == 1:
            axes = [axes]

        ax = axes[0]
        x = np.arange(len(present))
        vals = [tdata[m]["tau_mean"] for m in present]
        errs = [tdata[m]["tau_std"] for m in present]
        colors = [METHOD_COLORS[m] for m in present]
        ax.bar(x, vals, yerr=errs, color=colors, capsize=4, edgecolor="k", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(present, rotation=30, ha="right", fontsize=11)
        ax.set_ylabel(r"$\tau_{\mathrm{int}}$ (lower = better)", fontsize=13)
        ax.set_title(f"{tname}: autocorrelation time", fontsize=14)
        ax.tick_params(labelsize=11)

        if multimodal:
            ax2 = axes[1]
            vals2 = [tdata[m]["mode_crossings_mean"] for m in present]
            errs2 = [tdata[m]["mode_crossings_std"] for m in present]
            ax2.bar(x, vals2, yerr=errs2, color=colors, capsize=4,
                    edgecolor="k", linewidth=0.5)
            ax2.set_xticks(x)
            ax2.set_xticklabels(present, rotation=30, ha="right", fontsize=11)
            ax2.set_ylabel("Mode crossings (higher = better)", fontsize=13)
            ax2.set_title(f"{tname}: mode exploration", fontsize=14)
            ax2.tick_params(labelsize=11)

        for i, ax_ in enumerate(axes):
            ax_.text(-0.08, 1.05, f"({'abcdef'[i]})", transform=ax_.transAxes,
                     fontsize=15, fontweight="bold", va="top")

        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"fig_{tname}_metrics.png"), dpi=200,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved fig_{tname}_metrics.png")

    # ---- Density scatter plots for 2D targets ----
    for tname in ["2d_double_well", "2d_gmm"]:
        if tname not in summary:
            continue
        tdata = summary[tname]
        present = [m for m in METHOD_ORDER if m in tdata]
        if not present:
            continue

        n_methods = len(present)
        fig, axes = plt.subplots(1, n_methods, figsize=(3.5 * n_methods, 3.5),
                                 squeeze=False)

        for idx, mname in enumerate(present):
            ax = axes[0, idx]
            trajs = []
            for r in final_results:
                if (r["target"] == tname and r["method"] == mname
                        and r["phase"] == "final" and "traj_sample" in r):
                    trajs.append(np.array(r["traj_sample"]))
            if trajs:
                traj_all = np.concatenate(trajs, axis=0)
                if len(traj_all) > 8000:
                    idx_sub = np.random.default_rng(0).choice(
                        len(traj_all), 8000, replace=False)
                    traj_all = traj_all[idx_sub]
                ax.scatter(traj_all[:, 0], traj_all[:, 1], s=0.5, alpha=0.3,
                           color=METHOD_COLORS[mname], rasterized=True)

            ax.set_title(mname, fontsize=12, color=METHOD_COLORS[mname])
            ax.set_xlabel("$q_1$", fontsize=11)
            if idx == 0:
                ax.set_ylabel("$q_2$", fontsize=11)
            ax.tick_params(labelsize=9)
            ax.set_aspect("equal")

            if tname == "2d_double_well":
                ax.set_xlim(-3, 3)
                ax.set_ylim(-4, 4)
            elif tname == "2d_gmm":
                ax.set_xlim(-5, 5)
                ax.set_ylim(-5, 5)

        fig.suptitle(f"{tname}: density samples", fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"fig_{tname}_density.png"), dpi=200,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved fig_{tname}_density.png")

    # ---- Master summary table ----
    make_summary_table(summary, targets)


def make_summary_table(summary, targets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    METHOD_ORDER = ["LogOsc-3", "LogOsc-5", "NHC-3", "NHC-5", "Langevin", "NH-1"]
    TARGET_ORDER = ["1d_harmonic", "2d_double_well", "2d_gmm",
                    "5d_aniso_gauss", "10d_aniso_gauss", "10d_gmm"]
    TARGET_LABELS = ["1D Harm.", "2D DblWell", "2D GMM", "5D Aniso",
                     "10D Aniso", "10D GMM"]

    cell_text = []
    cell_colors = []

    for ti, tname in enumerate(TARGET_ORDER):
        row_text = []
        row_colors = []
        multimodal = targets.get(tname, {}).get("multimodal", False)

        vals = {}
        for mname in METHOD_ORDER:
            if tname in summary and mname in summary[tname]:
                entry = summary[tname][mname]
                if multimodal:
                    vals[mname] = entry["mode_crossings_mean"]
                else:
                    vals[mname] = entry["tau_mean"]

        best_val = None
        if vals:
            good_vals = [v for v in vals.values() if v < 1e5]
            if good_vals:
                if multimodal:
                    best_val = max(good_vals)
                else:
                    best_val = min(good_vals)

        for mname in METHOD_ORDER:
            if tname in summary and mname in summary[tname]:
                entry = summary[tname][mname]
                if multimodal:
                    val = entry["mode_crossings_mean"]
                    std = entry["mode_crossings_std"]
                    text = f"{val:.0f}\n+/-{std:.0f}"
                    is_best = (best_val is not None and val == best_val)
                else:
                    val = entry["tau_mean"]
                    std = entry["tau_std"]
                    if val > 9999:
                        text = "DIV"
                    elif val > 100:
                        text = f"{val:.0f}\n+/-{std:.0f}"
                    else:
                        text = f"{val:.1f}\n+/-{std:.1f}"
                    is_best = (best_val is not None and val == best_val and val < 1e5)

                if entry["diverged_mean"] > 0.1:
                    color = "#ffcccc"
                elif is_best:
                    color = "#ccffcc"
                else:
                    color = "white"
            else:
                text = "--"
                color = "#f0f0f0"

            row_text.append(text)
            row_colors.append(color)

        cell_text.append(row_text)
        cell_colors.append(row_colors)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        rowLabels=TARGET_LABELS,
        colLabels=METHOD_ORDER,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.2)

    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(fontweight="bold", fontsize=11)
            cell.set_facecolor("#e0e0e0")
        if j == -1:
            cell.set_text_props(fontweight="bold", fontsize=11)
            cell.set_facecolor("#e0e0e0")

    ax.set_title(
        r"Benchmark: $\tau_{\mathrm{int}}$ (unimodal) / mode crossings (multimodal)"
        "\nGreen = best per target; Red = >10% diverged",
        fontsize=13, pad=20)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_summary_table.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_summary_table.png")


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()

    targets = make_targets()
    print(f"Targets: {list(targets.keys())}")
    print(f"Methods: LogOsc-3, LogOsc-5, NHC-3, NHC-5, Langevin, NH-1\n")

    best_params, tune_results = run_tuning(
        targets, n_seeds=3, n_force_evals=100_000, n_workers=10
    )

    final_results = run_final(
        targets, best_params, n_seeds=10, n_force_evals=400_000, n_workers=10
    )

    summary = aggregate_results(final_results, targets, best_params)

    # Save (strip trajectory data)
    output = {
        "summary": {},
        "best_params": {},
        "elapsed_sec": time.time() - t_start,
    }
    for tname, tdata in summary.items():
        output["summary"][tname] = {}
        for mname, entry in tdata.items():
            output["summary"][tname][mname] = entry
    for k, v in best_params.items():
        output["best_params"][f"{k[0]}|{k[1]}"] = {
            "label": v["label"],
            "tuning_score": v["tuning_score"],
        }

    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x)
                  if isinstance(x, (np.floating, np.integer)) else x)
    print(f"\nSaved results to {out_path}")

    print("\nGenerating figures...")
    make_figures(summary, final_results, targets, best_params)

    total_time = time.time() - t_start
    print(f"\nTotal elapsed: {total_time:.1f}s")

    # Print summary
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    METHOD_ORDER = ["LogOsc-3", "LogOsc-5", "NHC-3", "NHC-5", "Langevin", "NH-1"]
    header = f"{'Target':20s}" + "".join(f"{m:>12s}" for m in METHOD_ORDER)
    print(header)
    print("-" * len(header))
    for tname in targets:
        if tname not in summary:
            continue
        multimodal = targets[tname]["multimodal"]
        row = f"{tname:20s}"
        for mname in METHOD_ORDER:
            if mname in summary[tname]:
                e = summary[tname][mname]
                if multimodal:
                    row += f"{e['mode_crossings_mean']:>10.0f}  "
                else:
                    if e["tau_mean"] > 9999:
                        row += f"{'DIV':>10s}  "
                    else:
                        row += f"{e['tau_mean']:>10.1f}  "
            else:
                row += f"{'--':>10s}  "
        print(row)

    return output


if __name__ == "__main__":
    main()
