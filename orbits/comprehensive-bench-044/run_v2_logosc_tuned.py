"""v2: Re-run LogOsc with properly tuned Q ranges.

Test two improvements:
1. Dimension-scaled Q: Q_i = D * kT / kappa_i (not kT/kappa_i)
2. Swept Q: same Q sweep as NHC to find true optimal
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
from research.eval.potentials import HarmonicOscillator1D, DoubleWell2D, GaussianMixture2D

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(OUT_DIR, "figures")

class AnisotropicGaussian:
    name = "anisotropic_gaussian"
    def __init__(self, kappas):
        self.kappas = np.asarray(kappas, dtype=float)
        self.dim = len(self.kappas)
    def energy(self, q):
        return 0.5 * float(np.sum(self.kappas * q * q))
    def gradient(self, q):
        return self.kappas * q

class GaussianMixtureND:
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
        if total < 1e-300: return 700.0
        return -np.log(total)
    def gradient(self, q):
        densities = self._component_densities(q)
        total = np.sum(densities)
        if total < 1e-300: return np.zeros(self.dim)
        diffs = self.centers - q[np.newaxis, :]
        weighted = densities[:, np.newaxis] * diffs / self.sigma**2
        return -np.sum(weighted, axis=0) / total

def g_func(xi):
    return 2.0 * xi / (1.0 + xi * xi)

def simulate_multiscale(potential, Qs, dt, n_steps, kT=1.0, mass=1.0, seed=0,
                        record_every=1):
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
        xi = xi + half * (K - dim * kT) / Qs
        gtot = float(np.sum(g_func(xi)))
        p = p * np.exp(-gtot * half)
        p = p - half * grad_U
        q = q + dt * p / mass
        grad_U = potential.gradient(q)
        p = p - half * grad_U
        gtot = float(np.sum(g_func(xi)))
        p = p * np.exp(-gtot * half)
        K = float(np.sum(p * p)) / mass
        xi = xi + half * (K - dim * kT) / Qs
        if (step + 1) % record_every == 0 and rec_i < n_rec:
            qs_rec[rec_i] = q
            rec_i += 1
        if not np.isfinite(p).all() or not np.isfinite(q).all():
            break
    return qs_rec[:rec_i]

def autocorr_time(x, c=5.0):
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    if n < 16 or np.std(x) < 1e-12: return float(n)
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f))[:n].real
    acf /= acf[0]
    tau = 1.0
    for k in range(1, n // 4):
        tau += 2.0 * acf[k]
        if k >= c * tau: break
    return float(max(tau, 1.0))

def tau_q2_mean(traj):
    if len(traj) < 64: return 1e6
    taus = []
    for d in range(traj.shape[1]):
        x = traj[:, d] ** 2
        if not np.isfinite(x).all(): return 1e6
        taus.append(autocorr_time(x))
    return float(np.mean(taus))

def count_mode_crossings(traj, potential):
    if len(traj) == 0: return 0
    centers = potential.centers
    d2 = np.sum((traj[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    assign = np.argmin(d2, axis=1)
    return int(np.sum(assign[1:] != assign[:-1]))


def run_single(args):
    """Single simulation task."""
    target_name, pot_info, N_thermo, Q_center, seed, n_steps, record_every = args
    
    pot = _make_pot(pot_info)
    dt = pot_info["dt"]
    dim = pot.dim
    kT = 1.0
    
    try:
        # Log-uniform Q around Q_center
        if N_thermo == 1:
            Qs = np.array([Q_center])
        else:
            log_spread = 1.5  # +/- 1.5 decades around center
            Qs = np.exp(np.linspace(
                np.log(Q_center) - log_spread,
                np.log(Q_center) + log_spread,
                N_thermo
            ))
        
        traj = simulate_multiscale(pot, Qs, dt, n_steps, kT=kT,
                                   seed=seed, record_every=record_every)
        
        if len(traj) < 64:
            return {"target": target_name, "N": N_thermo, "Q_center": Q_center,
                    "seed": seed, "tau": 1e6, "crossings": 0}
        
        result = {
            "target": target_name, "N": N_thermo, "Q_center": Q_center,
            "seed": seed, "tau": tau_q2_mean(traj),
        }
        if hasattr(pot, "centers"):
            result["crossings"] = count_mode_crossings(traj, pot)
        return result
    except Exception as e:
        return {"target": target_name, "N": N_thermo, "Q_center": Q_center,
                "seed": seed, "tau": 1e6, "crossings": 0, "error": str(e)}


def _make_pot(info):
    t = info["type"]
    if t == "aniso_5d":
        return AnisotropicGaussian(np.array([100.0 ** (i / 4.0) for i in range(5)]))
    elif t == "aniso_10d":
        return AnisotropicGaussian(np.array([100.0 ** (i / 9.0) for i in range(10)]))
    elif t == "gmm_10d":
        return GaussianMixtureND(dim=10, n_modes=5, radius=3.0, sigma=0.5, seed=0)
    elif t == "harmonic_1d":
        return HarmonicOscillator1D(omega=1.0)
    elif t == "dw_2d":
        return DoubleWell2D()
    elif t == "gmm_2d":
        return GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5)


def main():
    t0 = time.time()
    
    targets = {
        "1d_harmonic": {"type": "harmonic_1d", "dt": 0.03, "metric": "tau"},
        "2d_double_well": {"type": "dw_2d", "dt": 0.03, "metric": "tau"},
        "2d_gmm": {"type": "gmm_2d", "dt": 0.03, "metric": "crossings"},
        "5d_aniso_gauss": {"type": "aniso_5d", "dt": 0.005, "metric": "tau"},
        "10d_aniso_gauss": {"type": "aniso_10d", "dt": 0.005, "metric": "tau"},
        "10d_gmm": {"type": "gmm_10d", "dt": 0.03, "metric": "crossings"},
    }
    
    # Q center sweep: much wider range
    Q_centers = [0.01, 0.1, 1.0, 5.0, 10.0, 31.6, 100.0, 316.0]
    N_thermos = [3, 5]
    n_seeds_tune = 3
    n_steps_tune = 100_000
    record_every = 4
    
    # Build tuning tasks
    tasks = []
    for tname, tinfo in targets.items():
        for N in N_thermos:
            for Qc in Q_centers:
                for s in range(n_seeds_tune):
                    tasks.append((tname, tinfo, N, Qc, 1000 + s,
                                  n_steps_tune, record_every))
    
    print(f"LogOsc Q-sweep tuning: {len(tasks)} tasks")
    
    with Pool(processes=min(10, cpu_count())) as pool:
        results = pool.map(run_single, tasks)
    
    results = [r for r in results if r is not None]
    
    # Find best Q_center for each (target, N)
    print("\n" + "=" * 80)
    print("TUNING RESULTS: LogOsc with swept Q_center")
    print("=" * 80)
    
    best_params = {}
    for tname, tinfo in targets.items():
        metric = tinfo["metric"]
        for N in N_thermos:
            scores = {}
            for Qc in Q_centers:
                vals = []
                for r in results:
                    if r["target"] == tname and r["N"] == N and r["Q_center"] == Qc:
                        if metric == "crossings":
                            vals.append(r.get("crossings", 0))
                        else:
                            vals.append(r.get("tau", 1e6))
                if vals:
                    scores[Qc] = float(np.mean(vals))
            
            if metric == "crossings":
                best_Qc = max(scores, key=scores.get)
            else:
                best_Qc = min(scores, key=scores.get)
            
            best_params[(tname, N)] = best_Qc
            print(f"  {tname:20s} N={N}: best Q_center={best_Qc:>6.1f} "
                  f"score={scores[best_Qc]:.1f}")
            # Print full sweep
            for Qc in Q_centers:
                if Qc in scores:
                    marker = " <--" if Qc == best_Qc else ""
                    print(f"    Qc={Qc:>6.1f}: {scores[Qc]:>10.1f}{marker}")
    
    # Final evaluation with best Q_center
    print("\n" + "=" * 80)
    print("FINAL: LogOsc-tuned")
    print("=" * 80)
    
    n_seeds_final = 10
    n_steps_final = 400_000
    
    final_tasks = []
    for tname, tinfo in targets.items():
        for N in N_thermos:
            Qc = best_params[(tname, N)]
            for s in range(n_seeds_final):
                final_tasks.append((tname, tinfo, N, Qc, 5000 + s,
                                    n_steps_final, record_every))
    
    print(f"Final tasks: {len(final_tasks)}")
    with Pool(processes=min(10, cpu_count())) as pool:
        final_results = pool.map(run_single, final_tasks)
    
    final_results = [r for r in final_results if r is not None]
    
    # Aggregate
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    summary = {}
    for tname, tinfo in targets.items():
        metric = tinfo["metric"]
        summary[tname] = {}
        for N in N_thermos:
            Qc = best_params[(tname, N)]
            vals = []
            for r in final_results:
                if r["target"] == tname and r["N"] == N:
                    if metric == "crossings":
                        vals.append(r.get("crossings", 0))
                    else:
                        vals.append(r.get("tau", 1e6))
            if vals:
                mean_v = float(np.mean(vals))
                std_v = float(np.std(vals))
                summary[tname][f"LogOsc-{N}-tuned"] = {
                    "mean": mean_v, "std": std_v, "Q_center": Qc,
                }
                print(f"  {tname:20s} LogOsc-{N}-tuned (Qc={Qc:>6.1f}): "
                      f"{mean_v:.1f} +/- {std_v:.1f}")
    
    # Compare with v1 results
    v1_path = os.path.join(OUT_DIR, "results.json")
    if os.path.exists(v1_path):
        with open(v1_path) as f:
            v1 = json.load(f)
        print("\n  Comparison with v1 (auto Q range):")
        for tname in targets:
            v1_data = v1.get("summary", {}).get(tname, {})
            for N in N_thermos:
                old_key = f"LogOsc-{N}"
                new_key = f"LogOsc-{N}-tuned"
                if old_key in v1_data and tname in summary and new_key in summary[tname]:
                    metric = targets[tname]["metric"]
                    if metric == "crossings":
                        old_v = v1_data[old_key]["mode_crossings_mean"]
                    else:
                        old_v = v1_data[old_key]["tau_mean"]
                    new_v = summary[tname][new_key]["mean"]
                    if metric == "crossings":
                        ratio = new_v / max(old_v, 0.1)
                        better = "BETTER" if new_v > old_v else "worse"
                    else:
                        ratio = old_v / max(new_v, 0.1)
                        better = "BETTER" if new_v < old_v else "worse"
                    print(f"    {tname:20s} {old_key}: {old_v:.1f} -> {new_key}: "
                          f"{new_v:.1f} ({ratio:.1f}x {better})")
    
    # Save
    out = {"summary": summary, "best_params": {f"{k[0]}|N={k[1]}": v
           for k, v in best_params.items()}, "elapsed": time.time() - t0}
    with open(os.path.join(OUT_DIR, "results_v2.json"), "w") as f:
        json.dump(out, f, indent=2)
    
    print(f"\nTotal: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
