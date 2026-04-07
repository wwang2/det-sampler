"""q-optimization-035: numerical Q optimization and head-to-head NHC comparison.

Part 1: Optimize Q values freely on 5D anisotropic Gaussian; compare to log-uniform.
Part 2: Compare our parallel multi-scale log-osc vs NHC(M=N) at equal thermo count
        on 5D Gaussian (tau_int) and 2D Gaussian Mixture (mode crossings).
"""
import json
import os
import sys
import time
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from research.eval.integrators import ThermostatState
from research.eval.potentials import GaussianMixture2D


# ---------------------------------------------------------------------------
# Potentials
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Multi-scale parallel log-osc thermostat (vectorized BAOAB)
# ---------------------------------------------------------------------------
def g_func(xi):
    return 2.0 * xi / (1.0 + xi * xi)


def simulate_multiscale(potential, Qs, dt, n_steps, kT=1.0, mass=1.0, seed=0,
                        record_every=1):
    """BAOAB-like splitting for parallel multi-scale log-osc thermostats.

    dp/dt = -grad U - (sum g(xi_i)) * p
    dxi_i/dt = (K - dim*kT)/Q_i
    """
    rng = np.random.default_rng(seed)
    dim = potential.dim
    Qs = np.asarray(Qs, dtype=float)
    N = len(Qs)
    q = rng.normal(0, 1.0, size=dim) / np.sqrt(np.maximum(potential.kappas, 1e-6)) \
        if hasattr(potential, "kappas") else rng.normal(0, 1.0, size=dim)
    p = rng.normal(0, np.sqrt(mass * kT), size=dim)
    xi = np.zeros(N)

    half = 0.5 * dt
    grad_U = potential.gradient(q)
    n_rec = n_steps // record_every
    qs_rec = np.empty((n_rec, dim))
    rec_i = 0

    for step in range(n_steps):
        # Half-step thermostats
        K = float(np.sum(p * p)) / mass
        drive = (K - dim * kT) / Qs
        xi = xi + half * drive

        # Half-step momenta: friction + force
        gtot = float(np.sum(g_func(xi)))
        p = p * np.exp(-gtot * half)
        p = p - half * grad_U

        # Full-step positions
        q = q + dt * p / mass

        # Recompute force
        grad_U = potential.gradient(q)

        # Half-step momenta
        p = p - half * grad_U
        gtot = float(np.sum(g_func(xi)))
        p = p * np.exp(-gtot * half)

        # Half-step thermostats
        K = float(np.sum(p * p)) / mass
        drive = (K - dim * kT) / Qs
        xi = xi + half * drive

        if (step + 1) % record_every == 0 and rec_i < n_rec:
            qs_rec[rec_i] = q
            rec_i += 1

        if not np.isfinite(p).all() or not np.isfinite(q).all():
            qs_rec[rec_i:] = q
            break

    return qs_rec[:rec_i]


# ---------------------------------------------------------------------------
# NHC(M) baseline
# ---------------------------------------------------------------------------
def simulate_nhc(potential, Qs, dt, n_steps, kT=1.0, mass=1.0, seed=0,
                 record_every=1):
    """Nose-Hoover chain length M with BAOAB-like splitting.

    dp/dt = -grad U - xi_1 * p
    dxi_1/dt = (K - dim*kT)/Q_1 - xi_2 * xi_1
    dxi_i/dt = (Q_{i-1} xi_{i-1}^2 - kT)/Q_i - xi_{i+1} xi_i  (i=2..M-1)
    dxi_M/dt = (Q_{M-1} xi_{M-1}^2 - kT)/Q_M
    """
    rng = np.random.default_rng(seed)
    dim = potential.dim
    Qs = np.asarray(Qs, dtype=float)
    M = len(Qs)
    q = rng.normal(0, 1.0, size=dim) / np.sqrt(np.maximum(potential.kappas, 1e-6)) \
        if hasattr(potential, "kappas") else rng.normal(0, 1.0, size=dim)
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
        # Half-step xi
        xi = xi + half * chain_dxi(p, xi)

        # Half-step p: friction (xi_1) and force
        p = p * np.exp(-xi[0] * half)
        p = p - half * grad_U

        # Full-step q
        q = q + dt * p / mass
        grad_U = potential.gradient(q)

        # Half-step p
        p = p - half * grad_U
        p = p * np.exp(-xi[0] * half)

        # Half-step xi
        xi = xi + half * chain_dxi(p, xi)

        if (step + 1) % record_every == 0 and rec_i < n_rec:
            qs_rec[rec_i] = q
            rec_i += 1

        if not np.isfinite(p).all() or not np.isfinite(q).all():
            qs_rec[rec_i:] = q
            break

    return qs_rec[:rec_i]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def autocorr_time(x, c=5.0, max_lag=None):
    """Integrated autocorrelation time of 1D series x."""
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    if n < 16 or np.std(x) < 1e-12:
        return float(n)
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f))[:n].real
    acf /= acf[0]
    if max_lag is None:
        max_lag = n // 4
    tau = 1.0
    for k in range(1, max_lag):
        tau += 2.0 * acf[k]
        if k >= c * tau:
            break
    return float(max(tau, 1.0))


def tau_q2_mean(traj):
    """Mean over dimensions of tau_int for q^2."""
    if len(traj) < 64:
        return 1e6
    taus = []
    for d in range(traj.shape[1]):
        x = traj[:, d] ** 2
        if not np.isfinite(x).all():
            return 1e6
        taus.append(autocorr_time(x))
    return float(np.mean(taus))


def count_mode_crossings(traj, gmm):
    """Count transitions between nearest-mode assignments."""
    if len(traj) == 0:
        return 0
    centers = gmm.centers
    d2 = np.sum((traj[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    assign = np.argmin(d2, axis=1)
    return int(np.sum(assign[1:] != assign[:-1]))


# ---------------------------------------------------------------------------
# Part 1: Q optimization
# ---------------------------------------------------------------------------
def optimize_Qs(potential, N, n_force_evals, n_seeds, kT, kappa_max,
                n_inits=20, rng_seed=0, refine_seeds=3, refine_steps=None):
    dt = 0.05 / np.sqrt(kappa_max)
    n_steps = int(n_force_evals)
    log_lo = np.log(1.0 / np.sqrt(kappa_max) * 0.1)
    log_hi = np.log(1.0 / np.sqrt(potential.kappas.min()) * 10.0)

    def eval_logQ(logQ):
        Qs = np.exp(np.clip(logQ, log_lo - 2, log_hi + 2))
        taus = []
        for s in range(n_seeds):
            traj = simulate_multiscale(potential, Qs, dt, n_steps, kT=kT,
                                       seed=1000 * s + 7, record_every=2)
            taus.append(tau_q2_mean(traj))
        return float(np.mean(taus))

    rng = np.random.default_rng(rng_seed)
    best = (np.inf, None)
    history = []
    for init_i in range(n_inits):
        logQ0 = np.sort(rng.uniform(log_lo, log_hi, size=N))
        try:
            res = minimize(eval_logQ, logQ0, method="Nelder-Mead",
                           options={"xatol": 0.05, "fatol": 0.5,
                                    "maxiter": 60, "maxfev": 60})
            val = float(res.fun)
            logQ_opt = res.x
        except Exception as e:
            val = float("inf")
            logQ_opt = logQ0
        history.append({"init": logQ0.tolist(),
                        "opt": logQ_opt.tolist(), "tau": val})
        if val < best[0]:
            best = (val, np.exp(logQ_opt))
    return {"best_tau": best[0], "best_Qs": best[1].tolist(),
            "history": history, "log_lo": log_lo, "log_hi": log_hi}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def part1():
    print("=" * 60)
    print("PART 1: Q optimization")
    print("=" * 60)
    out = {}
    for kappa_ratio in [10.0, 100.0]:
        kappas = np.array([kappa_ratio ** (i / 4.0) for i in range(5)])
        pot = AnisotropicGaussian(kappas)
        for N in [3, 5]:
            key = f"kr{int(kappa_ratio)}_N{N}"
            print(f"\n[{key}] kappas={kappas}")
            t0 = time.time()
            # Fast optimization: shorter sims, 1 seed during search
            res = optimize_Qs(pot, N=N, n_force_evals=40_000, n_seeds=1,
                              kT=1.0, kappa_max=kappa_ratio, n_inits=8,
                              rng_seed=42 + N)
            # Refine top result with longer sims & more seeds
            best_Qs = np.array(res["best_Qs"])
            dt_r = 0.05 / np.sqrt(kappa_ratio)
            taus_refine = []
            for s in range(3):
                tr = simulate_multiscale(pot, best_Qs, dt_r, 200_000,
                                         seed=9000 + s, record_every=2)
                taus_refine.append(tau_q2_mean(tr))
            res["best_tau"] = float(np.mean(taus_refine))
            # log-uniform reference
            Qmin = 1.0 / np.sqrt(kappa_ratio)
            Qmax = 1.0 / np.sqrt(1.0)
            Qs_logu = np.exp(np.linspace(np.log(Qmin), np.log(Qmax), N))
            dt = 0.05 / np.sqrt(kappa_ratio)
            taus_logu = []
            for s in range(3):
                tr = simulate_multiscale(pot, Qs_logu, dt, 200_000,
                                         seed=2000 + s, record_every=2)
                taus_logu.append(tau_q2_mean(tr))
            tau_logu = float(np.mean(taus_logu))
            print(f"  best_tau={res['best_tau']:.2f}  Qs={res['best_Qs']}")
            print(f"  logu tau={tau_logu:.2f}  Qs={Qs_logu.tolist()}")
            print(f"  elapsed {time.time()-t0:.1f}s")
            out[key] = {
                "kappas": kappas.tolist(),
                "kappa_ratio": kappa_ratio,
                "N": N,
                "best_tau": res["best_tau"],
                "best_Qs": res["best_Qs"],
                "loguniform_Qs": Qs_logu.tolist(),
                "loguniform_tau": tau_logu,
            }
    return out


def part2():
    print("=" * 60)
    print("PART 2: Head-to-head vs NHC")
    print("=" * 60)
    out = {"gaussian": {}, "mixture": {}}

    n_seeds = 10
    n_steps_g = 400_000
    n_steps_m = 200_000

    # ---- Gaussian: tau_int ----
    for kappa_ratio in [10.0, 100.0]:
        kappas = np.array([kappa_ratio ** (i / 4.0) for i in range(5)])
        pot = AnisotropicGaussian(kappas)
        dt = 0.05 / np.sqrt(kappa_ratio)
        Qmin = 1.0 / np.sqrt(kappa_ratio)
        Qmax = 1.0
        results = {}
        for N in [3, 5]:
            Qs_par = np.exp(np.linspace(np.log(Qmin), np.log(Qmax), N))
            Qs_nhc = np.ones(N) * 1.0  # Q_ref = kT/omega^2 = 1
            tau_par = []
            tau_nhc = []
            for s in range(n_seeds):
                tr = simulate_multiscale(pot, Qs_par, dt, n_steps_g,
                                         seed=3000 + s, record_every=4)
                tau_par.append(tau_q2_mean(tr))
                tr = simulate_nhc(pot, Qs_nhc, dt, n_steps_g,
                                  seed=3000 + s, record_every=4)
                tau_nhc.append(tau_q2_mean(tr))
            results[f"parallel_N{N}"] = {
                "mean": float(np.mean(tau_par)),
                "std": float(np.std(tau_par)),
                "Qs": Qs_par.tolist(),
            }
            results[f"nhc_M{N}"] = {
                "mean": float(np.mean(tau_nhc)),
                "std": float(np.std(tau_nhc)),
                "Qs": Qs_nhc.tolist(),
            }
            print(f"  kr={kappa_ratio} N={N}: par={np.mean(tau_par):.1f}+-{np.std(tau_par):.1f}  "
                  f"nhc={np.mean(tau_nhc):.1f}+-{np.std(tau_nhc):.1f}")
        out["gaussian"][f"kr{int(kappa_ratio)}"] = results

    # ---- Gaussian Mixture: mode crossings ----
    gmm = GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5)
    dt_m = 0.02
    Qmin_m = 0.1
    Qmax_m = 10.0
    results_m = {}
    for N in [3, 5]:
        Qs_par = np.exp(np.linspace(np.log(Qmin_m), np.log(Qmax_m), N))
        Qs_nhc = np.ones(N) * 1.0
        cr_par = []
        cr_nhc = []
        for s in range(n_seeds):
            tr = simulate_multiscale(gmm, Qs_par, dt_m, n_steps_m,
                                     seed=4000 + s, record_every=4)
            cr_par.append(count_mode_crossings(tr, gmm))
            tr = simulate_nhc(gmm, Qs_nhc, dt_m, n_steps_m,
                              seed=4000 + s, record_every=4)
            cr_nhc.append(count_mode_crossings(tr, gmm))
        results_m[f"parallel_N{N}"] = {
            "mean": float(np.mean(cr_par)),
            "std": float(np.std(cr_par)),
            "Qs": Qs_par.tolist(),
        }
        results_m[f"nhc_M{N}"] = {
            "mean": float(np.mean(cr_nhc)),
            "std": float(np.std(cr_nhc)),
            "Qs": Qs_nhc.tolist(),
        }
        print(f"  GMM N={N}: par={np.mean(cr_par):.1f}+-{np.std(cr_par):.1f}  "
              f"nhc={np.mean(cr_nhc):.1f}+-{np.std(cr_nhc):.1f}")
    out["mixture"] = results_m
    return out


def main():
    out = {}
    t0 = time.time()
    out["part1"] = part1()
    out["part2"] = part2()
    out["elapsed_sec"] = time.time() - t0

    # Headline metric: NHC best vs our best at kr=100
    g100 = out["part2"]["gaussian"]["kr100"]
    nhc_best = min(g100["nhc_M3"]["mean"], g100["nhc_M5"]["mean"])
    par_best = min(g100["parallel_N3"]["mean"], g100["parallel_N5"]["mean"])
    out["headline_metric_nhc_over_ours"] = nhc_best / par_best
    print(f"\nHEADLINE: NHC_best/ours_best at kr=100 = {nhc_best/par_best:.3f}")

    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"Saved {out_path}  total {out['elapsed_sec']:.1f}s")
    return out


if __name__ == "__main__":
    main()
