"""esh-comparison-043: Head-to-head comparison of multi-scale log-osc thermostat vs ESH dynamics.

ESH (Energy Sampling Hamiltonian, Ver Steeg & Galstyan 2021, arXiv:2111.02434)
is a deterministic sampler that moves on the energy surface with unit-speed dynamics.
It samples the MICROCANONICAL ensemble at fixed total energy E = U(q) + |p|.
To get canonical samples, one must marginalize over energy levels or resample energy.

Our multi-scale log-osc thermostat samples the CANONICAL ensemble by construction.

This experiment compares:
1. Sampling accuracy (KL divergence, variance accuracy)
2. Mode-hopping ability (barrier crossings, mode visits)
3. Stability envelope (divergence rate vs step size)
4. Trajectory character (visual comparison)
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.eval.potentials import (
    DoubleWell2D, GaussianMixture2D,
)

ORBIT_DIR = Path(__file__).resolve().parent
FIG_DIR = ORBIT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


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


# ============================================================================
# ESH Dynamics (from Ver Steeg & Galstyan 2021)
# ============================================================================
def simulate_esh(potential, dt, n_steps, kT=1.0, seed=0, record_every=1,
                 energy_resample_interval=0, clamp_eps=1e-8):
    """ESH dynamics with velocity-Verlet-like integrator.

    State: (q, v, s) where v = p/|p| is unit direction, s = log|p| for stability.

    Equations:
        dq/dt = v
        ds/dt = -grad_U . v
        dv/dt = (-grad_U + (grad_U . v) v) * exp(-s)

    For canonical sampling, we optionally resample s (= log|p|) from its
    marginal distribution p(s) ~ exp((d-1)*s - exp(s)) every N steps,
    which is the log of a Gamma(d, 1) variate.

    Args:
        energy_resample_interval: if >0, resample |p| from Gamma(d,kT) every
            this many steps. 0 = pure microcanonical ESH (no resampling).
    """
    rng = np.random.default_rng(seed)
    dim = potential.dim

    # Initialize
    q = rng.normal(0, 1.0, size=dim)
    if hasattr(potential, 'kappas'):
        q = q / np.sqrt(np.maximum(potential.kappas, 1e-6))

    # Initialize v as random unit vector
    v = rng.normal(0, 1, size=dim)
    v = v / np.linalg.norm(v)

    # Initialize |p| from Gamma(d, kT) — the canonical marginal for ESH
    p_norm = rng.gamma(dim, kT)
    s = np.log(max(p_norm, clamp_eps))

    grad_U = potential.gradient(q)
    force_evals = 1

    n_rec = n_steps // record_every
    qs_rec = np.empty((n_rec, dim))
    rec_i = 0

    half = 0.5 * dt

    for step in range(n_steps):
        # --- Leapfrog-style integration ---
        # Half-step s
        gv = float(np.dot(grad_U, v))
        s = s - half * gv

        # Half-step v (geodesic on sphere)
        exp_neg_s = np.exp(-np.clip(s, -20, 20))
        a = (-grad_U + gv * v) * exp_neg_s
        a_norm = np.linalg.norm(a)
        if a_norm > 1e-14:
            # Rodrigues-like rotation: exact geodesic step
            theta = a_norm * half
            a_hat = a / a_norm
            if theta < 1e-6:
                v = v + half * a
            else:
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                v = v * cos_t + a_hat * sin_t
            # Re-project to unit sphere (numerical drift)
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-14:
                v = v / v_norm

        # Full-step q
        q = q + dt * v

        # Recompute gradient
        grad_U = potential.gradient(q)
        force_evals += 1

        # Half-step v
        gv = float(np.dot(grad_U, v))
        exp_neg_s = np.exp(-np.clip(s, -20, 20))
        a = (-grad_U + gv * v) * exp_neg_s
        a_norm = np.linalg.norm(a)
        if a_norm > 1e-14:
            theta = a_norm * half
            a_hat = a / a_norm
            if theta < 1e-6:
                v = v + half * a
            else:
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                v = v * cos_t + a_hat * sin_t
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-14:
                v = v / v_norm

        # Half-step s
        gv = float(np.dot(grad_U, v))
        s = s - half * gv

        # Energy resampling (for canonical ensemble)
        if energy_resample_interval > 0 and (step + 1) % energy_resample_interval == 0:
            p_norm_new = rng.gamma(dim, kT)
            s = np.log(max(p_norm_new, clamp_eps))

        # Record
        if (step + 1) % record_every == 0 and rec_i < n_rec:
            qs_rec[rec_i] = q
            rec_i += 1

        # Divergence check
        if not np.isfinite(q).all() or not np.isfinite(v).all() or not np.isfinite(s):
            break

    return qs_rec[:rec_i]


# ============================================================================
# Multi-scale log-osc thermostat (from q-optimization-035)
# ============================================================================
def g_func(xi):
    return 2.0 * xi / (1.0 + xi * xi)


def simulate_multiscale(potential, Qs, dt, n_steps, kT=1.0, mass=1.0, seed=0,
                        record_every=1):
    """BAOAB-like splitting for parallel multi-scale log-osc thermostats."""
    rng = np.random.default_rng(seed)
    dim = potential.dim
    Qs = np.asarray(Qs, dtype=float)
    N = len(Qs)
    q = rng.normal(0, 1.0, size=dim)
    if hasattr(potential, 'kappas'):
        q = q / np.sqrt(np.maximum(potential.kappas, 1e-6))
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
            qs_rec[rec_i:] = q
            break

    return qs_rec[:rec_i]


# ============================================================================
# NHC baseline
# ============================================================================
def simulate_nhc(potential, Qs, dt, n_steps, kT=1.0, mass=1.0, seed=0,
                 record_every=1):
    """Nose-Hoover chain with BAOAB-like splitting."""
    rng = np.random.default_rng(seed)
    dim = potential.dim
    Qs = np.asarray(Qs, dtype=float)
    M = len(Qs)
    q = rng.normal(0, 1.0, size=dim)
    if hasattr(potential, 'kappas'):
        q = q / np.sqrt(np.maximum(potential.kappas, 1e-6))
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
            qs_rec[rec_i:] = q
            break

    return qs_rec[:rec_i]


# ============================================================================
# Diagnostics
# ============================================================================
def autocorr_time(x, c=5.0):
    """Integrated autocorrelation time via FFT."""
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


def count_barrier_crossings(traj_x):
    """Count sign changes in x coordinate (double well crossings)."""
    signs = np.sign(traj_x)
    return int(np.sum(signs[1:] != signs[:-1]))


def count_mode_crossings(traj, gmm):
    """Count transitions between nearest-mode assignments."""
    if len(traj) == 0:
        return 0
    centers = gmm.centers
    d2 = np.sum((traj[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    assign = np.argmin(d2, axis=1)
    return int(np.sum(assign[1:] != assign[:-1]))


def modes_visited(traj, gmm, radius=1.5):
    """Fraction of modes visited (within radius*sigma of center)."""
    if len(traj) == 0:
        return 0.0
    centers = gmm.centers
    d2 = np.sum((traj[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    threshold = (radius * gmm.sigma) ** 2
    visited = np.any(d2 < threshold, axis=0)
    return float(np.mean(visited))


def kl_x_marginal_double_well(traj, kT=1.0, n_bins=200):
    """KL divergence of x-marginal vs analytical for double well."""
    x = traj[:, 0]
    x = x[np.isfinite(x)]
    if len(x) < 100:
        return 10.0

    x_range = (-3.0, 3.0)
    hist, edges = np.histogram(x, bins=n_bins, range=x_range, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Analytical: P(x) ~ exp(-(x^2-1)^2 / kT)
    log_p = -(centers**2 - 1)**2 / kT
    log_p -= np.max(log_p)  # numerical stability
    p_true = np.exp(log_p)
    p_true /= np.sum(p_true) * (centers[1] - centers[0])

    # KL(empirical || true)
    mask = (hist > 0) & (p_true > 0)
    if mask.sum() < 10:
        return 10.0
    kl = np.sum(hist[mask] * np.log(hist[mask] / p_true[mask])) * (centers[1] - centers[0])
    return float(max(kl, 0.0))


def variance_accuracy(traj, kappas, kT=1.0):
    """Max relative error of per-dimension variance vs analytical."""
    if len(traj) < 64:
        return 10.0
    expected = kT / kappas
    observed = np.var(traj, axis=0)
    rel_err = np.abs(observed - expected) / expected
    return float(np.max(rel_err))


# ============================================================================
# Q schedule for our thermostat (corrected F2 from q-optimization-035)
# ============================================================================
def get_multiscale_Qs(kappas, N=5):
    """Corrected Q range: Q_min = 2.34*omega_max^(-1.55), Q_max = 2.34*omega_min^(-1.55)."""
    omegas = np.sqrt(kappas)
    Q_min = 2.34 * np.max(omegas)**(-1.55)
    Q_max = 2.34 * np.min(omegas)**(-1.55)
    return np.exp(np.linspace(np.log(Q_min), np.log(Q_max), N))


def get_Qs_for_potential(potential, N=5):
    """Return appropriate Q values for a given potential."""
    if hasattr(potential, 'kappas'):
        return get_multiscale_Qs(potential.kappas, N)
    elif isinstance(potential, DoubleWell2D):
        # Effective curvatures: ~4 for x near minima, ~0.5*2=1 for y
        return np.exp(np.linspace(np.log(0.5), np.log(5.0), N))
    elif isinstance(potential, GaussianMixture2D):
        # Within-mode curvature ~ 1/sigma^2, between-mode ~ soft
        return np.exp(np.linspace(np.log(0.1), np.log(10.0), N))
    else:
        return np.ones(N)


# ============================================================================
# Benchmark 1: 2D Double Well
# ============================================================================
def benchmark_double_well(n_force_evals=400_000, n_seeds=10, kT=1.0):
    print("\n=== Benchmark: 2D Double Well ===")
    pot = DoubleWell2D()
    dt = 0.02
    Qs = get_Qs_for_potential(pot, N=5)
    Qs_nhc = np.ones(3) * 1.0

    results = {"ours": [], "esh_micro": [], "esh_resample": [], "nhc": []}

    for s in range(n_seeds):
        seed = 1000 + s
        # Our method
        traj = simulate_multiscale(pot, Qs, dt, n_force_evals, kT=kT, seed=seed, record_every=4)
        crossings = count_barrier_crossings(traj[:, 0])
        kl = kl_x_marginal_double_well(traj, kT)
        results["ours"].append({"crossings": crossings, "kl": kl, "len": len(traj)})

        # ESH microcanonical (no resampling)
        traj_esh = simulate_esh(pot, dt, n_force_evals, kT=kT, seed=seed,
                                record_every=4, energy_resample_interval=0)
        crossings_e = count_barrier_crossings(traj_esh[:, 0]) if len(traj_esh) > 0 else 0
        kl_e = kl_x_marginal_double_well(traj_esh, kT) if len(traj_esh) > 0 else 10.0
        results["esh_micro"].append({"crossings": crossings_e, "kl": kl_e, "len": len(traj_esh)})

        # ESH with energy resampling every 100 steps
        traj_esh_r = simulate_esh(pot, dt, n_force_evals, kT=kT, seed=seed,
                                  record_every=4, energy_resample_interval=100)
        crossings_er = count_barrier_crossings(traj_esh_r[:, 0]) if len(traj_esh_r) > 0 else 0
        kl_er = kl_x_marginal_double_well(traj_esh_r, kT) if len(traj_esh_r) > 0 else 10.0
        results["esh_resample"].append({"crossings": crossings_er, "kl": kl_er, "len": len(traj_esh_r)})

        # NHC baseline
        traj_nhc = simulate_nhc(pot, Qs_nhc, dt, n_force_evals, kT=kT, seed=seed, record_every=4)
        crossings_nhc = count_barrier_crossings(traj_nhc[:, 0])
        kl_nhc = kl_x_marginal_double_well(traj_nhc, kT)
        results["nhc"].append({"crossings": crossings_nhc, "kl": kl_nhc, "len": len(traj_nhc)})

    summary = {}
    for method, data in results.items():
        cr = [d["crossings"] for d in data]
        kls = [d["kl"] for d in data]
        lens = [d["len"] for d in data]
        summary[method] = {
            "crossings_mean": float(np.mean(cr)),
            "crossings_std": float(np.std(cr)),
            "kl_mean": float(np.mean(kls)),
            "kl_std": float(np.std(kls)),
            "avg_len": float(np.mean(lens)),
        }
        print(f"  {method:15s}: crossings={np.mean(cr):.0f}+-{np.std(cr):.0f}, "
              f"KL={np.mean(kls):.4f}+-{np.std(kls):.4f}")

    return summary, results


# ============================================================================
# Benchmark 2: 2D Gaussian Mixture
# ============================================================================
def benchmark_gmm(n_force_evals=400_000, n_seeds=10, kT=1.0):
    print("\n=== Benchmark: 2D Gaussian Mixture (5 modes) ===")
    gmm = GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5)
    dt = 0.02
    Qs = get_Qs_for_potential(gmm, N=5)
    Qs_nhc = np.ones(3) * 1.0

    results = {"ours": [], "esh_micro": [], "esh_resample": [], "nhc": []}

    for s in range(n_seeds):
        seed = 2000 + s
        # Our method
        traj = simulate_multiscale(gmm, Qs, dt, n_force_evals, kT=kT, seed=seed, record_every=4)
        mc = count_mode_crossings(traj, gmm)
        mv = modes_visited(traj, gmm)
        results["ours"].append({"mode_crossings": mc, "modes_visited": mv, "len": len(traj)})

        # ESH micro
        traj_e = simulate_esh(gmm, dt, n_force_evals, kT=kT, seed=seed,
                              record_every=4, energy_resample_interval=0)
        mc_e = count_mode_crossings(traj_e, gmm) if len(traj_e) > 0 else 0
        mv_e = modes_visited(traj_e, gmm) if len(traj_e) > 0 else 0.0
        results["esh_micro"].append({"mode_crossings": mc_e, "modes_visited": mv_e, "len": len(traj_e)})

        # ESH resample
        traj_er = simulate_esh(gmm, dt, n_force_evals, kT=kT, seed=seed,
                               record_every=4, energy_resample_interval=100)
        mc_er = count_mode_crossings(traj_er, gmm) if len(traj_er) > 0 else 0
        mv_er = modes_visited(traj_er, gmm) if len(traj_er) > 0 else 0.0
        results["esh_resample"].append({"mode_crossings": mc_er, "modes_visited": mv_er, "len": len(traj_er)})

        # NHC
        traj_nhc = simulate_nhc(gmm, Qs_nhc, dt, n_force_evals, kT=kT, seed=seed, record_every=4)
        mc_nhc = count_mode_crossings(traj_nhc, gmm)
        mv_nhc = modes_visited(traj_nhc, gmm)
        results["nhc"].append({"mode_crossings": mc_nhc, "modes_visited": mv_nhc, "len": len(traj_nhc)})

    summary = {}
    for method, data in results.items():
        mc = [d["mode_crossings"] for d in data]
        mv = [d["modes_visited"] for d in data]
        summary[method] = {
            "mode_crossings_mean": float(np.mean(mc)),
            "mode_crossings_std": float(np.std(mc)),
            "modes_visited_mean": float(np.mean(mv)),
            "modes_visited_std": float(np.std(mv)),
        }
        print(f"  {method:15s}: mode_crossings={np.mean(mc):.0f}+-{np.std(mc):.0f}, "
              f"modes_visited={np.mean(mv):.2f}+-{np.std(mv):.2f}")

    return summary, results


# ============================================================================
# Benchmark 3 & 4: Anisotropic Gaussian (5D and 10D)
# ============================================================================
def benchmark_anisotropic(dim=5, kappa_ratio=100.0, n_force_evals=400_000,
                          n_seeds=10, kT=1.0):
    label = f"{dim}D Anisotropic Gaussian (kappa_ratio={kappa_ratio})"
    print(f"\n=== Benchmark: {label} ===")
    kappas = np.array([kappa_ratio ** (i / (dim - 1)) for i in range(dim)])
    pot = AnisotropicGaussian(kappas)
    dt = 0.05 / np.sqrt(kappa_ratio)
    Qs = get_multiscale_Qs(kappas, N=5)
    Qs_nhc = np.ones(3) * 1.0

    results = {"ours": [], "esh_micro": [], "esh_resample": [], "nhc": []}

    for s in range(n_seeds):
        seed = 3000 + s + dim * 100
        # Our method
        traj = simulate_multiscale(pot, Qs, dt, n_force_evals, kT=kT, seed=seed, record_every=4)
        tau = tau_q2_mean(traj)
        vacc = variance_accuracy(traj, kappas, kT)
        results["ours"].append({"tau": tau, "var_acc": vacc, "len": len(traj)})

        # ESH micro
        traj_e = simulate_esh(pot, dt, n_force_evals, kT=kT, seed=seed,
                              record_every=4, energy_resample_interval=0)
        tau_e = tau_q2_mean(traj_e) if len(traj_e) > 64 else 1e6
        vacc_e = variance_accuracy(traj_e, kappas, kT) if len(traj_e) > 64 else 10.0
        results["esh_micro"].append({"tau": tau_e, "var_acc": vacc_e, "len": len(traj_e)})

        # ESH resample
        traj_er = simulate_esh(pot, dt, n_force_evals, kT=kT, seed=seed,
                               record_every=4, energy_resample_interval=100)
        tau_er = tau_q2_mean(traj_er) if len(traj_er) > 64 else 1e6
        vacc_er = variance_accuracy(traj_er, kappas, kT) if len(traj_er) > 64 else 10.0
        results["esh_resample"].append({"tau": tau_er, "var_acc": vacc_er, "len": len(traj_er)})

        # NHC
        traj_nhc = simulate_nhc(pot, Qs_nhc, dt, n_force_evals, kT=kT, seed=seed, record_every=4)
        tau_nhc = tau_q2_mean(traj_nhc)
        vacc_nhc = variance_accuracy(traj_nhc, kappas, kT)
        results["nhc"].append({"tau": tau_nhc, "var_acc": vacc_nhc, "len": len(traj_nhc)})

    summary = {}
    for method, data in results.items():
        taus = [d["tau"] for d in data]
        vaccs = [d["var_acc"] for d in data]
        summary[method] = {
            "tau_mean": float(np.mean(taus)),
            "tau_std": float(np.std(taus)),
            "var_acc_mean": float(np.mean(vaccs)),
            "var_acc_std": float(np.std(vaccs)),
        }
        print(f"  {method:15s}: tau={np.mean(taus):.1f}+-{np.std(taus):.1f}, "
              f"var_err={np.mean(vaccs):.4f}+-{np.std(vaccs):.4f}")

    return summary, results


# ============================================================================
# Stability sweep
# ============================================================================
def stability_sweep(n_force_evals=200_000, n_seeds=10, kT=1.0):
    print("\n=== Stability Sweep ===")
    dts = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    # Use 5D anisotropic Gaussian (kappa_ratio=100)
    kappas = np.array([100.0 ** (i / 4) for i in range(5)])
    pot = AnisotropicGaussian(kappas)
    Qs = get_multiscale_Qs(kappas, N=5)
    Qs_nhc = np.ones(3) * 1.0

    results = {method: {str(dt): [] for dt in dts}
               for method in ["ours", "esh_micro", "esh_resample", "nhc"]}

    for dt in dts:
        print(f"  dt={dt}")
        for s in range(n_seeds):
            seed = 5000 + s

            # Our method
            traj = simulate_multiscale(pot, Qs, dt, n_force_evals, kT=kT,
                                       seed=seed, record_every=4)
            diverged = len(traj) < n_force_evals // 4 * 0.9 or (
                len(traj) > 0 and (np.abs(traj[-1]).max() > 1000 or not np.isfinite(traj[-1]).all()))
            tau = tau_q2_mean(traj) if not diverged and len(traj) > 64 else np.nan
            results["ours"][str(dt)].append({"tau": tau, "diverged": diverged})

            # ESH micro
            traj_e = simulate_esh(pot, dt, n_force_evals, kT=kT, seed=seed,
                                  record_every=4, energy_resample_interval=0)
            diverged_e = len(traj_e) < n_force_evals // 4 * 0.9 or (
                len(traj_e) > 0 and (np.abs(traj_e[-1]).max() > 1000 or not np.isfinite(traj_e[-1]).all()))
            tau_e = tau_q2_mean(traj_e) if not diverged_e and len(traj_e) > 64 else np.nan
            results["esh_micro"][str(dt)].append({"tau": tau_e, "diverged": diverged_e})

            # ESH resample
            traj_er = simulate_esh(pot, dt, n_force_evals, kT=kT, seed=seed,
                                   record_every=4, energy_resample_interval=100)
            diverged_er = len(traj_er) < n_force_evals // 4 * 0.9 or (
                len(traj_er) > 0 and (np.abs(traj_er[-1]).max() > 1000 or not np.isfinite(traj_er[-1]).all()))
            tau_er = tau_q2_mean(traj_er) if not diverged_er and len(traj_er) > 64 else np.nan
            results["esh_resample"][str(dt)].append({"tau": tau_er, "diverged": diverged_er})

            # NHC
            traj_nhc = simulate_nhc(pot, Qs_nhc, dt, n_force_evals, kT=kT,
                                     seed=seed, record_every=4)
            diverged_nhc = len(traj_nhc) < n_force_evals // 4 * 0.9 or (
                len(traj_nhc) > 0 and (np.abs(traj_nhc[-1]).max() > 1000 or not np.isfinite(traj_nhc[-1]).all()))
            tau_nhc = tau_q2_mean(traj_nhc) if not diverged_nhc and len(traj_nhc) > 64 else np.nan
            results["nhc"][str(dt)].append({"tau": tau_nhc, "diverged": diverged_nhc})

    # Summarize
    summary = {}
    for method in results:
        summary[method] = {}
        for dt_str in results[method]:
            data = results[method][dt_str]
            divs = [d["diverged"] for d in data]
            taus = [d["tau"] for d in data if not np.isnan(d["tau"])]
            summary[method][dt_str] = {
                "divergence_rate": float(np.mean(divs)),
                "tau_mean": float(np.mean(taus)) if taus else float("nan"),
                "tau_std": float(np.std(taus)) if len(taus) > 1 else 0.0,
                "n_valid": len(taus),
            }
            print(f"    {method:15s} dt={dt_str}: div={np.mean(divs):.1%}, "
                  f"tau={np.mean(taus):.1f}" if taus else
                  f"    {method:15s} dt={dt_str}: div={np.mean(divs):.1%}, tau=N/A")

    return summary, results


# ============================================================================
# Plotting
# ============================================================================
def plot_stability_envelope(stability_summary):
    """Fig 2: tau_int vs dt with divergence rate annotation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dts = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    colors = {
        "ours": "#2ca02c",          # green (tab10 idx 2)
        "esh_micro": "#9467bd",     # purple (tab10 idx 4)
        "esh_resample": "#8c564b",  # brown (tab10 idx 5)
        "nhc": "#ff7f0e",           # orange
    }
    labels = {
        "ours": "Multi-scale log-osc (ours)",
        "esh_micro": "ESH (microcanonical)",
        "esh_resample": "ESH + energy resample",
        "nhc": "NHC (M=3)",
    }
    markers = {"ours": "o", "esh_micro": "s", "esh_resample": "D", "nhc": "^"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for method in ["ours", "esh_micro", "esh_resample", "nhc"]:
        tau_means = []
        tau_stds = []
        div_rates = []
        valid_dts = []
        for dt in dts:
            s = stability_summary[method][str(dt)]
            div_rates.append(s["divergence_rate"])
            if not np.isnan(s["tau_mean"]) and s["n_valid"] > 0:
                tau_means.append(s["tau_mean"])
                tau_stds.append(s["tau_std"])
                valid_dts.append(dt)

        # Panel (a): tau_int vs dt
        if valid_dts:
            ax1.errorbar(valid_dts, tau_means, yerr=tau_stds,
                         color=colors[method], marker=markers[method],
                         label=labels[method], capsize=3, linewidth=1.5,
                         markersize=6)

        # Panel (b): divergence rate vs dt
        ax2.plot(dts, div_rates, color=colors[method], marker=markers[method],
                 label=labels[method], linewidth=1.5, markersize=6)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Step size dt", fontsize=14)
    ax1.set_ylabel(r"$\tau_{\mathrm{int}}$ (autocorrelation time)", fontsize=14)
    ax1.set_title("(a) Mixing efficiency vs step size", fontsize=14)
    ax1.legend(fontsize=10, loc="upper left")
    ax1.tick_params(labelsize=12)
    ax1.grid(True, alpha=0.3)

    ax2.set_xscale("log")
    ax2.set_xlabel("Step size dt", fontsize=14)
    ax2.set_ylabel("Divergence rate", fontsize=14)
    ax2.set_title("(b) Stability: fraction of runs that diverge", fontsize=14)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=10, loc="upper left")
    ax2.tick_params(labelsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    fig.suptitle("Stability Envelope: 5D Anisotropic Gaussian ($\\kappa_{\\mathrm{ratio}}=100$)",
                 fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_stability_envelope.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig2_stability_envelope.png")


def plot_benchmark_table(dw_summary, gmm_summary, g5d_summary, g10d_summary):
    """Fig 1: Summary comparison table rendered as figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ["ours", "esh_micro", "esh_resample", "nhc"]
    method_labels = ["Log-osc\n(ours)", "ESH\n(micro)", "ESH\n(+resample)", "NHC\n(M=3)"]

    # Build data rows
    rows = []
    row_labels = []

    # DW: barrier crossings
    row_labels.append("DW: crossings\n(per 400k evals)")
    rows.append([dw_summary[m]["crossings_mean"] for m in methods])

    # DW: KL
    row_labels.append("DW: KL div\n(lower=better)")
    rows.append([dw_summary[m]["kl_mean"] for m in methods])

    # GMM: mode crossings
    row_labels.append("GMM: mode\ncrossings")
    rows.append([gmm_summary[m]["mode_crossings_mean"] for m in methods])

    # GMM: modes visited
    row_labels.append("GMM: frac\nmodes visited")
    rows.append([gmm_summary[m]["modes_visited_mean"] for m in methods])

    # 5D: tau
    row_labels.append("5D Gauss:\n" + r"$\tau_{\mathrm{int}}$")
    rows.append([g5d_summary[m]["tau_mean"] for m in methods])

    # 5D: var accuracy
    row_labels.append("5D Gauss:\nvar error")
    rows.append([g5d_summary[m]["var_acc_mean"] for m in methods])

    # 10D: tau
    row_labels.append("10D Gauss:\n" + r"$\tau_{\mathrm{int}}$")
    rows.append([g10d_summary[m]["tau_mean"] for m in methods])

    # 10D: var accuracy
    row_labels.append("10D Gauss:\nvar error")
    rows.append([g10d_summary[m]["var_acc_mean"] for m in methods])

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")

    cell_text = []
    for i, row in enumerate(rows):
        formatted = []
        # Determine best (depends on metric)
        is_lower_better = i in [1, 4, 5, 6, 7]  # KL, tau, var_error
        vals = [v for v in row if np.isfinite(v)]
        best_val = min(vals) if is_lower_better and vals else (max(vals) if vals else None)
        for v in row:
            if np.isnan(v) or v > 1e5:
                formatted.append("DIV")
            elif v > 100:
                formatted.append(f"{v:.0f}")
            elif v > 1:
                formatted.append(f"{v:.1f}")
            else:
                formatted.append(f"{v:.4f}")
        cell_text.append(formatted)

    colors_table = []
    for i, row in enumerate(rows):
        is_lower_better = i in [1, 4, 5, 6, 7]
        vals = [v for v in row if np.isfinite(v) and v < 1e5]
        if not vals:
            colors_table.append(["white"] * len(methods))
            continue
        best_val = min(vals) if is_lower_better else max(vals)
        row_colors = []
        for v in row:
            if np.isfinite(v) and v < 1e5 and abs(v - best_val) < 1e-10:
                row_colors.append("#d4edda")  # light green
            else:
                row_colors.append("white")
        colors_table.append(row_colors)

    table = ax.table(cellText=cell_text, rowLabels=row_labels,
                     colLabels=method_labels, cellColours=colors_table,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)

    # Style header
    for j in range(len(methods)):
        table[(0, j)].set_facecolor("#e8e8e8")
        table[(0, j)].set_text_props(fontweight="bold")

    ax.set_title("Head-to-Head Benchmark Comparison\n"
                 "(green = best per metric; 10 seeds, 400k force evals)",
                 fontsize=16, pad=20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_benchmark_table.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig1_benchmark_table.png")


def plot_trajectory_comparison(kT=1.0):
    """Fig 4: exemplar trajectories on 2D double well."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pot = DoubleWell2D()
    dt = 0.02
    n_steps = 50_000
    seed = 42
    Qs = get_Qs_for_potential(pot, N=5)

    traj_ours = simulate_multiscale(pot, Qs, dt, n_steps, kT=kT, seed=seed, record_every=1)
    traj_esh = simulate_esh(pot, dt, n_steps, kT=kT, seed=seed, record_every=1,
                            energy_resample_interval=100)
    traj_nhc = simulate_nhc(pot, np.ones(3), dt, n_steps, kT=kT, seed=seed, record_every=1)

    # Potential contours
    xg = np.linspace(-2.5, 2.5, 200)
    yg = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(xg, yg)
    Z = (X**2 - 1)**2 + 0.5 * Y**2

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    titles = ["(a) Multi-scale log-osc (ours)", "(b) ESH + energy resample", "(c) NHC (M=3)"]
    trajs = [traj_ours, traj_esh, traj_nhc]
    colors_traj = ["#2ca02c", "#8c564b", "#ff7f0e"]

    for ax, traj, title, col in zip(axes, trajs, titles, colors_traj):
        ax.contour(X, Y, Z, levels=np.arange(0, 5, 0.5), colors="gray", alpha=0.4, linewidths=0.5)
        ax.contourf(X, Y, Z, levels=np.arange(0, 5, 0.5), cmap="Greys", alpha=0.15)
        # Plot trajectory (subsample for clarity)
        n_plot = min(len(traj), 20000)
        step = max(1, len(traj) // n_plot)
        ax.plot(traj[::step, 0], traj[::step, 1], color=col, alpha=0.15, linewidth=0.3)
        ax.scatter(traj[::step*10, 0], traj[::step*10, 1], color=col, alpha=0.4, s=1)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-3, 3)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("x", fontsize=12)
        ax.set_aspect("equal")
        # Annotate crossings
        cx = count_barrier_crossings(traj[:, 0])
        ax.text(0.02, 0.98, f"{cx} crossings", transform=ax.transAxes,
                fontsize=10, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    axes[0].set_ylabel("y", fontsize=12)
    fig.suptitle("Trajectory Comparison on 2D Double Well (50k steps, dt=0.02)",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_trajectory_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig4_trajectory_comparison.png")


def plot_mode_hopping(kT=1.0):
    """Fig 3: mode occupation over time for GMM."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gmm = GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5)
    dt = 0.02
    n_steps = 100_000
    seed = 42
    Qs = get_Qs_for_potential(gmm, N=5)

    traj_ours = simulate_multiscale(gmm, Qs, dt, n_steps, kT=kT, seed=seed, record_every=1)
    traj_esh = simulate_esh(gmm, dt, n_steps, kT=kT, seed=seed, record_every=1,
                            energy_resample_interval=100)
    traj_nhc = simulate_nhc(gmm, np.ones(3), dt, n_steps, kT=kT, seed=seed, record_every=1)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    trajs = [traj_ours, traj_esh, traj_nhc]
    names = ["Multi-scale log-osc (ours)", "ESH + energy resample", "NHC (M=3)"]
    colors_method = ["#2ca02c", "#8c564b", "#ff7f0e"]
    mode_colors = plt.cm.tab10(np.arange(5))

    for row, (traj, name, col) in enumerate(zip(trajs, names, colors_method)):
        if len(traj) == 0:
            axes[row, 0].text(0.5, 0.5, "DIVERGED", transform=axes[row, 0].transAxes,
                              fontsize=16, ha="center", va="center", color="red")
            axes[row, 1].text(0.5, 0.5, "DIVERGED", transform=axes[row, 1].transAxes,
                              fontsize=16, ha="center", va="center", color="red")
            axes[row, 0].set_ylabel(name, fontsize=11)
            continue

        # Left: scatter plot with mode assignment
        centers = gmm.centers
        d2 = np.sum((traj[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        assign = np.argmin(d2, axis=1)

        step = max(1, len(traj) // 5000)
        ax = axes[row, 0]
        for m in range(5):
            mask = assign[::step] == m
            ax.scatter(traj[::step, 0][mask], traj[::step, 1][mask],
                       c=[mode_colors[m]], s=2, alpha=0.5)
        # Plot mode centers
        for m in range(5):
            ax.plot(centers[m, 0], centers[m, 1], 'kx', markersize=8, markeredgewidth=2)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")
        ax.set_ylabel(name, fontsize=11)
        mc = count_mode_crossings(traj, gmm)
        mv = modes_visited(traj, gmm)
        ax.set_title(f"Spatial ({mc} crossings, {mv:.0%} modes visited)" if row == 0 else
                     f"({mc} crossings, {mv:.0%} modes visited)", fontsize=10)

        # Right: mode index vs time
        ax2 = axes[row, 1]
        t = np.arange(len(assign))
        for m in range(5):
            mask = assign == m
            ax2.scatter(t[mask][::step], assign[mask][::step],
                        c=[mode_colors[m]], s=1, alpha=0.3)
        ax2.set_ylabel("Mode index", fontsize=11)
        ax2.set_ylim(-0.5, 4.5)
        ax2.set_yticks(range(5))
        if row == 0:
            ax2.set_title("Mode occupation over time", fontsize=10)

    axes[2, 0].set_xlabel("x", fontsize=12)
    axes[2, 1].set_xlabel("Integration step", fontsize=12)
    fig.suptitle("Mode Hopping: 2D Gaussian Mixture (5 modes, r=3, $\\sigma$=0.5)",
                 fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_mode_hopping_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig3_mode_hopping_comparison.png")


# ============================================================================
# Main
# ============================================================================
def main():
    t0 = time.time()
    results = {}

    # Run benchmarks (moderate budget for tractable runtime in pure Python)
    N_EVALS = 100_000
    N_SEEDS = 5

    dw_summary, dw_raw = benchmark_double_well(n_force_evals=N_EVALS, n_seeds=N_SEEDS)
    results["double_well"] = dw_summary

    gmm_summary, gmm_raw = benchmark_gmm(n_force_evals=N_EVALS, n_seeds=N_SEEDS)
    results["gmm"] = gmm_summary

    g5d_summary, g5d_raw = benchmark_anisotropic(dim=5, kappa_ratio=100.0,
                                                  n_force_evals=N_EVALS, n_seeds=N_SEEDS)
    results["gaussian_5d"] = g5d_summary

    g10d_summary, g10d_raw = benchmark_anisotropic(dim=10, kappa_ratio=100.0,
                                                    n_force_evals=N_EVALS, n_seeds=N_SEEDS)
    results["gaussian_10d"] = g10d_summary

    stab_summary, stab_raw = stability_sweep(n_force_evals=50_000, n_seeds=N_SEEDS)
    results["stability"] = stab_summary

    elapsed = time.time() - t0
    results["elapsed_sec"] = elapsed

    # Headline metric: our tau / ESH best tau on 5D
    our_tau_5d = g5d_summary["ours"]["tau_mean"]
    esh_tau_5d = min(g5d_summary["esh_micro"]["tau_mean"], g5d_summary["esh_resample"]["tau_mean"])
    results["headline_metric_our_tau_over_esh"] = our_tau_5d / esh_tau_5d if esh_tau_5d > 0 else float("inf")

    # Save results
    out_path = ORBIT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    print(f"\nSaved results to {out_path}")
    print(f"Total elapsed: {elapsed:.1f}s")
    print(f"Headline: our_tau/ESH_tau (5D) = {results['headline_metric_our_tau_over_esh']:.3f}")

    # Generate plots
    print("\n=== Generating figures ===")
    plot_benchmark_table(dw_summary, gmm_summary, g5d_summary, g10d_summary)
    plot_stability_envelope(stab_summary)
    plot_trajectory_comparison()
    plot_mode_hopping()

    return results


if __name__ == "__main__":
    main()
