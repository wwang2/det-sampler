"""thermostat-dynamics-046: Internal dynamics and spectral study of multi-scale log-osc thermostats.

Key finding from initial run: all parallel thermostats share the SAME drive signal
(K - dim*kT), so xi_i are perfectly correlated up to Q-dependent scaling.
This means Gamma(t) = sum g(xi_i) does NOT decompose into independent Lorentzians.

Experiments on 5D anisotropic Gaussian (kappa=[1, 3.16, 10, 31.6, 100], kT=1, dt=0.005):
1. PSD of Gamma(t) for N=3,5,10,20 + NHC(M=5) + single NH
2. Individual xi_i PSDs for N=5 (are they Lorentzians? at what frequencies?)
3. Cross-correlation between xi_i and q_d^2 (do thermostats couple to specific modes?)
4. Energy equilibration from non-equilibrium start (multi-scale vs NHC vs NH)
5. PSD comparison: multi-scale N=5 vs NHC(M=5) vs single NH (annotated key figure)
"""

import os
import sys
import json
import time
import numpy as np
from scipy.signal import welch

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
# Integrators with full state recording
# ---------------------------------------------------------------------------
def g_func(xi):
    """Log-oscillator friction: g(xi) = 2*xi / (1 + xi^2)."""
    return 2.0 * xi / (1.0 + xi * xi)


def simulate_multiscale_full(potential, Qs, dt, n_steps, kT=1.0, mass=1.0,
                              seed=0, record_every=1):
    """Multi-scale log-osc with full state recording (q, p, xi, Gamma).

    BAOAB splitting:
      dp/dt = -grad U - Gamma * p,  Gamma = sum_i g(xi_i)
      dxi_i/dt = (K - dim*kT) / Q_i
    """
    rng = np.random.default_rng(seed)
    dim = potential.dim
    Qs = np.asarray(Qs, dtype=float)
    N = len(Qs)

    if hasattr(potential, 'kappas'):
        q = rng.normal(0, 1.0, size=dim) / np.sqrt(np.maximum(potential.kappas, 1e-6))
    else:
        q = rng.normal(0, 1.0, size=dim)
    p = rng.normal(0, np.sqrt(mass * kT), size=dim)
    xi = np.zeros(N)

    half = 0.5 * dt
    grad_U = potential.gradient(q)
    n_rec = n_steps // record_every

    qs_rec = np.empty((n_rec, dim))
    ps_rec = np.empty((n_rec, dim))
    xi_rec = np.empty((n_rec, N))
    gamma_rec = np.empty(n_rec)
    rec_i = 0

    for step in range(n_steps):
        # B: Half-step thermostats
        K = float(np.sum(p * p)) / mass
        xi = xi + half * (K - dim * kT) / Qs

        # A: Half-step momenta (friction + force)
        gtot = float(np.sum(g_func(xi)))
        p = p * np.exp(-gtot * half)
        p = p - half * grad_U

        # O: Full-step positions
        q = q + dt * p / mass
        grad_U = potential.gradient(q)

        # A: Half-step momenta
        p = p - half * grad_U
        gtot = float(np.sum(g_func(xi)))
        p = p * np.exp(-gtot * half)

        # B: Half-step thermostats
        K = float(np.sum(p * p)) / mass
        xi = xi + half * (K - dim * kT) / Qs

        if (step + 1) % record_every == 0 and rec_i < n_rec:
            qs_rec[rec_i] = q
            ps_rec[rec_i] = p
            xi_rec[rec_i] = xi.copy()
            gamma_rec[rec_i] = float(np.sum(g_func(xi)))
            rec_i += 1

        if not np.isfinite(p).all() or not np.isfinite(q).all():
            break

    return qs_rec[:rec_i], ps_rec[:rec_i], xi_rec[:rec_i], gamma_rec[:rec_i]


def simulate_nhc_full(potential, Qs, dt, n_steps, kT=1.0, mass=1.0,
                       seed=0, record_every=1):
    """Nose-Hoover Chain with full state recording.

    dp/dt = -grad U - xi_1 * p
    dxi_1/dt = (K - dim*kT)/Q_1 - xi_2 * xi_1
    dxi_j/dt = (Q_{j-1} * xi_{j-1}^2 - kT)/Q_j - xi_{j+1} * xi_j
    dxi_M/dt = (Q_{M-1} * xi_{M-1}^2 - kT)/Q_M
    """
    rng = np.random.default_rng(seed)
    dim = potential.dim
    Qs = np.asarray(Qs, dtype=float)
    M = len(Qs)

    if hasattr(potential, 'kappas'):
        q = rng.normal(0, 1.0, size=dim) / np.sqrt(np.maximum(potential.kappas, 1e-6))
    else:
        q = rng.normal(0, 1.0, size=dim)
    p = rng.normal(0, np.sqrt(mass * kT), size=dim)
    xi = np.zeros(M)

    half = 0.5 * dt
    grad_U = potential.gradient(q)
    n_rec = n_steps // record_every

    qs_rec = np.empty((n_rec, dim))
    ps_rec = np.empty((n_rec, dim))
    xi_rec = np.empty((n_rec, M))
    gamma_rec = np.empty(n_rec)  # friction on p = xi[0]
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
            ps_rec[rec_i] = p
            xi_rec[rec_i] = xi.copy()
            gamma_rec[rec_i] = xi[0]  # NHC friction = xi_1
            rec_i += 1

        if not np.isfinite(p).all() or not np.isfinite(q).all():
            break

    return qs_rec[:rec_i], ps_rec[:rec_i], xi_rec[:rec_i], gamma_rec[:rec_i]


def simulate_single_nh_full(potential, Q, dt, n_steps, kT=1.0, mass=1.0,
                             seed=0, record_every=1):
    """Single Nose-Hoover with full state recording."""
    rng = np.random.default_rng(seed)
    dim = potential.dim
    if hasattr(potential, 'kappas'):
        q = rng.normal(0, 1.0, size=dim) / np.sqrt(np.maximum(potential.kappas, 1e-6))
    else:
        q = rng.normal(0, 1.0, size=dim)
    p = rng.normal(0, np.sqrt(mass * kT), size=dim)
    xi = 0.0

    half = 0.5 * dt
    grad_U = potential.gradient(q)
    n_rec = n_steps // record_every

    qs_rec = np.empty((n_rec, dim))
    ps_rec = np.empty((n_rec, dim))
    xi_rec = np.empty(n_rec)
    gamma_rec = np.empty(n_rec)
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
            ps_rec[rec_i] = p
            xi_rec[rec_i] = xi
            gamma_rec[rec_i] = xi
            rec_i += 1

        if not np.isfinite(p).all() or not np.isfinite(q).all():
            break

    return qs_rec[:rec_i], ps_rec[:rec_i], xi_rec[:rec_i], gamma_rec[:rec_i]


# ---------------------------------------------------------------------------
# Non-equilibrium start integrators (record KE per mode + q^2 per mode)
# ---------------------------------------------------------------------------
def simulate_multiscale_noneq(potential, Qs, dt, n_steps, kT=1.0, mass=1.0,
                               record_every=1):
    """Multi-scale log-osc from non-equilibrium: all modes start hot (5x kT)."""
    dim = potential.dim
    Qs = np.asarray(Qs, dtype=float)
    N = len(Qs)

    # Non-eq start: every mode has 5x the equilibrium KE
    rng = np.random.default_rng(123)
    q = rng.normal(0, 1.0, size=dim) / np.sqrt(np.maximum(potential.kappas, 1e-6))
    p = rng.normal(0, np.sqrt(5.0 * mass * kT), size=dim)  # 5x too hot
    xi = np.zeros(N)

    half = 0.5 * dt
    grad_U = potential.gradient(q)
    n_rec = n_steps // record_every
    ke_rec = np.empty((n_rec, dim))
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
            ke_rec[rec_i] = 0.5 * p * p / mass
            rec_i += 1

        if not np.isfinite(p).all() or not np.isfinite(q).all():
            break

    return ke_rec[:rec_i]


def simulate_nhc_noneq(potential, Qs, dt, n_steps, kT=1.0, mass=1.0,
                        record_every=1):
    """NHC from non-equilibrium: all modes start hot (5x kT)."""
    dim = potential.dim
    Qs = np.asarray(Qs, dtype=float)
    M = len(Qs)

    rng = np.random.default_rng(123)
    q = rng.normal(0, 1.0, size=dim) / np.sqrt(np.maximum(potential.kappas, 1e-6))
    p = rng.normal(0, np.sqrt(5.0 * mass * kT), size=dim)  # 5x too hot
    xi = np.zeros(M)

    half = 0.5 * dt
    grad_U = potential.gradient(q)
    n_rec = n_steps // record_every
    ke_rec = np.empty((n_rec, dim))
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
            ke_rec[rec_i] = 0.5 * p * p / mass
            rec_i += 1

        if not np.isfinite(p).all() or not np.isfinite(q).all():
            break

    return ke_rec[:rec_i]


def simulate_nh_noneq(potential, Q, dt, n_steps, kT=1.0, mass=1.0,
                       record_every=1):
    """Single NH from non-equilibrium: all modes start hot (5x kT)."""
    dim = potential.dim
    rng = np.random.default_rng(123)
    q = rng.normal(0, 1.0, size=dim) / np.sqrt(np.maximum(potential.kappas, 1e-6))
    p = rng.normal(0, np.sqrt(5.0 * mass * kT), size=dim)  # 5x too hot
    xi = 0.0

    half = 0.5 * dt
    grad_U = potential.gradient(q)
    n_rec = n_steps // record_every
    ke_rec = np.empty((n_rec, dim))
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
            ke_rec[rec_i] = 0.5 * p * p / mass
            rec_i += 1

        if not np.isfinite(p).all() or not np.isfinite(q).all():
            break

    return ke_rec[:rec_i]


# ---------------------------------------------------------------------------
# PSD helpers
# ---------------------------------------------------------------------------
def compute_psd(signal, dt, nperseg=8192):
    """Welch PSD estimate."""
    freqs, psd = welch(signal, fs=1.0 / dt, nperseg=min(nperseg, len(signal) // 2),
                       noverlap=nperseg // 2)
    return freqs, psd


def fit_power_law(freqs, psd, f_min=None, f_max=None):
    """Fit PSD ~ f^{-alpha} in log-log space. Returns (alpha, alpha_err)."""
    mask = freqs > 0
    if f_min is not None:
        mask &= freqs >= f_min
    if f_max is not None:
        mask &= freqs <= f_max
    f_fit = freqs[mask]
    p_fit = psd[mask]
    if len(f_fit) < 5:
        return 0.0, 0.0

    log_f = np.log10(f_fit)
    log_p = np.log10(np.maximum(p_fit, 1e-30))
    try:
        coeffs = np.polyfit(log_f, log_p, 1)
        slope = coeffs[0]          # PSD ~ f^slope
        alpha = -slope             # alpha > 0 for 1/f noise
        residuals = log_p - np.polyval(coeffs, log_f)
        alpha_err = np.std(residuals) / np.sqrt(len(residuals))
    except Exception:
        alpha, alpha_err = 0.0, 0.0
    return alpha, alpha_err


# ---------------------------------------------------------------------------
# Cross-correlation
# ---------------------------------------------------------------------------
def cross_corr_max(x, y, max_lag=5000):
    """Max |cross-correlation| between x and y using FFT."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    y = np.asarray(y, dtype=float) - np.mean(y)
    sx, sy = np.std(x), np.std(y)
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    fx = np.fft.rfft(x, n=2 * n)
    fy = np.fft.rfft(y, n=2 * n)
    corr = np.fft.irfft(fx * np.conj(fy))[:max_lag]
    corr /= (n * sx * sy)
    return float(np.max(np.abs(corr)))


def pearson_corr(x, y):
    """Simple Pearson correlation."""
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = np.sqrt(np.sum(x**2) * np.sum(y**2))
    if denom < 1e-30:
        return 0.0
    return float(np.sum(x * y) / denom)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
OUTDIR = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(OUTDIR, "figures")

KAPPAS = np.array([1.0, 3.16227766, 10.0, 31.6227766, 100.0])
DIM = 5
KT = 1.0
DT = 0.005
SEED = 42

# Q range: kT/kappa_max to kT/kappa_min
Q_MIN = KT / KAPPAS[-1]   # 0.01
Q_MAX = KT / KAPPAS[0]    # 1.0


def get_Qs(N):
    """Log-uniform Q values in [Q_MIN, Q_MAX]."""
    if N == 1:
        return np.array([np.sqrt(Q_MIN * Q_MAX)])
    return np.exp(np.linspace(np.log(Q_MIN), np.log(Q_MAX), N))


# ---------------------------------------------------------------------------
# Experiment 1: PSD of Gamma(t) for varying N
# ---------------------------------------------------------------------------
def exp1_gamma_psd():
    print("=" * 60)
    print("Exp 1: PSD of Gamma(t)")
    print("=" * 60)
    pot = AnisotropicGaussian(KAPPAS)
    n_steps = 500_000
    nperseg = 8192

    results = {}

    for N in [3, 5, 10, 20]:
        Qs = get_Qs(N)
        print(f"  N={N}, Qs_range=[{Qs[0]:.4f}, {Qs[-1]:.4f}]")
        _, _, xi_rec, gamma = simulate_multiscale_full(
            pot, Qs, DT, n_steps, kT=KT, seed=SEED, record_every=1)
        freqs, psd = compute_psd(gamma, DT, nperseg=nperseg)

        # Dutta-Horn band: frequency range set by Q values
        # f_i ~ sqrt(dim*kT/Q_i)/(2*pi), so f_min ~ sqrt(dim*kT/Q_max), f_max ~ sqrt(dim*kT/Q_min)
        f_dh_lo = np.sqrt(DIM * KT / Qs[-1]) / (2 * np.pi)
        f_dh_hi = np.sqrt(DIM * KT / Qs[0]) / (2 * np.pi)
        f_dh_lo = max(f_dh_lo, freqs[2])
        f_dh_hi = min(f_dh_hi, freqs[-2])

        # Fit in Dutta-Horn band (where 1/f is theoretically expected)
        alpha_dh, alpha_dh_err = fit_power_law(freqs, psd, f_min=f_dh_lo, f_max=f_dh_hi)

        # Also broad fit for reference (0.05 to 10 Hz, avoid high-f cutoff)
        f_lo = max(freqs[2], 0.05)
        f_hi = min(freqs[-1] * 0.5, 10.0)
        alpha, alpha_err = fit_power_law(freqs, psd, f_min=f_lo, f_max=f_hi)

        # Check xi correlation: Pearson between xi_0 and xi_1
        if N >= 2:
            rho_01 = pearson_corr(xi_rec[:, 0], xi_rec[:, 1])
        else:
            rho_01 = float('nan')

        print(f"    alpha_broad={alpha:.3f}+/-{alpha_err:.3f}, "
              f"alpha_DH={alpha_dh:.3f}+/-{alpha_dh_err:.3f}, "
              f"rho(xi0,xi1)={rho_01:.4f}")

        results[f"multiscale_N{N}"] = {
            "freqs": freqs.tolist(),
            "psd": psd.tolist(),
            "alpha_broad": alpha, "alpha_broad_err": alpha_err,
            "alpha_dh": alpha_dh, "alpha_dh_err": alpha_dh_err,
            "rho_xi01": rho_01,
            "Qs": Qs.tolist(),
        }

    # NHC(M=5) with Q=1.0
    Qs_nhc = np.ones(5) * 1.0
    print(f"  NHC M=5, Q_ref=1.0")
    _, _, xi_nhc, gamma_nhc = simulate_nhc_full(
        pot, Qs_nhc, DT, n_steps, kT=KT, seed=SEED, record_every=1)
    freqs_nhc, psd_nhc = compute_psd(gamma_nhc, DT, nperseg=nperseg)
    alpha_nhc, _ = fit_power_law(freqs_nhc, psd_nhc, f_min=0.05, f_max=20.0)
    rho_nhc_01 = pearson_corr(xi_nhc[:, 0], xi_nhc[:, 1]) if xi_nhc.shape[1] >= 2 else float('nan')
    print(f"    alpha={alpha_nhc:.3f}, rho(xi0,xi1)={rho_nhc_01:.4f}")
    results["nhc_M5"] = {
        "freqs": freqs_nhc.tolist(), "psd": psd_nhc.tolist(),
        "alpha_broad": alpha_nhc, "rho_xi01": rho_nhc_01,
    }

    # Single NH with Q=0.1
    Q_nh = 0.1
    print(f"  Single NH, Q={Q_nh}")
    _, _, _, gamma_nh = simulate_single_nh_full(
        pot, Q_nh, DT, n_steps, kT=KT, seed=SEED, record_every=1)
    freqs_nh, psd_nh = compute_psd(gamma_nh, DT, nperseg=nperseg)
    alpha_nh, _ = fit_power_law(freqs_nh, psd_nh, f_min=0.05, f_max=20.0)
    print(f"    alpha={alpha_nh:.3f}")
    results["nh_single"] = {
        "freqs": freqs_nh.tolist(), "psd": psd_nh.tolist(),
        "alpha_broad": alpha_nh,
    }

    return results


# ---------------------------------------------------------------------------
# Experiment 2: Individual xi_i PSDs (N=5)
# ---------------------------------------------------------------------------
def exp2_individual_xi_psd():
    print("\n" + "=" * 60)
    print("Exp 2: Individual xi_i PSDs (N=5)")
    print("=" * 60)
    pot = AnisotropicGaussian(KAPPAS)
    N = 5
    Qs = get_Qs(N)
    n_steps = 500_000
    nperseg = 8192

    _, _, xi_rec, gamma_rec = simulate_multiscale_full(
        pot, Qs, DT, n_steps, kT=KT, seed=SEED, record_every=1)

    results = {"Qs": Qs.tolist(), "individual": {}}

    # Compute PSD for each g(xi_i)
    for i in range(N):
        g_i = g_func(xi_rec[:, i])
        freqs, psd = compute_psd(g_i, DT, nperseg=nperseg)
        peak_idx = np.argmax(psd[1:]) + 1
        f_peak = freqs[peak_idx]
        # Theoretical natural frequency: f ~ sqrt(dim*kT/Q_i) / (2*pi)
        f_theory = np.sqrt(DIM * KT / Qs[i]) / (2 * np.pi)
        print(f"  xi_{i}: Q={Qs[i]:.4f}, f_peak={f_peak:.2f}, f_theory={f_theory:.2f}")
        results["individual"][i] = {
            "freqs": freqs.tolist(), "psd": psd.tolist(),
            "f_peak": float(f_peak), "f_theory": float(f_theory),
            "Q": float(Qs[i]),
        }

    # Composite Gamma PSD
    freqs_g, psd_g = compute_psd(gamma_rec, DT, nperseg=nperseg)
    results["composite"] = {"freqs": freqs_g.tolist(), "psd": psd_g.tolist()}

    # xi pairwise Pearson correlations (key diagnostic)
    corr_matrix = np.corrcoef(xi_rec.T)
    print(f"  xi pairwise correlation matrix:\n{corr_matrix}")
    results["xi_corr_matrix"] = corr_matrix.tolist()

    return results


# ---------------------------------------------------------------------------
# Experiment 3: Cross-correlation heatmap
# ---------------------------------------------------------------------------
def exp3_cross_correlation():
    print("\n" + "=" * 60)
    print("Exp 3: Cross-correlation xi_i vs q_d^2")
    print("=" * 60)
    pot = AnisotropicGaussian(KAPPAS)
    N = 5
    Qs = get_Qs(N)
    n_steps = 500_000

    qs, _, xi_rec, _ = simulate_multiscale_full(
        pot, Qs, DT, n_steps, kT=KT, seed=SEED, record_every=1)

    # Two heatmaps: Pearson correlation (simpler, more interpretable) and max |xcorr|
    heatmap_pearson = np.zeros((N, DIM))
    heatmap_xcorr = np.zeros((N, DIM))

    for i in range(N):
        for d in range(DIM):
            q2_d = qs[:, d] ** 2
            heatmap_pearson[i, d] = pearson_corr(xi_rec[:, i], q2_d)
            heatmap_xcorr[i, d] = cross_corr_max(xi_rec[:, i], q2_d, max_lag=10000)

    print("  Pearson correlation (xi_i vs q_d^2):")
    for i in range(N):
        vals = ", ".join(f"{heatmap_pearson[i,d]:.4f}" for d in range(DIM))
        print(f"    xi_{i} (Q={Qs[i]:.4f}): [{vals}]")

    # Also compute Pearson(xi_i, xi_j) to confirm lockstep
    xi_pearson = np.corrcoef(xi_rec.T)
    print(f"  xi-xi Pearson matrix (off-diagonal should be ~1 if locked):")
    for i in range(N):
        vals = ", ".join(f"{xi_pearson[i,j]:.4f}" for j in range(N))
        print(f"    [{vals}]")

    return {
        "heatmap_pearson": heatmap_pearson.tolist(),
        "heatmap_xcorr": heatmap_xcorr.tolist(),
        "xi_pearson": xi_pearson.tolist(),
        "Qs": Qs.tolist(),
        "kappas": KAPPAS.tolist(),
    }


# ---------------------------------------------------------------------------
# Experiment 4: Energy equilibration
# ---------------------------------------------------------------------------
def exp4_energy_equilibration():
    print("\n" + "=" * 60)
    print("Exp 4: Energy equilibration from non-equilibrium start")
    print("=" * 60)
    pot = AnisotropicGaussian(KAPPAS)
    n_steps = 200_000
    rec_every = 10

    # Multi-scale N=5
    Qs = get_Qs(5)
    print(f"  Multi-scale N=5, Qs_range=[{Qs[0]:.4f}, {Qs[-1]:.4f}]")
    ke_ms = simulate_multiscale_noneq(pot, Qs, DT, n_steps, kT=KT,
                                       record_every=rec_every)

    # NHC M=3
    Qs_nhc = np.ones(3) * 1.0
    print(f"  NHC M=3, Q=1.0")
    ke_nhc = simulate_nhc_noneq(pot, Qs_nhc, DT, n_steps, kT=KT,
                                 record_every=rec_every)

    # Single NH
    Q_nh = 0.1
    print(f"  Single NH, Q={Q_nh}")
    ke_nh = simulate_nh_noneq(pot, Q_nh, DT, n_steps, kT=KT,
                               record_every=rec_every)

    # Smoothing with running average
    def running_avg(arr, w=200):
        kernel = np.ones(w) / w
        out = np.empty_like(arr)
        for d in range(arr.shape[1]):
            out[:, d] = np.convolve(arr[:, d], kernel, mode='same')
        return out

    dt_rec = DT * rec_every

    # Report equilibration metrics: time for total KE to reach within 50% of dim*kT/2
    for name, ke in [("multiscale", ke_ms), ("nhc", ke_nhc), ("nh", ke_nh)]:
        target_total = DIM * 0.5 * KT  # dim*kT/2 total
        ke_smooth = running_avg(ke, w=200)
        total_ke = ke_smooth.sum(axis=1)
        within = np.where(np.abs(total_ke / target_total - 1.0) < 0.5)[0]
        if len(within) > 0:
            t_eq = within[0] * dt_rec
        else:
            t_eq = float('inf')
        mean_final = ke[-500:].mean(axis=0) if len(ke) > 500 else ke.mean(axis=0)
        print(f"    {name}: total KE equil time ~ {t_eq:.1f}, "
              f"final <K_d>/(kT/2) = {mean_final / (0.5*KT)}")

    return {
        "multiscale": running_avg(ke_ms).tolist(),
        "nhc": running_avg(ke_nhc).tolist(),
        "nh": running_avg(ke_nh).tolist(),
        "dt_rec": dt_rec,
        "n_rec": len(ke_ms),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_figures(r1, r2, r3, r4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    os.makedirs(FIGDIR, exist_ok=True)

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'axes.titlesize': 13,
    })

    tab10 = plt.cm.tab10
    colors_N = {3: tab10(2), 5: tab10(3), 10: tab10(4), 20: tab10(5)}
    color_nhc = '#ff7f0e'
    color_nh = '#1f77b4'

    # -----------------------------------------------------------------------
    # Fig 1: Gamma PSD for N=3,5,10,20 + NHC + NH
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    for N in [3, 5, 10, 20]:
        key = f"multiscale_N{N}"
        d = r1[key]
        f = np.array(d["freqs"]); p = np.array(d["psd"])
        mask = f > 0
        lbl = f"N={N} ($\\alpha_{{DH}}$={d['alpha_dh']:.2f})"
        ax.loglog(f[mask], p[mask], label=lbl, color=colors_N[N], lw=1.5)

    # NHC
    d = r1["nhc_M5"]
    f = np.array(d["freqs"]); p = np.array(d["psd"])
    mask = f > 0
    ax.loglog(f[mask], p[mask], label="NHC M=5", color=color_nhc, lw=1.5, ls='--')

    # Single NH
    d = r1["nh_single"]
    f = np.array(d["freqs"]); p = np.array(d["psd"])
    mask = f > 0
    ax.loglog(f[mask], p[mask], label="Single NH", color=color_nh, lw=1.5, ls='--')

    # 1/f reference
    d5 = r1["multiscale_N5"]
    f5 = np.array(d5["freqs"]); p5 = np.array(d5["psd"])
    idx1 = np.argmin(np.abs(f5 - 1.0))
    p_at_1 = p5[idx1] if idx1 > 0 else 1.0
    f_ref = np.logspace(-1, 2, 100)
    ax.loglog(f_ref, p_at_1 / f_ref, 'k--', alpha=0.4, lw=1, label=r"$1/f$ reference")

    # Dutta-Horn band
    f_dh_lo = 1.0 / (2 * np.pi * np.sqrt(Q_MAX))
    f_dh_hi = 1.0 / (2 * np.pi * np.sqrt(Q_MIN))
    ax.axvspan(f_dh_lo, f_dh_hi, alpha=0.07, color='gray',
               label=f"DH band [{f_dh_lo:.2f}, {f_dh_hi:.1f}] Hz")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"PSD of $\Gamma(t)$")
    ax.set_title(r"Power spectrum of total friction $\Gamma(t) = \sum_i g(\xi_i)$")
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(0.01, 100)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig1_gamma_psd.png"), dpi=150)
    plt.close(fig)
    print("  Saved fig1_gamma_psd.png")

    # -----------------------------------------------------------------------
    # Fig 2: Individual xi PSDs + composite (N=5)
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: individual g(xi_i) PSDs
    colors_xi = [tab10(i) for i in range(5)]
    for i in range(5):
        d = r2["individual"][i]
        f = np.array(d["freqs"]); p = np.array(d["psd"])
        mask = f > 0
        ax1.loglog(f[mask], p[mask], color=colors_xi[i], lw=1.2,
                   label=f"$g(\\xi_{i})$, Q={d['Q']:.3f}")
        # Mark peak
        ax1.axvline(d["f_peak"], color=colors_xi[i], ls=':', alpha=0.4, lw=0.8)
        # Mark theoretical
        ax1.axvline(d["f_theory"], color=colors_xi[i], ls='--', alpha=0.3, lw=0.8)

    # Composite
    dc = r2["composite"]
    f = np.array(dc["freqs"]); p = np.array(dc["psd"])
    mask = f > 0
    ax1.loglog(f[mask], p[mask], 'k-', lw=2.0, label=r"$\Gamma = \sum g(\xi_i)$")

    # 1/f reference
    idx1 = np.argmin(np.abs(f - 1.0))
    p_at_1 = p[idx1]
    f_ref = np.logspace(-1, 2, 100)
    ax1.loglog(f_ref, p_at_1 / f_ref, 'k--', alpha=0.3, lw=1, label=r"$1/f$")

    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("PSD")
    ax1.set_title("(a) Individual thermostat PSDs and composite")
    ax1.legend(fontsize=8, loc='upper right')
    ax1.set_xlim(0.01, 100)

    # Right: xi correlation matrix
    corr = np.array(r2["xi_corr_matrix"])
    im = ax2.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
    ax2.set_xlabel(r"Thermostat $j$")
    ax2.set_ylabel(r"Thermostat $i$")
    ax2.set_title(r"(b) $\rho(\xi_i, \xi_j)$ correlation matrix")
    Qs = np.array(r2["Qs"])
    labels = [f"$\\xi_{i}$\nQ={Qs[i]:.3f}" for i in range(5)]
    ax2.set_xticks(range(5)); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_yticks(range(5)); ax2.set_yticklabels(labels, fontsize=8)
    for i in range(5):
        for j in range(5):
            ax2.text(j, i, f"{corr[i,j]:.3f}", ha='center', va='center',
                     fontsize=9, color='white' if abs(corr[i,j]) > 0.5 else 'black')
    cb = fig.colorbar(im, ax=ax2, shrink=0.8)
    cb.set_label("Pearson correlation")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig2_individual_xi_psd.png"), dpi=150)
    plt.close(fig)
    print("  Saved fig2_individual_xi_psd.png")

    # -----------------------------------------------------------------------
    # Fig 3: Cross-correlation heatmap (Pearson + xcorr side by side)
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    Qs = np.array(r3["Qs"])
    kappas = np.array(r3["kappas"])

    for ax, hm_key, title in [(ax1, "heatmap_pearson", "(a) Pearson corr"),
                                (ax2, "heatmap_xcorr", r"(b) Max $|C(\tau)|$")]:
        hm = np.array(r3[hm_key])
        im = ax.imshow(hm, aspect='auto', cmap='viridis', origin='lower')
        ax.set_xlabel(r"Mode $d$ ($\kappa_d$)")
        ax.set_ylabel(r"Thermostat $i$ ($Q_i$)")
        ax.set_xticks(range(DIM))
        ax.set_xticklabels([f"d={d}\n$\\kappa$={kappas[d]:.1f}" for d in range(DIM)], fontsize=8)
        ax.set_yticks(range(5))
        ax.set_yticklabels([f"i={i}\nQ={Qs[i]:.3f}" for i in range(5)], fontsize=8)
        ax.set_title(title, fontsize=13)
        for i in range(5):
            for j in range(DIM):
                ax.text(j, i, f"{hm[i,j]:.3f}", ha='center', va='center',
                        fontsize=8, color='white' if hm[i,j] < 0.5 * hm.max() else 'black')
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(r"Cross-correlation: thermostat $\xi_i$ vs mode $q_d^2$", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig3_cross_correlation.png"), dpi=150)
    plt.close(fig)
    print("  Saved fig3_cross_correlation.png")

    # -----------------------------------------------------------------------
    # Fig 4: Energy equilibration (3 panels)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    titles = ["Multi-scale N=5", "NHC M=3", "Single NH"]
    data_keys = ["multiscale", "nhc", "nh"]
    dt_rec = r4["dt_rec"]

    mode_colors = [tab10(i) for i in range(5)]
    mode_labels = [f"d={d}, $\\kappa$={KAPPAS[d]:.1f}" for d in range(DIM)]

    for ax_i, (ax, title, key) in enumerate(zip(axes, titles, data_keys)):
        ke = np.array(r4[key])
        n_pts = len(ke)
        t = np.arange(n_pts) * dt_rec
        for d in range(DIM):
            ax.plot(t, ke[:, d] / (0.5 * KT), color=mode_colors[d],
                    lw=0.6, alpha=0.8, label=mode_labels[d] if ax_i == 0 else None)
        ax.axhline(1.0, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax.set_xlabel("Time")
        ax.set_title(f"({chr(97 + ax_i)}) {title}", fontsize=13)
        ax.set_xlim(0, t[-1] if len(t) > 0 else 1)
        ax.set_ylim(0, 12)

    axes[0].set_ylabel(r"$K_d(t) / (k_BT/2)$")
    axes[0].legend(fontsize=7, loc='upper right', ncol=1)
    fig.suptitle("Energy equilibration from non-equilibrium start (all modes 5x too hot)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig4_energy_equilibration.png"), dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig4_energy_equilibration.png")

    # -----------------------------------------------------------------------
    # Fig 5: PSD comparison (key paper figure)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    # Multi-scale N=5
    d = r1["multiscale_N5"]
    f = np.array(d["freqs"]); p = np.array(d["psd"])
    mask = f > 0
    ax.loglog(f[mask], p[mask], color=tab10(3), lw=2.0,
              label=f"Multi-scale N=5 ($\\alpha_{{DH}}$={d['alpha_dh']:.2f})")

    # NHC M=5
    d = r1["nhc_M5"]
    f = np.array(d["freqs"]); p = np.array(d["psd"])
    mask = f > 0
    ax.loglog(f[mask], p[mask], color=color_nhc, lw=2.0, ls='--',
              label="NHC M=5")

    # Single NH
    d = r1["nh_single"]
    f = np.array(d["freqs"]); p = np.array(d["psd"])
    mask = f > 0
    ax.loglog(f[mask], p[mask], color=color_nh, lw=2.0, ls='--',
              label="Single NH (Q=0.1)")

    # 1/f reference
    d5 = r1["multiscale_N5"]
    f5 = np.array(d5["freqs"]); p5 = np.array(d5["psd"])
    idx1 = np.argmin(np.abs(f5 - 1.0))
    p_at_1 = p5[idx1]
    f_ref = np.logspace(-1, 2, 100)
    ax.loglog(f_ref, p_at_1 / f_ref, 'k--', alpha=0.4, lw=1)

    # Annotations
    ax.annotate("1/f band\n(Dutta-Horn)", xy=(0.5, p_at_1 * 2), fontsize=11,
                style='italic', color='gray', ha='center')
    ax.annotate("Single NH:\nLorentzian peak", xy=(5, 0.01), fontsize=10,
                color=color_nh, ha='center',
                arrowprops=dict(arrowstyle='->', color=color_nh, lw=1.2),
                xytext=(10, 0.1))
    ax.annotate("NHC: broadened\nby chain coupling", xy=(0.3, 0.5), fontsize=10,
                color=color_nhc, ha='center')

    # DH band
    f_dh_lo = 1.0 / (2 * np.pi * np.sqrt(Q_MAX))
    f_dh_hi = 1.0 / (2 * np.pi * np.sqrt(Q_MIN))
    ax.axvspan(f_dh_lo, f_dh_hi, alpha=0.07, color='gray')

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"PSD of friction $\Gamma(t)$")
    ax.set_title("Friction PSD comparison: multi-scale log-osc vs NHC vs single NH")
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(0.01, 100)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig5_psd_comparison.png"), dpi=150)
    plt.close(fig)
    print("  Saved fig5_psd_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()

    r1 = exp1_gamma_psd()
    r2 = exp2_individual_xi_psd()
    r3 = exp3_cross_correlation()
    r4 = exp4_energy_equilibration()

    print("\n" + "=" * 60)
    print("Making figures...")
    print("=" * 60)
    make_figures(r1, r2, r3, r4)

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s")

    # Summary
    summary = {
        "psd_slopes": {},
        "xi_correlations": {},
        "cross_corr_uniformity": None,
        "elapsed_sec": elapsed,
    }
    for N in [3, 5, 10, 20]:
        d = r1[f"multiscale_N{N}"]
        summary["psd_slopes"][f"N{N}"] = {
            "alpha_broad": d["alpha_broad"],
            "alpha_dh": d.get("alpha_dh", None),
            "rho_xi01": d["rho_xi01"],
        }

    # Check if cross-corr heatmap rows are uniform (all xi see same thing)
    hm = np.array(r3["heatmap_pearson"])
    row_std = np.std(hm, axis=0).mean()
    col_std = np.std(hm, axis=1).mean()
    summary["cross_corr_uniformity"] = {
        "row_variation": float(row_std),
        "col_variation": float(col_std),
        "rows_uniform": bool(np.std(hm[0] - hm[1]) < 0.01),
    }

    with open(os.path.join(OUTDIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"Saved results.json")

    # Headline
    alpha_dh_5 = r1["multiscale_N5"]["alpha_dh"]
    alpha_broad_5 = r1["multiscale_N5"]["alpha_broad"]
    rho_01 = r1["multiscale_N5"]["rho_xi01"]
    print(f"\nHEADLINE: PSD slope N=5: alpha_DH={alpha_dh_5:.3f}, alpha_broad={alpha_broad_5:.3f} (target ~1.0)")
    print(f"          xi correlation rho(xi0,xi1) = {rho_01:.4f}")
    print(f"          Cross-corr rows uniform: {summary['cross_corr_uniformity']['rows_uniform']}")

    return summary


if __name__ == "__main__":
    main()
