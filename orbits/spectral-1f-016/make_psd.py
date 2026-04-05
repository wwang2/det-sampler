"""PSD Analysis: Multi-scale log-osc thermostats generate 1/f friction noise.

Runs multi-scale log-osc on 1D HO with N=1,3,5,7,10 scales (log-spaced Q).
Collects g_total(t) = sum_k 2*xi_k/(1+xi_k^2) over 2M steps (dt=0.005).
Computes PSD via scipy.signal.welch. Fits S(f) ~ f^{-alpha}.

Dutta-Horn mechanism: superposition of Lorentzian relaxation processes
with log-uniform relaxation rates produces 1/f power spectral density.
Each thermostat xi_k oscillates at frequency ~ sqrt(d*kT/Q_k).
Log-spaced Q -> log-uniform frequencies -> 1/f friction noise.

The 1/f regime should appear in the band [f_min, f_max] where
f_min ~ 1/sqrt(Q_max) and f_max ~ 1/sqrt(Q_min).

Reference: Dutta & Horn, Rev. Mod. Phys. 53, 497 (1981).
"""

import sys
import os
import json
import numpy as np
from scipy.signal import welch
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from research.eval.potentials import HarmonicOscillator1D
from research.eval.integrators import ThermostatState


def g_func(xi_val):
    """Bounded friction: g(xi) = 2*xi/(1+xi^2), in [-1, 1]."""
    return 2.0 * xi_val / (1.0 + xi_val**2)


def run_and_collect(Qs, n_steps=2_000_000, dt=0.005, seed=42):
    """Run multi-scale log-osc on 1D HO, collect per-thermostat friction time series."""
    potential = HarmonicOscillator1D(omega=1.0)
    dim = 1
    kT = 1.0
    mass = 1.0
    n_thermo = len(Qs)

    rng = np.random.default_rng(seed)
    q = np.array([0.5])
    p = rng.normal(0, np.sqrt(mass * kT), size=dim)
    xi = np.zeros(n_thermo)

    # Storage: per-thermostat g(xi_k) and total
    g_total_series = np.zeros(n_steps)
    g_per_thermo = np.zeros((n_thermo, n_steps))

    half_dt = 0.5 * dt

    # Initial gradient
    grad_U = potential.gradient(q)

    for step in range(n_steps):
        # Record friction BEFORE the step
        for k in range(n_thermo):
            gk = g_func(xi[k])
            g_per_thermo[k, step] = gk
        g_total_series[step] = np.sum(g_per_thermo[:, step])

        # Velocity Verlet step (same as MultiScaleLogOscVerlet)
        # Half-step xi
        kinetic = np.sum(p**2) / mass
        drive = kinetic - dim * kT
        xi_dot = np.array([drive / Q for Q in Qs])
        xi = xi + half_dt * xi_dot

        # Half-step p (friction + kick)
        total_g = sum(g_func(xi[k]) for k in range(n_thermo))
        scale = np.exp(-total_g * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale
        p = p - half_dt * grad_U

        # Full-step q
        q = q + dt * p / mass

        # New gradient
        grad_U = potential.gradient(q)

        # Half-step p (kick + friction)
        p = p - half_dt * grad_U
        total_g = sum(g_func(xi[k]) for k in range(n_thermo))
        scale = np.exp(-total_g * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale

        # Half-step xi
        kinetic = np.sum(p**2) / mass
        drive = kinetic - dim * kT
        xi_dot = np.array([drive / Q for Q in Qs])
        xi = xi + half_dt * xi_dot

    return g_total_series, g_per_thermo


def compute_psd(signal, dt, nperseg=8192):
    """Compute PSD via Welch's method."""
    freqs, psd = welch(signal, fs=1.0/dt, nperseg=nperseg, noverlap=nperseg//2)
    return freqs, psd


def fit_power_law(freqs, psd, f_min=None, f_max=None):
    """Fit S(f) ~ A * f^{-alpha} in log-log space."""
    mask = freqs > 0
    if f_min is not None:
        mask &= freqs >= f_min
    if f_max is not None:
        mask &= freqs <= f_max
    f_fit = freqs[mask]
    p_fit = psd[mask]

    if len(f_fit) < 5:
        return 0.0, 0.0, 0.0

    def power_law_log(log_f, log_A, alpha):
        return log_A - alpha * log_f

    log_f = np.log10(f_fit)
    log_p = np.log10(p_fit)

    try:
        popt, pcov = curve_fit(power_law_log, log_f, log_p, p0=[0.0, 1.0])
        log_A, alpha = popt
        perr = np.sqrt(np.diag(pcov))
        alpha_err = perr[1]
    except Exception:
        alpha = 0.0
        alpha_err = 0.0
        log_A = 0.0

    return alpha, alpha_err, 10**log_A


def get_log_spaced_Qs(n_scales, Q_min=0.01, Q_max=1000.0):
    """Generate log-spaced Q values."""
    if n_scales == 1:
        return [np.sqrt(Q_min * Q_max)]  # geometric mean
    return list(np.logspace(np.log10(Q_min), np.log10(Q_max), n_scales))


def main():
    print("=" * 70)
    print("PSD Analysis: Multi-scale log-osc thermostats and 1/f noise")
    print("=" * 70)

    dt = 0.005
    n_steps = 2_000_000
    nperseg = 16384  # larger segments for better low-freq resolution
    seed = 42

    N_scales_list = [1, 3, 5, 7, 10]
    results = {}

    for N in N_scales_list:
        Qs = get_log_spaced_Qs(N)
        print(f"\nN={N} scales, Qs={[f'{q:.4f}' for q in Qs]}")

        g_total, g_per = run_and_collect(Qs, n_steps=n_steps, dt=dt, seed=seed)

        freqs, psd = compute_psd(g_total, dt, nperseg=nperseg)

        # Dutta-Horn: 1/f regime in [1/sqrt(Q_max), 1/sqrt(Q_min)]
        # Thermostat oscillation frequency ~ sqrt(kT/Q) (for dim=1)
        f_lo = 1.0 / np.sqrt(max(Qs))
        f_hi = 1.0 / np.sqrt(min(Qs))
        # But clip to available frequency range
        f_lo = max(f_lo, freqs[1])
        f_hi = min(f_hi, freqs[-1])

        # Fit in the predicted 1/f band
        alpha_band, alpha_band_err, A_band = fit_power_law(freqs, psd, f_min=f_lo, f_max=f_hi)

        # Also fit over a broad "mid" range for comparison
        alpha_mid, alpha_mid_err, A_mid = fit_power_law(freqs, psd, f_min=0.1, f_max=10.0)

        # Also fit in a narrower central band (1 decade around geometric mean)
        f_center = np.sqrt(f_lo * f_hi) if f_lo > 0 and f_hi > 0 else 1.0
        f_narrow_lo = f_center / 3.0
        f_narrow_hi = f_center * 3.0
        alpha_narrow, alpha_narrow_err, _ = fit_power_law(freqs, psd, f_min=f_narrow_lo, f_max=f_narrow_hi)

        print(f"  Dutta-Horn band: [{f_lo:.3f}, {f_hi:.3f}] Hz")
        print(f"  alpha (DH band)  = {alpha_band:.3f} +/- {alpha_band_err:.3f}")
        print(f"  alpha (0.1-10)   = {alpha_mid:.3f} +/- {alpha_mid_err:.3f}")
        print(f"  alpha (narrow)   = {alpha_narrow:.3f} +/- {alpha_narrow_err:.3f}")

        results[N] = {
            'Qs': Qs,
            'freqs': freqs.tolist(),
            'psd': psd.tolist(),
            'alpha_dh_band': float(alpha_band),
            'alpha_dh_band_err': float(alpha_band_err),
            'alpha_mid': float(alpha_mid),
            'alpha_mid_err': float(alpha_mid_err),
            'alpha_narrow': float(alpha_narrow),
            'alpha_narrow_err': float(alpha_narrow_err),
            'f_lo': float(f_lo),
            'f_hi': float(f_hi),
        }

        # Store per-thermostat PSDs for N=3 (used by decomposition)
        if N == 3:
            per_thermo_psds = {}
            for k in range(len(Qs)):
                fk, pk = compute_psd(g_per[k], dt, nperseg=nperseg)
                per_thermo_psds[k] = {
                    'freqs': fk.tolist(),
                    'psd': pk.tolist(),
                    'Q': Qs[k],
                }
            results['per_thermo_N3'] = per_thermo_psds

        # Store per-thermostat PSDs for all N (for the consolidated figure)
        per_thermo_psds_all = {}
        for k in range(len(Qs)):
            fk, pk = compute_psd(g_per[k], dt, nperseg=nperseg)
            per_thermo_psds_all[k] = {
                'freqs': fk.tolist(),
                'psd': pk.tolist(),
                'Q': Qs[k],
            }
        results[f'per_thermo_N{N}'] = per_thermo_psds_all

    # Also save the N=1 and N=3 friction time series (subsampled) for panel (d)
    for N_ts in [1, 3]:
        Qs = get_log_spaced_Qs(N_ts)
        g_total, _ = run_and_collect(Qs, n_steps=200_000, dt=dt, seed=seed)
        # Subsample to 10000 points for plotting
        step = max(1, len(g_total) // 10000)
        results[f'timeseries_N{N_ts}'] = {
            'g_total': g_total[::step].tolist(),
            'dt': dt * step,
            'Qs': Qs,
        }

    # Save results
    outdir = os.path.dirname(os.path.abspath(__file__))
    outpath = os.path.join(outdir, 'psd_results.json')
    with open(outpath, 'w') as f:
        json.dump(results, f)
    print(f"\nResults saved to {outpath}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary: alpha vs N_scales (Dutta-Horn band)")
    print("=" * 70)
    for N in N_scales_list:
        r = results[N]
        print(f"  N={N:2d}: alpha(DH)={r['alpha_dh_band']:.3f}+/-{r['alpha_dh_band_err']:.3f}  "
              f"alpha(mid)={r['alpha_mid']:.3f}+/-{r['alpha_mid_err']:.3f}  "
              f"band=[{r['f_lo']:.3f},{r['f_hi']:.3f}]")

    return results


if __name__ == '__main__':
    main()
