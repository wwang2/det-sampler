"""Lorentzian Decomposition: Individual thermostat PSDs for N=3.

Uses Q=[0.1, 1.0, 10.0] where all three thermostats have measurable
oscillation frequencies within the simulation bandwidth.

For each thermostat k:
- Plot PSD of g(xi_k) = 2*xi_k/(1+xi_k^2)
- Show each is approximately Lorentzian: S_k(f) ~ tau_k / (1 + (2*pi*f*tau_k)^2)
- Overlay sum = total PSD ~= 1/f in the overlap region

Dutta-Horn: superposition of Lorentzians with log-distributed tau produces 1/f.

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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def g_func(xi_val):
    return 2.0 * xi_val / (1.0 + xi_val**2)


def lorentzian(f, A, tau):
    """Lorentzian PSD: S(f) = A * tau / (1 + (2*pi*f*tau)^2)."""
    return A * tau / (1.0 + (2.0 * np.pi * f * tau)**2)


def fit_lorentzian(freqs, psd):
    """Fit a Lorentzian to measured PSD."""
    mask = freqs > 0
    f_fit = freqs[mask]
    p_fit = psd[mask]

    # Initial guess from corner frequency
    psd_max = np.max(p_fit[:len(p_fit)//2])  # look at low-freq part
    half_idx = np.argmin(np.abs(p_fit - psd_max / 2))
    f_corner = f_fit[half_idx] if half_idx > 0 else 1.0
    tau_guess = 1.0 / (2.0 * np.pi * f_corner)
    A_guess = psd_max / tau_guess

    try:
        popt, pcov = curve_fit(lorentzian, f_fit, p_fit,
                               p0=[A_guess, tau_guess],
                               bounds=([0, 1e-6], [np.inf, 1e4]),
                               maxfev=10000)
        return popt
    except Exception:
        return [A_guess, tau_guess]


def run_and_collect_decomp(Qs, n_steps=2_000_000, dt=0.005, seed=42):
    """Run multi-scale log-osc on 1D HO, collect per-thermostat friction."""
    potential = HarmonicOscillator1D(omega=1.0)
    dim = 1
    kT = 1.0
    mass = 1.0
    n_thermo = len(Qs)

    rng = np.random.default_rng(seed)
    q = np.array([0.5])
    p = rng.normal(0, np.sqrt(mass * kT), size=dim)
    xi = np.zeros(n_thermo)

    g_per_thermo = np.zeros((n_thermo, n_steps))
    g_total_series = np.zeros(n_steps)
    half_dt = 0.5 * dt
    grad_U = potential.gradient(q)

    for step in range(n_steps):
        for k in range(n_thermo):
            gk = g_func(xi[k])
            g_per_thermo[k, step] = gk
        g_total_series[step] = np.sum(g_per_thermo[:, step])

        # Velocity Verlet step
        kinetic = np.sum(p**2) / mass
        drive = kinetic - dim * kT
        xi_dot = np.array([drive / Q for Q in Qs])
        xi = xi + half_dt * xi_dot

        total_g = sum(g_func(xi[k]) for k in range(n_thermo))
        scale = np.clip(np.exp(-total_g * half_dt), 1e-10, 1e10)
        p = p * scale
        p = p - half_dt * grad_U
        q = q + dt * p / mass
        grad_U = potential.gradient(q)
        p = p - half_dt * grad_U
        total_g = sum(g_func(xi[k]) for k in range(n_thermo))
        scale = np.clip(np.exp(-total_g * half_dt), 1e-10, 1e10)
        p = p * scale

        kinetic = np.sum(p**2) / mass
        drive = kinetic - dim * kT
        xi_dot = np.array([drive / Q for Q in Qs])
        xi = xi + half_dt * xi_dot

    return g_total_series, g_per_thermo


def main():
    print("=" * 70)
    print("Lorentzian Decomposition: N=3 multi-scale thermostat PSDs")
    print("=" * 70)

    # Use moderate Q range where all thermostats are well-resolved
    Qs = [0.1, 1.0, 10.0]
    dt = 0.005
    n_steps = 2_000_000
    nperseg = 16384

    print(f"Running simulation: Qs={Qs}, {n_steps} steps, dt={dt}")
    g_total, g_per = run_and_collect_decomp(Qs, n_steps=n_steps, dt=dt, seed=42)

    # Compute PSDs
    freqs_total, psd_total = welch(g_total, fs=1.0/dt, nperseg=nperseg, noverlap=nperseg//2)

    per_thermo_data = []
    fits = {}
    for k in range(len(Qs)):
        fk, pk = welch(g_per[k], fs=1.0/dt, nperseg=nperseg, noverlap=nperseg//2)
        A, tau = fit_lorentzian(fk, pk)
        f_corner = 1.0 / (2.0 * np.pi * tau)
        fits[k] = {'A': A, 'tau': tau, 'Q': Qs[k], 'f_corner': f_corner}
        per_thermo_data.append((fk, pk))

        print(f"  Thermostat k={k}: Q={Qs[k]:.2f}, tau_fit={tau:.4f}, "
              f"f_corner={f_corner:.3f} Hz")
        print(f"    Expected f ~ 1/sqrt(Q) = {1.0/np.sqrt(Qs[k]):.3f} Hz")

    # --- PLOT ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = [plt.cm.tab10(i) for i in [2, 3, 4]]
    labels_q = [f'Q={q}' for q in Qs]

    # Panel (a): Individual PSDs + Lorentzian fits
    ax = axes[0]
    for k in range(len(Qs)):
        fk, pk = per_thermo_data[k]
        mask = fk > 0
        ax.loglog(fk[mask], pk[mask], color=colors[k], alpha=0.6, linewidth=1.0,
                  label=f'$g(\\xi_{{{k+1}}})$, Q={Qs[k]}')

        # Lorentzian fit overlay
        A, tau = fits[k]['A'], fits[k]['tau']
        f_fit = np.logspace(np.log10(fk[mask][0]), np.log10(fk[mask][-1]), 300)
        ax.loglog(f_fit, lorentzian(f_fit, A, tau), '--', color=colors[k],
                  linewidth=2, alpha=0.8)

    ax.set_xlabel('Frequency (Hz)', fontsize=14)
    ax.set_ylabel('Power Spectral Density', fontsize=14)
    ax.set_title('Individual Thermostat PSDs', fontsize=16)
    ax.legend(fontsize=12, loc='upper right')
    ax.tick_params(labelsize=12)
    ax.set_xlim([5e-3, 100])
    ax.grid(True, alpha=0.3, which='both')

    # Panel (b): Sum vs total, with 1/f reference
    ax = axes[1]
    mask = freqs_total > 0
    ax.loglog(freqs_total[mask], psd_total[mask], 'k-', linewidth=1.5,
              label='Measured $g_{\\mathrm{total}}$', alpha=0.7)

    # Sum of fitted Lorentzians
    f_theory = np.logspace(-2, 2, 500)
    psd_sum = np.zeros_like(f_theory)
    for k in range(len(Qs)):
        A, tau = fits[k]['A'], fits[k]['tau']
        lor_k = lorentzian(f_theory, A, tau)
        psd_sum += lor_k
        ax.loglog(f_theory, lor_k, ':', color=colors[k], linewidth=1, alpha=0.5)

    ax.loglog(f_theory, psd_sum, 'r-', linewidth=2.5,
              label='$\\sum_k$ Lorentzian$_k$', alpha=0.9)

    # 1/f reference
    f_ref = np.logspace(-1, 1.5, 100)
    # Normalize to match data at f=0.3 Hz
    idx_ref = np.argmin(np.abs(freqs_total[mask] - 0.3))
    psd_at_ref = psd_total[mask][idx_ref]
    ax.loglog(f_ref, psd_at_ref * 0.3 / f_ref, 'g--', linewidth=2,
              alpha=0.7, label='$1/f$ reference')

    ax.set_xlabel('Frequency (Hz)', fontsize=14)
    ax.set_ylabel('Power Spectral Density', fontsize=14)
    ax.set_title('Dutta-Horn: $\\sum$ Lorentzians $\\approx 1/f$', fontsize=16)
    ax.legend(fontsize=12, loc='upper right')
    ax.tick_params(labelsize=12)
    ax.set_xlim([5e-3, 100])
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    figpath = os.path.join(figdir, 'lorentzian_decomposition.png')
    fig.savefig(figpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved to {figpath}")

    # Save fit results
    fit_results = {str(k): {'A': float(v['A']), 'tau': float(v['tau']),
                             'Q': float(v['Q']), 'f_corner': float(v['f_corner'])}
                   for k, v in fits.items()}
    outdir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(outdir, 'lorentzian_fits.json'), 'w') as f:
        json.dump(fit_results, f, indent=2)


if __name__ == '__main__':
    main()
