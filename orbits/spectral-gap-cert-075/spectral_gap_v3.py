"""Spectral gap via direct simulation — robust autocorrelation analysis.

Instead of diagonalizing the truncated Liouville operator (which suffers from
severe truncation artifacts for non-self-adjoint operators), we compute the
spectral gap empirically from long NH trajectories.

The spectral gap lambda_2 governs the exponential decay of autocorrelations:
  C(t) ~ exp(-lambda_2 * t) * cos(omega_osc * t)

We extract lambda_2 by fitting the autocorrelation envelope.

This approach:
1. Is numerically robust (no ill-conditioned eigenvector matrices)
2. Directly measures what we care about (mixing rate)
3. Can handle arbitrary g(xi) functions
4. Validates against the Hermite-basis eigenvalue analysis
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import time


def g_tanh(alpha):
    """Tanh damping: g(xi) = tanh(alpha*xi), g'(0) = alpha."""
    return lambda xi: np.tanh(alpha * xi)

def g_losc(xi):
    """Log-oscillator: g(xi) = 2xi/(1+xi^2), g'(0) = 2."""
    return 2.0 * xi / (1.0 + xi**2)

def g_linear(xi):
    """Standard NH: g(xi) = xi."""
    return xi


def run_nh_trajectory(omega=1.0, kT=1.0, Q=1.0, g_func=None,
                      dt=0.005, n_steps=2000000, n_skip=5, seed=None):
    """Run NH dynamics for 1D HO, return time series of (q, p, xi).

    Uses velocity-Verlet splitting (Martyna et al. 1996):
      1. half-step xi
      2. half-step p (friction rescaling + force kick)
      3. full-step q
      4. half-step p (force kick + friction rescaling)
      5. half-step xi
    """
    if seed is not None:
        np.random.seed(seed)

    sigma_q = np.sqrt(kT / omega**2)
    sigma_p = np.sqrt(kT)
    sigma_xi = np.sqrt(kT / Q)

    q = np.random.randn() * sigma_q
    p = np.random.randn() * sigma_p
    xi = np.random.randn() * sigma_xi

    n_samples = n_steps // n_skip
    qs = np.zeros(n_samples)
    ps = np.zeros(n_samples)
    xis = np.zeros(n_samples)

    for step in range(n_steps):
        # Half-step xi
        xi += 0.5 * dt * (p**2 - kT) / Q

        # Half-step p: friction rescaling + force kick
        g_val = g_func(xi)
        scale = np.exp(-g_val * 0.5 * dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p *= scale
        p -= 0.5 * dt * omega**2 * q

        # Full-step q
        q += dt * p

        # Half-step p: force kick + friction rescaling
        p -= 0.5 * dt * omega**2 * q
        g_val = g_func(xi)
        scale = np.exp(-g_val * 0.5 * dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p *= scale

        # Half-step xi
        xi += 0.5 * dt * (p**2 - kT) / Q

        if step % n_skip == 0:
            idx = step // n_skip
            qs[idx] = q
            ps[idx] = p
            xis[idx] = xi

    return qs, ps, xis


def compute_acf(x, max_lag=None):
    """Compute normalized autocorrelation function via FFT."""
    x = x - np.mean(x)
    n = len(x)
    if max_lag is None:
        max_lag = n // 2
    var = np.var(x)
    if var < 1e-15:
        return np.ones(max_lag)
    fft_x = np.fft.fft(x, n=2*n)
    acf_full = np.fft.ifft(fft_x * np.conj(fft_x)).real[:n] / (n * var)
    return acf_full[:max_lag]


def compute_tau_int(acf):
    """Integrated autocorrelation time from ACF using initial positive sequence."""
    tau = 0.5  # C(0)/2 = 0.5
    for t in range(1, len(acf)):
        if acf[t] < 0.05:  # stop when ACF drops below threshold
            break
        tau += acf[t]
    return max(tau, 1.0)


def fit_decay_rate(acf, dt_sample, max_fit_lag=None):
    """Extract decay rate by fitting the ACF envelope.

    Fit: |C(t)| envelope ~ A * exp(-gamma * t)

    Returns gamma (decay rate) and the fit quality.
    """
    n = len(acf)
    if max_fit_lag is None:
        max_fit_lag = min(n, 2000)

    ts = np.arange(max_fit_lag) * dt_sample
    acf_clip = acf[:max_fit_lag]

    # Find envelope by taking local maxima of |C(t)|
    abs_acf = np.abs(acf_clip)

    # Use running maximum to get envelope
    from scipy.signal import argrelextrema
    # Find local maxima
    max_idx = argrelextrema(abs_acf, np.greater, order=5)[0]
    if len(max_idx) < 3:
        # Not enough maxima, use all positive values
        mask = acf_clip > 0.01
        if np.sum(mask) < 5:
            return 0.0, 0.0
        ts_fit = ts[mask]
        vals_fit = acf_clip[mask]
    else:
        ts_fit = ts[max_idx]
        vals_fit = abs_acf[max_idx]

    # Filter out very small values
    keep = vals_fit > 0.01
    if np.sum(keep) < 3:
        return 0.0, 0.0
    ts_fit = ts_fit[keep]
    vals_fit = vals_fit[keep]

    # Fit log(envelope) = log(A) - gamma * t
    try:
        log_vals = np.log(vals_fit)
        coeffs = np.polyfit(ts_fit, log_vals, 1)
        gamma = -coeffs[0]
        # R^2 quality
        predicted = np.polyval(coeffs, ts_fit)
        ss_res = np.sum((log_vals - predicted)**2)
        ss_tot = np.sum((log_vals - np.mean(log_vals))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return gamma, r2
    except:
        return 0.0, 0.0


def main():
    print("=" * 80)
    print("SPECTRAL GAP via DIRECT SIMULATION — Autocorrelation Analysis")
    print("=" * 80)

    fig_dir = "/Users/wujiewang/code/det-sampler/.worktrees/spectral-gap-cert-075/orbits/spectral-gap-cert-075/figures"
    os.makedirs(fig_dir, exist_ok=True)

    omega = 1.0
    kT = 1.0
    dt = 0.005
    n_skip = 5
    dt_sample = dt * n_skip  # 0.025 time units between samples
    n_steps = 4000000  # 4M steps = 20000 time units
    n_runs = 5  # average over 5 independent runs for robustness

    alphas = [0.25, 0.5, 0.75, 1.0, np.sqrt(2), 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    Qs = [0.1, 0.3, 1.0, 3.0, 10.0]

    all_results = {}

    total_combos = len(Qs) * (len(alphas) + 1)  # +1 for log-osc
    combo_idx = 0

    for Q in Qs:
        print(f"\n{'='*60}")
        print(f"Q = {Q}   (Kac prediction: α_opt = {np.sqrt(2.0/Q):.4f})")
        print(f"{'='*60}")

        for alpha in alphas:
            combo_idx += 1
            t0 = time.time()

            taus = []
            gammas = []

            for run in range(n_runs):
                qs, ps, xis = run_nh_trajectory(
                    omega=omega, kT=kT, Q=Q, g_func=g_tanh(alpha),
                    dt=dt, n_steps=n_steps, n_skip=n_skip, seed=42+run*1000
                )

                acf = compute_acf(qs, max_lag=20000)
                tau = compute_tau_int(acf)
                gamma, r2 = fit_decay_rate(acf, dt_sample, max_fit_lag=10000)

                taus.append(tau * dt_sample)
                gammas.append(gamma)

            tau_mean = np.mean(taus)
            tau_std = np.std(taus)
            gamma_mean = np.mean(gammas)
            gamma_std = np.std(gammas)

            elapsed = time.time() - t0
            all_results[('tanh', alpha, Q)] = {
                'tau_int': tau_mean, 'tau_std': tau_std,
                'gamma': gamma_mean, 'gamma_std': gamma_std,
                'rate_from_tau': 1.0/tau_mean if tau_mean > 0 else 0
            }
            print(f"  [{combo_idx:3d}/{total_combos}] tanh(α={alpha:5.3f}): "
                  f"τ_int={tau_mean:7.2f}±{tau_std:.2f}  "
                  f"γ={gamma_mean:.4f}±{gamma_std:.4f}  "
                  f"1/τ={1/tau_mean:.4f}  ({elapsed:.1f}s)")

        # Log-oscillator
        combo_idx += 1
        t0 = time.time()
        taus = []
        gammas = []
        for run in range(n_runs):
            qs, ps, xis = run_nh_trajectory(
                omega=omega, kT=kT, Q=Q, g_func=g_losc,
                dt=dt, n_steps=n_steps, n_skip=n_skip, seed=42+run*1000
            )
            acf = compute_acf(qs, max_lag=20000)
            tau = compute_tau_int(acf)
            gamma, r2 = fit_decay_rate(acf, dt_sample, max_fit_lag=10000)
            taus.append(tau * dt_sample)
            gammas.append(gamma)

        tau_mean = np.mean(taus)
        tau_std = np.std(taus)
        gamma_mean = np.mean(gammas)
        gamma_std = np.std(gammas)
        elapsed = time.time() - t0
        all_results[('losc', 2.0, Q)] = {
            'tau_int': tau_mean, 'tau_std': tau_std,
            'gamma': gamma_mean, 'gamma_std': gamma_std,
            'rate_from_tau': 1.0/tau_mean if tau_mean > 0 else 0
        }
        print(f"  [{combo_idx:3d}/{total_combos}] log-osc(g'=2):     "
              f"τ_int={tau_mean:7.2f}±{tau_std:.2f}  "
              f"γ={gamma_mean:.4f}±{gamma_std:.4f}  "
              f"1/τ={1/tau_mean:.4f}  ({elapsed:.1f}s)")

    # Also run linear g (standard NH) for reference
    print(f"\n{'='*60}")
    print("Standard NH: g(ξ) = ξ (linear, known non-ergodic for 1D HO)")
    print(f"{'='*60}")
    for Q in [1.0]:
        taus = []
        gammas = []
        for run in range(n_runs):
            qs, ps, xis = run_nh_trajectory(
                omega=omega, kT=kT, Q=Q, g_func=g_linear,
                dt=dt, n_steps=n_steps, n_skip=n_skip, seed=42+run*1000
            )
            acf = compute_acf(qs, max_lag=20000)
            tau = compute_tau_int(acf)
            gamma, r2 = fit_decay_rate(acf, dt_sample, max_fit_lag=10000)
            taus.append(tau * dt_sample)
            gammas.append(gamma)
        tau_mean = np.mean(taus)
        gamma_mean = np.mean(gammas)
        all_results[('linear', 1.0, Q)] = {
            'tau_int': tau_mean, 'tau_std': np.std(taus),
            'gamma': gamma_mean, 'gamma_std': np.std(gammas),
            'rate_from_tau': 1.0/tau_mean if tau_mean > 0 else 0
        }
        print(f"  linear(Q={Q}): τ_int={tau_mean:.2f}  γ={gamma_mean:.4f}")

    # ====== SUMMARY TABLES ======
    print("\n\n" + "=" * 100)
    print("TABLE 1: Integrated autocorrelation time τ_int (in time units)")
    print("=" * 100)
    header = f"{'Q':>6s}"
    for a in alphas:
        header += f"  {a:>6.2f}"
    header += f"  {'losc':>6s}  {'best_α':>6s}  {'Kac':>6s}"
    print(header)
    print("-" * len(header))

    for Q in Qs:
        line = f"{Q:6.1f}"
        best_tau = 1e10
        best_alpha = 0
        for alpha in alphas:
            tau = all_results[('tanh', alpha, Q)]['tau_int']
            line += f"  {tau:6.2f}"
            if tau < best_tau:
                best_tau = tau
                best_alpha = alpha
        losc_tau = all_results[('losc', 2.0, Q)]['tau_int']
        kac = np.sqrt(2.0 / Q)
        line += f"  {losc_tau:6.2f}  {best_alpha:6.3f}  {kac:6.3f}"
        print(line)

    print("\n\n" + "=" * 100)
    print("TABLE 2: Envelope decay rate γ (from ACF envelope fit)")
    print("=" * 100)
    header = f"{'Q':>6s}"
    for a in alphas:
        header += f"  {a:>6.2f}"
    header += f"  {'losc':>6s}  {'best_α':>6s}  {'Kac':>6s}"
    print(header)
    print("-" * len(header))

    for Q in Qs:
        line = f"{Q:6.1f}"
        best_gamma = 0
        best_alpha = 0
        for alpha in alphas:
            gamma = all_results[('tanh', alpha, Q)]['gamma']
            line += f"  {gamma:6.4f}" if gamma > 0 else f"  {'~0':>6s}"
            if gamma > best_gamma:
                best_gamma = gamma
                best_alpha = alpha
        losc_gamma = all_results[('losc', 2.0, Q)]['gamma']
        kac = np.sqrt(2.0 / Q)
        line += f"  {losc_gamma:6.4f}  {best_alpha:6.3f}  {kac:6.3f}"
        print(line)

    # ====== KEY COMPARISON: log-osc vs tanh(2ξ) ======
    print("\n\n" + "=" * 80)
    print("TABLE 3: Log-osc vs tanh(2ξ) — Does log-osc have faster mixing?")
    print("=" * 80)
    print(f"{'Q':>6s}  {'τ_losc':>8s}  {'τ_tanh2':>8s}  {'speedup':>8s}  {'γ_losc':>8s}  {'γ_tanh2':>8s}  {'γ_ratio':>8s}")
    print("-" * 65)
    for Q in Qs:
        t_losc = all_results[('losc', 2.0, Q)]['tau_int']
        t_tanh = all_results[('tanh', 2.0, Q)]['tau_int']
        speedup = t_tanh / t_losc if t_losc > 0 else 0
        g_losc_val = all_results[('losc', 2.0, Q)]['gamma']
        g_tanh_val = all_results[('tanh', 2.0, Q)]['gamma']
        g_ratio = g_losc_val / g_tanh_val if g_tanh_val > 0 else float('inf')
        print(f"{Q:6.1f}  {t_losc:8.2f}  {t_tanh:8.2f}  {speedup:8.3f}  "
              f"{g_losc_val:8.4f}  {g_tanh_val:8.4f}  {g_ratio:8.3f}")

    # ====== OPTIMAL α per Q ======
    print("\n\n" + "=" * 80)
    print("TABLE 4: Optimal α (minimizes τ_int) vs Kac prediction √(2/Q)")
    print("=" * 80)
    print(f"{'Q':>6s}  {'α_opt':>8s}  {'Kac':>8s}  {'τ_opt':>8s}  {'τ_losc':>8s}  {'losc better?':>12s}")
    print("-" * 55)
    for Q in Qs:
        taus_tanh = [(alpha, all_results[('tanh', alpha, Q)]['tau_int']) for alpha in alphas]
        best = min(taus_tanh, key=lambda x: x[1])
        kac = np.sqrt(2.0 / Q)
        t_losc = all_results[('losc', 2.0, Q)]['tau_int']
        better = "YES" if t_losc < best[1] else "no"
        print(f"{Q:6.1f}  {best[0]:8.3f}  {kac:8.3f}  {best[1]:8.2f}  {t_losc:8.2f}  {better:>12s}")

    # ====== FIGURES ======
    generate_figures(all_results, alphas, Qs, fig_dir, dt, n_skip, n_steps, omega, kT)

    return all_results


def generate_figures(results, alphas, Qs, fig_dir, dt, n_skip, n_steps, omega, kT):
    """Generate publication-quality figures."""

    # Figure 1: τ_int vs α for each Q
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Panel A: τ_int
    ax = axes[0]
    for i, Q in enumerate(Qs):
        taus = [results[('tanh', a, Q)]['tau_int'] for a in alphas]
        ax.plot(alphas, taus, 'o-', color=colors[i], label=f'Q={Q}', markersize=5, linewidth=1.5)
        # Mark Kac prediction
        kac = np.sqrt(2.0 / Q)
        ax.axvline(kac, color=colors[i], linestyle=':', alpha=0.4, linewidth=1)
        # Mark log-osc
        losc_tau = results[('losc', 2.0, Q)]['tau_int']
        ax.plot(2.0, losc_tau, 's', color=colors[i], markersize=10,
                markeredgecolor='black', markeredgewidth=1.5, zorder=5)

    ax.set_xlabel(r"$\alpha$ = g'(0)", fontsize=13)
    ax.set_ylabel(r"$\tau_{\mathrm{int}}$ (time units)", fontsize=13)
    ax.set_title("(a) Autocorrelation time vs damping strength", fontsize=12)
    ax.legend(fontsize=9, title="Thermostat mass Q")
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # Panel B: Decay rate γ
    ax = axes[1]
    for i, Q in enumerate(Qs):
        gammas = [results[('tanh', a, Q)]['gamma'] for a in alphas]
        ax.plot(alphas, gammas, 'o-', color=colors[i], label=f'Q={Q}', markersize=5, linewidth=1.5)
        kac = np.sqrt(2.0 / Q)
        ax.axvline(kac, color=colors[i], linestyle=':', alpha=0.4, linewidth=1)
        losc_gamma = results[('losc', 2.0, Q)]['gamma']
        ax.plot(2.0, losc_gamma, 's', color=colors[i], markersize=10,
                markeredgecolor='black', markeredgewidth=1.5, zorder=5)

    ax.set_xlabel(r"$\alpha$ = g'(0)", fontsize=13)
    ax.set_ylabel(r"Decay rate $\gamma$", fontsize=13)
    ax.set_title("(b) ACF envelope decay rate", fontsize=12)
    ax.legend(fontsize=9, title="Thermostat mass Q")
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    fig.suptitle("Spectral gap analysis: 1D Harmonic Oscillator (ω=1, kT=1)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/tau_and_gamma_vs_alpha.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig_dir}/tau_and_gamma_vs_alpha.png")
    plt.close()

    # Figure 2: Example ACF traces for select cases
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    cases = [
        (g_tanh(1.0), 1.0, r'tanh($\xi$), Q=1'),
        (g_tanh(2.0), 1.0, r'tanh($2\xi$), Q=1'),
        (g_losc, 1.0, r'log-osc, Q=1'),
        (g_tanh(1.0), 0.1, r'tanh($\xi$), Q=0.1'),
        (g_tanh(np.sqrt(2)), 0.1, r'tanh($\sqrt{2}\xi$), Q=0.1'),
        (g_losc, 0.1, r'log-osc, Q=0.1'),
    ]

    for ax, (gfunc, Q, title) in zip(axes.flat, cases):
        qs, _, _ = run_nh_trajectory(omega=omega, kT=kT, Q=Q, g_func=gfunc,
                                     dt=dt, n_steps=n_steps, n_skip=n_skip, seed=42)
        acf = compute_acf(qs, max_lag=4000)
        ts = np.arange(len(acf)) * dt * n_skip
        ax.plot(ts, acf, linewidth=0.8)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_xlabel('t')
        ax.set_ylabel('C(t)/C(0)')
        ax.set_title(title, fontsize=11)
        ax.set_ylim(-0.5, 1.1)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Autocorrelation functions (1D HO, ω=1, kT=1)", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/acf_examples.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_dir}/acf_examples.png")
    plt.close()

    # Figure 3: Log-osc vs tanh(2ξ) head-to-head
    fig, axes = plt.subplots(1, len(Qs), figsize=(4*len(Qs), 4))
    if len(Qs) == 1:
        axes = [axes]

    for ax, Q in zip(axes, Qs):
        for gfunc, label, color in [(g_tanh(2.0), r'tanh($2\xi$)', '#d62728'),
                                     (g_losc, 'log-osc', '#1f77b4')]:
            qs, _, _ = run_nh_trajectory(omega=omega, kT=kT, Q=Q, g_func=gfunc,
                                         dt=dt, n_steps=n_steps, n_skip=n_skip, seed=42)
            acf = compute_acf(qs, max_lag=4000)
            ts = np.arange(len(acf)) * dt * n_skip
            ax.plot(ts, acf, label=label, color=color, linewidth=1.2)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_xlabel('t')
        ax.set_ylabel('C(t)/C(0)')
        ax.set_title(f'Q = {Q}')
        ax.legend(fontsize=9)
        ax.set_ylim(-0.5, 1.1)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Log-osc vs tanh: autocorrelation comparison (g'(0)=2)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/losc_vs_tanh_acf.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_dir}/losc_vs_tanh_acf.png")
    plt.close()


if __name__ == "__main__":
    results = main()
