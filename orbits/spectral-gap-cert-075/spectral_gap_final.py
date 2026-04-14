"""Spectral gap via direct simulation — vectorized for speed.

Runs multiple independent NH trajectories in parallel using numpy broadcasting.
Measures autocorrelation time tau_int and envelope decay rate gamma for each
(alpha, Q) combination.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time


def g_tanh_vec(xi, alpha):
    """Vectorized tanh damping."""
    return np.tanh(alpha * xi)

def g_losc_vec(xi):
    """Vectorized log-oscillator damping."""
    return 2.0 * xi / (1.0 + xi**2)

def g_linear_vec(xi):
    """Vectorized standard NH."""
    return xi


def run_nh_batch(omega, kT, Q, g_type, g_param, dt, n_steps, n_skip, n_traj, seed=42):
    """Run n_traj independent NH trajectories in parallel (vectorized).

    g_type: 'tanh', 'losc', or 'linear'
    g_param: alpha for tanh, ignored for losc/linear

    Returns q_samples: shape (n_traj, n_steps//n_skip)
    """
    rng = np.random.RandomState(seed)
    sigma_q = np.sqrt(kT / omega**2)
    sigma_p = np.sqrt(kT)
    sigma_xi = np.sqrt(kT / Q)

    # Initialize: shape (n_traj,)
    q = rng.randn(n_traj) * sigma_q
    p = rng.randn(n_traj) * sigma_p
    xi = rng.randn(n_traj) * sigma_xi

    n_samples = n_steps // n_skip
    q_out = np.zeros((n_traj, n_samples))

    omega2 = omega**2

    def g_eval(xi_val):
        if g_type == 'tanh':
            return np.tanh(g_param * xi_val)
        elif g_type == 'losc':
            return 2.0 * xi_val / (1.0 + xi_val**2)
        else:  # linear
            return xi_val

    sample_idx = 0
    for step in range(n_steps):
        # Half-step xi
        xi += 0.5 * dt * (p**2 - kT) / Q

        # Half-step p: friction + force
        g_val = g_eval(xi)
        scale = np.exp(-g_val * 0.5 * dt)
        np.clip(scale, 1e-10, 1e10, out=scale)
        p *= scale
        p -= 0.5 * dt * omega2 * q

        # Full-step q
        q += dt * p

        # Half-step p: force + friction
        p -= 0.5 * dt * omega2 * q
        g_val = g_eval(xi)
        scale = np.exp(-g_val * 0.5 * dt)
        np.clip(scale, 1e-10, 1e10, out=scale)
        p *= scale

        # Half-step xi
        xi += 0.5 * dt * (p**2 - kT) / Q

        if step % n_skip == 0:
            q_out[:, sample_idx] = q
            sample_idx += 1

    return q_out


def compute_acf_batch(q_batch, max_lag=None):
    """Compute ACF for each trajectory in batch, return mean ACF.

    q_batch: shape (n_traj, n_samples)
    """
    n_traj, n_samples = q_batch.shape
    if max_lag is None:
        max_lag = n_samples // 4

    acfs = np.zeros((n_traj, max_lag))
    for i in range(n_traj):
        x = q_batch[i] - np.mean(q_batch[i])
        var = np.var(q_batch[i])
        if var < 1e-15:
            acfs[i, :] = 1.0
            continue
        n = len(x)
        fft_x = np.fft.fft(x, n=2*n)
        acf_full = np.fft.ifft(fft_x * np.conj(fft_x)).real[:n] / (n * var)
        acfs[i, :max_lag] = acf_full[:max_lag]

    return np.mean(acfs, axis=0)


def compute_tau_int(acf):
    """Integrated autocorrelation time from ACF."""
    tau = 0.5
    for t in range(1, len(acf)):
        if acf[t] < 0.05:
            break
        tau += acf[t]
    return max(tau, 1.0)


def fit_envelope_decay(acf, dt_sample, max_fit_lag=None):
    """Fit exponential decay to ACF envelope."""
    n = len(acf)
    if max_fit_lag is None:
        max_fit_lag = min(n, 5000)

    acf_clip = acf[:max_fit_lag]
    abs_acf = np.abs(acf_clip)
    ts = np.arange(max_fit_lag) * dt_sample

    # Find local maxima for envelope
    peaks = []
    for i in range(1, len(abs_acf) - 1):
        if abs_acf[i] > abs_acf[i-1] and abs_acf[i] > abs_acf[i+1] and abs_acf[i] > 0.02:
            peaks.append(i)

    if len(peaks) < 3:
        # Fallback: use points where ACF > 0
        mask = (acf_clip > 0.02) & (ts > 0)
        if np.sum(mask) < 5:
            return 0.0, 0.0
        ts_fit = ts[mask]
        vals_fit = acf_clip[mask]
    else:
        peaks = np.array(peaks)
        ts_fit = ts[peaks]
        vals_fit = abs_acf[peaks]

    try:
        log_vals = np.log(vals_fit)
        coeffs = np.polyfit(ts_fit, log_vals, 1)
        gamma = -coeffs[0]
        predicted = np.polyval(coeffs, ts_fit)
        ss_res = np.sum((log_vals - predicted)**2)
        ss_tot = np.sum((log_vals - np.mean(log_vals))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return max(gamma, 0), r2
    except:
        return 0.0, 0.0


def main():
    print("=" * 80)
    print("SPECTRAL GAP — VECTORIZED DIRECT SIMULATION")
    print("=" * 80)

    fig_dir = "/Users/wujiewang/code/det-sampler/.worktrees/spectral-gap-cert-075/orbits/spectral-gap-cert-075/figures"
    os.makedirs(fig_dir, exist_ok=True)

    # Parameters
    omega = 1.0
    kT = 1.0
    dt = 0.01
    n_skip = 5
    dt_sample = dt * n_skip  # 0.05 time units
    n_steps = 2000000  # 2M steps = 20,000 time units per trajectory
    n_traj = 8  # parallel trajectories

    alphas = [0.25, 0.5, 0.75, 1.0, np.sqrt(2), 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    Qs = [0.1, 0.3, 1.0, 3.0, 10.0]

    all_results = {}
    total = len(Qs) * (len(alphas) + 1)
    count = 0

    for Q in Qs:
        print(f"\n{'='*60}")
        print(f"Q = {Q}   Kac prediction: α_opt = {np.sqrt(2.0/Q):.4f}")
        print(f"{'='*60}")

        for alpha in alphas:
            count += 1
            t0 = time.time()

            q_batch = run_nh_batch(omega, kT, Q, 'tanh', alpha,
                                   dt, n_steps, n_skip, n_traj)
            acf = compute_acf_batch(q_batch, max_lag=10000)
            tau = compute_tau_int(acf) * dt_sample
            gamma, r2 = fit_envelope_decay(acf, dt_sample)

            elapsed = time.time() - t0
            all_results[('tanh', alpha, Q)] = {
                'tau_int': tau, 'gamma': gamma, 'r2': r2, 'acf': acf
            }
            print(f"  [{count:3d}/{total}] tanh(α={alpha:5.3f}): "
                  f"τ={tau:7.2f}  γ={gamma:.4f} (R²={r2:.3f})  "
                  f"1/τ={1/tau:.4f}  ({elapsed:.1f}s)")

        # Log-oscillator
        count += 1
        t0 = time.time()
        q_batch = run_nh_batch(omega, kT, Q, 'losc', 0,
                               dt, n_steps, n_skip, n_traj)
        acf = compute_acf_batch(q_batch, max_lag=10000)
        tau = compute_tau_int(acf) * dt_sample
        gamma, r2 = fit_envelope_decay(acf, dt_sample)
        elapsed = time.time() - t0
        all_results[('losc', 2.0, Q)] = {
            'tau_int': tau, 'gamma': gamma, 'r2': r2, 'acf': acf
        }
        print(f"  [{count:3d}/{total}] log-osc(g'=2):     "
              f"τ={tau:7.2f}  γ={gamma:.4f} (R²={r2:.3f})  "
              f"1/τ={1/tau:.4f}  ({elapsed:.1f}s)")

    # Standard NH reference
    print(f"\n{'='*60}")
    print("Reference: Standard NH g(ξ)=ξ, Q=1 (non-ergodic for 1D HO)")
    print(f"{'='*60}")
    q_batch = run_nh_batch(omega, kT, 1.0, 'linear', 0,
                           dt, n_steps, n_skip, n_traj)
    acf = compute_acf_batch(q_batch, max_lag=10000)
    tau = compute_tau_int(acf) * dt_sample
    gamma, r2 = fit_envelope_decay(acf, dt_sample)
    all_results[('linear', 1.0, 1.0)] = {
        'tau_int': tau, 'gamma': gamma, 'r2': r2, 'acf': acf
    }
    print(f"  linear NH: τ={tau:.2f}  γ={gamma:.4f}  (non-ergodic => large τ or γ~0)")

    # ====== TABLES ======
    print("\n\n" + "=" * 110)
    print("TABLE 1: Autocorrelation time τ_int (time units) — lower is better")
    print("=" * 110)
    hdr = f"{'Q':>6s}"
    for a in alphas:
        hdr += f" {a:>6.2f}"
    hdr += f" {'losc':>6s} {'best_α':>6s} {'Kac':>6s}"
    print(hdr)
    print("-" * 110)

    for Q in Qs:
        line = f"{Q:6.1f}"
        best_tau, best_alpha = 1e10, 0
        for alpha in alphas:
            t = all_results[('tanh', alpha, Q)]['tau_int']
            line += f" {t:6.2f}"
            if t < best_tau:
                best_tau, best_alpha = t, alpha
        lt = all_results[('losc', 2.0, Q)]['tau_int']
        kac = np.sqrt(2.0 / Q)
        line += f" {lt:6.2f} {best_alpha:6.3f} {kac:6.3f}"
        print(line)

    print("\n\n" + "=" * 110)
    print("TABLE 2: Envelope decay rate γ — higher is better")
    print("=" * 110)
    hdr = f"{'Q':>6s}"
    for a in alphas:
        hdr += f" {a:>6.2f}"
    hdr += f" {'losc':>6s} {'best_α':>6s} {'Kac':>6s}"
    print(hdr)
    print("-" * 110)

    for Q in Qs:
        line = f"{Q:6.1f}"
        best_g, best_alpha = 0, 0
        for alpha in alphas:
            g = all_results[('tanh', alpha, Q)]['gamma']
            line += f" {g:6.4f}"
            if g > best_g:
                best_g, best_alpha = g, alpha
        lg = all_results[('losc', 2.0, Q)]['gamma']
        kac = np.sqrt(2.0 / Q)
        line += f" {lg:6.4f} {best_alpha:6.3f} {kac:6.3f}"
        print(line)

    # ====== KEY TABLE: log-osc vs tanh(2ξ) ======
    print("\n\n" + "=" * 80)
    print("TABLE 3: Log-osc vs tanh(2ξ) head-to-head (both have g'(0)=2)")
    print("=" * 80)
    print(f"{'Q':>6s}  {'τ_losc':>8s}  {'τ_tanh2':>8s}  {'speedup':>8s}  "
          f"{'γ_losc':>8s}  {'γ_tanh2':>8s}  {'γ_ratio':>8s}")
    print("-" * 65)
    for Q in Qs:
        tl = all_results[('losc', 2.0, Q)]['tau_int']
        tt = all_results[('tanh', 2.0, Q)]['tau_int']
        sp = tt / tl if tl > 0 else 0
        gl = all_results[('losc', 2.0, Q)]['gamma']
        gt = all_results[('tanh', 2.0, Q)]['gamma']
        gr = gl / gt if gt > 0 else float('inf')
        print(f"{Q:6.1f}  {tl:8.2f}  {tt:8.2f}  {sp:8.3f}  "
              f"{gl:8.4f}  {gt:8.4f}  {gr:8.3f}")

    # ====== OPTIMAL α TABLE ======
    print("\n\n" + "=" * 80)
    print("TABLE 4: Optimal α vs Kac prediction √(2/Q)")
    print("=" * 80)
    print(f"{'Q':>6s}  {'α_opt':>8s}  {'Kac':>8s}  {'τ_opt':>8s}  {'τ_losc':>8s}  {'losc_wins':>10s}")
    print("-" * 55)
    for Q in Qs:
        pairs = [(a, all_results[('tanh', a, Q)]['tau_int']) for a in alphas]
        best = min(pairs, key=lambda x: x[1])
        kac = np.sqrt(2.0 / Q)
        tl = all_results[('losc', 2.0, Q)]['tau_int']
        wins = "YES" if tl < best[1] else "no"
        print(f"{Q:6.1f}  {best[0]:8.3f}  {kac:8.3f}  {best[1]:8.2f}  {tl:8.2f}  {wins:>10s}")

    # ====== FIGURES ======
    generate_figures(all_results, alphas, Qs, fig_dir, dt_sample)

    return all_results


def generate_figures(results, alphas, Qs, fig_dir, dt_sample):
    """Generate figures."""

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # --- Figure 1: Two-panel: τ_int and γ vs α ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    for i, Q in enumerate(Qs):
        taus = [results[('tanh', a, Q)]['tau_int'] for a in alphas]
        ax.plot(alphas, taus, 'o-', color=colors[i], label=f'Q={Q}', ms=5, lw=1.5)
        kac = np.sqrt(2.0 / Q)
        ax.axvline(kac, color=colors[i], ls=':', alpha=0.4, lw=1)
        lt = results[('losc', 2.0, Q)]['tau_int']
        ax.plot(2.0, lt, 's', color=colors[i], ms=10,
                markeredgecolor='black', markeredgewidth=1.5, zorder=5)
    ax.set_xlabel(r"$\alpha$ = g'(0)", fontsize=13)
    ax.set_ylabel(r"$\tau_{\mathrm{int}}$ (time units)", fontsize=13)
    ax.set_title("(a) Integrated autocorrelation time", fontsize=12)
    ax.legend(fontsize=9, title="Q")
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i, Q in enumerate(Qs):
        gammas = [results[('tanh', a, Q)]['gamma'] for a in alphas]
        ax.plot(alphas, gammas, 'o-', color=colors[i], label=f'Q={Q}', ms=5, lw=1.5)
        kac = np.sqrt(2.0 / Q)
        ax.axvline(kac, color=colors[i], ls=':', alpha=0.4, lw=1)
        lg = results[('losc', 2.0, Q)]['gamma']
        ax.plot(2.0, lg, 's', color=colors[i], ms=10,
                markeredgecolor='black', markeredgewidth=1.5, zorder=5)
    ax.set_xlabel(r"$\alpha$ = g'(0)", fontsize=13)
    ax.set_ylabel(r"Decay rate $\gamma$", fontsize=13)
    ax.set_title("(b) ACF envelope decay rate", fontsize=12)
    ax.legend(fontsize=9, title="Q")
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    fig.suptitle("NH spectral gap: 1D HO ($\\omega$=1, kT=1)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/fig1_tau_gamma_vs_alpha.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig_dir}/fig1_tau_gamma_vs_alpha.png")
    plt.close()

    # --- Figure 2: ACF examples ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Select interesting cases
    cases = [
        ('tanh', 1.0, 1.0, r'tanh($\xi$), Q=1'),
        ('tanh', 2.0, 1.0, r'tanh($2\xi$), Q=1'),
        ('losc', 2.0, 1.0, r'log-osc, Q=1'),
        ('tanh', 1.0, 0.1, r'tanh($\xi$), Q=0.1'),
        ('tanh', 2.0, 0.1, r'tanh($2\xi$), Q=0.1'),
        ('losc', 2.0, 0.1, r'log-osc, Q=0.1'),
    ]

    for ax, (gtype, alpha, Q, title) in zip(axes.flat, cases):
        key = (gtype, alpha, Q)
        if key in results and 'acf' in results[key]:
            acf = results[key]['acf']
            ts = np.arange(len(acf)) * dt_sample
            ax.plot(ts[:4000], acf[:4000], lw=0.8)
            ax.axhline(0, color='k', lw=0.5)

            # Overlay envelope fit
            gamma = results[key]['gamma']
            if gamma > 0:
                env_t = ts[:4000]
                env = np.exp(-gamma * env_t)
                ax.plot(env_t, env, 'r--', lw=1, alpha=0.7, label=f'γ={gamma:.3f}')
                ax.plot(env_t, -env, 'r--', lw=1, alpha=0.7)
                ax.legend(fontsize=9)

        ax.set_xlabel('t')
        ax.set_ylabel('C(t)/C(0)')
        ax.set_title(title, fontsize=11)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Autocorrelation functions with envelope fits", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/fig2_acf_examples.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_dir}/fig2_acf_examples.png")
    plt.close()

    # --- Figure 3: log-osc vs tanh head-to-head ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, Q in zip(axes, [0.1, 1.0, 10.0]):
        for key, label, color in [
            (('tanh', 2.0, Q), r'tanh($2\xi$)', '#d62728'),
            (('losc', 2.0, Q), 'log-osc', '#1f77b4'),
        ]:
            if key in results and 'acf' in results[key]:
                acf = results[key]['acf']
                ts = np.arange(min(len(acf), 4000)) * dt_sample
                ax.plot(ts, acf[:len(ts)], label=label, color=color, lw=1.2)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel('t')
        ax.set_ylabel('C(t)/C(0)')
        ax.set_title(f'Q = {Q}')
        ax.legend(fontsize=10)
        ax.set_ylim(-0.8, 1.1)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Log-osc vs tanh(2ξ): both have g'(0)=2", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/fig3_losc_vs_tanh.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_dir}/fig3_losc_vs_tanh.png")
    plt.close()


if __name__ == "__main__":
    results = main()
