"""Final spectral gap study — lean and fast.

Uses hand-written Verlet integrator with carefully chosen parameters.
Focuses on the configurations that actually show differences.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import sys


def run_nh_nd(omega2, kT, Q, g_type, g_param, dt, n_steps, n_skip, seed=42):
    """Run d-dimensional NH with Verlet integrator. Return q samples.

    omega2: array of shape (d,) — squared frequencies
    Returns: q_samples of shape (d, n_steps//n_skip)
    """
    d = len(omega2)
    rng = np.random.RandomState(seed)

    sigma_q = np.sqrt(kT / omega2)
    q = rng.randn(d) * sigma_q
    p = rng.randn(d) * np.sqrt(kT)
    xi = rng.randn() * np.sqrt(kT / Q)

    n_out = n_steps // n_skip
    q_out = np.zeros((d, n_out))
    out_idx = 0

    half_dt = 0.5 * dt
    d_kT = d * kT

    for step in range(n_steps):
        # Half-step xi
        KE = np.dot(p, p)
        xi += half_dt * (KE - d_kT) / Q

        # Half-step p: friction + force
        if g_type == 'tanh':
            g = np.tanh(g_param * xi)
        elif g_type == 'losc':
            g = 2.0 * xi / (1.0 + xi * xi)
        else:
            g = xi

        s = np.exp(-g * half_dt)
        if s > 1e10: s = 1e10
        elif s < 1e-10: s = 1e-10
        p *= s
        p -= half_dt * omega2 * q

        # Full-step q
        q += dt * p

        # Half-step p: force + friction
        p -= half_dt * omega2 * q
        if g_type == 'tanh':
            g = np.tanh(g_param * xi)
        elif g_type == 'losc':
            g = 2.0 * xi / (1.0 + xi * xi)
        else:
            g = xi

        s = np.exp(-g * half_dt)
        if s > 1e10: s = 1e10
        elif s < 1e-10: s = 1e-10
        p *= s

        # Half-step xi
        KE = np.dot(p, p)
        xi += half_dt * (KE - d_kT) / Q

        if step % n_skip == 0:
            q_out[:, out_idx] = q
            out_idx += 1

    return q_out


def compute_acf(x, max_lag):
    """ACF via FFT."""
    x = x - np.mean(x)
    n = len(x)
    var = np.var(x)
    if var < 1e-15:
        return np.ones(max_lag)
    fft_x = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(fft_x * np.conj(fft_x)).real[:n] / (n * var)
    return acf[:max_lag]


def tau_int(acf, dt_s):
    """Integrated autocorrelation time."""
    tau = 0.5
    for t in range(1, len(acf)):
        if acf[t] < 0.05:
            break
        tau += acf[t]
    return max(tau, 1.0) * dt_s


def fit_env(acf, dt_s):
    """Fit exponential envelope to ACF peaks."""
    n = min(len(acf), 5000)
    a = np.abs(acf[:n])
    ts = np.arange(n) * dt_s

    peaks = [i for i in range(1, n-1)
             if a[i] > a[i-1] and a[i] > a[i+1] and a[i] > 0.01]

    if len(peaks) < 3:
        mask = (acf[:n] > 0.01) & (ts > 0)
        if np.sum(mask) < 5:
            return 0.0
        tf, vf = ts[mask], acf[:n][mask]
    else:
        peaks = np.array(peaks)
        tf, vf = ts[peaks], a[peaks]

    try:
        c = np.polyfit(tf, np.log(vf), 1)
        return max(-c[0], 0)
    except:
        return 0.0


def main():
    print("=" * 80)
    print("SPECTRAL GAP — FINAL STUDY")
    print("=" * 80)
    sys.stdout.flush()

    fig_dir = "/Users/wujiewang/code/det-sampler/.worktrees/spectral-gap-cert-075/orbits/spectral-gap-cert-075/figures"
    os.makedirs(fig_dir, exist_ok=True)

    kT = 1.0

    # ========== PART A: 1D HO at different kappa ==========
    # For 1D, omega^2 = omega_min^2 = 1/kappa^2
    # The slow mode period is 2*pi/omega_min = 2*pi*kappa
    # Need t_max >> period for ACF to show decay

    alphas = [0.25, 0.5, 0.75, 1.0, np.sqrt(2), 2.0, 3.0, 4.0, 6.0, 8.0]

    print("\n" + "=" * 80)
    print("PART A: 1D slow mode (ω² = 1/κ²) — varying κ")
    print("=" * 80)

    configs_1d = [
        # (kappa, Q, dt, n_steps, n_skip)
        (1,   1.0, 0.01, 500000, 5),
        (10,  1.0, 0.01, 500000, 5),
        (10,  0.1, 0.01, 500000, 5),
        (100, 1.0, 0.01, 1000000, 10),
        (100, 0.1, 0.01, 1000000, 10),
    ]

    results_1d = {}

    for kappa, Q, dt, n_steps, n_skip in configs_1d:
        omega2 = np.array([1.0 / kappa**2])
        dt_s = dt * n_skip
        max_lag = min(n_steps // n_skip // 2, 20000)
        n_runs = 3

        print(f"\n--- κ={kappa}, Q={Q}, ω={1/kappa:.4f} ---")
        sys.stdout.flush()

        for alpha in alphas:
            t0 = time.time()
            taus, gs = [], []
            for run in range(n_runs):
                qs = run_nh_nd(omega2, kT, Q, 'tanh', alpha, dt, n_steps, n_skip, seed=42+run*100)
                burn = qs.shape[1] // 10
                acf = compute_acf(qs[0, burn:], max_lag)
                taus.append(tau_int(acf, dt_s))
                gs.append(fit_env(acf, dt_s))
            tm, gm = np.mean(taus), np.mean(gs)
            elapsed = time.time() - t0
            results_1d[('tanh', alpha, kappa, Q)] = {'tau': tm, 'gamma': gm}
            print(f"  tanh(α={alpha:5.3f}): τ={tm:8.2f}  γ={gm:.5f}  ({elapsed:.1f}s)")
            sys.stdout.flush()

        # log-osc
        t0 = time.time()
        taus, gs = [], []
        for run in range(n_runs):
            qs = run_nh_nd(omega2, kT, Q, 'losc', 0, dt, n_steps, n_skip, seed=42+run*100)
            burn = qs.shape[1] // 10
            acf = compute_acf(qs[0, burn:], max_lag)
            taus.append(tau_int(acf, dt_s))
            gs.append(fit_env(acf, dt_s))
        tm, gm = np.mean(taus), np.mean(gs)
        elapsed = time.time() - t0
        results_1d[('losc', 2.0, kappa, Q)] = {'tau': tm, 'gamma': gm}
        print(f"  log-osc:           τ={tm:8.2f}  γ={gm:.5f}  ({elapsed:.1f}s)")
        sys.stdout.flush()

        # linear NH
        t0 = time.time()
        taus, gs = [], []
        for run in range(n_runs):
            qs = run_nh_nd(omega2, kT, Q, 'linear', 0, dt, n_steps, n_skip, seed=42+run*100)
            burn = qs.shape[1] // 10
            acf = compute_acf(qs[0, burn:], max_lag)
            taus.append(tau_int(acf, dt_s))
            gs.append(fit_env(acf, dt_s))
        tm, gm = np.mean(taus), np.mean(gs)
        elapsed = time.time() - t0
        results_1d[('linear', 1.0, kappa, Q)] = {'tau': tm, 'gamma': gm}
        print(f"  linear NH:         τ={tm:8.2f}  γ={gm:.5f}  ({elapsed:.1f}s)")
        sys.stdout.flush()

    # ========== PART B: Multi-D (d=5, kappa=10) ==========
    print("\n\n" + "=" * 80)
    print("PART B: d=5, κ=10")
    print("=" * 80)

    d = 5
    kappa = 10
    omega2 = np.logspace(-2*np.log10(kappa), 0, d)  # 0.01 to 1
    dt = 0.005
    n_steps = 1000000
    n_skip = 10
    dt_s = dt * n_skip
    max_lag = min(n_steps // n_skip // 2, 20000)
    n_runs = 3

    results_nd = {}

    for Q in [0.1, 1.0]:
        print(f"\n--- d={d}, κ={kappa}, Q={Q} ---")
        print(f"    Kac: α_opt = √(2d/Q) = {np.sqrt(2*d/Q):.3f}")
        sys.stdout.flush()

        for alpha in alphas:
            t0 = time.time()
            taus, gs = [], []
            for run in range(n_runs):
                qs = run_nh_nd(omega2, kT, Q, 'tanh', alpha, dt, n_steps, n_skip, seed=42+run*100)
                burn = qs.shape[1] // 10
                acf = compute_acf(qs[0, burn:], max_lag)
                taus.append(tau_int(acf, dt_s))
                gs.append(fit_env(acf, dt_s))
            tm, gm = np.mean(taus), np.mean(gs)
            elapsed = time.time() - t0
            results_nd[('tanh', alpha, d, kappa, Q)] = {'tau': tm, 'gamma': gm}
            print(f"  tanh(α={alpha:5.3f}): τ_slow={tm:8.2f}  γ={gm:.5f}  ({elapsed:.1f}s)")
            sys.stdout.flush()

        # log-osc
        t0 = time.time()
        taus, gs = [], []
        for run in range(n_runs):
            qs = run_nh_nd(omega2, kT, Q, 'losc', 0, dt, n_steps, n_skip, seed=42+run*100)
            burn = qs.shape[1] // 10
            acf = compute_acf(qs[0, burn:], max_lag)
            taus.append(tau_int(acf, dt_s))
            gs.append(fit_env(acf, dt_s))
        tm, gm = np.mean(taus), np.mean(gs)
        elapsed = time.time() - t0
        results_nd[('losc', 2.0, d, kappa, Q)] = {'tau': tm, 'gamma': gm}
        print(f"  log-osc:           τ_slow={tm:8.2f}  γ={gm:.5f}  ({elapsed:.1f}s)")
        sys.stdout.flush()

        # linear
        t0 = time.time()
        taus, gs = [], []
        for run in range(n_runs):
            qs = run_nh_nd(omega2, kT, Q, 'linear', 0, dt, n_steps, n_skip, seed=42+run*100)
            burn = qs.shape[1] // 10
            acf = compute_acf(qs[0, burn:], max_lag)
            taus.append(tau_int(acf, dt_s))
            gs.append(fit_env(acf, dt_s))
        tm, gm = np.mean(taus), np.mean(gs)
        elapsed = time.time() - t0
        results_nd[('linear', 1.0, d, kappa, Q)] = {'tau': tm, 'gamma': gm}
        print(f"  linear NH:         τ_slow={tm:8.2f}  γ={gm:.5f}  ({elapsed:.1f}s)")
        sys.stdout.flush()

    # ========== SUMMARY ==========
    print("\n\n" + "=" * 100)
    print("SUMMARY: 1D slow mode results")
    print("=" * 100)

    for kappa, Q, _, _, _ in configs_1d:
        print(f"\n  κ={kappa}, Q={Q}:")
        pairs = [(a, results_1d[('tanh', a, kappa, Q)]['tau']) for a in alphas]
        best_a, best_tau = min(pairs, key=lambda x: x[1])
        tl = results_1d[('losc', 2.0, kappa, Q)]['tau']
        tt2 = results_1d[('tanh', 2.0, kappa, Q)]['tau']
        tlin = results_1d[('linear', 1.0, kappa, Q)]['tau']
        kac = np.sqrt(2.0 / Q)
        print(f"    Best tanh: α={best_a:.3f} (τ={best_tau:.2f})  Kac={kac:.3f}")
        print(f"    log-osc: τ={tl:.2f}  tanh(2): τ={tt2:.2f}  linear: τ={tlin:.2f}")
        print(f"    losc speedup over tanh(2): {tt2/tl:.3f}x")
        if tlin > 0.01:
            print(f"    losc speedup over linear: {tlin/tl:.3f}x")

    print("\n\n" + "=" * 100)
    print("SUMMARY: d=5, κ=10 results")
    print("=" * 100)

    for Q in [0.1, 1.0]:
        print(f"\n  d=5, κ=10, Q={Q}:")
        pairs = [(a, results_nd[('tanh', a, 5, 10, Q)]['tau']) for a in alphas]
        best_a, best_tau = min(pairs, key=lambda x: x[1])
        tl = results_nd[('losc', 2.0, 5, 10, Q)]['tau']
        tt2 = results_nd[('tanh', 2.0, 5, 10, Q)]['tau']
        tlin = results_nd[('linear', 1.0, 5, 10, Q)]['tau']
        kac = np.sqrt(2*5/Q)
        print(f"    Best tanh: α={best_a:.3f} (τ={best_tau:.2f})  Kac={kac:.3f}")
        print(f"    log-osc: τ={tl:.2f}  tanh(2): τ={tt2:.2f}  linear: τ={tlin:.2f}")
        print(f"    losc speedup over tanh(2): {tt2/tl:.3f}x")

    # ========== FIGURES ==========
    generate_figures(results_1d, results_nd, alphas, configs_1d, fig_dir)

    return results_1d, results_nd


def generate_figures(R1d, Rnd, alphas, configs_1d, fig_dir):
    """Generate publication figures."""

    # Fig 1: τ vs α for different κ (1D)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (kappa, Q, _, _, _) in zip(axes, [(1, 1.0, 0, 0, 0), (10, 1.0, 0, 0, 0), (100, 1.0, 0, 0, 0)]):
        taus = [R1d.get(('tanh', a, kappa, Q), {}).get('tau', np.nan) for a in alphas]
        ax.plot(alphas, taus, 'bo-', ms=5, lw=1.5, label='tanh(αξ)')

        tl = R1d.get(('losc', 2.0, kappa, Q), {}).get('tau', np.nan)
        ax.axhline(tl, color='red', ls='--', lw=2, label=f'log-osc (τ={tl:.1f})')

        tlin = R1d.get(('linear', 1.0, kappa, Q), {}).get('tau', np.nan)
        ax.axhline(tlin, color='gray', ls=':', lw=2, label=f'linear (τ={tlin:.1f})')

        kac = np.sqrt(2.0 / Q)
        ax.axvline(kac, color='green', ls=':', alpha=0.5, label=f'Kac √(2/Q)={kac:.1f}')

        ax.set_xlabel(r"$\alpha$ = g'(0)", fontsize=12)
        ax.set_ylabel(r"$\tau_{\mathrm{int}}$", fontsize=12)
        ax.set_title(f"κ={kappa}, Q={Q}", fontsize=12)
        ax.legend(fontsize=8)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

    fig.suptitle("1D slow mode: autocorrelation time vs damping strength", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/fig1_1d_tau_vs_alpha.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig_dir}/fig1_1d_tau_vs_alpha.png")
    plt.close()

    # Fig 2: τ vs α for d=5, κ=10
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, Q in zip(axes, [0.1, 1.0]):
        taus = [Rnd.get(('tanh', a, 5, 10, Q), {}).get('tau', np.nan) for a in alphas]
        ax.plot(alphas, taus, 'bo-', ms=5, lw=1.5, label='tanh(αξ)')

        tl = Rnd.get(('losc', 2.0, 5, 10, Q), {}).get('tau', np.nan)
        ax.axhline(tl, color='red', ls='--', lw=2, label=f'log-osc (τ={tl:.1f})')

        tlin = Rnd.get(('linear', 1.0, 5, 10, Q), {}).get('tau', np.nan)
        ax.axhline(tlin, color='gray', ls=':', lw=2, label=f'linear (τ={tlin:.1f})')

        kac = np.sqrt(2*5/Q)
        ax.axvline(kac, color='green', ls=':', alpha=0.5, label=f'Kac √(2d/Q)={kac:.1f}')

        ax.set_xlabel(r"$\alpha$ = g'(0)", fontsize=12)
        ax.set_ylabel(r"$\tau_{\mathrm{int}}$ (slow mode)", fontsize=12)
        ax.set_title(f"d=5, κ=10, Q={Q}", fontsize=12)
        ax.legend(fontsize=8)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

    fig.suptitle("5D anisotropic HO: slow mode autocorrelation time", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/fig2_5d_tau_vs_alpha.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_dir}/fig2_5d_tau_vs_alpha.png")
    plt.close()

    # Fig 3: losc vs tanh(2) comparison across all configs
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    configs_labels = []
    speedups = []

    for kappa, Q, _, _, _ in configs_1d:
        tl = R1d.get(('losc', 2.0, kappa, Q), {}).get('tau', 999)
        tt = R1d.get(('tanh', 2.0, kappa, Q), {}).get('tau', 999)
        sp = tt / tl if tl > 0 else 0
        configs_labels.append(f"1D,κ={kappa},Q={Q}")
        speedups.append(sp)

    for Q in [0.1, 1.0]:
        tl = Rnd.get(('losc', 2.0, 5, 10, Q), {}).get('tau', 999)
        tt = Rnd.get(('tanh', 2.0, 5, 10, Q), {}).get('tau', 999)
        sp = tt / tl if tl > 0 else 0
        configs_labels.append(f"5D,κ=10,Q={Q}")
        speedups.append(sp)

    x = np.arange(len(configs_labels))
    bars = ax.bar(x, speedups, color=['#1f77b4' if s > 1 else '#d62728' for s in speedups])
    ax.axhline(1.0, color='k', ls='--', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(configs_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Speedup (τ_tanh2 / τ_losc)", fontsize=12)
    ax.set_title("Log-osc speedup over tanh(2ξ)", fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/fig3_losc_speedup.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_dir}/fig3_losc_speedup.png")
    plt.close()


if __name__ == "__main__":
    results = main()
