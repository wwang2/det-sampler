"""Spectral-gap sweep v2: stress the thermostat with harder problems.

Key insight: for 1D HO with omega=1, kT=1, the system mixes too fast
and all thermostats look equivalent. We need:
1. Condition number kappa >> 1 (slow mode)
2. Or measure the actual ACF decay rate (not just tau_int which bottoms out at ~1)

Strategy:
- Use the VECTORIZED batch approach (8 trajectories in parallel)
- Measure BOTH tau_int AND envelope decay rate gamma
- Use kappa in {1, 10, 50} to stress-test
- Focus on Q=1 (standard) and Q=0.1 (tight coupling)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import json
import sys


def run_nh_batch(omega2, kT, Q, g_type, g_param, dt, n_steps, n_skip, n_traj, seed=42):
    """Run n_traj independent 1D NH trajectories in parallel."""
    rng = np.random.RandomState(seed)
    sigma_q = np.sqrt(kT / omega2)
    sigma_p = np.sqrt(kT)

    q = rng.randn(n_traj) * sigma_q
    p = rng.randn(n_traj) * sigma_p
    xi = rng.randn(n_traj) * np.sqrt(kT / Q)

    n_out = n_steps // n_skip
    q_out = np.zeros((n_traj, n_out))
    out_idx = 0
    half_dt = 0.5 * dt

    for step in range(n_steps):
        xi += half_dt * (p * p - kT) / Q

        if g_type == 'tanh':
            g = np.tanh(g_param * xi)
        elif g_type == 'losc':
            g = 2.0 * xi / (1.0 + xi * xi)
        else:
            g = xi

        s = np.exp(-g * half_dt)
        np.clip(s, 1e-10, 1e10, out=s)
        p *= s
        p -= half_dt * omega2 * q

        q += dt * p

        p -= half_dt * omega2 * q
        if g_type == 'tanh':
            g = np.tanh(g_param * xi)
        elif g_type == 'losc':
            g = 2.0 * xi / (1.0 + xi * xi)
        else:
            g = xi
        s = np.exp(-g * half_dt)
        np.clip(s, 1e-10, 1e10, out=s)
        p *= s

        xi += half_dt * (p * p - kT) / Q

        if step % n_skip == 0:
            q_out[:, out_idx] = q
            out_idx += 1

    return q_out


def compute_acf_batch(q_batch, max_lag):
    """Compute mean ACF over batch of trajectories."""
    n_traj, n_samples = q_batch.shape
    acfs = np.zeros((n_traj, max_lag))
    for i in range(n_traj):
        x = q_batch[i] - np.mean(q_batch[i])
        var = np.var(q_batch[i])
        if var < 1e-15:
            acfs[i] = 1.0
            continue
        n = len(x)
        fft_x = np.fft.fft(x, n=2 * n)
        acf = np.fft.ifft(fft_x * np.conj(fft_x)).real[:n] / (n * var)
        acfs[i, :max_lag] = acf[:max_lag]
    return np.mean(acfs, axis=0)


def tau_int(acf, dt_s):
    """Integrated autocorrelation time."""
    tau = 0.5
    for t in range(1, len(acf)):
        if acf[t] < 0.02:
            break
        tau += acf[t]
    return max(tau, 1.0) * dt_s


def fit_gamma(acf, dt_s):
    """Fit exponential decay rate to ACF envelope."""
    n = min(len(acf), 5000)
    a = np.abs(acf[:n])
    ts = np.arange(n) * dt_s
    peaks = [i for i in range(1, n - 1)
             if a[i] > a[i - 1] and a[i] > a[i + 1] and a[i] > 0.01]
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
    print("=" * 90)
    print("ORBIT 075 v2: Spectral Gap Sweep — stress test with kappa >> 1")
    print("=" * 90)
    sys.stdout.flush()

    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    kT = 1.0
    n_traj = 8

    alphas = [0.5, 1.0, np.sqrt(2), 2.0, 4.0]
    # Configurations: (kappa, Q, dt, n_steps, n_skip)
    configs = [
        (1,   1.0,  0.005, 2_000_000, 10),   # easy baseline
        (1,   0.1,  0.005, 2_000_000, 10),   # tight coupling
        (10,  1.0,  0.005, 4_000_000, 10),   # kappa=10, harder
        (10,  0.1,  0.005, 4_000_000, 10),   # kappa=10, tight
        (50,  1.0,  0.002, 10_000_000, 20),  # kappa=50, hard
        (50,  0.1,  0.002, 10_000_000, 20),  # kappa=50, tight
    ]

    all_results = {}
    acf_store = {}  # for plotting

    for kappa, Q, dt, n_steps, n_skip in configs:
        omega2 = 1.0 / kappa**2
        dt_s = dt * n_skip
        max_lag = min(n_steps // n_skip // 4, 20000)

        print(f"\n{'='*70}")
        print(f"kappa={kappa}, Q={Q}, omega={1/kappa:.4f}, dt={dt}, "
              f"T_total={n_steps*dt:.0f} time units")
        print(f"{'='*70}")
        sys.stdout.flush()

        for alpha in alphas:
            t0 = time.time()
            q_batch = run_nh_batch(omega2, kT, Q, 'tanh', alpha,
                                   dt, n_steps, n_skip, n_traj)
            burn = q_batch.shape[1] // 10
            acf = compute_acf_batch(q_batch[:, burn:], max_lag)
            tau = tau_int(acf, dt_s)
            gamma = fit_gamma(acf, dt_s)
            elapsed = time.time() - t0
            key = ('tanh', alpha, kappa, Q)
            all_results[key] = {'tau': tau, 'gamma': gamma}
            acf_store[key] = (acf[:min(5000, len(acf))], dt_s)
            print(f"  tanh(a={alpha:5.3f}): tau={tau:8.2f}  gamma={gamma:.5f}  "
                  f"1/tau={1/tau:.5f}  ({elapsed:.1f}s)")
            sys.stdout.flush()

        # log-osc
        t0 = time.time()
        q_batch = run_nh_batch(omega2, kT, Q, 'losc', 0,
                               dt, n_steps, n_skip, n_traj)
        burn = q_batch.shape[1] // 10
        acf = compute_acf_batch(q_batch[:, burn:], max_lag)
        tau = tau_int(acf, dt_s)
        gamma = fit_gamma(acf, dt_s)
        elapsed = time.time() - t0
        key = ('losc', 2.0, kappa, Q)
        all_results[key] = {'tau': tau, 'gamma': gamma}
        acf_store[key] = (acf[:min(5000, len(acf))], dt_s)
        print(f"  log-osc (g'=2):    tau={tau:8.2f}  gamma={gamma:.5f}  "
              f"1/tau={1/tau:.5f}  ({elapsed:.1f}s)")
        sys.stdout.flush()

        # linear NH
        t0 = time.time()
        q_batch = run_nh_batch(omega2, kT, Q, 'linear', 0,
                               dt, n_steps, n_skip, n_traj)
        burn = q_batch.shape[1] // 10
        acf = compute_acf_batch(q_batch[:, burn:], max_lag)
        tau = tau_int(acf, dt_s)
        gamma = fit_gamma(acf, dt_s)
        elapsed = time.time() - t0
        key = ('linear', 1.0, kappa, Q)
        all_results[key] = {'tau': tau, 'gamma': gamma}
        acf_store[key] = (acf[:min(5000, len(acf))], dt_s)
        print(f"  linear NH:         tau={tau:8.2f}  gamma={gamma:.5f}  "
              f"1/tau={1/tau:.5f}  ({elapsed:.1f}s)")
        sys.stdout.flush()

    # ====== SUMMARY TABLES ======
    print("\n\n" + "=" * 90)
    print("SUMMARY: log-osc vs tanh(2x) head-to-head [both g'(0)=2]")
    print("=" * 90)
    print(f"{'kappa':>6} {'Q':>5}  {'tau_losc':>9} {'tau_tanh2':>10} {'speedup':>9}  "
          f"{'gamma_losc':>11} {'gamma_tanh2':>12} {'gamma_ratio':>12}")
    print("-" * 85)

    metric_vals = []
    for kappa, Q, _, _, _ in configs:
        tl = all_results[('losc', 2.0, kappa, Q)]['tau']
        tt = all_results[('tanh', 2.0, kappa, Q)]['tau']
        sp = tt / tl if tl > 0 else 0
        gl = all_results[('losc', 2.0, kappa, Q)]['gamma']
        gt = all_results[('tanh', 2.0, kappa, Q)]['gamma']
        gr = gl / gt if gt > 0 else float('inf')
        metric_vals.append(sp)
        print(f"{kappa:6d} {Q:5.1f}  {tl:9.2f} {tt:10.2f} {sp:9.3f}x  "
              f"{gl:11.5f} {gt:12.5f} {gr:12.3f}")

    print(f"\nMean speedup: {np.mean(metric_vals):.3f}x")

    # Optimal alpha table
    print("\n\n" + "=" * 90)
    print("OPTIMAL alpha per config")
    print("=" * 90)
    print(f"{'kappa':>6} {'Q':>5}  {'best_a':>7} {'tau_best':>9} {'tau_losc':>9} "
          f"{'losc_rank':>10} {'Kac':>6}")
    print("-" * 60)
    for kappa, Q, _, _, _ in configs:
        pairs = [(a, all_results[('tanh', a, kappa, Q)]['tau']) for a in alphas]
        best_a, best_tau = min(pairs, key=lambda x: x[1])
        tl = all_results[('losc', 2.0, kappa, Q)]['tau']
        kac = np.sqrt(2.0 / Q)
        rank = "BEST" if tl <= best_tau else f"+{(tl/best_tau - 1)*100:.0f}%"
        print(f"{kappa:6d} {Q:5.1f}  {best_a:7.3f} {best_tau:9.2f} {tl:9.2f} "
              f"{rank:>10} {kac:6.2f}")

    # ====== FIGURES ======

    # Fig A: tau vs alpha, panel per (kappa, Q)
    n_configs = len(configs)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (kappa, Q, _, _, _) in zip(axes.flat, configs):
        taus = [all_results[('tanh', a, kappa, Q)]['tau'] for a in alphas]
        ax.plot(alphas, taus, 'bo-', ms=6, lw=1.5, label=r'tanh($\alpha\xi$)')

        tl = all_results[('losc', 2.0, kappa, Q)]['tau']
        ax.axhline(tl, color='red', ls='--', lw=2, label=f'log-osc ({tl:.1f})')

        tlin = all_results[('linear', 1.0, kappa, Q)]['tau']
        ax.axhline(tlin, color='gray', ls=':', lw=1.5, label=f'linear ({tlin:.1f})')

        kac = np.sqrt(2.0 / Q)
        ax.axvline(kac, color='green', ls=':', alpha=0.5, lw=1.5)

        ax.set_xlabel(r"$\alpha$", fontsize=11)
        ax.set_ylabel(r"$\tau_{\mathrm{int}}$", fontsize=11)
        ax.set_title(f"$\\kappa$={kappa}, Q={Q}", fontsize=12)
        ax.legend(fontsize=7)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

    fig.suptitle(r"$\tau_{\mathrm{int}}$ vs $\alpha$ — 1D HO slow mode", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/v2_tau_vs_alpha.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig_dir}/v2_tau_vs_alpha.png")
    plt.close()

    # Fig B: ACF comparison losc vs tanh(2) for kappa=50, Q=1
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (kappa, Q) in zip(axes, [(1, 1.0), (10, 1.0), (50, 1.0)]):
        for gtype, label, color in [
            (('losc', 2.0, kappa, Q), 'log-osc', '#d62728'),
            (('tanh', 2.0, kappa, Q), r'tanh($2\xi$)', '#1f77b4'),
            (('linear', 1.0, kappa, Q), 'linear NH', 'gray'),
        ]:
            if gtype in acf_store:
                acf, dt_s = acf_store[gtype]
                ts = np.arange(len(acf)) * dt_s
                ax.plot(ts, acf, color=color, lw=1, label=label)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel('t', fontsize=11)
        ax.set_ylabel('C(t)/C(0)', fontsize=11)
        ax.set_title(f"$\\kappa$={kappa}, Q={Q}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, 1.05)

    fig.suptitle("ACF comparison: log-osc vs tanh(2x) vs linear", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/v2_acf_comparison.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_dir}/v2_acf_comparison.png")
    plt.close()

    # Fig C: Speedup bar chart across all configs
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    labels = [f"k={k},Q={Q}" for k, Q, _, _, _ in configs]
    x = np.arange(len(labels))
    colors = ['#2ca02c' if s > 1 else '#d62728' for s in metric_vals]
    bars = ax.bar(x, metric_vals, color=colors, edgecolor='black', lw=1)
    ax.axhline(1.0, color='k', ls='--', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(r"Speedup $\tau_{\mathrm{tanh2}} / \tau_{\mathrm{losc}}$", fontsize=12)
    ax.set_title("Log-osc speedup over tanh(2x) [g'(0)=2]", fontsize=13)
    for i, v in enumerate(metric_vals):
        ax.text(i, v + 0.02, f'{v:.2f}x', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylim(0, max(metric_vals) * 1.3 if max(metric_vals) > 0 else 2)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/v2_speedup_bar.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_dir}/v2_speedup_bar.png")
    plt.close()

    # Save results
    out = {}
    for key, val in all_results.items():
        k_str = f"{key[0]}_a{key[1]:.3f}_k{key[2]}_Q{key[3]:.1f}"
        out[k_str] = val
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_v2.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("Saved: results_v2.json")

    return all_results


if __name__ == "__main__":
    main()
