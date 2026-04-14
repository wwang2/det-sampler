"""Focused spectral-gap sweep: log-osc vs tanh(2x) for orbit 075.

1D harmonic oscillator, omega=1, kT=1.
Sweep alpha in {0.5, 1.0, sqrt(2), 2.0, 4.0}, Q in {0.1, 1.0, 10.0}.
For each (alpha, Q): run NH simulation, measure tau_int.
Key comparison: log-osc (g'(0)=2) vs tanh(2x) (g'(0)=2).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import json


def run_nh_1d(omega2, kT, Q, g_type, g_param, dt, n_steps, n_skip, seed=42):
    """Run 1D NH trajectory. Returns q samples."""
    rng = np.random.RandomState(seed)
    q = rng.randn() * np.sqrt(kT / omega2)
    p = rng.randn() * np.sqrt(kT)
    xi = rng.randn() * np.sqrt(kT / Q)

    n_out = n_steps // n_skip
    q_out = np.empty(n_out)
    out_idx = 0
    half_dt = 0.5 * dt

    for step in range(n_steps):
        # Half-step xi
        xi += half_dt * (p * p - kT) / Q

        # Half-step p: friction + force
        if g_type == 'tanh':
            g = np.tanh(g_param * xi)
        elif g_type == 'losc':
            g = 2.0 * xi / (1.0 + xi * xi)
        else:
            g = xi

        s = np.exp(-g * half_dt)
        s = min(max(s, 1e-10), 1e10)
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
        s = min(max(s, 1e-10), 1e10)
        p *= s

        # Half-step xi
        xi += half_dt * (p * p - kT) / Q

        if step % n_skip == 0:
            q_out[out_idx] = q
            out_idx += 1

    return q_out


def compute_acf(x, max_lag):
    """ACF via FFT."""
    x = x - np.mean(x)
    n = len(x)
    var = np.var(x)
    if var < 1e-15:
        return np.ones(max_lag)
    fft_x = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(fft_x * np.conj(fft_x)).real[:n] / (n * var)
    return acf[:max_lag]


def tau_int(acf, dt_s):
    """Integrated autocorrelation time with Sokal window."""
    tau = 0.5
    for t in range(1, len(acf)):
        if acf[t] < 0.05:
            break
        tau += acf[t]
    return max(tau, 1.0) * dt_s


def main():
    print("=" * 80)
    print("ORBIT 075: Spectral Gap Sweep — log-osc vs tanh")
    print("=" * 80)

    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    omega2 = 1.0
    kT = 1.0
    dt = 0.005
    n_skip = 10
    dt_s = dt * n_skip
    n_steps = 2_000_000  # 10,000 time units
    n_runs = 5  # average over seeds
    max_lag = 10000

    alphas = [0.5, 1.0, np.sqrt(2), 2.0, 4.0]
    Qs = [0.1, 1.0, 10.0]

    results = {}

    for Q in Qs:
        print(f"\n--- Q = {Q} ---")

        for alpha in alphas:
            t0 = time.time()
            taus = []
            for run in range(n_runs):
                qs = run_nh_1d(omega2, kT, Q, 'tanh', alpha, dt, n_steps, n_skip, seed=42 + run * 137)
                burn = len(qs) // 10
                acf = compute_acf(qs[burn:], max_lag)
                taus.append(tau_int(acf, dt_s))
            tm = np.mean(taus)
            ts = np.std(taus)
            elapsed = time.time() - t0
            results[('tanh', alpha, Q)] = {'tau': tm, 'tau_std': ts, 'taus': taus}
            print(f"  tanh(a={alpha:5.3f}): tau = {tm:7.2f} +/- {ts:5.2f}  ({elapsed:.1f}s)")

        # log-osc
        t0 = time.time()
        taus = []
        for run in range(n_runs):
            qs = run_nh_1d(omega2, kT, Q, 'losc', 0, dt, n_steps, n_skip, seed=42 + run * 137)
            burn = len(qs) // 10
            acf = compute_acf(qs[burn:], max_lag)
            taus.append(tau_int(acf, dt_s))
        tm = np.mean(taus)
        ts = np.std(taus)
        elapsed = time.time() - t0
        results[('losc', 2.0, Q)] = {'tau': tm, 'tau_std': ts, 'taus': taus}
        print(f"  log-osc (g'=2):    tau = {tm:7.2f} +/- {ts:5.2f}  ({elapsed:.1f}s)")

    # ====== KEY TABLE ======
    print("\n\n" + "=" * 80)
    print("KEY TABLE: log-osc vs tanh(2x) — both have g'(0) = 2")
    print("=" * 80)
    print(f"{'Q':>6}  {'tau_losc':>10}  {'tau_tanh2':>10}  {'speedup':>10}  {'interpretation':>20}")
    print("-" * 65)

    speedups = []
    for Q in Qs:
        tl = results[('losc', 2.0, Q)]['tau']
        tt = results[('tanh', 2.0, Q)]['tau']
        sp = tt / tl if tl > 0 else 0
        speedups.append(sp)
        interp = "losc FASTER" if sp > 1 else "tanh FASTER"
        print(f"{Q:6.1f}  {tl:10.2f}  {tt:10.2f}  {sp:10.3f}x  {interp:>20}")

    mean_speedup = np.mean(speedups)
    print(f"\nMean speedup: {mean_speedup:.3f}x")

    # ====== FULL TABLE ======
    print("\n\n" + "=" * 80)
    print("FULL TABLE: tau_int for all (alpha, Q)")
    print("=" * 80)
    header = f"{'Q':>6}"
    for a in alphas:
        header += f"  {'a=' + f'{a:.2f}':>8}"
    header += f"  {'losc':>8}  {'best_a':>8}"
    print(header)
    print("-" * (6 + 10 * len(alphas) + 20))

    for Q in Qs:
        line = f"{Q:6.1f}"
        best_tau, best_a = 1e10, 0
        for a in alphas:
            t = results[('tanh', a, Q)]['tau']
            line += f"  {t:8.2f}"
            if t < best_tau:
                best_tau, best_a = t, a
        tl = results[('losc', 2.0, Q)]['tau']
        line += f"  {tl:8.2f}  {best_a:8.3f}"
        print(line)

    # ====== FIGURES ======

    # Fig 1: tau vs alpha for each Q, with log-osc marked
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors_q = {0.1: '#1f77b4', 1.0: '#ff7f0e', 10.0: '#2ca02c'}

    for ax, Q in zip(axes, Qs):
        taus_tanh = [results[('tanh', a, Q)]['tau'] for a in alphas]
        ax.plot(alphas, taus_tanh, 'o-', color='#1f77b4', ms=7, lw=2, label=r'tanh($\alpha\xi$)')

        tl = results[('losc', 2.0, Q)]['tau']
        ax.axhline(tl, color='#d62728', ls='--', lw=2, label=f'log-osc (tau={tl:.1f})')

        # Mark tanh(2x) specifically
        tt2 = results[('tanh', 2.0, Q)]['tau']
        ax.plot(2.0, tt2, 's', color='#d62728', ms=10, markeredgecolor='black',
                markeredgewidth=1.5, zorder=5, label=f'tanh(2x) (tau={tt2:.1f})')

        kac = np.sqrt(2.0 / Q)
        ax.axvline(kac, color='green', ls=':', alpha=0.5, lw=1.5,
                   label=r'$\sqrt{2/Q}$' + f'={kac:.2f}')

        ax.set_xlabel(r"$\alpha$ = g'(0)", fontsize=12)
        ax.set_ylabel(r"$\tau_{\mathrm{int}}$", fontsize=12)
        ax.set_title(f"Q = {Q}", fontsize=13)
        ax.legend(fontsize=8)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

    fig.suptitle(r"1D HO ($\omega$=1, kT=1): autocorrelation time vs damping", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/sweep_tau_vs_alpha.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig_dir}/sweep_tau_vs_alpha.png")
    plt.close()

    # Fig 2: Speedup bar chart
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    x = np.arange(len(Qs))
    bars = ax.bar(x, speedups, color=['#2ca02c' if s > 1 else '#d62728' for s in speedups],
                  edgecolor='black', lw=1)
    ax.axhline(1.0, color='k', ls='--', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Q={Q}' for Q in Qs], fontsize=11)
    ax.set_ylabel(r"Speedup ($\tau_{\mathrm{tanh2}} / \tau_{\mathrm{losc}}$)", fontsize=12)
    ax.set_title("Log-osc speedup over tanh(2x)\n[both have g'(0)=2]", fontsize=13)
    for i, sp in enumerate(speedups):
        ax.text(i, sp + 0.02, f'{sp:.2f}x', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(speedups) * 1.3)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/sweep_speedup_bar.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_dir}/sweep_speedup_bar.png")
    plt.close()

    # Fig 3: ACF comparison at Q=1
    Q_show = 1.0
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    for run in range(1):  # just one seed for the plot
        qs_losc = run_nh_1d(omega2, kT, Q_show, 'losc', 0, dt, n_steps, n_skip, seed=42)
        qs_tanh = run_nh_1d(omega2, kT, Q_show, 'tanh', 2.0, dt, n_steps, n_skip, seed=42)
        burn = len(qs_losc) // 10
        acf_losc = compute_acf(qs_losc[burn:], 3000)
        acf_tanh = compute_acf(qs_tanh[burn:], 3000)
        ts = np.arange(3000) * dt_s
        ax.plot(ts, acf_losc, color='#d62728', lw=1.2, label='log-osc')
        ax.plot(ts, acf_tanh, color='#1f77b4', lw=1.2, label=r'tanh($2\xi$)')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('t (time units)', fontsize=12)
    ax.set_ylabel('C(t)/C(0)', fontsize=12)
    ax.set_title(f"ACF comparison: Q={Q_show}, g'(0)=2", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/sweep_acf_comparison.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_dir}/sweep_acf_comparison.png")
    plt.close()

    # Save numeric results
    out = {}
    for key, val in results.items():
        k_str = f"{key[0]}_a{key[1]:.3f}_Q{key[2]:.1f}"
        out[k_str] = {'tau': val['tau'], 'tau_std': val['tau_std']}
    with open(os.path.join(fig_dir, '..', 'results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: results.json")

    print(f"\n{'='*80}")
    print(f"METRIC (tau_tanh2 / tau_losc at Q=1): {speedups[1]:.3f}")
    print(f"{'='*80}")

    return results, speedups


if __name__ == "__main__":
    main()
