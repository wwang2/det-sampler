"""Fast final sweep: use batch vectorization, moderate parameters.

Focus on the required sweep: alpha x Q, 1D HO.
Also kappa=10 to check if differences emerge at higher condition number.
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
    """Vectorized batch: n_traj parallel 1D trajectories."""
    rng = np.random.RandomState(seed)
    q = rng.randn(n_traj) * np.sqrt(kT / omega2)
    p = rng.randn(n_traj) * np.sqrt(kT)
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
    tau = 0.5
    for t in range(1, len(acf)):
        if acf[t] < 0.02:
            break
        tau += acf[t]
    return max(tau, 1.0) * dt_s


def fit_gamma(acf, dt_s):
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
    print("=" * 80)
    print("ORBIT 075: Final Spectral Gap Sweep")
    print("=" * 80)
    sys.stdout.flush()

    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    kT = 1.0
    n_traj = 16

    alphas_req = [0.5, 1.0, np.sqrt(2), 2.0, 4.0]
    Qs_req = [0.1, 1.0, 10.0]

    # ====== PART 1: Required sweep (kappa=1) ======
    print("\n" + "=" * 70)
    print("PART 1: Required sweep -- kappa=1, alpha x Q")
    print("=" * 70)

    dt = 0.005
    n_skip = 5
    dt_s = dt * n_skip
    n_steps = 1_000_000
    max_lag = 8000

    R1 = {}
    acf_store = {}

    for Q in Qs_req:
        print(f"\n  Q = {Q}")
        sys.stdout.flush()
        for alpha in alphas_req:
            t0 = time.time()
            qb = run_nh_batch(1.0, kT, Q, 'tanh', alpha, dt, n_steps, n_skip, n_traj)
            burn = qb.shape[1] // 10
            acf = compute_acf_batch(qb[:, burn:], max_lag)
            t = tau_int(acf, dt_s)
            g = fit_gamma(acf, dt_s)
            R1[('tanh', alpha, Q)] = {'tau': t, 'gamma': g}
            acf_store[('tanh', alpha, 1, Q)] = (acf[:3000], dt_s)
            print(f"    tanh(a={alpha:5.3f}): tau={t:7.2f}  gamma={g:.5f}  ({time.time()-t0:.1f}s)")
            sys.stdout.flush()

        t0 = time.time()
        qb = run_nh_batch(1.0, kT, Q, 'losc', 0, dt, n_steps, n_skip, n_traj)
        burn = qb.shape[1] // 10
        acf = compute_acf_batch(qb[:, burn:], max_lag)
        t = tau_int(acf, dt_s)
        g = fit_gamma(acf, dt_s)
        R1[('losc', 2.0, Q)] = {'tau': t, 'gamma': g}
        acf_store[('losc', 2.0, 1, Q)] = (acf[:3000], dt_s)
        print(f"    log-osc:          tau={t:7.2f}  gamma={g:.5f}  ({time.time()-t0:.1f}s)")
        sys.stdout.flush()

    # ====== PART 2: kappa=10 (harder) ======
    print("\n" + "=" * 70)
    print("PART 2: kappa=10 -- stress test")
    print("=" * 70)

    dt = 0.005
    n_skip = 5
    dt_s = dt * n_skip
    n_steps = 2_000_000
    max_lag = 15000

    R2 = {}
    for Q in [0.1, 1.0]:
        print(f"\n  kappa=10, Q={Q}")
        sys.stdout.flush()
        for alpha in alphas_req:
            t0 = time.time()
            qb = run_nh_batch(0.01, kT, Q, 'tanh', alpha, dt, n_steps, n_skip, n_traj)
            burn = qb.shape[1] // 10
            acf = compute_acf_batch(qb[:, burn:], max_lag)
            t = tau_int(acf, dt_s)
            g = fit_gamma(acf, dt_s)
            R2[('tanh', alpha, Q)] = {'tau': t, 'gamma': g}
            acf_store[('tanh', alpha, 10, Q)] = (acf[:5000], dt_s)
            print(f"    tanh(a={alpha:5.3f}): tau={t:8.2f}  gamma={g:.5f}  ({time.time()-t0:.1f}s)")
            sys.stdout.flush()

        t0 = time.time()
        qb = run_nh_batch(0.01, kT, Q, 'losc', 0, dt, n_steps, n_skip, n_traj)
        burn = qb.shape[1] // 10
        acf = compute_acf_batch(qb[:, burn:], max_lag)
        t = tau_int(acf, dt_s)
        g = fit_gamma(acf, dt_s)
        R2[('losc', 2.0, Q)] = {'tau': t, 'gamma': g}
        acf_store[('losc', 2.0, 10, Q)] = (acf[:5000], dt_s)
        print(f"    log-osc:          tau={t:8.2f}  gamma={g:.5f}  ({time.time()-t0:.1f}s)")
        sys.stdout.flush()

    # ====== TABLES ======
    print("\n\n" + "=" * 80)
    print("TABLE 1: kappa=1 -- tau_int (lower = better)")
    print("=" * 80)
    hdr = f"{'Q':>6}"
    for a in alphas_req:
        hdr += f"  a={a:5.2f}"
    hdr += f"  {'losc':>8}"
    print(hdr)
    print("-" * 70)
    for Q in Qs_req:
        line = f"{Q:6.1f}"
        for a in alphas_req:
            line += f" {R1[('tanh', a, Q)]['tau']:8.2f}"
        line += f" {R1[('losc', 2.0, Q)]['tau']:8.2f}"
        print(line)

    print("\n\n" + "=" * 80)
    print("TABLE 2: kappa=10 -- tau_int")
    print("=" * 80)
    hdr = f"{'Q':>6}"
    for a in alphas_req:
        hdr += f"  a={a:5.2f}"
    hdr += f"  {'losc':>8}"
    print(hdr)
    print("-" * 70)
    for Q in [0.1, 1.0]:
        line = f"{Q:6.1f}"
        for a in alphas_req:
            line += f" {R2[('tanh', a, Q)]['tau']:8.2f}"
        line += f" {R2[('losc', 2.0, Q)]['tau']:8.2f}"
        print(line)

    print("\n\n" + "=" * 80)
    print("KEY RESULT: log-osc vs tanh(2x) [g'(0)=2]")
    print("=" * 80)
    print(f"{'config':>20} {'tau_losc':>9} {'tau_tanh2':>10} {'speedup':>9}")
    print("-" * 55)

    speedups = []
    configs_labels = []
    for Q in Qs_req:
        tl = R1[('losc', 2.0, Q)]['tau']
        tt = R1[('tanh', 2.0, Q)]['tau']
        sp = tt / tl if tl > 0 else 0
        speedups.append(sp)
        configs_labels.append(f"k=1,Q={Q}")
        print(f"{'k=1,Q='+str(Q):>20} {tl:9.2f} {tt:10.2f} {sp:9.3f}x")

    for Q in [0.1, 1.0]:
        tl = R2[('losc', 2.0, Q)]['tau']
        tt = R2[('tanh', 2.0, Q)]['tau']
        sp = tt / tl if tl > 0 else 0
        speedups.append(sp)
        configs_labels.append(f"k=10,Q={Q}")
        print(f"{'k=10,Q='+str(Q):>20} {tl:9.2f} {tt:10.2f} {sp:9.3f}x")

    mean_sp = np.mean(speedups)
    print(f"\nMean speedup: {mean_sp:.3f}x")
    print(f"Median speedup: {np.median(speedups):.3f}x")

    # ====== FIGURES ======

    # Fig 1: tau vs alpha, 3 panels for Q at kappa=1
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, Q in zip(axes, Qs_req):
        taus = [R1[('tanh', a, Q)]['tau'] for a in alphas_req]
        ax.plot(alphas_req, taus, 'bo-', ms=7, lw=2, label=r'tanh($\alpha\xi$)')
        tl = R1[('losc', 2.0, Q)]['tau']
        ax.axhline(tl, color='red', ls='--', lw=2, label=f'log-osc')
        ax.plot(2.0, R1[('tanh', 2.0, Q)]['tau'], 's', color='orange', ms=10,
                markeredgecolor='black', markeredgewidth=1.5, zorder=5,
                label=r'tanh($2\xi$)')
        kac = np.sqrt(2.0 / Q)
        ax.axvline(kac, color='green', ls=':', alpha=0.5, lw=1.5,
                   label=r'$\sqrt{2/Q}$')
        ax.set_xlabel(r"$\alpha$ = g'(0)", fontsize=12)
        ax.set_ylabel(r"$\tau_{\mathrm{int}}$", fontsize=12)
        ax.set_title(f"Q = {Q}", fontsize=13)
        ax.legend(fontsize=8)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
    fig.suptitle(r"1D HO ($\kappa$=1): $\tau_{\mathrm{int}}$ vs damping strength",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/final_tau_vs_alpha_k1.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig_dir}/final_tau_vs_alpha_k1.png")
    plt.close()

    # Fig 2: tau vs alpha at kappa=10
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, Q in zip(axes, [0.1, 1.0]):
        taus = [R2[('tanh', a, Q)]['tau'] for a in alphas_req]
        ax.plot(alphas_req, taus, 'bo-', ms=7, lw=2, label=r'tanh($\alpha\xi$)')
        tl = R2[('losc', 2.0, Q)]['tau']
        ax.axhline(tl, color='red', ls='--', lw=2, label=f'log-osc')
        ax.plot(2.0, R2[('tanh', 2.0, Q)]['tau'], 's', color='orange', ms=10,
                markeredgecolor='black', markeredgewidth=1.5, zorder=5,
                label=r'tanh($2\xi$)')
        kac = np.sqrt(2.0 / Q)
        ax.axvline(kac, color='green', ls=':', alpha=0.5, lw=1.5,
                   label=r'$\sqrt{2/Q}$')
        ax.set_xlabel(r"$\alpha$ = g'(0)", fontsize=12)
        ax.set_ylabel(r"$\tau_{\mathrm{int}}$", fontsize=12)
        ax.set_title(f"$\\kappa$=10, Q={Q}", fontsize=13)
        ax.legend(fontsize=8)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
    fig.suptitle(r"1D HO ($\kappa$=10): $\tau_{\mathrm{int}}$ vs damping strength",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/final_tau_vs_alpha_k10.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_dir}/final_tau_vs_alpha_k10.png")
    plt.close()

    # Fig 3: Speedup bar chart
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    x = np.arange(len(configs_labels))
    colors = ['#2ca02c' if s > 1 else '#d62728' for s in speedups]
    ax.bar(x, speedups, color=colors, edgecolor='black', lw=1)
    ax.axhline(1.0, color='k', ls='--', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(configs_labels, fontsize=10)
    ax.set_ylabel(r"$\tau_{\mathrm{tanh2}} / \tau_{\mathrm{losc}}$", fontsize=12)
    ax.set_title("Log-osc speedup over tanh(2x) [g'(0)=2]", fontsize=13)
    for i, v in enumerate(speedups):
        ax.text(i, v + 0.02, f'{v:.2f}x', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylim(0, max(max(speedups) * 1.3, 1.5))
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/final_speedup_bar.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_dir}/final_speedup_bar.png")
    plt.close()

    # Fig 4: ACF comparison at kappa=10, Q=1
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (kappa, Q) in zip(axes, [(1, 1.0), (10, 1.0)]):
        for key, label, color in [
            (('losc', 2.0, kappa, Q), 'log-osc', '#d62728'),
            (('tanh', 2.0, kappa, Q), r'tanh($2\xi$)', '#1f77b4'),
        ]:
            if key in acf_store:
                acf, dts = acf_store[key]
                ts = np.arange(len(acf)) * dts
                ax.plot(ts, acf, color=color, lw=1.2, label=label)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel('t', fontsize=11)
        ax.set_ylabel('C(t)/C(0)', fontsize=11)
        ax.set_title(f"kappa={kappa}, Q={Q}", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    fig.suptitle("ACF: log-osc vs tanh(2x)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/final_acf_comparison.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_dir}/final_acf_comparison.png")
    plt.close()

    # Save results
    out = {
        'speedups': dict(zip(configs_labels, [float(s) for s in speedups])),
        'mean_speedup': float(mean_sp),
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_final.json'), 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print("Saved: results_final.json")

    print(f"\n{'='*80}")
    print(f"FINAL METRIC: mean tau_tanh2/tau_losc = {mean_sp:.3f}")
    print(f"{'='*80}")

    return R1, R2, speedups


if __name__ == "__main__":
    main()
