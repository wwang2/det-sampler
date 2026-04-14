"""Spectral gap study — fast version.

Uses scipy.integrate.solve_ivp for trajectory integration (C-compiled ODE solver),
which is orders of magnitude faster than a Python for-loop.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time


def nh_rhs_factory(omega, kT, Q, g_type, g_param):
    """Return RHS function for NH dynamics."""
    omega2 = omega**2

    if g_type == 'tanh':
        def rhs(t, y):
            q, p, xi = y
            g = np.tanh(g_param * xi)
            dq = p
            dp = -omega2 * q - g * p
            dxi = (p**2 - kT) / Q
            return [dq, dp, dxi]
    elif g_type == 'losc':
        def rhs(t, y):
            q, p, xi = y
            g = 2.0 * xi / (1.0 + xi**2)
            dq = p
            dp = -omega2 * q - g * p
            dxi = (p**2 - kT) / Q
            return [dq, dp, dxi]
    elif g_type == 'linear':
        def rhs(t, y):
            q, p, xi = y
            dq = p
            dp = -omega2 * q - xi * p
            dxi = (p**2 - kT) / Q
            return [dq, dp, dxi]
    return rhs


def run_trajectory_ivp(omega, kT, Q, g_type, g_param, t_max=500, dt_out=0.05, seed=42):
    """Run one NH trajectory using solve_ivp (fast C backend)."""
    rng = np.random.RandomState(seed)
    sigma_q = np.sqrt(kT / omega**2)
    sigma_p = np.sqrt(kT)
    sigma_xi = np.sqrt(kT / Q)

    y0 = [rng.randn() * sigma_q, rng.randn() * sigma_p, rng.randn() * sigma_xi]
    t_eval = np.arange(0, t_max, dt_out)

    rhs = nh_rhs_factory(omega, kT, Q, g_type, g_param)

    sol = solve_ivp(rhs, [0, t_max], y0, method='RK45',
                    t_eval=t_eval, rtol=1e-8, atol=1e-10, max_step=0.05)

    if not sol.success:
        print(f"  WARNING: solve_ivp failed: {sol.message}")
        return None, None, None, None

    return sol.t, sol.y[0], sol.y[1], sol.y[2]  # t, q, p, xi


def compute_acf(x, max_lag=None):
    """Normalized ACF via FFT."""
    x = x - np.mean(x)
    n = len(x)
    if max_lag is None:
        max_lag = n // 4
    var = np.var(x)
    if var < 1e-15:
        return np.ones(max_lag)
    fft_x = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(fft_x * np.conj(fft_x)).real[:n] / (n * var)
    return acf[:max_lag]


def compute_tau_int(acf, dt_sample):
    """Integrated autocorrelation time."""
    tau = 0.5
    for t in range(1, len(acf)):
        if acf[t] < 0.05:
            break
        tau += acf[t]
    return max(tau, 1.0) * dt_sample


def fit_envelope(acf, dt_sample, max_fit=None):
    """Fit exponential envelope to ACF."""
    n = len(acf) if max_fit is None else min(len(acf), max_fit)
    abs_acf = np.abs(acf[:n])
    ts = np.arange(n) * dt_sample

    # Find peaks
    peaks = []
    for i in range(1, n-1):
        if abs_acf[i] > abs_acf[i-1] and abs_acf[i] > abs_acf[i+1] and abs_acf[i] > 0.02:
            peaks.append(i)

    if len(peaks) < 3:
        mask = (acf[:n] > 0.02) & (ts > 0)
        if np.sum(mask) < 5:
            return 0.0, 0.0
        t_f, v_f = ts[mask], acf[:n][mask]
    else:
        peaks = np.array(peaks)
        t_f, v_f = ts[peaks], abs_acf[peaks]

    try:
        log_v = np.log(v_f)
        c = np.polyfit(t_f, log_v, 1)
        gamma = -c[0]
        pred = np.polyval(c, t_f)
        ss_res = np.sum((log_v - pred)**2)
        ss_tot = np.sum((log_v - np.mean(log_v))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        return max(gamma, 0), r2
    except:
        return 0.0, 0.0


def main():
    print("=" * 80)
    print("SPECTRAL GAP STUDY — solve_ivp backend")
    print("=" * 80)

    fig_dir = "/Users/wujiewang/code/det-sampler/.worktrees/spectral-gap-cert-075/orbits/spectral-gap-cert-075/figures"
    os.makedirs(fig_dir, exist_ok=True)

    omega = 1.0
    kT = 1.0
    t_max = 2000.0  # 2000 time units per trajectory
    dt_out = 0.05
    n_runs = 4

    alphas = [0.25, 0.5, 0.75, 1.0, np.sqrt(2), 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    Qs = [0.1, 0.3, 1.0, 3.0, 10.0]

    all_results = {}
    total = len(Qs) * (len(alphas) + 1)
    count = 0

    for Q in Qs:
        print(f"\n{'='*60}")
        print(f"Q = {Q}   Kac: α_opt = {np.sqrt(2.0/Q):.4f}")
        print(f"{'='*60}")

        for alpha in alphas:
            count += 1
            t0 = time.time()

            taus, gammas = [], []
            acf_mean = None
            for run in range(n_runs):
                ts, qs, ps, xis = run_trajectory_ivp(
                    omega, kT, Q, 'tanh', alpha,
                    t_max=t_max, dt_out=dt_out, seed=42+run*100)
                if qs is None:
                    continue
                # Discard first 10% as burn-in
                burn = len(qs) // 10
                qs = qs[burn:]

                acf = compute_acf(qs, max_lag=min(len(qs)//2, 10000))
                tau = compute_tau_int(acf, dt_out)
                gamma, r2 = fit_envelope(acf, dt_out)
                taus.append(tau)
                gammas.append(gamma)
                if acf_mean is None:
                    acf_mean = acf.copy()
                else:
                    ml = min(len(acf_mean), len(acf))
                    acf_mean = (acf_mean[:ml] + acf[:ml]) / 2.0  # running average

            tau_m = np.mean(taus) if taus else 999
            gamma_m = np.mean(gammas) if gammas else 0
            elapsed = time.time() - t0

            all_results[('tanh', alpha, Q)] = {
                'tau': tau_m, 'gamma': gamma_m, 'acf': acf_mean,
                'taus': taus, 'gammas': gammas
            }
            print(f"  [{count:3d}/{total}] tanh(α={alpha:5.3f}): "
                  f"τ={tau_m:7.2f}  γ={gamma_m:.4f}  1/τ={1/tau_m:.4f}  ({elapsed:.1f}s)")

        # Log-osc
        count += 1
        t0 = time.time()
        taus, gammas = [], []
        acf_mean = None
        for run in range(n_runs):
            ts, qs, ps, xis = run_trajectory_ivp(
                omega, kT, Q, 'losc', 0,
                t_max=t_max, dt_out=dt_out, seed=42+run*100)
            if qs is None:
                continue
            burn = len(qs) // 10
            qs = qs[burn:]
            acf = compute_acf(qs, max_lag=min(len(qs)//2, 10000))
            tau = compute_tau_int(acf, dt_out)
            gamma, r2 = fit_envelope(acf, dt_out)
            taus.append(tau)
            gammas.append(gamma)
            if acf_mean is None:
                acf_mean = acf.copy()
            else:
                ml = min(len(acf_mean), len(acf))
                acf_mean = (acf_mean[:ml] + acf[:ml]) / 2.0

        tau_m = np.mean(taus) if taus else 999
        gamma_m = np.mean(gammas) if gammas else 0
        elapsed = time.time() - t0
        all_results[('losc', 2.0, Q)] = {
            'tau': tau_m, 'gamma': gamma_m, 'acf': acf_mean,
            'taus': taus, 'gammas': gammas
        }
        print(f"  [{count:3d}/{total}] log-osc(g'=2):     "
              f"τ={tau_m:7.2f}  γ={gamma_m:.4f}  1/τ={1/tau_m:.4f}  ({elapsed:.1f}s)")

    # Standard NH reference
    print(f"\n{'='*60}")
    print("Reference: Standard NH g(ξ)=ξ, Q=1")
    print(f"{'='*60}")
    taus, gammas = [], []
    for run in range(n_runs):
        ts, qs, ps, xis = run_trajectory_ivp(
            omega, kT, 1.0, 'linear', 0,
            t_max=t_max, dt_out=dt_out, seed=42+run*100)
        if qs is None:
            continue
        burn = len(qs) // 10
        qs = qs[burn:]
        acf = compute_acf(qs, max_lag=min(len(qs)//2, 10000))
        tau = compute_tau_int(acf, dt_out)
        gamma, _ = fit_envelope(acf, dt_out)
        taus.append(tau)
        gammas.append(gamma)
    tau_m = np.mean(taus) if taus else 999
    gamma_m = np.mean(gammas) if gammas else 0
    all_results[('linear', 1.0, 1.0)] = {'tau': tau_m, 'gamma': gamma_m}
    print(f"  linear NH: τ={tau_m:.2f}  γ={gamma_m:.4f}")

    # ====== TABLES ======
    print_tables(all_results, alphas, Qs)

    # ====== FIGURES ======
    generate_figures(all_results, alphas, Qs, fig_dir, dt_out, omega, kT, t_max)

    return all_results


def print_tables(R, alphas, Qs):
    print("\n\n" + "=" * 110)
    print("TABLE 1: τ_int (time units) — LOWER = FASTER MIXING")
    print("=" * 110)
    hdr = f"{'Q':>6s}"
    for a in alphas:
        hdr += f" {a:>6.2f}"
    hdr += f" {'losc':>6s} {'α*':>6s} {'Kac':>6s}"
    print(hdr)
    print("-" * 110)
    for Q in Qs:
        line = f"{Q:6.1f}"
        best_t, best_a = 1e10, 0
        for a in alphas:
            t = R[('tanh', a, Q)]['tau']
            line += f" {t:6.2f}"
            if t < best_t:
                best_t, best_a = t, a
        lt = R[('losc', 2.0, Q)]['tau']
        line += f" {lt:6.2f} {best_a:6.3f} {np.sqrt(2.0/Q):6.3f}"
        print(line)

    print("\n\n" + "=" * 110)
    print("TABLE 2: Envelope decay rate γ — HIGHER = FASTER MIXING")
    print("=" * 110)
    hdr = f"{'Q':>6s}"
    for a in alphas:
        hdr += f" {a:>6.2f}"
    hdr += f" {'losc':>6s} {'α*':>6s} {'Kac':>6s}"
    print(hdr)
    print("-" * 110)
    for Q in Qs:
        line = f"{Q:6.1f}"
        best_g, best_a = 0, 0
        for a in alphas:
            g = R[('tanh', a, Q)]['gamma']
            line += f" {g:6.4f}"
            if g > best_g:
                best_g, best_a = g, a
        lg = R[('losc', 2.0, Q)]['gamma']
        line += f" {lg:6.4f} {best_a:6.3f} {np.sqrt(2.0/Q):6.3f}"
        print(line)

    print("\n\n" + "=" * 80)
    print("TABLE 3: Log-osc vs tanh(2ξ) — both g'(0)=2")
    print("=" * 80)
    print(f"{'Q':>6s}  {'τ_losc':>8s}  {'τ_tanh2':>8s}  {'speedup':>8s}  "
          f"{'γ_losc':>8s}  {'γ_tanh2':>8s}  {'γ_ratio':>8s}")
    print("-" * 65)
    for Q in Qs:
        tl = R[('losc', 2.0, Q)]['tau']
        tt = R[('tanh', 2.0, Q)]['tau']
        sp = tt / tl if tl > 0 else 0
        gl = R[('losc', 2.0, Q)]['gamma']
        gt = R[('tanh', 2.0, Q)]['gamma']
        gr = gl / gt if gt > 0 else float('inf')
        print(f"{Q:6.1f}  {tl:8.2f}  {tt:8.2f}  {sp:8.3f}  "
              f"{gl:8.4f}  {gt:8.4f}  {gr:8.3f}")

    print("\n\n" + "=" * 80)
    print("TABLE 4: Optimal α vs Kac √(2/Q)")
    print("=" * 80)
    print(f"{'Q':>6s}  {'α*':>8s}  {'Kac':>8s}  {'τ*':>8s}  {'τ_losc':>8s}  {'losc_better':>11s}")
    print("-" * 55)
    for Q in Qs:
        pairs = [(a, R[('tanh', a, Q)]['tau']) for a in alphas]
        best = min(pairs, key=lambda x: x[1])
        kac = np.sqrt(2.0/Q)
        tl = R[('losc', 2.0, Q)]['tau']
        w = "YES" if tl < best[1] else "no"
        print(f"{Q:6.1f}  {best[0]:8.3f}  {kac:8.3f}  {best[1]:8.2f}  {tl:8.2f}  {w:>11s}")


def generate_figures(R, alphas, Qs, fig_dir, dt_out, omega, kT, t_max):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Fig 1: τ and γ vs α
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    for i, Q in enumerate(Qs):
        taus = [R[('tanh', a, Q)]['tau'] for a in alphas]
        ax.plot(alphas, taus, 'o-', color=colors[i], label=f'Q={Q}', ms=5, lw=1.5)
        kac = np.sqrt(2.0/Q)
        ax.axvline(kac, color=colors[i], ls=':', alpha=0.4)
        lt = R[('losc', 2.0, Q)]['tau']
        ax.plot(2.0, lt, 's', color=colors[i], ms=10,
                markeredgecolor='black', markeredgewidth=1.5, zorder=5)
    ax.set_xlabel(r"$\alpha$ = g'(0)", fontsize=13)
    ax.set_ylabel(r"$\tau_{\mathrm{int}}$", fontsize=13)
    ax.set_title("(a) Autocorrelation time", fontsize=12)
    ax.legend(fontsize=9, title="Q")
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i, Q in enumerate(Qs):
        gammas = [R[('tanh', a, Q)]['gamma'] for a in alphas]
        ax.plot(alphas, gammas, 'o-', color=colors[i], label=f'Q={Q}', ms=5, lw=1.5)
        kac = np.sqrt(2.0/Q)
        ax.axvline(kac, color=colors[i], ls=':', alpha=0.4)
        lg = R[('losc', 2.0, Q)]['gamma']
        ax.plot(2.0, lg, 's', color=colors[i], ms=10,
                markeredgecolor='black', markeredgewidth=1.5, zorder=5)
    ax.set_xlabel(r"$\alpha$ = g'(0)", fontsize=13)
    ax.set_ylabel(r"$\gamma$", fontsize=13)
    ax.set_title("(b) ACF envelope decay rate", fontsize=12)
    ax.legend(fontsize=9, title="Q")
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    fig.suptitle("NH spectral gap: 1D HO ($\\omega$=1, kT=1)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/fig1_tau_gamma_vs_alpha.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig_dir}/fig1_tau_gamma_vs_alpha.png")
    plt.close()

    # Fig 2: ACF examples
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    cases = [
        (('tanh', 1.0, 1.0), r'tanh($\xi$), Q=1'),
        (('tanh', 2.0, 1.0), r'tanh($2\xi$), Q=1'),
        (('losc', 2.0, 1.0), 'log-osc, Q=1'),
        (('tanh', 1.0, 0.1), r'tanh($\xi$), Q=0.1'),
        (('tanh', 2.0, 0.1), r'tanh($2\xi$), Q=0.1'),
        (('losc', 2.0, 0.1), 'log-osc, Q=0.1'),
    ]
    for ax, (key, title) in zip(axes.flat, cases):
        if key in R and R[key].get('acf') is not None:
            acf = R[key]['acf']
            n_show = min(len(acf), 4000)
            ts = np.arange(n_show) * dt_out
            ax.plot(ts, acf[:n_show], lw=0.8)
            ax.axhline(0, color='k', lw=0.5)
            g = R[key]['gamma']
            if g > 0.001:
                env = np.exp(-g * ts)
                ax.plot(ts, env, 'r--', lw=1, alpha=0.7, label=f'γ={g:.3f}')
                ax.plot(ts, -env, 'r--', lw=1, alpha=0.7)
                ax.legend(fontsize=9)
        ax.set_xlabel('t'); ax.set_ylabel('C(t)/C(0)')
        ax.set_title(title, fontsize=11)
        ax.set_ylim(-1.1, 1.1); ax.grid(True, alpha=0.3)
    fig.suptitle("Autocorrelation functions", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/fig2_acf_examples.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_dir}/fig2_acf_examples.png")
    plt.close()

    # Fig 3: log-osc vs tanh head-to-head
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, Q in zip(axes, [0.1, 1.0, 10.0]):
        for key, label, color in [
            (('tanh', 2.0, Q), r'tanh($2\xi$)', '#d62728'),
            (('losc', 2.0, Q), 'log-osc', '#1f77b4')]:
            if key in R and R[key].get('acf') is not None:
                acf = R[key]['acf']
                n_show = min(len(acf), 4000)
                ts = np.arange(n_show) * dt_out
                ax.plot(ts, acf[:n_show], label=label, color=color, lw=1.2)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel('t'); ax.set_ylabel('C(t)/C(0)')
        ax.set_title(f'Q = {Q}'); ax.legend(fontsize=10)
        ax.set_ylim(-0.8, 1.1); ax.grid(True, alpha=0.3)
    fig.suptitle("Log-osc vs tanh(2ξ): both g'(0)=2", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/fig3_losc_vs_tanh.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_dir}/fig3_losc_vs_tanh.png")
    plt.close()

    # Fig 4: Optimal α vs Kac prediction
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    opt_alphas = []
    kacs = []
    for Q in Qs:
        pairs = [(a, R[('tanh', a, Q)]['tau']) for a in alphas]
        best_a = min(pairs, key=lambda x: x[1])[0]
        opt_alphas.append(best_a)
        kacs.append(np.sqrt(2.0/Q))
    ax.scatter(kacs, opt_alphas, s=80, c='blue', zorder=5, label='Measured optimal')
    lim = max(max(kacs), max(opt_alphas)) * 1.2
    ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.5, label='Perfect agreement')
    for i, Q in enumerate(Qs):
        ax.annotate(f'Q={Q}', (kacs[i], opt_alphas[i]), fontsize=9,
                    xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel(r'Kac prediction $\sqrt{2/Q}$', fontsize=12)
    ax.set_ylabel(r'Measured optimal $\alpha^*$', fontsize=12)
    ax.set_title('Optimal damping strength vs Kac formula', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/fig4_optimal_vs_kac.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_dir}/fig4_optimal_vs_kac.png")
    plt.close()


if __name__ == "__main__":
    results = main()
