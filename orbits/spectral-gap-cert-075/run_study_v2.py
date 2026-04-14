"""Spectral gap study v2 — multi-dimensional anisotropic HO.

The 1D HO with omega=1 is too easy: all damping functions mix fast.
The real differences appear at high condition number kappa >> 1.

We test:
  U(q) = sum_i 0.5 * omega_i^2 * q_i^2
  with omega_i spread over [omega_min, omega_max], kappa = omega_max/omega_min

The slowest mode (omega_min) bottlenecks mixing. The spectral gap is determined
by the decay rate of C_{q_slow}(t) for the slowest coordinate.

NH dynamics:
  dq_i/dt = p_i
  dp_i/dt = -omega_i^2 q_i - g(xi) p_i
  dxi/dt = (sum p_i^2 - d*kT) / Q

Key prediction: for the slow mode, the effective friction is g(xi) which
couples to ALL modes through xi. The thermostat variable xi is driven by
the total kinetic energy, which is dominated by fast modes.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time


def nh_rhs_nd(omega2_vec, kT, Q, g_type, g_param, d):
    """Return RHS for d-dimensional NH dynamics."""
    def rhs(t, y):
        q = y[:d]
        p = y[d:2*d]
        xi = y[2*d]

        if g_type == 'tanh':
            g = np.tanh(g_param * xi)
        elif g_type == 'losc':
            g = 2.0 * xi / (1.0 + xi**2)
        else:
            g = xi

        dq = p.copy()
        dp = -omega2_vec * q - g * p
        dxi = (np.sum(p**2) - d * kT) / Q

        return np.concatenate([dq, dp, [dxi]])
    return rhs


def run_trajectory_nd(omega2_vec, kT, Q, g_type, g_param,
                      t_max=2000, dt_out=0.1, seed=42):
    """Run d-dimensional NH trajectory."""
    d = len(omega2_vec)
    rng = np.random.RandomState(seed)

    # Initialize from equilibrium
    sigma_q = np.sqrt(kT / omega2_vec)
    sigma_p = np.sqrt(kT)
    sigma_xi = np.sqrt(kT / Q)

    q0 = rng.randn(d) * sigma_q
    p0 = rng.randn(d) * sigma_p
    xi0 = rng.randn() * sigma_xi

    y0 = np.concatenate([q0, p0, [xi0]])
    t_eval = np.arange(0, t_max, dt_out)

    rhs = nh_rhs_nd(omega2_vec, kT, Q, g_type, g_param, d)

    sol = solve_ivp(rhs, [0, t_max], y0, method='RK45',
                    t_eval=t_eval, rtol=1e-8, atol=1e-10, max_step=dt_out)

    if not sol.success:
        print(f"  WARNING: {sol.message}")
        return None

    return sol  # sol.y shape: (2*d+1, n_times)


def compute_acf(x, max_lag=None):
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
    tau = 0.5
    for t in range(1, len(acf)):
        if acf[t] < 0.05:
            break
        tau += acf[t]
    return max(tau, 1.0) * dt_sample


def fit_envelope(acf, dt_sample, max_fit=None):
    n = len(acf) if max_fit is None else min(len(acf), max_fit)
    abs_acf = np.abs(acf[:n])
    ts = np.arange(n) * dt_sample

    peaks = []
    for i in range(1, n-1):
        if abs_acf[i] > abs_acf[i-1] and abs_acf[i] > abs_acf[i+1] and abs_acf[i] > 0.01:
            peaks.append(i)

    if len(peaks) < 3:
        mask = (acf[:n] > 0.01) & (ts > 0)
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
    print("SPECTRAL GAP STUDY v2 — Multi-dimensional anisotropic HO")
    print("=" * 80)

    fig_dir = "/Users/wujiewang/code/det-sampler/.worktrees/spectral-gap-cert-075/orbits/spectral-gap-cert-075/figures"
    os.makedirs(fig_dir, exist_ok=True)

    kT = 1.0
    n_runs = 3

    # Test configurations matching orbit 073
    configs = [
        {'d': 1, 'kappa': 1, 'label': 'd=1, κ=1'},
        {'d': 1, 'kappa': 10, 'label': 'd=1, κ=10'},
        {'d': 1, 'kappa': 100, 'label': 'd=1, κ=100'},
        {'d': 5, 'kappa': 10, 'label': 'd=5, κ=10'},
        {'d': 5, 'kappa': 100, 'label': 'd=5, κ=100'},
        {'d': 10, 'kappa': 100, 'label': 'd=10, κ=100'},
    ]

    alphas = [0.25, 0.5, 1.0, np.sqrt(2), 2.0, 3.0, 4.0, 6.0, 8.0]
    Qs_per_config = {
        'd=1, κ=1': [1.0],
        'd=1, κ=10': [0.1, 1.0],
        'd=1, κ=100': [0.1, 1.0],
        'd=5, κ=10': [0.1, 1.0],
        'd=5, κ=100': [0.1, 1.0],
        'd=10, κ=100': [0.1, 1.0],
    }

    all_results = {}

    for cfg in configs:
        d = cfg['d']
        kappa = cfg['kappa']
        label = cfg['label']

        # Frequencies: omega_i^2 log-spaced from 1 to kappa^2
        if d == 1:
            omega2_vec = np.array([1.0])  # omega_min = 1
            if kappa > 1:
                omega2_vec = np.array([1.0])  # Still 1D, but we test the single slow mode
                # For 1D kappa>1, use omega^2 = 1/kappa (slow mode)
                omega2_vec = np.array([1.0 / kappa])
        else:
            omega2_vec = np.logspace(0, 2*np.log10(kappa), d) / kappa**2
            # This gives omega^2 from 1/kappa^2 to 1, so omega from 1/kappa to 1
            # Slowest mode has omega_min = 1/kappa

        omega_min = np.sqrt(np.min(omega2_vec))

        # Sampling params: need dt_out << 1/omega_min, t_max >> tau for slow mode
        dt_out = min(0.5, 0.5 / np.sqrt(np.max(omega2_vec)))
        t_max = max(5000, 500 * kappa)  # Need enough time for slow mode decorrelation
        if d >= 10:
            t_max = min(t_max, 5000)  # Limit for speed

        Qs = Qs_per_config[label]

        for Q in Qs:
            print(f"\n{'='*70}")
            print(f"{label}, Q={Q}  (ω_min={omega_min:.4f}, t_max={t_max}, dt={dt_out:.3f})")
            print(f"Kac prediction: α_opt = √(2d/{Q}) = {np.sqrt(2*d/Q):.4f}")
            print(f"{'='*70}")

            for alpha in alphas:
                t0 = time.time()
                taus_slow = []
                gammas_slow = []

                for run in range(n_runs):
                    sol = run_trajectory_nd(omega2_vec, kT, Q, 'tanh', alpha,
                                           t_max=t_max, dt_out=dt_out, seed=42+run*100)
                    if sol is None:
                        continue

                    # Measure ACF of the SLOWEST coordinate (q[0])
                    q_slow = sol.y[0]
                    burn = len(q_slow) // 10
                    q_slow = q_slow[burn:]

                    acf = compute_acf(q_slow, max_lag=min(len(q_slow)//2, 20000))
                    tau = compute_tau_int(acf, dt_out)
                    gamma, r2 = fit_envelope(acf, dt_out, max_fit=10000)
                    taus_slow.append(tau)
                    gammas_slow.append(gamma)

                tau_m = np.mean(taus_slow) if taus_slow else 999
                gamma_m = np.mean(gammas_slow) if gammas_slow else 0
                elapsed = time.time() - t0

                key = ('tanh', alpha, Q, label)
                all_results[key] = {'tau': tau_m, 'gamma': gamma_m}
                print(f"  tanh(α={alpha:5.3f}): τ_slow={tau_m:8.2f}  γ={gamma_m:.4f}  ({elapsed:.1f}s)")

            # Log-osc
            t0 = time.time()
            taus_slow, gammas_slow = [], []
            for run in range(n_runs):
                sol = run_trajectory_nd(omega2_vec, kT, Q, 'losc', 0,
                                       t_max=t_max, dt_out=dt_out, seed=42+run*100)
                if sol is None:
                    continue
                q_slow = sol.y[0]
                burn = len(q_slow) // 10
                q_slow = q_slow[burn:]
                acf = compute_acf(q_slow, max_lag=min(len(q_slow)//2, 20000))
                tau = compute_tau_int(acf, dt_out)
                gamma, r2 = fit_envelope(acf, dt_out, max_fit=10000)
                taus_slow.append(tau)
                gammas_slow.append(gamma)

            tau_m = np.mean(taus_slow) if taus_slow else 999
            gamma_m = np.mean(gammas_slow) if gammas_slow else 0
            elapsed = time.time() - t0
            key = ('losc', 2.0, Q, label)
            all_results[key] = {'tau': tau_m, 'gamma': gamma_m}
            print(f"  log-osc(g'=2):     τ_slow={tau_m:8.2f}  γ={gamma_m:.4f}  ({elapsed:.1f}s)")

            # Linear NH reference
            t0 = time.time()
            taus_slow, gammas_slow = [], []
            for run in range(n_runs):
                sol = run_trajectory_nd(omega2_vec, kT, Q, 'linear', 0,
                                       t_max=t_max, dt_out=dt_out, seed=42+run*100)
                if sol is None:
                    continue
                q_slow = sol.y[0]
                burn = len(q_slow) // 10
                q_slow = q_slow[burn:]
                acf = compute_acf(q_slow, max_lag=min(len(q_slow)//2, 20000))
                tau = compute_tau_int(acf, dt_out)
                gamma, r2 = fit_envelope(acf, dt_out, max_fit=10000)
                taus_slow.append(tau)
                gammas_slow.append(gamma)
            tau_m = np.mean(taus_slow) if taus_slow else 999
            gamma_m = np.mean(gammas_slow) if gammas_slow else 0
            elapsed = time.time() - t0
            key = ('linear', 1.0, Q, label)
            all_results[key] = {'tau': tau_m, 'gamma': gamma_m}
            print(f"  linear NH:         τ_slow={tau_m:8.2f}  γ={gamma_m:.4f}  ({elapsed:.1f}s)")

    # ====== SUMMARY TABLES ======
    print("\n\n" + "=" * 100)
    print("COMPREHENSIVE RESULTS: τ_int of slowest mode")
    print("=" * 100)

    for cfg in configs:
        label = cfg['label']
        Qs = Qs_per_config[label]
        for Q in Qs:
            print(f"\n--- {label}, Q={Q} ---")
            print(f"  {'g(ξ)':>12s}  {'α':>6s}  {'τ_slow':>8s}  {'γ':>8s}")
            print(f"  {'-'*40}")

            best_tau = 1e10
            best_label = ""
            for alpha in alphas:
                key = ('tanh', alpha, Q, label)
                if key in all_results:
                    r = all_results[key]
                    print(f"  {'tanh':>12s}  {alpha:6.3f}  {r['tau']:8.2f}  {r['gamma']:8.4f}")
                    if r['tau'] < best_tau:
                        best_tau = r['tau']
                        best_label = f"tanh(α={alpha:.2f})"

            key = ('losc', 2.0, Q, label)
            if key in all_results:
                r = all_results[key]
                print(f"  {'log-osc':>12s}  {'2.0':>6s}  {r['tau']:8.2f}  {r['gamma']:8.4f}")
                if r['tau'] < best_tau:
                    best_tau = r['tau']
                    best_label = "log-osc"

            key = ('linear', 1.0, Q, label)
            if key in all_results:
                r = all_results[key]
                print(f"  {'linear':>12s}  {'1.0':>6s}  {r['tau']:8.2f}  {r['gamma']:8.4f}")
                if r['tau'] < best_tau:
                    best_tau = r['tau']
                    best_label = "linear"

            print(f"  BEST: {best_label} (τ={best_tau:.2f})")

    # ====== KEY COMPARISON TABLE ======
    print("\n\n" + "=" * 100)
    print("KEY COMPARISON: log-osc vs tanh(2ξ) vs linear NH (for each config, Q=best)")
    print("=" * 100)
    print(f"{'config':>16s}  {'Q':>5s}  {'τ_losc':>8s}  {'τ_tanh2':>8s}  {'τ_linear':>8s}  "
          f"{'speedup':>8s}  {'τ_best_tanh':>11s}  {'α_best':>6s}")
    print("-" * 90)

    for cfg in configs:
        label = cfg['label']
        Qs = Qs_per_config[label]
        for Q in Qs:
            tl = all_results.get(('losc', 2.0, Q, label), {}).get('tau', 999)
            tt2 = all_results.get(('tanh', 2.0, Q, label), {}).get('tau', 999)
            tlin = all_results.get(('linear', 1.0, Q, label), {}).get('tau', 999)
            sp = tt2 / tl if tl > 0 else 0

            # Find best tanh
            best_tau, best_a = 1e10, 0
            for a in alphas:
                t = all_results.get(('tanh', a, Q, label), {}).get('tau', 999)
                if t < best_tau:
                    best_tau, best_a = t, a

            print(f"{label:>16s}  {Q:5.1f}  {tl:8.2f}  {tt2:8.2f}  {tlin:8.2f}  "
                  f"{sp:8.3f}  {best_tau:11.2f}  {best_a:6.3f}")

    # ====== GENERATE FIGURES ======
    generate_figures(all_results, alphas, configs, Qs_per_config, fig_dir)

    return all_results


def generate_figures(R, alphas, configs, Qs_per_config, fig_dir):
    """Generate figures."""

    # Main figure: τ_slow vs α for each config
    n_panels = sum(len(Qs_per_config[c['label']]) for c in configs)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flat

    panel_idx = 0
    for cfg in configs:
        label = cfg['label']
        Qs = Qs_per_config[label]
        for Q in Qs:
            if panel_idx >= 6:
                break
            ax = axes[panel_idx]
            panel_idx += 1

            taus = [R.get(('tanh', a, Q, label), {}).get('tau', np.nan) for a in alphas]
            ax.plot(alphas, taus, 'bo-', ms=5, lw=1.5, label='tanh')

            # log-osc
            tl = R.get(('losc', 2.0, Q, label), {}).get('tau', np.nan)
            ax.axhline(tl, color='red', ls='--', lw=1.5, label=f'log-osc (τ={tl:.1f})')

            # linear
            tlin = R.get(('linear', 1.0, Q, label), {}).get('tau', np.nan)
            ax.axhline(tlin, color='gray', ls=':', lw=1.5, label=f'linear (τ={tlin:.1f})')

            ax.set_xlabel(r"$\alpha$", fontsize=11)
            ax.set_ylabel(r"$\tau_{\mathrm{int}}$ (slow mode)", fontsize=11)
            ax.set_title(f"{label}, Q={Q}", fontsize=11)
            ax.legend(fontsize=8)
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)

    fig.suptitle("Spectral gap: autocorrelation time of slowest mode", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/fig_main_tau_vs_alpha_nd.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig_dir}/fig_main_tau_vs_alpha_nd.png")
    plt.close()


if __name__ == "__main__":
    results = main()
