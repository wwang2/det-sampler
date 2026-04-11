"""
corrected-dft-067: Crooks Detailed Fluctuation Theorem test on sigma_bath
per-trajectory from temperature quench simulations.

Orbit 066 tested P(sigma=+s)/P(sigma=-s)=exp(s) on the WRONG variable
(sigma_bath - sigma_exact, which is estimator residual).  This orbit tests
on sigma_bath ALONE, which is the physical heat flow into the NH thermostat
bath and should satisfy a fluctuation-theorem-like relation under
non-equilibrium driving.

Theory:
-------
For the NH-tanh thermostat under a sudden quench T0 -> T1, the bath entropy
production per trajectory is:

    sigma_bath = beta_1 * integral_0^t tanh(xi(s)) |p(s)|^2 ds

where beta_1 = 1/T1 is the final inverse temperature.

The Gallavotti-Cohen / Evans-Searles fluctuation theorem for the TOTAL
entropy production sigma_tot = sigma_bath - sigma_exact predicts:

    P(+sigma_tot) / P(-sigma_tot) = exp(sigma_tot)

i.e. slope = 1.0 in a log-ratio plot.

For sigma_bath ALONE, the situation is more subtle. sigma_bath is NOT the
total entropy production -- it's only the heat-flow component.  The DFT
slope for sigma_bath depends on the specific protocol and is NOT guaranteed
to be 1.0 or beta_1.  We test empirically what slope emerges.

We test sigma_bath, sigma_exact, AND sigma_tot to see which (if any)
satisfies the DFT.

Usage:
    python3 analysis.py
"""

import json
import os
import time
import numpy as np
from multiprocessing import Pool
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

# ============================================================================
#  Plotting defaults (style.md compliant)
# ============================================================================
mpl.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,

    'savefig.pad_inches': 0.2,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.6,
})

C_BATH = "#1f77b4"     # blue
C_EXACT = "#ff7f0e"    # orange
C_TOT = "#2ca02c"      # green
C_FIT = "#d62728"      # red
C_THEORY = "#9467bd"   # purple
C_GRAY = "#7f7f7f"

HERE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(HERE, "results")
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================================
#  2D double-well NH-tanh system (copied from orbit 065)
# ============================================================================

def dw_grad_V(q):
    """grad V for V(q) = (q1^2-1)^2 + 0.5*q2^2.  q: (N,2)."""
    out = np.empty_like(q)
    out[:, 0] = 4.0 * q[:, 0] * (q[:, 0]**2 - 1.0)
    out[:, 1] = q[:, 1]
    return out


def make_dw_stepper(d, Q_nh, dt):
    """Return rk4_step(q, p, xi, kT) for the 2D double-well."""
    def vf(q, p, xi, kT):
        g = np.tanh(xi)
        dq = p
        dp = -dw_grad_V(q) - g[:, None] * p
        psum = np.sum(p * p, axis=1)
        dxi = (psum - d * kT) / Q_nh
        return dq, dp, dxi

    def rk4(q, p, xi, kT):
        dq1, dp1, dx1 = vf(q, p, xi, kT)
        dq2, dp2, dx2 = vf(q + .5*dt*dq1, p + .5*dt*dp1, xi + .5*dt*dx1, kT)
        dq3, dp3, dx3 = vf(q + .5*dt*dq2, p + .5*dt*dp2, xi + .5*dt*dx2, kT)
        dq4, dp4, dx4 = vf(q + dt*dq3, p + dt*dp3, xi + dt*dx3, kT)
        return (q + dt/6*(dq1+2*dq2+2*dq3+dq4),
                p + dt/6*(dp1+2*dp2+2*dp3+dp4),
                xi + dt/6*(dx1+2*dx2+2*dx3+dx4))
    return rk4


# ============================================================================
#  Quench simulation -- saves per-trajectory bath, exact, and tot
# ============================================================================

def run_dw_quench_components(
    T0, T1, Q_nh=1.0, dt=0.005,
    t_burn=200.0, t_decorr=20.0, t_post=200.0,
    n_parents=60, n_branches=40,
    seed=0,
):
    """Run the 2D double-well sudden quench and return per-trajectory
    sigma_bath, sigma_exact, and sigma_tot arrays."""
    t0_wall = time.time()
    d = 2
    M = int(n_parents)
    K = int(n_branches)
    N = M * K
    n_burn = int(round(t_burn / dt))
    n_decorr = int(round(t_decorr / dt))
    n_post = int(round(t_post / dt))
    beta1 = 1.0 / T1

    rng = np.random.default_rng(seed)
    rk4 = make_dw_stepper(d, Q_nh, dt)

    # random IC: near left well
    q = rng.standard_normal((M, d)) * 0.3
    q[:, 0] -= 1.0
    p = rng.standard_normal((M, d)) * np.sqrt(T0)
    xi = rng.standard_normal(M)

    # burn-in at T0
    for _ in range(n_burn):
        q, p, xi = rk4(q, p, xi, T0)

    # per-trajectory accumulators
    final_bath = np.zeros(N)
    final_exact = np.zeros(N)

    for k_branch in range(K):
        qb = q.copy(); pb = p.copy(); xib = xi.copy()
        bath_cum = np.zeros(M)
        exact_cum = np.zeros(M)

        for step in range(n_post):
            xi_old = xib; p_old = pb
            br_old = beta1 * np.tanh(xi_old) * np.sum(p_old*p_old, axis=1)
            er_old = d * np.tanh(xi_old)

            qb, pb, xib = rk4(qb, pb, xib, T1)

            br_new = beta1 * np.tanh(xib) * np.sum(pb*pb, axis=1)
            er_new = d * np.tanh(xib)

            bath_cum += 0.5 * (br_old + br_new) * dt
            exact_cum += 0.5 * (er_old + er_new) * dt

        slc = slice(k_branch * M, (k_branch + 1) * M)
        final_bath[slc] = bath_cum
        final_exact[slc] = exact_cum

        # advance parents at T0 (decorrelation between branches)
        for _ in range(n_decorr):
            q, p, xi = rk4(q, p, xi, T0)

    wall = time.time() - t0_wall
    final_tot = final_bath - final_exact

    return dict(
        sigma_bath=final_bath,
        sigma_exact=final_exact,
        sigma_tot=final_tot,
        n_traj=N, d=d, T0=T0, T1=T1, beta1=beta1,
        Q_nh=Q_nh, dt=dt, t_post=t_post,
        wall_time=wall, seed=seed,
    )


# ============================================================================
#  DFT (Detailed Fluctuation Theorem) analysis
# ============================================================================

def dft_analysis(sigma, n_bins=40, min_count=5):
    """Compute the DFT log-ratio: log P(+s) / P(-s) for a distribution of
    entropy production values.

    For each bin center s > 0:
      - P(+s) = fraction of samples in [s - ds/2, s + ds/2]
      - P(-s) = fraction of samples in [-s - ds/2, -s + ds/2]
      - log_ratio = log(P(+s) / P(-s))

    Returns dict with s_values, log_ratios, weights, slope, intercept.
    """
    sigma = np.asarray(sigma)

    # Bin width ~ 0.3 * std, symmetric about 0
    std = np.std(sigma)
    bin_width = 0.3 * std

    # Create symmetric bins
    max_abs = np.max(np.abs(sigma))
    edges = np.arange(0, max_abs + bin_width, bin_width)

    s_vals = []
    log_rats = []
    weights = []
    count_pos_list = []
    count_neg_list = []

    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        s_center = 0.5 * (lo + hi)

        count_pos = np.sum((sigma >= lo) & (sigma < hi))
        count_neg = np.sum((sigma >= -hi) & (sigma < -lo))

        if count_pos >= min_count and count_neg >= min_count:
            lr = np.log(count_pos / count_neg)
            w = np.sqrt(count_pos * count_neg)
            s_vals.append(s_center)
            log_rats.append(lr)
            weights.append(w)
            count_pos_list.append(int(count_pos))
            count_neg_list.append(int(count_neg))

    s_vals = np.array(s_vals)
    log_rats = np.array(log_rats)
    weights = np.array(weights)

    # Weighted least squares: log_ratio = slope * s + intercept
    if len(s_vals) >= 2:
        W = np.diag(weights)
        A = np.column_stack([s_vals, np.ones_like(s_vals)])
        AW = A.T @ W
        params = np.linalg.solve(AW @ A, AW @ log_rats)
        slope_fit = params[0]
        intercept_fit = params[1]
    else:
        slope_fit = np.nan
        intercept_fit = np.nan

    return dict(
        s_values=s_vals,
        log_ratios=log_rats,
        weights=weights,
        count_pos=count_pos_list,
        count_neg=count_neg_list,
        slope=slope_fit,
        intercept=intercept_fit,
        n_points=len(s_vals),
        sigma_mean=float(np.mean(sigma)),
        sigma_std=float(np.std(sigma, ddof=1)),
    )


def bootstrap_dft_slope(sigma, n_bootstrap=1000, n_bins=40, min_count=5, rng_seed=42):
    """Bootstrap 95% CI on the DFT slope."""
    rng = np.random.default_rng(rng_seed)
    N = len(sigma)
    slopes = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        res = dft_analysis(sigma[idx], n_bins=n_bins, min_count=min_count)
        if not np.isnan(res['slope']):
            slopes.append(res['slope'])
    slopes = np.array(slopes)
    if len(slopes) < 10:
        return np.nan, np.nan, slopes
    ci_lo = np.percentile(slopes, 2.5)
    ci_hi = np.percentile(slopes, 97.5)
    return ci_lo, ci_hi, slopes


# ============================================================================
#  Exact KL computation (from orbit 065)
# ============================================================================

def compute_exact_qp_KL(T0, T1):
    """Exact numerical KL for (q,p) canonical measures of the 2D double-well."""
    from scipy import integrate

    def Z_q1(T, limit=10):
        f = lambda q: np.exp(-(q**2 - 1)**2 / T)
        val, _ = integrate.quad(f, -limit, limit)
        return val

    Zq0 = Z_q1(T0)
    Zq1 = Z_q1(T1)

    def integrand_q1(q):
        V = (q**2 - 1)**2
        rho0 = np.exp(-V / T0) / Zq0
        lr = V * (1.0/T1 - 1.0/T0) + np.log(Zq1 / Zq0)
        return rho0 * lr

    kl_q1, _ = integrate.quad(integrand_q1, -10, 10)
    kl_q2 = 0.5 * (np.log(T1/T0) - 1 + T0/T1)
    kl_p = (np.log(T1/T0) - 1 + T0/T1)
    return dict(kl_q1=float(kl_q1), kl_q2=float(kl_q2), kl_p=float(kl_p),
                kl_qp=float(kl_q1 + kl_q2 + kl_p))


# ============================================================================
#  Run one seed (for multiprocessing)
# ============================================================================

def run_phase(args):
    """Run one seed of a quench phase. Returns the result dict."""
    phase_label, T0, T1, seed = args
    print(f"  [{phase_label}] seed={seed} starting...")
    r = run_dw_quench_components(
        T0=T0, T1=T1, Q_nh=1.0, dt=0.005,
        t_burn=200.0, t_decorr=20.0, t_post=200.0,
        n_parents=60, n_branches=40, seed=seed,
    )
    print(f"  [{phase_label}] seed={seed} done in {r['wall_time']:.1f}s, "
          f"N={r['n_traj']}, "
          f"mean(bath)={np.mean(r['sigma_bath']):.4f}, "
          f"mean(exact)={np.mean(r['sigma_exact']):.4f}, "
          f"mean(tot)={np.mean(r['sigma_tot']):.4f}")
    return r


def skewness(x):
    """Fisher skewness."""
    x = np.asarray(x)
    n = len(x)
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-15:
        return 0.0
    return float(np.mean(((x - m) / s)**3) * n / max(n-1, 1))


# ============================================================================
#  Figure
# ============================================================================

def make_figure(
    p1_summary, p1_dft_bath, p1_dft_tot, p1_dft_exact,
    p1_all_bath, p1_all_tot, p1_all_exact,
    p2_summary, p2_dft_bath, p2_dft_tot, p2_dft_exact,
    p2_all_bath, p2_all_tot, p2_all_exact,
):
    """2x2 panel figure:
    (a) Phase 1 DFT log-ratio for sigma_bath, sigma_tot, sigma_exact
    (b) Phase 1 histograms
    (c) Phase 2 DFT log-ratio
    (d) Phase 2 histograms
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    for row_idx, (summary, dft_bath, dft_tot, dft_exact,
                  all_bath, all_tot, all_exact, phase_label) in enumerate([
        (p1_summary, p1_dft_bath, p1_dft_tot, p1_dft_exact,
         p1_all_bath, p1_all_tot, p1_all_exact, "Phase 1"),
        (p2_summary, p2_dft_bath, p2_dft_tot, p2_dft_exact,
         p2_all_bath, p2_all_tot, p2_all_exact, "Phase 2"),
    ]):
        beta1 = summary['beta1']
        T0 = summary['T0']
        T1 = summary['T1']

        # ---- Left panel: DFT log-ratio plot ----
        ax = axes[row_idx, 0]

        # Plot sigma_tot DFT (the control -- expected slope=1)
        if dft_tot['n_points'] >= 2:
            ax.scatter(dft_tot['s_values'], dft_tot['log_ratios'],
                      color=C_TOT, s=40, zorder=5,
                      label=r'$\sigma_{\rm tot}$ (slope=%.2f)' % dft_tot['slope'],
                      marker='o')
            s_range = np.linspace(0, np.max(dft_tot['s_values']), 100)
            ax.plot(s_range, dft_tot['slope'] * s_range + dft_tot['intercept'],
                   color=C_TOT, ls='--', lw=1.2, alpha=0.7)

        # Plot sigma_bath DFT
        if dft_bath['n_points'] >= 2:
            ax.scatter(dft_bath['s_values'], dft_bath['log_ratios'],
                      color=C_BATH, s=40, zorder=5,
                      label=r'$\sigma_{\rm bath}$ (slope=%.2f)' % dft_bath['slope'],
                      marker='s')
            s_range_b = np.linspace(0, np.max(dft_bath['s_values']), 100)
            ax.plot(s_range_b, dft_bath['slope'] * s_range_b + dft_bath['intercept'],
                   color=C_BATH, ls='--', lw=1.2, alpha=0.7)

        # Plot sigma_exact DFT
        if dft_exact['n_points'] >= 2:
            ax.scatter(dft_exact['s_values'], dft_exact['log_ratios'],
                      color=C_EXACT, s=30, zorder=5,
                      label=r'$\sigma_{\rm exact}$ (slope=%.2f)' % dft_exact['slope'],
                      marker='^')
            s_range_e = np.linspace(0, np.max(dft_exact['s_values']), 100)
            ax.plot(s_range_e, dft_exact['slope'] * s_range_e + dft_exact['intercept'],
                   color=C_EXACT, ls='--', lw=1.2, alpha=0.7)

        # Reference line: slope=1
        s_max = 0.1
        for dft_res in [dft_tot, dft_bath, dft_exact]:
            if len(dft_res['s_values']) > 0:
                s_max = max(s_max, np.max(dft_res['s_values']))
        s_ref = np.linspace(0, s_max * 1.1, 100)
        ax.plot(s_ref, 1.0 * s_ref, color=C_FIT, ls='-', lw=1.5, alpha=0.5,
               label='slope = 1.0 (FT)')

        ci_tot = summary['dft_tot']['ci_95']
        ci_bath = summary['dft_bath']['ci_95']

        panel_letter = '(a)' if row_idx == 0 else '(c)'
        ax.set_title(
            f'{panel_letter} {phase_label}: $T_0={T0}\\to T_1={T1}$\n'
            f'slope(tot)={dft_tot["slope"]:.3f} CI=[{ci_tot[0]:.2f},{ci_tot[1]:.2f}]',
            fontweight='bold', fontsize=12)
        ax.set_xlabel(r'$s$ (entropy production magnitude)')
        ax.set_ylabel(r'$\log\, P(+s) / P(-s)$')
        ax.legend(frameon=False, fontsize=9)

        # ---- Right panel: histograms ----
        ax = axes[row_idx, 1]
        n_hist_bins = 60

        ax.hist(all_tot, bins=n_hist_bins, density=True, alpha=0.4,
                color=C_TOT, label=r'$\sigma_{\rm tot}$', edgecolor='none')
        ax.hist(all_bath, bins=n_hist_bins, density=True, alpha=0.4,
                color=C_BATH, label=r'$\sigma_{\rm bath}$', edgecolor='none')
        ax.hist(all_exact, bins=n_hist_bins, density=True, alpha=0.4,
                color=C_EXACT, label=r'$\sigma_{\rm exact}$', edgecolor='none')

        # Vertical lines at means
        mean_tot = np.mean(all_tot)
        mean_bath = np.mean(all_bath)
        mean_exact = np.mean(all_exact)
        ax.axvline(mean_tot, color=C_TOT, ls='--', lw=1.5,
                   label=r'$\langle\sigma_{\rm tot}\rangle=%.3f$' % mean_tot)
        ax.axvline(mean_bath, color=C_BATH, ls='--', lw=1.5,
                   label=r'$\langle\sigma_{\rm bath}\rangle=%.3f$' % mean_bath)
        ax.axvline(mean_exact, color=C_EXACT, ls='--', lw=1.5,
                   label=r'$\langle\sigma_{\rm exact}\rangle=%.3f$' % mean_exact)
        ax.axvline(0, color='black', ls=':', lw=1, alpha=0.5)

        panel_letter_r = '(b)' if row_idx == 0 else '(d)'
        jar = summary['mean_jarzynski']
        ax.set_title(
            f'{panel_letter_r} {phase_label}: distributions\n'
            r'Jarzynski $\langle e^{-\sigma_{\rm tot}}\rangle = %.3f$' % jar,
            fontweight='bold', fontsize=12)
        ax.set_xlabel(r'$\sigma$ (entropy production)')
        ax.set_ylabel('density')
        ax.legend(frameon=False, fontsize=9, loc='upper right')

    fig.suptitle(
        'Crooks DFT test on NH-tanh temperature quench\n'
        '2D double-well, per-trajectory entropy production',
        fontsize=15, fontweight='bold', y=1.02)

    for fmt in ['png', 'pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'fig_corrected_dft.{fmt}'), bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {FIG_DIR}/fig_corrected_dft.png/pdf")


# ============================================================================
#  Main
# ============================================================================

def main():
    seeds = [42, 123, 7]

    # ---- Phase 1: T0=1.0 -> T1=2.0 ----
    print("=" * 70)
    print("PHASE 1: 2D double-well, T0=1.0 -> T1=2.0")
    print("=" * 70)

    kl1 = compute_exact_qp_KL(1.0, 2.0)
    print(f"Exact KL(q,p) = {kl1['kl_qp']:.4f}")

    # Run 3 seeds in parallel
    phase1_args = [("phase1", 1.0, 2.0, s) for s in seeds]
    with Pool(3) as pool:
        phase1_results = pool.map(run_phase, phase1_args)

    # ---- Phase 2: T0=0.8 -> T1=1.5 ----
    print("\n" + "=" * 70)
    print("PHASE 2: 2D double-well, T0=0.8 -> T1=1.5")
    print("=" * 70)

    kl2 = compute_exact_qp_KL(0.8, 1.5)
    print(f"Exact KL(q,p) = {kl2['kl_qp']:.4f}")

    phase2_args = [("phase2", 0.8, 1.5, s) for s in seeds]
    with Pool(3) as pool:
        phase2_results = pool.map(run_phase, phase2_args)

    # ---- DFT analysis on each phase ----
    all_phase_data = {}
    for phase_label, results, kl_exact, T1_val in [
        ("phase1", phase1_results, kl1, 2.0),
        ("phase2", phase2_results, kl2, 1.5),
    ]:
        beta1 = 1.0 / T1_val
        print(f"\n{'='*70}")
        print(f"DFT ANALYSIS: {phase_label} (beta1 = {beta1:.4f})")
        print(f"{'='*70}")

        # Combine all per-trajectory data across seeds
        all_bath = np.concatenate([r['sigma_bath'] for r in results])
        all_exact = np.concatenate([r['sigma_exact'] for r in results])
        all_tot = np.concatenate([r['sigma_tot'] for r in results])

        print(f"Total trajectories: {len(all_bath)}")
        print(f"sigma_bath:  mean={np.mean(all_bath):.4f} +/- {np.std(all_bath, ddof=1):.4f}")
        print(f"sigma_exact: mean={np.mean(all_exact):.4f} +/- {np.std(all_exact, ddof=1):.4f}")
        print(f"sigma_tot:   mean={np.mean(all_tot):.4f} +/- {np.std(all_tot, ddof=1):.4f}")

        # DFT on sigma_bath
        dft_bath = dft_analysis(all_bath)
        ci_lo_bath, ci_hi_bath, boot_slopes_bath = bootstrap_dft_slope(all_bath)
        print(f"\nDFT on sigma_bath:")
        print(f"  Slope = {dft_bath['slope']:.4f}  (expected: beta1={beta1:.4f} or 1.0?)")
        print(f"  Intercept = {dft_bath['intercept']:.4f}")
        print(f"  95% CI = [{ci_lo_bath:.4f}, {ci_hi_bath:.4f}]")
        print(f"  N data points = {dft_bath['n_points']}")

        # DFT on sigma_tot (control -- should have slope ~1.0)
        dft_tot = dft_analysis(all_tot)
        ci_lo_tot, ci_hi_tot, boot_slopes_tot = bootstrap_dft_slope(all_tot)
        print(f"\nDFT on sigma_tot (control):")
        print(f"  Slope = {dft_tot['slope']:.4f}  (expected: 1.0)")
        print(f"  Intercept = {dft_tot['intercept']:.4f}")
        print(f"  95% CI = [{ci_lo_tot:.4f}, {ci_hi_tot:.4f}]")
        print(f"  N data points = {dft_tot['n_points']}")

        # DFT on sigma_exact
        dft_exact = dft_analysis(all_exact)
        ci_lo_exact, ci_hi_exact, boot_slopes_exact = bootstrap_dft_slope(all_exact)
        print(f"\nDFT on sigma_exact:")
        print(f"  Slope = {dft_exact['slope']:.4f}")
        print(f"  Intercept = {dft_exact['intercept']:.4f}")
        print(f"  95% CI = [{ci_lo_exact:.4f}, {ci_hi_exact:.4f}]")
        print(f"  N data points = {dft_exact['n_points']}")

        # Per-seed statistics
        seed_stats = []
        for r in results:
            s_bath = dft_analysis(r['sigma_bath'])
            s_tot = dft_analysis(r['sigma_tot'])
            s_exact = dft_analysis(r['sigma_exact'])
            jar = float(np.mean(np.exp(-r['sigma_tot'])))
            seed_stats.append(dict(
                seed=int(r['seed']),
                mean_bath=float(np.mean(r['sigma_bath'])),
                mean_exact=float(np.mean(r['sigma_exact'])),
                mean_tot=float(np.mean(r['sigma_tot'])),
                std_bath=float(np.std(r['sigma_bath'], ddof=1)),
                std_exact=float(np.std(r['sigma_exact'], ddof=1)),
                std_tot=float(np.std(r['sigma_tot'], ddof=1)),
                jarzynski=jar,
                dft_slope_bath=float(s_bath['slope']) if not np.isnan(s_bath['slope']) else None,
                dft_slope_tot=float(s_tot['slope']) if not np.isnan(s_tot['slope']) else None,
                dft_slope_exact=float(s_exact['slope']) if not np.isnan(s_exact['slope']) else None,
                wall_time=float(r['wall_time']),
            ))

        # Jarzynski check
        jars = [s['jarzynski'] for s in seed_stats]
        mean_jar = np.mean(jars)
        print(f"\nJarzynski <exp(-sigma_tot)>: {mean_jar:.4f} (target: 1.0)")

        # Save results
        summary = dict(
            phase=phase_label,
            T0=results[0]['T0'], T1=results[0]['T1'],
            beta1=beta1,
            kl_qp_exact=kl_exact['kl_qp'],
            n_traj_total=len(all_bath),
            dft_bath=dict(
                slope=float(dft_bath['slope']),
                intercept=float(dft_bath['intercept']),
                ci_95=[float(ci_lo_bath), float(ci_hi_bath)],
                n_points=int(dft_bath['n_points']),
                s_values=dft_bath['s_values'].tolist(),
                log_ratios=dft_bath['log_ratios'].tolist(),
            ),
            dft_tot=dict(
                slope=float(dft_tot['slope']),
                intercept=float(dft_tot['intercept']),
                ci_95=[float(ci_lo_tot), float(ci_hi_tot)],
                n_points=int(dft_tot['n_points']),
                s_values=dft_tot['s_values'].tolist(),
                log_ratios=dft_tot['log_ratios'].tolist(),
            ),
            dft_exact=dict(
                slope=float(dft_exact['slope']),
                intercept=float(dft_exact['intercept']),
                ci_95=[float(ci_lo_exact), float(ci_hi_exact)],
                n_points=int(dft_exact['n_points']),
                s_values=dft_exact['s_values'].tolist(),
                log_ratios=dft_exact['log_ratios'].tolist(),
            ),
            per_seed=seed_stats,
            mean_jarzynski=float(mean_jar),
            sigma_bath_stats=dict(
                mean=float(np.mean(all_bath)),
                std=float(np.std(all_bath, ddof=1)),
                median=float(np.median(all_bath)),
                skew=float(skewness(all_bath)),
            ),
            sigma_tot_stats=dict(
                mean=float(np.mean(all_tot)),
                std=float(np.std(all_tot, ddof=1)),
                median=float(np.median(all_tot)),
                skew=float(skewness(all_tot)),
            ),
            sigma_exact_stats=dict(
                mean=float(np.mean(all_exact)),
                std=float(np.std(all_exact, ddof=1)),
                median=float(np.median(all_exact)),
                skew=float(skewness(all_exact)),
            ),
        )
        with open(os.path.join(RES_DIR, f"dft_{phase_label}.json"), "w") as f:
            json.dump(summary, f, indent=2)

        all_phase_data[phase_label] = dict(
            summary=summary,
            dft_bath=dft_bath, dft_tot=dft_tot, dft_exact=dft_exact,
            all_bath=all_bath, all_tot=all_tot, all_exact=all_exact,
        )

    # ---- Figure ----
    print("\n" + "=" * 70)
    print("GENERATING FIGURE")
    print("=" * 70)

    p1 = all_phase_data["phase1"]
    p2 = all_phase_data["phase2"]

    make_figure(
        p1['summary'], p1['dft_bath'], p1['dft_tot'], p1['dft_exact'],
        p1['all_bath'], p1['all_tot'], p1['all_exact'],
        p2['summary'], p2['dft_bath'], p2['dft_tot'], p2['dft_exact'],
        p2['all_bath'], p2['all_tot'], p2['all_exact'],
    )

    print("\nDone. Results in:", RES_DIR)
    print("Figures in:", FIG_DIR)

    # Return the Phase 1 DFT slope on sigma_tot as the orbit metric
    return p1['summary']['dft_tot']['slope']


if __name__ == "__main__":
    metric = main()
    print(f"\n[METRIC] Phase 1 DFT slope on sigma_tot = {metric:.4f}")
