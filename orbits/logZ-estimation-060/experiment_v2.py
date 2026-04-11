"""
logZ Estimation v2 — Better-equilibrated AIS/TI
================================================

Key changes from v1:
- More equilibration steps for AIS and TI
- Focus on the dimensional scaling story (IS fails in high-d, AIS/TI should not)
- Only run E1 (comparison) and E3 (scaling) with better params
- Keep E2 and E4 from v1
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import logsumexp
import os
import sys
import time

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
})

SEED = 42
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

C_NHCNF = '#1f77b4'
C_TI    = '#ff7f0e'
C_IS    = '#2ca02c'
C_TRUE  = '#d62728'


# =============================================================================
# GMM utilities (vectorized)
# =============================================================================

def gmm_params(n_modes=5, radius=3.0, sigma=0.5, d=2):
    angles = np.linspace(0, 2 * np.pi, n_modes, endpoint=False)
    means = np.zeros((n_modes, d))
    means[:, 0] = radius * np.cos(angles)
    means[:, 1] = radius * np.sin(angles)
    weights = np.ones(n_modes) / n_modes
    return means, sigma, weights


def gmm_log_prob(q, means, sigma):
    """log p_GMM(q) for single point."""
    K, d = means.shape
    diff = q[None, :] - means
    sq_dist = np.sum(diff**2, axis=-1)
    log_norm = -0.5 * d * np.log(2 * np.pi * sigma**2)
    log_components = log_norm - 0.5 * sq_dist / sigma**2 - np.log(K)
    return logsumexp(log_components)


def gmm_log_prob_batch(qs, means, sigma):
    """Vectorized log p_GMM. qs: (N, d)."""
    K, d = means.shape
    diff = qs[:, None, :] - means[None, :, :]
    sq_dist = np.sum(diff**2, axis=-1)
    log_norm = -0.5 * d * np.log(2 * np.pi * sigma**2)
    log_components = log_norm - 0.5 * sq_dist / sigma**2 - np.log(K)
    return logsumexp(log_components, axis=-1)


def gmm_grad_V(q, means, sigma):
    """Gradient of V = -log p_GMM."""
    K, d = means.shape
    diff = q[None, :] - means
    sq_dist = np.sum(diff**2, axis=-1)
    log_components = -0.5 * sq_dist / sigma**2
    r = np.exp(log_components - logsumexp(log_components))
    grad_log_p = np.sum(r[:, None] * (means - q[None, :]), axis=0) / sigma**2
    return -grad_log_p


def annealed_grad_V(q, means, sigma, sigma_ref, beta):
    """Gradient of V_beta = (1-beta)*V_ref + beta*V_target."""
    grad_V_ref = q / sigma_ref**2
    grad_V_target = gmm_grad_V(q, means, sigma)
    return (1 - beta) * grad_V_ref + beta * grad_V_target


# =============================================================================
# NH-tanh RK4 integrator
# =============================================================================

def nh_tanh_rk4_step(q, p, xi, grad_V_fn, dt, Q=1.0, kT=1.0, d=None):
    if d is None:
        d = q.shape[0]

    def f(q_, p_, xi_):
        gv = grad_V_fn(q_)
        g = np.tanh(xi_)
        dq = p_.copy()
        dp = -gv - g * p_
        dxi = (1.0 / Q) * (np.sum(p_**2) - d * kT)
        return dq, dp, dxi

    k1q, k1p, k1x = f(q, p, xi)
    k2q, k2p, k2x = f(q + 0.5*dt*k1q, p + 0.5*dt*k1p, xi + 0.5*dt*k1x)
    k3q, k3p, k3x = f(q + 0.5*dt*k2q, p + 0.5*dt*k2p, xi + 0.5*dt*k2x)
    k4q, k4p, k4x = f(q + dt*k3q, p + dt*k3p, xi + dt*k3x)

    q_new = q + (dt/6.0) * (k1q + 2*k2q + 2*k3q + k4q)
    p_new = p + (dt/6.0) * (k1p + 2*k2p + 2*k3p + k4p)
    xi_new = xi + (dt/6.0) * (k1x + 2*k2x + 2*k3x + k4x)

    g_start = np.tanh(xi)
    g_end = np.tanh(xi_new)
    div_integral = -d * 0.5 * (g_start + g_end) * dt

    return q_new, p_new, xi_new, div_integral


# =============================================================================
# NH-CNF AIS
# =============================================================================

def nhcnf_ais(means, sigma, d, n_beta=11, n_equil=2000, dt=0.01, Q=1.0,
              kT=1.0, sigma_ref=5.0, n_chains=5, seed=42):
    """AIS with NH-CNF transition kernel."""
    betas = np.linspace(0, 1, n_beta)
    log_Z_ref = (d / 2.0) * np.log(2 * np.pi * sigma_ref**2)
    log_weights = np.zeros(n_chains)
    nfe = 0
    trace = []

    rng = np.random.RandomState(seed)
    q_chains = [rng.randn(d) * sigma_ref for _ in range(n_chains)]
    p_chains = [rng.randn(d) * np.sqrt(kT) for _ in range(n_chains)]
    xi_chains = [0.0 for _ in range(n_chains)]

    for k in range(len(betas) - 1):
        beta_curr = betas[k]
        beta_next = betas[k + 1]

        # Weight increment
        for i in range(n_chains):
            q = q_chains[i]
            V_ref = np.sum(q**2) / (2 * sigma_ref**2)
            V_target = -gmm_log_prob(q, means, sigma)
            log_u_curr = -(1 - beta_curr) * V_ref - beta_curr * V_target
            log_u_next = -(1 - beta_next) * V_ref - beta_next * V_target
            log_weights[i] += log_u_next - log_u_curr

        # Equilibrate at beta_next
        for i in range(n_chains):
            q, p, xi = q_chains[i].copy(), p_chains[i].copy(), xi_chains[i]
            gv_fn = lambda q_, _b=beta_next: annealed_grad_V(q_, means, sigma, sigma_ref, _b)
            for _ in range(n_equil):
                q, p, xi, _ = nh_tanh_rk4_step(q, p, xi, gv_fn, dt, Q, kT, d)
                nfe += 1
            q_chains[i], p_chains[i], xi_chains[i] = q, p, xi

        log_Z_est = logsumexp(log_weights) - np.log(n_chains) + log_Z_ref
        trace.append((nfe, log_Z_est))

    return log_weights, log_Z_ref, nfe, trace


# =============================================================================
# TI
# =============================================================================

def thermodynamic_integration(means, sigma, d, n_beta=11, n_burn=500,
                              n_samples=1000, dt=0.01, Q=1.0, kT=1.0,
                              sigma_ref=5.0, seed=42):
    """TI: log(Z1/Z0) = -int_0^1 <V_target - V_ref>_beta dbeta."""
    betas = np.linspace(0, 1, n_beta)
    dF = np.zeros(n_beta)
    nfe = 0
    trace = []
    log_Z_ref = (d / 2.0) * np.log(2 * np.pi * sigma_ref**2)

    rng = np.random.RandomState(seed)
    q = rng.randn(d) * sigma_ref
    p = rng.randn(d) * np.sqrt(kT)
    xi = 0.0

    for k, beta in enumerate(betas):
        gv_fn = lambda q_, _b=beta: annealed_grad_V(q_, means, sigma, sigma_ref, _b)

        for _ in range(n_burn):
            q, p, xi, _ = nh_tanh_rk4_step(q, p, xi, gv_fn, dt, Q, kT, d)
            nfe += 1

        vals = []
        for step in range(n_samples):
            q, p, xi, _ = nh_tanh_rk4_step(q, p, xi, gv_fn, dt, Q, kT, d)
            nfe += 1
            if step % 5 == 0:
                V_ref = np.sum(q**2) / (2 * sigma_ref**2)
                V_target = -gmm_log_prob(q, means, sigma)
                vals.append(V_target - V_ref)

        dF[k] = np.mean(vals)
        if k > 0:
            integral = np.trapezoid(dF[:k+1], betas[:k+1])
            trace.append((nfe, log_Z_ref - integral))

    integral = np.trapezoid(dF, betas)
    return log_Z_ref - integral, dF, betas, nfe, trace


# =============================================================================
# IS (vectorized)
# =============================================================================

def importance_sampling(means, sigma, d, n_samples=50000, sigma_ref=5.0, seed=42):
    rng = np.random.RandomState(seed)
    qs = rng.randn(n_samples, d) * sigma_ref
    log_p_ref = (-0.5 * d * np.log(2 * np.pi * sigma_ref**2)
                 - np.sum(qs**2, axis=1) / (2 * sigma_ref**2))
    log_p_target = gmm_log_prob_batch(qs, means, sigma)
    log_ws = log_p_target - log_p_ref

    trace = []
    for cp in range(500, n_samples + 1, 500):
        trace.append((cp, logsumexp(log_ws[:cp]) - np.log(cp)))

    return logsumexp(log_ws) - np.log(n_samples), trace


# =============================================================================
# E1: Main comparison (better params)
# =============================================================================

def experiment_e1():
    print("=" * 60)
    print("E1: logZ estimation for 2D GMM", flush=True)
    print("=" * 60)

    d = 2
    means, sigma, _ = gmm_params(5, 3.0, 0.5, d)
    log_Z_true = 0.0
    sigma_ref = 5.0

    n_runs = 10
    results = {'ais': [], 'ti': [], 'is': []}
    traces = {'ais': [], 'ti': [], 'is': []}

    for run in range(n_runs):
        seed = 42 + run * 7
        print(f"  Run {run+1}/{n_runs}...", flush=True)

        # AIS: 5 chains, 11 betas, 1000 equil steps
        _, _, nfe, trace = nhcnf_ais(
            means, sigma, d, n_beta=11, n_equil=1000,
            dt=0.01, Q=1.0, sigma_ref=sigma_ref, n_chains=5, seed=seed)
        results['ais'].append(trace[-1][1])
        traces['ais'].append(trace)

        # TI: 11 windows, 500 burn + 1000 samples
        log_Z_ti, _, _, _, trace_ti = thermodynamic_integration(
            means, sigma, d, n_beta=11, n_burn=500, n_samples=1000,
            dt=0.01, sigma_ref=sigma_ref, seed=seed)
        results['ti'].append(log_Z_ti)
        traces['ti'].append(trace_ti)

        # IS: 100k (fast)
        log_Z_is, trace_is = importance_sampling(
            means, sigma, d, n_samples=100000, sigma_ref=sigma_ref, seed=seed)
        results['is'].append(log_Z_is)
        traces['is'].append(trace_is)

    # Summary
    for name, key in [('AIS', 'ais'), ('TI', 'ti'), ('IS', 'is')]:
        vals = np.array(results[key])
        print(f"  {name}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
              f"|bias|={abs(vals.mean()):.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    ax = axes[0]
    run_ids = np.arange(1, n_runs + 1)
    ax.plot(run_ids, results['ais'], 'o-', color=C_NHCNF, label='NH-CNF AIS', ms=5)
    ax.plot(run_ids, results['ti'], 's-', color=C_TI, label='TI (NH)', ms=5)
    ax.plot(run_ids, results['is'], '^-', color=C_IS, label='IS (100k)', ms=5)
    ax.axhline(0, color=C_TRUE, ls='--', lw=2, label='True (log Z=0)')
    ax.set_xlabel('Run index')
    ax.set_ylabel('log Z estimate')
    ax.set_title('(a) log Z estimates (2D GMM)', fontweight='bold')
    ax.legend(fontsize=9, frameon=False)

    # Zoomed inset: IS and AIS only (TI variance compresses them near zero)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_inset = inset_axes(ax, width="40%", height="40%", loc='lower left',
                          bbox_to_anchor=(0.05, 0.05, 1, 1), bbox_transform=ax.transAxes)
    ax_inset.plot(run_ids, results['ais'], 'o-', color=C_NHCNF, ms=3, lw=1)
    ax_inset.plot(run_ids, results['is'], '^-', color=C_IS, ms=3, lw=1)
    ax_inset.axhline(0, color=C_TRUE, ls='--', lw=1)
    ais_arr = np.array(results['ais'])
    is_arr = np.array(results['is'])
    y_lo = min(ais_arr.min(), is_arr.min()) - 0.1
    y_hi = max(ais_arr.max(), is_arr.max()) + 0.1
    ax_inset.set_ylim(y_lo, y_hi)
    ax_inset.set_title('AIS & IS zoom', fontsize=8)
    ax_inset.tick_params(labelsize=7)

    ax = axes[1]
    errors = {
        'NH-CNF\nAIS': np.abs(np.array(results['ais'])),
        'TI': np.abs(np.array(results['ti'])),
        'IS': np.abs(np.array(results['is'])),
    }
    colors_box = [C_NHCNF, C_TI, C_IS]
    bplot = ax.boxplot(list(errors.values()), tick_labels=list(errors.keys()),
                       patch_artist=True, widths=0.5)
    for patch, c in zip(bplot['boxes'], colors_box):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    ax.set_ylabel('|log Z error|')
    ax.set_title('(b) Error distribution (10 runs)', fontweight='bold')

    ax = axes[2]
    if traces['ais'][0]:
        nfes, logZs = zip(*traces['ais'][0])
        ax.plot(nfes, np.abs(np.array(logZs)), '-', color=C_NHCNF, label='AIS', lw=2)
    if traces['ti'][0]:
        nfes, logZs = zip(*traces['ti'][0])
        ax.plot(nfes, np.abs(np.array(logZs)), '-', color=C_TI, label='TI', lw=2)
    if traces['is'][0]:
        nfes, logZs = zip(*traces['is'][0])
        ax.plot(nfes, np.abs(np.array(logZs)), '-', color=C_IS, label='IS', lw=2, alpha=0.7)
    ax.set_xlabel('NFE / samples')
    ax.set_ylabel('|log Z error|')
    ax.set_yscale('log')
    ax.set_title('(c) Convergence (run 1)', fontweight='bold')
    ax.legend(fontsize=9, frameon=False)

    fig.savefig(os.path.join(FIGDIR, 'e1_logZ.png'))
    plt.close(fig)
    print("Saved e1_logZ.png", flush=True)
    return results


# =============================================================================
# E3: Scaling with dimension
# =============================================================================

def experiment_e3():
    print("\n" + "=" * 60)
    print("E3: Scaling with dimension", flush=True)
    print("=" * 60)

    dims = [2, 5, 10, 20]
    sigma = 0.5
    sigma_ref = 5.0
    n_runs = 5

    all_results = {}

    for d in dims:
        print(f"\n  d = {d}", flush=True)
        means, _, _ = gmm_params(5, 3.0, sigma, d)
        ais_vals, ti_vals, is_vals = [], [], []

        for run in range(n_runs):
            seed = 42 + run * 13
            print(f"    run {run+1}/{n_runs}...", flush=True)

            # IS (vectorized, fast)
            log_Z_is, _ = importance_sampling(
                means, sigma, d, n_samples=100000, sigma_ref=sigma_ref, seed=seed)
            is_vals.append(log_Z_is)

            # AIS: scale equil with d
            n_equil = min(2000, 500 + d * 50)
            _, _, _, trace = nhcnf_ais(
                means, sigma, d, n_beta=11, n_equil=n_equil,
                dt=0.01, Q=1.0, sigma_ref=sigma_ref, n_chains=3, seed=seed)
            ais_vals.append(trace[-1][1])

            # TI: scale with d
            n_burn_ti = min(1000, 300 + d * 30)
            n_samples_ti = min(1500, 500 + d * 30)
            log_Z_ti, _, _, _, _ = thermodynamic_integration(
                means, sigma, d, n_beta=11, n_burn=n_burn_ti,
                n_samples=n_samples_ti, dt=0.01, sigma_ref=sigma_ref, seed=seed)
            ti_vals.append(log_Z_ti)

        all_results[d] = {
            'ais': np.array(ais_vals),
            'ti': np.array(ti_vals),
            'is': np.array(is_vals)
        }
        for name, vals in [('AIS', ais_vals), ('TI', ti_vals), ('IS', is_vals)]:
            print(f"    {name}: mean={np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    # (a) Bias
    ax = axes[0]
    for method, color, label in [('ais', C_NHCNF, 'AIS (NH-CNF)'),
                                  ('ti', C_TI, 'TI (NH)'),
                                  ('is', C_IS, 'IS')]:
        biases = [np.abs(np.mean(all_results[d][method])) for d in dims]
        stds = [np.std(all_results[d][method]) for d in dims]
        ax.errorbar(dims, biases, yerr=stds, fmt='o-', color=color, label=label,
                    capsize=5, ms=7, lw=2)
    ax.set_xlabel('Dimension d')
    ax.set_ylabel('|Bias| in log Z')
    ax.set_title('(a) Bias vs dimension', fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=10, frameon=False)

    # (b) Variance
    ax = axes[1]
    for method, color, label in [('ais', C_NHCNF, 'AIS'),
                                  ('ti', C_TI, 'TI'),
                                  ('is', C_IS, 'IS')]:
        variances = [np.var(all_results[d][method]) for d in dims]
        ax.plot(dims, variances, 'o-', color=color, label=label, ms=7, lw=2)
    ax.set_xlabel('Dimension d')
    ax.set_ylabel('Variance of log Z')
    ax.set_title('(b) Variance vs dimension', fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=10, frameon=False)

    # (c) Mean estimate (clipped y-axis; IS at d=20 drops to ~-100)
    ax = axes[2]
    for method, color, label in [('ais', C_NHCNF, 'AIS'),
                                  ('ti', C_TI, 'TI'),
                                  ('is', C_IS, 'IS')]:
        means_est = [np.mean(all_results[d][method]) for d in dims]
        stds = [np.std(all_results[d][method]) for d in dims]
        ax.errorbar(dims, means_est, yerr=stds, fmt='o-', color=color, label=label,
                    capsize=5, ms=7, lw=2)
    ax.axhline(0, color=C_TRUE, ls='--', lw=2, label='True (log Z=0)')
    ax.set_xlabel('Dimension d')
    ax.set_ylabel('log Z estimate')
    ax.set_title('(c) Mean estimate vs dimension', fontweight='bold')
    # Clip y-axis so True line is visible; annotate IS off-scale
    ais_means = [np.mean(all_results[d]['ais']) for d in dims]
    ti_means = [np.mean(all_results[d]['ti']) for d in dims]
    is_means = [np.mean(all_results[d]['is']) for d in dims]
    y_lo = min(min(ais_means), min(ti_means)) * 1.3
    y_hi = max(max(ais_means), max(ti_means)) * 0.3 if max(max(ais_means), max(ti_means)) > 0 else 5
    y_hi = max(y_hi, 5)
    ax.set_ylim(y_lo, y_hi)
    # Annotate IS values that fall off-scale
    for di, d_val in enumerate(dims):
        if is_means[di] < y_lo:
            ax.annotate(f'IS: {is_means[di]:.0f}', xy=(d_val, y_lo),
                        fontsize=8, color=C_IS, ha='center', va='bottom',
                        fontweight='bold')
            ax.plot(d_val, y_lo, 'v', color=C_IS, ms=10, clip_on=False)
    ax.legend(fontsize=9, frameon=False)

    fig.savefig(os.path.join(FIGDIR, 'e3_scaling.png'))
    plt.close(fig)
    print("\nSaved e3_scaling.png", flush=True)
    return all_results


if __name__ == '__main__':
    t0 = time.time()
    print("logZ Estimation v2 — Better equilibration")
    print("=" * 60, flush=True)

    results_e1 = experiment_e1()
    results_e3 = experiment_e3()

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print("Done.", flush=True)
