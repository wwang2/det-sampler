"""
logZ Estimation via NH-CNF Exact Density Tracking
==================================================

Can NH-CNF's exact log-density tracking estimate normalizing constants
(partition functions) from a single trajectory?

The idea: start from a distribution with KNOWN Z (e.g., a Gaussian),
flow to the target, and the accumulated divergence integral gives
log(Z_target/Z_ref).

We implement and compare:
1. NH-CNF + Annealed Importance Sampling (AIS)
2. Thermodynamic Integration (TI) with NH sampler
3. Simple Importance Sampling (IS)

Test system: 2D Gaussian Mixture Model with 5 modes on a ring.
All numpy/scipy -- no torch dependency.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import logsumexp
import os
import time

# --- Global plot defaults ---
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

# Colors
C_NHCNF = '#1f77b4'
C_TI    = '#ff7f0e'
C_IS    = '#2ca02c'
C_TRUE  = '#d62728'


# =============================================================================
# GMM utilities (pure numpy)
# =============================================================================

def gmm_params(n_modes=5, radius=3.0, sigma=0.5, d=2):
    """GMM: modes on a ring in first 2 dims, equal weights."""
    angles = np.linspace(0, 2 * np.pi, n_modes, endpoint=False)
    means = np.zeros((n_modes, d))
    means[:, 0] = radius * np.cos(angles)
    means[:, 1] = radius * np.sin(angles)
    weights = np.ones(n_modes) / n_modes
    return means, sigma, weights


def gmm_log_prob(q, means, sigma):
    """log p_GMM(q) for a single point q (shape (d,))."""
    K, d = means.shape
    diff = q[None, :] - means  # (K, d)
    sq_dist = np.sum(diff**2, axis=-1)  # (K,)
    log_norm = -0.5 * d * np.log(2 * np.pi * sigma**2)
    log_components = log_norm - 0.5 * sq_dist / sigma**2 - np.log(K)
    return logsumexp(log_components)


def gmm_grad_V(q, means, sigma):
    """Gradient of V(q) = -log p_GMM(q). Analytical formula.

    grad V = -grad log p = -sum_k r_k * (mu_k - q) / sigma^2
    where r_k = softmax(log component_k).
    """
    K, d = means.shape
    diff = q[None, :] - means  # (K, d)
    sq_dist = np.sum(diff**2, axis=-1)  # (K,)
    log_components = -0.5 * sq_dist / sigma**2  # unnormalized log responsibilities
    # softmax to get responsibilities
    r = np.exp(log_components - logsumexp(log_components))  # (K,)
    # grad log p = sum_k r_k * (mu_k - q) / sigma^2
    grad_log_p = np.sum(r[:, None] * (means - q[None, :]), axis=0) / sigma**2
    return -grad_log_p  # grad V = -grad log p


def annealed_V_and_grad(q, means, sigma, sigma_ref, beta):
    """
    Annealed potential: V_beta = (1-beta)*V_ref + beta*V_target
    V_ref = ||q||^2 / (2*sigma_ref^2)
    V_target = -log p_GMM(q)
    Returns (V_beta, grad_V_beta).
    """
    V_ref = np.sum(q**2) / (2 * sigma_ref**2)
    grad_V_ref = q / sigma_ref**2

    log_p = gmm_log_prob(q, means, sigma)
    V_target = -log_p
    grad_V_target = gmm_grad_V(q, means, sigma)

    V = (1 - beta) * V_ref + beta * V_target
    grad_V = (1 - beta) * grad_V_ref + beta * grad_V_target
    return V, grad_V


# =============================================================================
# NH-tanh RK4 integrator (numpy version)
# =============================================================================

def nh_tanh_rk4_step(q, p, xi, grad_V_fn, dt, Q=1.0, kT=1.0, d=None):
    """One RK4 step of NH-tanh ODE. All arrays are 1D numpy."""
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

    # Divergence integral (trapezoidal)
    g_start = np.tanh(xi)
    g_end = np.tanh(xi_new)
    div_integral = -d * 0.5 * (g_start + g_end) * dt

    return q_new, p_new, xi_new, div_integral


# =============================================================================
# Method 1: NH-CNF AIS
# =============================================================================

def nhcnf_ais(means, sigma, d, n_beta=11, n_steps_per_beta=3000,
              dt=0.005, Q=1.0, kT=1.0, sigma_ref=5.0, n_chains=20, seed=42):
    """
    Annealed Importance Sampling using NH-CNF as the transition kernel.

    At each temperature transition beta_k -> beta_{k+1}, the AIS weight
    increment is computed from the exact potential evaluations (not from
    density estimation). The NH-CNF then equilibrates at the new temperature.

    Returns: log_weights (n_chains,), log_Z_ref, nfe, trace
    """
    betas = np.linspace(0, 1, n_beta)
    log_Z_ref = (d / 2.0) * np.log(2 * np.pi * sigma_ref**2)

    log_weights = np.zeros(n_chains)
    nfe_total = 0
    trace = []  # (nfe, log_Z_est) pairs

    rng = np.random.RandomState(seed)

    # Initialize chains from reference Gaussian
    q_chains = [rng.randn(d) * sigma_ref for _ in range(n_chains)]
    p_chains = [rng.randn(d) * np.sqrt(kT) for _ in range(n_chains)]
    xi_chains = [0.0 for _ in range(n_chains)]

    for k in range(len(betas) - 1):
        beta_curr = betas[k]
        beta_next = betas[k + 1]

        # AIS weight increment: log p_{beta_next}(q) - log p_{beta_curr}(q)
        # where log p_beta(q) = -(1-beta)*V_ref(q) - beta*V_target(q) + const
        for i in range(n_chains):
            q = q_chains[i]
            V_ref = np.sum(q**2) / (2 * sigma_ref**2)
            V_target = -gmm_log_prob(q, means, sigma)

            log_u_curr = -(1 - beta_curr) * V_ref - beta_curr * V_target
            log_u_next = -(1 - beta_next) * V_ref - beta_next * V_target
            log_weights[i] += log_u_next - log_u_curr

        # Equilibrate at beta_next using NH-CNF
        for i in range(n_chains):
            q = q_chains[i].copy()
            p = p_chains[i].copy()
            xi = xi_chains[i]

            def grad_V_fn(q_):
                _, gv = annealed_V_and_grad(q_, means, sigma, sigma_ref, beta_next)
                return gv

            for step in range(n_steps_per_beta):
                q, p, xi, div = nh_tanh_rk4_step(q, p, xi, grad_V_fn, dt, Q, kT, d)
                nfe_total += 1

            q_chains[i] = q
            p_chains[i] = p
            xi_chains[i] = xi

        # Current log Z estimate
        log_Z_ratio = logsumexp(log_weights) - np.log(n_chains)
        log_Z_est = log_Z_ratio + log_Z_ref
        # Since Z_target = 1 (normalized GMM), the "true" log Z = 0,
        # and log_Z_est should approach 0.
        # But what we compute is log Z_target = log(Z_target/Z_ref) + log Z_ref
        # = log_Z_ratio + log_Z_ref.
        # Since Z_target = 1 => log Z_target = 0 => log_Z_ratio should = -log_Z_ref
        trace.append((nfe_total, log_Z_est))

    return log_weights, log_Z_ref, nfe_total, trace


# =============================================================================
# Method 2: Thermodynamic Integration (TI)
# =============================================================================

def thermodynamic_integration(means, sigma, d, n_beta=11, n_samples=3000,
                              dt=0.005, Q=1.0, kT=1.0, sigma_ref=5.0, seed=42):
    """
    TI: log(Z_target/Z_ref) = -integral_0^1 <V_target - V_ref>_beta dbeta

    At each beta, sample with NH and average V_target - V_ref.
    """
    betas = np.linspace(0, 1, n_beta)
    dF_dbeta = np.zeros(n_beta)
    nfe_total = 0
    trace = []

    rng = np.random.RandomState(seed)
    q = rng.randn(d) * sigma_ref
    p = rng.randn(d) * np.sqrt(kT)
    xi = 0.0

    log_Z_ref = (d / 2.0) * np.log(2 * np.pi * sigma_ref**2)

    for k, beta in enumerate(betas):
        def grad_V_fn(q_, _beta=beta):
            _, gv = annealed_V_and_grad(q_, means, sigma, sigma_ref, _beta)
            return gv

        # Burn-in
        n_burn = 300
        for step in range(n_burn):
            q, p, xi, _ = nh_tanh_rk4_step(q, p, xi, grad_V_fn, dt, Q, kT, d)
            nfe_total += 1

        # Collect <V_target - V_ref>
        vals = []
        for step in range(n_samples):
            q, p, xi, _ = nh_tanh_rk4_step(q, p, xi, grad_V_fn, dt, Q, kT, d)
            nfe_total += 1
            if step % 10 == 0:
                V_ref = np.sum(q**2) / (2 * sigma_ref**2)
                V_target = -gmm_log_prob(q, means, sigma)
                vals.append(V_target - V_ref)

        dF_dbeta[k] = np.mean(vals)

        if k > 0:
            integral = np.trapezoid(dF_dbeta[:k+1], betas[:k+1])
            log_Z_est = log_Z_ref - integral
            trace.append((nfe_total, log_Z_est))

    integral = np.trapezoid(dF_dbeta, betas)
    log_Z_final = log_Z_ref - integral
    return log_Z_final, dF_dbeta, betas, nfe_total, trace


# =============================================================================
# Method 3: Simple Importance Sampling
# =============================================================================

def gmm_log_prob_batch(qs, means, sigma):
    """Vectorized log p_GMM for batch of points. qs: (N, d), means: (K, d)."""
    K, d = means.shape
    # qs: (N, d), means: (K, d) -> diff: (N, K, d)
    diff = qs[:, None, :] - means[None, :, :]
    sq_dist = np.sum(diff**2, axis=-1)  # (N, K)
    log_norm = -0.5 * d * np.log(2 * np.pi * sigma**2)
    log_components = log_norm - 0.5 * sq_dist / sigma**2 - np.log(K)
    return logsumexp(log_components, axis=-1)  # (N,)


def importance_sampling(means, sigma, d, n_samples=50000, sigma_ref=5.0, seed=42):
    """
    IS: Z_target = E_ref[p_target(q) / p_ref(q)], vectorized.
    """
    rng = np.random.RandomState(seed)

    qs = rng.randn(n_samples, d) * sigma_ref  # (N, d)
    log_p_ref = (-0.5 * d * np.log(2 * np.pi * sigma_ref**2)
                 - np.sum(qs**2, axis=1) / (2 * sigma_ref**2))  # (N,)
    log_p_target = gmm_log_prob_batch(qs, means, sigma)  # (N,)
    log_ws = log_p_target - log_p_ref

    # Build convergence trace
    trace = []
    checkpoints = np.arange(200, n_samples + 1, 200)
    for cp in checkpoints:
        log_Z_est = logsumexp(log_ws[:cp]) - np.log(cp)
        trace.append((cp, log_Z_est))

    log_Z_final = logsumexp(log_ws) - np.log(n_samples)
    return log_Z_final, trace


# =============================================================================
# E1: Main comparison
# =============================================================================

def experiment_e1():
    """Compare methods for estimating log Z of a 2D GMM."""
    print("=" * 60)
    print("E1: logZ estimation for 2D GMM (5 modes, ring)")
    print("=" * 60)

    d = 2
    means, sigma, weights = gmm_params(5, 3.0, 0.5, d)
    log_Z_true = 0.0  # GMM is normalized
    sigma_ref = 5.0
    print(f"True log Z = {log_Z_true:.4f}")
    print(f"Reference: N(0, {sigma_ref}^2 I), log Z_ref = {(d/2)*np.log(2*np.pi*sigma_ref**2):.4f}")

    n_runs = 10
    results = {
        'nhcnf_ais': {'logZ': [], 'traces': []},
        'ti': {'logZ': [], 'traces': []},
        'is': {'logZ': [], 'traces': []},
    }

    import sys
    for run in range(n_runs):
        seed = 42 + run * 7
        print(f"  Run {run+1}/{n_runs} (seed={seed})...", flush=True)

        # NH-CNF AIS: 2 chains, 6 betas, 500 steps = 6k NFE (fast)
        log_w, log_Z_ref, nfe, trace = nhcnf_ais(
            means, sigma, d, n_beta=6, n_steps_per_beta=500,
            dt=0.01, Q=1.0, kT=1.0, sigma_ref=sigma_ref, n_chains=2, seed=seed
        )
        log_Z_ais = trace[-1][1] if trace else log_Z_ref
        results['nhcnf_ais']['logZ'].append(log_Z_ais)
        results['nhcnf_ais']['traces'].append(trace)

        # TI: 6 windows, 500 samples each (+ 500 burn) = 6k NFE
        log_Z_ti, _, _, nfe_ti, trace_ti = thermodynamic_integration(
            means, sigma, d, n_beta=6, n_samples=500,
            dt=0.01, Q=1.0, kT=1.0, sigma_ref=sigma_ref, seed=seed
        )
        results['ti']['logZ'].append(log_Z_ti)
        results['ti']['traces'].append(trace_ti)

        # IS: 50k samples (vectorized, fast)
        log_Z_is, trace_is = importance_sampling(
            means, sigma, d, n_samples=50000, sigma_ref=sigma_ref, seed=seed
        )
        results['is']['logZ'].append(log_Z_is)
        results['is']['traces'].append(trace_is)
        sys.stdout.flush()

    # Print summary
    for name, key in [('NH-CNF AIS', 'nhcnf_ais'), ('TI', 'ti'), ('IS', 'is')]:
        vals = np.array(results[key]['logZ'])
        print(f"  {name}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
              f"|bias|={abs(vals.mean() - log_Z_true):.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    # (a) log Z estimates across runs
    ax = axes[0]
    run_ids = np.arange(1, n_runs + 1)
    ax.plot(run_ids, results['nhcnf_ais']['logZ'], 'o-', color=C_NHCNF,
            label='NH-CNF AIS', markersize=4, alpha=0.8)
    ax.plot(run_ids, results['ti']['logZ'], 's-', color=C_TI,
            label='TI (11 windows)', markersize=4, alpha=0.8)
    ax.plot(run_ids, results['is']['logZ'], '^-', color=C_IS,
            label='IS (50k)', markersize=4, alpha=0.8)
    ax.axhline(log_Z_true, color=C_TRUE, ls='--', lw=2, label=f'True (log Z={log_Z_true:.1f})')
    ax.set_xlabel('Run index')
    ax.set_ylabel('log Z estimate')
    ax.set_title('(a) log Z estimates across runs', fontweight='bold')
    ax.legend(fontsize=9, frameon=False)

    # (b) |error| boxplot
    ax = axes[1]
    errors = {
        'NH-CNF\nAIS': np.abs(np.array(results['nhcnf_ais']['logZ']) - log_Z_true),
        'TI': np.abs(np.array(results['ti']['logZ']) - log_Z_true),
        'IS': np.abs(np.array(results['is']['logZ']) - log_Z_true),
    }
    colors = [C_NHCNF, C_TI, C_IS]
    bplot = ax.boxplot(list(errors.values()), labels=list(errors.keys()),
                       patch_artist=True, widths=0.5)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel('|log Z error|')
    ax.set_title('(b) Error distribution (20 runs)', fontweight='bold')

    # (c) Convergence trace (first run)
    ax = axes[2]
    if results['nhcnf_ais']['traces'][0]:
        nfes, logZs = zip(*results['nhcnf_ais']['traces'][0])
        ax.plot(nfes, np.abs(np.array(logZs) - log_Z_true), '-', color=C_NHCNF,
                label='NH-CNF AIS', lw=2)
    if results['ti']['traces'][0]:
        nfes, logZs = zip(*results['ti']['traces'][0])
        ax.plot(nfes, np.abs(np.array(logZs) - log_Z_true), '-', color=C_TI,
                label='TI', lw=2)
    if results['is']['traces'][0]:
        nfes, logZs = zip(*results['is']['traces'][0])
        ax.plot(nfes, np.abs(np.array(logZs) - log_Z_true), '-', color=C_IS,
                label='IS', lw=2, alpha=0.7)
    ax.set_xlabel('NFE / samples')
    ax.set_ylabel('|log Z error|')
    ax.set_yscale('log')
    ax.set_title('(c) Convergence (run 1)', fontweight='bold')
    ax.legend(fontsize=9, frameon=False)

    fig.savefig(os.path.join(FIGDIR, 'e1_logZ.png'))
    plt.close(fig)
    print(f"Saved e1_logZ.png")
    return results


# =============================================================================
# E2: Z(beta) curve
# =============================================================================

def experiment_e2():
    """Compute Z(beta) for the GMM at different inverse temperatures."""
    print("\n" + "=" * 60)
    print("E2: Partition function curve Z(beta)")
    print("=" * 60)

    d = 2
    means, sigma, weights = gmm_params(5, 3.0, 0.5, d)
    sigma_ref = 5.0

    betas_test = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0])

    # Compute Z(beta) via IS with large sample for "ground truth"
    # Z(beta) = int p_GMM(q)^beta dq
    # = E_ref[ p_GMM(q)^beta / p_ref(q) ] where ref = N(0, sigma_ref^2 I)
    n_is = 100000
    rng = np.random.RandomState(42)
    q_samples = rng.randn(n_is, d) * sigma_ref
    log_p_ref = -0.5 * d * np.log(2 * np.pi * sigma_ref**2) - np.sum(q_samples**2, axis=1) / (2 * sigma_ref**2)

    # Pre-compute log p_GMM for all samples (vectorized)
    log_p_gmm = gmm_log_prob_batch(q_samples, means, sigma)

    log_Z_is = np.zeros(len(betas_test))
    log_Z_ti = np.zeros(len(betas_test))

    for j, beta in enumerate(betas_test):
        # IS: Z(beta) = E_ref[ exp(beta * log_p_gmm) / p_ref ]
        # = int exp(beta*log_p_gmm(q)) dq (since p_GMM is normalized, this is int p^beta dq)
        log_w = beta * log_p_gmm - log_p_ref
        log_Z_is[j] = logsumexp(log_w) - np.log(n_is)
        print(f"  beta={beta:.1f}: log Z(IS, 200k) = {log_Z_is[j]:.4f}")

    # Analytical for well-separated modes
    K = 5
    w = 1.0 / K
    log_Z_analytical = np.zeros(len(betas_test))
    for j, beta in enumerate(betas_test):
        # int (sum_k w_k N(q;mu_k,s^2))^beta dq
        # ~ K * w^beta * (2*pi*s^2)^(d/2*(1-beta)) * beta^(-d/2)
        log_single = (d/2.0) * (1 - beta) * np.log(2 * np.pi * sigma**2) - (d/2.0) * np.log(beta)
        log_Z_analytical[j] = np.log(K) + beta * np.log(w) + log_single

    # Also compute with TI for a few beta values
    for j, beta in enumerate(betas_test):
        if beta <= 1.0:
            # TI from 0 to beta
            n_b = max(5, int(beta * 11))
            betas_path = np.linspace(0, beta, n_b)
            log_Z_ref = (d / 2.0) * np.log(2 * np.pi * sigma_ref**2)

            rng2 = np.random.RandomState(42)
            q = rng2.randn(d) * sigma_ref
            p = rng2.randn(d)
            xi = 0.0

            dF = np.zeros(n_b)
            for ki, b in enumerate(betas_path):
                def gv_fn(q_, _b=b):
                    _, gv = annealed_V_and_grad(q_, means, sigma, sigma_ref, _b)
                    return gv

                # burn + sample (reduced for speed)
                for _ in range(200):
                    q, p, xi, _ = nh_tanh_rk4_step(q, p, xi, gv_fn, 0.01, Q=1.0, kT=1.0, d=d)
                vals = []
                for s in range(300):
                    q, p, xi, _ = nh_tanh_rk4_step(q, p, xi, gv_fn, 0.01, Q=1.0, kT=1.0, d=d)
                    if s % 5 == 0:
                        V_r = np.sum(q**2) / (2 * sigma_ref**2)
                        V_t = -gmm_log_prob(q, means, sigma)
                        vals.append(V_t - V_r)
                dF[ki] = np.mean(vals)

            integral = np.trapezoid(dF, betas_path)
            log_Z_ti[j] = log_Z_ref - integral
        else:
            log_Z_ti[j] = np.nan

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    ax = axes[0]
    ax.plot(betas_test, log_Z_is, 'o-', color=C_IS, lw=2, markersize=7, label='IS (200k)', zorder=3)
    mask_ti = ~np.isnan(log_Z_ti)
    ax.plot(betas_test[mask_ti], log_Z_ti[mask_ti], 's--', color=C_TI, lw=2, markersize=7, label='TI (NH)')
    ax.plot(betas_test, log_Z_analytical, '^:', color=C_TRUE, lw=1.5, markersize=6,
            label='Analytical (separated)', alpha=0.8)
    ax.set_xlabel(r'Inverse temperature $\beta$')
    ax.set_ylabel(r'$\log Z(\beta)$')
    ax.set_title(r'(a) Partition function $\log Z(\beta)$', fontweight='bold')
    ax.legend(fontsize=10, frameon=False)

    ax = axes[1]
    err_analytical = np.abs(log_Z_is - log_Z_analytical)
    err_ti = np.abs(log_Z_is[mask_ti] - log_Z_ti[mask_ti])
    ax.plot(betas_test, err_analytical, '^-', color=C_TRUE, lw=2, markersize=6,
            label='|IS - Analytical|')
    ax.plot(betas_test[mask_ti], err_ti, 's-', color=C_TI, lw=2, markersize=6,
            label='|IS - TI|')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel('Discrepancy')
    ax.set_title('(b) Method agreement', fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=10, frameon=False)

    fig.savefig(os.path.join(FIGDIR, 'e2_partition.png'))
    plt.close(fig)
    print("Saved e2_partition.png")
    return betas_test, log_Z_is, log_Z_analytical


# =============================================================================
# E3: Scaling with dimension
# =============================================================================

def experiment_e3():
    """How does log Z estimation accuracy scale with dimension?"""
    print("\n" + "=" * 60)
    print("E3: Scaling with dimension")
    print("=" * 60)

    dims = [2, 5, 10, 20]
    sigma = 0.5
    sigma_ref = 5.0
    n_runs = 5
    log_Z_true = 0.0

    all_results = {}

    for d in dims:
        print(f"\n  d = {d}", flush=True)
        means, _, _ = gmm_params(5, 3.0, sigma, d)
        ti_vals = []
        is_vals = []

        for run in range(n_runs):
            seed = 42 + run * 13

            # IS (vectorized, fast)
            rng = np.random.RandomState(seed)
            n_is = 50000
            qs = rng.randn(n_is, d) * sigma_ref
            log_p_ref = (-0.5 * d * np.log(2 * np.pi * sigma_ref**2)
                         - np.sum(qs**2, axis=1) / (2 * sigma_ref**2))
            log_p_gmm = gmm_log_prob_batch(qs, means, sigma)
            log_ws = log_p_gmm - log_p_ref
            log_Z_is = logsumexp(log_ws) - np.log(n_is)
            is_vals.append(log_Z_is)

            # TI (reduced params)
            n_samples_ti = max(300, 500 - d * 10)
            log_Z_ti, _, _, _, _ = thermodynamic_integration(
                means, sigma, d, n_beta=6, n_samples=n_samples_ti,
                dt=0.01, Q=1.0, kT=1.0, sigma_ref=sigma_ref, seed=seed
            )
            ti_vals.append(log_Z_ti)

        all_results[d] = {'ti': np.array(ti_vals), 'is': np.array(is_vals)}
        print(f"    IS: mean={np.mean(is_vals):.4f} +/- {np.std(is_vals):.4f}")
        print(f"    TI: mean={np.mean(ti_vals):.4f} +/- {np.std(ti_vals):.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    ax = axes[0]
    for method, color, label in [('ti', C_TI, 'TI (NH)'), ('is', C_IS, 'IS')]:
        biases = [np.abs(np.mean(all_results[d][method]) - log_Z_true) for d in dims]
        stds = [np.std(all_results[d][method]) for d in dims]
        ax.errorbar(dims, biases, yerr=stds, fmt='o-', color=color, label=label,
                    capsize=5, markersize=7, lw=2)
    ax.set_xlabel('Dimension d')
    ax.set_ylabel('|Bias| in log Z')
    ax.set_title('(a) Bias vs dimension', fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=10, frameon=False)

    ax = axes[1]
    for method, color, label in [('ti', C_TI, 'TI (NH)'), ('is', C_IS, 'IS')]:
        variances = [np.var(all_results[d][method]) for d in dims]
        ax.plot(dims, variances, 'o-', color=color, label=label, markersize=7, lw=2)
    ax.set_xlabel('Dimension d')
    ax.set_ylabel('Variance of log Z estimate')
    ax.set_title('(b) Variance vs dimension', fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=10, frameon=False)

    fig.savefig(os.path.join(FIGDIR, 'e3_scaling.png'))
    plt.close(fig)
    print("Saved e3_scaling.png")
    return all_results


# =============================================================================
# E4: Conceptual figure
# =============================================================================

def experiment_e4():
    """Conceptual figure: how Z estimation works."""
    print("\n" + "=" * 60)
    print("E4: Conceptual figure")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # (a) Flow from reference to target
    ax = axes[0]
    x = np.linspace(-6, 6, 300)

    # Reference (broad Gaussian)
    ref = np.exp(-x**2 / (2 * 5**2))
    ref /= ref.max()
    ax.fill_between(x, ref, alpha=0.15, color=C_NHCNF)
    ax.plot(x, ref, color=C_NHCNF, lw=2, label=r'Reference $p_0 = \mathcal{N}(0, \sigma_{ref}^2)$')

    # Target (GMM-like)
    target = np.zeros_like(x)
    for mu in [-3, -1.5, 0, 1.5, 3]:
        target += np.exp(-(x - mu)**2 / (2 * 0.4**2))
    target /= (5 * target.max())
    ax.fill_between(x, target, alpha=0.15, color=C_TRUE)
    ax.plot(x, target, color=C_TRUE, lw=2, label=r'Target $p_1 \propto e^{-V(q)}$')

    # Intermediate
    for beta in [0.3, 0.6]:
        inter = np.exp(-x**2 / (2 * 5**2))**(1 - beta) * (target / target.max())**beta
        inter /= inter.max() * 1.5
        ax.plot(x, inter, '--', color='gray', lw=1, alpha=0.5)

    # Arrow
    ax.annotate('', xy=(4.5, 0.25), xytext=(4.5, 0.75),
                arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
    ax.text(5.0, 0.50, r'$\beta: 0 \to 1$', fontsize=12, va='center', style='italic')

    # Key equation box
    eq = (r'$\log Z_{target} = \log Z_{ref}$' + '\n'
          r'$- \int_0^1 \langle V_{target} - V_{ref} \rangle_\beta \, d\beta$')
    ax.text(0.03, 0.97, eq, transform=ax.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9, edgecolor='gray'))

    ax.set_xlabel('q')
    ax.set_ylabel('Density')
    ax.set_title('(a) Annealing from reference to target', fontweight='bold')
    ax.legend(fontsize=9, loc='upper right', frameon=False)
    ax.set_ylim(0, 1.1)

    # (b) AIS weight accumulation
    ax = axes[1]
    betas = np.linspace(0, 1, 11)

    # Schematic free energy curve
    F_beta = 3.0 * (1 - betas) + 0.0 * betas - 1.5 * betas * (1 - betas)
    ax.plot(betas, F_beta, 'k-', lw=2.5, zorder=3)
    ax.plot(betas, F_beta, 'o', color=C_NHCNF, markersize=9, zorder=4,
            label='NH-CNF equilibration')

    # Fill area = integral
    ax.fill_between(betas, F_beta, alpha=0.12, color=C_TI)
    ax.text(0.45, 1.0, r'$\Delta F = -\log(Z_1/Z_0)$', fontsize=12, ha='center',
            color=C_TI, fontweight='bold')

    # Show NH-CNF at each window
    for i, b in enumerate(betas):
        if i % 2 == 0 and i > 0 and i < 10:
            ax.annotate('', xy=(b, F_beta[i] - 0.15), xytext=(b, F_beta[i] - 0.6),
                        arrowprops=dict(arrowstyle='->', color=C_NHCNF, lw=1.2, alpha=0.5))

    ax.set_xlabel(r'Annealing parameter $\beta$')
    ax.set_ylabel(r'$\langle V_{target} - V_{ref} \rangle_\beta$')
    ax.set_title('(b) Thermodynamic integration path', fontweight='bold')
    ax.text(0.0, -0.5, 'Reference\n(known Z)', fontsize=10, ha='center', color=C_NHCNF)
    ax.text(1.0, -0.5, 'Target\n(unknown Z)', fontsize=10, ha='center', color=C_TRUE)
    ax.legend(fontsize=9, loc='upper right', frameon=False)

    fig.savefig(os.path.join(FIGDIR, 'e4_concept.png'))
    plt.close(fig)
    print("Saved e4_concept.png")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    t0 = time.time()
    print("logZ Estimation via NH-CNF Exact Density Tracking")
    print("=" * 60)

    experiment_e4()
    results_e1 = experiment_e1()
    results_e2 = experiment_e2()
    results_e3 = experiment_e3()

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print("All experiments complete.")
