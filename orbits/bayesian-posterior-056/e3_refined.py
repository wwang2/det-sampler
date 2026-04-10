"""
E3 Refined: Analytical divergence speedup benchmark.

Pure numpy implementation (no torch dependency).

Produces: figures/e3_analysis.png — 4-panel publication figure
  (a) Speedup ratio vs dimension (log-log)
  (b) Divergence estimation variance vs dimension
  (c) Relative error vs dimension
  (d) Effective throughput (steps/sec at given accuracy)

Benchmarks:
  - Analytical: div(f) = -d * tanh(xi), exact, O(1)
  - Hutchinson(1): 1 random vector, stochastic
  - Hutchinson(5): 5 random vectors, stochastic
  - Brute-force: full Jacobian trace via finite differences, O(d^2)

All runs on CPU, 10 repeats for timing, 100 random vectors for variance estimation.
"""

import numpy as np
from scipy.special import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import os

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

# Style colors
C_NH = '#1f77b4'       # Analytical (NH blue)
C_HUTCH1 = '#ff7f0e'   # Hutchinson(1) orange
C_HUTCH5 = '#2ca02c'   # Hutchinson(5) green
C_BRUTE = '#d62728'    # Brute force red


# =============================================================================
# GMM potential and gradient (pure numpy)
# =============================================================================

def gmm_potential_and_grad(x, means, weights):
    """Gaussian mixture model: V(x) = -log sum_k w_k N(x; mu_k, I).
    x: (d,), means: (K, d), weights: (K,)
    Returns (V, grad_V) both as numpy arrays.
    """
    d = x.shape[-1]
    diff = x[np.newaxis, :] - means  # (K, d)
    log_probs = -0.5 * np.sum(diff**2, axis=-1) - 0.5 * d * np.log(2 * np.pi)
    log_probs = log_probs + np.log(weights)
    log_sum = logsumexp(log_probs)
    V = -log_sum
    # Gradient: sum_k alpha_k * (x - mu_k)
    alpha = np.exp(log_probs - logsumexp(log_probs))  # softmax
    grad_V = np.sum(alpha[:, np.newaxis] * diff, axis=0)
    return V, grad_V


# =============================================================================
# NH-tanh RK4 step (pure numpy)
# =============================================================================

def nh_tanh_rk4_step(q, p, xi, grad_V_fn, dt, Q=1.0, kT=1.0, d=None):
    """One RK4 step of NH-tanh ODE. Returns (q, p, xi, div_integral)."""
    if d is None:
        d = q.shape[-1]

    def f(q_, p_, xi_):
        _, gv = grad_V_fn(q_)
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
# Setup GMM for a given dimension
# =============================================================================

def make_gmm(d, K=5, seed=42):
    """Create a K-component GMM in d dimensions. Returns grad_fn."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2*np.pi, K, endpoint=False)
    means = np.zeros((K, d))
    means[:, 0] = 3.0 * np.cos(angles)
    if d >= 2:
        means[:, 1] = 3.0 * np.sin(angles)
    weights = np.ones(K) / K

    def grad_fn(x, m=means, w=weights):
        _, g = gmm_potential_and_grad(x, m, w)
        return g

    def potential_and_grad_fn(x, m=means, w=weights):
        return gmm_potential_and_grad(x, m, w)

    return potential_and_grad_fn, grad_fn, means, weights


# =============================================================================
# Hutchinson divergence estimator (finite differences)
# =============================================================================

def hutchinson_div_step(q, p, xi, grad_fn, d, n_vec, rng):
    """Compute Hutchinson trace estimate of d(dp/dt)/dp.

    dp/dt = -grad_V(q) - tanh(xi)*p
    True: d(dp/dt)/dp = -tanh(xi)*I, trace = -d*tanh(xi)

    Hutchinson uses finite differences to estimate v^T J v.
    """
    g = np.tanh(xi)
    eps_fd = 1e-4
    gv = grad_fn(q)  # this returns just the gradient

    estimates = []
    for _ in range(n_vec):
        v = rng.randn(d)
        f_plus = -gv - g * (p + eps_fd * v)
        f_minus = -gv - g * (p - eps_fd * v)
        jv_approx = (f_plus - f_minus) / (2 * eps_fd)
        trace_est = np.dot(v, jv_approx)
        estimates.append(trace_est)

    return np.mean(estimates), estimates


# =============================================================================
# Brute force Jacobian trace (finite differences, O(d^2))
# =============================================================================

def brute_force_div(q, p, xi, grad_fn, d):
    """Compute full Jacobian trace via column-by-column finite differences."""
    g = np.tanh(xi)
    eps_fd = 1e-5
    gv = grad_fn(q)

    trace = 0.0
    for i in range(d):
        e_i = np.zeros(d)
        e_i[i] = 1.0
        f_plus = -gv - g * (p + eps_fd * e_i)
        f_minus = -gv - g * (p - eps_fd * e_i)
        df_dp_i = (f_plus - f_minus) / (2 * eps_fd)
        trace += df_dp_i[i]

    return trace


# =============================================================================
# Benchmark: timing with multiple repeats
# =============================================================================

def benchmark_timing(d, n_steps=200, n_repeats=10, dt=0.01, Q=1.0, kT=1.0):
    """Time each divergence method over n_steps, repeated n_repeats times."""
    pot_grad_fn, grad_fn, _, _ = make_gmm(d)
    results = {}

    # --- Analytical ---
    times_ana = []
    for rep in range(n_repeats):
        rng = np.random.RandomState(SEED + rep)
        q = rng.randn(d)
        p = rng.randn(d)
        xi = 0.0

        t0 = time.perf_counter()
        for _ in range(n_steps):
            q, p, xi, div_int = nh_tanh_rk4_step(q, p, xi, pot_grad_fn, dt, Q, kT, d)
            # Analytical divergence is already computed inside the step
        t1 = time.perf_counter()
        times_ana.append(t1 - t0)

    results['analytical'] = {
        'times': np.array(times_ana),
        'mean': np.mean(times_ana),
        'std': np.std(times_ana, ddof=1),
        'per_step_mean': np.mean(times_ana) / n_steps,
        'per_step_std': np.std(times_ana, ddof=1) / n_steps,
    }

    # --- Hutchinson(1) and Hutchinson(5) ---
    for n_vec, label in [(1, 'hutch1'), (5, 'hutch5')]:
        times_h = []
        for rep in range(n_repeats):
            rng = np.random.RandomState(SEED + rep)
            q = rng.randn(d)
            p = rng.randn(d)
            xi = 0.0
            hutch_rng = np.random.RandomState(SEED + 5000 + rep)

            t0 = time.perf_counter()
            for _ in range(n_steps):
                q, p, xi, _ = nh_tanh_rk4_step(q, p, xi, pot_grad_fn, dt, Q, kT, d)
                # Extra cost: Hutchinson trace estimation
                g = np.tanh(xi)
                eps_fd = 1e-4
                gv = grad_fn(q)
                for _ in range(n_vec):
                    v = hutch_rng.randn(d)
                    f_plus = -gv - g * (p + eps_fd * v)
                    f_minus = -gv - g * (p - eps_fd * v)
                    trace_est = np.dot(v, (f_plus - f_minus) / (2 * eps_fd))
            t1 = time.perf_counter()
            times_h.append(t1 - t0)

        results[label] = {
            'times': np.array(times_h),
            'mean': np.mean(times_h),
            'std': np.std(times_h, ddof=1),
            'per_step_mean': np.mean(times_h) / n_steps,
            'per_step_std': np.std(times_h, ddof=1) / n_steps,
        }

    # --- Brute force (skip for d > 100) ---
    if d <= 100:
        times_bf = []
        for rep in range(n_repeats):
            rng = np.random.RandomState(SEED + rep)
            q = rng.randn(d)
            p = rng.randn(d)
            xi = 0.0

            t0 = time.perf_counter()
            for _ in range(n_steps):
                q, p, xi, _ = nh_tanh_rk4_step(q, p, xi, pot_grad_fn, dt, Q, kT, d)
                # Brute force trace
                g = np.tanh(xi)
                eps_fd = 1e-5
                gv = grad_fn(q)
                for i in range(d):
                    e_i = np.zeros(d)
                    e_i[i] = 1.0
                    f_plus = -gv - g * (p + eps_fd * e_i)
                    f_minus = -gv - g * (p - eps_fd * e_i)
                    df_dp_i = (f_plus - f_minus) / (2 * eps_fd)
                    # trace += df_dp_i[i]  (we just need to do the work)
            t1 = time.perf_counter()
            times_bf.append(t1 - t0)

        results['brute'] = {
            'times': np.array(times_bf),
            'mean': np.mean(times_bf),
            'std': np.std(times_bf, ddof=1),
            'per_step_mean': np.mean(times_bf) / n_steps,
            'per_step_std': np.std(times_bf, ddof=1) / n_steps,
        }
    else:
        results['brute'] = None

    return results


# =============================================================================
# Variance and error analysis at a single state
# =============================================================================

def variance_and_error_analysis(d, n_draws=100):
    """At a single (q, p, xi) state, compute variance and error of each method."""
    pot_grad_fn, grad_fn, _, _ = make_gmm(d)

    # Run a few steps to get a non-trivial state
    rng = np.random.RandomState(SEED)
    q = rng.randn(d)
    p = rng.randn(d)
    xi = 0.0
    for _ in range(50):
        q, p, xi, _ = nh_tanh_rk4_step(q, p, xi, pot_grad_fn, 0.01, 1.0, 1.0, d)

    # True divergence
    true_div = -d * np.tanh(xi)

    g = np.tanh(xi)
    eps_fd = 1e-4
    gv = grad_fn(q)

    # Hutchinson(1): n_draws independent estimates
    hutch_rng = np.random.RandomState(SEED + 1000)
    hutch1_estimates = []
    for _ in range(n_draws):
        v = hutch_rng.randn(d)
        f_plus = -gv - g * (p + eps_fd * v)
        f_minus = -gv - g * (p - eps_fd * v)
        trace_est = np.dot(v, (f_plus - f_minus) / (2 * eps_fd))
        hutch1_estimates.append(trace_est)
    hutch1_estimates = np.array(hutch1_estimates)

    # Hutchinson(5): n_draws independent estimates, each averaging 5 vectors
    hutch_rng5 = np.random.RandomState(SEED + 2000)
    hutch5_estimates = []
    for _ in range(n_draws):
        traces = []
        for _ in range(5):
            v = hutch_rng5.randn(d)
            f_plus = -gv - g * (p + eps_fd * v)
            f_minus = -gv - g * (p - eps_fd * v)
            trace_est = np.dot(v, (f_plus - f_minus) / (2 * eps_fd))
            traces.append(trace_est)
        hutch5_estimates.append(np.mean(traces))
    hutch5_estimates = np.array(hutch5_estimates)

    # Brute force (for d <= 100)
    if d <= 100:
        bf_div = brute_force_div(q, p, xi, grad_fn, d)
    else:
        bf_div = true_div  # Same by definition for this system

    abs_true = max(abs(true_div), 1e-10)

    return {
        'true_div': true_div,
        'hutch1_mean': np.mean(hutch1_estimates),
        'hutch1_var': np.var(hutch1_estimates, ddof=1),
        'hutch1_std': np.std(hutch1_estimates, ddof=1),
        'hutch1_rel_error': np.mean(np.abs(hutch1_estimates - true_div)) / abs_true,
        'hutch5_mean': np.mean(hutch5_estimates),
        'hutch5_var': np.var(hutch5_estimates, ddof=1),
        'hutch5_std': np.std(hutch5_estimates, ddof=1),
        'hutch5_rel_error': np.mean(np.abs(hutch5_estimates - true_div)) / abs_true,
        'brute_div': bf_div,
        'brute_error': abs(bf_div - true_div) / abs_true,
        'analytical_error': 0.0,
    }


# =============================================================================
# Main
# =============================================================================

def run_e3_refined():
    print("=" * 70)
    print("E3 REFINED: Analytical Divergence Speedup Benchmark")
    print("=" * 70)

    dims = [2, 5, 10, 20, 50, 100, 200, 500]
    n_steps = 200
    n_repeats = 10
    n_draws = 100

    all_timing = {}
    all_variance = {}

    for d in dims:
        print(f"\n--- Dimension d={d} ---")

        # Timing benchmark
        n_reps_actual = n_repeats if d <= 100 else 5  # fewer reps for large d
        n_steps_actual = n_steps if d <= 200 else 100  # fewer steps for d=500
        print(f"  Timing ({n_reps_actual} repeats, {n_steps_actual} steps each)...")
        timing = benchmark_timing(d, n_steps=n_steps_actual, n_repeats=n_reps_actual)
        all_timing[d] = timing

        t_ana = timing['analytical']['per_step_mean']
        t_h1 = timing['hutch1']['per_step_mean']
        t_h5 = timing['hutch5']['per_step_mean']

        print(f"    Analytical:   {t_ana*1e3:.3f} ms/step")
        print(f"    Hutch(1):     {t_h1*1e3:.3f} ms/step  ({t_h1/t_ana:.2f}x)")
        print(f"    Hutch(5):     {t_h5*1e3:.3f} ms/step  ({t_h5/t_ana:.2f}x)")
        if timing['brute'] is not None:
            t_bf = timing['brute']['per_step_mean']
            print(f"    Brute force:  {t_bf*1e3:.3f} ms/step  ({t_bf/t_ana:.2f}x)")

        # Variance / error analysis
        print(f"  Variance analysis ({n_draws} draws)...")
        var_result = variance_and_error_analysis(d, n_draws=n_draws)
        all_variance[d] = var_result

        print(f"    True div:      {var_result['true_div']:.4f}")
        print(f"    Hutch(1) var:  {var_result['hutch1_var']:.4f}, rel_err: {var_result['hutch1_rel_error']:.4f}")
        print(f"    Hutch(5) var:  {var_result['hutch5_var']:.4f}, rel_err: {var_result['hutch5_rel_error']:.4f}")

    # =========================================================================
    # Figure: e3_analysis.png — 4-panel publication figure
    # =========================================================================
    print("\n--- Generating e3_analysis.png ---")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    dims_arr = np.array(dims)

    # Extract timing data
    t_ana_arr = np.array([all_timing[d]['analytical']['per_step_mean'] for d in dims])
    t_ana_std = np.array([all_timing[d]['analytical']['per_step_std'] for d in dims])
    t_h1_arr = np.array([all_timing[d]['hutch1']['per_step_mean'] for d in dims])
    t_h1_std = np.array([all_timing[d]['hutch1']['per_step_std'] for d in dims])
    t_h5_arr = np.array([all_timing[d]['hutch5']['per_step_mean'] for d in dims])
    t_h5_std = np.array([all_timing[d]['hutch5']['per_step_std'] for d in dims])

    dims_bf = [d for d in dims if all_timing[d]['brute'] is not None]
    t_bf_arr = np.array([all_timing[d]['brute']['per_step_mean'] for d in dims_bf])
    t_bf_std = np.array([all_timing[d]['brute']['per_step_std'] for d in dims_bf])

    # ---- Panel (a): Speedup ratio vs dimension ----
    ax = axes[0, 0]

    speedup_h1 = t_h1_arr / t_ana_arr
    speedup_h5 = t_h5_arr / t_ana_arr
    speedup_bf = t_bf_arr / np.array([all_timing[d]['analytical']['per_step_mean'] for d in dims_bf])

    # Error propagation for ratio: sigma(a/b) ~ (a/b)*sqrt((sa/a)^2+(sb/b)^2)
    speedup_h1_err = speedup_h1 * np.sqrt((t_h1_std/t_h1_arr)**2 + (t_ana_std/t_ana_arr)**2)
    speedup_h5_err = speedup_h5 * np.sqrt((t_h5_std/t_h5_arr)**2 + (t_ana_std/t_ana_arr)**2)
    t_ana_bf = np.array([all_timing[d]['analytical']['per_step_std'] for d in dims_bf])
    t_ana_bf_m = np.array([all_timing[d]['analytical']['per_step_mean'] for d in dims_bf])
    speedup_bf_err = speedup_bf * np.sqrt((t_bf_std/t_bf_arr)**2 + (t_ana_bf/t_ana_bf_m)**2)

    ax.errorbar(dims_arr, speedup_h1, yerr=speedup_h1_err, fmt='o-', color=C_HUTCH1,
                label='Hutch(1) / Analytical', markersize=6, linewidth=2, capsize=3)
    ax.errorbar(dims_arr, speedup_h5, yerr=speedup_h5_err, fmt='s-', color=C_HUTCH5,
                label='Hutch(5) / Analytical', markersize=6, linewidth=2, capsize=3)
    ax.errorbar(np.array(dims_bf), speedup_bf, yerr=speedup_bf_err, fmt='D--', color=C_BRUTE,
                label='Brute / Analytical', markersize=6, linewidth=2, capsize=3)

    ax.axhline(1.0, color='gray', ls=':', alpha=0.5, label='Analytical baseline')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Dimension $d$')
    ax.set_ylabel('Cost ratio (method / analytical)')
    ax.set_title('(a) Computational cost scaling', fontweight='bold')
    ax.legend(frameon=False, fontsize=10, loc='upper left')

    # Reference O(d) slope for brute force
    if len(dims_bf) >= 2:
        d_ref = np.array([dims_bf[0], dims_bf[-1]])
        bf_start = speedup_bf[0]
        slope_ref = bf_start * (d_ref / d_ref[0])**1.0
        ax.plot(d_ref, slope_ref, ':', color=C_BRUTE, alpha=0.3, linewidth=1)
        ax.text(d_ref[-1]*0.6, slope_ref[-1]*1.3, '$O(d)$', fontsize=10,
                color=C_BRUTE, alpha=0.5)

    # ---- Panel (b): Divergence estimation variance vs dimension ----
    ax = axes[0, 1]

    var_h1 = np.array([all_variance[d]['hutch1_var'] for d in dims])
    var_h5 = np.array([all_variance[d]['hutch5_var'] for d in dims])

    ax.semilogy(dims_arr, var_h1, 'o-', color=C_HUTCH1, label='Hutch(1)',
                markersize=6, linewidth=2)
    ax.semilogy(dims_arr, var_h5, 's-', color=C_HUTCH5, label='Hutch(5)',
                markersize=6, linewidth=2)

    # Analytical: exact, zero variance — plot at machine epsilon
    eps_line = np.finfo(np.float32).eps
    ax.axhline(eps_line, color=C_NH, ls='--', linewidth=2,
               label='Analytical (exact)', alpha=0.8)
    ax.fill_between([dims_arr[0]*0.7, dims_arr[-1]*1.4], 0, eps_line,
                    color=C_NH, alpha=0.08)
    ax.text(dims_arr[-1]*0.5, eps_line*3, 'Zero variance', fontsize=10,
            color=C_NH, fontweight='bold', ha='center')

    ax.set_xscale('log')
    ax.set_xlabel('Dimension $d$')
    ax.set_ylabel('Variance of trace estimate')
    ax.set_title('(b) Estimation variance', fontweight='bold')
    ax.legend(frameon=False, fontsize=10)
    ax.set_xlim(dims_arr[0]*0.7, dims_arr[-1]*1.4)

    # ---- Panel (c): Relative error vs dimension ----
    ax = axes[1, 0]

    rel_err_h1 = np.array([all_variance[d]['hutch1_rel_error'] for d in dims])
    rel_err_h5 = np.array([all_variance[d]['hutch5_rel_error'] for d in dims])

    ax.semilogy(dims_arr, rel_err_h1, 'o-', color=C_HUTCH1, label='Hutch(1)',
                markersize=6, linewidth=2)
    ax.semilogy(dims_arr, rel_err_h5, 's-', color=C_HUTCH5, label='Hutch(5)',
                markersize=6, linewidth=2)

    # Analytical: always exact — numerical precision floor
    ax.axhline(1e-7, color=C_NH, ls='--', linewidth=2,
               label='Analytical (float64 floor)', alpha=0.8)

    ax.set_xscale('log')
    ax.set_xlabel('Dimension $d$')
    ax.set_ylabel('Mean relative error')
    ax.set_title('(c) Divergence estimation error', fontweight='bold')
    ax.legend(frameon=False, fontsize=10)

    # ---- Panel (d): Effective throughput ----
    ax = axes[1, 1]

    # Analytical throughput: always achieves any accuracy target
    throughput_ana = 1.0 / t_ana_arr
    ax.plot(dims_arr, throughput_ana, 'o-', color=C_NH, label='Analytical (exact)',
            markersize=7, linewidth=2.5, zorder=5)

    # For Hutchinson: estimate n_vectors needed for target accuracy
    # rel_error ~ std_1/sqrt(n) / |true_div| => n = var_1 / (eps * |true_div|)^2
    # Cost = t_base + n * t_extra_per_vec
    target_errors = [0.1, 0.05, 0.01]
    markers = ['v', '^', 'x']
    for idx, target_eps in enumerate(target_errors):
        throughputs = []
        for i, d in enumerate(dims):
            var1 = all_variance[d]['hutch1_var']
            true_div = abs(all_variance[d]['true_div'])
            if true_div < 1e-10:
                throughputs.append(np.nan)
                continue

            n_needed = var1 / (target_eps * true_div)**2
            n_needed = max(1.0, n_needed)

            t_extra_per_vec = max(0, (t_h1_arr[i] - t_ana_arr[i]))
            t_total = t_ana_arr[i] + n_needed * t_extra_per_vec
            throughputs.append(1.0 / t_total)

        ax.plot(dims_arr, throughputs, f'{markers[idx]}--', color=C_HUTCH1,
                label=f'Hutch, $\\epsilon={target_eps}$', linewidth=1.5,
                markersize=5, alpha=0.7)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Dimension $d$')
    ax.set_ylabel('Steps per second')
    ax.set_title('(d) Effective throughput at target accuracy', fontweight='bold')
    ax.legend(frameon=False, fontsize=9, loc='lower left')

    outpath = os.path.join(FIGDIR, 'e3_analysis.png')
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {outpath}")

    # =========================================================================
    # Print summary tables for log.md
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY TABLE: Wall-clock time per step (ms)")
    print("=" * 70)
    header = f"{'d':>5} | {'Analytical':>14} | {'Hutch(1)':>14} | {'Hutch(5)':>14} | {'Brute':>14} | {'H1/Ana':>8} | {'H5/Ana':>8} | {'BF/Ana':>8}"
    print(header)
    print("-" * len(header))
    for d in dims:
        t_a = all_timing[d]['analytical']['per_step_mean'] * 1e3
        t_a_s = all_timing[d]['analytical']['per_step_std'] * 1e3
        t_1 = all_timing[d]['hutch1']['per_step_mean'] * 1e3
        t_1_s = all_timing[d]['hutch1']['per_step_std'] * 1e3
        t_5 = all_timing[d]['hutch5']['per_step_mean'] * 1e3
        t_5_s = all_timing[d]['hutch5']['per_step_std'] * 1e3

        ratio_1 = t_1 / t_a
        ratio_5 = t_5 / t_a

        if all_timing[d]['brute'] is not None:
            t_b = all_timing[d]['brute']['per_step_mean'] * 1e3
            t_b_s = all_timing[d]['brute']['per_step_std'] * 1e3
            ratio_b = t_b / t_a
            print(f"{d:>5} | {t_a:>6.2f}+/-{t_a_s:>5.2f} | {t_1:>6.2f}+/-{t_1_s:>5.2f} | {t_5:>6.2f}+/-{t_5_s:>5.2f} | {t_b:>6.2f}+/-{t_b_s:>5.2f} | {ratio_1:>7.2f}x | {ratio_5:>7.2f}x | {ratio_b:>7.2f}x")
        else:
            print(f"{d:>5} | {t_a:>6.2f}+/-{t_a_s:>5.2f} | {t_1:>6.2f}+/-{t_1_s:>5.2f} | {t_5:>6.2f}+/-{t_5_s:>5.2f} | {'(too slow)':>14} | {ratio_1:>7.2f}x | {ratio_5:>7.2f}x | {'N/A':>8}")

    print("\n" + "=" * 70)
    print("SUMMARY TABLE: Variance and relative error")
    print("=" * 70)
    print(f"{'d':>5} | {'True div':>10} | {'H1 var':>10} | {'H5 var':>10} | {'H1 rel_err':>10} | {'H5 rel_err':>10}")
    print("-" * 70)
    for d in dims:
        v = all_variance[d]
        print(f"{d:>5} | {v['true_div']:>10.4f} | {v['hutch1_var']:>10.4f} | {v['hutch5_var']:>10.4f} | {v['hutch1_rel_error']:>10.4f} | {v['hutch5_rel_error']:>10.4f}")

    return all_timing, all_variance


if __name__ == '__main__':
    run_e3_refined()
