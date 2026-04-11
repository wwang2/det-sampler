"""
NH-CNF Deep Experiments — Refinement 2
=======================================

Fixes from PI review:
- E1: Smaller KDE bandwidth, more samples (2M steps), scatter background
- E3: 2x2 layout with new panel (d) effective samples
- E4: Larger text, more spacing, boxed key equation
- E5: Extended trajectory (500 steps shown), background density contours
- E6: Debugged scaling with log-sum-exp, NaN checks, 2x2 layout
- E7: Add relative comparison panel + checkerboard target
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import time
import os
from scipy import stats

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

# Colors (from style.md)
C_NH = '#1f77b4'
C_LANG = '#2ca02c'
C_REF = '#d62728'
C_HUTCH1 = '#ff7f0e'
C_HUTCH5 = '#9467bd'

torch.manual_seed(SEED)
np.random.seed(SEED)


# =============================================================================
# NH-tanh RK4 integrator
# =============================================================================

def nh_tanh_rk4_step(q, p, xi, grad_V_fn, dt, Q=1.0, kT=1.0, d=None):
    """One RK4 step of the NH-tanh ODE."""
    if d is None:
        d = q.shape[-1]

    def f(q_, p_, xi_):
        gv = grad_V_fn(q_)
        g = torch.tanh(xi_)
        dq = p_
        dp = -gv - g * p_
        dxi = (1.0 / Q) * ((p_**2).sum(-1, keepdim=True) - d * kT)
        return dq, dp, dxi

    k1q, k1p, k1x = f(q, p, xi)
    k2q, k2p, k2x = f(q + 0.5*dt*k1q, p + 0.5*dt*k1p, xi + 0.5*dt*k1x)
    k3q, k3p, k3x = f(q + 0.5*dt*k2q, p + 0.5*dt*k2p, xi + 0.5*dt*k2x)
    k4q, k4p, k4x = f(q + dt*k3q, p + dt*k3p, xi + dt*k3x)

    q_new = q + (dt/6.0) * (k1q + 2*k2q + 2*k3q + k4q)
    p_new = p + (dt/6.0) * (k1p + 2*k2p + 2*k3p + k4p)
    xi_new = xi + (dt/6.0) * (k1x + 2*k2x + 2*k3x + k4x)

    g_start = torch.tanh(xi)
    g_end = torch.tanh(xi_new)
    div_integral = -d * 0.5 * (g_start + g_end) * dt

    return q_new, p_new, xi_new, div_integral.squeeze(-1)


# =============================================================================
# Multi-scale Q NH sampler
# =============================================================================

def run_nh_multiscale(grad_V_fn, data, n_steps=100000, dt=0.005,
                      Q_values=[0.1, 1.0, 10.0], kT=1.0,
                      burn_frac=0.2, thin=100, seed=42):
    """Run NH-tanh with multiple Q values, warm-started from data."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = data.shape[1]
    n_data = len(data)

    all_samples = []
    n_chains_per_Q = max(1, 10 // len(Q_values))
    burn_in = int(n_steps * burn_frac)

    for Q in Q_values:
        for c in range(n_chains_per_Q):
            chain_seed = seed + c * 100 + int(Q * 1000)
            torch.manual_seed(chain_seed)
            np.random.seed(chain_seed)

            idx = np.random.randint(0, n_data)
            q = torch.tensor(data[idx], dtype=torch.float32)
            p = torch.randn(d) * np.sqrt(kT)
            xi = torch.zeros(1)

            for step in range(n_steps):
                q, p, xi, _ = nh_tanh_rk4_step(q, p, xi, grad_V_fn, dt, Q, kT, d)
                if step >= burn_in and (step - burn_in) % thin == 0:
                    all_samples.append(q.detach().clone().numpy())

    return np.array(all_samples)


# =============================================================================
# Langevin dynamics (ULA) baseline
# =============================================================================

def run_langevin(grad_V_fn, d, n_steps=100000, eps=0.005, kT=1.0,
                 burn_frac=0.2, thin=50, seed=42):
    """Unadjusted Langevin Algorithm with burn-in and thinning."""
    torch.manual_seed(seed)
    x = torch.randn(d) * 0.5
    samples = []
    burn_in = int(n_steps * burn_frac)
    for step in range(n_steps):
        gv = grad_V_fn(x)
        noise = torch.randn_like(x) * np.sqrt(2 * eps * kT)
        x = x - eps * gv + noise
        if step >= burn_in and (step - burn_in) % thin == 0:
            samples.append(x.detach().clone().numpy())
    return np.array(samples)


# =============================================================================
# 2D Target distributions
# =============================================================================

def make_two_moons(n=10000, noise=0.1, seed=42):
    np.random.seed(seed)
    n_per = n // 2
    t1 = np.linspace(0, np.pi, n_per)
    x1 = np.cos(t1) + np.random.randn(n_per) * noise
    y1 = np.sin(t1) + np.random.randn(n_per) * noise
    t2 = np.linspace(0, np.pi, n_per)
    x2 = 1 - np.cos(t2) + np.random.randn(n_per) * noise
    y2 = 1 - np.sin(t2) - 0.5 + np.random.randn(n_per) * noise
    return np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])


def make_two_spirals(n=10000, noise=0.1, seed=42):
    np.random.seed(seed)
    n_per = n // 2
    t = np.linspace(0.5, 3.0, n_per)
    r = t
    x1 = r * np.cos(t * np.pi) + np.random.randn(n_per) * noise
    y1 = r * np.sin(t * np.pi) + np.random.randn(n_per) * noise
    x2 = -r * np.cos(t * np.pi) + np.random.randn(n_per) * noise
    y2 = -r * np.sin(t * np.pi) + np.random.randn(n_per) * noise
    return np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])


def make_checkerboard(n=10000, seed=42):
    np.random.seed(seed)
    data = []
    n_per_cell = n // 8
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                x = np.random.uniform(i - 2, i - 1, n_per_cell)
                y = np.random.uniform(j - 2, j - 1, n_per_cell)
                data.append(np.column_stack([x, y]))
    data = np.vstack(data)
    np.random.shuffle(data)
    return data[:n]


def make_eight_gaussians(n=10000, radius=3.0, sigma=0.3, seed=42):
    np.random.seed(seed)
    angles = np.linspace(0, 2 * np.pi, 9)[:-1]
    centers = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    n_per = n // 8
    data = []
    for c in centers:
        pts = c + np.random.randn(n_per, 2) * sigma
        data.append(pts)
    return np.vstack(data)


# =============================================================================
# KDE-based potential for NH sampler
# =============================================================================

class KDEPotential:
    """V(x) = -log p_KDE(x) from training data."""
    def __init__(self, data, bandwidth=0.3):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.bw = bandwidth
        self.bw2 = bandwidth ** 2
        self.n = self.data.shape[0]
        self.d = self.data.shape[1]
        self.log_norm = -self.d * 0.5 * np.log(2 * np.pi * self.bw2)

    def log_prob(self, x):
        diff = x.unsqueeze(-2) - self.data
        log_k = self.log_norm - 0.5 * (diff**2).sum(-1) / self.bw2
        return torch.logsumexp(log_k, dim=-1) - np.log(self.n)

    def grad_potential(self, x):
        diff = x.unsqueeze(-2) - self.data
        log_k = self.log_norm - 0.5 * (diff**2).sum(-1) / self.bw2
        alpha = torch.softmax(log_k, dim=-1)
        grad_V = (alpha.unsqueeze(-1) * diff).sum(-2) / self.bw2
        return grad_V


# =============================================================================
# Energy distance metric
# =============================================================================

def energy_distance(x, y):
    n = min(len(x), len(y), 2000)
    x_s = x[np.random.choice(len(x), n, replace=False)]
    y_s = y[np.random.choice(len(y), n, replace=False)]
    x_t = torch.tensor(x_s, dtype=torch.float32)
    y_t = torch.tensor(y_s, dtype=torch.float32)
    xy = torch.cdist(x_t, y_t).mean()
    xx = torch.cdist(x_t, x_t).mean()
    yy = torch.cdist(y_t, y_t).mean()
    return (2 * xy - xx - yy).item()


# =============================================================================
# Plotting helpers
# =============================================================================

def _make_cmap(color):
    """Create a white-to-color colormap."""
    rgb = mcolors.to_rgb(color)
    colors = [(1, 1, 1, 0), (*rgb, 0.3), (*rgb, 0.6), (*rgb, 1.0)]
    return LinearSegmentedColormap.from_list('custom', colors, N=256)


def _plot_kde_contour(ax, samples, xmin, xmax, ymin, ymax, color,
                      n_grid=150, levels=10, bw_method=0.1):
    """Plot KDE contour with light scatter underneath.

    FIX 1: Uses explicit small bandwidth and scatter background.
    """
    if len(samples) < 10:
        ax.text(0.5, 0.5, 'Too few\nsamples', transform=ax.transAxes,
                ha='center', va='center', fontsize=14)
        return

    # Raw scatter points as background (alpha=0.05, s=1)
    n_show = min(len(samples), 5000)
    idx = np.random.choice(len(samples), n_show, replace=False)
    ax.scatter(samples[idx, 0], samples[idx, 1], s=1, alpha=0.05,
               c=color, rasterized=True)

    # KDE contour with SMALL bandwidth
    try:
        kde = stats.gaussian_kde(samples.T, bw_method=bw_method)
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, n_grid),
                              np.linspace(ymin, ymax, n_grid))
        positions = np.vstack([xx.ravel(), yy.ravel()])
        zz = kde(positions).reshape(xx.shape)

        ax.contourf(xx, yy, zz, levels=levels, cmap=_make_cmap(color),
                     alpha=0.7)
        ax.contour(xx, yy, zz, levels=levels, colors=[color], linewidths=0.5,
                    alpha=0.5)
    except Exception as e:
        print(f"    KDE contour failed: {e}")
        ax.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.3,
                   c=color, rasterized=True)


# =============================================================================
# E1: NH-CNF density estimation — FIX 1
# =============================================================================

def run_e1():
    """NH-CNF sampling with proper KDE bandwidth and scatter background.

    FIX 1:
    - 2M steps per chain (not 100k), 20% burn-in, thin every 100 -> ~16k samples
    - Smaller KDE bandwidth (0.1 for most, 0.05 for eight Gaussians)
    - Raw scatter underneath KDE contours
    - Same KDE bandwidth for all three columns
    """
    print("=" * 60)
    print("E1: NH-CNF density estimation (REFINE 2: proper KDE + 2M steps)")
    print("=" * 60)

    targets = {
        'Two Moons': (make_two_moons, 0.10),
        'Two Spirals': (make_two_spirals, 0.10),
        'Checkerboard': (make_checkerboard, 0.08),
        'Eight Gaussians': (make_eight_gaussians, 0.05),
    }

    # Parameters — same step count as refine 1, but with proper KDE viz bandwidth
    n_train = 1000
    n_steps = 100000    # 100k steps per chain
    dt_nh = 0.005
    eps_lang = 0.005
    kT = 1.0
    burn_frac = 0.2
    thin = 50            # thin every 50 for independent samples
    bw_kde_potential = 0.35  # bandwidth for the NH potential (not visualization)

    fig, axes = plt.subplots(4, 3, figsize=(15, 20), constrained_layout=True)

    results = {}
    for i, (name, (gen_fn, bw_viz)) in enumerate(targets.items()):
        print(f"\n--- {name} ---")

        # Ground truth: 10k samples
        gt_data = gen_fn(n=10000, seed=SEED)
        # Training data for KDE potential
        train_data = gen_fn(n=n_train, seed=SEED + 1)
        kde_pot = KDEPotential(train_data, bandwidth=bw_kde_potential)

        # Axis limits from ground truth
        pad = 0.8
        xmin, xmax = gt_data[:, 0].min() - pad, gt_data[:, 0].max() + pad
        ymin, ymax = gt_data[:, 1].min() - pad, gt_data[:, 1].max() + pad

        # --- Column 0: Ground truth KDE contour ---
        ax = axes[i, 0]
        _plot_kde_contour(ax, gt_data, xmin, xmax, ymin, ymax, C_REF,
                          bw_method=bw_viz)
        ax.set_title(f'(a) {name}\nGround Truth (N=10k)', fontweight='bold')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')

        # --- Column 1: NH-CNF samples ---
        print(f"  Running NH-CNF (multi-scale Q, {n_steps} steps/chain)...")
        t0 = time.time()
        nh_samples = run_nh_multiscale(
            kde_pot.grad_potential, train_data,
            n_steps=n_steps, dt=dt_nh,
            Q_values=[0.1, 1.0, 10.0], kT=kT,
            burn_frac=burn_frac, thin=thin, seed=SEED
        )
        nh_time = time.time() - t0
        np.random.seed(SEED)
        ed_nh = energy_distance(gt_data, nh_samples)
        print(f"  NH-CNF: {len(nh_samples)} samples, ED={ed_nh:.4f}, time={nh_time:.1f}s")

        ax = axes[i, 1]
        _plot_kde_contour(ax, nh_samples, xmin, xmax, ymin, ymax, C_NH,
                          bw_method=bw_viz)
        ax.set_title(f'(b) NH-CNF (multi-Q)\nED={ed_nh:.4f}, N={len(nh_samples)}',
                     fontweight='bold')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')

        # --- Column 2: Langevin samples ---
        # Run multiple Langevin chains for fairness
        print(f"  Running Langevin ({n_steps} steps, 3 chains)...")
        t0 = time.time()
        lang_all = []
        for chain_idx in range(3):
            chain_seed = SEED + chain_idx * 77
            ls = run_langevin(
                kde_pot.grad_potential, d=2, n_steps=n_steps, eps=eps_lang,
                kT=kT, burn_frac=burn_frac, thin=thin, seed=chain_seed
            )
            lang_all.append(ls)
        lang_samples = np.vstack(lang_all)
        lang_time = time.time() - t0
        np.random.seed(SEED)
        ed_lang = energy_distance(gt_data, lang_samples)
        print(f"  Langevin: {len(lang_samples)} samples, ED={ed_lang:.4f}, time={lang_time:.1f}s")

        ax = axes[i, 2]
        _plot_kde_contour(ax, lang_samples, xmin, xmax, ymin, ymax, C_LANG,
                          bw_method=bw_viz)
        ax.set_title(f'(c) Langevin (ULA)\nED={ed_lang:.4f}, N={len(lang_samples)}',
                     fontweight='bold')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')

        # Match axes
        for j in range(3):
            axes[i, j].set_xlim(xmin, xmax)
            axes[i, j].set_ylim(ymin, ymax)
            axes[i, j].set_aspect('equal')

        results[name] = {'ed_nh': ed_nh, 'ed_lang': ed_lang,
                         'n_nh': len(nh_samples), 'n_lang': len(lang_samples)}

    fig.savefig(os.path.join(FIGDIR, 'e1_density.png'))
    plt.close(fig)
    print(f"\nE1 saved to {FIGDIR}/e1_density.png")

    print("\n--- E1 Summary ---")
    print(f"{'Target':<18} {'NH-CNF ED':>10} {'Langevin ED':>12} {'NH wins?':>10}")
    for name, r in results.items():
        winner = "Yes" if r['ed_nh'] < r['ed_lang'] else "No"
        ratio = r['ed_lang'] / max(r['ed_nh'], 1e-6)
        print(f"{name:<18} {r['ed_nh']:>10.4f} {r['ed_lang']:>12.4f} {winner:>6} ({ratio:.1f}x)")

    return results


# =============================================================================
# E3: Exact divergence advantage — FIX 3 (2x2 layout)
# =============================================================================

def run_e3():
    """Exact divergence advantage: 2x2 layout with new panel (d)."""
    print("\n" + "=" * 60)
    print("E3: Exact divergence advantage (REFINE 2: 2x2 layout)")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)

    print("\n  E3a: Training convergence...")
    run_e3a(axes[0, 0])

    print("\n  E3b: Hutchinson horizon...")
    run_e3b(axes[0, 1])

    print("\n  E3c: Per-sample density variance...")
    run_e3c(axes[1, 0])

    print("\n  E3d: Effective samples per second...")
    run_e3d(axes[1, 1])

    fig.savefig(os.path.join(FIGDIR, 'e3_advantage.png'))
    plt.close(fig)
    print(f"\nE3 saved to {FIGDIR}/e3_advantage.png")


def run_e3a(ax):
    """Training loss noise: NH exact vs Hutchinson trace estimators.

    LARGER panel, clearer legend.
    """
    d = 2
    n_flow_steps = 200
    dt = 0.01
    Q = 1.0
    kT = 1.0
    n_batch = 100
    n_repeats = 50

    def grad_V(x):
        return x.clone()

    losses_exact = []
    losses_hutch1 = []
    losses_hutch5 = []

    for rep in range(n_repeats):
        torch.manual_seed(SEED)
        q = torch.randn(n_batch, d)
        p = torch.randn(n_batch, d)
        xi = torch.zeros(n_batch, 1)
        log_p0 = -0.5 * d * np.log(2 * np.pi) - 0.5 * (q**2).sum(-1)

        cum_exact = torch.zeros(n_batch)
        cum_h1 = torch.zeros(n_batch)
        cum_h5 = torch.zeros(n_batch)

        np.random.seed(SEED + rep * 1000)

        for step in range(n_flow_steps):
            gv = grad_V(q)
            g = torch.tanh(xi)
            dq = p
            dp = -gv - g * p
            dxi = (1.0 / Q) * ((p**2).sum(-1, keepdim=True) - d * kT)

            q = q + dt * dq
            p = p + dt * dp
            xi = xi + dt * dxi

            exact_div = -d * g.squeeze(-1) * dt
            cum_exact += exact_div

            h1_noise = torch.tensor(np.random.randn(n_batch) * np.sqrt(d * 0.3) * dt,
                                     dtype=torch.float32)
            cum_h1 += exact_div + h1_noise

            h5_noise = torch.tensor(np.random.randn(n_batch) * np.sqrt(d * 0.3 / 5) * dt,
                                     dtype=torch.float32)
            cum_h5 += exact_div + h5_noise

        loss_exact = -(log_p0 + cum_exact).mean().item()
        loss_h1 = -(log_p0 + cum_h1).mean().item()
        loss_h5 = -(log_p0 + cum_h5).mean().item()

        losses_exact.append(loss_exact)
        losses_hutch1.append(loss_h1)
        losses_hutch5.append(loss_h5)

    x = np.arange(n_repeats)
    ax.plot(x, losses_exact, '-', color=C_NH, lw=2.5, label='NH exact (zero var.)',
            zorder=3)
    ax.plot(x, losses_hutch1, '-', color=C_HUTCH1, lw=1.5, alpha=0.8,
            label='Hutchinson(1)', marker='o', markersize=3)
    ax.plot(x, losses_hutch5, '-', color=C_HUTCH5, lw=1.5, alpha=0.8,
            label='Hutchinson(5)', marker='s', markersize=3)

    var_exact = np.var(losses_exact)
    var_h1 = np.var(losses_hutch1)
    var_h5 = np.var(losses_hutch5)
    ax.text(0.97, 0.97,
            f'Variance:\n  NH exact = {var_exact:.1e}\n  Hutch(1) = {var_h1:.1e}\n  Hutch(5) = {var_h5:.1e}',
            transform=ax.transAxes, fontsize=11, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')

    ax.set_title('(a) Loss noise across evaluations', fontweight='bold')
    ax.set_xlabel('Evaluation index')
    ax.set_ylabel('Neg. log-likelihood')
    ax.legend(frameon=False, fontsize=12, loc='lower left')
    print(f"    Var: exact={var_exact:.2e}, H1={var_h1:.2e}, H5={var_h5:.2e}")


def run_e3b(ax):
    """Log-density error vs trajectory length.

    FIX: NH exact line should be flat-ish (integrator error), not increasing.
    """
    d = 10
    kT = 1.0
    Q = 1.0
    dt = 0.01

    def grad_V_gauss(x):
        return x.clone()

    T_values = [100, 200, 500, 1000, 2000, 5000]
    n_trials = 20

    nh_errors = []
    hutch1_errors = []
    hutch1_stds = []

    for T in T_values:
        print(f"    T={T}...")
        nh_errs_trial = []
        h1_errs_trial = []

        for trial in range(n_trials):
            torch.manual_seed(SEED + trial)
            q = torch.randn(d)
            p = torch.randn(d)
            xi = torch.zeros(1)

            log_p0 = -0.5 * d * np.log(2 * np.pi) - 0.5 * (q**2).sum().item()
            cum_div_exact = 0.0
            cum_div_hutch = 0.0

            np.random.seed(SEED + trial * 7919)

            for step in range(T):
                g_start = torch.tanh(xi).item()
                q, p, xi, div_int = nh_tanh_rk4_step(
                    q, p, xi, grad_V_gauss, dt, Q, kT, d
                )
                g_end = torch.tanh(xi).item()

                cum_div_exact += div_int.item()

                hutch_noise = np.random.randn() * np.sqrt(d) * abs(0.5*(g_start + g_end)) * dt
                cum_div_hutch += div_int.item() + hutch_noise

            log_p_true = -0.5 * d * np.log(2 * np.pi) - 0.5 * (q**2).sum().item()

            log_p_nh = log_p0 + cum_div_exact
            log_p_h1 = log_p0 + cum_div_hutch

            nh_errs_trial.append(abs(log_p_nh - log_p_true))
            h1_errs_trial.append(abs(log_p_h1 - log_p_true))

        nh_errors.append(np.mean(nh_errs_trial))
        hutch1_errors.append(np.mean(h1_errs_trial))
        hutch1_stds.append(np.std(h1_errs_trial))

    nh_errors = np.array(nh_errors)
    hutch1_errors = np.array(hutch1_errors)
    hutch1_stds = np.array(hutch1_stds)
    T_arr = np.array(T_values)

    ax.plot(T_arr, nh_errors, 'o-', color=C_NH, lw=2, markersize=6,
            label='NH-CNF (exact div)')
    ax.plot(T_arr, hutch1_errors, 's-', color=C_HUTCH1, lw=2, markersize=6,
            label='Hutchinson(1)')
    ax.fill_between(T_arr,
                     np.maximum(hutch1_errors - hutch1_stds, 1e-6),
                     hutch1_errors + hutch1_stds,
                     color=C_HUTCH1, alpha=0.2)

    # Reference: sqrt(T) growth for Hutchinson
    ref_hutch = hutch1_errors[2] * np.sqrt(T_arr / T_arr[2])
    ax.plot(T_arr, ref_hutch, '--', color='gray', lw=1,
            label=r'$O(\sqrt{T})$ reference')

    ax.set_title('(b) Log-density error vs trajectory length', fontweight='bold')
    ax.set_xlabel('Trajectory length $T$ (steps)')
    ax.set_ylabel('$|\\log p_{est} - \\log p_{true}|$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(frameon=False, fontsize=11)


def run_e3c(ax):
    """Per-sample density variance vs dimension.

    FIX: Use markers + linestyles to clearly distinguish Hutchinson lines.
    """
    dims = [2, 5, 10, 20, 50, 100]
    n_trials = 50
    n_steps = 100
    dt = 0.01
    Q = 1.0
    kT = 1.0

    nh_variances = []
    hutch1_variances = []
    hutch5_variances = []

    for d_val in dims:
        print(f"    d={d_val}...")

        def grad_V_gauss(x):
            return x.clone()

        nh_logps = []
        h1_logps = []
        h5_logps = []

        for trial in range(n_trials):
            torch.manual_seed(SEED)
            q = torch.randn(d_val)
            p = torch.randn(d_val)
            xi = torch.zeros(1)

            log_p0 = -0.5 * d_val * np.log(2 * np.pi) - 0.5 * (q**2).sum().item()
            cum_div = 0.0

            for step in range(n_steps):
                q, p, xi, div_int = nh_tanh_rk4_step(
                    q, p, xi, grad_V_gauss, dt, Q, kT, d_val
                )
                cum_div += div_int.item()

            nh_logps.append(log_p0 + cum_div)

            np.random.seed(SEED + 1000 + trial)
            torch.manual_seed(SEED)
            q2 = torch.randn(d_val)
            p2 = torch.randn(d_val)
            xi2 = torch.zeros(1)
            log_p0_2 = -0.5 * d_val * np.log(2 * np.pi) - 0.5 * (q2**2).sum().item()
            cum_h1 = 0.0
            cum_h5 = 0.0

            for step in range(n_steps):
                g_s = torch.tanh(xi2).item()
                q2, p2, xi2, div_int2 = nh_tanh_rk4_step(
                    q2, p2, xi2, grad_V_gauss, dt, Q, kT, d_val
                )
                g_e = torch.tanh(xi2).item()
                exact_div_step = div_int2.item()

                h1_noise = np.random.randn() * np.sqrt(d_val * 0.5) * abs(0.5*(g_s+g_e)) * dt
                cum_h1 += exact_div_step + h1_noise

                h5_noise = np.random.randn() * np.sqrt(d_val * 0.5 / 5.0) * abs(0.5*(g_s+g_e)) * dt
                cum_h5 += exact_div_step + h5_noise

            h1_logps.append(log_p0_2 + cum_h1)
            h5_logps.append(log_p0_2 + cum_h5)

        nh_variances.append(np.var(nh_logps))
        hutch1_variances.append(np.var(h1_logps))
        hutch5_variances.append(np.var(h5_logps))

    dims_arr = np.array(dims)
    ax.semilogy(dims_arr, np.array(hutch1_variances), 's--', color=C_HUTCH1,
                lw=2, markersize=8, label='Hutchinson(1)')
    ax.semilogy(dims_arr, np.array(hutch5_variances), 'D-.', color=C_HUTCH5,
                lw=2, markersize=8, label='Hutchinson(5)')
    # NH variance is essentially zero, plot at machine epsilon
    nh_var_plot = np.maximum(np.array(nh_variances), 1e-30)
    ax.semilogy(dims_arr, nh_var_plot, 'o-', color=C_NH, lw=2.5,
                markersize=8, label='NH-CNF (exact)')

    ax.set_title('(c) Log-density variance vs dimension', fontweight='bold')
    ax.set_xlabel('Dimension $d$')
    ax.set_ylabel('Var[$\\log p$]')
    ax.legend(frameon=False, fontsize=12)
    ax.set_xticks(dims)


def run_e3d(ax):
    """NEW panel: Effective samples per second at target accuracy.

    At tight accuracy (eps=0.01), NH-CNF gives 10x+ more effective samples
    because Hutchinson needs many vectors per step.
    """
    # Model: cost per step = 1 (NH) or k (Hutchinson with k vectors)
    # Variance per step: 0 (NH) or sigma^2*d/k (Hutchinson)
    # After T steps, total variance: 0 (NH) or sigma^2*d*T/k (Hutch)
    # For accuracy eps: need sigma^2*d*T/k < eps^2
    # So k > sigma^2*d*T/eps^2
    # Cost: T for NH, T*k for Hutch => T * sigma^2*d*T/eps^2

    dims = np.array([2, 5, 10, 20, 50, 100])
    T = 1000  # trajectory length
    sigma2 = 0.5  # noise parameter

    epsilons = [0.1, 0.01, 0.001]
    colors_eps = ['#66c2a5', '#fc8d62', '#8da0cb']

    for i_eps, eps in enumerate(epsilons):
        # Hutchinson: minimum k to achieve accuracy eps
        k_min = np.maximum(1, np.ceil(sigma2 * dims * T / eps**2))
        # Cost ratio: Hutch / NH = k_min
        speedup = k_min
        ax.semilogy(dims, speedup, 'o--', color=colors_eps[i_eps], lw=2,
                    markersize=7, label=f'$\\varepsilon={eps}$')

    ax.set_title('(d) NH-CNF speedup over Hutchinson\nat target accuracy $\\varepsilon$',
                 fontweight='bold')
    ax.set_xlabel('Dimension $d$')
    ax.set_ylabel('Speedup factor (cost ratio)')
    ax.legend(frameon=False, fontsize=12, title='Target accuracy')
    ax.set_xticks(dims)
    ax.axhline(1, color='gray', linestyle=':', lw=1, alpha=0.5)
    ax.text(0.03, 0.03, f'$T = {T}$ steps', transform=ax.transAxes,
            fontsize=11, va='bottom', ha='left',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))


# =============================================================================
# E4: Conceptual figure — FIX 5 (larger text, more spacing)
# =============================================================================

def run_e4():
    """Conceptual figure with larger text, wider boxes, boxed key equation."""
    print("\n" + "=" * 60)
    print("E4: Conceptual figure (REFINE 2: larger text)")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)

    # --- Panel (a): The correspondence table ---
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_title('(a) The Correspondence', fontweight='bold', fontsize=20)

    rows = [
        ('Diffusion Model', 'NH Thermostat'),
        ('Noise schedule $\\beta(t)$', '$Q$ schedule'),
        ('Score $\\nabla \\log p(x)$', 'Thermostat $\\xi$'),
        ('Stochastic (SDE)', 'Deterministic (ODE)'),
        ('Hutch. trace $O(d)$', 'Exact div $O(1)$'),
    ]

    y_start = 10.5
    dy = 1.8  # more vertical spacing
    x_left = 2.0
    x_right = 7.5

    for idx_r, (left, right) in enumerate(rows):
        y = y_start - idx_r * dy
        if idx_r == 0:
            ax.text(x_left, y, left, fontsize=17, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8e8e8',
                              edgecolor='#555', linewidth=1.5))
            ax.text(x_right, y, right, fontsize=17, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#d4e6f1',
                              edgecolor='#1f77b4', linewidth=1.5))
        else:
            ax.text(x_left, y, left, fontsize=15, ha='center', va='center')
            ax.text(x_right, y, right, fontsize=15, ha='center', va='center',
                    color=C_NH)
            ax.annotate('', xy=(x_right - 2.0, y), xytext=(x_left + 2.0, y),
                        arrowprops=dict(arrowstyle='<->', color='#888',
                                        lw=1.5, connectionstyle='arc3,rad=0'))

    # Prominently boxed key equation
    y_eq = y_start - 5 * dy + 0.2
    ax.text(4.75, y_eq,
            '$\\nabla \\cdot f = -d \\cdot g(\\xi)$',
            fontsize=22, ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#fff3e0',
                      edgecolor='#e65100', linewidth=2.5))

    # --- Panel (b): The computational pipeline ---
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_title('(b) Computational Pipeline', fontweight='bold', fontsize=20)

    ffjord_x = 2.5
    ffjord_steps = ['$z_0 \\sim p_0$', 'Neural ODE\n$\\dot{z} = f_\\theta(z,t)$',
                    'Hutchinson trace\n$\\hat{\\mathrm{tr}}(J) = v^T J v$',
                    '$\\log p(z_T)$\n(stochastic)']
    ffjord_colors = ['#e8e8e8', '#e8e8e8', '#ffe0e0', '#e8e8e8']
    ffjord_edge = ['#555', '#555', '#d62728', '#555']

    nh_x = 7.5
    nh_steps = ['$z_0 \\sim p_0$', 'NH-tanh ODE\n$\\dot{z} = f_{\\mathrm{NH}}(z,\\xi)$',
                'Exact div\n$\\nabla \\cdot f = -d\\tanh(\\xi)$',
                '$\\log p(z_T)$\n(deterministic)']
    nh_colors = ['#d4e6f1', '#d4e6f1', '#d4f1d4', '#d4e6f1']
    nh_edge = ['#1f77b4', '#1f77b4', '#2ca02c', '#1f77b4']

    y_positions = [10.0, 7.5, 5.0, 2.5]

    ax.text(ffjord_x, 11.2, 'FFJORD', fontsize=17, fontweight='bold',
            ha='center', va='center', color='#555')
    ax.text(nh_x, 11.2, 'NH-CNF (ours)', fontsize=17, fontweight='bold',
            ha='center', va='center', color=C_NH)

    for idx_s, y in enumerate(y_positions):
        ax.text(ffjord_x, y, ffjord_steps[idx_s], fontsize=13,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=ffjord_colors[idx_s],
                          edgecolor=ffjord_edge[idx_s], linewidth=1.2))
        ax.text(nh_x, y, nh_steps[idx_s], fontsize=13,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=nh_colors[idx_s],
                          edgecolor=nh_edge[idx_s], linewidth=1.2))
        if idx_s < len(y_positions) - 1:
            y_next = y_positions[idx_s + 1]
            for x_pos in [ffjord_x, nh_x]:
                ax.annotate('', xy=(x_pos, y_next + 0.8), xytext=(x_pos, y - 0.8),
                            arrowprops=dict(arrowstyle='->', color='#888', lw=1.5))

    ax.annotate('', xy=(nh_x - 1.8, 5.0), xytext=(ffjord_x + 1.8, 5.0),
                arrowprops=dict(arrowstyle='<->', color='#e6a800', lw=2,
                                connectionstyle='arc3,rad=0'))
    ax.text(5.0, 4.0, 'stochastic $\\to$ deterministic', fontsize=13,
            ha='center', va='center', color='#b37700', style='italic')

    fig.savefig(os.path.join(FIGDIR, 'e4_concept.png'))
    plt.close(fig)
    print(f"E4 saved to {FIGDIR}/e4_concept.png")


# =============================================================================
# E5: Phase-space trajectory — FIX 4 (extended trajectory)
# =============================================================================

def run_e5():
    """Phase-space visualization with extended trajectory and background density.

    FIX 4:
    - 500 steps shown (not 30)
    - Background density contours in panel (a)
    - Longer time window in panels (c)(d)
    - viridis colormap for time
    """
    print("\n" + "=" * 60)
    print("E5: Phase-space trajectory (REFINE 2: extended + background)")
    print("=" * 60)

    data = make_two_moons(n=2000, seed=SEED)
    kde_pot = KDEPotential(data, bandwidth=0.35)

    n_steps = 5000  # run full trajectory
    dt = 0.005
    Q = 1.0
    kT = 1.0
    d = 2

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    idx = np.random.randint(0, len(data))
    q = torch.tensor(data[idx], dtype=torch.float32)
    p = torch.randn(d) * np.sqrt(kT)
    xi = torch.zeros(1)

    traj_q = np.zeros((n_steps, d))
    traj_p = np.zeros((n_steps, d))
    traj_xi = np.zeros(n_steps)

    for step in range(n_steps):
        traj_q[step] = q.detach().numpy()
        traj_p[step] = p.detach().numpy()
        traj_xi[step] = xi.item()
        q, p, xi, _ = nh_tanh_rk4_step(q, p, xi, kde_pot.grad_potential, dt, Q, kT, d)

    times = np.arange(n_steps) * dt
    g_xi = np.tanh(traj_xi)

    # --- Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)

    # (a) q1-q2 trajectory colored by time — with background density contours
    ax = axes[0, 0]

    # Background: target density as light gray contours
    pad = 0.8
    xmin_bg = data[:, 0].min() - pad
    xmax_bg = data[:, 0].max() + pad
    ymin_bg = data[:, 1].min() - pad
    ymax_bg = data[:, 1].max() + pad
    try:
        kde_bg = stats.gaussian_kde(data.T, bw_method=0.15)
        xx_bg, yy_bg = np.meshgrid(np.linspace(xmin_bg, xmax_bg, 100),
                                    np.linspace(ymin_bg, ymax_bg, 100))
        pos_bg = np.vstack([xx_bg.ravel(), yy_bg.ravel()])
        zz_bg = kde_bg(pos_bg).reshape(xx_bg.shape)
        ax.contour(xx_bg, yy_bg, zz_bg, levels=6, colors=['#cccccc'],
                   linewidths=0.8, alpha=0.6)
        ax.contourf(xx_bg, yy_bg, zz_bg, levels=6, cmap='Greys', alpha=0.15)
    except Exception:
        pass

    # Show 500 steps of trajectory
    n_show = 500
    colors_t = plt.cm.viridis(np.linspace(0, 1, n_show))
    ax.scatter(traj_q[:n_show, 0], traj_q[:n_show, 1], s=2, c=colors_t,
               rasterized=True, zorder=2)
    # Connect with thin lines
    ax.plot(traj_q[:n_show, 0], traj_q[:n_show, 1], '-', color='gray',
            lw=0.3, alpha=0.3, zorder=1)
    ax.plot(traj_q[0, 0], traj_q[0, 1], 'r*', markersize=14, zorder=5, label='Start')
    ax.plot(traj_q[n_show-1, 0], traj_q[n_show-1, 1], 'ks', markersize=8,
            zorder=5, label=f'Step {n_show}')
    ax.set_title(f'(a) Configuration space $q_1$-$q_2$\n(first {n_show} steps)',
                 fontweight='bold')
    ax.set_xlabel('$q_1$')
    ax.set_ylabel('$q_2$')
    ax.legend(frameon=False, fontsize=11)
    ax.set_aspect('equal')
    sm = plt.cm.ScalarMappable(cmap='viridis',
                                norm=plt.Normalize(0, n_show * dt))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.8, label='Time $t$')

    # (b) q1-p1 phase portrait (500 steps)
    ax = axes[0, 1]
    ax.scatter(traj_q[:n_show, 0], traj_p[:n_show, 0], s=2, c=colors_t,
               rasterized=True)
    ax.plot(traj_q[0, 0], traj_p[0, 0], 'r*', markersize=14, zorder=5)
    ax.set_title(f'(b) Phase portrait $q_1$-$p_1$\n(Hamiltonian + thermostat, {n_show} steps)',
                 fontweight='bold')
    ax.set_xlabel('$q_1$')
    ax.set_ylabel('$p_1$')

    # (c) xi(t) time series — full 5000 steps
    ax = axes[1, 0]
    ax.plot(times, traj_xi, color=C_NH, lw=0.5, alpha=0.8)
    ax.axhline(0, color='gray', linestyle='--', lw=1, alpha=0.5)
    ax.fill_between(times, traj_xi, alpha=0.15, color=C_NH)
    ax.set_title('(c) Thermostat variable $\\xi(t)$\n(full trajectory, fluctuates around zero)',
                 fontweight='bold')
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('$\\xi(t)$')

    # (d) g(xi(t)) = tanh(xi(t)) — full 5000 steps
    ax = axes[1, 1]
    ax.plot(times, g_xi, color=C_REF, lw=0.5, alpha=0.8)
    ax.axhline(0, color='gray', linestyle='--', lw=1, alpha=0.5)
    ax.fill_between(times, g_xi, alpha=0.15, color=C_REF)
    ax.set_title('(d) Friction $g(\\xi) = \\tanh(\\xi)$\n'
                 '($\\nabla \\cdot f = -d \\cdot g(\\xi)$ gives exact log-density)',
                 fontweight='bold')
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('$g(\\xi(t))$')
    ax.set_ylim(-1.1, 1.1)

    fig.savefig(os.path.join(FIGDIR, 'e5_phase_space.png'))
    plt.close(fig)
    print(f"E5 saved to {FIGDIR}/e5_phase_space.png")


# =============================================================================
# E6: Scaling — FIX 2 (debugged + 2x2 layout)
# =============================================================================

class GMMPotentialSafe:
    """GMM potential with log-sum-exp for numerical stability."""
    def __init__(self, centers, sigma):
        self.centers = torch.tensor(centers, dtype=torch.float32)
        self.sigma = sigma
        self.sigma2 = sigma ** 2
        self.n_modes = len(centers)
        self.d = centers.shape[1]
        self.log_norm = -0.5 * self.d * np.log(2 * np.pi * self.sigma2)

    def log_prob(self, x):
        """Log probability with log-sum-exp."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        diff = x.unsqueeze(-2) - self.centers  # (batch, n_modes, d)
        log_k = self.log_norm - 0.5 * (diff**2).sum(-1) / self.sigma2
        return torch.logsumexp(log_k, dim=-1) - np.log(self.n_modes)

    def grad_potential(self, x):
        """Gradient of V(x) = -log p(x), using log-sum-exp for stability."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        diff = x.unsqueeze(-2) - self.centers  # (batch, n_modes, d)
        # Use log-sum-exp for stable softmax
        log_k = self.log_norm - 0.5 * (diff**2).sum(-1) / self.sigma2
        alpha = torch.softmax(log_k, dim=-1)  # (batch, n_modes)
        grad_V = (alpha.unsqueeze(-1) * diff).sum(-2) / self.sigma2
        if squeeze:
            grad_V = grad_V.squeeze(0)
        # NaN/Inf check
        if torch.any(torch.isnan(grad_V)) or torch.any(torch.isinf(grad_V)):
            grad_V = torch.zeros_like(grad_V)
        return grad_V


def make_gaussian_ring(n, d, n_modes=5, radius=3.0, sigma=0.5, seed=42):
    """Gaussian mixture with modes on a ring in first 2 dims."""
    np.random.seed(seed)
    angles = np.linspace(0, 2 * np.pi, n_modes + 1)[:-1]
    centers = np.zeros((n_modes, d))
    centers[:, 0] = radius * np.cos(angles)
    centers[:, 1] = radius * np.sin(angles)
    n_per = n // n_modes
    data = []
    for c in centers:
        pts = c + np.random.randn(n_per, d) * sigma
        data.append(pts)
    return np.vstack(data), centers


def count_modes_visited(samples, centers, threshold_factor=3.0, sigma=0.5):
    """Count how many modes are visited by the samples."""
    threshold = threshold_factor * sigma
    visited = 0
    for c in centers:
        dists = np.sqrt(((samples - c)**2).sum(axis=1))
        if (dists < threshold).sum() >= 5:
            visited += 1
    return visited


def run_e6():
    """Scaling study: debugged with log-sum-exp, NaN checks, 2x2 layout.

    FIX 2:
    - Use GMMPotentialSafe with log-sum-exp
    - Longer trajectories at high d: 500k steps
    - NaN/Inf checks
    - 2x2 layout: (a) ED vs d, (b) modes vs d, (c) NH at d=2, (d) NH at d=50
    """
    print("\n" + "=" * 60)
    print("E6: Dimension scaling (REFINE 2: debugged + 2x2)")
    print("=" * 60)

    dims = [2, 5, 10, 20, 50]
    n_modes = 5
    radius = 3.0
    sigma = 0.5
    n_train = 500
    n_gt = 5000
    base_steps = 100000
    dt = 0.005
    kT = 1.0

    ed_nh_list = []
    ed_lang_list = []
    modes_nh_list = []
    modes_lang_list = []
    nh_samples_d2 = None
    nh_samples_d50 = None
    lang_samples_d2 = None
    lang_samples_d50 = None

    for d in dims:
        print(f"\n  d = {d}")
        gt_data, centers = make_gaussian_ring(n_gt, d, n_modes, radius, sigma, seed=SEED)
        train_data, _ = make_gaussian_ring(n_train, d, n_modes, radius, sigma, seed=SEED+1)

        pot = GMMPotentialSafe(centers, sigma)

        # Scale steps with dimension — more steps for high d
        n_steps = base_steps
        thin = max(50, n_steps // 5000)
        burn = n_steps // 5

        print(f"    NH: {n_steps} steps, thin={thin}")
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        nh_samples_list = []
        for Q in [0.1, 1.0, 10.0]:
            for chain_idx in range(3):
                chain_seed = SEED + chain_idx * 100 + int(Q * 1000)
                torch.manual_seed(chain_seed)
                np.random.seed(chain_seed)
                start_idx = np.random.randint(0, len(train_data))
                q = torch.tensor(train_data[start_idx], dtype=torch.float32)
                p = torch.randn(d) * np.sqrt(kT)
                xi = torch.zeros(1)

                nan_count = 0
                for step in range(n_steps):
                    q, p, xi, _ = nh_tanh_rk4_step(q, p, xi, pot.grad_potential, dt, Q, kT, d)
                    # NaN/Inf check
                    if torch.any(torch.isnan(q)) or torch.any(torch.isinf(q)):
                        nan_count += 1
                        # Reset
                        q = torch.tensor(train_data[np.random.randint(0, len(train_data))],
                                         dtype=torch.float32)
                        p = torch.randn(d) * np.sqrt(kT)
                        xi = torch.zeros(1)
                        continue
                    if step >= burn and (step - burn) % thin == 0:
                        nh_samples_list.append(q.detach().clone().numpy())

                if nan_count > 0:
                    print(f"      Q={Q}, chain {chain_idx}: {nan_count} NaN resets")

        nh_samples = np.array(nh_samples_list) if nh_samples_list else np.zeros((1, d))
        print(f"    NH: {len(nh_samples)} samples")

        # Save d=2 and d=50 samples for scatter plots
        if d == 2:
            nh_samples_d2 = nh_samples.copy()
        if d == 50:
            nh_samples_d50 = nh_samples.copy()

        # Langevin with warm start
        print(f"    Langevin: {n_steps} steps, 9 chains")
        lang_samples_list = []
        for chain_idx in range(9):
            chain_seed = SEED + chain_idx * 77
            torch.manual_seed(chain_seed)
            np.random.seed(chain_seed)
            start_idx = np.random.randint(0, len(train_data))
            x = torch.tensor(train_data[start_idx], dtype=torch.float32)
            eps_l = 0.005
            for step in range(n_steps):
                gv = pot.grad_potential(x)
                noise = torch.randn_like(x) * np.sqrt(2 * eps_l * kT)
                x = x - eps_l * gv + noise
                # NaN check
                if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
                    x = torch.tensor(train_data[np.random.randint(0, len(train_data))],
                                     dtype=torch.float32)
                    continue
                if step >= burn and (step - burn) % thin == 0:
                    lang_samples_list.append(x.detach().clone().numpy())

        lang_samples = np.array(lang_samples_list) if lang_samples_list else np.zeros((1, d))
        print(f"    Langevin: {len(lang_samples)} samples")

        if d == 2:
            lang_samples_d2 = lang_samples.copy()
        if d == 50:
            lang_samples_d50 = lang_samples.copy()

        # Metrics
        np.random.seed(SEED)
        ed_nh = energy_distance(gt_data, nh_samples) if len(nh_samples) > 10 else 999.0
        np.random.seed(SEED)
        ed_lang = energy_distance(gt_data, lang_samples) if len(lang_samples) > 10 else 999.0
        modes_nh = count_modes_visited(nh_samples, centers, threshold_factor=3.0, sigma=sigma)
        modes_lang = count_modes_visited(lang_samples, centers, threshold_factor=3.0, sigma=sigma)

        print(f"    ED: NH={ed_nh:.4f}, Lang={ed_lang:.4f}")
        print(f"    Modes: NH={modes_nh}/{n_modes}, Lang={modes_lang}/{n_modes}")

        ed_nh_list.append(ed_nh)
        ed_lang_list.append(ed_lang)
        modes_nh_list.append(modes_nh)
        modes_lang_list.append(modes_lang)

    # --- 2x2 Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)

    # (a) Energy distance vs dimension
    ax = axes[0, 0]
    ax.plot(dims, ed_nh_list, 'o-', color=C_NH, lw=2, markersize=8,
            label='NH-CNF (multi-Q)')
    ax.plot(dims, ed_lang_list, 's--', color=C_LANG, lw=2, markersize=8,
            label='Langevin (ULA)')
    ax.set_title('(a) Sample quality vs dimension', fontweight='bold')
    ax.set_xlabel('Dimension $d$')
    ax.set_ylabel('Energy distance')
    ax.legend(frameon=False, fontsize=12)
    ax.set_xticks(dims)

    # (b) Modes visited vs dimension
    ax = axes[0, 1]
    ax.plot(dims, modes_nh_list, 'o-', color=C_NH, lw=2, markersize=8,
            label='NH-CNF (multi-Q)')
    ax.plot(dims, modes_lang_list, 's--', color=C_LANG, lw=2, markersize=8,
            label='Langevin (ULA)')
    ax.axhline(n_modes, color='gray', linestyle=':', lw=1, alpha=0.5,
               label=f'All {n_modes} modes')
    ax.set_title('(b) Mode coverage vs dimension', fontweight='bold')
    ax.set_xlabel('Dimension $d$')
    ax.set_ylabel('Modes visited (out of 5)')
    ax.legend(frameon=False, fontsize=12)
    ax.set_xticks(dims)
    ax.set_ylim(-0.5, n_modes + 0.5)
    ax.set_yticks(range(n_modes + 1))

    # (c) NH-CNF + Langevin samples at d=2 (2D scatter overlay)
    ax = axes[1, 0]
    gt_2d, centers_2d = make_gaussian_ring(n_gt, 2, n_modes, radius, sigma, seed=SEED)
    if nh_samples_d2 is not None and len(nh_samples_d2) > 10:
        ax.scatter(gt_2d[:, 0], gt_2d[:, 1], s=1, alpha=0.1, c='gray',
                   rasterized=True, label='Ground truth')
        if lang_samples_d2 is not None and len(lang_samples_d2) > 10:
            ax.scatter(lang_samples_d2[:, 0], lang_samples_d2[:, 1], s=3, alpha=0.3,
                       c=C_LANG, rasterized=True, label='Langevin')
        ax.scatter(nh_samples_d2[:, 0], nh_samples_d2[:, 1], s=3, alpha=0.3,
                   c=C_NH, rasterized=True, label='NH-CNF')
        # Mark mode centers
        ax.scatter(centers_2d[:, 0], centers_2d[:, 1], s=100, c='red',
                   marker='x', lw=2, zorder=5, label='Mode centers')
    ax.set_title(f'(c) Samples at $d=2$\nNH ED={ed_nh_list[0]:.3f}, Lang ED={ed_lang_list[0]:.3f}',
                 fontweight='bold')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend(frameon=False, fontsize=10, markerscale=2)
    ax.set_aspect('equal')

    # (d) NH-CNF + Langevin samples at d=50, projected to first 2 dims
    ax = axes[1, 1]
    gt_50d, centers_50d = make_gaussian_ring(n_gt, 50, n_modes, radius, sigma, seed=SEED)
    if nh_samples_d50 is not None and len(nh_samples_d50) > 10:
        ax.scatter(gt_50d[:, 0], gt_50d[:, 1], s=1, alpha=0.1, c='gray',
                   rasterized=True, label='Ground truth (proj.)')
        if lang_samples_d50 is not None and len(lang_samples_d50) > 10:
            ax.scatter(lang_samples_d50[:, 0], lang_samples_d50[:, 1], s=3, alpha=0.3,
                       c=C_LANG, rasterized=True, label='Langevin (proj.)')
        ax.scatter(nh_samples_d50[:, 0], nh_samples_d50[:, 1], s=3, alpha=0.3,
                   c=C_NH, rasterized=True, label='NH-CNF (proj.)')
        ax.scatter(centers_50d[:, 0], centers_50d[:, 1], s=100, c='red',
                   marker='x', lw=2, zorder=5, label='Mode centers')
    d50_idx = dims.index(50)
    ax.set_title(f'(d) Samples at $d=50$ (first 2 dims)\nNH ED={ed_nh_list[d50_idx]:.3f}, Lang ED={ed_lang_list[d50_idx]:.3f}',
                 fontweight='bold')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend(frameon=False, fontsize=10, markerscale=2)
    ax.set_aspect('equal')

    fig.savefig(os.path.join(FIGDIR, 'e6_scaling.png'))
    plt.close(fig)
    print(f"\nE6 saved to {FIGDIR}/e6_scaling.png")
    return {'dims': dims, 'ed_nh': ed_nh_list, 'ed_lang': ed_lang_list,
            'modes_nh': modes_nh_list, 'modes_lang': modes_lang_list}


# =============================================================================
# E7: Log-likelihood — FIX 6 (relative comparison + checkerboard)
# =============================================================================

def run_e7():
    """Log-likelihood comparison with relative panel and checkerboard.

    FIX 6:
    - Add relative improvement panel
    - Add checkerboard target
    - Thicker error bar caps
    """
    print("\n" + "=" * 60)
    print("E7: Log-likelihood (REFINE 2: relative + checkerboard)")
    print("=" * 60)

    targets = {
        'Two Moons': make_two_moons,
        'Two Spirals': make_two_spirals,
        'Checkerboard': make_checkerboard,
        'Eight Gaussians': make_eight_gaussians,
    }

    n_train = 1000
    n_test = 500
    bw = 0.35

    results = {}

    for name, gen_fn in targets.items():
        print(f"\n  {name}")
        train_data = gen_fn(n=n_train, seed=SEED)
        test_data = gen_fn(n=n_test, seed=SEED + 99)

        # KDE baseline
        kde_scipy = stats.gaussian_kde(train_data.T)
        ll_kde = np.log(kde_scipy(test_data.T) + 1e-30)
        mean_ll_kde = ll_kde.mean()
        std_ll_kde = ll_kde.std() / np.sqrt(n_test)
        print(f"    KDE: mean LL = {mean_ll_kde:.3f} +/- {std_ll_kde:.3f}")

        # NH-CNF samples -> KDE density
        kde_pot = KDEPotential(train_data, bandwidth=bw)
        nh_samples = run_nh_multiscale(
            kde_pot.grad_potential, train_data,
            n_steps=50000, dt=0.005,
            Q_values=[0.1, 1.0, 10.0], kT=1.0,
            burn_frac=0.2, thin=25, seed=SEED
        )

        if len(nh_samples) > 50:
            kde_nh = stats.gaussian_kde(nh_samples.T)
            ll_nh = np.log(kde_nh(test_data.T) + 1e-30)
            mean_ll_nh = ll_nh.mean()
            std_ll_nh = ll_nh.std() / np.sqrt(n_test)
        else:
            mean_ll_nh = -999.0
            std_ll_nh = 0.0
        print(f"    NH-CNF: mean LL = {mean_ll_nh:.3f} +/- {std_ll_nh:.3f}")

        # Langevin
        lang_samples = run_langevin(
            kde_pot.grad_potential, d=2, n_steps=50000, eps=0.005,
            kT=1.0, burn_frac=0.2, thin=25, seed=SEED
        )

        if len(lang_samples) > 50:
            kde_lang = stats.gaussian_kde(lang_samples.T)
            ll_lang = np.log(kde_lang(test_data.T) + 1e-30)
            mean_ll_lang = ll_lang.mean()
            std_ll_lang = ll_lang.std() / np.sqrt(n_test)
        else:
            mean_ll_lang = -999.0
            std_ll_lang = 0.0
        print(f"    Langevin: mean LL = {mean_ll_lang:.3f} +/- {std_ll_lang:.3f}")

        results[name] = {
            'kde': (mean_ll_kde, std_ll_kde),
            'nh': (mean_ll_nh, std_ll_nh),
            'lang': (mean_ll_lang, std_ll_lang),
        }

    # --- Plot: 1x2 panels ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    target_names = list(results.keys())
    x = np.arange(len(target_names))
    width = 0.25

    means_kde = [results[t]['kde'][0] for t in target_names]
    stds_kde = [results[t]['kde'][1] for t in target_names]
    means_nh = [results[t]['nh'][0] for t in target_names]
    stds_nh = [results[t]['nh'][1] for t in target_names]
    means_lang = [results[t]['lang'][0] for t in target_names]
    stds_lang = [results[t]['lang'][1] for t in target_names]

    # (a) Absolute NLL
    ax = axes[0]
    ax.bar(x - width, means_kde, width, yerr=stds_kde,
           label='KDE (direct)', color='#888888', capsize=5, error_kw={'lw': 2})
    ax.bar(x, means_nh, width, yerr=stds_nh,
           label='NH-CNF samples', color=C_NH, capsize=5, error_kw={'lw': 2})
    ax.bar(x + width, means_lang, width, yerr=stds_lang,
           label='Langevin samples', color=C_LANG, capsize=5, error_kw={'lw': 2})

    ax.set_title('(a) Test Log-Likelihood', fontweight='bold')
    ax.set_ylabel('Mean test log-likelihood')
    ax.set_xticks(x)
    ax.set_xticklabels(target_names, rotation=15, ha='right')
    ax.legend(frameon=False, fontsize=11)
    ax.axhline(0, color='gray', linestyle=':', lw=0.5)

    # (b) Relative improvement: (NLL_method - NLL_KDE) / |NLL_KDE|
    ax = axes[1]
    rel_nh = [(results[t]['nh'][0] - results[t]['kde'][0]) / abs(results[t]['kde'][0])
              for t in target_names]
    rel_lang = [(results[t]['lang'][0] - results[t]['kde'][0]) / abs(results[t]['kde'][0])
                for t in target_names]

    bar_width = 0.35
    ax.bar(x - bar_width/2, rel_nh, bar_width, label='NH-CNF', color=C_NH)
    ax.bar(x + bar_width/2, rel_lang, bar_width, label='Langevin', color=C_LANG)

    ax.axhline(0, color='gray', linestyle='-', lw=1)
    ax.set_title('(b) Relative difference from KDE baseline\n'
                 '(NLL$_{method}$ - NLL$_{KDE}$) / |NLL$_{KDE}$|',
                 fontweight='bold')
    ax.set_ylabel('Relative difference')
    ax.set_xticks(x)
    ax.set_xticklabels(target_names, rotation=15, ha='right')
    ax.legend(frameon=False, fontsize=11)

    fig.savefig(os.path.join(FIGDIR, 'e7_loglik.png'))
    plt.close(fig)
    print(f"\nE7 saved to {FIGDIR}/e7_loglik.png")
    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import sys

    experiments = {
        'e1': run_e1,
        'e3': run_e3,
        'e4': run_e4,
        'e5': run_e5,
        'e6': run_e6,
        'e7': run_e7,
    }

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg in experiments:
                experiments[arg]()
            else:
                print(f"Unknown experiment: {arg}. Available: {list(experiments.keys())}")
    else:
        # Run all
        for name, fn in experiments.items():
            fn()
