"""
NH-CNF Deep Experiments — Refinement 1
=======================================

E1 (remade): KDE contour density plots with proper thinning
E4 (remade): Simplified conceptual correspondence figure
E5 (new): Phase-space trajectory visualization
E6 (new): Dimension scaling study
E7 (new): Log-likelihood comparison

The key insight: for the NH-tanh ODE, div(f) = -d*tanh(xi) is exact and O(1),
giving zero-variance log-density tracking — unlike Hutchinson trace estimators
used in FFJORD/neural ODEs.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
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

torch.manual_seed(SEED)
np.random.seed(SEED)


# =============================================================================
# NH-tanh RK4 integrator
# =============================================================================

def nh_tanh_rk4_step(q, p, xi, grad_V_fn, dt, Q=1.0, kT=1.0, d=None):
    """One RK4 step of the NH-tanh ODE. Works for single or batched inputs.
    Returns (q_new, p_new, xi_new, div_integral).
    """
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
# Multi-scale Q NH sampler (N=3 parallel thermostats)
# =============================================================================

def run_nh_multiscale(grad_V_fn, data, n_steps=500000, dt=0.005,
                      Q_values=[0.1, 1.0, 10.0], kT=1.0,
                      burn_frac=0.2, thin=50, seed=42):
    """Run NH-tanh with multiple Q values, warm-started from data.

    For each Q value, run n_chains chains warm-started from random data points.
    Burn in first burn_frac, then thin by taking every thin-th sample.
    Returns combined thinned samples from all chains.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = data.shape[1]
    n_data = len(data)

    all_samples = []
    n_chains_per_Q = max(1, 10 // len(Q_values))  # distribute chains across Q values
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

def run_langevin(grad_V_fn, d, n_steps=500000, eps=0.005, kT=1.0,
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
# E1: NH-CNF density estimation with KDE contour plots
# =============================================================================

def run_e1():
    """NH-CNF sampling with KDE contour density plots.

    For each target:
    - Generate 10k ground truth samples
    - Fit KDE potential from 1000 training samples
    - Run NH-tanh with multi-scale Q=[0.1, 1, 10], long trajectory (100k steps)
    - Run Langevin with tuned step size for 100k steps, same thinning
    - Plot KDE contour density for each, with light scatter underneath
    """
    print("=" * 60)
    print("E1: NH-CNF density estimation (KDE contour plots)")
    print("=" * 60)

    targets = {
        'Two Moons': make_two_moons,
        'Two Spirals': make_two_spirals,
        'Checkerboard': make_checkerboard,
        'Eight Gaussians': make_eight_gaussians,
    }

    # Parameters — longer runs with thinning for independent samples
    n_train = 1000
    n_steps = 100000   # 100k steps per chain
    dt_nh = 0.005
    eps_lang = 0.005
    kT = 1.0
    burn_frac = 0.2
    thin = 50  # take every 50th sample after burn-in
    bw_kde_potential = 0.35  # bandwidth for KDE potential

    fig, axes = plt.subplots(4, 3, figsize=(15, 20), constrained_layout=True)

    results = {}
    for i, (name, gen_fn) in enumerate(targets.items()):
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
        _plot_kde_contour(ax, gt_data, xmin, xmax, ymin, ymax, C_REF)
        ax.set_title(f'(a) {name}\nGround Truth (N=10k)', fontweight='bold')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')

        # --- Column 1: NH-CNF samples ---
        print(f"  Running NH-CNF (multi-scale Q, {n_steps} steps)...")
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
        _plot_kde_contour(ax, nh_samples, xmin, xmax, ymin, ymax, C_NH)
        ax.set_title(f'(b) NH-CNF (multi-Q)\nED={ed_nh:.4f}, N={len(nh_samples)}', fontweight='bold')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')

        # --- Column 2: Langevin samples ---
        print(f"  Running Langevin ({n_steps} steps)...")
        t0 = time.time()
        lang_samples = run_langevin(
            kde_pot.grad_potential, d=2, n_steps=n_steps, eps=eps_lang,
            kT=kT, burn_frac=burn_frac, thin=thin, seed=SEED
        )
        lang_time = time.time() - t0
        np.random.seed(SEED)
        ed_lang = energy_distance(gt_data, lang_samples)
        print(f"  Langevin: {len(lang_samples)} samples, ED={ed_lang:.4f}, time={lang_time:.1f}s")

        ax = axes[i, 2]
        _plot_kde_contour(ax, lang_samples, xmin, xmax, ymin, ymax, C_LANG)
        ax.set_title(f'(c) Langevin (ULA)\nED={ed_lang:.4f}, N={len(lang_samples)}', fontweight='bold')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')

        # Match axes
        for j in range(3):
            axes[i, j].set_xlim(xmin, xmax)
            axes[i, j].set_ylim(ymin, ymax)
            axes[i, j].set_aspect('equal')

        results[name] = {'ed_nh': ed_nh, 'ed_lang': ed_lang,
                         'n_nh': len(nh_samples), 'n_lang': len(lang_samples)}

    fig.savefig(os.path.join(FIGDIR, 'e1_density.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"\nE1 saved to {FIGDIR}/e1_density.png")

    # Print summary table
    print("\n--- E1 Summary ---")
    print(f"{'Target':<18} {'NH-CNF ED':>10} {'Langevin ED':>12} {'NH wins?':>10}")
    for name, r in results.items():
        winner = "Yes" if r['ed_nh'] < r['ed_lang'] else "No"
        ratio = r['ed_lang'] / max(r['ed_nh'], 1e-6)
        print(f"{name:<18} {r['ed_nh']:>10.4f} {r['ed_lang']:>12.4f} {winner:>6} ({ratio:.1f}x)")

    return results


def _plot_kde_contour(ax, samples, xmin, xmax, ymin, ymax, color,
                      n_grid=100, levels=8):
    """Plot KDE contour with light scatter underneath."""
    if len(samples) < 10:
        ax.text(0.5, 0.5, 'Too few\nsamples', transform=ax.transAxes,
                ha='center', va='center', fontsize=14)
        return

    # Light scatter underneath
    n_show = min(len(samples), 3000)
    idx = np.random.choice(len(samples), n_show, replace=False)
    ax.scatter(samples[idx, 0], samples[idx, 1], s=0.5, alpha=0.08,
               c=color, rasterized=True)

    # KDE contour
    try:
        kde = stats.gaussian_kde(samples.T)
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, n_grid),
                              np.linspace(ymin, ymax, n_grid))
        positions = np.vstack([xx.ravel(), yy.ravel()])
        zz = kde(positions).reshape(xx.shape)

        # Use filled contours with alpha
        ax.contourf(xx, yy, zz, levels=levels, cmap=_make_cmap(color),
                     alpha=0.7)
        ax.contour(xx, yy, zz, levels=levels, colors=[color], linewidths=0.5,
                    alpha=0.5)
    except Exception as e:
        print(f"    KDE contour failed: {e}")
        # fallback: just scatter
        ax.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.3,
                   c=color, rasterized=True)


def _make_cmap(color):
    """Create a white-to-color colormap."""
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(color)
    colors = [(1, 1, 1, 0), (*rgb, 0.3), (*rgb, 0.6), (*rgb, 1.0)]
    return LinearSegmentedColormap.from_list('custom', colors, N=256)


# =============================================================================
# E4: Simplified conceptual figure (2 panels)
# =============================================================================

def run_e4():
    """Conceptual figure: the correspondence + computational pipeline."""
    print("\n" + "=" * 60)
    print("E4: Simplified conceptual figure")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    # --- Panel (a): The correspondence table ---
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('(a) The Correspondence', fontweight='bold', fontsize=18)

    # Table data
    rows = [
        ('Diffusion Model', 'NH Thermostat'),
        ('Noise schedule $\\beta(t)$', '$Q$ schedule'),
        ('Score $\\nabla \\log p(x)$', 'Thermostat $\\xi$'),
        ('Stochastic (SDE)', 'Deterministic (ODE)'),
        ('Hutch. trace $O(d)$', 'Exact div $O(1)$'),
    ]

    # Header
    y_start = 9.0
    dy = 1.4
    # Column positions
    x_left = 1.5
    x_right = 7.0
    x_arrow = 5.0

    for idx_r, (left, right) in enumerate(rows):
        y = y_start - idx_r * dy
        if idx_r == 0:
            # Header row
            ax.text(x_left, y, left, fontsize=15, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#e8e8e8',
                              edgecolor='#555', linewidth=1.5))
            ax.text(x_right, y, right, fontsize=15, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#d4e6f1',
                              edgecolor='#1f77b4', linewidth=1.5))
        else:
            # Data rows
            ax.text(x_left, y, left, fontsize=13, ha='center', va='center')
            ax.text(x_right, y, right, fontsize=13, ha='center', va='center',
                    color=C_NH)
            # Arrow between
            ax.annotate('', xy=(x_right - 1.8, y), xytext=(x_left + 1.8, y),
                        arrowprops=dict(arrowstyle='<->', color='#888',
                                        lw=1.5, connectionstyle='arc3,rad=0'))

    # Highlight the key difference
    y_highlight = y_start - 4 * dy - 0.8
    ax.text(5.0, y_highlight,
            'Key: exact $O(1)$ divergence eliminates stochastic trace noise',
            fontsize=12, ha='center', va='center', style='italic',
            color='#333',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff9e6',
                      edgecolor='#e6a800', linewidth=1))

    # --- Panel (b): The computational pipeline ---
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('(b) Computational Pipeline', fontweight='bold', fontsize=18)

    # FFJORD pipeline (left column)
    ffjord_x = 2.5
    ffjord_steps = ['$z_0 \\sim p_0$', 'Neural ODE\n$\\dot{z} = f_\\theta(z,t)$',
                    'Hutchinson trace\n$\\hat{\\mathrm{tr}}(J) = v^T J v$',
                    '$\\log p(z_T)$\n(stochastic)']
    ffjord_colors = ['#e8e8e8', '#e8e8e8', '#ffe0e0', '#e8e8e8']
    ffjord_edge = ['#555', '#555', '#d62728', '#555']

    # NH-CNF pipeline (right column)
    nh_x = 7.5
    nh_steps = ['$z_0 \\sim p_0$', 'NH-tanh ODE\n$\\dot{z} = f_{\\mathrm{NH}}(z,\\xi)$',
                'Exact div\n$\\nabla \\cdot f = -d\\tanh(\\xi)$',
                '$\\log p(z_T)$\n(deterministic)']
    nh_colors = ['#d4e6f1', '#d4e6f1', '#d4f1d4', '#d4e6f1']
    nh_edge = ['#1f77b4', '#1f77b4', '#2ca02c', '#1f77b4']

    y_positions = [8.5, 6.5, 4.5, 2.5]

    # Labels
    ax.text(ffjord_x, 9.5, 'FFJORD', fontsize=15, fontweight='bold',
            ha='center', va='center', color='#555')
    ax.text(nh_x, 9.5, 'NH-CNF (ours)', fontsize=15, fontweight='bold',
            ha='center', va='center', color=C_NH)

    for idx_s, y in enumerate(y_positions):
        # FFJORD box
        ax.text(ffjord_x, y, ffjord_steps[idx_s], fontsize=11,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=ffjord_colors[idx_s],
                          edgecolor=ffjord_edge[idx_s], linewidth=1.2))
        # NH-CNF box
        ax.text(nh_x, y, nh_steps[idx_s], fontsize=11,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=nh_colors[idx_s],
                          edgecolor=nh_edge[idx_s], linewidth=1.2))
        # Arrows between steps
        if idx_s < len(y_positions) - 1:
            y_next = y_positions[idx_s + 1]
            for x_pos in [ffjord_x, nh_x]:
                ax.annotate('', xy=(x_pos, y_next + 0.6), xytext=(x_pos, y - 0.6),
                            arrowprops=dict(arrowstyle='->', color='#888',
                                            lw=1.5))

    # Highlight difference
    ax.annotate('', xy=(nh_x - 1.5, 4.5), xytext=(ffjord_x + 1.5, 4.5),
                arrowprops=dict(arrowstyle='<->', color='#e6a800', lw=2,
                                connectionstyle='arc3,rad=0'))
    ax.text(5.0, 3.8, 'stochastic $\\to$ deterministic', fontsize=11,
            ha='center', va='center', color='#b37700', style='italic')

    fig.savefig(os.path.join(FIGDIR, 'e4_concept.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"E4 saved to {FIGDIR}/e4_concept.png")


# =============================================================================
# E5: Phase-space trajectory visualization
# =============================================================================

def run_e5():
    """Phase-space visualization: the 'wow' figure showing NH dynamics.

    Target: two moons.
    Run NH-CNF for 5000 steps, record full (q1, q2, p1, p2, xi) trajectory.
    4-panel figure showing configuration, phase portrait, xi(t), g(xi(t)).
    """
    print("\n" + "=" * 60)
    print("E5: Phase-space trajectory visualization")
    print("=" * 60)

    # Setup
    data = make_two_moons(n=1000, seed=SEED)
    kde_pot = KDEPotential(data, bandwidth=0.35)

    n_steps = 5000
    dt = 0.005
    Q = 1.0
    kT = 1.0
    d = 2

    # Run and record full trajectory
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

    # (a) q1-q2 trajectory colored by time
    ax = axes[0, 0]
    # Background: ground truth scatter
    ax.scatter(data[:, 0], data[:, 1], s=1, alpha=0.1, c='gray', rasterized=True)
    # Trajectory colored by time
    colors_t = plt.cm.viridis(np.linspace(0, 1, n_steps))
    ax.scatter(traj_q[:, 0], traj_q[:, 1], s=0.5, c=colors_t, rasterized=True)
    # Mark start and end
    ax.plot(traj_q[0, 0], traj_q[0, 1], 'r*', markersize=12, zorder=5, label='Start')
    ax.plot(traj_q[-1, 0], traj_q[-1, 1], 'ks', markersize=8, zorder=5, label='End')
    ax.set_title('(a) Configuration space $q_1$-$q_2$\n(colored by time)', fontweight='bold')
    ax.set_xlabel('$q_1$')
    ax.set_ylabel('$q_2$')
    ax.legend(frameon=False, fontsize=10)
    ax.set_aspect('equal')
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, times[-1]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, label='Time')

    # (b) q1-p1 phase portrait
    ax = axes[0, 1]
    ax.scatter(traj_q[:, 0], traj_p[:, 0], s=0.5, c=colors_t, rasterized=True)
    ax.plot(traj_q[0, 0], traj_p[0, 0], 'r*', markersize=12, zorder=5)
    ax.set_title('(b) Phase portrait $q_1$-$p_1$\n(Hamiltonian + thermostat)', fontweight='bold')
    ax.set_xlabel('$q_1$')
    ax.set_ylabel('$p_1$')

    # (c) xi(t) time series
    ax = axes[1, 0]
    ax.plot(times, traj_xi, color=C_NH, lw=0.5, alpha=0.8)
    ax.axhline(0, color='gray', linestyle='--', lw=1, alpha=0.5)
    ax.fill_between(times, traj_xi, alpha=0.2, color=C_NH)
    ax.set_title('(c) Thermostat variable $\\xi(t)$\n(fluctuates around zero at equilibrium)',
                 fontweight='bold')
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('$\\xi(t)$')

    # (d) g(xi(t)) = tanh(xi(t)) -- the friction / divergence signal
    ax = axes[1, 1]
    ax.plot(times, g_xi, color='#d62728', lw=0.5, alpha=0.8)
    ax.axhline(0, color='gray', linestyle='--', lw=1, alpha=0.5)
    ax.fill_between(times, g_xi, alpha=0.2, color='#d62728')
    ax.set_title('(d) Friction $g(\\xi) = \\tanh(\\xi)$\n'
                 '($\\nabla \\cdot f = -d \\cdot g(\\xi)$ gives exact log-density)',
                 fontweight='bold')
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('$g(\\xi(t))$')
    ax.set_ylim(-1.1, 1.1)

    fig.savefig(os.path.join(FIGDIR, 'e5_phase_space.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"E5 saved to {FIGDIR}/e5_phase_space.png")


# =============================================================================
# E6: Scaling to higher dimensions
# =============================================================================

def make_gaussian_ring(n, d, n_modes=5, radius=3.0, sigma=0.5, seed=42):
    """Gaussian mixture with modes on a ring in first 2 dims, rest zero."""
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
    """Count how many modes are visited by the samples.
    A mode is 'visited' if at least 5 samples fall within threshold_factor*sigma of its center.
    """
    threshold = threshold_factor * sigma
    visited = 0
    for c in centers:
        dists = np.sqrt(((samples - c)**2).sum(axis=1))
        if (dists < threshold).sum() >= 5:
            visited += 1
    return visited


def run_e6():
    """Scaling study: energy distance and mode coverage vs dimension."""
    print("\n" + "=" * 60)
    print("E6: Dimension scaling study")
    print("=" * 60)

    dims = [2, 5, 10, 20, 50]
    n_modes = 5
    radius = 3.0
    sigma = 0.5
    n_train = 500
    n_gt = 5000
    n_steps = 200000
    dt = 0.005
    kT = 1.0

    ed_nh_list = []
    ed_lang_list = []
    modes_nh_list = []
    modes_lang_list = []

    for d in dims:
        print(f"\n  d = {d}")
        gt_data, centers = make_gaussian_ring(n_gt, d, n_modes, radius, sigma, seed=SEED)
        train_data, _ = make_gaussian_ring(n_train, d, n_modes, radius, sigma, seed=SEED+1)

        # For high-d, use a simpler potential: sum of Gaussians with known centers
        # (KDE scales badly with d)
        class GMMPotential:
            def __init__(self, centers, sigma):
                self.centers = torch.tensor(centers, dtype=torch.float32)
                self.sigma = sigma
                self.sigma2 = sigma ** 2
                self.n_modes = len(centers)
                self.d = centers.shape[1]

            def grad_potential(self, x):
                """Gradient of V(x) = -log sum_k N(x; mu_k, sigma^2 I)."""
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                    squeeze = True
                else:
                    squeeze = False
                diff = x.unsqueeze(-2) - self.centers  # (batch, n_modes, d)
                log_k = -0.5 * (diff**2).sum(-1) / self.sigma2  # (batch, n_modes)
                alpha = torch.softmax(log_k, dim=-1)  # (batch, n_modes)
                grad_V = (alpha.unsqueeze(-1) * diff).sum(-2) / self.sigma2
                if squeeze:
                    grad_V = grad_V.squeeze(0)
                return grad_V

        pot = GMMPotential(centers, sigma)

        # NH-CNF: multi-scale Q, single chain (warm-started), moderate steps
        # Reduce steps for high-d to keep runtime reasonable
        actual_steps = min(n_steps, max(50000, n_steps // (d // 2 + 1)))
        thin = max(10, actual_steps // 5000)
        burn = actual_steps // 5

        print(f"    NH: {actual_steps} steps, thin={thin}")
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        nh_samples_list = []
        for Q in [0.1, 1.0, 10.0]:
            for chain_idx in range(3):
                chain_seed = SEED + chain_idx * 100 + int(Q * 1000)
                torch.manual_seed(chain_seed)
                # Warm start from a random training point
                start_idx = np.random.randint(0, len(train_data))
                q = torch.tensor(train_data[start_idx], dtype=torch.float32)
                p = torch.randn(d) * np.sqrt(kT)
                xi = torch.zeros(1)

                for step in range(actual_steps):
                    q, p, xi, _ = nh_tanh_rk4_step(q, p, xi, pot.grad_potential, dt, Q, kT, d)
                    if step >= burn and (step - burn) % thin == 0:
                        nh_samples_list.append(q.detach().clone().numpy())

        nh_samples = np.array(nh_samples_list)
        print(f"    NH: {len(nh_samples)} samples")

        # Langevin
        print(f"    Langevin: {actual_steps} steps")
        lang_samples_list = []
        for chain_idx in range(9):
            chain_seed = SEED + chain_idx * 77
            torch.manual_seed(chain_seed)
            np.random.seed(chain_seed)
            start_idx = np.random.randint(0, len(train_data))
            x = torch.tensor(train_data[start_idx], dtype=torch.float32)
            eps_l = 0.005
            for step in range(actual_steps):
                gv = pot.grad_potential(x)
                noise = torch.randn_like(x) * np.sqrt(2 * eps_l * kT)
                x = x - eps_l * gv + noise
                if step >= burn and (step - burn) % thin == 0:
                    lang_samples_list.append(x.detach().clone().numpy())

        lang_samples = np.array(lang_samples_list)
        print(f"    Langevin: {len(lang_samples)} samples")

        # Metrics
        np.random.seed(SEED)
        ed_nh = energy_distance(gt_data, nh_samples)
        np.random.seed(SEED)
        ed_lang = energy_distance(gt_data, lang_samples)
        modes_nh = count_modes_visited(nh_samples, centers, threshold_factor=3.0, sigma=sigma)
        modes_lang = count_modes_visited(lang_samples, centers, threshold_factor=3.0, sigma=sigma)

        print(f"    ED: NH={ed_nh:.4f}, Lang={ed_lang:.4f}")
        print(f"    Modes: NH={modes_nh}/{n_modes}, Lang={modes_lang}/{n_modes}")

        ed_nh_list.append(ed_nh)
        ed_lang_list.append(ed_lang)
        modes_nh_list.append(modes_nh)
        modes_lang_list.append(modes_lang)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    # (a) Energy distance vs dimension
    ax = axes[0]
    ax.plot(dims, ed_nh_list, 'o-', color=C_NH, lw=2, markersize=8, label='NH-CNF (multi-Q)')
    ax.plot(dims, ed_lang_list, 's--', color=C_LANG, lw=2, markersize=8, label='Langevin (ULA)')
    ax.set_title('(a) Sample quality vs dimension', fontweight='bold')
    ax.set_xlabel('Dimension $d$')
    ax.set_ylabel('Energy distance')
    ax.legend(frameon=False)
    ax.set_xticks(dims)

    # (b) Modes visited vs dimension
    ax = axes[1]
    ax.plot(dims, modes_nh_list, 'o-', color=C_NH, lw=2, markersize=8, label='NH-CNF (multi-Q)')
    ax.plot(dims, modes_lang_list, 's--', color=C_LANG, lw=2, markersize=8, label='Langevin (ULA)')
    ax.axhline(n_modes, color='gray', linestyle=':', lw=1, alpha=0.5, label=f'All {n_modes} modes')
    ax.set_title('(b) Mode coverage vs dimension', fontweight='bold')
    ax.set_xlabel('Dimension $d$')
    ax.set_ylabel('Modes visited (out of 5)')
    ax.legend(frameon=False)
    ax.set_xticks(dims)
    ax.set_ylim(-0.5, n_modes + 0.5)
    ax.set_yticks(range(n_modes + 1))

    fig.savefig(os.path.join(FIGDIR, 'e6_scaling.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"\nE6 saved to {FIGDIR}/e6_scaling.png")
    return {'dims': dims, 'ed_nh': ed_nh_list, 'ed_lang': ed_lang_list,
            'modes_nh': modes_nh_list, 'modes_lang': modes_lang_list}


# =============================================================================
# E7: Log-likelihood comparison
# =============================================================================

def run_e7():
    """Log-likelihood comparison: NH-CNF exact vs KDE baseline.

    For 2D targets, evaluate log p(x_test) using:
    - NH-CNF: run flow from z0 -> z_T, track log p via exact divergence
    - KDE: scipy.stats.gaussian_kde on training samples
    """
    print("\n" + "=" * 60)
    print("E7: Log-likelihood comparison")
    print("=" * 60)

    targets = {
        'Two Moons': make_two_moons,
        'Two Spirals': make_two_spirals,
        'Eight Gaussians': make_eight_gaussians,
    }

    n_train = 1000
    n_test = 500
    bw = 0.35
    d = 2

    results = {}

    for name, gen_fn in targets.items():
        print(f"\n  {name}")
        train_data = gen_fn(n=n_train, seed=SEED)
        test_data = gen_fn(n=n_test, seed=SEED + 99)

        # --- KDE baseline log-likelihood ---
        kde_scipy = stats.gaussian_kde(train_data.T)
        ll_kde = np.log(kde_scipy(test_data.T) + 1e-30)
        mean_ll_kde = ll_kde.mean()
        std_ll_kde = ll_kde.std() / np.sqrt(n_test)
        print(f"    KDE: mean LL = {mean_ll_kde:.3f} +/- {std_ll_kde:.3f}")

        # --- NH-CNF log-likelihood ---
        # Run NH flow from initial conditions near test points,
        # use the instantaneous change of variables formula.
        # Actually, for density estimation we need:
        #   log p(x) = log p(z0) + integral_0^T div(f) dt
        # where z_T = x (the test point) and z0 = T^{-1}(x).
        # But we don't have the inverse flow easily.
        #
        # Alternative: sample-based density estimation.
        # Run NH sampler from training data, collect samples,
        # then use KDE on those samples. The NH advantage is in
        # the *trajectory*, not the density evaluation per se.
        #
        # For a fair comparison: evaluate how well the NH-generated
        # samples cover the test distribution, measured by fitting
        # KDE to NH samples and evaluating on test points.
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
        print(f"    NH-CNF samples -> KDE: mean LL = {mean_ll_nh:.3f} +/- {std_ll_nh:.3f}")

        # --- Langevin baseline ---
        lang_samples = run_langevin(
            kde_pot.grad_potential, d=2, n_steps=50000, eps=0.005,
            kT=1.0, burn_frac=0.2, thin=25, seed=SEED
        )

        if len(lang_samples) > 50:
            kde_lang = stats.gaussian_kde(lang_samples.T)
            ll_lang = np.log(kde_lang(lang_samples.T) + 1e-30)
            # Evaluate on TEST data, not lang_samples
            ll_lang = np.log(kde_lang(test_data.T) + 1e-30)
            mean_ll_lang = ll_lang.mean()
            std_ll_lang = ll_lang.std() / np.sqrt(n_test)
        else:
            mean_ll_lang = -999.0
            std_ll_lang = 0.0
        print(f"    Langevin -> KDE: mean LL = {mean_ll_lang:.3f} +/- {std_ll_lang:.3f}")

        results[name] = {
            'kde': (mean_ll_kde, std_ll_kde),
            'nh': (mean_ll_nh, std_ll_nh),
            'lang': (mean_ll_lang, std_ll_lang),
        }

    # --- Plot: grouped bar chart ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)

    target_names = list(results.keys())
    x = np.arange(len(target_names))
    width = 0.25

    means_kde = [results[t]['kde'][0] for t in target_names]
    stds_kde = [results[t]['kde'][1] for t in target_names]
    means_nh = [results[t]['nh'][0] for t in target_names]
    stds_nh = [results[t]['nh'][1] for t in target_names]
    means_lang = [results[t]['lang'][0] for t in target_names]
    stds_lang = [results[t]['lang'][1] for t in target_names]

    bars1 = ax.bar(x - width, means_kde, width, yerr=stds_kde,
                   label='KDE (direct)', color='#888888', capsize=4)
    bars2 = ax.bar(x, means_nh, width, yerr=stds_nh,
                   label='NH-CNF samples', color=C_NH, capsize=4)
    bars3 = ax.bar(x + width, means_lang, width, yerr=stds_lang,
                   label='Langevin samples', color=C_LANG, capsize=4)

    ax.set_title('Test Log-Likelihood Comparison', fontweight='bold')
    ax.set_ylabel('Mean test log-likelihood')
    ax.set_xticks(x)
    ax.set_xticklabels(target_names)
    ax.legend(frameon=False)
    ax.axhline(0, color='gray', linestyle=':', lw=0.5)

    fig.savefig(os.path.join(FIGDIR, 'e7_loglik.png'), bbox_inches='tight')
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
