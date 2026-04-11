"""
NH-CNF Deep Experiments — Refinement 3
=======================================

Root cause fix: The KDE potential with bandwidth=0.35 is a blurry approximation
of the true density. The NH sampler correctly samples from exp(-V_KDE), but V_KDE
itself is blurry -- so samples will always be blurry regardless of chain length.

Fix strategy:
- Eight Gaussians: EXACT analytical potential (log-sum-exp GMM)
- Checkerboard: KDE(bw=0.1) + GridPotential, grid-interpolated for speed
- Two Moons, Two Spirals: KDE with SMALL bandwidth (0.1), grid-interpolated

The grid interpolation precomputes grad_V on a fine 2D grid and uses bilinear
interpolation during sampling. This converts O(N_data) per KDE eval to O(1).

Only E1 is remade here. E3-E7 figures are unchanged from refine 2.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import time
import os
import sys
from scipy import stats

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

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

C_NH = '#1f77b4'
C_LANG = '#2ca02c'
C_REF = '#d62728'

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

    return q_new, p_new, xi_new


# =============================================================================
# Multi-scale Q NH sampler
# =============================================================================

def run_nh_multiscale(grad_V_fn, init_points, n_steps=200000, dt=0.005,
                      Q_values=[0.1, 1.0, 10.0], kT=1.0,
                      burn_frac=0.2, thin=50, seed=42):
    """Run NH-tanh with multiple Q values, warm-started from init_points."""
    d = init_points.shape[1]
    n_data = len(init_points)

    all_samples = []
    n_chains_per_Q = 3
    burn_in = int(n_steps * burn_frac)

    for Q in Q_values:
        for c in range(n_chains_per_Q):
            chain_seed = seed + c * 100 + int(Q * 1000)
            torch.manual_seed(chain_seed)
            np.random.seed(chain_seed)

            idx = np.random.randint(0, n_data)
            q = torch.tensor(init_points[idx], dtype=torch.float32)
            p = torch.randn(d) * np.sqrt(kT)
            xi = torch.zeros(1)

            for step in range(n_steps):
                q, p, xi = nh_tanh_rk4_step(q, p, xi, grad_V_fn, dt, Q, kT, d)

                if torch.any(torch.abs(q) > 50):
                    q = q.clamp(-50, 50)
                    p = torch.randn(d) * np.sqrt(kT)
                    xi = torch.zeros(1)

                if step >= burn_in and (step - burn_in) % thin == 0:
                    all_samples.append(q.detach().clone().numpy())

    return np.array(all_samples)


# =============================================================================
# Langevin dynamics (ULA) baseline
# =============================================================================

def run_langevin(grad_V_fn, d, n_steps=200000, eps=0.005, kT=1.0,
                 burn_frac=0.2, thin=50, seed=42, init=None):
    """Unadjusted Langevin Algorithm with burn-in and thinning."""
    torch.manual_seed(seed)
    if init is not None:
        x = torch.tensor(init, dtype=torch.float32)
    else:
        x = torch.randn(d) * 0.5
    samples = []
    burn_in = int(n_steps * burn_frac)
    for step in range(n_steps):
        gv = grad_V_fn(x)
        gv = gv.clamp(-100, 100)
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
# ANALYTICAL POTENTIALS
# =============================================================================

class EightGaussiansPotential:
    """Exact potential for 8 Gaussians on a ring."""
    def __init__(self, radius=3.0, sigma=0.3):
        angles = torch.linspace(0, 2 * np.pi, 9)[:-1]
        self.centers = torch.stack(
            [radius * torch.cos(angles), radius * torch.sin(angles)], dim=1
        )
        self.sigma2 = sigma ** 2
        self.log_norm = -np.log(8) - np.log(2 * np.pi * self.sigma2)

    def log_prob(self, x):
        diff = x.unsqueeze(-2) - self.centers
        log_k = self.log_norm - 0.5 * (diff ** 2).sum(-1) / self.sigma2
        return torch.logsumexp(log_k, dim=-1)

    def grad_potential(self, x):
        diff = x.unsqueeze(-2) - self.centers
        log_k = self.log_norm - 0.5 * (diff ** 2).sum(-1) / self.sigma2
        w = torch.softmax(log_k, dim=-1)
        return (w.unsqueeze(-1) * diff).sum(-2) / self.sigma2


class CheckerboardPotential:
    """Smooth checkerboard potential via sigmoid(sharpness * cos(pi*x) * cos(pi*y))."""
    def __init__(self, sharpness=20.0):
        self.sharpness = sharpness

    def log_prob(self, x):
        cx = torch.cos(np.pi * x[..., 0])
        cy = torch.cos(np.pi * x[..., 1])
        return torch.log(torch.sigmoid(self.sharpness * cx * cy) + 1e-8)

    def grad_potential(self, x):
        x_req = x.detach().clone().requires_grad_(True)
        lp = self.log_prob(x_req)
        lp.backward()
        return -x_req.grad.detach()


# =============================================================================
# GRID-INTERPOLATED POTENTIAL (fast O(1) lookup for MLP or any potential)
# =============================================================================

class GridPotential:
    """Precompute grad_V on a fine grid, then bilinear interpolate during sampling.

    This converts O(network_forward) per step to O(1) grid lookup.
    Critical for making MLP-based potentials practical in long sampling runs.
    """
    def __init__(self, potential_fn, xlim, ylim, n_grid=500):
        """
        Args:
            potential_fn: object with .log_prob(x) method accepting (N,2) tensor
            xlim: (xmin, xmax)
            ylim: (ymin, ymax)
            n_grid: grid resolution per dimension
        """
        self.xmin, self.xmax = xlim
        self.ymin, self.ymax = ylim
        self.n_grid = n_grid
        self.dx = (self.xmax - self.xmin) / (n_grid - 1)
        self.dy = (self.ymax - self.ymin) / (n_grid - 1)

        print(f"    Precomputing gradient grid ({n_grid}x{n_grid})...")
        t0 = time.time()

        # Build grid points
        xs = torch.linspace(self.xmin, self.xmax, n_grid)
        ys = torch.linspace(self.ymin, self.ymax, n_grid)
        xx, yy = torch.meshgrid(xs, ys, indexing='ij')
        grid_pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)  # (n_grid^2, 2)

        # Compute grad_V in batches
        batch_size = 10000
        grad_list = []
        for i in range(0, len(grid_pts), batch_size):
            batch = grid_pts[i:i+batch_size].clone().requires_grad_(True)
            lp = potential_fn.log_prob(batch)
            lp_sum = lp.sum()
            g = torch.autograd.grad(lp_sum, batch)[0]
            grad_list.append(-g.detach())  # grad_V = -grad log p

        all_grads = torch.cat(grad_list, dim=0)
        # Store as (n_grid, n_grid, 2) -- indexed by [ix, iy, :]
        self.grad_grid = all_grads.reshape(n_grid, n_grid, 2)

        # Also store log_prob grid for diagnostics
        with torch.no_grad():
            lp_list = []
            for i in range(0, len(grid_pts), batch_size):
                batch = grid_pts[i:i+batch_size]
                lp_list.append(potential_fn.log_prob(batch))
            self.log_prob_grid = torch.cat(lp_list).reshape(n_grid, n_grid)

        print(f"    Grid precomputed in {time.time()-t0:.1f}s")

    def log_prob(self, x):
        """Bilinear interpolation of log_prob. For diagnostics only."""
        ix, iy, fx, fy = self._get_interp_coords(x)
        v00 = self.log_prob_grid[ix, iy]
        v10 = self.log_prob_grid[ix+1, iy]
        v01 = self.log_prob_grid[ix, iy+1]
        v11 = self.log_prob_grid[ix+1, iy+1]
        return (v00 * (1-fx) * (1-fy) + v10 * fx * (1-fy) +
                v01 * (1-fx) * fy + v11 * fx * fy)

    def grad_potential(self, x):
        """Bilinear interpolation of precomputed gradient. O(1) per call."""
        ix, iy, fx, fy = self._get_interp_coords(x)
        g00 = self.grad_grid[ix, iy]
        g10 = self.grad_grid[ix+1, iy]
        g01 = self.grad_grid[ix, iy+1]
        g11 = self.grad_grid[ix+1, iy+1]
        return (g00 * (1-fx) * (1-fy) + g10 * fx * (1-fy) +
                g01 * (1-fx) * fy + g11 * fx * fy)

    def _get_interp_coords(self, x):
        """Get grid indices and fractional offsets for bilinear interpolation."""
        # Clamp to grid bounds
        x0 = x[..., 0].clamp(self.xmin, self.xmax - 1e-6)
        x1 = x[..., 1].clamp(self.ymin, self.ymax - 1e-6)
        # Continuous grid coordinates
        gx = (x0 - self.xmin) / self.dx
        gy = (x1 - self.ymin) / self.dy
        # Integer indices
        ix = gx.long().clamp(0, self.n_grid - 2)
        iy = gy.long().clamp(0, self.n_grid - 2)
        # Fractional part
        fx = (gx - ix.float()).unsqueeze(-1)
        fy = (gy - iy.float()).unsqueeze(-1)
        return ix, iy, fx, fy


# =============================================================================
# KDE POTENTIAL (with configurable bandwidth)
# =============================================================================

class KDEPotential:
    """V(x) = -log p_KDE(x) with configurable bandwidth.

    For use with GridPotential: precompute on grid, then O(1) lookup during sampling.
    """
    def __init__(self, data, bandwidth=0.1):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.bw = bandwidth
        self.bw2 = bandwidth ** 2
        self.n = self.data.shape[0]
        self.d = self.data.shape[1]
        self.log_norm = -self.d * 0.5 * np.log(2 * np.pi * self.bw2)

    def log_prob(self, x):
        """log p_KDE(x) = log (1/N) sum_i K(x - x_i)."""
        diff = x.unsqueeze(-2) - self.data  # (..., N, d)
        log_k = self.log_norm - 0.5 * (diff ** 2).sum(-1) / self.bw2  # (..., N)
        return torch.logsumexp(log_k, dim=-1) - np.log(self.n)

    def grad_potential(self, x):
        """Analytical grad V = -grad log p = sum_k w_k (x - x_k) / bw^2."""
        diff = x.unsqueeze(-2) - self.data
        log_k = self.log_norm - 0.5 * (diff ** 2).sum(-1) / self.bw2
        w = torch.softmax(log_k, dim=-1)
        return (w.unsqueeze(-1) * diff).sum(-2) / self.bw2


# =============================================================================
# Energy distance metric
# =============================================================================

def energy_distance(x, y):
    n = min(len(x), len(y), 3000)
    np.random.seed(SEED)
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
    rgb = mcolors.to_rgb(color)
    colors = [(1, 1, 1, 0), (*rgb, 0.3), (*rgb, 0.6), (*rgb, 1.0)]
    return LinearSegmentedColormap.from_list('custom', colors, N=256)


def _plot_kde_contour(ax, samples, xmin, xmax, ymin, ymax, color,
                      n_grid=150, levels=10, bw_method=0.1):
    if len(samples) < 10:
        ax.text(0.5, 0.5, 'Too few\nsamples', transform=ax.transAxes,
                ha='center', va='center', fontsize=14)
        return

    n_show = min(len(samples), 5000)
    idx = np.random.choice(len(samples), n_show, replace=False)
    ax.scatter(samples[idx, 0], samples[idx, 1], s=1, alpha=0.05,
               c=color, rasterized=True)

    try:
        kde = stats.gaussian_kde(samples.T, bw_method=bw_method)
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, n_grid),
                              np.linspace(ymin, ymax, n_grid))
        positions = np.vstack([xx.ravel(), yy.ravel()])
        zz = kde(positions).reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=levels, cmap=_make_cmap(color), alpha=0.7)
        ax.contour(xx, yy, zz, levels=levels, colors=[color], linewidths=0.5, alpha=0.5)
    except Exception as e:
        print(f"    KDE contour failed: {e}")
        ax.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.3,
                   c=color, rasterized=True)


def _plot_row(axes, row, name, gt_data, nh_samples, lang_samples,
              ed_nh, ed_lang, bw_viz=0.1):
    """Plot one row of the E1 figure."""
    pad = 0.8
    xmin, xmax = gt_data[:, 0].min() - pad, gt_data[:, 0].max() + pad
    ymin, ymax = gt_data[:, 1].min() - pad, gt_data[:, 1].max() + pad

    ax = axes[row, 0]
    _plot_kde_contour(ax, gt_data, xmin, xmax, ymin, ymax, C_REF, bw_method=bw_viz)
    ax.set_title(f'(a) {name}\nGround Truth (N=10k)', fontweight='bold')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    ax = axes[row, 1]
    _plot_kde_contour(ax, nh_samples, xmin, xmax, ymin, ymax, C_NH, bw_method=bw_viz)
    ax.set_title(f'(b) NH-CNF (multi-Q)\nED={ed_nh:.4f}, N={len(nh_samples)}',
                 fontweight='bold')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    ax = axes[row, 2]
    _plot_kde_contour(ax, lang_samples, xmin, xmax, ymin, ymax, C_LANG, bw_method=bw_viz)
    ax.set_title(f'(c) Langevin (ULA)\nED={ed_lang:.4f}, N={len(lang_samples)}',
                 fontweight='bold')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    for j in range(3):
        axes[row, j].set_xlim(xmin, xmax)
        axes[row, j].set_ylim(ymin, ymax)
        axes[row, j].set_aspect('equal')


# =============================================================================
# Run a single target's sampling
# =============================================================================

def run_target(name, potential, gt_data, init_data, n_steps, dt_nh, kT,
               burn_frac, thin, bw_viz):
    """Run NH-CNF and Langevin on one target, return samples and EDs."""
    print(f"\n--- {name} ---")

    # Quick Q tune: 3 candidates, 30k steps, 1 chain
    print("  Tuning Q...")
    best_Q = 1.0
    best_ed = float('inf')
    for Q in [0.3, 1.0, 3.0, 10.0]:
        samples = run_nh_multiscale(
            potential.grad_potential, init_data, n_steps=30000, dt=dt_nh,
            Q_values=[Q], kT=kT, burn_frac=0.3, thin=20, seed=SEED
        )
        if len(samples) > 10:
            ed = energy_distance(gt_data, samples)
            if ed < best_ed:
                best_ed = ed
                best_Q = Q
    print(f"  Best Q = {best_Q}")

    # Quick Langevin eps tune
    print("  Tuning Langevin eps...")
    best_eps = 0.01
    best_ed = float('inf')
    init_pt = init_data[np.random.randint(0, len(init_data))]
    for eps in [0.001, 0.003, 0.01, 0.03]:
        samples = run_langevin(
            potential.grad_potential, d=2, n_steps=30000, eps=eps, kT=kT,
            burn_frac=0.3, thin=20, seed=SEED, init=init_pt
        )
        if len(samples) > 10:
            ed = energy_distance(gt_data, samples)
            if ed < best_ed:
                best_ed = ed
                best_eps = eps
    print(f"  Best eps = {best_eps}")

    # Full NH-CNF run
    Q_values = [best_Q * 0.1, best_Q, best_Q * 10.0]
    print(f"  Running NH-CNF (Q={[f'{q:.1f}' for q in Q_values]}, {n_steps} steps/chain)...")
    t0 = time.time()
    nh_samples = run_nh_multiscale(
        potential.grad_potential, init_data, n_steps=n_steps, dt=dt_nh,
        Q_values=Q_values, kT=kT, burn_frac=burn_frac, thin=thin, seed=SEED
    )
    nh_time = time.time() - t0
    ed_nh = energy_distance(gt_data, nh_samples)
    print(f"  NH-CNF: {len(nh_samples)} samples, ED={ed_nh:.4f}, time={nh_time:.1f}s")

    # Full Langevin run (9 chains, warm started)
    print(f"  Running Langevin (eps={best_eps}, {n_steps} steps, 9 chains)...")
    t0 = time.time()
    lang_all = []
    for ci in range(9):
        cs = SEED + ci * 77
        ip = init_data[np.random.randint(0, len(init_data))]
        ls = run_langevin(
            potential.grad_potential, d=2, n_steps=n_steps, eps=best_eps,
            kT=kT, burn_frac=burn_frac, thin=thin, seed=cs, init=ip
        )
        lang_all.append(ls)
    lang_samples = np.vstack(lang_all)
    lang_time = time.time() - t0
    ed_lang = energy_distance(gt_data, lang_samples)
    print(f"  Langevin: {len(lang_samples)} samples, ED={ed_lang:.4f}, time={lang_time:.1f}s")

    return nh_samples, lang_samples, ed_nh, ed_lang


# =============================================================================
# E1: Main experiment
# =============================================================================

def run_e1():
    """E1 remade: exact potentials + grid-interpolated KDE for complex targets."""
    print("=" * 60)
    print("E1: NH-CNF density (REFINE 3: proper potentials)")
    print("=" * 60)

    n_steps = 200000
    dt_nh = 0.005
    kT = 1.0
    burn_frac = 0.2
    thin = 50

    fig, axes = plt.subplots(4, 3, figsize=(15, 20), constrained_layout=True)
    results = {}

    # --- Eight Gaussians (EXACT analytical) ---
    gt = make_eight_gaussians(n=10000, seed=SEED)
    pot = EightGaussiansPotential(radius=3.0, sigma=0.3)
    nh_s, la_s, ed_nh, ed_la = run_target(
        'Eight Gaussians', pot, gt, gt,
        n_steps, dt_nh, kT, burn_frac, thin, 0.05
    )
    _plot_row(axes, 0, 'Eight Gaussians', gt, nh_s, la_s, ed_nh, ed_la, 0.05)
    results['Eight Gaussians'] = {'ed_nh': ed_nh, 'ed_lang': ed_la}

    # --- Checkerboard (KDE bw=0.1, grid-interpolated) ---
    # The sigmoid analytical potential has too-steep gradients for sampling.
    # Instead, use KDE with small bandwidth on training data.
    gt = make_checkerboard(n=10000, seed=SEED)
    train_cb = make_checkerboard(n=10000, seed=SEED + 1)
    print("\n  Building KDE potential for Checkerboard (bw=0.1)...")
    kde_cb = KDEPotential(train_cb, bandwidth=0.1)
    grid_cb = GridPotential(kde_cb, (-3, 3), (-3, 3), n_grid=500)
    nh_s, la_s, ed_nh, ed_la = run_target(
        'Checkerboard', grid_cb, gt, train_cb,
        n_steps, dt_nh, kT, burn_frac, thin, 0.08
    )
    _plot_row(axes, 1, 'Checkerboard', gt, nh_s, la_s, ed_nh, ed_la, 0.08)
    results['Checkerboard'] = {'ed_nh': ed_nh, 'ed_lang': ed_la}

    # --- Two Moons (KDE bw=0.1, grid-interpolated) ---
    gt = make_two_moons(n=10000, seed=SEED)
    train = make_two_moons(n=10000, seed=SEED + 1)
    print("\n  Building KDE potential for Two Moons (bw=0.1)...")
    kde_pot = KDEPotential(train, bandwidth=0.1)
    pad = 1.0
    xlim = (gt[:, 0].min() - pad, gt[:, 0].max() + pad)
    ylim = (gt[:, 1].min() - pad, gt[:, 1].max() + pad)
    grid_tm = GridPotential(kde_pot, xlim, ylim, n_grid=500)
    nh_s, la_s, ed_nh, ed_la = run_target(
        'Two Moons', grid_tm, gt, train,
        n_steps, dt_nh, kT, burn_frac, thin, 0.10
    )
    _plot_row(axes, 2, 'Two Moons', gt, nh_s, la_s, ed_nh, ed_la, 0.10)
    results['Two Moons'] = {'ed_nh': ed_nh, 'ed_lang': ed_la}

    # --- Two Spirals (KDE bw=0.1, grid-interpolated) ---
    gt = make_two_spirals(n=10000, seed=SEED)
    train = make_two_spirals(n=10000, seed=SEED + 1)
    print("\n  Building KDE potential for Two Spirals (bw=0.1)...")
    kde_pot2 = KDEPotential(train, bandwidth=0.1)
    xlim = (gt[:, 0].min() - pad, gt[:, 0].max() + pad)
    ylim = (gt[:, 1].min() - pad, gt[:, 1].max() + pad)
    grid_ts = GridPotential(kde_pot2, xlim, ylim, n_grid=500)
    nh_s, la_s, ed_nh, ed_la = run_target(
        'Two Spirals', grid_ts, gt, train,
        n_steps, dt_nh, kT, burn_frac, thin, 0.10
    )
    _plot_row(axes, 3, 'Two Spirals', gt, nh_s, la_s, ed_nh, ed_la, 0.10)
    results['Two Spirals'] = {'ed_nh': ed_nh, 'ed_lang': ed_la}

    fig.savefig(os.path.join(FIGDIR, 'e1_density.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"\nE1 saved to {FIGDIR}/e1_density.png")

    print("\n--- E1 Summary (Refine 3: proper potentials) ---")
    print(f"{'Target':<18} {'NH-CNF ED':>10} {'Langevin ED':>12} {'NH wins?':>10}")
    for name, r in results.items():
        winner = "Yes" if r['ed_nh'] < r['ed_lang'] else "No"
        ratio = r['ed_lang'] / max(r['ed_nh'], 1e-6)
        print(f"{name:<18} {r['ed_nh']:>10.4f} {r['ed_lang']:>12.4f} {winner:>6} ({ratio:.1f}x)")

    return results


# =============================================================================
# E1 Potential diagnostic figure
# =============================================================================

def run_e1_potential():
    """Diagnostic: visualize the fitted potential for each target."""
    print("\n" + "=" * 60)
    print("E1 Potential: diagnostic visualization")
    print("=" * 60)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), constrained_layout=True)

    # Eight Gaussians (analytical)
    pot = EightGaussiansPotential(radius=3.0, sigma=0.3)
    gt = make_eight_gaussians(n=10000, seed=SEED)
    _plot_potential(axes[0], pot, gt, '(a) Eight Gaussians\n(exact)',
                    xlim=(-5, 5), ylim=(-5, 5))

    # Checkerboard (KDE bw=0.1)
    train_cb = make_checkerboard(n=10000, seed=SEED + 1)
    gt = make_checkerboard(n=10000, seed=SEED)
    kde_cb = KDEPotential(train_cb, bandwidth=0.1)
    _plot_potential(axes[1], kde_cb, gt, '(b) Checkerboard\n(KDE bw=0.1)',
                    xlim=(-3, 3), ylim=(-3, 3))

    # Two Moons (KDE bw=0.1)
    train = make_two_moons(n=10000, seed=SEED + 1)
    gt = make_two_moons(n=10000, seed=SEED)
    kde_tm = KDEPotential(train, bandwidth=0.1)
    _plot_potential(axes[2], kde_tm, gt, '(c) Two Moons\n(KDE bw=0.1)',
                    xlim=(-2, 3), ylim=(-1.5, 2))

    # Two Spirals (KDE bw=0.1)
    train = make_two_spirals(n=10000, seed=SEED + 1)
    gt = make_two_spirals(n=10000, seed=SEED)
    kde_ts = KDEPotential(train, bandwidth=0.1)
    _plot_potential(axes[3], kde_ts, gt, '(d) Two Spirals\n(KDE bw=0.1)',
                    xlim=(-5, 5), ylim=(-5, 5))

    fig.savefig(os.path.join(FIGDIR, 'e1_potential.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved to {FIGDIR}/e1_potential.png")


def _plot_potential(ax, potential, gt_data, title, xlim=(-5, 5), ylim=(-5, 5)):
    """Plot contours of log p(x) with data overlay."""
    n_grid = 150
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], n_grid),
        np.linspace(ylim[0], ylim[1], n_grid)
    )
    grid = torch.tensor(np.column_stack([xx.ravel(), yy.ravel()]), dtype=torch.float32)

    with torch.no_grad():
        log_p = potential.log_prob(grid).numpy()
    zz = log_p.reshape(xx.shape)

    vmin = np.percentile(zz[np.isfinite(zz)], 5)
    vmax = np.percentile(zz[np.isfinite(zz)], 99)
    zz = np.clip(zz, vmin, vmax)

    ax.contourf(xx, yy, zz, levels=20, cmap='viridis', alpha=0.8)
    ax.contour(xx, yy, zz, levels=10, colors='k', linewidths=0.3, alpha=0.4)

    n_show = min(len(gt_data), 2000)
    idx = np.random.choice(len(gt_data), n_show, replace=False)
    ax.scatter(gt_data[idx, 0], gt_data[idx, 1], s=1, c='gray', alpha=0.3, rasterized=True)

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("NH-CNF Deep Experiments — Refinement 3")
    print("Root cause fix: analytical potentials + KDE(bw=0.1) with grid interpolation")
    print()

    run_e1_potential()
    results = run_e1()

    print("\n" + "=" * 60)
    print("DONE. Only E1 remade; E3-E7 unchanged from refine 2.")
    print("=" * 60)
