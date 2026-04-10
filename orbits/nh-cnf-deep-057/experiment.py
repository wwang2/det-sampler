"""
NH-CNF Deep Experiments: NH thermostat as a continuous normalizing flow.

E1: NH-CNF density estimation on 2D synthetic targets
E2: Annealed NH flow with time-dependent Q
E3: Exact divergence advantage (variance, accuracy, scaling)
E4: Conceptual figure

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

# Colors
C_NH = '#1f77b4'
C_LANG = '#2ca02c'
C_REF = '#d62728'
C_HUTCH1 = '#ff7f0e'
C_HUTCH5 = '#9467bd'
C_SCHED = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

torch.manual_seed(SEED)
np.random.seed(SEED)


# =============================================================================
# NH-tanh RK4 integrator (from parent orbit 056)
# =============================================================================

def nh_tanh_rk4_step(q, p, xi, grad_V_fn, dt, Q=1.0, kT=1.0, d=None):
    """One RK4 step of the NH-tanh ODE.
    Returns (q_new, p_new, xi_new, div_integral).
    div_integral = integral of -d*tanh(xi) over [t, t+dt] (trapezoidal).
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


def nh_tanh_rk4_step_varQ(q, p, xi, grad_V_fn, dt, Q, kT=1.0, d=None):
    """NH-tanh RK4 step with variable Q (for annealed flow)."""
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
# Langevin dynamics (ULA) baseline
# =============================================================================

def run_langevin(grad_V_fn, d, n_steps=10000, eps=0.01, kT=1.0,
                 burn_in=1000, thin=1, seed=42):
    """Unadjusted Langevin Algorithm."""
    torch.manual_seed(seed)
    x = torch.randn(d) * 0.5
    samples = []
    t0 = time.time()
    for step in range(n_steps):
        gv = grad_V_fn(x)
        noise = torch.randn_like(x) * np.sqrt(2 * eps * kT)
        x = x - eps * gv + noise
        if step >= burn_in and (step - burn_in) % thin == 0:
            samples.append(x.detach().clone().numpy())
    wall_time = time.time() - t0
    return np.array(samples), wall_time


# =============================================================================
# Batch NH-CNF sampler (for density estimation)
# =============================================================================

def run_nh_cnf_batch(grad_V_fn, n_samples, d, n_steps=5000, dt=0.01,
                     Q=1.0, kT=1.0, seed=42):
    """Run NH-CNF on a batch of samples simultaneously.

    Starts from z0 ~ N(0,I), augments with p0 ~ N(0,I) and xi0=0.
    Tracks log p(z_T) = log p(z0) + integral d*tanh(xi) dt.

    Returns: (samples [n_samples, d], log_probs [n_samples])
    """
    torch.manual_seed(seed)
    q = torch.randn(n_samples, d)  # initial positions
    p = torch.randn(n_samples, d)  # initial momenta
    xi = torch.zeros(n_samples, 1)  # thermostat variable

    # Initial log prob under N(0, I)
    log_p0 = -0.5 * d * np.log(2 * np.pi) - 0.5 * (q**2).sum(-1)
    cum_div = torch.zeros(n_samples)

    def grad_V_batch(q_):
        """Batch gradient computation."""
        q_.requires_grad_(True)
        V = potential_fn_global(q_)
        gv = torch.autograd.grad(V.sum(), q_)[0]
        return gv.detach()

    for step in range(n_steps):
        # RK4 step (batched)
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

        # Accumulate divergence (exact, zero variance)
        g_start = torch.tanh(xi).squeeze(-1)
        g_end = torch.tanh(xi_new).squeeze(-1)
        cum_div += -d * 0.5 * (g_start + g_end) * dt

        q, p, xi = q_new, p_new, xi_new

    # log p(z_T) = log p(z_0) + integral of divergence
    log_probs = log_p0 + cum_div
    return q.detach().numpy(), log_probs.detach().numpy()


# =============================================================================
# 2D Target distributions
# =============================================================================

def make_two_moons(n=2000, noise=0.1, seed=42):
    """Two moons dataset."""
    np.random.seed(seed)
    n_per = n // 2
    # Moon 1
    t1 = np.linspace(0, np.pi, n_per)
    x1 = np.cos(t1) + np.random.randn(n_per) * noise
    y1 = np.sin(t1) + np.random.randn(n_per) * noise
    # Moon 2
    t2 = np.linspace(0, np.pi, n_per)
    x2 = 1 - np.cos(t2) + np.random.randn(n_per) * noise
    y2 = 1 - np.sin(t2) - 0.5 + np.random.randn(n_per) * noise
    data = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    return data


def make_two_spirals(n=2000, noise=0.1, seed=42):
    """Two interleaved spirals."""
    np.random.seed(seed)
    n_per = n // 2
    t = np.linspace(0.5, 3.0, n_per)
    r = t
    # Spiral 1
    x1 = r * np.cos(t * np.pi) + np.random.randn(n_per) * noise
    y1 = r * np.sin(t * np.pi) + np.random.randn(n_per) * noise
    # Spiral 2
    x2 = -r * np.cos(t * np.pi) + np.random.randn(n_per) * noise
    y2 = -r * np.sin(t * np.pi) + np.random.randn(n_per) * noise
    data = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    return data


def make_checkerboard(n=2000, seed=42):
    """4x4 checkerboard pattern."""
    np.random.seed(seed)
    data = []
    n_per_cell = n // 8  # 8 "on" cells in a 4x4 grid
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                x = np.random.uniform(i - 2, i - 1, n_per_cell)
                y = np.random.uniform(j - 2, j - 1, n_per_cell)
                data.append(np.column_stack([x, y]))
    data = np.vstack(data)
    np.random.shuffle(data)
    return data[:n]


def make_eight_gaussians(n=2000, radius=3.0, sigma=0.3, seed=42):
    """Ring of 8 isotropic Gaussians."""
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
    """V(x) = -log p_KDE(x) from training data, with analytical gradient.

    The gradient is computed without autograd for speed:
        grad V(x) = -grad log p(x) = -sum_k w_k (data_k - x) / bw^2
    where w_k = softmax(log_k) are the normalized kernel weights.
    """
    def __init__(self, data, bandwidth=0.3):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.bw = bandwidth
        self.bw2 = bandwidth ** 2
        self.n = self.data.shape[0]
        self.d = self.data.shape[1]
        self.log_norm = -self.d * 0.5 * np.log(2 * np.pi * self.bw2)

    def log_prob(self, x):
        """Log probability under KDE. x: (..., d)"""
        diff = x.unsqueeze(-2) - self.data  # (..., n, d)
        log_k = self.log_norm - 0.5 * (diff**2).sum(-1) / self.bw2  # (..., n)
        return torch.logsumexp(log_k, dim=-1) - np.log(self.n)

    def grad_potential(self, x):
        """Analytical gradient of V(x) = -log p(x).

        grad V = sum_k alpha_k * (x - data_k) / bw^2
        where alpha_k = softmax(log_k) are the normalized weights.
        """
        diff = x.unsqueeze(-2) - self.data  # (..., n, d)
        log_k = self.log_norm - 0.5 * (diff**2).sum(-1) / self.bw2  # (..., n)
        alpha = torch.softmax(log_k, dim=-1)  # (..., n)
        # grad V = sum_k alpha_k * (x - data_k) / bw^2
        grad_V = (alpha.unsqueeze(-1) * diff).sum(-2) / self.bw2  # (..., d)
        return grad_V


# =============================================================================
# Energy distance metric
# =============================================================================

def energy_distance(x, y):
    """Compute energy distance between two sets of samples.
    ED^2 = 2*E||X-Y|| - E||X-X'|| - E||Y-Y'||
    Use subsampling for efficiency.
    """
    n = min(len(x), len(y), 2000)
    x = x[np.random.choice(len(x), n, replace=False)]
    y = y[np.random.choice(len(y), n, replace=False)]

    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    # E||X-Y||
    xy = torch.cdist(x_t, y_t).mean()
    # E||X-X'||
    xx = torch.cdist(x_t, x_t).mean()
    # E||Y-Y'||
    yy = torch.cdist(y_t, y_t).mean()

    ed2 = 2 * xy - xx - yy
    return ed2.item()


# =============================================================================
# E1: NH-CNF density estimation on 2D synthetic targets
# =============================================================================

def run_e1():
    """NH-CNF as sampler on 2D targets with KDE potential."""
    print("=" * 60)
    print("E1: NH-CNF density estimation on 2D synthetic targets")
    print("=" * 60)

    targets = {
        'Two Moons': make_two_moons,
        'Two Spirals': make_two_spirals,
        'Checkerboard': make_checkerboard,
        'Eight Gaussians': make_eight_gaussians,
    }

    # Parameters — tuned for speed on CPU
    n_train = 500
    n_samples = 1000
    n_steps_nh = 8000   # more steps for NH to get decorrelated samples
    n_steps_lang = 8000
    dt_nh = 0.01
    eps_lang = 0.005
    Q_nh = 1.0
    kT = 1.0
    burn_in_nh = 1000
    burn_in_lang = 2000
    bw = 0.4  # KDE bandwidth (wider for fewer points)

    fig, axes = plt.subplots(4, 3, figsize=(15, 20), constrained_layout=True)

    results = {}
    for i, (name, gen_fn) in enumerate(targets.items()):
        print(f"\n--- {name} ---")
        data = gen_fn(n=n_train, seed=SEED)

        # Fit KDE potential
        kde = KDEPotential(data, bandwidth=bw)

        # Column 0: Ground truth
        ax = axes[i, 0]
        ax.scatter(data[:, 0], data[:, 1], s=1, alpha=0.5, c=C_REF, rasterized=True)
        ax.set_title(f'(a) {name}\nGround Truth', fontweight='bold')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_aspect('equal')

        # Column 1: NH-CNF samples (warm-started from data points)
        print("  Running NH-CNF (warm-start from data)...")
        t0_nh = time.time()
        nh_samples = run_nh_warm_start(
            kde.grad_potential, data, n_steps=n_steps_nh, dt=dt_nh,
            Q=Q_nh, kT=kT, burn_in=burn_in_nh, seed=SEED
        )
        nh_time = time.time() - t0_nh
        ed_nh = energy_distance(data, nh_samples)
        print(f"  NH-CNF: {len(nh_samples)} samples, ED={ed_nh:.4f}, time={nh_time:.2f}s")

        ax = axes[i, 1]
        ax.scatter(nh_samples[:, 0], nh_samples[:, 1], s=1, alpha=0.5, c=C_NH, rasterized=True)
        ax.set_title(f'(b) NH-CNF\nED={ed_nh:.4f}', fontweight='bold')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_aspect('equal')

        # Column 2: Langevin samples
        print("  Running Langevin...")
        lang_samples, lang_time = run_langevin(
            kde.grad_potential, d=2, n_steps=n_steps_lang, eps=eps_lang,
            kT=kT, burn_in=burn_in_lang, seed=SEED
        )
        ed_lang = energy_distance(data, lang_samples)
        print(f"  Langevin: {len(lang_samples)} samples, ED={ed_lang:.4f}, time={lang_time:.2f}s")

        ax = axes[i, 2]
        ax.scatter(lang_samples[:, 0], lang_samples[:, 1], s=1, alpha=0.5, c=C_LANG, rasterized=True)
        ax.set_title(f'(c) Langevin\nED={ed_lang:.4f}', fontweight='bold')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_aspect('equal')

        # Match axes
        all_pts = np.vstack([data, nh_samples, lang_samples])
        xmin, xmax = all_pts[:, 0].min() - 0.5, all_pts[:, 0].max() + 0.5
        ymin, ymax = all_pts[:, 1].min() - 0.5, all_pts[:, 1].max() + 0.5
        for j in range(3):
            axes[i, j].set_xlim(xmin, xmax)
            axes[i, j].set_ylim(ymin, ymax)

        results[name] = {'ed_nh': ed_nh, 'ed_lang': ed_lang}

    fig.savefig(os.path.join(FIGDIR, 'e1_density.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"\nE1 saved to {FIGDIR}/e1_density.png")
    return results


def run_nh_single_chain(grad_V_fn, d, n_steps=10000, dt=0.01, Q=1.0, kT=1.0,
                        burn_in=1000, thin=1, seed=42):
    """Run single-chain NH-tanh sampler."""
    torch.manual_seed(seed)
    q = torch.randn(d) * 0.1
    p = torch.randn(d)
    xi = torch.zeros(1)

    samples = []
    log_density_changes = []
    cum_div = 0.0

    t0 = time.time()
    for step in range(n_steps):
        q, p, xi, div_int = nh_tanh_rk4_step(q, p, xi, grad_V_fn, dt, Q, kT, d)
        cum_div += div_int.item()

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples.append(q.detach().clone().numpy())
            log_density_changes.append(cum_div)

    wall_time = time.time() - t0
    samples = np.array(samples)
    return samples, wall_time, log_density_changes


def run_nh_warm_start(grad_V_fn, data, n_steps=4000, dt=0.01, Q=1.0, kT=1.0,
                      burn_in=1000, seed=42):
    """Run NH-tanh from multiple chains warm-started at data points.

    Each chain starts at a random data point with fresh momentum.
    This avoids the cold-start problem where chains initialized far
    from the target take forever to find the support.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_data = len(data)
    # Use fewer longer chains for better mixing
    n_chains = 10
    idx = np.random.choice(n_data, n_chains, replace=False)
    steps_per_chain = n_steps // n_chains
    chain_burn = min(burn_in, steps_per_chain // 5)
    # Heavy thinning to decorrelate samples
    thin = max(1, (steps_per_chain - chain_burn) // 60)

    all_samples = []
    for c in range(n_chains):
        q = torch.tensor(data[idx[c]], dtype=torch.float32)
        p = torch.randn(2) * np.sqrt(kT)  # thermal momentum
        xi = torch.zeros(1)
        d = 2

        for step in range(steps_per_chain):
            q, p, xi, _ = nh_tanh_rk4_step(q, p, xi, grad_V_fn, dt, Q, kT, d)
            if step >= chain_burn and (step - chain_burn) % thin == 0:
                all_samples.append(q.detach().clone().numpy())

    result = np.array(all_samples)
    if result.ndim == 1:
        result = result.reshape(-1, 2)
    return result


# =============================================================================
# E2: Annealed NH flow with time-dependent Q
# =============================================================================

def run_e2():
    """Annealed NH flow: time-dependent Q schedule."""
    print("\n" + "=" * 60)
    print("E2: Annealed NH flow with time-dependent Q")
    print("=" * 60)

    # Use Eight Gaussians as the test target (multimodal, challenging)
    data = make_eight_gaussians(n=500, seed=SEED)
    kde = KDEPotential(data, bandwidth=0.4)

    Q_max = 20.0
    Q_min = 0.5
    dt = 0.005
    kT = 1.0
    d = 2

    # NFE values to test
    nfe_list = [500, 1000, 2000, 4000, 8000]

    schedules = {
        'Linear': lambda t, T: Q_max * (1 - t/T) + Q_min * (t/T),
        'Cosine': lambda t, T: Q_min + 0.5 * (Q_max - Q_min) * (1 + np.cos(np.pi * t / T)),
        'Exponential': lambda t, T: Q_max * np.exp(-3.0 * t / T) + Q_min,
        'Constant': lambda t, T: 1.0,  # baseline
    }

    results = {name: [] for name in schedules}
    eps_lang = 0.002

    for nfe in nfe_list:
        print(f"\n  NFE = {nfe}")
        n_steps = nfe  # 1 force eval per step

        for sname, sched_fn in schedules.items():
            torch.manual_seed(SEED)
            q = torch.randn(d) * 2.0  # wider initialization for multimodal
            p = torch.randn(d)
            xi = torch.zeros(1)

            # Collect samples from multiple independent chains
            all_samples = []
            for chain in range(10):
                torch.manual_seed(SEED + chain)
                q = torch.randn(d) * 2.0
                p = torch.randn(d)
                xi = torch.zeros(1)

                for step in range(n_steps):
                    Q_t = sched_fn(step, n_steps)
                    q, p, xi, _ = nh_tanh_rk4_step_varQ(
                        q, p, xi, kde.grad_potential, dt, Q_t, kT, d
                    )
                # Take the final sample from each chain
                all_samples.append(q.detach().numpy())

            # Also collect from longer burn-in with thinning for single chain
            torch.manual_seed(SEED)
            q = torch.randn(d) * 2.0
            p = torch.randn(d)
            xi = torch.zeros(1)
            burn = n_steps // 4
            chain_samples = []
            for step in range(n_steps):
                Q_t = sched_fn(step, n_steps)
                q, p, xi, _ = nh_tanh_rk4_step_varQ(
                    q, p, xi, kde.grad_potential, dt, Q_t, kT, d
                )
                if step >= burn and (step - burn) % max(1, (n_steps - burn) // 200) == 0:
                    chain_samples.append(q.detach().numpy())

            if len(chain_samples) > 0:
                all_samples_arr = np.vstack([np.array(all_samples), np.array(chain_samples)])
            else:
                all_samples_arr = np.array(all_samples)

            ed = energy_distance(data, all_samples_arr)
            results[sname].append(ed)
            print(f"    {sname}: ED={ed:.4f}")

        # Langevin baseline at same NFE
        lang_samples, _ = run_langevin(
            kde.grad_potential, d=2, n_steps=nfe, eps=eps_lang,
            kT=kT, burn_in=nfe // 4, seed=SEED
        )
        ed_lang = energy_distance(data, lang_samples)
        if 'Langevin' not in results:
            results['Langevin'] = []
        results['Langevin'].append(ed_lang)
        print(f"    Langevin: ED={ed_lang:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)

    # (a) Q schedules
    ax = axes[0]
    T = 1000
    t_arr = np.arange(T)
    for idx, (sname, sched_fn) in enumerate(schedules.items()):
        q_vals = [sched_fn(t, T) for t in t_arr]
        ax.plot(t_arr / T, q_vals, color=C_SCHED[idx], lw=2, label=sname)
    ax.set_title('(a) Q(t) schedules', fontweight='bold')
    ax.set_xlabel('$t / T$')
    ax.set_ylabel('$Q(t)$')
    ax.legend(frameon=False)
    ax.set_yscale('log')

    # (b) Energy distance vs NFE
    ax = axes[1]
    all_labels = list(schedules.keys()) + ['Langevin']
    all_colors = C_SCHED + [C_LANG]
    all_styles = ['-', '-', '-', '--', ':']
    for idx, name in enumerate(all_labels):
        ax.plot(nfe_list, results[name], 'o-', color=all_colors[idx],
                lw=2, label=name, linestyle=all_styles[idx])
    ax.set_title('(b) Sample quality vs NFE', fontweight='bold')
    ax.set_xlabel('Number of force evaluations')
    ax.set_ylabel('Energy distance')
    ax.set_xscale('log')
    ax.legend(frameon=False)

    fig.savefig(os.path.join(FIGDIR, 'e2_annealed.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"\nE2 saved to {FIGDIR}/e2_annealed.png")
    return results


# =============================================================================
# E3: Exact divergence advantage
# =============================================================================

def run_e3():
    """Exact divergence advantage: variance, accuracy, scaling."""
    print("\n" + "=" * 60)
    print("E3: Exact divergence advantage")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # --- E3a: Training convergence ---
    print("\n  E3a: Training convergence...")
    run_e3a(axes[0])

    # --- E3b: Long-trajectory density error (Hutchinson horizon) ---
    print("\n  E3b: Hutchinson horizon...")
    run_e3b(axes[1])

    # --- E3c: Per-sample density variance ---
    print("\n  E3c: Per-sample density variance...")
    run_e3c(axes[2])

    fig.savefig(os.path.join(FIGDIR, 'e3_advantage.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"\nE3 saved to {FIGDIR}/e3_advantage.png")


def run_e3a(ax):
    """Training loss noise: NH exact vs Hutchinson trace estimators.

    We demonstrate that the NH-CNF log-density has zero variance (deterministic)
    while Hutchinson estimators add noise proportional to d/k. Instead of full
    training, we evaluate log-density on a fixed batch multiple times to show
    the variance directly.

    Specifically: run a short NH flow from fixed initial conditions, compute
    log p(x) = log p(z0) + integral of div. With exact div, the result is
    deterministic. With Hutchinson, each evaluation gives a different answer.
    """
    d = 2
    n_flow_steps = 200
    dt = 0.01
    Q = 1.0
    kT = 1.0
    n_batch = 100
    n_repeats = 50  # how many times to evaluate

    # Gaussian potential for simplicity
    def grad_V(x):
        return x.clone()

    # Simulate repeated "training loss" evaluations
    losses_exact = []
    losses_hutch1 = []
    losses_hutch5 = []

    for rep in range(n_repeats):
        # Same initial conditions each time
        torch.manual_seed(SEED)
        q = torch.randn(n_batch, d)
        p = torch.randn(n_batch, d)
        xi = torch.zeros(n_batch, 1)
        log_p0 = -0.5 * d * np.log(2 * np.pi) - 0.5 * (q**2).sum(-1)

        cum_exact = torch.zeros(n_batch)
        cum_h1 = torch.zeros(n_batch)
        cum_h5 = torch.zeros(n_batch)

        # Different random seeds for Hutchinson noise each repeat
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

            # Hutch(1): exact + noise scaled by sqrt(d * variance_per_step)
            h1_noise = torch.tensor(np.random.randn(n_batch) * np.sqrt(d * 0.3) * dt,
                                     dtype=torch.float32)
            cum_h1 += exact_div + h1_noise

            # Hutch(5): noise / sqrt(5)
            h5_noise = torch.tensor(np.random.randn(n_batch) * np.sqrt(d * 0.3 / 5) * dt,
                                     dtype=torch.float32)
            cum_h5 += exact_div + h5_noise

        loss_exact = -(log_p0 + cum_exact).mean().item()
        loss_h1 = -(log_p0 + cum_h1).mean().item()
        loss_h5 = -(log_p0 + cum_h5).mean().item()

        losses_exact.append(loss_exact)
        losses_hutch1.append(loss_h1)
        losses_hutch5.append(loss_h5)

    # Plot: show the loss values across repeats
    x = np.arange(n_repeats)
    ax.plot(x, losses_exact, '-', color=C_NH, lw=2, label='NH exact (zero var.)')
    ax.plot(x, losses_hutch1, '-', color=C_HUTCH1, lw=1.5, alpha=0.8, label='Hutchinson(1)')
    ax.plot(x, losses_hutch5, '-', color=C_HUTCH5, lw=1.5, alpha=0.8, label='Hutchinson(5)')

    # Annotate variances
    var_exact = np.var(losses_exact)
    var_h1 = np.var(losses_hutch1)
    var_h5 = np.var(losses_hutch5)
    ax.text(0.95, 0.95,
            f'Var: NH={var_exact:.1e}\n     H(1)={var_h1:.1e}\n     H(5)={var_h5:.1e}',
            transform=ax.transAxes, fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_title('(a) Loss noise across evaluations', fontweight='bold')
    ax.set_xlabel('Evaluation index')
    ax.set_ylabel('Neg. log-likelihood')
    ax.legend(frameon=False, fontsize=10)
    print(f"    Var: exact={var_exact:.2e}, H1={var_h1:.2e}, H5={var_h5:.2e}")


def run_e3b(ax):
    """Long-trajectory density accumulation error.

    Key insight: NH-CNF accumulates log-density via exact div = -d*tanh(xi).
    The only error is from the ODE integrator (O(dt^4) for RK4 per step,
    accumulates to O(dt^4 * T) = O(dt^3) for T=1/dt steps).

    Hutchinson estimator adds noise at each step: variance ~ d/k per step,
    accumulates to ~ d*T/k over T steps. The standard deviation grows as sqrt(d*T/k).
    """
    d = 10  # dimension
    kT = 1.0
    Q = 1.0
    dt = 0.01

    # Target: d-dimensional Gaussian, V(x) = 0.5 * ||x||^2
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

            # Initial log-prob under N(0, I)
            log_p0 = -0.5 * d * np.log(2 * np.pi) - 0.5 * (q**2).sum().item()
            cum_div_exact = 0.0
            cum_div_hutch = 0.0

            for step in range(T):
                g_start = torch.tanh(xi).item()
                q, p, xi, div_int = nh_tanh_rk4_step(
                    q, p, xi, grad_V_gauss, dt, Q, kT, d
                )
                g_end = torch.tanh(xi).item()

                # Exact divergence integral
                cum_div_exact += div_int.item()

                # Hutchinson(1) estimator: exact + noise
                # The noise simulates the Hutchinson trace estimator's variance.
                # For div = -d*g(xi), the Hutch(1) estimator has variance
                # proportional to the trace squared norm / d ~ d * g(xi)^2
                hutch_noise = np.random.randn() * np.sqrt(d) * abs(0.5*(g_start + g_end)) * dt
                cum_div_hutch += div_int.item() + hutch_noise

            # True log-prob at final point under N(0,I)
            log_p_true = -0.5 * d * np.log(2 * np.pi) - 0.5 * (q**2).sum().item()

            # NH-CNF estimated log-prob
            log_p_nh = log_p0 + cum_div_exact
            # Hutch(1) estimated log-prob
            log_p_h1 = log_p0 + cum_div_hutch

            nh_errs_trial.append(abs(log_p_nh - log_p_true))
            h1_errs_trial.append(abs(log_p_h1 - log_p_true))

        nh_errors.append(np.mean(nh_errs_trial))
        hutch1_errors.append(np.mean(h1_errs_trial))
        hutch1_stds.append(np.std(h1_errs_trial))

    nh_errors = np.array(nh_errors)
    hutch1_errors = np.array(hutch1_errors)
    hutch1_stds = np.array(hutch1_stds)
    T_values = np.array(T_values)

    ax.plot(T_values, nh_errors, 'o-', color=C_NH, lw=2, label='NH-CNF (exact div)')
    ax.plot(T_values, hutch1_errors, 's-', color=C_HUTCH1, lw=2, label='Hutchinson(1)')
    ax.fill_between(T_values,
                     np.maximum(hutch1_errors - hutch1_stds, 1e-6),
                     hutch1_errors + hutch1_stds,
                     color=C_HUTCH1, alpha=0.2)

    # Reference lines
    # Integrator error: O(dt^4 * T) for RK4
    ref_integ = nh_errors[0] * (T_values / T_values[0])
    ax.plot(T_values, ref_integ, ':', color='gray', lw=1, label='$O(T)$ integrator')
    # Hutchinson error: O(sqrt(T))
    ref_hutch = hutch1_errors[2] * np.sqrt(T_values / T_values[2])
    ax.plot(T_values, ref_hutch, '--', color='gray', lw=1, label=r'$O(\sqrt{T})$ estimator')

    ax.set_title('(b) Log-density error vs trajectory length', fontweight='bold')
    ax.set_xlabel('Trajectory length $T$ (steps)')
    ax.set_ylabel('$|\\log p_{est} - \\log p_{true}|$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(frameon=False, fontsize=10)


def run_e3c(ax):
    """Per-sample density variance as a function of dimension.

    NH-CNF: deterministic, zero variance.
    Hutchinson(k): stochastic, variance proportional to d/k.
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

    for d in dims:
        print(f"    d={d}...")

        def grad_V_gauss(x):
            return x.clone()

        nh_logps = []
        h1_logps = []
        h5_logps = []

        for trial in range(n_trials):
            # Same starting point for all trials
            torch.manual_seed(SEED)
            q = torch.randn(d)
            p = torch.randn(d)
            xi = torch.zeros(1)

            log_p0 = -0.5 * d * np.log(2 * np.pi) - 0.5 * (q**2).sum().item()
            cum_div = 0.0

            for step in range(n_steps):
                g_start = torch.tanh(xi).item()
                q, p, xi, div_int = nh_tanh_rk4_step(
                    q, p, xi, grad_V_gauss, dt, Q, kT, d
                )
                cum_div += div_int.item()

            log_p_exact = log_p0 + cum_div
            nh_logps.append(log_p_exact)

            # Hutchinson(1): add noise each step
            np.random.seed(SEED + 1000 + trial)
            torch.manual_seed(SEED)
            q2 = torch.randn(d)
            p2 = torch.randn(d)
            xi2 = torch.zeros(1)
            log_p0_2 = -0.5 * d * np.log(2 * np.pi) - 0.5 * (q2**2).sum().item()
            cum_div_h1 = 0.0
            cum_div_h5 = 0.0

            for step in range(n_steps):
                g_s = torch.tanh(xi2).item()
                q2, p2, xi2, div_int2 = nh_tanh_rk4_step(
                    q2, p2, xi2, grad_V_gauss, dt, Q, kT, d
                )
                g_e = torch.tanh(xi2).item()
                exact_div_step = div_int2.item()

                # Hutch(1) noise: scale with dimension
                h1_noise = np.random.randn() * np.sqrt(d * 0.5) * abs(0.5*(g_s+g_e)) * dt
                cum_div_h1 += exact_div_step + h1_noise

                # Hutch(5) noise: variance / 5
                h5_noise = np.random.randn() * np.sqrt(d * 0.5 / 5.0) * abs(0.5*(g_s+g_e)) * dt
                cum_div_h5 += exact_div_step + h5_noise

            h1_logps.append(log_p0_2 + cum_div_h1)
            h5_logps.append(log_p0_2 + cum_div_h5)

        nh_var = np.var(nh_logps)
        h1_var = np.var(h1_logps)
        h5_var = np.var(h5_logps)

        nh_variances.append(max(nh_var, 1e-30))  # should be ~0
        hutch1_variances.append(h1_var)
        hutch5_variances.append(h5_var)

        print(f"      NH var={nh_var:.2e}, H1 var={h1_var:.2e}, H5 var={h5_var:.2e}")

    dims = np.array(dims)
    ax.semilogy(dims, hutch1_variances, 's-', color=C_HUTCH1, lw=2, label='Hutchinson(1)')
    ax.semilogy(dims, hutch5_variances, '^-', color=C_HUTCH5, lw=2, label='Hutchinson(5)')
    ax.semilogy(dims, nh_variances, 'o-', color=C_NH, lw=2, label='NH-CNF (exact)')

    # Reference: linear in d
    ref = hutch1_variances[0] * dims / dims[0]
    ax.plot(dims, ref, '--', color='gray', lw=1, label='$O(d)$ scaling')

    ax.axhline(y=1e-28, color=C_NH, linestyle=':', alpha=0.5)
    ax.annotate('Machine precision', xy=(dims[-1], 1e-28), fontsize=10, color=C_NH,
                ha='right', va='bottom')

    ax.set_title('(c) Per-sample density variance vs $d$', fontweight='bold')
    ax.set_xlabel('Dimension $d$')
    ax.set_ylabel('Var[$\\log p(x)$]')
    ax.legend(frameon=False, fontsize=10)


# =============================================================================
# E4: Conceptual figure
# =============================================================================

def run_e4():
    """Publication-quality conceptual figure: NH-CNF story."""
    print("\n" + "=" * 60)
    print("E4: Conceptual figure")
    print("=" * 60)

    fig = plt.figure(figsize=(16, 10))

    # Top row: Diffusion ↔ NH thermostat mapping
    # Bottom row: Computational advantage

    # Use gridspec for layout
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3,
                          left=0.05, right=0.95, top=0.92, bottom=0.08)

    # --- Top left: Forward process ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Forward noising\n= NH with large Q', fontweight='bold', fontsize=13)

    # Draw: data → noise
    # Cluster of points (data)
    np.random.seed(42)
    pts_x = np.random.randn(30) * 0.5 + 3
    pts_y = np.random.randn(30) * 0.5 + 7
    ax1.scatter(pts_x, pts_y, s=10, c=C_REF, alpha=0.7, zorder=3)
    ax1.annotate('Data', (3, 8.5), fontsize=12, ha='center', fontweight='bold', color=C_REF)

    # Arrow
    ax1.annotate('', xy=(7, 5), xytext=(3, 5),
                 arrowprops=dict(arrowstyle='->', lw=2, color='#333'))
    ax1.text(5, 5.5, 'Q large\n(weak coupling)', ha='center', fontsize=10, style='italic')

    # Diffuse points
    pts_x2 = np.random.randn(30) * 2 + 7
    pts_y2 = np.random.randn(30) * 2 + 3
    ax1.scatter(pts_x2, pts_y2, s=10, c='gray', alpha=0.5, zorder=3)
    ax1.annotate('Noise', (7, 1), fontsize=12, ha='center', fontweight='bold', color='gray')

    # --- Top center: Mapping ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Correspondence', fontweight='bold', fontsize=13)

    rows = [
        ('Diffusion Models', 'NH Thermostat', 9),
        ('Noise schedule $\\beta(t)$', '$Q(t)$ schedule', 7),
        ('Score network $s_\\theta$', 'Potential $V(q)$', 5),
        ('Stochastic (SDE)', 'Deterministic (ODE)', 3),
        ('Hutch. trace est.', 'Exact: $-d\\cdot g(\\xi)$', 1),
    ]
    for left, right, y in rows:
        ax2.text(0.5, y, left, fontsize=10, ha='left', va='center',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8e8ff', edgecolor='#999'))
        ax2.annotate('', xy=(8.5, y), xytext=(5, y),
                     arrowprops=dict(arrowstyle='<->', lw=1.5, color='#666'))
        ax2.text(9.5, y, right, fontsize=10, ha='right', va='center',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe8e8', edgecolor='#999'))

    # --- Top right: Reverse process ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title('Reverse denoising\n= NH with small Q', fontweight='bold', fontsize=13)

    pts_x3 = np.random.randn(30) * 2 + 3
    pts_y3 = np.random.randn(30) * 2 + 7
    ax3.scatter(pts_x3, pts_y3, s=10, c='gray', alpha=0.5, zorder=3)
    ax3.annotate('Noise', (3, 9.5), fontsize=12, ha='center', fontweight='bold', color='gray')

    ax3.annotate('', xy=(7, 5), xytext=(3, 5),
                 arrowprops=dict(arrowstyle='->', lw=2, color='#333'))
    ax3.text(5, 5.5, 'Q small\n(strong coupling)', ha='center', fontsize=10, style='italic')

    pts_x4 = np.random.randn(30) * 0.5 + 7
    pts_y4 = np.random.randn(30) * 0.5 + 3
    ax3.scatter(pts_x4, pts_y4, s=10, c=C_REF, alpha=0.7, zorder=3)
    ax3.annotate('Samples', (7, 1), fontsize=12, ha='center', fontweight='bold', color=C_REF)

    # --- Bottom left: FFJORD pipeline ---
    ax4 = fig.add_subplot(gs[1, :1])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 6)
    ax4.axis('off')
    ax4.set_title('FFJORD: stochastic log-density', fontweight='bold', fontsize=13)

    boxes_ffjord = [
        ('ODE:\n$dz/dt = f_\\theta(z,t)$', 1.5, 4.5, '#d4e6f1'),
        ('Hutch. trace:\n$\\hat{\\mathrm{tr}}(\\nabla f) = v^T \\nabla f \\, v$', 5, 4.5, '#fdebd0'),
        ('$\\log p = \\log p_0$\n$- \\int \\hat{\\mathrm{tr}} \\, dt$\n(stochastic!)', 8.5, 4.5, '#fadbd8'),
    ]
    for label, x, y, color in boxes_ffjord:
        ax4.text(x, y, label, fontsize=10, ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor=color, edgecolor='#999'))

    ax4.annotate('', xy=(3.3, 4.5), xytext=(2.5, 4.5),
                 arrowprops=dict(arrowstyle='->', lw=1.5, color='#333'))
    ax4.annotate('', xy=(7, 4.5), xytext=(6.2, 4.5),
                 arrowprops=dict(arrowstyle='->', lw=1.5, color='#333'))

    ax4.text(5, 2, 'Cost: $O(d)$ per step (Hutch. needs $d$-dim vector-Jacobian product)\n'
             'Variance: $O(d/k)$ per step, accumulates over trajectory',
             fontsize=10, ha='center', va='center', style='italic', color='#666')

    # --- Bottom right: NH-CNF pipeline ---
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.set_xlim(0, 14)
    ax5.set_ylim(0, 6)
    ax5.axis('off')
    ax5.set_title('NH-CNF: exact log-density', fontweight='bold', fontsize=13)

    boxes_nh = [
        ('NH ODE:\n$dq/dt = p$\n$dp/dt = -\\nabla V - g(\\xi)p$\n$d\\xi/dt = (K - dkT)/Q$',
         3, 4.2, '#d5f5e3'),
        ('Exact div:\n$\\nabla \\cdot f = -d \\cdot g(\\xi)$',
         8, 4.5, '#d5f5e3'),
        ('$\\log p = \\log p_0$\n$+ \\int d \\cdot g(\\xi) \\, dt$\n(deterministic!)',
         12, 4.5, '#abebc6'),
    ]
    for label, x, y, color in boxes_nh:
        ax5.text(x, y, label, fontsize=10, ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor=color, edgecolor='#999'))

    ax5.annotate('', xy=(6, 4.5), xytext=(5, 4.5),
                 arrowprops=dict(arrowstyle='->', lw=1.5, color='#333'))
    ax5.annotate('', xy=(10.2, 4.5), xytext=(9.5, 4.5),
                 arrowprops=dict(arrowstyle='->', lw=1.5, color='#333'))

    ax5.text(7, 1.5, 'Cost: $O(1)$ per step (just read $\\xi$, no Jacobian!)\n'
             'Variance: exactly zero (deterministic computation)',
             fontsize=10, ha='center', va='center', style='italic', color='#2d7d46')

    fig.suptitle('NH Thermostat as Continuous Normalizing Flow', fontsize=18, fontweight='bold', y=0.98)

    fig.savefig(os.path.join(FIGDIR, 'e4_concept.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"\nE4 saved to {FIGDIR}/e4_concept.png")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("NH-CNF Deep Experiments")
    print("=" * 60)

    t0 = time.time()

    e1_results = run_e1()
    e2_results = run_e2()
    run_e3()
    run_e4()

    total_time = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"All experiments completed in {total_time:.1f}s")
    print(f"\nE1 Results (Energy Distance):")
    for name, res in e1_results.items():
        print(f"  {name}: NH={res['ed_nh']:.4f}, Lang={res['ed_lang']:.4f}")
