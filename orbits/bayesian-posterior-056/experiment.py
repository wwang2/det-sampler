"""
NH thermostat as CNF for Bayesian posterior sampling.

Three experiments:
  E1: 2D banana distribution
  E2: BNN posterior (1D sinusoidal regression)
  E3: 10D Gaussian mixture divergence speedup benchmark
"""

import torch
import numpy as np
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
FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# Style colors from style.md
C_NH = '#1f77b4'
C_SGLD = '#2ca02c'
C_REF = '#d62728'
C_ENSEMBLE = '#9467bd'


# =============================================================================
# Potentials
# =============================================================================

def banana_potential(x):
    """V(x) = 0.5*(x1^2 - x2)^2 + 0.5*(x1 - 1)^2"""
    return 0.5 * (x[..., 0]**2 - x[..., 1])**2 + 0.5 * (x[..., 0] - 1.0)**2


def banana_grad(x):
    """Gradient of banana potential."""
    x1, x2 = x[..., 0], x[..., 1]
    t = x1**2 - x2
    dv_dx1 = t * 2.0 * x1 + (x1 - 1.0)
    dv_dx2 = -t
    return torch.stack([dv_dx1, dv_dx2], dim=-1)


def gmm_potential_and_grad(x, means, weights):
    """Gaussian mixture model: V(x) = -log sum_k w_k N(x; mu_k, I).
    Returns (V, grad_V).
    """
    # x: (..., d), means: (K, d), weights: (K,)
    d = x.shape[-1]
    diff = x.unsqueeze(-2) - means  # (..., K, d)
    log_probs = -0.5 * (diff**2).sum(-1) - 0.5 * d * np.log(2 * np.pi)  # (..., K)
    log_probs = log_probs + torch.log(weights)
    log_sum = torch.logsumexp(log_probs, dim=-1)  # (...)
    V = -log_sum
    # Gradient: -sum_k w_k * N_k * (-(x - mu_k)) / sum_k w_k * N_k
    # = sum_k alpha_k * (x - mu_k) where alpha_k = softmax(log_probs)
    alpha = torch.softmax(log_probs, dim=-1)  # (..., K)
    grad_V = (alpha.unsqueeze(-1) * diff).sum(-2)  # (..., d)
    return V, grad_V


# =============================================================================
# NH-tanh sampler (RK4)
# =============================================================================

def nh_tanh_rk4_step(q, p, xi, grad_V_fn, dt, Q=1.0, kT=1.0, d=None):
    """One RK4 step of the NH-tanh ODE.
    dq/dt = p
    dp/dt = -grad_V(q) - tanh(xi)*p
    dxi/dt = (1/Q)*(p^2/1 - d*kT)  [m=1]

    Returns (q_new, p_new, xi_new, div_integral).
    div_integral = -d * tanh(xi) * dt  (midpoint approx for the integral)
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

    # Analytical divergence: div = -d * tanh(xi), integrate with Simpson-like approx
    # Using midpoint (average of start and end xi for trapezoidal)
    g_start = torch.tanh(xi)
    g_end = torch.tanh(xi_new)
    div_integral = -d * 0.5 * (g_start + g_end) * dt  # per component, sum over xi dims

    return q_new, p_new, xi_new, div_integral.squeeze(-1)


def run_nh_tanh(grad_V_fn, d, n_steps=10000, dt=0.01, Q=1.0, kT=1.0,
                burn_in=1000, thin=1, seed=42):
    """Run NH-tanh sampler, return position samples and timing."""
    torch.manual_seed(seed)
    np.random.seed(seed)

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


# =============================================================================
# NH-tanh with multi-scale Q (parallel thermostats)
# =============================================================================

def run_nh_multiscale(grad_V_fn, d, n_steps=10000, dt=0.01,
                      Qs=[0.1, 1.0, 10.0], kT=1.0, burn_in=1000, thin=1, seed=42):
    """NH-tanh with N parallel thermostats at different Q values."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    N_therm = len(Qs)
    q = torch.randn(d) * 0.1
    p = torch.randn(d)
    xis = torch.zeros(N_therm)
    Qs_t = torch.tensor(Qs, dtype=torch.float32)

    samples = []
    t0 = time.time()

    for step in range(n_steps):
        # Compute total friction: Gamma = sum_k tanh(xi_k)
        gv = grad_V_fn(q)
        gs = torch.tanh(xis)
        total_g = gs.sum()

        # Simple RK4 for the coupled system
        def f_full(q_, p_, xis_):
            gv_ = grad_V_fn(q_)
            gs_ = torch.tanh(xis_)
            total_g_ = gs_.sum()
            dq = p_
            dp = -gv_ - total_g_ * p_
            # Each thermostat driven by same kinetic energy signal
            KE = (p_**2).sum()
            dxis = (1.0 / Qs_t) * (KE - d * kT)
            return dq, dp, dxis

        k1q, k1p, k1x = f_full(q, p, xis)
        k2q, k2p, k2x = f_full(q + 0.5*dt*k1q, p + 0.5*dt*k1p, xis + 0.5*dt*k1x)
        k3q, k3p, k3x = f_full(q + 0.5*dt*k2q, p + 0.5*dt*k2p, xis + 0.5*dt*k2x)
        k4q, k4p, k4x = f_full(q + dt*k3q, p + dt*k3p, xis + dt*k3x)

        q = q + (dt/6.0) * (k1q + 2*k2q + 2*k3q + k4q)
        p = p + (dt/6.0) * (k1p + 2*k2p + 2*k3p + k4p)
        xis = xis + (dt/6.0) * (k1x + 2*k2x + 2*k3x + k4x)

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples.append(q.detach().clone().numpy())

    wall_time = time.time() - t0
    return np.array(samples), wall_time


# =============================================================================
# SGLD sampler
# =============================================================================

def run_sgld(grad_V_fn, d, n_steps=100000, eps=0.001, burn_in=10000, thin=1, seed=42):
    """Stochastic Gradient Langevin Dynamics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    theta = torch.randn(d) * 0.1
    samples = []
    t0 = time.time()

    for step in range(n_steps):
        gv = grad_V_fn(theta)
        noise = torch.randn_like(theta) * np.sqrt(2 * eps)
        theta = theta - eps * gv + noise

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples.append(theta.detach().clone().numpy())

    wall_time = time.time() - t0
    return np.array(samples), wall_time


# =============================================================================
# E1: 2D Banana Distribution
# =============================================================================

def compute_kl_2d(samples, log_prob_fn, xlim=(-3, 5), ylim=(-2, 12), nbins=50):
    """Histogram-based KL divergence for 2D samples."""
    # Build histogram of samples
    H_samp, xedges, yedges = np.histogram2d(
        samples[:, 0], samples[:, 1], bins=nbins,
        range=[xlim, ylim], density=True
    )

    # Evaluate true density on grid centers
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xc, yc, indexing='ij')
    grid = np.stack([X, Y], axis=-1)
    log_p = log_prob_fn(grid)
    p_true = np.exp(log_p)
    p_true = p_true / (p_true.sum() * (xedges[1]-xedges[0]) * (yedges[1]-yedges[0]))

    # KL(samples || true) = sum p_samp * log(p_samp / p_true)
    mask = (H_samp > 0) & (p_true > 0)
    kl = np.sum(H_samp[mask] * np.log(H_samp[mask] / p_true[mask])) * \
         (xedges[1]-xedges[0]) * (yedges[1]-yedges[0])
    return kl


def banana_log_prob(x):
    """Log probability of banana distribution (unnormalized is fine for density plots)."""
    return -0.5 * (x[..., 0]**2 - x[..., 1])**2 - 0.5 * (x[..., 0] - 1.0)**2


def run_e1():
    """E1: 2D banana distribution comparison."""
    print("=" * 60)
    print("E1: 2D Banana Distribution")
    print("=" * 60)

    d = 2

    # NH-tanh with multi-scale Q for better mixing
    # Q=[0.01, 0.1, 1.0] found by parameter sweep to give best mixing
    print("Running NH-tanh (multi-scale Q)...")
    nh_samples, nh_time = run_nh_multiscale(
        banana_grad, d, n_steps=500000, dt=0.005, Qs=[0.01, 0.1, 1.0],
        kT=1.0, burn_in=50000, thin=4, seed=SEED
    )
    print(f"  NH-tanh: {len(nh_samples)} samples in {nh_time:.2f}s")

    # SGLD sampler
    print("Running SGLD...")
    sgld_samples, sgld_time = run_sgld(
        banana_grad, d, n_steps=200000, eps=0.002,
        burn_in=20000, thin=18, seed=SEED
    )
    print(f"  SGLD: {len(sgld_samples)} samples in {sgld_time:.2f}s")

    # Compute KL
    kl_nh = compute_kl_2d(nh_samples, banana_log_prob)
    kl_sgld = compute_kl_2d(sgld_samples, banana_log_prob)
    print(f"  KL(NH-tanh || true) = {kl_nh:.4f}")
    print(f"  KL(SGLD || true) = {kl_sgld:.4f}")

    # --- KL convergence curves ---
    print("Computing KL convergence curves...")
    n_nh = 500000
    n_sgld = 200000
    checkpoints_nh = np.logspace(2.5, np.log10(n_nh), 15).astype(int)
    checkpoints_sgld = np.logspace(2.5, np.log10(n_sgld), 15).astype(int)

    nh_all, _ = run_nh_multiscale(
        banana_grad, d, n_steps=n_nh, dt=0.005, Qs=[0.01, 0.1, 1.0],
        kT=1.0, burn_in=0, thin=1, seed=SEED
    )
    sgld_all, _ = run_sgld(
        banana_grad, d, n_steps=n_sgld, eps=0.002, burn_in=0, thin=1, seed=SEED
    )

    kl_nh_curve, fe_nh_curve = [], []
    for cp in checkpoints_nh:
        if cp > 200:
            kl_nh_curve.append(compute_kl_2d(nh_all[:cp], banana_log_prob))
            fe_nh_curve.append(cp * 4)  # 4 force evals per RK4 step (approx for Euler too)

    kl_sgld_curve, fe_sgld_curve = [], []
    for cp in checkpoints_sgld:
        if cp > 200:
            kl_sgld_curve.append(compute_kl_2d(sgld_all[:cp], banana_log_prob))
            fe_sgld_curve.append(cp)  # 1 force eval per SGLD step

    # --- Plot ---
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), constrained_layout=True)

    # (a) Ground truth density
    xg = np.linspace(-3, 5, 200)
    yg = np.linspace(-2, 12, 200)
    X, Y = np.meshgrid(xg, yg)
    grid = np.stack([X, Y], axis=-1)
    Z = np.exp(banana_log_prob(grid))
    axes[0].contourf(X, Y, Z, levels=30, cmap='viridis')
    axes[0].set_title('(a) Ground truth density', fontweight='bold')
    axes[0].set_xlabel('$\\theta_1$')
    axes[0].set_ylabel('$\\theta_2$')

    # (b) NH-tanh samples
    axes[1].scatter(nh_samples[:, 0], nh_samples[:, 1], s=1, alpha=0.3, c=C_NH)
    axes[1].contour(X, Y, Z, levels=8, colors='gray', alpha=0.5, linewidths=0.5)
    axes[1].set_title(f'(b) NH-tanh multi-Q (KL={kl_nh:.3f})', fontweight='bold')
    axes[1].set_xlabel('$\\theta_1$')
    axes[1].set_ylabel('$\\theta_2$')
    axes[1].set_xlim(-3, 5)
    axes[1].set_ylim(-2, 12)

    # (c) SGLD samples
    axes[2].scatter(sgld_samples[:, 0], sgld_samples[:, 1], s=1, alpha=0.3, c=C_SGLD)
    axes[2].contour(X, Y, Z, levels=8, colors='gray', alpha=0.5, linewidths=0.5)
    axes[2].set_title(f'(c) SGLD (KL={kl_sgld:.3f})', fontweight='bold')
    axes[2].set_xlabel('$\\theta_1$')
    axes[2].set_ylabel('$\\theta_2$')
    axes[2].set_xlim(-3, 5)
    axes[2].set_ylim(-2, 12)

    # (d) KL convergence
    axes[3].plot(fe_nh_curve, kl_nh_curve, 'o-', color=C_NH,
                 label='NH-tanh multi-Q', markersize=3)
    axes[3].plot(fe_sgld_curve, kl_sgld_curve, 's-', color=C_SGLD,
                 label='SGLD', markersize=3)
    axes[3].axhline(0.01, color='gray', ls='--', label='KL=0.01 threshold')
    axes[3].set_xscale('log')
    axes[3].set_yscale('log')
    axes[3].set_title('(d) KL convergence', fontweight='bold')
    axes[3].set_xlabel('Force evaluations')
    axes[3].set_ylabel('KL divergence')
    axes[3].legend(frameon=False)

    fig.savefig(os.path.join(FIGDIR, 'e1_banana.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved e1_banana.png")

    return kl_nh, kl_sgld


# =============================================================================
# E2: BNN posterior — 1D sinusoidal regression
# =============================================================================

class TinyBNN(torch.nn.Module):
    """2-layer MLP: input(1) -> hidden(20) -> hidden(20) -> output(1)."""
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 20)
        self.fc2 = torch.nn.Linear(20, 20)
        self.fc3 = torch.nn.Linear(20, 1)

    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        return self.fc3(h)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def set_flat_params(self, flat):
        idx = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(flat[idx:idx+n].view_as(p))
            idx += n

    def get_flat_params(self):
        return torch.cat([p.data.flatten() for p in self.parameters()])


def bnn_log_posterior(model, x_data, y_data, sigma_noise=0.1, sigma_prior=1.0):
    """Negative log posterior = -log p(D|theta) - log p(theta)."""
    pred = model(x_data)
    nll = 0.5 * ((pred.squeeze() - y_data)**2).sum() / sigma_noise**2
    prior = 0.5 * sum((p**2).sum() for p in model.parameters()) / sigma_prior**2
    return nll + prior


def run_e2():
    """E2: BNN posterior sampling for 1D sinusoidal regression."""
    print("=" * 60)
    print("E2: BNN Posterior (1D sinusoidal regression)")
    print("=" * 60)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Data
    x_data = torch.linspace(-1, 1, 20).unsqueeze(-1)
    y_data = torch.sin(3 * x_data.squeeze()) + 0.1 * torch.randn(20)
    x_test = torch.linspace(-2, 2, 200).unsqueeze(-1)

    model = TinyBNN()
    d = model.n_params()
    print(f"  BNN has {d} parameters")

    # --- NH-tanh with multi-scale Q ---
    print("Running NH-tanh multi-scale...")

    def bnn_forward_functional(theta_flat, x):
        """Functional forward pass: manually extract weights from flat vector."""
        # fc1: weight (20,1), bias (20,)  -> 20 + 20 = 40
        # fc2: weight (20,20), bias (20,) -> 400 + 20 = 420
        # fc3: weight (1,20), bias (1,)   -> 20 + 1 = 21
        idx = 0
        w1 = theta_flat[idx:idx+20].view(20, 1); idx += 20
        b1 = theta_flat[idx:idx+20]; idx += 20
        w2 = theta_flat[idx:idx+400].view(20, 20); idx += 400
        b2 = theta_flat[idx:idx+20]; idx += 20
        w3 = theta_flat[idx:idx+20].view(1, 20); idx += 20
        b3 = theta_flat[idx:idx+1]; idx += 1
        h = torch.tanh(x @ w1.T + b1)
        h = torch.tanh(h @ w2.T + b2)
        return (h @ w3.T + b3).squeeze(-1)

    def bnn_grad_fn(theta_flat):
        """Compute gradient of V(theta) = -log posterior."""
        theta = theta_flat.detach().requires_grad_(True)
        pred = bnn_forward_functional(theta, x_data)
        nll = 0.5 * ((pred - y_data)**2).sum() / 0.01  # sigma_noise=0.1
        prior = 0.5 * (theta**2).sum()
        V = nll + prior
        V.backward()
        return theta.grad.detach()

    # --- Find MAP estimate to initialize from (using Adam via manual param) ---
    print("  Finding MAP estimate...")
    torch.manual_seed(SEED)
    map_model = TinyBNN()
    map_opt = torch.optim.Adam(map_model.parameters(), lr=0.005)
    for epoch in range(2000):
        pred = map_model(x_data)
        # sigma_noise=0.1, so sigma^2=0.01
        nll = 0.5 * ((pred.squeeze() - y_data)**2).sum() / 0.01
        prior = 0.5 * sum((pp**2).sum() for pp in map_model.parameters())
        loss = nll + prior
        map_opt.zero_grad()
        loss.backward()
        map_opt.step()
        if epoch % 500 == 0:
            print(f"    MAP epoch {epoch}: loss = {loss.item():.2f}")
    theta_map = map_model.get_flat_params().detach()

    # --- NH-tanh with multi-scale Q ---
    torch.manual_seed(SEED)
    Qs = [1.0, 10.0, 100.0]  # Larger Q for high-d to slow xi dynamics
    N_therm = len(Qs)
    q = theta_map.detach().clone()
    p = torch.randn(d) * 0.1  # small initial momentum
    xis = torch.zeros(N_therm)
    Qs_t = torch.tensor(Qs, dtype=torch.float32)
    dt = 0.0002  # very small dt for high-d stability
    n_steps = 50000
    burn_in = 10000
    thin = 40

    nh_samples = []
    t0 = time.time()
    for step in range(n_steps):
        gv = bnn_grad_fn(q)
        # Clip gradient for stability
        gv = torch.clamp(gv, -100, 100)
        gs = torch.tanh(xis)
        total_g = gs.sum()

        # Velocity Verlet with friction (splitting)
        # Half-step momentum
        p = p + 0.5 * dt * (-gv - total_g * p)
        # Full-step position
        q = q + dt * p
        # Update xi (half step before, half after)
        KE = (p**2).sum()
        xis = xis + dt * (1.0 / Qs_t) * (KE - d * 1.0)
        # Recompute gradient at new position
        gv = bnn_grad_fn(q)
        gv = torch.clamp(gv, -100, 100)
        gs = torch.tanh(xis)
        total_g = gs.sum()
        # Half-step momentum
        p = p + 0.5 * dt * (-gv - total_g * p)

        if step >= burn_in and (step - burn_in) % thin == 0:
            nh_samples.append(q.detach().clone())

    nh_time = time.time() - t0
    print(f"  NH-tanh: {len(nh_samples)} samples in {nh_time:.2f}s")

    # --- SGLD ---
    print("Running SGLD...")
    torch.manual_seed(SEED)
    theta = theta_map.detach().clone()
    sgld_samples = []
    eps = 1e-5  # smaller step for stability
    sgld_steps = 50000
    sgld_burn = 10000
    sgld_thin = 40

    t0 = time.time()
    for step in range(sgld_steps):
        gv = bnn_grad_fn(theta)
        gv = torch.clamp(gv, -100, 100)
        theta = theta - eps * gv + np.sqrt(2 * eps) * torch.randn_like(theta)

        if step >= sgld_burn and (step - sgld_burn) % sgld_thin == 0:
            sgld_samples.append(theta.detach().clone())

    sgld_time = time.time() - t0
    print(f"  SGLD: {len(sgld_samples)} samples in {sgld_time:.2f}s")

    # --- Deep ensemble (5 MAP estimates) ---
    print("Running Deep Ensemble...")
    ensemble_params = []
    t0 = time.time()
    for ens_i in range(5):
        torch.manual_seed(SEED + ens_i + 100)
        m = TinyBNN()
        opt = torch.optim.Adam(m.parameters(), lr=0.01)
        for epoch in range(2000):
            pred = m(x_data)
            loss = 0.5 * ((pred.squeeze() - y_data)**2).sum() / 0.01 + \
                   0.5 * sum((p**2).sum() for p in m.parameters())
            opt.zero_grad()
            loss.backward()
            opt.step()
        ensemble_params.append(m.get_flat_params().detach().clone())
    ens_time = time.time() - t0
    print(f"  Ensemble: 5 models in {ens_time:.2f}s")

    # --- Evaluate predictions ---
    def predict_with_params(params_list, x):
        preds = []
        for p_vec in params_list:
            with torch.no_grad():
                pred = bnn_forward_functional(p_vec, x)
                preds.append(pred.numpy())
        return np.array(preds)

    nh_preds = predict_with_params(nh_samples, x_test)
    sgld_preds = predict_with_params(sgld_samples, x_test)
    ens_preds = predict_with_params(ensemble_params, x_test)

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    x_np = x_test.squeeze().numpy()
    xd_np = x_data.squeeze().numpy()
    yd_np = y_data.numpy()
    true_fn = np.sin(3 * x_np)

    for ax, preds, title, color in [
        (axes[0], nh_preds, f'(a) NH-tanh (N={len(nh_samples)})', C_NH),
        (axes[1], sgld_preds, f'(b) SGLD (N={len(sgld_samples)})', C_SGLD),
        (axes[2], ens_preds, f'(c) Deep Ensemble (N=5)', C_ENSEMBLE),
    ]:
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        ax.plot(x_np, true_fn, 'k--', label='True', linewidth=1)
        ax.plot(x_np, mean, color=color, label='Mean pred.', linewidth=1.5)
        ax.fill_between(x_np, mean - 2*std, mean + 2*std, color=color, alpha=0.2,
                        label='$\\pm 2\\sigma$')
        ax.scatter(xd_np, yd_np, c='black', s=20, zorder=5, label='Data')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title, fontweight='bold')
        ax.set_ylim(-3, 3)
        ax.legend(frameon=False, fontsize=9, loc='upper left')

    fig.savefig(os.path.join(FIGDIR, 'e2_bnn.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved e2_bnn.png")

    # Compute calibration (fraction of test points within 95% CI)
    y_true_test = np.sin(3 * x_np)
    for name, preds in [('NH-tanh', nh_preds), ('SGLD', sgld_preds), ('Ensemble', ens_preds)]:
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        in_ci = np.mean(np.abs(y_true_test - mean) < 2 * std)
        print(f"  {name} calibration (95% CI coverage): {in_ci:.3f}")

    return nh_preds, sgld_preds, ens_preds


# =============================================================================
# E3: Analytical divergence speedup benchmark
# =============================================================================

def run_e3():
    """E3: Compare divergence computation methods on 10D Gaussian mixture."""
    print("=" * 60)
    print("E3: Analytical Divergence Speedup (10D GMM)")
    print("=" * 60)

    torch.manual_seed(SEED)
    d = 10
    K = 5

    # Set up GMM
    angles = torch.linspace(0, 2*np.pi, K+1)[:K]
    means = 3.0 * torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    # Embed in 10D (first 2 dims have structure, rest are standard normal)
    means_10d = torch.zeros(K, d)
    means_10d[:, :2] = means
    weights = torch.ones(K) / K

    def gmm_grad(x):
        _, g = gmm_potential_and_grad(x, means_10d, weights)
        return g

    n_steps = 1000
    dt = 0.01
    Q = 1.0

    q = torch.randn(d)
    p = torch.randn(d)
    xi = torch.zeros(1)

    # Method 1: Analytical divergence
    print("  Method 1: Analytical div...")
    q1, p1, xi1 = q.clone(), p.clone(), xi.clone()
    t0 = time.time()
    for _ in range(n_steps):
        q1, p1, xi1, div_int = nh_tanh_rk4_step(q1, p1, xi1, gmm_grad, dt, Q, 1.0, d)
    t_analytical = time.time() - t0
    print(f"    Time: {t_analytical:.4f}s")

    # Method 2: Hutchinson trace estimator (1, 5, 10 random vectors)
    def nh_step_with_hutchinson(q, p, xi, grad_fn, dt, n_vec=1):
        """One step + Hutchinson trace estimator."""
        q_new, p_new, xi_new, _ = nh_tanh_rk4_step(q, p, xi, grad_fn, dt, Q, 1.0, d)

        # Hutchinson: div = E[v^T J v] where v is random
        # We need Jacobian of the full dynamics wrt (q, p, xi)
        # But for fair comparison, we estimate div of dp/dt wrt p = -d*tanh(xi)
        # using Hutchinson with random projections through autograd
        state = torch.cat([q, p, xi])
        state.requires_grad_(True)

        trace_est = 0.0
        for _ in range(n_vec):
            v = torch.randn(d)  # random vector for p-component
            # Compute v^T (d/dp)(dp/dt) using forward-mode AD approximation
            # dp/dt = -grad_V(q) - tanh(xi)*p
            # d(dp/dt)/dp = -tanh(xi)*I  => trace = -d*tanh(xi)
            # But Hutchinson doesn't know this, so we compute via finite diff
            eps_fd = 1e-4
            p_plus = p + eps_fd * v
            p_minus = p - eps_fd * v
            g = torch.tanh(xi)
            f_plus = -grad_fn(q) - g * p_plus
            f_minus = -grad_fn(q) - g * p_minus
            trace_est += (v * (f_plus - f_minus) / (2 * eps_fd)).sum()
        trace_est /= n_vec

        return q_new, p_new, xi_new, trace_est

    times_hutch = {}
    for n_vec in [1, 5, 10]:
        q2, p2, xi2 = q.clone(), p.clone(), xi.clone()
        t0 = time.time()
        for _ in range(n_steps):
            q2, p2, xi2, _ = nh_step_with_hutchinson(q2, p2, xi2, gmm_grad, dt, n_vec)
        t_hutch = time.time() - t0
        times_hutch[n_vec] = t_hutch
        print(f"  Hutchinson({n_vec}): {t_hutch:.4f}s")

    # Method 3: Brute force Jacobian trace
    print("  Method 3: Brute force Jacobian...")
    q3, p3, xi3 = q.clone(), p.clone(), xi.clone()
    t0 = time.time()
    for _ in range(n_steps):
        q3, p3, xi3, _ = nh_tanh_rk4_step(q3, p3, xi3, gmm_grad, dt, Q, 1.0, d)
        # Brute force: compute full Jacobian of dp/dt wrt p
        p3_req = p3.detach().requires_grad_(True)
        g = torch.tanh(xi3)
        f_p = -gmm_grad(q3) - g * p3_req
        jac = torch.zeros(d, d)
        for i in range(d):
            if p3_req.grad is not None:
                p3_req.grad.zero_()
            f_p[i].backward(retain_graph=True)
            jac[i] = p3_req.grad
        trace_bf = jac.diagonal().sum()
    t_brute = time.time() - t0
    print(f"  Brute force: {t_brute:.4f}s")

    # --- Speedup ratios ---
    methods = ['Analytical', 'Hutch(1)', 'Hutch(5)', 'Hutch(10)', 'Brute force']
    times = [t_analytical, times_hutch[1], times_hutch[5], times_hutch[10], t_brute]
    speedups = [t / t_analytical for t in times]

    print("\n  Speedup ratios (relative to analytical=1.0x):")
    for m, t, s in zip(methods, times, speedups):
        print(f"    {m}: {t:.4f}s ({s:.1f}x)")

    # --- Dimension scaling study ---
    print("\n  Dimension scaling study (Analytical vs Hutch(1) vs Brute)...")
    dims_to_test = [10, 50, 100, 500]
    scaling_results = {dd: {} for dd in dims_to_test}
    n_steps_scale = 200  # fewer steps for scaling study

    for dd in dims_to_test:
        torch.manual_seed(SEED)
        K_dd = 5
        means_dd = torch.zeros(K_dd, dd)
        means_dd[:, :2] = means[:K_dd, :2] if K_dd <= K else means[:, :2]
        weights_dd = torch.ones(K_dd) / K_dd

        def gmm_grad_dd(x, m=means_dd, w=weights_dd):
            _, g = gmm_potential_and_grad(x, m, w)
            return g

        q_dd = torch.randn(dd)
        p_dd = torch.randn(dd)
        xi_dd = torch.zeros(1)

        # Analytical
        q_, p_, xi_ = q_dd.clone(), p_dd.clone(), xi_dd.clone()
        t0 = time.time()
        for _ in range(n_steps_scale):
            q_, p_, xi_, _ = nh_tanh_rk4_step(q_, p_, xi_, gmm_grad_dd, dt, Q, 1.0, dd)
        scaling_results[dd]['analytical'] = time.time() - t0

        # Hutchinson(1)
        q_, p_, xi_ = q_dd.clone(), p_dd.clone(), xi_dd.clone()
        t0 = time.time()
        for _ in range(n_steps_scale):
            q_, p_, xi_, _ = nh_tanh_rk4_step(q_, p_, xi_, gmm_grad_dd, dt, Q, 1.0, dd)
            v = torch.randn(dd)
            eps_fd = 1e-4
            g_xi = torch.tanh(xi_)
            f_plus = -gmm_grad_dd(q_) - g_xi * (p_ + eps_fd * v)
            f_minus = -gmm_grad_dd(q_) - g_xi * (p_ - eps_fd * v)
            trace_h = (v * (f_plus - f_minus) / (2 * eps_fd)).sum()
        scaling_results[dd]['hutch1'] = time.time() - t0

        # Brute force (skip for d>=100 -- too slow)
        if dd <= 100:
            q_, p_, xi_ = q_dd.clone(), p_dd.clone(), xi_dd.clone()
            t0 = time.time()
            for _ in range(n_steps_scale):
                q_, p_, xi_, _ = nh_tanh_rk4_step(q_, p_, xi_, gmm_grad_dd, dt, Q, 1.0, dd)
                p_req = p_.detach().requires_grad_(True)
                g_xi = torch.tanh(xi_)
                f_p = -gmm_grad_dd(q_) - g_xi * p_req
                jac = torch.zeros(dd, dd)
                for i in range(dd):
                    if p_req.grad is not None:
                        p_req.grad.zero_()
                    f_p[i].backward(retain_graph=True)
                    jac[i] = p_req.grad
                trace_bf = jac.diagonal().sum()
            scaling_results[dd]['brute'] = time.time() - t0
        else:
            scaling_results[dd]['brute'] = np.nan

        ratio_h = scaling_results[dd]['hutch1'] / scaling_results[dd]['analytical']
        print(f"    d={dd}: analytical={scaling_results[dd]['analytical']:.3f}s, "
              f"hutch1={scaling_results[dd]['hutch1']:.3f}s ({ratio_h:.1f}x), "
              f"brute={scaling_results[dd].get('brute', 'N/A')}")

    # --- Plot: 2-panel figure ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # Panel (a): Bar chart at d=10
    colors_bar = [C_NH, C_SGLD, '#ff7f0e', '#d62728', '#9467bd']
    bars = axes[0].bar(methods, times, color=colors_bar, edgecolor='black', linewidth=0.5)
    for bar, s in zip(bars, speedups):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02*max(times),
                     f'{s:.1f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Wall-clock time (s)')
    axes[0].set_title('(a) Divergence cost at d=10', fontweight='bold')
    axes[0].set_ylim(0, max(times) * 1.25)

    # Panel (b): Scaling with dimension
    dims_arr = np.array(dims_to_test)
    t_ana = [scaling_results[dd]['analytical'] for dd in dims_to_test]
    t_h1 = [scaling_results[dd]['hutch1'] for dd in dims_to_test]
    t_bf = [scaling_results[dd]['brute'] for dd in dims_to_test]
    speedup_h1 = [h/a for h, a in zip(t_h1, t_ana)]

    axes[1].plot(dims_arr, speedup_h1, 'o-', color=C_SGLD, label='Hutch(1) / Analytical',
                 markersize=6, linewidth=2)
    # Add brute force speedup where available
    speedup_bf = []
    dims_bf = []
    for dd in dims_to_test:
        if not np.isnan(scaling_results[dd]['brute']):
            speedup_bf.append(scaling_results[dd]['brute'] / scaling_results[dd]['analytical'])
            dims_bf.append(dd)
    if dims_bf:
        axes[1].plot(dims_bf, speedup_bf, 's--', color=C_REF, label='Brute / Analytical',
                     markersize=6, linewidth=2)

    axes[1].axhline(1.0, color='gray', ls=':', alpha=0.5)
    axes[1].set_xlabel('Dimension d')
    axes[1].set_ylabel('Slowdown ratio vs analytical')
    axes[1].set_title('(b) Scaling with dimension', fontweight='bold')
    axes[1].legend(frameon=False)
    axes[1].set_xscale('log')

    fig.savefig(os.path.join(FIGDIR, 'e3_speedup.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved e3_speedup.png")

    return dict(zip(methods, times))


# =============================================================================
# E3b: Verify analytical divergence is correct
# =============================================================================

def verify_analytical_divergence():
    """Verify that analytical div = -d*tanh(xi) matches autograd Jacobian trace.

    The NH ODE for the momentum is dp/dt = -grad_V(q) - g(xi)*p.
    The divergence wrt the full state (q, p, xi) of the RHS is:
      d/dq_i(dq_i/dt) = d/dq_i(p_i) = 0
      d/dp_i(dp_i/dt) = d/dp_i(-grad_V_i - g(xi)*p_i) = -g(xi)  (for each i)
      d/dxi(dxi/dt) = d/dxi((1/Q)(p^2 - d*kT)) = 0
    Total divergence = sum_i(-g(xi)) = -d * g(xi).

    We verify using torch autograd on the p-component.
    """
    print("\n" + "=" * 60)
    print("Verification: Analytical divergence vs autograd")
    print("=" * 60)

    torch.manual_seed(SEED)
    d = 5
    xi_val = 0.7

    # Analytical: div = -d * tanh(xi)
    analytical_div = -d * np.tanh(xi_val)

    # Autograd verification: dp_i/dt = -g(xi)*p_i + f_i(q)
    # d(dp_i/dt)/dp_i = -g(xi) for each i, so trace = -d*g(xi)
    p = torch.randn(d, requires_grad=True)
    xi = torch.tensor([xi_val])
    g = torch.tanh(xi)

    # dp/dt (the p-dependent part only, grad_V doesn't depend on p)
    dp_dt = -g * p  # shape (d,)
    # Compute trace of Jacobian d(dp/dt)/dp via autograd
    trace = 0.0
    for i in range(d):
        grad_i = torch.autograd.grad(dp_dt[i], p, retain_graph=True)[0]
        trace += grad_i[i].item()

    print(f"  Analytical divergence: {analytical_div:.6f}")
    print(f"  Autograd trace:       {trace:.6f}")
    print(f"  Absolute error:       {abs(analytical_div - trace):.2e}")

    assert abs(analytical_div - trace) < 1e-6, "Divergence mismatch!"
    print("  PASSED: analytical divergence matches autograd trace.")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    verify_analytical_divergence()

    kl_nh, kl_sgld = run_e1()

    run_e2()

    e3_times = run_e3()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"E1 banana KL:  NH-tanh={kl_nh:.4f}  SGLD={kl_sgld:.4f}")
    print(f"E3 speedup:    Analytical={e3_times['Analytical']:.3f}s  "
          f"Hutch(1)={e3_times['Hutch(1)']:.3f}s  "
          f"Brute={e3_times['Brute force']:.3f}s")
    speedup = e3_times['Hutch(1)'] / e3_times['Analytical']
    print(f"E3 speedup ratio (Hutch(1)/Analytical): {speedup:.1f}x")
