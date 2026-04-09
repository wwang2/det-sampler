#!/usr/bin/env python3
"""E2: Learn g(xi) as a constrained MLP.

Fix Q = Q_learned from E1 (best Q for D=10, kappa=100).
Parameterize g(xi) as odd-symmetric MLP, train jointly with log_Q.
Run 5 seeds to test whether shape converges or only g'(0) matters.
"""

import sys
import json
import time
import numpy as np

sys.path.insert(0, "/Users/wujiewang/code/uni-diffsim")

import torch
import torch.nn as nn

from uni_diffsim.integrators import Integrator

SEED_BASE = 42
OUT_DIR = "/Users/wujiewang/code/det-sampler/.worktrees/diff-thermostat-055/orbits/diff-thermostat-055"


class AnisotropicHarmonic(nn.Module):
    def __init__(self, kappas):
        super().__init__()
        self.kappas = kappas.float()
    def force(self, x):
        return -self.kappas * x


def variance_loss(traj_x, true_var):
    n = traj_x.shape[0]
    traj = traj_x[n // 4:]
    x2 = (traj**2).mean(dim=0).mean(dim=0)
    return ((x2 - true_var) / true_var).pow(2).mean()


def abs_rho_loss(traj_x):
    n = traj_x.shape[0]
    traj = traj_x[n // 4:]
    mu = traj.mean(dim=0, keepdim=True)
    x = traj[:-1] - mu
    x1 = traj[1:] - mu
    var = x.pow(2).mean(dim=0) + 1e-10
    rho1 = (x * x1).mean(dim=0) / var
    return rho1.abs().mean()


def combined_loss(traj_x, true_var, alpha=0.5):
    return alpha * variance_loss(traj_x, true_var) + (1 - alpha) * abs_rho_loss(traj_x)


class LearnableFrictionNH(Integrator):
    """Nose-Hoover with learnable odd-symmetric friction g(xi).

    g(xi) is enforced odd via: g(xi) = net(xi) - net(-xi).
    The net is a small MLP. We also learn log_Q jointly.
    """
    def __init__(self, kT=1.0, Q=1.0, mass=1.0, hidden=32):
        super().__init__()
        self.kT = nn.Parameter(torch.tensor(float(kT)))
        self.mass = nn.Parameter(torch.tensor(float(mass)))
        self.log_Q = nn.Parameter(torch.tensor(float(np.log(Q))))
        self.g_net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Initialize near identity: g(xi) ~ xi
        with torch.no_grad():
            for m in self.g_net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    nn.init.zeros_(m.bias)

    @property
    def Q(self):
        return torch.exp(self.log_Q)

    def g(self, alpha):
        """Odd-symmetric g: g(-xi) = -g(xi)."""
        shape = alpha.shape
        a_flat = alpha.reshape(-1, 1)
        out_pos = self.g_net(a_flat)
        out_neg = self.g_net(-a_flat)
        delta = (out_pos - out_neg).reshape(shape)
        # Residual connection: g(xi) = xi + small_odd_correction
        return alpha + delta

    def step(self, x, v, alpha, force_fn, dt):
        ndof = x.shape[-1]
        Q = self.Q

        v2 = (v**2).sum(dim=-1)
        alpha = alpha + (dt / 4) * (v2 - ndof * self.kT) / Q
        g_alpha = self.g(alpha)
        v = v * torch.exp(-g_alpha.unsqueeze(-1) * dt / 2)
        v2 = (v**2).sum(dim=-1)
        alpha = alpha + (dt / 4) * (v2 - ndof * self.kT) / Q

        v = v + (dt / 2) * force_fn(x) / self.mass
        x = x + dt * v
        v = v + (dt / 2) * force_fn(x) / self.mass

        v2 = (v**2).sum(dim=-1)
        alpha = alpha + (dt / 4) * (v2 - ndof * self.kT) / Q
        g_alpha = self.g(alpha)
        v = v * torch.exp(-g_alpha.unsqueeze(-1) * dt / 2)
        v2 = (v**2).sum(dim=-1)
        alpha = alpha + (dt / 4) * (v2 - ndof * self.kT) / Q
        return x, v, alpha

    def run(self, x0, v0, force_fn, dt, n_steps, store_every=1, final_only=False):
        v = v0 if v0 is not None else torch.randn_like(x0) * torch.sqrt(self.kT / self.mass)
        alpha = torch.zeros(x0.shape[:-1], device=x0.device, dtype=x0.dtype)
        state = (x0, v, alpha)
        step_fn = lambda x, v, a: self.step(x, v, a, force_fn, dt)
        return self._integrate(state, step_fn, n_steps, store_every, final_only, store_indices=[0, 1])


def train_g_seed(pot, dim, kappas, kT, Q_init, seed, n_epochs=100, n_steps=4000, batch_size=8):
    """Train one instance of LearnableFrictionNH from a given seed."""
    true_var = kT / kappas
    kappa_max = kappas.max().item()
    dt = min(0.01, 0.05 / np.sqrt(kappa_max))

    torch.manual_seed(seed)
    model = LearnableFrictionNH(kT=kT, Q=Q_init, mass=1.0, hidden=32)
    model.kT.requires_grad_(False)
    model.mass.requires_grad_(False)

    param_groups = [
        {"params": [model.log_Q], "lr": 0.05},
        {"params": model.g_net.parameters(), "lr": 0.003},
    ]
    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    q_hist, loss_hist = [], []
    for epoch in range(n_epochs):
        torch.manual_seed(epoch * 11 + seed)
        x0 = torch.randn(batch_size, dim) * 0.1
        v0 = torch.randn(batch_size, dim) * np.sqrt(kT)

        traj_x, _ = model.run(x0, v0, pot.force, dt=dt, n_steps=n_steps)
        loss = combined_loss(traj_x, true_var, alpha=0.5)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()

        q_hist.append(model.Q.item())
        loss_hist.append(loss.item())
        if epoch % 25 == 0 or epoch == n_epochs - 1:
            print(f"    seed={seed} epoch {epoch:3d}: Q={model.Q.item():.5f}, loss={loss.item():.4f}")

    # Evaluate g on a grid
    xi_range = torch.linspace(-4, 4, 200)
    with torch.no_grad():
        g_vals = model.g(xi_range).numpy()

    # Compute g'(0) via finite difference
    eps = 0.001
    with torch.no_grad():
        gp = model.g(torch.tensor(eps)).item()
        gm = model.g(torch.tensor(-eps)).item()
    g_prime_0 = (gp - gm) / (2 * eps)

    return {
        "Q_learned": model.Q.item(),
        "Q_history": q_hist,
        "loss_history": loss_hist,
        "g_vals": g_vals.tolist(),
        "g_prime_0": g_prime_0,
        "final_loss": loss_hist[-1],
        "seed": seed,
    }


def run_e2():
    print("=" * 60)
    print("E2: Learn g(xi) as constrained MLP")
    print("=" * 60)

    dim = 10
    kappa_max = 100
    kT = 1.0
    exponents = torch.linspace(0, 1, dim)
    kappas = kappa_max ** exponents
    pot = AnisotropicHarmonic(kappas)

    # Load Q_learned from E1 if available, else use default
    try:
        with open(f"{OUT_DIR}/e1_results.json") as f:
            e1 = json.load(f)
        Q_init = e1["D=10_k=100"]["Q_learned"]
        print(f"Using Q_init from E1: {Q_init:.5f}")
    except (FileNotFoundError, KeyError):
        Q_init = 0.5
        print(f"E1 results not found, using Q_init={Q_init}")

    seeds = [42, 123, 256, 789, 1024]
    all_results = {"xi_range": torch.linspace(-4, 4, 200).tolist()}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        r = train_g_seed(pot, dim, kappas, kT, Q_init, seed,
                         n_epochs=100, n_steps=4000, batch_size=8)
        all_results[f"seed_{seed}"] = r
        print(f"  Q_learned={r['Q_learned']:.5f}, g'(0)={r['g_prime_0']:.4f}, loss={r['final_loss']:.5f}")

    with open(f"{OUT_DIR}/e2_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved e2_results.json")
    return all_results


def plot_e2(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams.update({
        'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11,
        'figure.dpi': 150, 'savefig.dpi': 300,
        'savefig.bbox_inches': 'tight', 'savefig.pad_inches': 0.2,
    })

    xi = np.array(results["xi_range"])
    seeds = [k for k in results.keys() if k.startswith("seed_")]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # (a) Learned g(xi) curves, 5 seeds overlaid
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(seeds)))
    g_prime_vals = []
    for i, sk in enumerate(seeds):
        r = results[sk]
        g_vals = np.array(r["g_vals"])
        ax.plot(xi, g_vals, color=colors[i], alpha=0.8,
                label=f'seed {r["seed"]}: g\'(0)={r["g_prime_0"]:.2f}')
        g_prime_vals.append(r["g_prime_0"])
    # Reference functions
    ax.plot(xi, xi, 'k--', alpha=0.4, label=r'$g(\xi)=\xi$ (linear NH)')
    ax.plot(xi, np.tanh(xi), 'k:', alpha=0.4, label=r'$g(\xi)=\tanh(\xi)$')
    ax.set_xlabel(r'$\xi$')
    ax.set_ylabel(r'$g(\xi)$')
    ax.set_title(r"(a) Learned $g(\xi)$ (5 seeds)", fontweight='bold')
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlim(-4, 4)
    ax.axhline(0, color='gray', alpha=0.2)
    ax.axvline(0, color='gray', alpha=0.2)

    # (b) g'(0) distribution
    ax = axes[1]
    Q_vals = [results[sk]["Q_learned"] for sk in seeds]
    ax.bar(range(len(seeds)), g_prime_vals, color='steelblue', alpha=0.7, label="g'(0)")
    ax.axhline(np.mean(g_prime_vals), color='red', ls='--',
               label=f"mean g'(0) = {np.mean(g_prime_vals):.3f}")
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f's{results[sk]["seed"]}' for sk in seeds], fontsize=10)
    ax.set_ylabel("g'(0)")
    ax.set_title("(b) g'(0) across seeds", fontweight='bold')
    ax.legend(fontsize=10, frameon=False)

    # Add Q values as text
    for i, q in enumerate(Q_vals):
        ax.text(i, g_prime_vals[i] + 0.02, f'Q={q:.3f}', ha='center', fontsize=8)

    # (c) Loss curves
    ax = axes[2]
    for i, sk in enumerate(seeds):
        r = results[sk]
        ax.plot(r["loss_history"], color=colors[i], alpha=0.7,
                label=f'seed {r["seed"]}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Combined loss')
    ax.set_title('(c) Training convergence', fontweight='bold')
    ax.legend(fontsize=9, frameon=False)

    fig.savefig(f"{OUT_DIR}/figures/e2_learned_g.png")
    plt.close(fig)
    print("Saved figures/e2_learned_g.png")


if __name__ == "__main__":
    t0 = time.time()
    results = run_e2()
    plot_e2(results)
    print(f"\nE2 total wall time: {time.time()-t0:.1f}s")
