#!/usr/bin/env python3
"""E3: Learn (Q_1,...,Q_N) jointly for multi-scale thermostats.

N parallel thermostat variables, each with independent Q_k.
Total friction = sum of g(xi_k) = sum of xi_k (linear NH).
Learn the Q_k distribution to test whether 1/f log-spacing is optimal.
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


class MultiThermostatNH(Integrator):
    """N parallel Nose-Hoover thermostats with independent Q_k.

    Each thermostat variable xi_k evolves as:
        dxi_k/dt = (1/Q_k) * (v^2 - D*kT)
    Total friction on velocities:
        v *= exp(-sum_k(xi_k) * dt/2)

    Q_k are log-parameterized: Q_k = exp(log_Q_k).
    """
    def __init__(self, kT=1.0, mass=1.0, N_thermo=3, Q_init=None):
        super().__init__()
        self.kT = nn.Parameter(torch.tensor(float(kT)))
        self.mass = nn.Parameter(torch.tensor(float(mass)))
        self.N_thermo = N_thermo

        if Q_init is None:
            Q_init = torch.ones(N_thermo)
        else:
            Q_init = torch.tensor(Q_init, dtype=torch.float32)

        self.log_Qs = nn.Parameter(torch.log(Q_init))

    @property
    def Qs(self):
        return torch.exp(self.log_Qs)

    def step(self, x, v, alphas, force_fn, dt):
        """
        alphas: (..., N_thermo) — one xi per thermostat
        """
        ndof = x.shape[-1]
        Qs = self.Qs  # (N_thermo,)

        v2 = (v**2).sum(dim=-1)  # (...,)

        # First thermostat half-step
        driving = (v2 - ndof * self.kT)  # (...,)
        # Update each alpha_k: alpha_k += (dt/4) * driving / Q_k
        alphas = alphas + (dt / 4) * driving.unsqueeze(-1) / Qs.unsqueeze(0)

        # Total friction = sum of xi_k
        total_friction = alphas.sum(dim=-1)  # (...,)
        v = v * torch.exp(-total_friction.unsqueeze(-1) * dt / 2)

        v2 = (v**2).sum(dim=-1)
        driving = (v2 - ndof * self.kT)
        alphas = alphas + (dt / 4) * driving.unsqueeze(-1) / Qs.unsqueeze(0)

        # Velocity-Verlet
        v = v + (dt / 2) * force_fn(x) / self.mass
        x = x + dt * v
        v = v + (dt / 2) * force_fn(x) / self.mass

        # Second thermostat half-step
        v2 = (v**2).sum(dim=-1)
        driving = (v2 - ndof * self.kT)
        alphas = alphas + (dt / 4) * driving.unsqueeze(-1) / Qs.unsqueeze(0)

        total_friction = alphas.sum(dim=-1)
        v = v * torch.exp(-total_friction.unsqueeze(-1) * dt / 2)

        v2 = (v**2).sum(dim=-1)
        driving = (v2 - ndof * self.kT)
        alphas = alphas + (dt / 4) * driving.unsqueeze(-1) / Qs.unsqueeze(0)

        return x, v, alphas

    def run(self, x0, v0, force_fn, dt, n_steps, store_every=1, final_only=False):
        v = v0 if v0 is not None else torch.randn_like(x0) * torch.sqrt(self.kT / self.mass)
        batch_shape = x0.shape[:-1]
        alphas = torch.zeros(*batch_shape, self.N_thermo, device=x0.device, dtype=x0.dtype)
        state = (x0, v, alphas)
        step_fn = lambda x, v, a: self.step(x, v, a, force_fn, dt)
        return self._integrate(state, step_fn, n_steps, store_every, final_only, store_indices=[0, 1])


def train_multi_thermo(pot, dim, kappas, kT, N_thermo, seed,
                       n_epochs=100, n_steps=4000, batch_size=8):
    """Train N parallel thermostats from given seed."""
    true_var = kT / kappas
    kappa_max = kappas.max().item()
    dt = min(0.01, 0.05 / np.sqrt(kappa_max))

    torch.manual_seed(seed)
    # Initialize with log-uniform spread
    Q_init = torch.logspace(-1, 1, N_thermo)
    model = MultiThermostatNH(kT=kT, mass=1.0, N_thermo=N_thermo, Q_init=Q_init)
    model.kT.requires_grad_(False)
    model.mass.requires_grad_(False)

    optimizer = torch.optim.Adam([model.log_Qs], lr=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    q_history, loss_history = [], []
    for epoch in range(n_epochs):
        torch.manual_seed(epoch * 13 + seed)
        x0 = torch.randn(batch_size, dim) * 0.1
        v0 = torch.randn(batch_size, dim) * np.sqrt(kT)

        traj_x, _ = model.run(x0, v0, pot.force, dt=dt, n_steps=n_steps)
        loss = combined_loss(traj_x, true_var, alpha=0.5)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([model.log_Qs], 2.0)
        optimizer.step()
        scheduler.step()

        q_history.append(model.Qs.detach().tolist())
        loss_history.append(loss.item())
        if epoch % 25 == 0 or epoch == n_epochs - 1:
            qs_str = ", ".join([f"{q:.4f}" for q in model.Qs.detach().tolist()])
            print(f"    seed={seed} epoch {epoch:3d}: Qs=[{qs_str}], loss={loss.item():.4f}")

    return {
        "Qs_learned": model.Qs.detach().tolist(),
        "log_Qs_learned": model.log_Qs.detach().tolist(),
        "Q_history": q_history,
        "loss_history": loss_history,
        "final_loss": loss_history[-1],
        "seed": seed,
        "N_thermo": N_thermo,
    }


def run_e3():
    print("=" * 60)
    print("E3: Learn (Q_1,...,Q_N) jointly")
    print("=" * 60)

    dim = 10
    kappa_max = 100
    kT = 1.0
    exponents = torch.linspace(0, 1, dim)
    kappas = kappa_max ** exponents
    pot = AnisotropicHarmonic(kappas)

    # F1 prescription from orbit #34
    Q_min_f1 = 1.0 / np.sqrt(kappa_max)
    Q_max_f1 = 1.0 / np.sqrt(1.0)  # kappa_min = 1
    print(f"F1 prescription: Q_min={Q_min_f1:.4f}, Q_max={Q_max_f1:.4f}")

    seeds = [42, 123, 256, 789, 1024]
    results = {"kappas": kappas.tolist(), "Q_min_f1": Q_min_f1, "Q_max_f1": Q_max_f1}

    for N_thermo in [3, 5]:
        print(f"\n=== N_thermo = {N_thermo} ===")
        for seed in seeds:
            label = f"N={N_thermo}_seed={seed}"
            print(f"\n--- {label} ---")
            r = train_multi_thermo(pot, dim, kappas, kT, N_thermo, seed,
                                   n_epochs=100, n_steps=4000, batch_size=8)
            results[label] = r
            qs = sorted(r["Qs_learned"])
            print(f"  Qs (sorted): {[f'{q:.4f}' for q in qs]}")
            print(f"  loss={r['final_loss']:.5f}")

    with open(f"{OUT_DIR}/e3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved e3_results.json")
    return results


def plot_e3(results):
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

    Q_min_f1 = results["Q_min_f1"]
    Q_max_f1 = results["Q_max_f1"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # (a) Learned Q_k for N=3, 5 seeds overlaid
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 0.5, 5))
    seeds = [42, 123, 256, 789, 1024]
    for N_thermo, marker in [(3, 'o'), (5, 's')]:
        for i, seed in enumerate(seeds):
            key = f"N={N_thermo}_seed={seed}"
            if key not in results:
                continue
            qs = sorted(results[key]["Qs_learned"])
            ax.semilogy(range(len(qs)), qs, marker=marker, color=colors[i],
                        alpha=0.7, markersize=8,
                        label=f'N={N_thermo} s{seed}' if N_thermo == 3 else None,
                        linestyle='-' if N_thermo == 3 else '--')
    # F1 reference
    ax.axhline(Q_min_f1, color='gray', ls='--', alpha=0.5, label=f'F1 Q_min={Q_min_f1:.3f}')
    ax.axhline(Q_max_f1, color='gray', ls=':', alpha=0.5, label=f'F1 Q_max={Q_max_f1:.3f}')
    ax.set_xlabel('Thermostat index k (sorted)')
    ax.set_ylabel(r'$Q_k$')
    ax.set_title('(a) Learned Q distribution (N=3)', fontweight='bold')
    ax.legend(fontsize=8, frameon=False)

    # (b) Same for N=5
    ax = axes[1]
    for i, seed in enumerate(seeds):
        key = f"N=5_seed={seed}"
        if key not in results:
            continue
        qs = sorted(results[key]["Qs_learned"])
        ax.semilogy(range(len(qs)), qs, 's-', color=colors[i],
                    alpha=0.7, markersize=8, label=f's{seed}')
    # F1 log-uniform reference for N=5
    f1_qs = np.logspace(np.log10(Q_min_f1), np.log10(Q_max_f1), 5)
    ax.semilogy(range(5), f1_qs, 'kx--', markersize=10, label='F1 log-uniform')
    ax.axhline(Q_min_f1, color='gray', ls='--', alpha=0.3)
    ax.axhline(Q_max_f1, color='gray', ls=':', alpha=0.3)
    ax.set_xlabel('Thermostat index k (sorted)')
    ax.set_ylabel(r'$Q_k$')
    ax.set_title('(b) Learned Q distribution (N=5)', fontweight='bold')
    ax.legend(fontsize=8, frameon=False)

    # (c) Loss comparison: N=1 (from E1) vs N=3 vs N=5
    ax = axes[2]
    # Collect final losses
    for N_thermo in [3, 5]:
        losses = []
        for seed in seeds:
            key = f"N={N_thermo}_seed={seed}"
            if key in results:
                losses.append(results[key]["final_loss"])
        if losses:
            ax.bar(N_thermo, np.mean(losses), yerr=np.std(losses),
                   width=0.8, color='steelblue' if N_thermo == 3 else 'coral',
                   alpha=0.7, label=f'N={N_thermo}', capsize=5)

    # Try to load E1 result for N=1
    try:
        with open(f"{OUT_DIR}/e1_results.json") as f:
            e1 = json.load(f)
        e1_loss = e1["D=10_k=100"]["final_loss"]
        ax.bar(1, e1_loss, width=0.8, color='gray', alpha=0.7, label='N=1 (E1)')
    except (FileNotFoundError, KeyError):
        pass

    ax.set_xlabel('Number of thermostats N')
    ax.set_ylabel('Final combined loss')
    ax.set_title('(c) Loss vs N thermostats', fontweight='bold')
    ax.set_xticks([1, 3, 5])
    ax.legend(fontsize=10, frameon=False)

    fig.savefig(f"{OUT_DIR}/figures/e3_learned_Q_distribution.png")
    plt.close(fig)
    print("Saved figures/e3_learned_Q_distribution.png")


if __name__ == "__main__":
    t0 = time.time()
    results = run_e3()
    plot_e3(results)
    print(f"\nE3 total wall time: {time.time()-t0:.1f}s")
