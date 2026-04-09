#!/usr/bin/env python3
"""E3: Learn (Q_1,...,Q_N) jointly for multi-scale thermostats.

Reduced: n_steps=1000, batch=2, epochs=60.
"""

import sys
import json
import time
import numpy as np

sys.path.insert(0, "/Users/wujiewang/code/uni-diffsim")

import torch
import torch.nn as nn

from uni_diffsim.integrators import Integrator

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

    Each xi_k evolves as: dxi_k/dt = (1/Q_k) * (v^2 - D*kT)
    Total friction: v *= exp(-sum(xi_k) * dt/2)
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
        ndof = x.shape[-1]
        Qs = self.Qs  # (N_thermo,)
        v2 = (v**2).sum(dim=-1)  # (batch,)
        driving = (v2 - ndof * self.kT)  # (batch,)

        # First thermostat half-step
        alphas = alphas + (dt / 4) * driving.unsqueeze(-1) / Qs.unsqueeze(0)
        total_friction = alphas.sum(dim=-1)
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
                       n_epochs=60, n_steps=1000, batch_size=2):
    true_var = kT / kappas
    kappa_max = kappas.max().item()
    dt = min(0.01, 0.05 / np.sqrt(kappa_max))

    torch.manual_seed(seed)
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
        if epoch % 15 == 0 or epoch == n_epochs - 1:
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

    Q_min_f1 = 1.0 / np.sqrt(kappa_max)
    Q_max_f1 = 1.0 / np.sqrt(1.0)
    print(f"F1 prescription: Q_min={Q_min_f1:.4f}, Q_max={Q_max_f1:.4f}")

    seeds = [42, 123, 256, 789, 1024]
    results = {"kappas": kappas.tolist(), "Q_min_f1": Q_min_f1, "Q_max_f1": Q_max_f1}

    for N_thermo in [3, 5]:
        print(f"\n=== N_thermo = {N_thermo} ===")
        for seed in seeds:
            label = f"N={N_thermo}_seed={seed}"
            t_run = time.time()
            print(f"\n--- {label} ---")
            r = train_multi_thermo(pot, dim, kappas, kT, N_thermo, seed,
                                   n_epochs=60, n_steps=1000, batch_size=2)
            dt_run = time.time() - t_run
            results[label] = r
            qs = sorted(r["Qs_learned"])
            print(f"  Qs (sorted): {[f'{q:.4f}' for q in qs]}")
            print(f"  loss={r['final_loss']:.5f}, wall={dt_run:.1f}s")

    with open(f"{OUT_DIR}/e3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved e3_results.json")
    return results


if __name__ == "__main__":
    t0 = time.time()
    results = run_e3()
    print(f"\nE3 total wall time: {time.time()-t0:.1f}s")
