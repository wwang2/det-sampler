#!/usr/bin/env python3
"""Learn thermostat parameters (Q and g(xi)) via backpropagation through
differentiable Nose-Hoover dynamics.

Key design decisions:
  - Q = exp(log_Q) parameterization -> multiplicative updates in log space.
    Without this, Adam on raw Q cannot move 1 -> 0.04 (orders of magnitude).
  - Loss is a combination of:
      (a) variance_loss: relative error of <x_i^2> vs kT/kappa_i
          (forces correct marginal widths -> invariant measure fidelity)
      (b) abs_rho_loss: |lag-1 autocorrelation|
          (monotone in mixing quality; never rewards oscillatory anti-correlation)
  - kT and mass are frozen; only the thermostat parameters learn.
"""

import sys
import json
import time
import numpy as np

sys.path.insert(0, "/Users/wujiewang/code/uni-diffsim")

import torch
import torch.nn as nn

from uni_diffsim import NoseHoover, DoubleWell
from uni_diffsim.integrators import Integrator

torch.manual_seed(42)
OUT_DIR = "/Users/wujiewang/code/det-sampler/.worktrees/learn-thermostat-050/orbits/learn-thermostat-050"


class AnisotropicHarmonic(nn.Module):
    """U(x) = 0.5 * sum_i kappa_i * x_i^2"""
    def __init__(self, kappas):
        super().__init__()
        self.kappas = kappas.float()
    def force(self, x):
        return -self.kappas * x


class LogQNoseHoover(NoseHoover):
    """NoseHoover with log-parameterized Q = exp(log_Q) for multiplicative updates."""
    def __init__(self, kT=1.0, mass=1.0, Q=1.0):
        super().__init__(kT=kT, mass=mass, Q=Q)
        # Replace Q with log_Q; delete original Q param
        del self._parameters["Q"]
        self.log_Q = nn.Parameter(torch.tensor(float(np.log(Q))))

    @property
    def Q(self):
        return torch.exp(self.log_Q)


def variance_loss(traj_x, true_var):
    """Relative squared error of <x_i^2> vs kT/kappa_i.
    traj_x: (n_steps, batch, dim); true_var: (dim,)
    """
    n = traj_x.shape[0]
    traj = traj_x[n // 4:]  # burn-in 25%
    # Use <x^2> (not var) -- we want marginal second moment since center is 0
    x2 = (traj**2).mean(dim=0).mean(dim=0)  # -> (dim,)
    return ((x2 - true_var) / true_var).pow(2).mean()


def abs_rho_loss(traj_x):
    """|lag-1 autocorrelation| averaged over batch/dim.
    Monotone in tau_int in the range rho in [0,1]; never rewards negative rho.
    """
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


def train_Q(pot, dim, kappas, n_epochs=60, n_steps=2000, batch_size=4, dt=0.01, lr=0.1, kT=1.0, Q0=1.0, seed_offset=42, verbose_every=15):
    true_var = kT / kappas
    nh = LogQNoseHoover(kT=kT, mass=1.0, Q=Q0)
    nh.kT.requires_grad_(False)
    nh.mass.requires_grad_(False)

    optimizer = torch.optim.Adam([nh.log_Q], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    q_history, loss_history = [], []
    for epoch in range(n_epochs):
        torch.manual_seed(epoch * 7 + seed_offset)
        x0 = torch.randn(batch_size, dim) * 0.1
        v0 = torch.randn(batch_size, dim) * np.sqrt(kT)

        traj_x, _ = nh.run(x0, v0, pot.force, dt=dt, n_steps=n_steps)
        loss = combined_loss(traj_x, true_var, alpha=0.5)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([nh.log_Q], max_norm=2.0)
        optimizer.step()
        scheduler.step()

        q_history.append(nh.Q.item())
        loss_history.append(loss.item())
        if epoch % verbose_every == 0 or epoch == n_epochs - 1:
            print(f"    epoch {epoch:3d}: Q={nh.Q.item():.5f}, loss={loss.item():.4f}")
    return nh.Q.item(), q_history, loss_history


def run_e1():
    print("=" * 60); print("E1: Learn Q on anisotropic Gaussian"); print("=" * 60)
    results = {}
    kT = 1.0
    for dim in [2, 5, 10]:
        print(f"\n--- dim={dim} ---")
        kappas = torch.logspace(0, 2, dim)
        pot = AnisotropicHarmonic(kappas)
        Q_theory = dim * kT / kappas.mean().item()
        print(f"  kappas = {[round(k,2) for k in kappas.tolist()]}")
        print(f"  Q_theory (D*kT/<kappa>) = {Q_theory:.4f}")

        Q_final, q_hist, loss_hist = train_Q(
            pot, dim, kappas, n_epochs=60, n_steps=2000, batch_size=4, lr=0.1, Q0=1.0,
        )
        results[f"dim_{dim}"] = {
            "kappas": kappas.tolist(),
            "Q_theory": Q_theory,
            "Q_learned": Q_final,
            "Q_history": q_hist,
            "loss_history": loss_hist,
        }
    return results


def run_e2():
    print("\n" + "=" * 60); print("E2: Learn Q on DoubleWell"); print("=" * 60)
    pot = DoubleWell(barrier_height=2.0)
    kT = 1.0

    # For DoubleWell, true_var = <x^2> under Boltzmann ~ 1.0 for a=2 (wells at +-1)
    # Numerical integration: for U = 2(x^2-1)^2, kT=1
    # <x^2> computed analytically via Gaussian quadrature
    from math import exp
    xs = np.linspace(-3, 3, 2000)
    U = 2*(xs**2 - 1)**2
    w = np.exp(-U/kT)
    true_x2 = float(np.sum(xs**2 * w) / np.sum(w))
    print(f"  true <x^2> = {true_x2:.4f}")

    nh = LogQNoseHoover(kT=kT, mass=1.0, Q=1.0)
    nh.kT.requires_grad_(False); nh.mass.requires_grad_(False)
    optimizer = torch.optim.Adam([nh.log_Q], lr=0.1)
    n_epochs = 60
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    dt = 0.01
    n_steps = 3000
    batch = 8
    true_var = torch.tensor([true_x2])

    q_hist, loss_hist = [], []
    for epoch in range(n_epochs):
        torch.manual_seed(epoch * 13 + 99)
        x0 = torch.ones(batch, 1) * (1.0 if epoch % 2 == 0 else -1.0) + 0.1*torch.randn(batch, 1)
        v0 = torch.randn(batch, 1) * np.sqrt(kT)
        traj_x, _ = nh.run(x0, v0, pot.force, dt=dt, n_steps=n_steps)
        loss = combined_loss(traj_x, true_var, alpha=0.5)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([nh.log_Q], 2.0)
        optimizer.step()
        scheduler.step()

        q_hist.append(nh.Q.item())
        loss_hist.append(loss.item())
        if epoch % 15 == 0 or epoch == n_epochs-1:
            print(f"  epoch {epoch:3d}: Q={nh.Q.item():.5f}, loss={loss.item():.4f}")

    # harmonic local curvature: U''(1)= 16 -> kappa_eff=16; Q~kT/kappa_eff
    Q_harmonic = kT / 16.0
    print(f"  Q_harmonic (local U''(1)=16) = {Q_harmonic:.4f}")
    print(f"  Q_learned = {nh.Q.item():.4f}")
    return {
        "Q_learned": nh.Q.item(),
        "Q_harmonic_equiv": Q_harmonic,
        "Q_history": q_hist,
        "loss_history": loss_hist,
        "true_x2": true_x2,
        "barrier_height": 2.0,
    }


class LearnableFrictionNH(Integrator):
    """Nose-Hoover with learnable friction g(alpha).

    Standard NH uses friction linear in alpha: v <- exp(-alpha*dt/2)*v
    Here we replace alpha with g(alpha) where g is a small MLP.
    Q is also log-parameterized.
    """
    def __init__(self, kT=1.0, Q=1.0, mass=1.0, hidden=16):
        super().__init__()
        self.kT = nn.Parameter(torch.tensor(float(kT)))
        self.mass = nn.Parameter(torch.tensor(float(mass)))
        self.log_Q = nn.Parameter(torch.tensor(float(np.log(Q))))
        self.g_net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Initialize g_net to ~identity via residual
        with torch.no_grad():
            for m in self.g_net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    nn.init.zeros_(m.bias)

    @property
    def Q(self):
        return torch.exp(self.log_Q)

    def g(self, alpha):
        # residual: g(alpha) = alpha + small_mlp(alpha)
        shape = alpha.shape
        delta = self.g_net(alpha.reshape(-1, 1)).reshape(shape)
        return alpha + delta

    def step(self, x, v, alpha, force_fn, dt):
        ndof = x.shape[-1]
        Q = self.Q
        v2 = (v**2).sum(dim=-1)
        alpha = alpha + (dt/4)*(v2 - ndof*self.kT)/Q
        g_alpha = self.g(alpha)
        v = v * torch.exp(-g_alpha.unsqueeze(-1)*dt/2)
        v2 = (v**2).sum(dim=-1)
        alpha = alpha + (dt/4)*(v2 - ndof*self.kT)/Q

        v = v + (dt/2) * force_fn(x) / self.mass
        x = x + dt * v
        v = v + (dt/2) * force_fn(x) / self.mass

        v2 = (v**2).sum(dim=-1)
        alpha = alpha + (dt/4)*(v2 - ndof*self.kT)/Q
        g_alpha = self.g(alpha)
        v = v * torch.exp(-g_alpha.unsqueeze(-1)*dt/2)
        v2 = (v**2).sum(dim=-1)
        alpha = alpha + (dt/4)*(v2 - ndof*self.kT)/Q
        return x, v, alpha

    def run(self, x0, v0, force_fn, dt, n_steps, store_every=1, final_only=False):
        v = v0 if v0 is not None else torch.randn_like(x0)*torch.sqrt(self.kT/self.mass)
        alpha = torch.zeros(x0.shape[:-1], device=x0.device, dtype=x0.dtype)
        state = (x0, v, alpha)
        step_fn = lambda x,v,a: self.step(x,v,a,force_fn,dt)
        return self._integrate(state, step_fn, n_steps, store_every, final_only, store_indices=[0,1])


def run_e3():
    print("\n" + "=" * 60); print("E3: Learn g(xi) as neural network"); print("=" * 60)
    dim = 5
    kappas = torch.logspace(0, 2, dim)
    pot = AnisotropicHarmonic(kappas)
    true_var = 1.0 / kappas
    kT, dt = 1.0, 0.01
    n_steps, n_epochs, batch = 2000, 80, 4

    model = LearnableFrictionNH(kT=kT, Q=0.1, mass=1.0, hidden=16)
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
        torch.manual_seed(epoch*11 + 7)
        x0 = torch.randn(batch, dim) * 0.1
        v0 = torch.randn(batch, dim) * np.sqrt(kT)
        traj_x, _ = model.run(x0, v0, pot.force, dt=dt, n_steps=n_steps)
        loss = combined_loss(traj_x, true_var, alpha=0.5)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()
        q_hist.append(model.Q.item())
        loss_hist.append(loss.item())
        if epoch % 20 == 0 or epoch == n_epochs-1:
            print(f"  epoch {epoch:3d}: Q={model.Q.item():.5f}, loss={loss.item():.4f}")

    xi_range = torch.linspace(-4, 4, 200)
    with torch.no_grad():
        g_learned = model.g(xi_range).numpy()
    return {
        "Q_learned": model.Q.item(),
        "Q_history": q_hist,
        "loss_history": loss_hist,
        "xi_range": xi_range.numpy().tolist(),
        "g_learned": g_learned.tolist(),
        "g_linear": xi_range.numpy().tolist(),
        "g_tanh": np.tanh(xi_range.numpy()).tolist(),
        "dim": dim,
        "kappas": kappas.tolist(),
    }


def compute_ess(traj_x):
    n = traj_x.shape[0]
    traj = traj_x[n // 4:]
    mu = traj.mean(dim=0, keepdim=True)
    x = traj[:-1] - mu
    x1 = traj[1:] - mu
    var = x.pow(2).mean(dim=0) + 1e-10
    rho1 = (x * x1).mean(dim=0) / var
    rho1_mean = rho1.abs().mean().item()
    rho1_mean = min(max(rho1_mean, 0.0), 0.999)
    tau_int = (1 + rho1_mean) / (1 - rho1_mean + 1e-6)
    n_eff = traj.shape[0]
    return n_eff / (2 * max(tau_int, 1.0))


def run_e4(e1_results):
    print("\n" + "=" * 60); print("E4: Summary comparison"); print("=" * 60)
    kT, dt = 1.0, 0.01
    n_steps, n_seeds = 10000, 5
    results = {}
    for dim in [2, 5, 10]:
        print(f"\n--- dim={dim} ---")
        kappas = torch.logspace(0, 2, dim)
        pot = AnisotropicHarmonic(kappas)
        Q_l = e1_results[f"dim_{dim}"]["Q_learned"]
        Q_t = e1_results[f"dim_{dim}"]["Q_theory"]
        configs = {"Q_default": 1.0, "Q_theory": Q_t, "Q_learned": Q_l}
        dim_res = {}
        for lbl, Q_val in configs.items():
            ess_vals = []
            for seed in range(n_seeds):
                torch.manual_seed(seed*31 + 100)
                nh = NoseHoover(kT=kT, Q=Q_val, mass=1.0)
                x0 = torch.randn(1, dim) * 0.1
                v0 = torch.randn(1, dim) * np.sqrt(kT)
                with torch.no_grad():
                    traj_x, _ = nh.run(x0, v0, pot.force, dt=dt, n_steps=n_steps)
                ess_vals.append(compute_ess(traj_x))
            m, s = float(np.mean(ess_vals)), float(np.std(ess_vals))
            dim_res[lbl] = {"mean": m, "std": s, "Q": float(Q_val),
                            "ess_per_force_eval": m / (n_steps*2)}
            print(f"  {lbl} (Q={Q_val:.4f}): ESS = {m:.1f} +/- {s:.1f}")
        results[f"dim_{dim}"] = dim_res
    return results


def main():
    t0 = time.time()
    all_results = {}
    all_results["e1"] = run_e1()
    all_results["e2"] = run_e2()
    all_results["e3"] = run_e3()
    all_results["e4"] = run_e4(all_results["e1"])
    all_results["wall_time_seconds"] = time.time() - t0
    print(f"\nTotal wall time: {all_results['wall_time_seconds']:.1f}s")

    class NpEnc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.integer,)): return int(o)
            return super().default(o)

    with open(f"{OUT_DIR}/results.json", "w") as f:
        json.dump(all_results, f, indent=2, cls=NpEnc)
    print(f"Saved results.json")
    return all_results


if __name__ == "__main__":
    main()
