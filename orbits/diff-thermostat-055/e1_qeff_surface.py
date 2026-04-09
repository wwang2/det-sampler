#!/usr/bin/env python3
"""E1: Map the optimal Q_eff(kappa, D) surface via differentiable simulation.

Reduced parameters for practical runtime:
- n_steps=2000 (backprop through 2000 steps is feasible)
- batch_size=4 for D>=10
- 60 epochs
"""

import sys
import json
import time
import numpy as np

sys.path.insert(0, "/Users/wujiewang/code/uni-diffsim")

import torch
import torch.nn as nn

from uni_diffsim import NoseHoover

SEED_BASE = 42
OUT_DIR = "/Users/wujiewang/code/det-sampler/.worktrees/diff-thermostat-055/orbits/diff-thermostat-055"


class AnisotropicHarmonic(nn.Module):
    def __init__(self, kappas):
        super().__init__()
        self.kappas = kappas.float()
    def force(self, x):
        return -self.kappas * x


class LogQNoseHoover(NoseHoover):
    def __init__(self, kT=1.0, mass=1.0, Q=1.0):
        super().__init__(kT=kT, mass=mass, Q=Q)
        del self._parameters["Q"]
        self.log_Q = nn.Parameter(torch.tensor(float(np.log(Q))))

    @property
    def Q(self):
        return torch.exp(self.log_Q)


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


def train_Q(pot, dim, kappas, kappa_max, n_epochs=60, n_steps=2000,
            batch_size=4, lr=0.05, kT=1.0, Q0=1.0, seed_offset=42):
    true_var = kT / kappas
    dt = min(0.01, 0.05 / np.sqrt(kappa_max))

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
        if epoch % 15 == 0 or epoch == n_epochs - 1:
            print(f"    epoch {epoch:3d}: Q={nh.Q.item():.5f}, loss={loss.item():.4f}")

    return nh.Q.item(), q_history, loss_history, loss.item()


def run_e1():
    print("=" * 60)
    print("E1: Optimal Q_eff(kappa, D) surface")
    print("=" * 60)

    kT = 1.0
    dims = [2, 5, 10, 20]
    kappa_maxes = [10, 100, 1000]
    results = {}

    for D in dims:
        # Scale batch/steps to keep runtime manageable
        if D <= 5:
            n_steps, batch_size, n_epochs = 2000, 4, 60
        elif D <= 10:
            n_steps, batch_size, n_epochs = 1500, 4, 50
        else:
            n_steps, batch_size, n_epochs = 1000, 2, 40

        for kappa_max in kappa_maxes:
            label = f"D={D}_k={kappa_max}"
            t_cell = time.time()
            print(f"\n--- {label} (steps={n_steps}, batch={batch_size}, epochs={n_epochs}) ---")

            if D == 1:
                kappas = torch.tensor([float(kappa_max)])
            else:
                exponents = torch.linspace(0, 1, D)
                kappas = kappa_max ** exponents

            pot = AnisotropicHarmonic(kappas)
            Q_f1 = 1.0 / np.sqrt(kappa_max)
            Q_DkT = D * kT

            Q_final, q_hist, loss_hist, final_loss = train_Q(
                pot, D, kappas, kappa_max,
                n_epochs=n_epochs, n_steps=n_steps, batch_size=batch_size,
                lr=0.05, kT=kT, Q0=1.0, seed_offset=SEED_BASE
            )

            dt_cell = time.time() - t_cell
            print(f"  kappas: [{kappas[0]:.2f}, {kappas[-1]:.2f}]")
            print(f"  Q_learned={Q_final:.5f}  Q_f1={Q_f1:.5f}  Q_DkT={Q_DkT:.1f}")
            print(f"  final loss={final_loss:.5f}  wall={dt_cell:.1f}s")

            results[label] = {
                "D": D, "kappa_max": kappa_max,
                "kappas": kappas.tolist(),
                "Q_learned": Q_final, "Q_f1": Q_f1, "Q_DkT": Q_DkT,
                "Q_history": q_hist, "loss_history": loss_hist,
                "final_loss": final_loss, "wall_seconds": dt_cell,
            }

    with open(f"{OUT_DIR}/e1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved e1_results.json")
    return results


def plot_e1(results):
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

    dims = [2, 5, 10, 20]
    kappa_maxes = [10, 100, 1000]

    Q_learned = np.zeros((len(dims), len(kappa_maxes)))
    Q_f1 = np.zeros_like(Q_learned)
    Q_DkT = np.zeros_like(Q_learned)

    for i, D in enumerate(dims):
        for j, k in enumerate(kappa_maxes):
            key = f"D={D}_k={k}"
            Q_learned[i, j] = results[key]["Q_learned"]
            Q_f1[i, j] = results[key]["Q_f1"]
            Q_DkT[i, j] = results[key]["Q_DkT"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # (a) Heatmap
    ax = axes[0]
    im = ax.imshow(np.log10(np.clip(Q_learned, 1e-6, None)), aspect='auto',
                   cmap='viridis', origin='lower')
    ax.set_xticks(range(len(kappa_maxes)))
    ax.set_xticklabels([str(k) for k in kappa_maxes])
    ax.set_yticks(range(len(dims)))
    ax.set_yticklabels([str(d) for d in dims])
    ax.set_xlabel(r'$\kappa_{\max}$')
    ax.set_ylabel('D (dimension)')
    ax.set_title(r'(a) $\log_{10}(Q_{\rm learned})$', fontweight='bold')
    for i in range(len(dims)):
        for j in range(len(kappa_maxes)):
            ax.text(j, i, f"{Q_learned[i,j]:.3f}",
                    ha='center', va='center', color='white', fontsize=10, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8)

    # (b) Scatter: learned vs predicted
    ax = axes[1]
    colors_D = {2: '#1f77b4', 5: '#ff7f0e', 10: '#2ca02c', 20: '#d62728'}
    for i, D in enumerate(dims):
        ax.scatter(Q_f1[i, :], Q_learned[i, :], c=colors_D[D], marker='o', s=80,
                   label=f'D={D} vs F1', edgecolors='k', linewidths=0.5)
        ax.scatter(Q_DkT[i, :], Q_learned[i, :], c=colors_D[D], marker='s', s=80,
                   edgecolors='k', linewidths=0.5, alpha=0.5)
    all_q = np.concatenate([Q_learned.ravel(), Q_f1.ravel(), Q_DkT.ravel()])
    pos_q = all_q[all_q > 0]
    if len(pos_q) > 0:
        lims = [min(pos_q)*0.3, max(pos_q)*3]
        ax.plot(lims, lims, 'k--', alpha=0.3, label='1:1')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Analytical Q prediction')
    ax.set_ylabel(r'$Q_{\rm learned}$')
    ax.set_title('(b) Learned vs predicted Q', fontweight='bold')
    ax.legend(fontsize=8, frameon=False, ncol=2)

    # (c) Q vs D at fixed kappa
    ax = axes[2]
    for j, k in enumerate(kappa_maxes):
        ax.plot(dims, Q_learned[:, j], 'o-', label=f'$\\kappa_{{\\max}}$={k}', markersize=8)
    d_arr = np.array(dims, dtype=float)
    ax.plot(d_arr, d_arr * 1.0, 'k--', alpha=0.4, label=r'D$\cdot$kT')
    ax.set_xlabel('Dimension D')
    ax.set_ylabel(r'$Q_{\rm learned}$')
    ax.set_title('(c) Q scaling with dimension', fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=9, frameon=False)

    fig.savefig(f"{OUT_DIR}/figures/e1_qeff_surface.png")
    plt.close(fig)
    print("Saved figures/e1_qeff_surface.png")

    # Loss curves
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    rep_cells = [("D=2_k=100", "D=2, $\\kappa$=100"),
                 ("D=10_k=10", "D=10, $\\kappa$=10"),
                 ("D=10_k=100", "D=10, $\\kappa$=100"),
                 ("D=20_k=1000", "D=20, $\\kappa$=1000")]
    for ax, (key, title) in zip(axes2.ravel(), rep_cells):
        r = results[key]
        ax2b = ax.twinx()
        l1, = ax.plot(r["loss_history"], 'b-', alpha=0.7, label='loss')
        l2, = ax2b.plot(r["Q_history"], 'r-', alpha=0.7, label='Q')
        ax2b.axhline(r["Q_f1"], color='gray', ls='--', alpha=0.5)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss', color='b')
        ax2b.set_ylabel('Q', color='r')
        ax.set_title(title, fontweight='bold')
        ax.legend([l1, l2], ['loss', 'Q'], fontsize=9, frameon=False, loc='upper right')
    fig2.savefig(f"{OUT_DIR}/figures/e1_loss_curves.png")
    plt.close(fig2)
    print("Saved figures/e1_loss_curves.png")


if __name__ == "__main__":
    t0 = time.time()
    results = run_e1()
    plot_e1(results)
    print(f"\nE1 total wall time: {time.time()-t0:.1f}s")
