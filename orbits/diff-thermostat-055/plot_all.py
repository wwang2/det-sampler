#!/usr/bin/env python3
"""Plot all figures for E1, E2, E3. Run after experiments complete."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

OUT_DIR = "/Users/wujiewang/code/det-sampler/.worktrees/diff-thermostat-055/orbits/diff-thermostat-055"

mpl.rcParams.update({
    'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.pad_inches': 0.2,
})


def plot_e1():
    with open(f"{OUT_DIR}/e1_results.json") as f:
        results = json.load(f)

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

    # (b) Scatter: learned vs F1
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

    fig.savefig(f"{OUT_DIR}/figures/e1_qeff_surface.png", bbox_inches='tight')
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
    fig2.savefig(f"{OUT_DIR}/figures/e1_loss_curves.png", bbox_inches='tight')
    plt.close(fig2)
    print("Saved figures/e1_loss_curves.png")


def plot_e2():
    with open(f"{OUT_DIR}/e2_results.json") as f:
        results = json.load(f)

    xi = np.array(results["xi_range"])
    seeds = [k for k in results.keys() if k.startswith("seed_")]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # (a) Learned g(xi)
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(seeds)))
    g_prime_vals = []
    for i, sk in enumerate(seeds):
        r = results[sk]
        g_vals = np.array(r["g_vals"])
        ax.plot(xi, g_vals, color=colors[i], alpha=0.8,
                label=f'seed {r["seed"]}: g\'(0)={r["g_prime_0"]:.2f}')
        g_prime_vals.append(r["g_prime_0"])
    ax.plot(xi, xi, 'k--', alpha=0.4, label=r'$g(\xi)=\xi$ (linear NH)')
    ax.plot(xi, np.tanh(xi), 'k:', alpha=0.4, label=r'$g(\xi)=\tanh(\xi)$')
    ax.set_xlabel(r'$\xi$')
    ax.set_ylabel(r'$g(\xi)$')
    ax.set_title(r"(a) Learned $g(\xi)$ (5 seeds)", fontweight='bold')
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlim(-4, 4)
    ax.axhline(0, color='gray', alpha=0.2)
    ax.axvline(0, color='gray', alpha=0.2)

    # (b) g'(0) and Q across seeds
    ax = axes[1]
    Q_vals = [results[sk]["Q_learned"] for sk in seeds]
    x_pos = np.arange(len(seeds))
    w = 0.35
    ax.bar(x_pos - w/2, g_prime_vals, w, color='steelblue', alpha=0.7, label="g'(0)")
    ax.bar(x_pos + w/2, Q_vals, w, color='coral', alpha=0.7, label="Q")
    ax.axhline(np.mean(g_prime_vals), color='steelblue', ls='--',
               label=f"mean g'(0)={np.mean(g_prime_vals):.3f}")
    ax.axhline(np.mean(Q_vals), color='coral', ls='--',
               label=f"mean Q={np.mean(Q_vals):.3f}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f's{results[sk]["seed"]}' for sk in seeds], fontsize=10)
    ax.set_ylabel("Value")
    ax.set_title("(b) g'(0) and Q across seeds", fontweight='bold')
    ax.legend(fontsize=9, frameon=False)

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

    fig.savefig(f"{OUT_DIR}/figures/e2_learned_g.png", bbox_inches='tight')
    plt.close(fig)
    print("Saved figures/e2_learned_g.png")


def plot_e3():
    with open(f"{OUT_DIR}/e3_results.json") as f:
        results = json.load(f)

    Q_min_f1 = results["Q_min_f1"]
    Q_max_f1 = results["Q_max_f1"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    colors = plt.cm.tab10(np.linspace(0, 0.5, 5))
    seeds = [42, 123, 256, 789, 1024]

    # (a) N=3
    ax = axes[0]
    for i, seed in enumerate(seeds):
        key = f"N=3_seed={seed}"
        if key not in results:
            continue
        qs = sorted(results[key]["Qs_learned"])
        ax.semilogy(range(len(qs)), qs, 'o-', color=colors[i],
                    alpha=0.7, markersize=8, label=f's{seed}')
    f1_qs_3 = np.logspace(np.log10(Q_min_f1), np.log10(Q_max_f1), 3)
    ax.semilogy(range(3), f1_qs_3, 'kx--', markersize=10, label='F1 log-uniform')
    ax.axhline(Q_min_f1, color='gray', ls='--', alpha=0.3)
    ax.axhline(Q_max_f1, color='gray', ls=':', alpha=0.3)
    ax.set_xlabel('Thermostat index k (sorted)')
    ax.set_ylabel(r'$Q_k$')
    ax.set_title('(a) Learned Q distribution (N=3)', fontweight='bold')
    ax.legend(fontsize=8, frameon=False)

    # (b) N=5
    ax = axes[1]
    for i, seed in enumerate(seeds):
        key = f"N=5_seed={seed}"
        if key not in results:
            continue
        qs = sorted(results[key]["Qs_learned"])
        ax.semilogy(range(len(qs)), qs, 's-', color=colors[i],
                    alpha=0.7, markersize=8, label=f's{seed}')
    f1_qs_5 = np.logspace(np.log10(Q_min_f1), np.log10(Q_max_f1), 5)
    ax.semilogy(range(5), f1_qs_5, 'kx--', markersize=10, label='F1 log-uniform')
    ax.axhline(Q_min_f1, color='gray', ls='--', alpha=0.3)
    ax.axhline(Q_max_f1, color='gray', ls=':', alpha=0.3)
    ax.set_xlabel('Thermostat index k (sorted)')
    ax.set_ylabel(r'$Q_k$')
    ax.set_title('(b) Learned Q distribution (N=5)', fontweight='bold')
    ax.legend(fontsize=8, frameon=False)

    # (c) Loss comparison
    ax = axes[2]
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

    fig.savefig(f"{OUT_DIR}/figures/e3_learned_Q_distribution.png", bbox_inches='tight')
    plt.close(fig)
    print("Saved figures/e3_learned_Q_distribution.png")


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("e1", "all"):
        plot_e1()
    if which in ("e2", "all"):
        plot_e2()
    if which in ("e3", "all"):
        plot_e3()
