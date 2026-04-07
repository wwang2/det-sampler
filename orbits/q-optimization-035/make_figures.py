"""Make figures from results.json."""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
with open(os.path.join(HERE, "results.json")) as f:
    R = json.load(f)


def fig_q_optimization():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, kr in zip(axes, [10, 100]):
        for offset, N in enumerate([3, 5]):
            key = f"kr{kr}_N{N}"
            d = R["part1"][key]
            opt = np.array(d["best_Qs"])
            logu = np.array(d["loguniform_Qs"])
            y = N + offset * 0.0
            ax.scatter(np.sort(opt), [y + 0.15] * N, marker="o", s=80,
                       color="C0", label="optimized" if offset == 0 else None,
                       edgecolor="k", linewidth=0.5, zorder=3)
            ax.scatter(np.sort(logu), [y - 0.15] * N, marker="s", s=80,
                       color="C1", label="log-uniform" if offset == 0 else None,
                       edgecolor="k", linewidth=0.5, zorder=3)
            ax.text(opt.min() * 0.5, y, f"N={N}", va="center", ha="right",
                    fontsize=9)
            ax.text(opt.max() * 2.0, y + 0.15,
                    f"τ={d['best_tau']:.0f}", fontsize=8, color="C0", va="center")
            ax.text(opt.max() * 2.0, y - 0.15,
                    f"τ={d['loguniform_tau']:.0f}", fontsize=8, color="C1", va="center")
        ax.set_xscale("log")
        ax.set_xlabel("Q")
        ax.set_title(f"κ_ratio = {kr}")
        ax.set_yticks([])
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="upper left", framealpha=0.9, fontsize=9)
    fig.suptitle("Optimized Q values vs log-uniform reference", fontsize=12)
    fig.tight_layout()
    out = os.path.join(HERE, "figures", "q_optimization.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def fig_nhc_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel a/b: Gaussian tau vs N for kr in {10, 100}
    for ax, kr in zip(axes[:2], [10, 100]):
        gd = R["part2"]["gaussian"][f"kr{kr}"]
        Ns = [3, 5]
        par_m = [gd[f"parallel_N{N}"]["mean"] for N in Ns]
        par_s = [gd[f"parallel_N{N}"]["std"] for N in Ns]
        nhc_m = [gd[f"nhc_M{N}"]["mean"] for N in Ns]
        nhc_s = [gd[f"nhc_M{N}"]["std"] for N in Ns]
        x = np.array(Ns)
        ax.errorbar(x - 0.08, par_m, yerr=par_s, fmt="o-", color="C0",
                    label="parallel log-osc", capsize=3)
        ax.errorbar(x + 0.08, nhc_m, yerr=nhc_s, fmt="s-", color="C3",
                    label="NHC", capsize=3)
        ax.set_xlabel("# thermostats (N or M)")
        ax.set_ylabel(r"$\tau_{int}$ on $q^2$")
        ax.set_xticks(Ns)
        ax.set_title(f"5D Gaussian, κ_ratio={kr}")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    # Panel c: GMM mode crossings
    md = R["part2"]["mixture"]
    Ns = [3, 5]
    par_m = [md[f"parallel_N{N}"]["mean"] for N in Ns]
    par_s = [md[f"parallel_N{N}"]["std"] for N in Ns]
    nhc_m = [md[f"nhc_M{N}"]["mean"] for N in Ns]
    nhc_s = [md[f"nhc_M{N}"]["std"] for N in Ns]
    x = np.array(Ns)
    ax = axes[2]
    ax.errorbar(x - 0.08, par_m, yerr=par_s, fmt="o-", color="C0",
                label="parallel log-osc", capsize=3)
    ax.errorbar(x + 0.08, nhc_m, yerr=nhc_s, fmt="s-", color="C3",
                label="NHC", capsize=3)
    ax.set_xlabel("# thermostats")
    ax.set_ylabel("mode crossings / 200k force evals")
    ax.set_xticks(Ns)
    ax.set_title("2D GMM (5 modes)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle("Parallel multi-scale log-osc vs Nose-Hoover Chain "
                 "(equal thermostat count)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(HERE, "figures", "nhc_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_q_optimization()
    fig_nhc_comparison()
