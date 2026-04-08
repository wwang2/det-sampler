#!/usr/bin/env python3
"""Generate figures from results.json"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

OUT_DIR = "/Users/wujiewang/code/det-sampler/.worktrees/learn-thermostat-050/orbits/learn-thermostat-050"
FIG_DIR = f"{OUT_DIR}/figures"

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

NH_BLUE = "#1f77b4"
NHC_ORANGE = "#ff7f0e"
LEARNED_GREEN = "#2ca02c"
THEORY_RED = "#d62728"
GRAY = "#888888"


def load():
    with open(f"{OUT_DIR}/results.json") as f:
        return json.load(f)


def fig1_Q_learning(results):
    """Q vs epoch for d=2,5,10 with Q_theory marked."""
    e1 = results["e1"]
    dims = [2, 5, 10]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, dim in zip(axes, dims):
        d = e1[f"dim_{dim}"]
        q_hist = d["Q_history"]
        q_theory = d["Q_theory"]
        q_learned = d["Q_learned"]
        epochs = np.arange(len(q_hist))
        ax.plot(epochs, q_hist, color=LEARNED_GREEN, lw=2, label="learned Q")
        ax.axhline(q_theory, color=THEORY_RED, lw=1.5, ls="--",
                   label=f"Q_theory = {q_theory:.3f}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"$Q$")
        ax.set_title(f"d = {dim}  |  Q_learned = {q_learned:.3f}")
        ax.set_yscale("log")
        ax.legend(loc="best", frameon=False)
        ax.grid(alpha=0.3)
    fig.suptitle("E1: Learned Q converges toward analytical Q = D kT / <kappa>",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig1_Q_learning.png", bbox_inches="tight")
    plt.close(fig)
    print("saved fig1_Q_learning.png")


def fig2_Q_nonGaussian(results):
    """DoubleWell: Q history + loss curve."""
    e2 = results["e2"]
    q_hist = e2["Q_history"]
    loss_hist = e2["loss_history"]
    q_learned = e2["Q_learned"]
    q_harm = e2["Q_harmonic_equiv"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = np.arange(len(q_hist))

    ax = axes[0]
    ax.plot(epochs, q_hist, color=LEARNED_GREEN, lw=2, label="learned Q")
    ax.axhline(q_harm, color=THEORY_RED, ls="--", lw=1.5,
               label=f"Q (harmonic approx) = {q_harm:.3f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$Q$")
    ax.set_title(f"DoubleWell (a=2): Q_learned = {q_learned:.3f}")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, loss_hist, color=NH_BLUE, lw=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\tau_{\mathrm{int}}$ proxy loss")
    ax.set_title("Training loss")
    ax.grid(alpha=0.3)

    fig.suptitle("E2: Learned Q on DoubleWell (non-Gaussian)", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig2_Q_nonGaussian.png", bbox_inches="tight")
    plt.close(fig)
    print("saved fig2_Q_nonGaussian.png")


def fig3_learned_friction(results):
    """Learned g(xi) vs tanh vs linear."""
    e3 = results["e3"]
    xi = np.array(e3["xi_range"])
    g_learned = np.array(e3["g_learned"])
    g_linear = np.array(e3["g_linear"])
    g_tanh = np.array(e3["g_tanh"])
    q_hist = e3["Q_history"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(xi, g_linear, color=GRAY, ls=":", lw=1.5, label=r"linear (NH): $g(\xi)=\xi$")
    ax.plot(xi, g_tanh, color=NHC_ORANGE, ls="--", lw=1.5, label=r"$\tanh(\xi)$ (bounded)")
    ax.plot(xi, g_learned, color=LEARNED_GREEN, lw=2.5, label="learned $g(\\xi)$")
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.axvline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel(r"$\xi$ (thermostat variable)")
    ax.set_ylabel(r"$g(\xi)$")
    ax.set_title("Learned friction function")
    ax.legend(frameon=False, loc="best")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(np.arange(len(q_hist)), q_hist, color=LEARNED_GREEN, lw=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$Q$")
    ax.set_title(f"Q_learned = {e3['Q_learned']:.3f}")
    ax.grid(alpha=0.3)

    fig.suptitle("E3: Learnable friction g(xi) on 5D anisotropic Gaussian", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig3_learned_friction.png", bbox_inches="tight")
    plt.close(fig)
    print("saved fig3_learned_friction.png")


def fig4_comparison(results):
    """ESS bar chart across {default, theory, learned} for each dim."""
    e4 = results["e4"]
    dims = [2, 5, 10]
    labels = ["Q_default", "Q_theory", "Q_learned"]
    colors = [GRAY, THEORY_RED, LEARNED_GREEN]
    nice_labels = ["Q = 1 (default)", "Q_theory", "Q_learned"]

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.25
    x = np.arange(len(dims))
    for i, (lbl, col, nice) in enumerate(zip(labels, colors, nice_labels)):
        means = [e4[f"dim_{d}"][lbl]["mean"] for d in dims]
        stds = [e4[f"dim_{d}"][lbl]["std"] for d in dims]
        ax.bar(x + (i - 1) * width, means, width,
               yerr=stds, capsize=4, label=nice, color=col, alpha=0.85,
               edgecolor="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"d={d}" for d in dims])
    ax.set_ylabel("ESS (effective sample size)")
    ax.set_title("E4: ESS comparison  -  learned Q vs theory vs default (Q=1)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, axis="y")

    # Annotate Q values
    for i, d in enumerate(dims):
        dim_res = e4[f"dim_{d}"]
        q_l = dim_res["Q_learned"]["Q"]
        q_t = dim_res["Q_theory"]["Q"]
        ax.annotate(f"Q_t={q_t:.3f}\nQ_l={q_l:.3f}",
                    xy=(i, ax.get_ylim()[1] * 0.02),
                    ha="center", fontsize=8, color="#444")

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig4_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("saved fig4_comparison.png")


def main():
    results = load()
    fig1_Q_learning(results)
    fig2_Q_nonGaussian(results)
    fig3_learned_friction(results)
    fig4_comparison(results)
    print("All figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
