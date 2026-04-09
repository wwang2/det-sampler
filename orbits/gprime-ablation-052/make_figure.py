"""Consolidated 3-panel figure for gprime-ablation-052."""
import json, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)

mpl.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 220,
})

COLORS = {
    "log-osc":         "#1f77b4",
    "clipped-log-osc": "#9467bd",
    "tanh-scaled":     "#ff7f0e",
    "tanh-ref":        "#2ca02c",
}


def defs():
    def dlogosc(xi):
        return 2*(1-xi**2)/(1+xi**2)**2
    def gclipped(xi):
        return np.where(np.abs(xi) <= 1, 2*xi/(1+xi**2), np.sign(xi))
    def dclipped(xi):
        return np.where(np.abs(xi) <= 1, 2*(1-xi**2)/(1+xi**2)**2, 0.0)
    return {
        "log-osc":         (lambda x: 2*x/(1+x**2), dlogosc),
        "clipped-log-osc": (gclipped, dclipped),
        "tanh-scaled":     (lambda x: 2*np.tanh(x), lambda x: 2/np.cosh(x)**2),
        "tanh-ref":        (lambda x: np.tanh(x),   lambda x: 1/np.cosh(x)**2),
    }


def main():
    with open(os.path.join(OUT, "results.json")) as f:
        res = json.load(f)
    per = res["per_method_per_Q"]
    order = ["log-osc", "clipped-log-osc", "tanh-scaled", "tanh-ref"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.0))
    plt.subplots_adjust(wspace=0.36, left=0.05, right=0.99, top=0.88, bottom=0.18)

    # -------- Panel (a): friction functions g and g' -----------------------
    ax = axes[0]
    xi = np.linspace(-4, 4, 600)
    fns = defs()
    for name in order:
        g, dg = fns[name]
        ax.plot(xi, g(xi), lw=2.2, color=COLORS[name], label=name)
        ax.plot(xi, dg(xi), lw=1.2, color=COLORS[name], ls='--', alpha=0.75)
    ax.axhline(0, color='k', lw=0.5, alpha=0.4)
    ax.axhline(2.0, color='grey', lw=0.5, ls=':', alpha=0.5)
    ax.text(3.9, 2.03, "g'(0)=2", fontsize=9, ha='right', va='bottom', alpha=0.6)
    ax.axhline(1.0, color='grey', lw=0.5, ls=':', alpha=0.5)
    ax.text(3.9, 1.03, "g'(0)=1", fontsize=9, ha='right', va='bottom', alpha=0.6)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$g(\xi)$  solid,    $g'(\xi)$  dashed")
    ax.set_title("(a) Friction functions and derivatives", fontweight='bold')
    ax.set_ylim(-1.5, 2.6)
    ax.legend(frameon=False, loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # -------- Panel (b): tau_int vs Q (the key ablation) -------------------
    ax = axes[1]
    for name in order:
        Qcs = sorted(per[name].keys(), key=lambda s: float(s.split("=")[1]))
        Qs_num = [float(k.split("=")[1]) for k in Qcs]
        meds, q25s, q75s = [], [], []
        for qk in Qcs:
            t = np.array([tt for _, tt in per[name][qk]["taus"]])
            meds.append(np.median(t)); q25s.append(np.percentile(t, 25)); q75s.append(np.percentile(t, 75))
        meds, q25s, q75s = map(np.array, (meds, q25s, q75s))
        ax.plot(Qs_num, meds, 'o-', color=COLORS[name], lw=2, ms=7, label=name)
        ax.fill_between(Qs_num, q25s, q75s, color=COLORS[name], alpha=0.18, lw=0)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r"Thermostat mass center  $Q_c$")
    ax.set_ylabel(r"$\tau_\mathrm{int}(q^2)$  (median across 20 seeds)")
    ax.set_title(r"(b) $\tau_\mathrm{int}$ vs $Q$  ($d\!=\!10$ aniso, $\kappa\!=\!100$)",
                 fontweight='bold')
    ax.axhline(5.0, color='grey', lw=0.8, ls=':')
    ax.text(0.35, 5.5, "Hamiltonian floor", fontsize=9, color='grey')
    ax.legend(frameon=False, loc='lower left', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    # -------- Panel (c): best tau distribution (box) -----------------------
    ax = axes[2]
    # "best ACTIVE" tau: pick Qc minimizing median *subject to median > 10*
    # (below 10 is the Hamiltonian-disabled-thermostat artifact)
    data = []
    best_Qs = []
    for name in order:
        best = None
        for qk, rec in per[name].items():
            t = np.array([tt for _, tt in rec["taus"]])
            med = np.median(t)
            # take min of all medians: tanh-ref's Q=30 gives 5.7 < 10, but that's because
            # tanh-ref's actual best is nearly at the floor; include it honestly.
            if best is None or med < best[0]:
                best = (med, qk, t)
        data.append(best[2]); best_Qs.append(best[1])

    bp = ax.boxplot(data, tick_labels=order, patch_artist=True,
                    widths=0.55, showfliers=True,
                    medianprops=dict(color='black', lw=2.0))
    for patch, name in zip(bp['boxes'], order):
        patch.set_facecolor(COLORS[name]); patch.set_alpha(0.55)
        patch.set_edgecolor(COLORS[name])
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        jitter = rng.normal(0, 0.05, size=len(d))
        ax.scatter(np.full_like(d, i + 1) + jitter, d, s=14,
                   color=COLORS[order[i]], edgecolor='white', lw=0.4, alpha=0.9, zorder=3)
    for i, name in enumerate(order):
        m = np.median(data[i])
        ax.text(i + 1, m * 1.15, f"{m:.1f}\n({best_Qs[i]})", ha='center', va='bottom', fontsize=9)
    ax.set_yscale('log')
    ax.set_ylabel(r"$\tau_\mathrm{int}$  at each method's best $Q_c$")
    ax.set_title(r"(c) Best-$Q$ $\tau_\mathrm{int}$ across 20 seeds", fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(True, which='both', alpha=0.3)

    fig.suptitle(
        r"Ablation: does sign of $g'(\xi)$ drive the tanh-vs-logosc gap?"
        "   (answer: no — clipped-log-osc tracks log-osc at every $Q$)",
        fontsize=14, fontweight='bold', y=1.00)

    fig.savefig(os.path.join(FIG, "ablation.png"), bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)
    print("Saved", os.path.join(FIG, "ablation.png"))


if __name__ == "__main__":
    main()
