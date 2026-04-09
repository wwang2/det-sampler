"""Consolidated 3-panel figure for gprime-ablation-052."""
import json, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)

mpl.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
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

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.0), constrained_layout=True)

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
    ax.text(3.9, 2.03, "g'(0)=2", fontsize=10, ha='right', va='bottom', alpha=0.6)
    ax.axhline(1.0, color='grey', lw=0.5, ls=':', alpha=0.5)
    ax.text(3.9, 1.03, "g'(0)=1", fontsize=10, ha='right', va='bottom', alpha=0.6)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$g(\xi)$  solid,    $g'(\xi)$  dashed")
    ax.set_title("(a) Friction functions and derivatives", fontweight='bold')
    ax.set_ylim(-1.5, 2.6)
    ax.legend(frameon=False, loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    # -------- Panel (b): tau_int vs Q (the key ablation) -------------------
    ax = axes[1]
    LSTYLES = {"log-osc": "-", "clipped-log-osc": "--", "tanh-scaled": "-", "tanh-ref": "-"}
    MARKERS = {"log-osc": "o", "clipped-log-osc": "s", "tanh-scaled": "o", "tanh-ref": "o"}
    for name in order:
        Qcs = sorted(per[name].keys(), key=lambda s: float(s.split("=")[1]))
        Qs_num = [float(k.split("=")[1]) for k in Qcs]
        meds, q25s, q75s = [], [], []
        for qk in Qcs:
            t = np.array([tt for _, tt in per[name][qk]["taus"]])
            meds.append(np.median(t)); q25s.append(np.percentile(t, 25)); q75s.append(np.percentile(t, 75))
        meds, q25s, q75s = map(np.array, (meds, q25s, q75s))
        ax.plot(Qs_num, meds, marker=MARKERS[name], ls=LSTYLES[name],
                color=COLORS[name], lw=2, ms=7, label=name)
        ax.fill_between(Qs_num, q25s, q75s, color=COLORS[name], alpha=0.18, lw=0)
    # Annotate overlap between log-osc and clipped-log-osc
    ax.annotate("log-osc \u2248 clipped", xy=(30, 16), fontsize=10,
                color='#555555', ha='center', va='bottom')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r"Thermostat mass center  $Q_c$")
    ax.set_ylabel(r"$\tau_\mathrm{int}(q^2)$  (median, 20 seeds)")
    ax.set_title(r"(b) $\tau_\mathrm{int}$ vs $Q$  ($d\!=\!10$, $\kappa\!=\!100$)",
                 fontweight='bold')
    ax.axhline(5.0, color='grey', lw=0.8, ls=':')
    ax.text(0.35, 5.5, "Hamiltonian floor", fontsize=10, color='grey')
    ax.legend(frameon=False, loc='lower left', fontsize=11)
    ax.grid(True, which='both', alpha=0.3)

    # -------- Panel (c): grouped box plots at Q=10 and Q=30 ----------------
    ax = axes[2]
    show_Qs = ["Q=10.0", "Q=30.0"]
    show_labels = ["$Q_c = 10$", "$Q_c = 30$"]
    n_methods = len(order)
    width = 0.6
    rng = np.random.default_rng(0)
    gap = n_methods + 1.5  # spacing between groups
    for gi, qk in enumerate(show_Qs):
        for mi, name in enumerate(order):
            pos = gi * gap + mi
            t = np.array([tt for _, tt in per[name][qk]["taus"]])
            bp = ax.boxplot([t], positions=[pos], widths=width, patch_artist=True,
                            showfliers=False, medianprops=dict(color='black', lw=2.0))
            bp['boxes'][0].set_facecolor(COLORS[name])
            bp['boxes'][0].set_alpha(0.55)
            bp['boxes'][0].set_edgecolor(COLORS[name])
            jitter = rng.normal(0, 0.05, size=len(t))
            ax.scatter(np.full_like(t, pos) + jitter, t, s=12,
                       color=COLORS[name], edgecolor='white', lw=0.3, alpha=0.85, zorder=3)
            med = np.median(t)
            ax.text(pos, max(med * 1.12, med + 5), f"{med:.0f}", ha='center',
                    va='bottom', fontsize=9, color=COLORS[name])
    # Group labels
    for gi, label in enumerate(show_labels):
        center = gi * gap + (n_methods - 1) / 2
        ax.text(center, -0.12, label, ha='center', va='top', fontsize=13,
                fontweight='bold', transform=ax.get_xaxis_transform())
    all_pos = [gi * gap + mi for gi in range(len(show_Qs)) for mi in range(n_methods)]
    ax.set_xticks(all_pos)
    short_names = ["log", "clip", "tanh2", "tanh1"]
    ax.set_xticklabels(short_names * len(show_Qs), fontsize=10)
    ax.set_yscale('log')
    ax.set_ylabel(r"$\tau_\mathrm{int}$ (20 seeds)")
    ax.set_title(r"(c) $\tau_\mathrm{int}$ at thermostat-active $Q$", fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)

    fig.suptitle(
        r"Ablation: does sign of $g'(\xi)$ drive the tanh-vs-logosc gap?"
        r"   (no — clipped tracks log-osc at every $Q$)",
        fontsize=15, fontweight='bold')

    fig.savefig(os.path.join(FIG, "ablation.png"), pad_inches=0.25)
    plt.close(fig)
    print("Saved", os.path.join(FIG, "ablation.png"))


if __name__ == "__main__":
    main()
