"""Regenerate figures from results.json."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "results.json")) as f:
    results = json.load(f)

fig_dir = os.path.join(HERE, "figures")
os.makedirs(fig_dir, exist_ok=True)

COLOR_NHC = "#ff7f0e"
COLOR_PAR = "#2ca02c"
COLOR_LANG = "#9467bd"
FS_L = 14; FS_T = 12; FS_TT = 16

def get_metric(d, metric, default_mean=0, default_std=0):
    """Safely extract mean/std from nested dict."""
    if metric not in d:
        return default_mean, default_std
    m = d[metric]
    return m.get("mean", default_mean), m.get("std", default_std)

# ---- Fig 1: Barrier sweep ----
exp1 = results["exp1"]
barriers = sorted([float(k) for k in exp1.keys()])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for label, color, marker, key in [("Multi-scale log-osc (N=5)", COLOR_PAR, "o", "parallel"),
                                   ("NHC (M=5, best Q)", COLOR_NHC, "s", "nhc"),
                                   ("Langevin (best gamma)", COLOR_LANG, "^", "langevin")]:
    means, stds = [], []
    for a in barriers:
        m, s = get_metric(exp1[str(a)][key], "barrier_crossings")
        means.append(m); stds.append(s)
    ax1.errorbar(barriers, means, yerr=stds, marker=marker, capsize=4,
                 color=color, label=label, linewidth=2)
ax1.set_xlabel("Barrier height a", fontsize=FS_L)
ax1.set_ylabel("Barrier crossings (200k steps)", fontsize=FS_L)
ax1.set_title("(a) Barrier crossings vs height", fontsize=FS_TT)
ax1.legend(fontsize=11); ax1.tick_params(labelsize=FS_T)
ax1.set_yscale("log"); ax1.set_xscale("log")

par_m = [get_metric(exp1[str(a)]["parallel"], "barrier_crossings")[0] for a in barriers]
nhc_m = [get_metric(exp1[str(a)]["nhc"], "barrier_crossings")[0] for a in barriers]
lang_m = [get_metric(exp1[str(a)]["langevin"], "barrier_crossings")[0] for a in barriers]
ratio_nhc = [p / max(n, 1) for p, n in zip(par_m, nhc_m)]
ratio_lang = [p / max(l, 1) for p, l in zip(par_m, lang_m)]
ax2.plot(barriers, ratio_nhc, marker="s", color=COLOR_NHC, label="vs NHC", linewidth=2)
ax2.plot(barriers, ratio_lang, marker="^", color=COLOR_LANG, label="vs Langevin", linewidth=2)
ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1)
ax2.set_xlabel("Barrier height a", fontsize=FS_L)
ax2.set_ylabel("Crossings ratio (ours / baseline)", fontsize=FS_L)
ax2.set_title("(b) Advantage ratio", fontsize=FS_TT)
ax2.legend(fontsize=11); ax2.tick_params(labelsize=FS_T); ax2.set_xscale("log")
plt.tight_layout()
fig.savefig(os.path.join(fig_dir, "fig1_barrier_sweep.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved fig1_barrier_sweep.png")

# ---- Fig 2: Mode count sweep ----
exp2 = results["exp2"]
mcs = sorted([int(k) for k in exp2.keys()])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for label, color, marker, key in [("Multi-scale log-osc", COLOR_PAR, "o", "parallel"),
                                   ("NHC (best Q)", COLOR_NHC, "s", "nhc"),
                                   ("Langevin (best gamma)", COLOR_LANG, "^", "langevin")]:
    cr_means, cr_stds, vis_means, vis_stds = [], [], [], []
    for n in mcs:
        m, s = get_metric(exp2[str(n)][key], "mode_crossings")
        cr_means.append(m); cr_stds.append(s)
        m, s = get_metric(exp2[str(n)][key], "modes_visited")
        vis_means.append(m); vis_stds.append(s)
    ax1.errorbar(mcs, cr_means, yerr=cr_stds, marker=marker, capsize=4,
                 color=color, label=label, linewidth=2)
    ax2.errorbar(mcs, vis_means, yerr=vis_stds, marker=marker, capsize=4,
                 color=color, label=label, linewidth=2)
ax1.set_xlabel("Number of modes", fontsize=FS_L)
ax1.set_ylabel("Mode crossings (200k steps)", fontsize=FS_L)
ax1.set_title("(a) Total mode crossings", fontsize=FS_TT)
ax1.legend(fontsize=11); ax1.tick_params(labelsize=FS_T)
ax2.set_xlabel("Number of modes", fontsize=FS_L)
ax2.set_ylabel("Fraction of modes visited", fontsize=FS_L)
ax2.set_title("(b) Mode coverage", fontsize=FS_TT)
ax2.legend(fontsize=11); ax2.tick_params(labelsize=FS_T); ax2.set_ylim(0, 1.1)
plt.tight_layout()
fig.savefig(os.path.join(fig_dir, "fig2_mode_count_sweep.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved fig2_mode_count_sweep.png")

# ---- Fig 3: High-dim ----
exp3 = results["exp3"]
dims = sorted([int(k) for k in exp3.keys()])
fig, axes = plt.subplots(1, len(dims), figsize=(6 * len(dims), 5), sharey=True)
if len(dims) == 1:
    axes = [axes]
for i, dim in enumerate(dims):
    ax = axes[i]
    data = exp3[str(dim)]
    par_cross, par_cross_s = get_metric(data["parallel"], "mode_crossings")
    nhc_cross, nhc_cross_s = get_metric(data["nhc"], "mode_crossings")
    means = [par_cross, nhc_cross]
    stds = [par_cross_s, nhc_cross_s]
    x = np.arange(2)
    ax.bar(x, means, width=0.5, yerr=stds, capsize=5,
           color=[COLOR_PAR, COLOR_NHC], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(["Log-osc\n(N=5)", "NHC\n(M=5)"], fontsize=FS_T)
    ax.set_title(f"({chr(97+i)}) {dim}D, 5 modes", fontsize=FS_TT)
    ax.tick_params(labelsize=FS_T)
    par_mv = get_metric(data["parallel"], "modes_visited")[0]
    nhc_mv = get_metric(data["nhc"], "modes_visited")[0]
    ymax = max(max(means), 1)
    ax.text(0, ymax * 0.15, f"visited:\n{par_mv:.0%}", ha="center",
            fontsize=11, fontweight="bold", color="white")
    ax.text(1, ymax * 0.15, f"visited:\n{nhc_mv:.0%}", ha="center",
            fontsize=11, fontweight="bold", color="white")
axes[0].set_ylabel("Mode crossings (200k steps)", fontsize=FS_L)
plt.tight_layout()
fig.savefig(os.path.join(fig_dir, "fig3_high_dim.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved fig3_high_dim.png")

# ---- Fig 4: Summary table ----
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis("off")
col_labels = ["Experiment", "Setting", "Log-osc (ours)", "NHC (best)", "Langevin (best)", "Ratio (ours/NHC)"]
table_data = []
for a in [1.0, 4.0, 8.0]:
    key = str(float(a))
    if key in exp1:
        d = exp1[key]
        p = get_metric(d["parallel"], "barrier_crossings")[0]
        n = get_metric(d["nhc"], "barrier_crossings")[0]
        l = get_metric(d["langevin"], "barrier_crossings")[0]
        ratio = p / max(n, 1)
        table_data.append(["Barrier sweep", f"a={a}", f"{p:.0f}", f"{n:.0f}", f"{l:.0f}", f"{ratio:.2f}x"])
for nm in [5, 10, 20]:
    key = str(nm)
    if key in exp2:
        d = exp2[key]
        p = get_metric(d["parallel"], "mode_crossings")[0]
        n = get_metric(d["nhc"], "mode_crossings")[0]
        l = get_metric(d["langevin"], "mode_crossings")[0]
        ratio = p / max(n, 1)
        table_data.append(["Mode sweep", f"n={nm}", f"{p:.0f}", f"{n:.0f}", f"{l:.0f}", f"{ratio:.2f}x"])
for dim in dims:
    key = str(dim)
    if key in exp3:
        d = exp3[key]
        p = get_metric(d["parallel"], "mode_crossings")[0]
        n = get_metric(d["nhc"], "mode_crossings")[0]
        ratio = p / max(n, 1)
        table_data.append(["High-D", f"d={dim}", f"{p:.0f}", f"{n:.0f}", "---", f"{ratio:.2f}x"])
table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.0, 1.6)
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#d4e6f1")
    table[0, j].set_text_props(fontweight="bold")
for i in range(1, len(table_data) + 1):
    try:
        val = float(table_data[i-1][5].replace("x", ""))
        if val > 1.5:
            table[i, 5].set_facecolor("#d5f5e3")
        elif val < 0.8:
            table[i, 5].set_facecolor("#fadbd8")
    except:
        pass
ax.set_title("Summary: Mode-Hopping Benchmark", fontsize=FS_TT, pad=20)
plt.tight_layout()
fig.savefig(os.path.join(fig_dir, "fig4_summary_table.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved fig4_summary_table.png")
print("All figures done.")
