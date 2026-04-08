"""Generate the definitive combined summary figure."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(OUT_DIR, "figures")

# Load results
with open(os.path.join(OUT_DIR, "results.json")) as f:
    v1 = json.load(f)
with open(os.path.join(OUT_DIR, "results_v2.json")) as f:
    v2 = json.load(f)

# Method display names and colors
METHODS = [
    ("LogOsc-3", "LogOsc-3\n(auto Q)", "#2ca02c"),
    ("LogOsc-5", "LogOsc-5\n(auto Q)", "#d62728"),
    ("LogOsc-3t", "LogOsc-3\n(tuned Q)", "#98df8a"),
    ("LogOsc-5t", "LogOsc-5\n(tuned Q)", "#ff9896"),
    ("NHC-3", "NHC-3", "#ff7f0e"),
    ("NHC-5", "NHC-5", "#e377c2"),
    ("Langevin", "Langevin", "#7f7f7f"),
    ("NH-1", "NH-1", "#1f77b4"),
]

TARGETS_TAU = [
    ("1d_harmonic", "1D Harmonic"),
    ("2d_double_well", "2D DoubleWell"),
    ("5d_aniso_gauss", "5D Aniso Gauss"),
    ("10d_aniso_gauss", "10D Aniso Gauss"),
]
TARGETS_MC = [
    ("2d_gmm", "2D GMM"),
    ("10d_gmm", "10D GMM"),
]

def get_val(method_key, target, metric="tau"):
    """Get value for method/target combo."""
    if method_key.endswith("t"):
        # Tuned from v2
        base = method_key[:-1]
        N = int(base[-1])
        key2 = f"{base}-tuned"
        if target in v2["summary"] and key2 in v2["summary"][target]:
            return v2["summary"][target][key2]["mean"], v2["summary"][target][key2]["std"]
    else:
        # From v1
        if target in v1["summary"] and method_key in v1["summary"][target]:
            e = v1["summary"][target][method_key]
            if metric == "crossings":
                return e["mode_crossings_mean"], e["mode_crossings_std"]
            else:
                return e["tau_mean"], e["tau_std"]
    return None, None

# =====================================================================
# Figure 1: tau_int bar chart (4 targets x 8 methods)
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (tname, tlabel) in enumerate(TARGETS_TAU):
    ax = axes[idx]
    x = np.arange(len(METHODS))
    vals = []
    errs = []
    colors = []
    labels = []
    for mk, ml, mc in METHODS:
        v, e = get_val(mk, tname, "tau")
        vals.append(v if v is not None else 0)
        errs.append(e if e is not None else 0)
        colors.append(mc)
        labels.append(ml)
    
    # Cap values for display
    max_display = max(v for v in vals if v < 1e5) * 1.5 if any(v < 1e5 for v in vals) else 100
    vals_display = [min(v, max_display) for v in vals]
    
    bars = ax.bar(x, vals_display, yerr=[min(e, max_display*0.3) for e in errs],
                  color=colors, capsize=3, edgecolor="k", linewidth=0.4)
    
    # Add value labels on bars
    for i, (v, bar) in enumerate(zip(vals, bars)):
        if v > max_display * 0.9:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5,
                    f"{v:.0f}" if v > 100 else f"{v:.1f}",
                    ha="center", va="center", fontsize=7, fontweight="bold", color="white")
        else:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_display*0.02,
                    f"{v:.1f}" if v < 100 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel(r"$\tau_{\mathrm{int}}$ (lower = better)", fontsize=11)
    ax.set_title(tlabel, fontsize=13, fontweight="bold")
    ax.set_ylim(0, max_display)
    ax.text(-0.06, 1.05, f"({'abcd'[idx]})", transform=ax.transAxes,
            fontsize=14, fontweight="bold")

fig.suptitle(r"Autocorrelation Time $\tau_{\mathrm{int}}$ — All Methods (tuned)",
             fontsize=15, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_tau_all_methods.png"), dpi=250,
            bbox_inches="tight")
plt.close(fig)
print("Saved fig_tau_all_methods.png")

# =====================================================================
# Figure 2: Mode crossings bar chart (2 targets)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (tname, tlabel) in enumerate(TARGETS_MC):
    ax = axes[idx]
    x = np.arange(len(METHODS))
    vals = []
    errs = []
    colors = []
    labels = []
    for mk, ml, mc in METHODS:
        v, e = get_val(mk, tname, "crossings")
        vals.append(v if v is not None else 0)
        errs.append(e if e is not None else 0)
        colors.append(mc)
        labels.append(ml)
    
    bars = ax.bar(x, vals, yerr=errs, color=colors, capsize=3,
                  edgecolor="k", linewidth=0.4)
    
    for i, (v, bar) in enumerate(zip(vals, bars)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Mode crossings (higher = better)", fontsize=11)
    ax.set_title(tlabel, fontsize=13, fontweight="bold")
    ax.text(-0.06, 1.05, f"({'ab'[idx]})", transform=ax.transAxes,
            fontsize=14, fontweight="bold")

fig.suptitle("Mode Exploration — All Methods (tuned)", fontsize=15, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_crossings_all_methods.png"), dpi=250,
            bbox_inches="tight")
plt.close(fig)
print("Saved fig_crossings_all_methods.png")

# =====================================================================
# Figure 3: Summary table (updated with tuned LogOsc)
# =====================================================================
METHOD_ORDER = ["LogOsc-3\n(auto)", "LogOsc-5\n(auto)", "LogOsc-3\n(tuned)", 
                "LogOsc-5\n(tuned)", "NHC-3", "NHC-5", "Langevin", "NH-1"]
METHOD_KEYS = ["LogOsc-3", "LogOsc-5", "LogOsc-3t", "LogOsc-5t", 
               "NHC-3", "NHC-5", "Langevin", "NH-1"]
TARGET_ORDER = ["1d_harmonic", "2d_double_well", "2d_gmm",
                "5d_aniso_gauss", "10d_aniso_gauss", "10d_gmm"]
TARGET_LABELS = ["1D Harm.", "2D DblWell", "2D GMM", "5D Aniso",
                 "10D Aniso", "10D GMM"]
MULTIMODAL = {"2d_gmm", "10d_gmm"}

cell_text = []
cell_colors = []

for tname, tlabel in zip(TARGET_ORDER, TARGET_LABELS):
    row_text = []
    row_colors = []
    is_mm = tname in MULTIMODAL
    
    # Collect all values for this target
    all_vals = {}
    for mk in METHOD_KEYS:
        metric = "crossings" if is_mm else "tau"
        v, e = get_val(mk, tname, metric)
        if v is not None:
            all_vals[mk] = v
    
    # Find best
    good = {k: v for k, v in all_vals.items() if v < 1e5}
    best_val = None
    if good:
        best_val = max(good.values()) if is_mm else min(good.values())
    
    for mk in METHOD_KEYS:
        v, e = get_val(mk, tname, "crossings" if is_mm else "tau")
        if v is not None:
            if is_mm:
                text = f"{v:.0f}\n+/-{e:.0f}"
                is_best = (best_val is not None and v >= best_val * 0.95)
            else:
                if v > 500:
                    text = f"{v:.0f}"
                elif v > 100:
                    text = f"{v:.0f}\n+/-{e:.0f}"
                else:
                    text = f"{v:.1f}\n+/-{e:.1f}"
                is_best = (best_val is not None and v <= best_val * 1.05 and v < 1e5)
            
            if v > 500 and not is_mm:
                color = "#ffcccc"
            elif is_best:
                color = "#ccffcc"
            else:
                color = "white"
        else:
            text = "--"
            color = "#f0f0f0"
        
        row_text.append(text)
        row_colors.append(color)
    
    cell_text.append(row_text)
    cell_colors.append(row_colors)

fig, ax = plt.subplots(figsize=(16, 5.5))
ax.axis("off")

table = ax.table(
    cellText=cell_text,
    cellColours=cell_colors,
    rowLabels=TARGET_LABELS,
    colLabels=METHOD_ORDER,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 2.2)

for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_text_props(fontweight="bold", fontsize=9)
        cell.set_facecolor("#e0e0e0")
    if j == -1:
        cell.set_text_props(fontweight="bold", fontsize=10)
        cell.set_facecolor("#e0e0e0")

ax.set_title(
    r"Comprehensive Benchmark: $\tau_{\mathrm{int}}$ (unimodal, lower=better) / "
    "mode crossings (multimodal, higher=better)\n"
    "Green = best (within 5%); Red tint = high autocorrelation (>500)",
    fontsize=12, pad=20)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_summary_table_v2.png"), dpi=250,
            bbox_inches="tight")
plt.close(fig)
print("Saved fig_summary_table_v2.png")

# =====================================================================
# Figure 4: Q-tuning importance (before/after for LogOsc-5)
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel a: tau_int before/after for anisotropic targets
targets_aniso = [("5d_aniso_gauss", "5D"), ("10d_aniso_gauss", "10D")]
ax = axes[0]
x = np.arange(len(targets_aniso))
w = 0.25
before = [get_val("LogOsc-5", t, "tau")[0] for t, _ in targets_aniso]
after = [get_val("LogOsc-5t", t, "tau")[0] for t, _ in targets_aniso]
nhc5 = [get_val("NHC-5", t, "tau")[0] for t, _ in targets_aniso]
nh1 = [get_val("NH-1", t, "tau")[0] for t, _ in targets_aniso]

ax.bar(x - 1.5*w, before, w, label="LogOsc-5 (auto Q)", color="#d62728", alpha=0.7)
ax.bar(x - 0.5*w, after, w, label="LogOsc-5 (tuned Q)", color="#ff9896")
ax.bar(x + 0.5*w, nhc5, w, label="NHC-5", color="#e377c2")
ax.bar(x + 1.5*w, nh1, w, label="NH-1", color="#1f77b4")
ax.set_xticks(x)
ax.set_xticklabels([l for _, l in targets_aniso])
ax.set_ylabel(r"$\tau_{\mathrm{int}}$", fontsize=13)
ax.set_title("Anisotropic Gaussian", fontsize=13)
ax.legend(fontsize=8, loc="upper left")
ax.set_yscale("log")
ax.text(-0.08, 1.05, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold")

# Panel b: mode crossings before/after for GMMs
targets_mm = [("2d_gmm", "2D GMM"), ("10d_gmm", "10D GMM")]
ax = axes[1]
x = np.arange(len(targets_mm))
before = [get_val("LogOsc-5", t, "crossings")[0] or 0 for t, _ in targets_mm]
after = [get_val("LogOsc-5t", t, "crossings")[0] or 0 for t, _ in targets_mm]
nhc5 = [get_val("NHC-5", t, "crossings")[0] or 0 for t, _ in targets_mm]
nh1 = [get_val("NH-1", t, "crossings")[0] or 0 for t, _ in targets_mm]

ax.bar(x - 1.5*w, before, w, label="LogOsc-5 (auto Q)", color="#d62728", alpha=0.7)
ax.bar(x - 0.5*w, after, w, label="LogOsc-5 (tuned Q)", color="#ff9896")
ax.bar(x + 0.5*w, nhc5, w, label="NHC-5", color="#e377c2")
ax.bar(x + 1.5*w, nh1, w, label="NH-1", color="#1f77b4")
ax.set_xticks(x)
ax.set_xticklabels([l for _, l in targets_mm])
ax.set_ylabel("Mode crossings", fontsize=13)
ax.set_title("Gaussian Mixtures", fontsize=13)
ax.legend(fontsize=8)
ax.text(-0.08, 1.05, "(b)", transform=ax.transAxes, fontsize=14, fontweight="bold")

# Panel c: Improvement ratios
ax = axes[2]
all_targets = ["1d_harmonic", "2d_double_well", "2d_gmm",
               "5d_aniso_gauss", "10d_aniso_gauss", "10d_gmm"]
all_labels = ["1D H", "2D DW", "2D GMM", "5D AG", "10D AG", "10D GMM"]
is_mm_list = [False, False, True, False, False, True]

ratios = []
for t, is_mm in zip(all_targets, is_mm_list):
    metric = "crossings" if is_mm else "tau"
    v_before = get_val("LogOsc-5", t, metric)[0]
    v_after = get_val("LogOsc-5t", t, metric)[0]
    if v_before and v_after and v_before > 0 and v_after > 0:
        if is_mm:
            ratios.append(v_after / v_before)
        else:
            ratios.append(v_before / v_after)
    else:
        ratios.append(1.0)

colors_r = ["#2ca02c" if r > 1.1 else "#d62728" if r < 0.9 else "#7f7f7f" for r in ratios]
x = np.arange(len(all_targets))
ax.barh(x, ratios, color=colors_r, edgecolor="k", linewidth=0.4)
ax.axvline(1.0, color="k", ls="--", lw=0.8)
ax.set_yticks(x)
ax.set_yticklabels(all_labels, fontsize=10)
ax.set_xlabel("Improvement ratio (tuned / auto)", fontsize=11)
ax.set_title("Q-tuning improvement\nfor LogOsc-5", fontsize=13)
ax.set_xscale("log")
for i, r in enumerate(ratios):
    ax.text(r * 1.1, i, f"{r:.1f}x", va="center", fontsize=9)
ax.text(-0.08, 1.05, "(c)", transform=ax.transAxes, fontsize=14, fontweight="bold")

fig.suptitle("Impact of Q-range tuning on LogOsc performance", fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_q_tuning_impact.png"), dpi=250,
            bbox_inches="tight")
plt.close(fig)
print("Saved fig_q_tuning_impact.png")

print("\nAll figures generated.")
