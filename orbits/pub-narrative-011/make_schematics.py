#!/usr/bin/env python3
"""Generate conceptual schematic figures for the paper.

These are SCHEMATICS, not data plots. They use matplotlib patches,
arrows, and text to create textbook-style diagrams.

Generates:
  - fig_schematic_thermostat.png  (the thermostat concept)
  - fig_schematic_friction.png    (friction function gallery)
  - fig_schematic_kam.png         (KAM tori vs space-filling)
  - fig_schematic_multiscale.png  (multi-scale thermostat)
  - fig_schematic_progression.png (discovery progression)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle, Arc
import matplotlib.patheffects as pe

# --- Style constants (from research/style.md) ---
COLOR_NH = "#1f77b4"       # blue
COLOR_NHC = "#ff7f0e"      # orange
COLOR_LOGOSC = "#2ca02c"   # green (tab10 index 2)
COLOR_TANH = "#d62728"     # red (tab10 index 3)
COLOR_ARCTAN = "#9467bd"   # purple (tab10 index 4)
COLOR_GRAY = "#888888"
COLOR_BG_LIGHT = "#f7f7f7"

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)

DPI = 300
FONT_LABEL = 14
FONT_TICK = 12
FONT_TITLE = 16
FONT_ANNOT = 11


def styled_text(ax, x, y, text, fontsize=FONT_ANNOT, **kwargs):
    """Text with white outline for readability."""
    defaults = dict(ha="center", va="center", fontsize=fontsize, fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    defaults.update(kwargs)
    return ax.text(x, y, text, **defaults)


# =============================================================================
# Schematic 1: The Thermostat Concept
# =============================================================================
def make_thermostat_concept():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("The Thermostat Concept", fontsize=FONT_TITLE, fontweight="bold", y=0.98)

    for idx, (ax, title, color, label_top, label_bot, desc) in enumerate(zip(
        axes,
        ["Standard Nose-Hoover", "Log-Osc Thermostat"],
        [COLOR_NH, COLOR_LOGOSC],
        ["Friction grows without bound", "Friction saturates at |g|=1"],
        ["System locks onto KAM torus", "System stays loose, explores"],
        ["g(\\xi) = \\xi  (unbounded)", "g(\\xi) = 2\\xi/(1+\\xi^2)  (bounded)"]
    )):
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-1, 9)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=FONT_LABEL, fontweight="bold", color=color, pad=10)

        # Physical system box
        box_phys = FancyBboxPatch((0.3, 4.5), 3.5, 3.2, boxstyle="round,pad=0.2",
                                   facecolor="#e8f4fd", edgecolor="black", linewidth=1.5)
        ax.add_patch(box_phys)
        ax.text(2.05, 7.2, "Physical System", ha="center", fontsize=FONT_ANNOT, fontweight="bold")

        # Particles inside box
        for cx, cy in [(1.0, 5.8), (1.8, 5.2), (2.8, 6.0), (2.2, 6.5), (3.0, 5.3)]:
            circ = Circle((cx, cy), 0.25, facecolor=color, edgecolor="black", alpha=0.7, linewidth=1)
            ax.add_patch(circ)

        # Heat bath box
        box_bath = FancyBboxPatch((6.0, 4.5), 3.8, 3.2, boxstyle="round,pad=0.2",
                                   facecolor="#fdf2e8", edgecolor="black", linewidth=1.5)
        ax.add_patch(box_bath)
        ax.text(7.9, 7.2, "Heat Bath", ha="center", fontsize=FONT_ANNOT, fontweight="bold")
        ax.text(7.9, 6.5, r"$\xi$ variable", ha="center", fontsize=FONT_ANNOT, fontstyle="italic")

        # Coupling knob (circle with indicator)
        knob_x, knob_y = 7.9, 5.4
        knob = Circle((knob_x, knob_y), 0.55, facecolor="white", edgecolor=color, linewidth=2.5)
        ax.add_patch(knob)
        # Knob indicator
        if idx == 0:
            # NH: needle pointing far (over-tightened)
            angle = np.radians(30)
            ax.annotate("", xy=(knob_x + 0.5*np.cos(angle), knob_y + 0.5*np.sin(angle)),
                        xytext=(knob_x, knob_y),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5))
            # Red warning zone
            arc = Arc((knob_x, knob_y), 1.1, 1.1, angle=0, theta1=0, theta2=90,
                      color="red", linewidth=3, linestyle="--")
            ax.add_patch(arc)
            ax.text(knob_x + 0.9, knob_y + 0.5, "!", color="red", fontsize=14, fontweight="bold")
        else:
            # Log-Osc: needle in moderate position, with soft stop
            angle = np.radians(60)
            ax.annotate("", xy=(knob_x + 0.45*np.cos(angle), knob_y + 0.45*np.sin(angle)),
                        xytext=(knob_x, knob_y),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5))
            # Green safe zone
            arc = Arc((knob_x, knob_y), 1.1, 1.1, angle=0, theta1=30, theta2=90,
                      color=COLOR_LOGOSC, linewidth=3)
            ax.add_patch(arc)
            ax.text(knob_x + 0.85, knob_y + 0.6, "ok", color=COLOR_LOGOSC, fontsize=10, fontweight="bold")

        # Coupling arrow
        ax.annotate("", xy=(6.0, 6.0), xytext=(3.8, 6.0),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=2.5,
                                    connectionstyle="arc3,rad=0"))
        ax.text(4.9, 6.35, "coupling", ha="center", fontsize=9, fontstyle="italic")

        # Description formula
        ax.text(5.25, 3.5, f"${desc}$", ha="center", fontsize=FONT_ANNOT,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor=color, alpha=0.9))

        # Top label
        ax.text(5.25, 2.3, label_top, ha="center", fontsize=FONT_ANNOT, color=color, fontweight="bold")

        # Bottom label (outcome)
        outcome_color = "red" if idx == 0 else COLOR_LOGOSC
        ax.text(5.25, 1.2, label_bot, ha="center", fontsize=FONT_ANNOT,
                color=outcome_color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=outcome_color, alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIGDIR, "fig_schematic_thermostat.png"), dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  -> fig_schematic_thermostat.png")


# =============================================================================
# Schematic 2: Friction Function Gallery
# =============================================================================
def make_friction_gallery():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Friction Function Gallery", fontsize=FONT_TITLE, fontweight="bold", y=1.0)

    xi = np.linspace(-5, 5, 500)

    # g(xi) functions
    g_nh = xi
    g_logosc = 2 * xi / (1 + xi**2)
    g_tanh = np.tanh(xi)
    g_arctan = np.arctan(xi)

    # V(xi) potentials (scaled by Q=1)
    v_nh = xi**2 / 2
    v_logosc = np.log(1 + xi**2)
    v_tanh = np.log(np.cosh(xi))
    v_arctan = xi * np.arctan(xi) - 0.5 * np.log(1 + xi**2)

    # p(xi) marginals ~ exp(-V(xi)/kT) with kT=1
    def normalize(p):
        dx = xi[1] - xi[0]
        return p / (np.sum(p) * dx)

    p_nh = normalize(np.exp(-v_nh))
    p_logosc = normalize(np.exp(-v_logosc))
    p_tanh = normalize(np.exp(-v_tanh))
    p_arctan = normalize(np.exp(-v_arctan))

    # --- Panel 1: g(xi) ---
    ax = axes[0]
    ax.plot(xi, g_nh, color=COLOR_NH, lw=2.5, label="NH: $g=\\xi$")
    ax.plot(xi, g_logosc, color=COLOR_LOGOSC, lw=2.5, label="Log-Osc: $2\\xi/(1+\\xi^2)$")
    ax.plot(xi, g_tanh, color=COLOR_TANH, lw=2.5, label="Tanh: $\\tanh(\\xi)$", ls="--")
    ax.plot(xi, g_arctan, color=COLOR_ARCTAN, lw=2.5, label="Arctan: $\\arctan(\\xi)$", ls=":")
    ax.axhline(1, color=COLOR_GRAY, ls="--", lw=1, alpha=0.5)
    ax.axhline(-1, color=COLOR_GRAY, ls="--", lw=1, alpha=0.5)
    ax.set_ylim(-3, 3)
    ax.set_xlabel(r"$\xi$", fontsize=FONT_LABEL)
    ax.set_ylabel(r"$g(\xi)$", fontsize=FONT_LABEL)
    ax.set_title("Friction Function", fontsize=FONT_LABEL, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.tick_params(labelsize=FONT_TICK)

    # Annotate bounded region
    ax.annotate("bounded\nregion", xy=(3.5, 1), xytext=(3.5, 2.2),
                fontsize=9, ha="center", color=COLOR_LOGOSC, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COLOR_LOGOSC))

    # --- Panel 2: V(xi) ---
    ax = axes[1]
    ax.plot(xi, v_nh, color=COLOR_NH, lw=2.5, label="$Q\\xi^2/2$")
    ax.plot(xi, v_logosc, color=COLOR_LOGOSC, lw=2.5, label="$Q\\log(1+\\xi^2)$")
    ax.plot(xi, v_tanh, color=COLOR_TANH, lw=2.5, label="$Q\\log\\cosh(\\xi)$", ls="--")
    ax.plot(xi, v_arctan, color=COLOR_ARCTAN, lw=2.5, label="$Q[\\xi\\arctan\\xi - \\frac{1}{2}\\log(1+\\xi^2)]$", ls=":")
    ax.set_ylim(-0.5, 8)
    ax.set_xlabel(r"$\xi$", fontsize=FONT_LABEL)
    ax.set_ylabel(r"$V(\xi)$", fontsize=FONT_LABEL)
    ax.set_title("Thermostat Potential", fontsize=FONT_LABEL, fontweight="bold")
    ax.legend(fontsize=7, loc="upper center")
    ax.tick_params(labelsize=FONT_TICK)

    # Annotate: log grows slowly
    ax.annotate("slow growth\n(log)", xy=(4, np.log(1+16)), xytext=(3, 6.5),
                fontsize=9, ha="center", color=COLOR_LOGOSC, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COLOR_LOGOSC))

    # --- Panel 3: p(xi) marginals ---
    ax = axes[2]
    ax.plot(xi, p_nh, color=COLOR_NH, lw=2.5, label="Gaussian")
    ax.plot(xi, p_logosc, color=COLOR_LOGOSC, lw=2.5, label="Cauchy-like")
    ax.plot(xi, p_tanh, color=COLOR_TANH, lw=2.5, label="sech$^2$-like", ls="--")
    ax.plot(xi, p_arctan, color=COLOR_ARCTAN, lw=2.5, label="sub-Gaussian", ls=":")
    ax.set_xlabel(r"$\xi$", fontsize=FONT_LABEL)
    ax.set_ylabel(r"$p(\xi) \propto \exp(-V(\xi)/kT)$", fontsize=FONT_LABEL)
    ax.set_title("Thermostat Marginal", fontsize=FONT_LABEL, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.tick_params(labelsize=FONT_TICK)

    # Annotate heavy tails
    ax.annotate("heavy tails\n(Cauchy-like)", xy=(3.5, p_logosc[xi > 3.4][0]),
                xytext=(3.2, 0.15),
                fontsize=9, ha="center", color=COLOR_LOGOSC, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COLOR_LOGOSC))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIGDIR, "fig_schematic_friction.png"), dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  -> fig_schematic_friction.png")


# =============================================================================
# Schematic 3: KAM Tori vs Space-Filling Orbits
# =============================================================================
def make_kam_schematic():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle("Phase Space Structure: KAM Tori vs Ergodic Orbits",
                 fontsize=FONT_TITLE, fontweight="bold", y=1.0)

    # --- Left: NH (KAM tori) ---
    ax = axes[0]
    ax.set_title("Nose-Hoover (NH)", fontsize=FONT_LABEL, fontweight="bold", color=COLOR_NH)
    ax.set_xlabel("$q$", fontsize=FONT_LABEL)
    ax.set_ylabel("$p$", fontsize=FONT_LABEL)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=FONT_TICK)

    # Draw nested elliptical tori
    theta = np.linspace(0, 2 * np.pi, 500)
    for a, b, alpha in [(2.8, 2.8, 0.15), (2.2, 2.2, 0.2), (1.5, 1.5, 0.3),
                         (0.9, 0.9, 0.4), (0.4, 0.4, 0.5)]:
        q_torus = a * np.cos(theta)
        p_torus = b * np.sin(theta)
        ax.plot(q_torus, p_torus, color=COLOR_NH, lw=1.5, alpha=alpha)

    # Draw a single "trapped" trajectory on one torus
    a_trap, b_trap = 1.8, 1.8
    # Add slight wobble to make it look like a trajectory stuck on a torus
    wobble = 0.08 * np.sin(7 * theta)
    q_trap = (a_trap + wobble) * np.cos(theta)
    p_trap = (b_trap + wobble) * np.sin(theta)
    ax.plot(q_trap, p_trap, color=COLOR_NH, lw=2.5, alpha=0.9)

    # Arrow showing direction
    idx_arr = 100
    ax.annotate("", xy=(q_trap[idx_arr+5], p_trap[idx_arr+5]),
                xytext=(q_trap[idx_arr], p_trap[idx_arr]),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_NH, lw=2))

    # Labels
    styled_text(ax, 0, 0, "trapped\ntrajectory", fontsize=11, color="red")
    ax.annotate("KAM tori\ndivide\nphase space", xy=(-2.5, 2.5), xytext=(-2.5, 2.5),
                fontsize=10, ha="center", color=COLOR_GRAY, fontstyle="italic")

    # Shade the "unvisited" regions
    ax.fill_between(np.linspace(-3.5, 3.5, 100), -3.5, 3.5, alpha=0.03, color="red")

    # --- Right: Log-Osc (space-filling) ---
    ax = axes[1]
    ax.set_title("Log-Osc Thermostat", fontsize=FONT_LABEL, fontweight="bold", color=COLOR_LOGOSC)
    ax.set_xlabel("$q$", fontsize=FONT_LABEL)
    ax.set_ylabel("$p$", fontsize=FONT_LABEL)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=FONT_TICK)

    # Draw a space-filling orbit (Lissajous-like with incommensurate frequencies + noise)
    rng = np.random.default_rng(42)
    t = np.linspace(0, 200, 20000)
    # Simulate a "space-filling" trajectory with multiple frequencies
    q_fill = (1.8 * np.sin(t) + 0.7 * np.sin(np.sqrt(2) * t + 1.3)
              + 0.4 * np.sin(np.pi * t + 0.7))
    p_fill = (1.8 * np.cos(t + 0.5) + 0.7 * np.cos(np.sqrt(3) * t + 2.1)
              + 0.4 * np.cos(np.e * t + 1.1))
    # Clip to range
    mask = (np.abs(q_fill) < 3.3) & (np.abs(p_fill) < 3.3)
    ax.plot(q_fill[mask], p_fill[mask], color=COLOR_LOGOSC, lw=0.3, alpha=0.4)

    # Overlay Gaussian contours (target distribution)
    for sigma in [1, 2, 3]:
        circle = plt.Circle((0, 0), sigma, fill=False, color=COLOR_GRAY,
                            ls="--", lw=1, alpha=0.4)
        ax.add_patch(circle)

    # Labels
    styled_text(ax, 0, -0.3, "ergodic\ntrajectory", fontsize=11, color=COLOR_LOGOSC)
    ax.annotate("bounded friction\ndeforms tori\n$\\rightarrow$ space-filling",
                xy=(2.5, -2.5), xytext=(2.5, -2.5),
                fontsize=10, ha="center", color=COLOR_GRAY, fontstyle="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIGDIR, "fig_schematic_kam.png"), dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  -> fig_schematic_kam.png")


# =============================================================================
# Schematic 4: Multi-Scale Thermostat
# =============================================================================
def make_multiscale_schematic():
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle("Multi-Scale Thermostat: Broadband Friction from Log-Spaced Masses",
                 fontsize=FONT_TITLE, fontweight="bold", y=0.98)

    # Layout: left half = 3 oscillator panels, right half = combined signal + barrier
    gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.4,
                          left=0.06, right=0.97, top=0.88, bottom=0.08)

    t = np.linspace(0, 10, 1000)

    configs = [
        ("Fast: $Q=0.1$", 0.1, "#e74c3c", "local temperature\ncontrol"),
        ("Medium: $Q=0.7$", 0.7, "#f39c12", "intermediate\nmixing"),
        ("Slow: $Q=10$", 10.0, "#3498db", "barrier crossing\nboost"),
    ]

    signals = []
    for i, (label, Q, color, desc) in enumerate(configs):
        ax = fig.add_subplot(gs[i, 0])
        freq = np.sqrt(1.0 / Q)  # characteristic frequency
        signal = np.sin(freq * t * 2 * np.pi) * (1.0 / (1 + 0.3 * freq))
        signals.append(signal)

        ax.plot(t, signal, color=color, lw=1.5)
        ax.set_xlim(0, 10)
        ax.set_ylim(-1.5, 1.5)
        ax.set_ylabel(r"$\xi_{" + str(i+1) + "}(t)$", fontsize=10)
        if i == 2:
            ax.set_xlabel("Time", fontsize=FONT_ANNOT)
        else:
            ax.set_xticklabels([])
        ax.set_title(label, fontsize=FONT_ANNOT, fontweight="bold", color=color, loc="left")
        ax.tick_params(labelsize=9)

        # Description on right
        ax.text(10.3, 0, desc, fontsize=9, va="center", ha="left", color=color,
                fontstyle="italic", transform=ax.transData)

    # Combined signal
    ax_combined = fig.add_subplot(gs[:2, 1:])
    combined = sum(signals)
    ax_combined.plot(t, combined, color="black", lw=1.2)
    ax_combined.fill_between(t, 0, combined, alpha=0.15, color=COLOR_LOGOSC)
    ax_combined.set_xlim(0, 10)
    ax_combined.set_xlabel("Time", fontsize=FONT_ANNOT)
    ax_combined.set_ylabel("Combined friction $g(t)$", fontsize=FONT_ANNOT)
    ax_combined.set_title("Broadband Friction Signal (sum of 3 frequencies)",
                          fontsize=FONT_ANNOT, fontweight="bold")
    ax_combined.tick_params(labelsize=FONT_TICK)

    # Annotate: "complex, ~1/f spectrum"
    ax_combined.text(5, max(combined)*0.85, "complex, broadband, $\\sim 1/f$ spectrum",
                     fontsize=10, ha="center", fontstyle="italic", color=COLOR_GRAY,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

    # Energy landscape with barrier crossing
    ax_land = fig.add_subplot(gs[2, 1:])
    x = np.linspace(-2.5, 2.5, 500)
    U = (x**2 - 1)**2  # double well
    ax_land.plot(x, U, color="black", lw=2.5)
    ax_land.fill_between(x, 0, U, alpha=0.08, color="black")

    # Trajectory hopping
    hop_x = np.array([-1, -0.8, -0.3, 0.3, 0.8, 1.0, 0.8, 0.3, -0.3, -0.8, -1])
    hop_y = (hop_x**2 - 1)**2 + 0.15
    ax_land.plot(hop_x, hop_y, "o-", color=COLOR_LOGOSC, lw=2, markersize=5, alpha=0.8)

    # Arrow at barrier top
    ax_land.annotate("barrier\ncrossing", xy=(0, 1.0), xytext=(1.8, 1.5),
                     fontsize=10, ha="center", color=COLOR_LOGOSC, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=COLOR_LOGOSC, lw=1.5))

    ax_land.set_xlabel("Position $q$", fontsize=FONT_ANNOT)
    ax_land.set_ylabel("$U(q)$", fontsize=FONT_ANNOT)
    ax_land.set_title("System crosses barriers with multi-scale friction",
                      fontsize=FONT_ANNOT, fontweight="bold")
    ax_land.set_ylim(-0.1, 2.0)
    ax_land.tick_params(labelsize=FONT_TICK)

    fig.savefig(os.path.join(FIGDIR, "fig_schematic_multiscale.png"), dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  -> fig_schematic_multiscale.png")


# =============================================================================
# Schematic 5: Discovery Progression
# =============================================================================
def make_progression_schematic():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-1, 7)
    ax.axis("off")
    fig.suptitle("The Discovery Progression", fontsize=FONT_TITLE, fontweight="bold", y=0.97)

    # Define nodes
    nodes = [
        {"x": 1.2, "y": 4.5, "label": "NH\n(1985)", "color": COLOR_NH,
         "desc": "single $\\xi$\nnon-ergodic\n(KAM tori)"},
        {"x": 4.0, "y": 4.5, "label": "NHC\n(1992)", "color": COLOR_NHC,
         "desc": "chain of $\\xi_j$\nimproved erg.\n(more DOF)"},
        {"x": 7.0, "y": 4.5, "label": "Log-Osc\n(this work)", "color": COLOR_LOGOSC,
         "desc": "bounded $g(\\xi)$\nbreaks KAM\n(erg. 0.94)"},
        {"x": 10.0, "y": 4.5, "label": "Multi-Scale\n(this work)", "color": "#e74c3c",
         "desc": "multi-Q values\nmode hopping\n(KL: 0.38$\\to$0.054)"},
        {"x": 13.0, "y": 4.5, "label": "LOCR\n(this work)", "color": "#9467bd",
         "desc": "selective chain\ndominates all\n(40% better NHC)"},
    ]

    # Draw boxes and labels
    box_w, box_h = 2.2, 1.6
    for node in nodes:
        x, y = node["x"], node["y"]
        box = FancyBboxPatch((x - box_w/2, y - box_h/2), box_w, box_h,
                              boxstyle="round,pad=0.15",
                              facecolor=node["color"], edgecolor="black",
                              linewidth=2, alpha=0.2)
        ax.add_patch(box)
        border = FancyBboxPatch((x - box_w/2, y - box_h/2), box_w, box_h,
                                 boxstyle="round,pad=0.15",
                                 facecolor="none", edgecolor=node["color"],
                                 linewidth=2.5)
        ax.add_patch(border)
        ax.text(x, y, node["label"], ha="center", va="center",
                fontsize=12, fontweight="bold", color=node["color"])

        # Description below
        ax.text(x, y - box_h/2 - 0.4, node["desc"], ha="center", va="top",
                fontsize=9, color=COLOR_GRAY, linespacing=1.3)

    # Draw arrows between nodes
    for i in range(len(nodes) - 1):
        x1 = nodes[i]["x"] + box_w/2 + 0.05
        x2 = nodes[i+1]["x"] - box_w/2 - 0.05
        y = nodes[i]["y"]
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=2,
                                    connectionstyle="arc3,rad=0"))

    # Innovation type labels (color-coded)
    ax.text(2.6, 6.3, "Strategy:", fontsize=11, fontweight="bold", color="black")
    labels_top = [
        (1.2, "extend system", COLOR_NH),
        (4.0, "chain coupling", COLOR_NHC),
        (7.0, "reshape friction", COLOR_LOGOSC),
        (10.0, "multi-frequency", "#e74c3c"),
        (13.0, "selective targeting", "#9467bd"),
    ]
    for x, text, color in labels_top:
        ax.text(x, 6.0, text, ha="center", fontsize=9, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.12, edgecolor=color))

    # Timeline arrow at bottom
    ax.annotate("", xy=(13.5, 0.5), xytext=(0.0, 0.5),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_GRAY, lw=1.5))
    ax.text(7, 0.15, "increasing ergodicity and mixing quality", ha="center",
            fontsize=10, fontstyle="italic", color=COLOR_GRAY)

    fig.savefig(os.path.join(FIGDIR, "fig_schematic_progression.png"), dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  -> fig_schematic_progression.png")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("Generating schematic figures...")
    make_thermostat_concept()
    make_friction_gallery()
    make_kam_schematic()
    make_multiscale_schematic()
    make_progression_schematic()
    print("Done. All schematics saved to:", FIGDIR)
