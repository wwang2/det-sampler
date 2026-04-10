"""
E3 Schematic: Conceptual diagram comparing Standard CNF vs NH Thermostat CNF
divergence computation.

Panel (a): Standard CNF path (blue) — ODE solver needs div(f) via Hutchinson estimator
Panel (b): NH Thermostat CNF path (green) — ODE solver gets div(f) = -d*g(xi) analytically

Pure matplotlib, no data dependencies. Produces figures/e3_schematic.png.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib as mpl
import os

# --- Global defaults ---
mpl.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.pad_inches': 0.2,
})

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# Colors
C_BLUE = '#1f77b4'       # Standard CNF path
C_BLUE_LIGHT = '#aec7e8'
C_GREEN = '#2ca02c'       # NH path
C_GREEN_LIGHT = '#98df8a'
C_GRAY = '#7f7f7f'        # Shared components
C_GRAY_LIGHT = '#d9d9d9'
C_RED = '#d62728'         # Warning/variance
C_BG = '#fafafa'


def draw_box(ax, xy, width, height, text, color='white', edgecolor='black',
             fontsize=12, fontweight='normal', alpha=1.0, text_color='black',
             linewidth=1.5, zorder=2):
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor=edgecolor,
        linewidth=linewidth, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(box)
    cx = xy[0] + width / 2
    cy = xy[1] + height / 2
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight, color=text_color,
            zorder=zorder + 1)
    return box


def draw_arrow(ax, start, end, color='black', linewidth=1.5, style='->', zorder=1):
    """Draw an arrow from start to end."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        color=color,
        linewidth=linewidth,
        mutation_scale=15,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def make_schematic():
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), constrained_layout=True)

    for ax in axes:
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-1.0, 7.5)
        ax.set_aspect('equal')
        ax.axis('off')

    # =========================================================================
    # Panel (a): Standard CNF
    # =========================================================================
    ax = axes[0]
    ax.set_title('(a) Standard CNF (e.g., FFJORD)', fontweight='bold', fontsize=16, pad=12)

    # Background tint
    bg = FancyBboxPatch((-0.3, -0.8), 10.6, 8.0, boxstyle="round,pad=0.2",
                        facecolor=C_BLUE_LIGHT, alpha=0.12, edgecolor='none')
    ax.add_patch(bg)

    # Box 1: ODE Solver
    draw_box(ax, (0.5, 5.5), 3.5, 1.2, 'ODE Solver\n(adaptive step)',
             color=C_GRAY_LIGHT, edgecolor=C_GRAY, fontsize=12, fontweight='bold')

    # Arrow down: "needs div(f)"
    draw_arrow(ax, (2.25, 5.5), (2.25, 4.3), color=C_BLUE, linewidth=2.0)
    ax.text(3.6, 4.85, 'needs\ndiv(f)', fontsize=11, color=C_BLUE,
            ha='left', va='center', style='italic')

    # Box 2: Hutchinson estimator (wider, more detail)
    draw_box(ax, (0.0, 2.2), 5.0, 2.0,
             '',  # We'll add text manually for multi-line
             color='white', edgecolor=C_BLUE, fontsize=11, linewidth=2.0)
    ax.text(2.5, 3.7, 'Hutchinson Trace Estimator', fontsize=12, fontweight='bold',
            ha='center', va='center', color=C_BLUE)
    ax.text(2.5, 3.05, r'$\mathrm{div}(\mathbf{f}) \approx \frac{1}{n}\sum_{i=1}^{n} \mathbf{v}_i^\top \mathbf{J} \, \mathbf{v}_i$',
            fontsize=13, ha='center', va='center', color='black')
    ax.text(2.5, 2.45, r'$\mathbf{v}_i \sim \mathcal{N}(0, I)$   —   $n_\mathrm{vec}$ random draws',
            fontsize=10, ha='center', va='center', color=C_GRAY)

    # Arrow down from Hutchinson to cost box
    draw_arrow(ax, (2.5, 2.2), (2.5, 1.1), color=C_BLUE, linewidth=1.5)

    # Cost annotation box
    draw_box(ax, (0.3, 0.0), 4.4, 1.0,
             r'Cost: $O(n_\mathrm{vec} \times d)$ per step',
             color='white', edgecolor=C_RED, fontsize=12, fontweight='bold',
             text_color=C_RED, linewidth=1.5)

    # Variance annotation (right side)
    draw_box(ax, (5.8, 3.3), 4.2, 1.3, '',
             color='#fff3f3', edgecolor=C_RED, fontsize=11, linewidth=1.5)
    ax.text(7.9, 4.15, 'Stochastic', fontsize=12, fontweight='bold',
            ha='center', va='center', color=C_RED)
    ax.text(7.9, 3.6, r'Variance $\sim 1/n_\mathrm{vec}$',
            fontsize=11, ha='center', va='center', color=C_RED)

    # Arrow from Hutch box to variance box
    draw_arrow(ax, (5.0, 3.2), (5.8, 3.6), color=C_RED, linewidth=1.5, style='->')

    # Dense Jacobian note
    ax.text(7.9, 2.3, 'Dense Jacobian $\\mathbf{J} \\in \\mathbb{R}^{d \\times d}$\n'
            'from neural network $f_\\theta$',
            fontsize=10, ha='center', va='center', color=C_GRAY,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=C_GRAY_LIGHT, linewidth=1))

    # =========================================================================
    # Panel (b): NH Thermostat CNF
    # =========================================================================
    ax = axes[1]
    ax.set_title('(b) NH Thermostat CNF', fontweight='bold', fontsize=16, pad=12)

    # Background tint
    bg = FancyBboxPatch((-0.3, -0.8), 10.6, 8.0, boxstyle="round,pad=0.2",
                        facecolor=C_GREEN_LIGHT, alpha=0.12, edgecolor='none')
    ax.add_patch(bg)

    # Box 1: ODE Solver (same as left)
    draw_box(ax, (0.5, 5.5), 3.5, 1.2, 'ODE Solver\n(adaptive step)',
             color=C_GRAY_LIGHT, edgecolor=C_GRAY, fontsize=12, fontweight='bold')

    # Arrow down: "needs div(f)"
    draw_arrow(ax, (2.25, 5.5), (2.25, 4.3), color=C_GREEN, linewidth=2.0)
    ax.text(3.6, 4.85, 'needs\ndiv(f)', fontsize=11, color=C_GREEN,
            ha='left', va='center', style='italic')

    # Box 2: Analytical divergence (clean, simple)
    draw_box(ax, (0.0, 2.5), 5.0, 1.7,
             '',
             color='white', edgecolor=C_GREEN, fontsize=11, linewidth=2.0)
    ax.text(2.5, 3.65, 'Analytical Divergence', fontsize=12, fontweight='bold',
            ha='center', va='center', color=C_GREEN)
    ax.text(2.5, 2.95, r'$\mathrm{div}(\mathbf{f}) = -d \cdot g(\xi)$',
            fontsize=15, ha='center', va='center', color='black')

    # Arrow down to cost
    draw_arrow(ax, (2.5, 2.5), (2.5, 1.1), color=C_GREEN, linewidth=1.5)

    # Cost annotation box
    draw_box(ax, (0.8, 0.0), 3.4, 1.0,
             r'Cost: $O(1)$ per step',
             color='white', edgecolor=C_GREEN, fontsize=12, fontweight='bold',
             text_color=C_GREEN, linewidth=1.5)

    # Zero variance annotation (right side)
    draw_box(ax, (5.8, 3.3), 4.2, 1.3, '',
             color='#f3fff3', edgecolor=C_GREEN, fontsize=11, linewidth=1.5)
    ax.text(7.9, 4.15, 'Exact', fontsize=12, fontweight='bold',
            ha='center', va='center', color=C_GREEN)
    ax.text(7.9, 3.6, 'Zero variance',
            fontsize=11, ha='center', va='center', color=C_GREEN)

    # Arrow from analytical box to zero-variance box
    draw_arrow(ax, (5.0, 3.35), (5.8, 3.6), color=C_GREEN, linewidth=1.5, style='->')

    # Block-diagonal Jacobian note
    ax.text(7.9, 2.3, 'Block-diagonal Jacobian\n'
            r'$\partial \dot{p}_i / \partial p_j = -g(\xi)\,\delta_{ij}$' '\n'
            r'$\mathrm{Tr}(\mathbf{J}) = -d \cdot g(\xi)$',
            fontsize=10, ha='center', va='center', color=C_GRAY,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=C_GRAY_LIGHT, linewidth=1))

    # "Just read xi" annotation
    ax.annotate('Just read $\\xi$\nand evaluate $g$',
                xy=(2.5, 2.5), xytext=(6.0, 1.5),
                fontsize=11, color=C_GREEN, fontweight='bold',
                ha='center', va='center',
                arrowprops=dict(arrowstyle='->', color=C_GREEN, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f3fff3',
                          edgecolor=C_GREEN, linewidth=1))

    outpath = os.path.join(FIGDIR, 'e3_schematic.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved {outpath}")


if __name__ == '__main__':
    make_schematic()
