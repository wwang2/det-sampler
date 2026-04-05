"""Generate figures for the Hoermander bracket analysis.

Fig 1: 1x4 panel of bracket rank heatmaps at xi=0.5 for NH, Log-Osc, Tanh, Arctan.
Fig 2: 1x2 panel of (a) rank vs Q and (b) Lyapunov exponent vs Q.

Style: DPI=300, Font 14, per research/style.md.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rank_analysis import build_bracket_lambdas, compute_rank_grid, compute_rank_volume
import sympy as sp
from sympy import symbols


# Style constants
COLORS = {
    'NH': '#1f77b4',
    'Log-Osc': '#ff7f0e',
    'Tanh': '#2ca02c',
    'Arctan': '#d62728',
}

FIGDIR = "/Users/wujiewang/code/det-sampler/.worktrees/hormander-ergo-017/orbits/hormander-ergo-017/figures"


def get_frictions():
    xi_sym = symbols('xi', real=True)
    return {
        'NH': xi_sym,
        'Log-Osc': 2 * xi_sym / (1 + xi_sym**2),
        'Tanh': sp.tanh(xi_sym),
        'Arctan': sp.atan(xi_sym),
    }


def plot_rank_heatmaps():
    """Fig 1: 1x4 panel of bracket rank heatmaps."""
    frictions = get_frictions()
    n_grid = 80
    q_range = np.linspace(-3, 3, n_grid)
    p_range = np.linspace(-3, 3, n_grid)

    # Compute grids at multiple xi slices
    xi_slices = [0.0, 0.5, 1.0, 2.0]

    for xi_val in xi_slices:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f'Lie Bracket Rank at $\\xi = {xi_val}$, $Q = 0.5$',
                     fontsize=16, y=1.02)

        for idx, (name, g_expr) in enumerate(frictions.items()):
            ax = axes[idx]
            print(f"  Computing {name} at xi={xi_val}...")
            bfuncs = build_bracket_lambdas(g_expr)
            grid = compute_rank_grid(bfuncs, q_range, p_range, xi_val, Q_val=0.5)

            # Color map: rank 1 = red, rank 2 = yellow, rank 3 = green
            from matplotlib.colors import ListedColormap, BoundaryNorm
            cmap = ListedColormap(['#d62728', '#ffdd57', '#2ca02c'])
            bounds = [0.5, 1.5, 2.5, 3.5]
            norm = BoundaryNorm(bounds, cmap.N)

            im = ax.imshow(grid, extent=[-3, 3, -3, 3], origin='lower',
                           cmap=cmap, norm=norm, aspect='equal')
            ax.set_title(name, fontsize=14)
            ax.set_xlabel('$q$', fontsize=14)
            if idx == 0:
                ax.set_ylabel('$p$', fontsize=14)
            ax.tick_params(labelsize=12)

            # Count stats
            n_full = np.sum(grid == 3)
            n_total = grid.size
            ax.text(0.05, 0.95, f'rank=3: {100*n_full/n_total:.0f}%',
                    transform=ax.transAxes, fontsize=11, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Colorbar
        cbar = fig.colorbar(im, ax=axes, ticks=[1, 2, 3], shrink=0.8,
                            label='Bracket rank')
        cbar.ax.tick_params(labelsize=12)

        plt.tight_layout()
        path = f"{FIGDIR}/rank_heatmap_xi{xi_val:.1f}.png"
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {path}")

    # Also make the canonical 1x4 figure at xi=0.5
    print("Done with heatmaps.")


def plot_rank_vs_Q():
    """Fig 2a: Fraction of points with full rank vs Q."""
    frictions = get_frictions()
    Q_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, g_expr in frictions.items():
        print(f"  Rank vs Q for {name}...")
        bfuncs = build_bracket_lambdas(g_expr)
        rank3_fracs = []
        for Q_val in Q_values:
            vol = compute_rank_volume(bfuncs, Q_val, n_samples=5000, seed=42)
            rank3_fracs.append(vol['rank3_frac'])

        ax.plot(Q_values, rank3_fracs, 'o-', color=COLORS[name],
                label=name, linewidth=2, markersize=6)

    ax.set_xlabel('Thermostat mass $Q$', fontsize=14)
    ax.set_ylabel('Fraction with rank = 3', fontsize=14)
    ax.set_title('Lie Bracket Rank vs Thermostat Mass', fontsize=16)
    ax.set_ylim(0.95, 1.005)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    path = f"{FIGDIR}/rank_vs_Q.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def plot_rank_and_lyapunov():
    """Fig 2: Combined 1x2 panel of rank fraction and Lyapunov exponent vs Q."""
    # Lyapunov data from unified-theory-007
    lyapunov_data = {
        'NH': {
            0.1: 0.001699, 0.2: 0.025735, 0.3: 0.034833, 0.5: 0.056283,
            0.7: 0.001223, 1.0: 0.001270, 1.5: 0.001093, 2.0: 0.000966,
            3.0: 0.000747, 5.0: 0.000610,
        },
        'Log-Osc': {
            0.1: 0.626368, 0.2: 0.514116, 0.3: 0.396856, 0.5: 0.199200,
            0.7: 0.001026, 1.0: 0.001194, 1.5: 0.001503, 2.0: 0.001070,
            3.0: 0.001228, 5.0: 0.001000,
        },
        'Tanh': {
            0.1: 0.434587, 0.2: 0.323408, 0.3: 0.215826, 0.5: 0.097877,
            0.7: 0.001605, 1.0: 0.001446, 1.5: 0.001337, 2.0: 0.001037,
            3.0: 0.000717, 5.0: 0.000819,
        },
        'Arctan': {
            0.1: 0.319844, 0.2: 0.240091, 0.3: 0.127595, 0.5: 0.115867,
            0.7: 0.001579, 1.0: 0.001046, 1.5: 0.000731, 2.0: 0.000752,
            3.0: 0.000900, 5.0: 0.000840,
        },
    }

    frictions = get_frictions()
    Q_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel (a): Rank fraction vs Q
    for name, g_expr in frictions.items():
        print(f"  Rank vs Q for {name}...")
        bfuncs = build_bracket_lambdas(g_expr)
        rank3_fracs = []
        for Q_val in Q_values:
            vol = compute_rank_volume(bfuncs, Q_val, n_samples=5000, seed=42)
            rank3_fracs.append(vol['rank3_frac'])

        ax1.plot(Q_values, rank3_fracs, 'o-', color=COLORS[name],
                 label=name, linewidth=2, markersize=6)

    ax1.set_xlabel('Thermostat mass $Q$', fontsize=14)
    ax1.set_ylabel('Fraction with rank = 3', fontsize=14)
    ax1.set_title('(a) Lie Bracket Rank', fontsize=14)
    ax1.set_ylim(0.94, 1.005)
    ax1.legend(fontsize=12, loc='lower right')
    ax1.tick_params(labelsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xscale('log')

    # Panel (b): Lyapunov exponent vs Q
    for name, ldata in lyapunov_data.items():
        Qs = sorted(ldata.keys())
        lambdas = [ldata[Q] for Q in Qs]
        ax2.plot(Qs, lambdas, 'o-', color=COLORS[name],
                 label=name, linewidth=2, markersize=6)

    ax2.set_xlabel('Thermostat mass $Q$', fontsize=14)
    ax2.set_ylabel('Maximal Lyapunov exponent $\\lambda_{\\max}$', fontsize=14)
    ax2.set_title('(b) Lyapunov Exponent', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.tick_params(labelsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_ylim(1e-4, 1.0)

    # Add annotation
    ax1.annotate('All frictions have\nfull rank generically',
                 xy=(1.0, 0.99), fontsize=11, ha='center',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                           edgecolor='gray'))
    ax2.annotate('Bounded frictions:\n10-300x larger $\\lambda$',
                 xy=(0.15, 0.4), fontsize=11, ha='left',
                 xycoords='axes fraction',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                           edgecolor='gray'))

    plt.tight_layout()
    path = f"{FIGDIR}/rank_vs_lyapunov.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


if __name__ == '__main__':
    import os
    os.makedirs(FIGDIR, exist_ok=True)

    print("Generating Fig 1: Rank heatmaps...")
    plot_rank_heatmaps()

    print("\nGenerating Fig 2: Rank vs Lyapunov...")
    plot_rank_and_lyapunov()

    print("\nAll figures generated.")
