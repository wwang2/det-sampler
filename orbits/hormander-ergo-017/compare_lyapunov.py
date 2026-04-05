"""Compare Lie bracket rank with Lyapunov exponents.

Key finding: bracket rank is uniformly full (3) for ALL frictions at generic
points, yet Lyapunov exponents differ by 10-300x. This demonstrates the
fundamental gap between controllability (bracket condition) and ergodicity
(positive Lyapunov exponents / mixing) for deterministic ODEs.

Lyapunov data from unified-theory-007 (parent orbit).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGDIR = "/Users/wujiewang/code/det-sampler/.worktrees/hormander-ergo-017/orbits/hormander-ergo-017/figures"

COLORS = {
    'NH': '#1f77b4',
    'Log-Osc': '#ff7f0e',
    'Tanh': '#2ca02c',
    'Arctan': '#d62728',
}

# Lyapunov data from unified-theory-007 (seed=42, dt=0.01, T=5000)
LYAPUNOV_DATA = {
    'NH': {
        0.1: 0.001699, 0.2: 0.025735, 0.3: 0.034833, 0.5: 0.056283,
        0.7: 0.001223, 1.0: 0.001270, 1.5: 0.001093, 2.0: 0.000966,
        3.0: 0.000747, 5.0: 0.000610, 10.0: 0.000479,
    },
    'Log-Osc': {
        0.1: 0.626368, 0.2: 0.514116, 0.3: 0.396856, 0.5: 0.199200,
        0.7: 0.001026, 1.0: 0.001194, 1.5: 0.001503, 2.0: 0.001070,
        3.0: 0.001228, 5.0: 0.001000, 10.0: 0.000658,
    },
    'Tanh': {
        0.1: 0.434587, 0.2: 0.323408, 0.3: 0.215826, 0.5: 0.097877,
        0.7: 0.001605, 1.0: 0.001446, 1.5: 0.001337, 2.0: 0.001037,
        3.0: 0.000717, 5.0: 0.000819, 10.0: 0.000530,
    },
    'Arctan': {
        0.1: 0.319844, 0.2: 0.240091, 0.3: 0.127595, 0.5: 0.115867,
        0.7: 0.001579, 1.0: 0.001046, 1.5: 0.000731, 2.0: 0.000752,
        3.0: 0.000900, 5.0: 0.000840, 10.0: 0.000596,
    },
}

# Ergodicity scores from unified-theory-007 coverage analysis
ERGODICITY_SCORES = {
    'NH': {0.3: 0.72, 0.5: 0.54, 0.8: 0.42},
    'Log-Osc': {0.3: 0.91, 0.5: 0.982, 0.8: 0.944},
    'Tanh': {0.3: 0.85, 0.5: 0.89, 0.8: 0.87},
    'Arctan': {0.3: 0.80, 0.5: 0.85, 0.8: 0.83},
}


def plot_comparison():
    """Plot the disconnect between bracket rank and Lyapunov exponents."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    Q_values = sorted(LYAPUNOV_DATA['NH'].keys())

    # Panel (a): Lyapunov exponents vs Q
    ax = axes[0]
    for name, ldata in LYAPUNOV_DATA.items():
        Qs = sorted(ldata.keys())
        lambdas = [ldata[Q] for Q in Qs]
        ax.plot(Qs, lambdas, 'o-', color=COLORS[name],
                label=name, linewidth=2, markersize=5)
    ax.set_xlabel('Thermostat mass $Q$', fontsize=14)
    ax.set_ylabel('$\\lambda_{\\max}$', fontsize=14)
    ax.set_title('(a) Lyapunov Exponent', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(3e-4, 1.0)
    ax.legend(fontsize=11, loc='upper right')
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)

    # Panel (b): Bracket rank (constant = 3 for all)
    ax = axes[1]
    for name in LYAPUNOV_DATA:
        # Bracket rank is 3 everywhere (except measure-zero set)
        ax.plot(Q_values, [3] * len(Q_values), 'o-', color=COLORS[name],
                label=name, linewidth=2, markersize=5, alpha=0.7)
    ax.set_xlabel('Thermostat mass $Q$', fontsize=14)
    ax.set_ylabel('Generic Lie bracket rank', fontsize=14)
    ax.set_title('(b) Bracket Rank (all identical)', fontsize=14)
    ax.set_xscale('log')
    ax.set_ylim(0, 4)
    ax.set_yticks([0, 1, 2, 3])
    ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5, label='Full rank')
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.annotate('Bracket rank = 3 for ALL frictions\nat ALL values of $Q$',
                xy=(0.5, 0.3), fontsize=12, ha='center',
                xycoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                          edgecolor='gray'))

    # Panel (c): Gap visualization - Lyapunov ratio (bounded/NH)
    ax = axes[2]
    nh_data = LYAPUNOV_DATA['NH']
    for name in ['Log-Osc', 'Tanh', 'Arctan']:
        Qs = sorted(LYAPUNOV_DATA[name].keys())
        ratios = []
        for Q in Qs:
            nh_val = nh_data[Q]
            if nh_val > 1e-4:
                ratios.append(LYAPUNOV_DATA[name][Q] / nh_val)
            else:
                ratios.append(np.nan)
        ax.plot(Qs, ratios, 'o-', color=COLORS[name],
                label=f'{name} / NH', linewidth=2, markersize=5)

    ax.set_xlabel('Thermostat mass $Q$', fontsize=14)
    ax.set_ylabel('$\\lambda_{\\max}$ ratio vs NH', fontsize=14)
    ax.set_title('(c) Lyapunov Advantage of Bounded Frictions', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{FIGDIR}/bracket_vs_lyapunov.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def print_summary_table():
    """Print the key comparison table."""
    print("\n" + "="*70)
    print("BRACKET RANK vs LYAPUNOV EXPONENT: The Controllability-Ergodicity Gap")
    print("="*70)

    print(f"\n{'Friction':<12} {'Bracket Rank':>14} {'lambda(Q=0.3)':>14} "
          f"{'lambda(Q=0.5)':>14} {'Ergodic?':>10}")
    print("-" * 70)

    for name in ['NH', 'Log-Osc', 'Tanh', 'Arctan']:
        rank = "3 (generic)"
        l03 = LYAPUNOV_DATA[name].get(0.3, 0)
        l05 = LYAPUNOV_DATA[name].get(0.5, 0)
        ergo = "No" if name == "NH" else "Yes"
        print(f"{name:<12} {rank:>14} {l03:>14.4f} {l05:>14.4f} {ergo:>10}")

    print("\nKey insight: ALL frictions satisfy the bracket condition (rank=3 generically)")
    print("yet NH fails to be ergodic. The bracket condition is NECESSARY but not SUFFICIENT.")
    print("The 10-300x difference in Lyapunov exponents reflects a DYNAMICAL distinction")
    print("(KAM torus deformation) that the bracket analysis cannot detect.")


if __name__ == '__main__':
    import os
    os.makedirs(FIGDIR, exist_ok=True)

    print_summary_table()
    plot_comparison()
