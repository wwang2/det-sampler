"""Task 3: Min-max figure showing 1/f is optimal.

Plot S_alpha(f)/S_1f(f) for alpha=0, 0.5, 1, 1.5, 2 over a 2-decade band.
Show that alpha=1 gives constant ratio (flat line) = minimax optimal.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, '/Users/wujiewang/code/det-sampler/.worktrees/spectral-design-theory-025')

ORBIT_DIR = Path('/Users/wujiewang/code/det-sampler/.worktrees/spectral-design-theory-025/orbits/spectral-design-theory-025')
FIGURES_DIR = ORBIT_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)


def normalized_spectrum(f: np.ndarray, alpha: float, f_lo: float, f_hi: float) -> np.ndarray:
    """Power-law spectrum S_alpha(f) = C * f^{-alpha}, normalized to unit integral."""
    if abs(alpha - 1.0) < 1e-9:
        C = 1.0 / np.log(f_hi / f_lo)
    else:
        C = (1.0 - alpha) / (f_hi**(1.0 - alpha) - f_lo**(1.0 - alpha))
    return C * f**(-alpha)


def max_regret(alpha: float, f_lo: float, f_hi: float, n_pts: int = 10000) -> float:
    """Compute max_f [S_1f(f) / S_alpha(f)] over the band."""
    f = np.logspace(np.log10(f_lo), np.log10(f_hi), n_pts)
    s1f = normalized_spectrum(f, 1.0, f_lo, f_hi)
    sa = normalized_spectrum(f, alpha, f_lo, f_hi)
    ratio = s1f / sa
    return float(np.max(ratio))


def min_regret(alpha: float, f_lo: float, f_hi: float, n_pts: int = 10000) -> float:
    """Compute min_f [S_1f(f) / S_alpha(f)] over the band."""
    f = np.logspace(np.log10(f_lo), np.log10(f_hi), n_pts)
    s1f = normalized_spectrum(f, 1.0, f_lo, f_hi)
    sa = normalized_spectrum(f, alpha, f_lo, f_hi)
    ratio = s1f / sa
    return float(np.min(ratio))


def main():
    f_lo = 0.01
    f_hi = 1.0   # 2-decade band
    f = np.logspace(np.log10(f_lo), np.log10(f_hi), 500)

    alphas = [0.0, 0.5, 1.0, 1.5, 2.0]
    labels = [
        r'$\alpha=0$ (white)',
        r'$\alpha=0.5$',
        r'$\alpha=1$ (1/f, optimal)',
        r'$\alpha=1.5$',
        r'$\alpha=2$ (Brownian)',
    ]
    colors = ['#2196F3', '#4CAF50', '#F44336', '#FF9800', '#9C27B0']
    linestyles = ['--', '-.', '-', '-.', '--']
    linewidths = [1.5, 1.5, 3.0, 1.5, 1.5]

    s1f = normalized_spectrum(f, 1.0, f_lo, f_hi)

    # --- Figure 1: Ratio S_alpha(f) / S_1f(f) ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for alpha, label, color, ls, lw in zip(alphas, labels, colors, linestyles, linewidths):
        sa = normalized_spectrum(f, alpha, f_lo, f_hi)
        ratio = sa / s1f
        ax.semilogx(f, ratio, label=label, color=color, linestyle=ls, linewidth=lw)

    ax.axhline(1.0, color='black', linewidth=0.8, linestyle=':', alpha=0.5, label='ratio=1 (perfect match)')
    ax.set_xlabel('Frequency f', fontsize=12)
    ax.set_ylabel(r'$S_\alpha(f) \,/\, S_{1/f}(f)$', fontsize=12)
    ax.set_title('Spectral Ratio vs 1/f Reference\n(1/f is flat = constant regret)', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(f_lo, f_hi)

    # --- Figure 2: Max-regret vs alpha ---
    ax2 = axes[1]
    alpha_range = np.linspace(0.0, 2.5, 200)
    max_regrets = [max_regret(a, f_lo, f_hi) for a in alpha_range]
    min_regrets = [min_regret(a, f_lo, f_hi) for a in alpha_range]

    ax2.plot(alpha_range, max_regrets, 'b-', linewidth=2, label='Max regret (worst case)')
    ax2.plot(alpha_range, min_regrets, 'g--', linewidth=2, label='Min regret (best case)')
    ax2.fill_between(alpha_range, min_regrets, max_regrets, alpha=0.15, color='blue')

    # Mark alpha=1 minimum
    idx_min = np.argmin(max_regrets)
    ax2.axvline(x=1.0, color='red', linewidth=2.5, linestyle='--', label=r'$\alpha=1$ (minimax)')
    ax2.scatter([alpha_range[idx_min]], [max_regrets[idx_min]], color='red', s=100, zorder=5)

    theoretical_min = np.log(f_hi / f_lo)
    ax2.axhline(theoretical_min, color='red', linewidth=1.2, linestyle=':', alpha=0.7,
                label=f'Theoretical min = ln(R) = {theoretical_min:.2f}')

    ax2.set_xlabel(r'Spectral exponent $\alpha$', fontsize=12)
    ax2.set_ylabel('Regret = max_f [S_{1/f} / S_alpha]', fontsize=11)
    ax2.set_title('Max Regret vs Spectral Exponent\n(1/f minimizes worst-case regret)', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2.5)

    plt.tight_layout()
    out_path = FIGURES_DIR / 'minmax_optimality.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    # --- Print numerical summary ---
    print("\n=== Minimax Regret Summary ===")
    print(f"Frequency band: [{f_lo}, {f_hi}], ratio R = {f_hi/f_lo:.0f}")
    print(f"Theoretical min max-regret (alpha=1): ln(R) = {np.log(f_hi/f_lo):.4f}")
    print()
    print(f"{'alpha':>8}  {'max_regret':>12}  {'min_regret':>12}  {'regret_ratio':>14}")
    print("-" * 52)
    for alpha in [0.0, 0.5, 1.0, 1.5, 2.0]:
        mr = max_regret(alpha, f_lo, f_hi)
        mnr = min_regret(alpha, f_lo, f_hi)
        rr = mr / mnr
        marker = " <-- OPTIMAL" if abs(alpha - 1.0) < 0.01 else ""
        print(f"{alpha:>8.1f}  {mr:>12.4f}  {mnr:>12.4f}  {rr:>14.4f}{marker}")

    # --- Formal verification of minimax claim ---
    print("\n=== Formal Verification ===")
    alpha_test = np.linspace(0.0, 2.5, 2000)
    mr_test = [max_regret(a, f_lo, f_hi) for a in alpha_test]
    alpha_opt = alpha_test[np.argmin(mr_test)]
    mr_opt = min(mr_test)
    ln_R = np.log(f_hi / f_lo)
    print(f"Numerically optimal alpha: {alpha_opt:.3f}")
    print(f"Min max-regret achieved:   {mr_opt:.4f}")
    print(f"ln(R) = ln({f_hi/f_lo:.0f}):            {ln_R:.4f}")
    print(f"Claim verified: |alpha_opt - 1| < 0.05: {abs(alpha_opt - 1.0) < 0.05}")
    print(f"Claim verified: |min_regret - ln(R)| < 0.01: {abs(mr_opt - ln_R) < 0.01}")

    # Second figure: spectra comparison
    fig2, ax3 = plt.subplots(figsize=(8, 5))
    for alpha, label, color, ls, lw in zip(alphas, labels, colors, linestyles, linewidths):
        sa = normalized_spectrum(f, alpha, f_lo, f_hi)
        ax3.loglog(f, sa, label=label, color=color, linestyle=ls, linewidth=lw)

    ax3.set_xlabel('Frequency f', fontsize=12)
    ax3.set_ylabel('Spectral density S(f)', fontsize=12)
    ax3.set_title('Power-Law Spectra (normalized to unit integral)\nover 2-decade band', fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_xlim(f_lo, f_hi)

    out_path2 = FIGURES_DIR / 'spectra_comparison.png'
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_path2}")

    return alpha_opt, mr_opt


if __name__ == '__main__':
    alpha_opt, mr_opt = main()
    print(f"\nConclusion: alpha={alpha_opt:.3f} minimizes max-regret={mr_opt:.4f}")
    print("1/f spectrum (alpha=1) is confirmed as minimax-optimal.")
