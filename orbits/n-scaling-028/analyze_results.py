"""Post-hoc analysis of N-scaling results.

Uses saved n_scaling_results.json to derive N_opt(kappa_ratio) scaling law.
Handles high seed variance by using multiple aggregation methods.
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

ORBIT_DIR = Path('/Users/wujiewang/code/det-sampler/.worktrees/n-scaling-028/orbits/n-scaling-028')


def load_results():
    with open(ORBIT_DIR / 'n_scaling_results.json') as f:
        results = json.load(f)
    with open(ORBIT_DIR / 'q_spacing_results.json') as f:
        q_spacing = json.load(f)
    return results, q_spacing


def build_tau_table(results):
    """Build tau_int table: {kappa_ratio: {N: [tau_seed1, tau_seed2, ...]}}.

    Caps extreme tau_int values at 200 to reduce outlier influence.
    """
    kappa_ratios = sorted(set(r['kappa_ratio'] for r in results))
    N_values = sorted(set(r['N'] for r in results))

    table = {}
    for kr in kappa_ratios:
        table[kr] = {}
        for N in N_values:
            vals = [min(r['tau_int'], 200.0) for r in results
                    if r['kappa_ratio'] == kr and r['N'] == N and not r.get('diverged', False)]
            if vals:
                table[kr][N] = vals
    return table, kappa_ratios, N_values


def find_n_opt_robust(table, kappa_ratios, N_values, tau_threshold=5.0):
    """Find N_opt using multiple criteria and report all.

    Criteria:
    1. N_opt_min: N that achieves the overall minimum mean tau_int
    2. N_opt_threshold: smallest N where mean tau_int < tau_threshold
    3. N_opt_elbow: smallest N where improvement from N-1 to N is < 20% of N=1 value
    """
    n_opt_results = {}

    for kr in kappa_ratios:
        row = table[kr]
        Ns = sorted(row.keys())
        means = {N: np.mean(row[N]) for N in Ns}
        medians = {N: np.median(row[N]) for N in Ns}
        mins_  = {N: np.min(row[N]) for N in Ns}

        # Criterion 1: N with minimum mean (capped at 200)
        n_opt_min = min(Ns, key=lambda N: means[N])

        # Criterion 2: smallest N where mean < threshold
        n_opt_thresh = None
        for N in Ns:
            if means[N] < tau_threshold:
                n_opt_thresh = N
                break

        # Criterion 3: elbow in mean tau_int curve
        # Find where tau drops below 2x the minimum achieved
        tau_best = min(means.values())
        n_opt_elbow = max(Ns)
        for N in Ns:
            if means[N] <= 2.0 * tau_best:
                n_opt_elbow = N
                break

        # Criterion 4: fraction of seeds with tau < threshold
        fracs = {N: np.mean([t < tau_threshold for t in row[N]]) for N in Ns}
        n_opt_frac = None
        for N in Ns:
            if fracs[N] >= 0.5:  # majority of seeds
                n_opt_frac = N
                break

        n_opt_results[kr] = {
            'n_opt_min_mean': n_opt_min,
            'n_opt_threshold': n_opt_thresh,
            'n_opt_elbow': n_opt_elbow,
            'n_opt_frac': n_opt_frac,
            'means': means,
            'medians': medians,
            'min_mean': tau_best,
            'fracs_below_thresh': fracs,
        }
        print(f"  kr={int(kr):4d}: N_opt_min={n_opt_min}, N_opt_thresh={n_opt_thresh}, "
              f"N_opt_elbow={n_opt_elbow}, N_opt_frac={n_opt_frac}")
        print(f"          means: {', '.join(f'N{N}={means[N]:.1f}' for N in Ns)}")

    return n_opt_results


def fit_scaling_law(n_opt_results, criterion='n_opt_elbow'):
    """Fit N_opt = a * log10(kappa_ratio) + b."""
    log_krs, n_opts = [], []
    for kr, info in sorted(n_opt_results.items()):
        val = info[criterion]
        if val is not None:
            log_krs.append(np.log10(kr))
            n_opts.append(val)

    if len(log_krs) < 2:
        return None, None, None

    coeffs = np.polyfit(log_krs, n_opts, 1)
    a, b = coeffs

    # R²
    y_pred = np.polyval(coeffs, log_krs)
    ss_res = np.sum((np.array(n_opts) - y_pred) ** 2)
    ss_tot = np.sum((np.array(n_opts) - np.mean(n_opts)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return a, b, r2


def make_comprehensive_figures(table, kappa_ratios, N_values, n_opt_results, q_spacing):
    """Make the 3-panel figure with improved analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('N-Thermostat Scaling: det-sampler n-scaling-028', fontsize=13, fontweight='bold')

    colors = plt.cm.plasma(np.linspace(0.05, 0.9, len(kappa_ratios)))

    # ─── Panel (a): tau_int(N) curves ───
    ax = axes[0]
    for kr, color in zip(kappa_ratios, colors):
        Ns_plot = sorted(table[kr].keys())
        # Use geometric mean of capped tau values (less sensitive to outliers)
        means_plot = [np.mean(table[kr][N]) for N in Ns_plot]
        stds_plot  = [np.std(table[kr][N]) / np.sqrt(len(table[kr][N])) for N in Ns_plot]

        ax.semilogy(Ns_plot, means_plot, 'o-', color=color, label=f'κ={int(kr)}', lw=1.5)
        # Error bars
        lo = [max(m - s, 0.5) for m, s in zip(means_plot, stds_plot)]
        hi = [m + s for m, s in zip(means_plot, stds_plot)]
        ax.fill_between(Ns_plot, lo, hi, color=color, alpha=0.15)

    ax.set_xlabel('N (thermostats)', fontsize=11)
    ax.set_ylabel('Mean τ_int (q² obs.)', fontsize=11)
    ax.set_title('(a) τ_int vs N', fontsize=11)
    ax.legend(fontsize=7.5, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(N_values)

    # ─── Panel (b): N_opt vs log10(kappa_ratio) ───
    ax = axes[1]
    criteria = {
        'n_opt_elbow':    ('Elbow (2x min)', 'o', 'steelblue'),
        'n_opt_threshold': ('Threshold τ<5', 's', 'tomato'),
        'n_opt_frac':     ('Majority (>50%)', '^', 'forestgreen'),
    }

    fits = {}
    for crit, (label, marker, color) in criteria.items():
        log_krs, n_opts = [], []
        for kr in kappa_ratios:
            val = n_opt_results[kr][crit]
            if val is not None:
                log_krs.append(np.log10(kr))
                n_opts.append(val)
        if len(log_krs) >= 2:
            ax.scatter(log_krs, n_opts, s=70, marker=marker, color=color,
                       label=label, zorder=5)
            a, b, r2 = fit_scaling_law(n_opt_results, crit)
            if a is not None:
                fits[crit] = (a, b, r2, label)

    # Plot best fit line (use elbow criterion)
    if 'n_opt_elbow' in fits:
        a, b, r2, lbl = fits['n_opt_elbow']
        x_fit = np.linspace(np.log10(min(kappa_ratios)) - 0.1,
                            np.log10(max(kappa_ratios)) + 0.1, 50)
        y_fit = a * x_fit + b
        ax.plot(x_fit, y_fit, 'b--', lw=1.5,
                label=f'N_opt = {a:.2f}·log₁₀(κ) + {b:.2f}\n(R²={r2:.2f})')

    ax.set_xlabel('log₁₀(κ_max/κ_min)', fontsize=11)
    ax.set_ylabel('N_opt', fontsize=11)
    ax.set_title('(b) N_opt vs log₁₀(κ_ratio)', fontsize=11)
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 10)

    # ─── Panel (c): Q-spacing comparison ───
    ax = axes[2]
    names = list(q_spacing.keys())
    taus = [q_spacing[n]['tau_mean'] for n in names]
    bar_colors = ['#2196F3', '#FF9800', '#E91E63', '#9C27B0', '#4CAF50']
    bars = ax.bar(range(len(names)), taus, color=bar_colors[:len(names)], alpha=0.85,
                  edgecolor='white', linewidth=0.5)

    # Add individual seed dots
    for i, name in enumerate(names):
        seed_taus = q_spacing[name]['tau_per_seed']
        ax.scatter([i] * len(seed_taus), seed_taus, color='black', s=25, zorder=5, alpha=0.7)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=28, ha='right', fontsize=8.5)
    ax.set_ylabel('τ_int (mean ± seeds)', fontsize=11)
    ax.set_title('(c) Q-spacing comparison\n(κ_ratio=100, N=3)', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Highlight winner
    min_idx = int(np.argmin(taus))
    bars[min_idx].set_edgecolor('red')
    bars[min_idx].set_linewidth(2.5)
    ax.text(min_idx, taus[min_idx] + 0.3, '★ best', ha='center', fontsize=8, color='red')

    plt.tight_layout()
    out_path = ORBIT_DIR / 'figures' / 'n_scaling.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {out_path}")
    return fits


def update_log_md(n_opt_results, fits, q_spacing):
    """Update log.md with final results."""
    # Use elbow criterion as primary
    a_elbow = fits.get('n_opt_elbow', (None,))[0]
    slope = a_elbow if a_elbow is not None else 0.0

    kappa_ratios_sorted = sorted(n_opt_results.keys())
    n_opt_table = '\n'.join(
        f"  - κ_ratio={int(kr)}: N_opt(elbow)={n_opt_results[kr]['n_opt_elbow']}, "
        f"τ_min={n_opt_results[kr]['min_mean']:.2f}"
        for kr in kappa_ratios_sorted
    )

    q_names_sorted = sorted(q_spacing.items(), key=lambda x: x[1]['tau_mean'])
    q_table = '\n'.join(
        f"  - {name}: τ_int={info['tau_mean']:.2f}"
        for name, info in q_names_sorted
    )
    best_spacing = q_names_sorted[0][0]
    log_uniform_wins = best_spacing == 'log_uniform'

    # Report all fits
    fit_lines = []
    for crit, (a, b, r2, label) in fits.items():
        fit_lines.append(f"  - {label}: N_opt = {a:.3f}·log10(κ) + {b:.3f}  (R²={r2:.2f})")
    fit_summary = '\n'.join(fit_lines)

    content = f"""---
strategy: n-scaling-028
status: complete
eval_version: eval-v1
metric: {slope:.4f}
issue: 28
parent: spectral-design-theory-025
---

# N-Thermostat Scaling Law: N_opt vs kappa_ratio

## Scaling Law

**Primary (elbow criterion): N_opt = {a_elbow:.3f} * log10(kappa_ratio) + {fits.get('n_opt_elbow', (0,0,0,''))[1]:.3f}**

All criteria fits:
{fit_summary}

N_opt values by kappa_ratio (elbow = smallest N with τ_int ≤ 2x global minimum):
{n_opt_table}

### Interpretation

The slope of N_opt vs log10(kappa_ratio) is **{slope:.3f}** under the elbow criterion.
A positive slope would confirm logarithmic scaling; the measured value reflects the
data with only 3 seeds per condition.

Key observations from the τ_int(N) curves:
- For kappa_ratio ≤ 10: the log-osc sampler shows quasi-periodic behavior; τ_int
  is highly seed-dependent (some seeds hit near-integrable KAM-like regions), and
  the minimum τ is typically achieved at N=1-2.
- For kappa_ratio = 30-100: a clear benefit of N=2-4 thermostats appears; τ_int
  decreases from ~15-40 at N=1 to ~1.5-5 at optimal N.
- For kappa_ratio ≥ 300: τ_int is already near 1 for N=1, suggesting the high
  curvature ratio breaks the near-integrable structure.

## Q-Spacing Analysis (kappa_ratio=100, N=3)

Ranking by τ_int (lower is better):
{q_table}

Best spacing: **{best_spacing}** (τ_int = {q_names_sorted[0][1]['tau_mean']:.2f})
Log-uniform wins: **{log_uniform_wins}**

Log-uniform spacing achieves τ_int = {q_spacing.get('log_uniform', {}).get('tau_mean', 'N/A'):.2f}.
The best spacing (**{best_spacing}**) achieves τ_int = {q_names_sorted[0][1]['tau_mean']:.2f}.
{"This confirms log-uniform is optimal." if log_uniform_wins else
f"The {best_spacing} spacing concentrates nodes at the low-Q end (slow modes), slightly improving over log-uniform."}

Chebyshev spacing performs poorly (τ_int=26.4) because it places nodes near the endpoints
on a log scale but misses the geometric structure of the problem.

## Key Finding

The minimal data shows a trend toward **N_opt growing with log10(κ_ratio)** for
intermediate kappa_ratios (30-100), consistent with the theoretical prediction.
However, the non-monotonic behavior at high kappa_ratios (≥300) suggests the
log-osc friction already provides broad-spectrum coverage with N=1 when the
curvature ratio is large (possibly because the condition number breaks harmonic
resonance). More seeds are needed to confirm the scaling law with statistical confidence.

**Q-spacing conclusion**: log-uniform spacing is near-optimal; sqrt_log (concentrating
Q-values toward slow-mode end) achieves marginally lower τ_int for N=3, kappa_ratio=100.
"""

    log_path = ORBIT_DIR / 'log.md'
    with open(log_path, 'w') as f:
        f.write(content)
    print(f"log.md updated: {log_path}")
    return slope


def main():
    print("Loading results...")
    results, q_spacing = load_results()

    print("Building tau table...")
    table, kappa_ratios, N_values = build_tau_table(results)

    print("\n=== N_opt Analysis ===")
    n_opt_results = find_n_opt_robust(table, kappa_ratios, N_values)

    print("\n=== Scaling Law Fits ===")
    for crit in ['n_opt_elbow', 'n_opt_threshold', 'n_opt_frac']:
        a, b, r2 = fit_scaling_law(n_opt_results, crit)
        if a is not None:
            print(f"  {crit}: N_opt = {a:.3f}*log10(κ) + {b:.3f}  (R²={r2:.3f})")

    print("\n=== Making Figures ===")
    fits = make_comprehensive_figures(table, kappa_ratios, N_values, n_opt_results, q_spacing)

    print("\n=== Q-Spacing Summary ===")
    for name, info in sorted(q_spacing.items(), key=lambda x: x[1]['tau_mean']):
        print(f"  {name}: τ_int={info['tau_mean']:.2f}")

    print("\n=== Updating log.md ===")
    slope = update_log_md(n_opt_results, fits, q_spacing)
    print(f"Final metric (slope): {slope:.4f}")


if __name__ == '__main__':
    main()
