"""Publication figures for the 1/f thermostat paper.

Generates 4 figures from completed orbit data:
  Fig 2: 1/f optimality (alpha-031)
  Fig 3: Resonance structure (chirikov-032 + ergodicity-phase-diagram-027)
  Fig 4: N-scaling (n-scaling-robust-029, or n-scaling-028 as fallback)

Run from the orbit directory. Requires completed sibling orbits.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Roots
REPO = Path('/Users/wujiewang/code/det-sampler')

# Data sources
ALPHA_RESULTS   = REPO / '.worktrees/alpha-spectrum-comparison-031/orbits/alpha-spectrum-comparison-031/alpha_results.json'
CHIRIKOV_RESULTS= REPO / '.worktrees/chirikov-exponent-032/orbits/chirikov-exponent-032/chirikov_results.json'
N_SCALING_029   = REPO / '.worktrees/n-scaling-robust-029/orbits/n-scaling-robust-029/n_scaling_results.json'
N_SCALING_028   = REPO / '.worktrees/n-scaling-028/orbits/n-scaling-028/n_scaling_results.json'
PHASE_027       = REPO / '.worktrees/ergodicity-phase-diagram-027/orbits/ergodicity-phase-diagram-027'

ORBIT_DIR = Path('/Users/wujiewang/code/det-sampler/.worktrees/figure-update-033/orbits/figure-update-033')

# ── Style ──────────────────────────────────────────────────────────────────────
COLORS = {
    'alpha0':  '#E91E63',  # pink
    'alpha05': '#FF9800',  # orange
    'alpha1':  '#2E7D32',  # dark green (1/f winner)
    'alpha15': '#1565C0',  # dark blue
    'alpha2':  '#6A1B9A',  # purple
    'resonance': '#D32F2F',
    'theory':  '#388E3C',
}
ALPHA_LABELS = {0.0: 'α=0 (white)', 0.5: 'α=0.5', 1.0: 'α=1 (1/f)', 1.5: 'α=1.5', 2.0: 'α=2 (red)'}
ALPHA_COLORS = {0.0: COLORS['alpha0'], 0.5: COLORS['alpha05'], 1.0: COLORS['alpha1'],
                1.5: COLORS['alpha15'], 2.0: COLORS['alpha2']}

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.8,
    'ytick.major.width': 0.8, 'lines.linewidth': 1.5,
})


# ── Fig 2: 1/f optimality ─────────────────────────────────────────────────────

def make_fig2():
    """Two-panel: (a) tau_int vs alpha, (b) ergodicity score vs alpha."""
    with open(ALPHA_RESULTS) as f:
        results = json.load(f)

    alphas = sorted(set(r['alpha'] for r in results))

    tau_means, tau_stds = [], []
    erg_means, erg_stds = [], []
    for a in alphas:
        runs = [r for r in results if r['alpha'] == a and not r.get('diverged', False)]
        taus = [r['tau_int'] for r in runs]
        ergs = [r['ergodicity_score'] for r in runs]
        tau_means.append(np.mean(taus)); tau_stds.append(np.std(taus))
        erg_means.append(np.mean(ergs)); erg_stds.append(np.std(ergs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    fig.suptitle('1/f Spectrum is Uniquely Ergodicity-Optimal', fontsize=10, y=1.02)

    colors = [ALPHA_COLORS[a] for a in alphas]
    x = np.arange(len(alphas))

    # Panel a: tau_int
    bars = ax1.bar(x, tau_means, yerr=tau_stds, color=colors, alpha=0.85,
                   capsize=4, error_kw={'linewidth': 0.8})
    ax1.set_xticks(x)
    ax1.set_xticklabels([ALPHA_LABELS[a] for a in alphas], rotation=15, ha='right', fontsize=8)
    ax1.set_ylabel('τ_int (integrated autocorr. time)')
    ax1.set_title('(a) Raw mixing speed\n(lower = faster)')
    # Annotate: α=1 marker
    idx1 = alphas.index(1.0)
    ax1.annotate('1/f', xy=(idx1, tau_means[idx1]),
                 xytext=(idx1+0.4, tau_means[idx1]*1.3),
                 fontsize=8, color=COLORS['alpha1'],
                 arrowprops=dict(arrowstyle='->', color=COLORS['alpha1'], lw=0.8))
    ax1.grid(axis='y', alpha=0.3)

    # Panel b: ergodicity score
    bars2 = ax2.bar(x, erg_means, yerr=erg_stds, color=colors, alpha=0.85,
                    capsize=4, error_kw={'linewidth': 0.8})
    ax2.set_xticks(x)
    ax2.set_xticklabels([ALPHA_LABELS[a] for a in alphas], rotation=15, ha='right', fontsize=8)
    ax2.set_ylabel('Ergodicity score\n(frac. dims within 20% of true var)')
    ax2.set_title('(b) Correct variance coverage\n(higher = better)')
    ax2.set_ylim(0, max(erg_means) * 1.5 + 0.05)
    ax2.annotate('α>1: fast but\nwrong variance', xy=(alphas.index(1.5), erg_means[alphas.index(1.5)]),
                 xytext=(alphas.index(1.5)+0.3, erg_means[alphas.index(1.5)]+0.03),
                 fontsize=7, color=COLORS['alpha15'],
                 arrowprops=dict(arrowstyle='->', color=COLORS['alpha15'], lw=0.8))
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = ORBIT_DIR / 'figures' / 'fig2_1f_optimality.png'
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Fig 2 saved → {out.name}")
    return out


# ── Fig 3: Resonance structure ────────────────────────────────────────────────

def make_fig3():
    """Two-panel: (a) C(κ) resonance structure, (b) schematic of mechanism."""
    with open(CHIRIKOV_RESULTS) as f:
        raw = json.load(f)
    results = {float(k): v for k, v in raw.items()}

    kappas_all = sorted(results.keys())
    kappas_found = [k for k in kappas_all if results[k]['crit_ratio'] is not None]
    crits_found  = [results[k]['crit_ratio'] for k in kappas_found]
    kappas_nf    = [k for k in kappas_all if results[k]['crit_ratio'] is None]

    omega_found = [np.sqrt(k) for k in kappas_found]
    omega_nf    = [np.sqrt(k) for k in kappas_nf]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    fig.suptitle('Ergodicity Requires Incommensurate Thermostat Frequencies', fontsize=10, y=1.02)

    # Panel (a): C(κ) vs ω×Q₁
    ax = axes[0]
    ax.scatter(omega_found, crits_found, s=60, color='steelblue', zorder=5, label='C(κ) measured')
    ax.plot(omega_found, crits_found, '-', color='steelblue', alpha=0.4)

    # NOT FOUND points at ceiling
    if kappas_nf:
        ax.scatter(omega_nf, [100]*len(kappas_nf), s=80, color=COLORS['resonance'],
                   marker='^', zorder=5, label='C > 100 (resonance)')
        for k, w in zip(kappas_nf, omega_nf):
            ax.annotate('resonance\nsingularity', xy=(w, 100), xytext=(w+0.5, 60),
                        fontsize=7, color=COLORS['resonance'],
                        arrowprops=dict(arrowstyle='->', color=COLORS['resonance'], lw=0.8))

    ax.axvline(1.0, color=COLORS['resonance'], linestyle='--', alpha=0.7,
               linewidth=1.2, label='ω×Q₁ = 1 (resonance)')
    ax.axhline(1.05, color='gray', linestyle=':', alpha=0.4)
    ax.set_yscale('log')

    ax.set_xlabel('ω × Q₁  (oscillator freq × thermostat time)')
    ax.set_ylabel('C(κ) = min Q₂/Q₁ for ergodicity')
    ax.set_title('(a) Non-monotonic critical ratio\nsingularity at ω×Q₁ = 1, drops to 1 for ω>>1')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, which='both')

    # Annotations for key regions
    ax.text(0.25, 1.3, 'sub-resonance\n(C~1.5-1.7)', ha='center', fontsize=7, color='gray')
    ax.text(3.5, 5.0, 'near-resonance\n(C=8)', ha='center', fontsize=7, color='steelblue')
    ax.text(12, 1.15, 'fast oscillators\n(C→1)', ha='center', fontsize=7, color='gray')

    # Panel (b): Design implication — show how F1 design avoids resonance
    ax = axes[1]
    ax.set_xlim(0, 3.5)
    ax.set_ylim(-0.5, 4)
    ax.axis('off')
    ax.set_title('(b) The F1 design principle\n(log-uniform Q spans all mode frequencies)')

    # Draw the frequency axis
    ax.annotate('', xy=(3.2, 1.5), xytext=(0.2, 1.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.text(3.3, 1.5, 'ω', fontsize=10, va='center')
    ax.text(0.2, 1.2, 'slow\n(κ_min)', fontsize=7, ha='center', color='gray')
    ax.text(2.9, 1.2, 'fast\n(κ_max)', fontsize=7, ha='center', color='gray')

    # Q values (log-spaced thermostats)
    qs = [0.4, 0.8, 1.4, 2.1, 2.8]
    colors_q = ['#1B5E20', '#2E7D32', '#388E3C', '#43A047', '#66BB6A']
    for i, (q, c) in enumerate(zip(qs, colors_q)):
        ax.annotate('', xy=(q, 1.5), xytext=(q, 2.6),
                    arrowprops=dict(arrowstyle='->', lw=2, color=c))
        ax.text(q, 2.8, f'Q_{i+1}', fontsize=7, ha='center', color=c)

    ax.text(1.7, 3.3, 'N log-uniform thermostats\ncover all resonances', ha='center',
            fontsize=8, color=COLORS['alpha1'])
    ax.text(1.7, 0.5, '→ 1/f spectrum\n→ K(t) ~ 1/t memory kernel', ha='center',
            fontsize=8, style='italic', color='steelblue')

    plt.tight_layout()
    out = ORBIT_DIR / 'figures' / 'fig3_resonance.png'
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Fig 3 saved → {out.name}")
    return out


# ── Fig 4: N-scaling ──────────────────────────────────────────────────────────

def make_fig4():
    """N_opt vs log10(kappa_ratio), with error bands and linear fit.
    Uses orbit 029 if available, otherwise falls back to 028.
    """
    if N_SCALING_029.exists():
        results_path = N_SCALING_029
        label_suffix = '(10 seeds, 800k evals)'
        orbit_label = 'orbit #029'
        print("Using n-scaling-029 data")
    elif N_SCALING_028.exists():
        results_path = N_SCALING_028
        label_suffix = '(3 seeds, 400k evals — prelim.)'
        orbit_label = 'orbit #028'
        print("Falling back to n-scaling-028 data")
    else:
        print("No N-scaling data available — skipping Fig 4")
        return None

    with open(results_path) as f:
        results = json.load(f)

    kappa_ratios = sorted(set(r['kappa_ratio'] for r in results))
    N_values = sorted(set(r['N'] for r in results))

    # Compute N_opt per kappa_ratio (median over seeds)
    summary = {}
    for kr in kappa_ratios:
        tau_by_N = {}
        for N in N_values:
            vals = [r['tau_int'] for r in results
                    if r['kappa_ratio'] == kr and r['N'] == N and not r.get('diverged', False)]
            if vals:
                tau_by_N[N] = {'mean': np.mean(vals), 'std': np.std(vals), 'median': np.median(vals)}
        if not tau_by_N:
            continue
        Ns = sorted(tau_by_N.keys())
        medians = [tau_by_N[n]['median'] for n in Ns]
        means   = [tau_by_N[n]['mean'] for n in Ns]
        stds    = [tau_by_N[n]['std'] for n in Ns]
        n_opt = Ns[int(np.argmin(medians))]
        tau_n1 = tau_by_N[1]['mean'] if 1 in tau_by_N else None
        tau_nopt = tau_by_N[n_opt]['mean']
        gain = tau_n1 / tau_nopt if tau_n1 else None
        summary[kr] = {
            'n_opt': n_opt, 'gain': gain,
            'tau_by_N': tau_by_N, 'Ns': Ns, 'means': means, 'stds': stds,
        }

    if len(summary) < 3:
        print("Insufficient data for Fig 4 — skipping")
        return None

    from scipy import stats as scipy_stats
    log_krs = np.array([np.log10(kr) for kr in sorted(summary.keys())])
    n_opts  = np.array([summary[kr]['n_opt'] for kr in sorted(summary.keys())])
    res = scipy_stats.linregress(log_krs, n_opts)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    fig.suptitle(f'N_opt Scales Logarithmically with Condition Number {label_suffix}', fontsize=9, y=1.02)

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(kappa_ratios)))

    # Panel (a): tau_int(N) curves
    ax = axes[0]
    for kr, color in zip(kappa_ratios, colors):
        if kr not in summary:
            continue
        s = summary[kr]
        ax.semilogy(s['Ns'], s['means'], 'o-', color=color, linewidth=1.5, markersize=4,
                    label=f'κ={int(kr)}')
        means = np.array(s['means']); stds = np.array(s['stds'])
        ax.fill_between(s['Ns'], np.maximum(means - stds, 0.1), means + stds,
                        color=color, alpha=0.12)

    ax.set_xlabel('N (number of thermostats)')
    ax.set_ylabel('τ_int (mean ± std)')
    ax.set_title(f'(a) τ_int vs N for each κ_ratio\n{orbit_label}')
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    ax.grid(alpha=0.3)

    # Panel (b): N_opt vs log10(kappa_ratio) + fit
    ax = axes[1]
    krs_sorted = sorted(summary.keys())
    n_opts_sorted = [summary[kr]['n_opt'] for kr in krs_sorted]
    log_krs_sorted = [np.log10(kr) for kr in krs_sorted]

    ax.scatter(log_krs_sorted, n_opts_sorted, s=80, color='steelblue',
               zorder=5, label=f'N_opt (median)')

    # Linear fit line
    x_fit = np.linspace(min(log_krs_sorted) - 0.1, max(log_krs_sorted) + 0.1, 50)
    y_fit = res.slope * x_fit + res.intercept
    ax.plot(x_fit, y_fit, 'r--', linewidth=1.5,
            label=f'N_opt = {res.slope:.2f}·log₁₀(κ) + {res.intercept:.2f}\n'
                  f'R²={res.rvalue**2:.2f}, p={res.pvalue:.3f}')

    ax.set_xlabel('log₁₀(κ_max/κ_min)')
    ax.set_ylabel('N_opt')
    ax.set_title('(b) Log-linear scaling law:\nmore curvature range → more thermostats needed')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = ORBIT_DIR / 'figures' / 'fig4_n_scaling.png'
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Fig 4 saved → {out.name}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    figs_made = []

    print("=== Fig 2: 1/f optimality ===")
    if ALPHA_RESULTS.exists():
        figs_made.append(make_fig2())
    else:
        print(f"  SKIP: {ALPHA_RESULTS} not found")

    print("=== Fig 3: Resonance structure ===")
    if CHIRIKOV_RESULTS.exists():
        figs_made.append(make_fig3())
    else:
        print(f"  SKIP: {CHIRIKOV_RESULTS} not found")

    print("=== Fig 4: N-scaling ===")
    figs_made.append(make_fig4())

    made = [f for f in figs_made if f is not None]
    print(f"\nDone. {len(made)} figures generated:")
    for f in made:
        print(f"  {f}")

    # Update log
    log = ORBIT_DIR / 'log.md'
    with open(log, 'w') as fp:
        fp.write(f"""---
strategy: figure-update-033
status: complete
eval_version: eval-v1
metric: {len(made):.1f}
issue: 33
parent: spectral-design-theory-025
---

# Figure Update

Generated {len(made)} publication figures from completed orbits.

## Figures
""")
        for f in made:
            fp.write(f"- `{f.name}`\n")
        fp.write("""
## Data sources
- Fig 2: alpha-spectrum-comparison-031/alpha_results.json
- Fig 3: chirikov-exponent-032/chirikov_results.json
- Fig 4: n-scaling-robust-029 (or n-scaling-028 fallback)
""")
    print("log.md updated")


if __name__ == '__main__':
    main()
