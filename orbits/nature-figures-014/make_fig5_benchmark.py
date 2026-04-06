#!/usr/bin/env python3
"""Figure 5: Comprehensive Benchmark.

2x4 panel layout:
  Top row: KL convergence traces for HO, DW, GMM, RB
    Each panel: NHC, Log-Osc, MultiScale, NHCTail -- mean +/- std over 5 seeds
    Log-log axes, KL=0.01 threshold line
  Bottom row: Summary metrics
    (e) Final KL by system (grouped bars)
    (f) ESS/force-eval by system
    (g) Time-to-threshold
    (h) HO ergodicity scores
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from shared import (
    COLOR_NHC, COLOR_LO, COLOR_MS, COLOR_NHCT,
    FONTSIZE_LABEL, FONTSIZE_TICK, FONTSIZE_TITLE, FONTSIZE_ANNOT, DPI,
    SEEDS, get_potential, get_sampler_config, run_trajectory,
    compute_kl_trace, _kl_1d, _kl_2d,
    SAMPLER_LABELS, SAMPLER_COLORS,
)

N_EVALS = 50_000  # 50K for practical runtime (5 seeds x 4 systems x 4 samplers)
POT_NAMES = ['HO', 'DW', 'GMM', 'RB']
POT_TITLES = ['Harmonic Osc.', 'Double Well', 'GMM (5-mode)', 'Rosenbrock']
SAMPLER_NAMES = ['NHC', 'LogOsc', 'MultiScale', 'NHCTail']


def run_single(sampler_name, pot_name, seed):
    """Run one sampler on one potential with one seed, return trajectory."""
    pot = get_potential(pot_name)
    cls, integ_cls, dt, kw = get_sampler_config(sampler_name, pot_name)
    return run_trajectory(cls, integ_cls, pot, dt=dt, n_force_evals=N_EVALS,
                          seed=seed, **kw)


def compute_ess(samples, max_lag=2000):
    """Estimate effective sample size per sample via autocorrelation."""
    n = len(samples)
    if samples.ndim > 1:
        samples = samples[:, 0]  # use first coordinate
    mean = np.mean(samples)
    var = np.var(samples)
    if var < 1e-30:
        return 0.0

    # Compute autocorrelation
    tau = 1.0
    for lag in range(1, min(max_lag, n // 2)):
        acf = np.mean((samples[:n-lag] - mean) * (samples[lag:] - mean)) / var
        if acf < 0.05:
            break
        tau += 2 * acf

    return 1.0 / tau


def compute_ergodicity_score(qs, ps, kT=1.0):
    """Compute ergodicity score for 1D HO."""
    from scipy.stats import kstest

    burn = len(qs) // 10
    q_post = qs[burn:]
    p_post = ps[burn:]

    # KS test for position
    ks_q, _ = kstest(q_post, 'norm', args=(0, np.sqrt(kT)))
    # KS test for momentum
    ks_p, _ = kstest(p_post, 'norm', args=(0, np.sqrt(kT)))

    # Variance match
    var_q_err = abs(np.var(q_post) / kT - 1)
    var_p_err = abs(np.var(p_post) / kT - 1)

    # Phase space coverage: 20x20 grid
    q_bins = np.linspace(-4, 4, 21)
    p_bins = np.linspace(-4, 4, 21)
    H, _, _ = np.histogram2d(q_post, p_post, bins=[q_bins, p_bins])
    coverage = np.sum(H > 0) / (20 * 20)

    score = (1 - max(ks_q, ks_p)) * 0.33 + (1 - max(var_q_err, var_p_err)) * 0.33 + coverage * 0.34
    return np.clip(score, 0, 1)


def make_figure():
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    panel_labels = [chr(ord('a') + i) for i in range(8)]

    # Storage for summary metrics
    all_kl = {s: {p: [] for p in POT_NAMES} for s in SAMPLER_NAMES}
    all_ess = {s: {p: [] for p in POT_NAMES} for s in SAMPLER_NAMES}
    all_ttt = {s: {p: [] for p in POT_NAMES} for s in SAMPLER_NAMES}  # time-to-threshold
    all_ergo = {s: [] for s in SAMPLER_NAMES}

    # ── Top row: KL convergence traces ──
    for col, (pname, ptitle) in enumerate(zip(POT_NAMES, POT_TITLES)):
        ax = axes[0, col]
        pot = get_potential(pname)

        for sname in SAMPLER_NAMES:
            all_evals = []
            all_kls = []

            for seed in SEEDS:
                print(f"  {sname} on {pname}, seed={seed}...")
                result = run_single(sname, pname, seed)

                # KL trace
                evals, kl_trace = compute_kl_trace(result['q'], pot)
                all_evals.append(evals)
                all_kls.append(kl_trace)

                # Final KL
                q_post = result['q'][len(result['q'])//10:]
                if pot.dim == 1:
                    final_kl = _kl_1d(q_post.ravel(), pot, 1.0)
                else:
                    final_kl = _kl_2d(q_post, pot, 1.0)
                all_kl[sname][pname].append(final_kl)

                # ESS
                ess = compute_ess(result['q'])
                all_ess[sname][pname].append(ess / N_EVALS)

                # Time-to-threshold (first eval where KL < 0.01)
                below = np.where(kl_trace < 0.01)[0]
                if len(below) > 0:
                    all_ttt[sname][pname].append(evals[below[0]])
                else:
                    all_ttt[sname][pname].append(N_EVALS)

                # Ergodicity for HO
                if pname == 'HO':
                    ergo = compute_ergodicity_score(
                        result['q'].ravel(), result['p'].ravel() if 'p' in result else result['q'].ravel()
                    )
                    all_ergo[sname].append(ergo)

            # Interpolate to common x-axis for mean/std
            common_evals = all_evals[0]
            kl_matrix = np.array(all_kls)
            kl_mean = np.mean(kl_matrix, axis=0)
            kl_std = np.std(kl_matrix, axis=0)

            color = SAMPLER_COLORS[sname]
            label = SAMPLER_LABELS[sname]
            ax.loglog(common_evals, kl_mean, color=color, lw=2, label=label)
            ax.fill_between(common_evals,
                           np.maximum(kl_mean - kl_std, 1e-5),
                           kl_mean + kl_std,
                           color=color, alpha=0.2)

        ax.axhline(0.01, color='gray', ls='--', lw=1, alpha=0.7)
        ax.set_title(ptitle, fontsize=FONTSIZE_TITLE - 2)
        ax.set_xlabel('Force evaluations', fontsize=FONTSIZE_LABEL - 1)
        if col == 0:
            ax.set_ylabel('KL divergence', fontsize=FONTSIZE_LABEL)
        ax.tick_params(labelsize=FONTSIZE_TICK)
        if col == 0:
            ax.legend(fontsize=8, loc='upper right')
        ax.text(0.03, 0.95, f'({panel_labels[col]})', transform=ax.transAxes,
                fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    # ── Bottom row: Summary metrics ──

    # (e) Final KL by system
    ax = axes[1, 0]
    x = np.arange(len(POT_NAMES))
    width = 0.18
    for i, sname in enumerate(SAMPLER_NAMES):
        means = [np.mean(all_kl[sname][p]) for p in POT_NAMES]
        stds = [np.std(all_kl[sname][p]) for p in POT_NAMES]
        offset = (i - 1.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               color=SAMPLER_COLORS[sname], label=SAMPLER_LABELS[sname], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(POT_NAMES, fontsize=FONTSIZE_TICK)
    ax.set_ylabel('Final KL', fontsize=FONTSIZE_LABEL)
    ax.set_yscale('log')
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.text(0.03, 0.95, f'({panel_labels[4]})', transform=ax.transAxes,
            fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    # (f) ESS/force-eval by system
    ax = axes[1, 1]
    for i, sname in enumerate(SAMPLER_NAMES):
        means = [np.mean(all_ess[sname][p]) for p in POT_NAMES]
        stds = [np.std(all_ess[sname][p]) for p in POT_NAMES]
        offset = (i - 1.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               color=SAMPLER_COLORS[sname], label=SAMPLER_LABELS[sname], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(POT_NAMES, fontsize=FONTSIZE_TICK)
    ax.set_ylabel('ESS / force eval', fontsize=FONTSIZE_LABEL)
    ax.set_yscale('log')
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.text(0.03, 0.95, f'({panel_labels[5]})', transform=ax.transAxes,
            fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    # (g) Time-to-threshold
    ax = axes[1, 2]
    for i, sname in enumerate(SAMPLER_NAMES):
        means = [np.mean(all_ttt[sname][p]) for p in POT_NAMES]
        stds = [np.std(all_ttt[sname][p]) for p in POT_NAMES]
        offset = (i - 1.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               color=SAMPLER_COLORS[sname], label=SAMPLER_LABELS[sname], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(POT_NAMES, fontsize=FONTSIZE_TICK)
    ax.set_ylabel('Time to KL < 0.01', fontsize=FONTSIZE_LABEL)
    ax.set_yscale('log')
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.text(0.03, 0.95, f'({panel_labels[6]})', transform=ax.transAxes,
            fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    # (h) HO ergodicity scores
    ax = axes[1, 3]
    ergo_means = [np.mean(all_ergo[s]) for s in SAMPLER_NAMES]
    ergo_stds = [np.std(all_ergo[s]) for s in SAMPLER_NAMES]
    colors = [SAMPLER_COLORS[s] for s in SAMPLER_NAMES]
    labels = [SAMPLER_LABELS[s] for s in SAMPLER_NAMES]
    bars = ax.bar(range(len(SAMPLER_NAMES)), ergo_means, yerr=ergo_stds,
                  color=colors, capsize=5, alpha=0.8)
    ax.set_xticks(range(len(SAMPLER_NAMES)))
    ax.set_xticklabels(labels, fontsize=FONTSIZE_TICK - 1, rotation=15)
    ax.set_ylabel('Ergodicity score', fontsize=FONTSIZE_LABEL)
    ax.axhline(0.85, color='gray', ls='--', lw=1, alpha=0.5, label='Threshold')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=FONTSIZE_ANNOT)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.text(0.03, 0.95, f'({panel_labels[7]})', transform=ax.transAxes,
            fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    fig.suptitle('Comprehensive Benchmark: 4 Potentials, 4 Samplers, 5 Seeds',
                 fontsize=FONTSIZE_TITLE, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fig5_benchmark.png')
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Print summary table
    print("\n=== BENCHMARK SUMMARY ===")
    for pname in POT_NAMES:
        print(f"\n{pname}:")
        for sname in SAMPLER_NAMES:
            kl_m = np.mean(all_kl[sname][pname])
            kl_s = np.std(all_kl[sname][pname])
            ess_m = np.mean(all_ess[sname][pname])
            ttt_m = np.mean(all_ttt[sname][pname])
            print(f"  {SAMPLER_LABELS[sname]:15s}: KL={kl_m:.4f}+/-{kl_s:.4f}, "
                  f"ESS/eval={ess_m:.6f}, TTT={ttt_m:.0f}")

    print("\nErgodicity (HO):")
    for sname in SAMPLER_NAMES:
        e_m = np.mean(all_ergo[sname])
        e_s = np.std(all_ergo[sname])
        print(f"  {SAMPLER_LABELS[sname]:15s}: {e_m:.3f}+/-{e_s:.3f}")

    return out_path


if __name__ == '__main__':
    make_figure()
