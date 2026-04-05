"""Consolidated Figure: 2x3 Nature-style panel for 1/f spectral analysis.

Panels:
(a) PSDs for N=1,3,5,10 with 1/f reference line
(b) Lorentzian decomposition for N=3, Q=[0.1,1,10]
(c) alpha vs N (fitted exponent)
(d) Friction time series: N=1 vs N=3
(e) GMM KL vs N_scales
(f) Spectral match comparison (bar chart)
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def lorentzian(f, A, tau):
    return A * tau / (1.0 + (2.0 * np.pi * f * tau)**2)


def main():
    outdir = os.path.dirname(os.path.abspath(__file__))
    figdir = os.path.join(outdir, 'figures')

    # Load all data
    with open(os.path.join(outdir, 'psd_results.json'), 'r') as f:
        psd_data = json.load(f)

    with open(os.path.join(outdir, 'lorentzian_fits.json'), 'r') as f:
        lor_fits = json.load(f)

    spec_path = os.path.join(outdir, 'spectral_match_results.json')
    has_spec = os.path.exists(spec_path)
    if has_spec:
        with open(spec_path, 'r') as f:
            spec_data = json.load(f)

    gmm_path = os.path.join(outdir, 'gmm_vs_n_results.json')
    has_gmm = os.path.exists(gmm_path)
    if has_gmm:
        with open(gmm_path, 'r') as f:
            gmm_data = json.load(f)

    # --- FIGURE ---
    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.30)

    # Color scheme (style.md)
    nhc_color = '#ff7f0e'
    novel_colors = {1: '#888888', 3: plt.cm.tab10(2), 5: plt.cm.tab10(3),
                    7: plt.cm.tab10(4), 10: plt.cm.tab10(5)}

    # ===== Panel (a): PSDs for N=1,3,5,10 =====
    ax = fig.add_subplot(gs[0, 0])
    for N in [1, 3, 5, 10]:
        key = str(N)
        if key not in psd_data:
            continue
        freqs = np.array(psd_data[key]['freqs'])
        psd = np.array(psd_data[key]['psd'])
        mask = freqs > 0
        ax.loglog(freqs[mask], psd[mask], color=novel_colors[N],
                  linewidth=1.8, alpha=0.85, label=f'N={N}')

    # 1/f reference
    f_ref = np.logspace(-1.5, 1.5, 100)
    d3 = psd_data.get('3')
    if d3:
        f3 = np.array(d3['freqs'])
        p3 = np.array(d3['psd'])
        m3 = f3 > 0
        idx05 = np.argmin(np.abs(f3[m3] - 0.5))
        ref_scale = p3[m3][idx05] * 0.5
        ax.loglog(f_ref, ref_scale / f_ref, 'k--', linewidth=2,
                  alpha=0.4, label='$1/f$')
        # 1/f^2 reference
        ax.loglog(f_ref, ref_scale * 0.5 / f_ref**2, 'k:', linewidth=1.5,
                  alpha=0.3, label='$1/f^2$')

    ax.set_xlabel('Frequency (Hz)', fontsize=14)
    ax.set_ylabel('PSD of $g_{\\mathrm{total}}(t)$', fontsize=14)
    ax.set_title('(a) Friction PSD vs N scales', fontsize=16)
    ax.legend(fontsize=10, loc='lower left')
    ax.tick_params(labelsize=12)
    ax.set_xlim([1e-2, 100])
    ax.grid(True, alpha=0.2, which='both')

    # ===== Panel (b): Lorentzian decomposition =====
    ax = fig.add_subplot(gs[0, 1])
    decomp_colors = [plt.cm.tab10(2), plt.cm.tab10(3), plt.cm.tab10(4)]

    # Plot per-thermostat PSDs from N=3 decomposition data
    # Use per_thermo_N3 from psd_results (wide Q range)
    ptd = psd_data.get('per_thermo_N3', {})
    for k_str, data in ptd.items():
        k = int(k_str)
        fk = np.array(data['freqs'])
        pk = np.array(data['psd'])
        Q = data['Q']
        mask = fk > 0
        ax.loglog(fk[mask], pk[mask], color=decomp_colors[k % 3],
                  alpha=0.5, linewidth=1.0,
                  label=f'$g(\\xi_{{{k+1}}})$, Q={Q:.2g}')

    # Lorentzian fits overlay
    f_fit = np.logspace(-2, 2, 400)
    psd_sum = np.zeros_like(f_fit)
    for k_str, fit in lor_fits.items():
        k = int(k_str)
        A, tau = fit['A'], fit['tau']
        lor_k = lorentzian(f_fit, A, tau)
        psd_sum += lor_k
        ax.loglog(f_fit, lor_k, '--', color=decomp_colors[k % 3],
                  linewidth=2, alpha=0.6)

    ax.loglog(f_fit, psd_sum, 'r-', linewidth=2.5, alpha=0.9,
              label='$\\Sigma$ Lorentzians')

    # 1/f reference
    if len(lor_fits) > 0:
        # Normalize 1/f line to match sum at f=0.3
        idx_03 = np.argmin(np.abs(f_fit - 0.3))
        scale_1f = psd_sum[idx_03] * 0.3
        f_1f = np.logspace(-1, 1.5, 100)
        ax.loglog(f_1f, scale_1f / f_1f, 'k--', linewidth=1.5, alpha=0.4,
                  label='$1/f$')

    ax.set_xlabel('Frequency (Hz)', fontsize=14)
    ax.set_ylabel('PSD', fontsize=14)
    ax.set_title('(b) Lorentzian decomposition (N=3)', fontsize=16)
    ax.legend(fontsize=9, loc='lower left')
    ax.tick_params(labelsize=12)
    ax.set_xlim([1e-2, 100])
    ax.grid(True, alpha=0.2, which='both')

    # ===== Panel (c): alpha vs N =====
    ax = fig.add_subplot(gs[0, 2])
    N_list = [1, 3, 5, 7, 10]
    alphas_dh = []
    alphas_dh_err = []
    alphas_mid = []
    alphas_mid_err = []

    for N in N_list:
        key = str(N)
        if key in psd_data:
            alphas_dh.append(psd_data[key].get('alpha_dh_band', 0))
            alphas_dh_err.append(psd_data[key].get('alpha_dh_band_err', 0))
            alphas_mid.append(psd_data[key].get('alpha_mid', 0))
            alphas_mid_err.append(psd_data[key].get('alpha_mid_err', 0))
        else:
            alphas_dh.append(0)
            alphas_dh_err.append(0)
            alphas_mid.append(0)
            alphas_mid_err.append(0)

    ax.errorbar(N_list, alphas_dh, yerr=alphas_dh_err,
                fmt='o-', color=plt.cm.tab10(2), linewidth=2,
                markersize=10, capsize=5, label='Dutta-Horn band', zorder=5)
    ax.errorbar(N_list, alphas_mid, yerr=alphas_mid_err,
                fmt='s--', color=plt.cm.tab10(3), linewidth=2,
                markersize=8, capsize=5, label='0.1-10 Hz band')

    ax.axhline(y=1.0, color='green', linestyle=':', linewidth=2, alpha=0.6,
               label='$\\alpha=1$ (1/f)')
    ax.axhline(y=2.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.4,
               label='$\\alpha=2$ (Brownian)')

    # Highlight N=3 as the 1/f sweet spot
    ax.axvspan(2.5, 3.5, alpha=0.1, color='green')

    ax.set_xlabel('Number of scales $N$', fontsize=14)
    ax.set_ylabel('PSD exponent $\\alpha$', fontsize=14)
    ax.set_title('(c) Spectral exponent vs N', fontsize=16)
    ax.legend(fontsize=9, loc='center right')
    ax.tick_params(labelsize=12)
    ax.set_xticks(N_list)
    ax.set_ylim([-0.5, 3.5])
    ax.grid(True, alpha=0.2)

    # ===== Panel (d): Friction time series =====
    ax = fig.add_subplot(gs[1, 0])

    for N_ts, color, lw in [(1, '#888888', 0.5), (3, plt.cm.tab10(2), 0.5)]:
        key = f'timeseries_N{N_ts}'
        if key in psd_data:
            ts = psd_data[key]
            g = np.array(ts['g_total'])
            dt_ts = ts['dt']
            t = np.arange(len(g)) * dt_ts
            mask = t < 150
            ax.plot(t[mask], g[mask], color=color, linewidth=lw,
                    alpha=0.7, label=f'N={N_ts}')

    ax.set_xlabel('Time (simulation units)', fontsize=14)
    ax.set_ylabel('$g_{\\mathrm{total}}(t)$', fontsize=14)
    ax.set_title('(d) Friction time series', fontsize=16)
    ax.legend(fontsize=12, loc='upper right')
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.2)

    # ===== Panel (e): GMM KL vs N_scales =====
    ax = fig.add_subplot(gs[1, 1])

    if has_gmm:
        N_gmm = []
        kl_means = []
        kl_stds = []
        for key in sorted(gmm_data.keys(), key=lambda x: int(x)):
            N_gmm.append(int(key))
            kl_means.append(gmm_data[key]['mean'])
            kl_stds.append(gmm_data[key]['std'])

        ax.errorbar(N_gmm, kl_means, yerr=kl_stds,
                    fmt='o-', color=plt.cm.tab10(2), linewidth=2,
                    markersize=10, capsize=5, zorder=5)

        # Add NHC baseline reference
        ax.axhline(y=0.544, color=nhc_color, linestyle='--', linewidth=1.5,
                    alpha=0.7, label='NHC (M=3)')
        # Highlight the drop from N=2 to N=3
        ax.axvspan(2.5, 3.5, alpha=0.1, color='green')
        ax.annotate('1/f onset', xy=(3, kl_means[N_gmm.index(3)]),
                    xytext=(5, 1.2), fontsize=11,
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                    color='green')

    ax.set_xlabel('Number of scales $N$', fontsize=14)
    ax.set_ylabel('GMM KL divergence', fontsize=14)
    ax.set_title('(e) GMM KL vs N (1M evals)', fontsize=16)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=12)
    ax.set_ylim([0, 2.5])
    ax.grid(True, alpha=0.2)

    # ===== Panel (f): Spectral match comparison =====
    ax = fig.add_subplot(gs[1, 2])

    if has_spec:
        configs = ['log_spaced_champion', 'wide_log_spaced', 'spectral_matched']
        labels = ['Champion\n[0.1, 0.7, 10]', 'Wide log\n[0.01, 3.16, 1000]',
                  'Spectral match\n[6.3k, 63k, 630k]']
        means = [spec_data[c]['mean_kl'] for c in configs]
        stds = [spec_data[c]['std_kl'] for c in configs]
        bar_colors = [plt.cm.tab10(2), plt.cm.tab10(4), '#cc4444']

        x_pos = np.arange(len(configs))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=6,
                      color=bar_colors, alpha=0.85, edgecolor='black',
                      linewidth=0.8, width=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=10)

        # Value labels
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.03, f'{m:.3f}', ha='center', fontsize=11,
                    weight='bold')

        # NHC baseline
        ax.axhline(y=0.544, color=nhc_color, linestyle='--', linewidth=1.5,
                    alpha=0.7, label='NHC (M=3)')
        ax.legend(fontsize=10)

    ax.set_ylabel('GMM KL divergence', fontsize=14)
    ax.set_title('(f) Spectral matching vs 1/f', fontsize=16)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.2, axis='y')

    # Save
    figpath = os.path.join(figdir, 'spectral_1f_consolidated.png')
    fig.savefig(figpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Consolidated figure saved to {figpath}")

    return figpath


if __name__ == '__main__':
    main()
