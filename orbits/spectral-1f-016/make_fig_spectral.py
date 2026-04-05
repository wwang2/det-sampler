"""Consolidated Figure: 2x3 Nature-style panel for 1/f spectral analysis.

Panels:
(a) PSDs for N=1,3,5,10 with 1/f reference line
(b) Lorentzian decomposition for N=3
(c) alpha vs N (fitted exponent)
(d) Friction time series: N=1 vs N=3
(e) GMM KL vs N_scales
(f) Spectral match comparison

Style: research/style.md conventions.
"""

import sys
import os
import json
import numpy as np
from scipy.signal import welch
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def lorentzian(f, A, tau):
    return A * tau / (1.0 + (2.0 * np.pi * f * tau)**2)


def fit_power_law(freqs, psd, f_min=None, f_max=None):
    mask = freqs > 0
    if f_min is not None:
        mask &= freqs >= f_min
    if f_max is not None:
        mask &= freqs <= f_max
    f_fit = freqs[mask]
    p_fit = psd[mask]
    if len(f_fit) < 5:
        return 0.0, 0.0

    def power_law_log(log_f, log_A, alpha):
        return log_A - alpha * log_f

    log_f = np.log10(f_fit)
    log_p = np.log10(p_fit)
    try:
        popt, pcov = curve_fit(power_law_log, log_f, log_p, p0=[0.0, 1.0])
        return popt[1], np.sqrt(pcov[1, 1])
    except Exception:
        return 0.0, 0.0


def main():
    outdir = os.path.dirname(os.path.abspath(__file__))
    figdir = os.path.join(outdir, 'figures')

    # Load data
    with open(os.path.join(outdir, 'psd_results.json'), 'r') as f:
        psd_data = json.load(f)

    # Try loading spectral match results
    spectral_match_path = os.path.join(outdir, 'spectral_match_results.json')
    has_spectral = os.path.exists(spectral_match_path)
    if has_spectral:
        with open(spectral_match_path, 'r') as f:
            spec_data = json.load(f)

    # Load lorentzian fits
    lor_fits_path = os.path.join(outdir, 'lorentzian_fits.json')
    has_lor = os.path.exists(lor_fits_path)
    if has_lor:
        with open(lor_fits_path, 'r') as f:
            lor_fits = json.load(f)

    # --- CREATE FIGURE ---
    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.32)

    # Color scheme
    colors_N = {1: '#888888', 3: plt.cm.tab10(2), 5: plt.cm.tab10(3),
                7: plt.cm.tab10(4), 10: plt.cm.tab10(5)}
    nhc_color = '#ff7f0e'

    # ===== Panel (a): PSDs for N=1,3,5,10 =====
    ax_a = fig.add_subplot(gs[0, 0])
    for N in [1, 3, 5, 10]:
        key = str(N)
        if key not in psd_data:
            continue
        freqs = np.array(psd_data[key]['freqs'])
        psd = np.array(psd_data[key]['psd'])
        mask = freqs > 0
        ax_a.loglog(freqs[mask], psd[mask], color=colors_N[N],
                    linewidth=1.5, alpha=0.8, label=f'N={N}')

    # 1/f reference line
    f_ref = np.logspace(-1.5, 1.5, 100)
    # Normalize: match N=3 data at f=0.5
    d3 = psd_data.get('3')
    if d3:
        f3 = np.array(d3['freqs'])
        p3 = np.array(d3['psd'])
        m3 = f3 > 0
        idx05 = np.argmin(np.abs(f3[m3] - 0.5))
        ref_scale = p3[m3][idx05] * 0.5
        ax_a.loglog(f_ref, ref_scale / f_ref, 'k--', linewidth=1.5,
                    alpha=0.5, label='$1/f$')

    ax_a.set_xlabel('Frequency (Hz)', fontsize=14)
    ax_a.set_ylabel('PSD of $g_{\\mathrm{total}}(t)$', fontsize=14)
    ax_a.set_title('(a) Friction PSD vs N scales', fontsize=16)
    ax_a.legend(fontsize=10, loc='lower left')
    ax_a.tick_params(labelsize=12)
    ax_a.set_xlim([1e-2, 100])
    ax_a.grid(True, alpha=0.2, which='both')

    # ===== Panel (b): Lorentzian decomposition (N=3, Q=[0.1,1,10]) =====
    ax_b = fig.add_subplot(gs[0, 1])

    # Use per_thermo_N3 from the wide-range run for visual
    # But better: use the stored data or re-plot from decomposition fits
    decomp_colors = [plt.cm.tab10(2), plt.cm.tab10(3), plt.cm.tab10(4)]

    per_thermo_key = 'per_thermo_N3'
    if per_thermo_key in psd_data:
        ptd = psd_data[per_thermo_key]
        for k_str, data in ptd.items():
            k = int(k_str)
            fk = np.array(data['freqs'])
            pk = np.array(data['psd'])
            Q = data['Q']
            mask = fk > 0
            ax_b.loglog(fk[mask], pk[mask], color=decomp_colors[k % 3],
                        alpha=0.6, linewidth=1.0,
                        label=f'$g(\\xi_{{{k+1}}})$, Q={Q:.2g}')

    # If we have Lorentzian fits, overlay them
    if has_lor:
        f_fit = np.logspace(-2, 2, 300)
        psd_sum = np.zeros_like(f_fit)
        for k_str, fit in lor_fits.items():
            k = int(k_str)
            A, tau = fit['A'], fit['tau']
            lor_k = lorentzian(f_fit, A, tau)
            psd_sum += lor_k
            ax_b.loglog(f_fit, lor_k, '--', color=decomp_colors[k % 3],
                        linewidth=1.5, alpha=0.6)

        ax_b.loglog(f_fit, psd_sum, 'r-', linewidth=2, alpha=0.8,
                    label='$\\Sigma$ Lorentzians')

    ax_b.set_xlabel('Frequency (Hz)', fontsize=14)
    ax_b.set_ylabel('PSD', fontsize=14)
    ax_b.set_title('(b) Lorentzian decomposition', fontsize=16)
    ax_b.legend(fontsize=9, loc='lower left')
    ax_b.tick_params(labelsize=12)
    ax_b.set_xlim([1e-2, 100])
    ax_b.grid(True, alpha=0.2, which='both')

    # ===== Panel (c): alpha vs N =====
    ax_c = fig.add_subplot(gs[0, 2])
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

    ax_c.errorbar(N_list, alphas_dh, yerr=alphas_dh_err,
                  fmt='o-', color=plt.cm.tab10(2), linewidth=2,
                  markersize=8, capsize=4, label='$\\alpha$ (DH band)')
    ax_c.errorbar(N_list, alphas_mid, yerr=alphas_mid_err,
                  fmt='s--', color=plt.cm.tab10(3), linewidth=2,
                  markersize=8, capsize=4, label='$\\alpha$ (0.1-10 Hz)')

    ax_c.axhline(y=1.0, color='k', linestyle=':', linewidth=1.5, alpha=0.5,
                 label='$\\alpha=1$ (1/f)')
    ax_c.axhline(y=2.0, color='gray', linestyle=':', linewidth=1, alpha=0.4,
                 label='$\\alpha=2$ (Brownian)')

    ax_c.set_xlabel('Number of scales N', fontsize=14)
    ax_c.set_ylabel('PSD exponent $\\alpha$', fontsize=14)
    ax_c.set_title('(c) $\\alpha$ vs N scales', fontsize=16)
    ax_c.legend(fontsize=9, loc='center right')
    ax_c.tick_params(labelsize=12)
    ax_c.set_xticks(N_list)
    ax_c.set_ylim([-0.5, 3.5])
    ax_c.grid(True, alpha=0.2)

    # ===== Panel (d): Friction time series N=1 vs N=3 =====
    ax_d = fig.add_subplot(gs[1, 0])

    for N_ts, color, ls in [(1, '#888888', '-'), (3, plt.cm.tab10(2), '-')]:
        key = f'timeseries_N{N_ts}'
        if key in psd_data:
            ts = psd_data[key]
            g = np.array(ts['g_total'])
            dt_ts = ts['dt']
            t = np.arange(len(g)) * dt_ts
            # Show first 2000 time units for clarity
            mask = t < 200
            ax_d.plot(t[mask], g[mask], color=color, linewidth=0.5,
                      alpha=0.7, label=f'N={N_ts}')

    ax_d.set_xlabel('Time', fontsize=14)
    ax_d.set_ylabel('$g_{\\mathrm{total}}(t)$', fontsize=14)
    ax_d.set_title('(d) Friction time series', fontsize=16)
    ax_d.legend(fontsize=12)
    ax_d.tick_params(labelsize=12)
    ax_d.grid(True, alpha=0.2)

    # ===== Panel (e): GMM KL vs N_scales =====
    ax_e = fig.add_subplot(gs[1, 1])

    # We need to run GMM for different N_scales - use stored results if available
    # For now, use known data points from parent orbit + our analysis
    # multiscale-chain-009: Qs=[0.1, 0.7, 10.0] -> GMM KL mean=0.071
    # We can estimate from spectral match results
    if has_spectral:
        configs = list(spec_data.keys())
        names_short = []
        means = []
        stds = []
        bar_colors = []
        for name in configs:
            d = spec_data[name]
            names_short.append(name.replace('_', '\n'))
            means.append(d['mean_kl'])
            stds.append(d['std_kl'])
            if 'champion' in name:
                bar_colors.append(nhc_color)
            elif 'spectral' in name:
                bar_colors.append(plt.cm.tab10(2))
            else:
                bar_colors.append(plt.cm.tab10(4))

        x_pos = np.arange(len(configs))
        bars = ax_e.bar(x_pos, means, yerr=stds, capsize=5,
                        color=bar_colors, alpha=0.8, edgecolor='black',
                        linewidth=0.5)
        ax_e.set_xticks(x_pos)
        ax_e.set_xticklabels(names_short, fontsize=9)
        ax_e.set_ylabel('GMM KL divergence', fontsize=14)
        ax_e.set_title('(e) GMM KL: spectral match', fontsize=16)
        ax_e.tick_params(labelsize=12)
        ax_e.grid(True, alpha=0.2, axis='y')

        # Add value labels on bars
        for i, (m, s) in enumerate(zip(means, stds)):
            ax_e.text(i, m + s + 0.005, f'{m:.3f}', ha='center', fontsize=10)
    else:
        ax_e.text(0.5, 0.5, 'Spectral match\nresults pending',
                  transform=ax_e.transAxes, ha='center', va='center', fontsize=14)
        ax_e.set_title('(e) GMM KL: spectral match', fontsize=16)

    # ===== Panel (f): Spectral match comparison (bar chart or placeholder) =====
    ax_f = fig.add_subplot(gs[1, 2])

    # Show the Q configurations and their predicted vs actual performance
    if has_spectral:
        # Table-like visualization
        ax_f.axis('off')
        table_data = []
        headers = ['Config', 'Q values', 'Mean KL', 'Std']
        for name in configs:
            d = spec_data[name]
            qs_str = ', '.join([f'{q:.2g}' for q in d['Qs']])
            table_data.append([
                name.replace('_', ' '),
                qs_str,
                f"{d['mean_kl']:.4f}",
                f"{d['std_kl']:.4f}"
            ])

        table = ax_f.table(cellText=table_data, colLabels=headers,
                           loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.0, 1.8)

        # Color header
        for j in range(len(headers)):
            table[0, j].set_facecolor('#d4e6f1')
            table[0, j].set_text_props(weight='bold')

        # Highlight best row
        if means:
            best_idx = np.argmin(means)
            for j in range(len(headers)):
                table[best_idx + 1, j].set_facecolor('#d5f5e3')

        ax_f.set_title('(f) Configuration comparison', fontsize=16)
    else:
        ax_f.text(0.5, 0.5, 'Spectral match\nresults pending',
                  transform=ax_f.transAxes, ha='center', va='center', fontsize=14)
        ax_f.set_title('(f) Configuration comparison', fontsize=16)

    # Save
    figpath = os.path.join(figdir, 'spectral_1f_consolidated.png')
    fig.savefig(figpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Consolidated figure saved to {figpath}")


if __name__ == '__main__':
    main()
