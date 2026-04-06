"""Task 4: Generate 2x2 consolidated figure for high-dimensional validation.

Panels:
  (a) Ergodicity score vs dimension (isotropic Gaussian): NHC vs NHCTail
  (b) Ergodicity score vs curvature ratio (anisotropic): NHC vs NHCTail
  (c) Variance match per dimension for d=50 anisotropic (bar chart)
  (d) KL divergence proxy vs force evals for d=50 anisotropic

Nature-style, DPI=300, NHC=#ff7f0e, NHCTail=#9467bd.
"""

from __future__ import annotations
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

WORKTREE = '/Users/wujiewang/code/det-sampler/.worktrees/high-dim-022'
ORBIT_DIR = f'{WORKTREE}/orbits/high-dim-022'

# Nature-style color palette
COLOR_NHC = '#ff7f0e'
COLOR_MS  = '#9467bd'

# Nature journal figure style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'lines.linewidth': 1.2,
    'lines.markersize': 5,
})


def panel_a(ax, gaussian_results):
    """Mean relative variance error vs dimension for isotropic Gaussian.

    Both samplers are non-ergodic on separable isotropic Gaussian (each
    dimension is an independent 1D harmonic oscillator — known KAM tori problem).
    We show mean relative error; lower is better.
    """
    dims = sorted(int(d) for d in gaussian_results.keys())
    nhc_err = [gaussian_results[str(d)]['nhc_mean_rel_err_q'] for d in dims]
    ms_err  = [gaussian_results[str(d)]['ms_mean_rel_err_q']  for d in dims]

    ax.plot(dims, nhc_err, 'o-', color=COLOR_NHC, label='NHC (M=3)',
            markerfacecolor='white', markeredgewidth=1.2)
    ax.plot(dims, ms_err, 's-', color=COLOR_MS, label='NHCTail (MS)',
            markerfacecolor='white', markeredgewidth=1.2)

    ax.set_xlabel('Dimension $d$')
    ax.set_ylabel('Mean relative var error')
    ax.set_title('(a) Isotropic Gaussian: mean var error vs $d$')
    ax.set_xticks(dims)
    ax.legend(frameon=False)
    ax.text(0.05, 0.92, 'Both non-ergodic\n(KAM tori, expected)',
            transform=ax.transAxes, fontsize=6, color='gray', va='top')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def panel_b(ax, aniso_results):
    """Ergodicity score vs curvature ratio for anisotropic Gaussian (per-dim)."""
    kappas = np.array(aniso_results['kappas'])
    nhc_rel = np.array(aniso_results['nhc_rel_err'])
    ms_rel  = np.array(aniso_results['ms_rel_err'])
    tol = 0.20

    # Plot 1 - within tolerance as colored markers
    nhc_ok = nhc_rel < tol
    ms_ok  = ms_rel  < tol

    ax.scatter(kappas[nhc_ok],  nhc_rel[nhc_ok],  color=COLOR_NHC, marker='o',
               s=18, label='NHC OK', zorder=3, alpha=0.8)
    ax.scatter(kappas[~nhc_ok], nhc_rel[~nhc_ok], color=COLOR_NHC, marker='x',
               s=18, label='NHC fail', zorder=3, alpha=0.5)
    ax.scatter(kappas[ms_ok],   ms_rel[ms_ok],    color=COLOR_MS,  marker='s',
               s=18, label='NHCTail OK', zorder=4, alpha=0.8)
    ax.scatter(kappas[~ms_ok],  ms_rel[~ms_ok],   color=COLOR_MS,  marker='+',
               s=18, label='NHCTail fail', zorder=4, alpha=0.5)

    ax.axhline(tol, ls='--', color='gray', lw=0.8, alpha=0.7, label='20% threshold')
    ax.set_xscale('log')
    ax.set_xlabel(r'Curvature $\kappa_i$')
    ax.set_ylabel('Relative variance error')
    ax.set_title(r'(b) Anisotropic Gaussian ($d=20$)')
    ax.legend(frameon=False, ncol=2, fontsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate scores
    nhc_sc = aniso_results['nhc_score']
    ms_sc  = aniso_results['ms_score']
    ax.text(0.02, 0.97,
            f'NHC score={nhc_sc:.2f}  NHCTail={ms_sc:.2f}  Δ={ms_sc-nhc_sc:+.2f}',
            transform=ax.transAxes, fontsize=6.5, va='top',
            color='#333333')


def panel_c(ax, aniso_results):
    """Per-dimension variance match bar chart."""
    kappas    = np.array(aniso_results['kappas'])
    exp_var   = np.array(aniso_results['expected_var'])
    nhc_var   = np.array(aniso_results['nhc_obs_var'])
    ms_var    = np.array(aniso_results['ms_obs_var'])
    d = len(kappas)
    x = np.arange(d)
    w = 0.30

    # Normalise: ratio observed / expected (1 = perfect)
    nhc_ratio = nhc_var / exp_var
    ms_ratio  = ms_var  / exp_var

    ax.bar(x - w/2, nhc_ratio, width=w, color=COLOR_NHC, alpha=0.75, label='NHC', linewidth=0)
    ax.bar(x + w/2, ms_ratio,  width=w, color=COLOR_MS,  alpha=0.75, label='NHCTail', linewidth=0)
    ax.axhline(1.0,  ls='-',  color='black', lw=0.8, alpha=0.5)
    ax.axhline(1.20, ls='--', color='gray',  lw=0.7, alpha=0.5)
    ax.axhline(0.80, ls='--', color='gray',  lw=0.7, alpha=0.5)

    ax.set_xlabel('Dimension index')
    ax.set_ylabel(r'var$(q_i)$ / $(kT/\kappa_i)$')
    ax.set_title(r'(c) Per-dim variance ratio ($d=20$ anisotropic)')
    ax.set_xticks(x[::4])
    ax.set_xlim(-0.8, d - 0.2)
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def panel_d(ax, gaussian_results):
    """Per-dimension variance for d=100 isotropic Gaussian: NHC vs NHCTail.

    Shows individual dim variances; expected = 1.0 (dashed line).
    Both are similarly non-ergodic due to KAM tori in independent HO dims.
    """
    d = 100
    key = str(d)
    if key not in gaussian_results or 'nhc_q_var' not in gaussian_results[key]:
        ax.text(0.5, 0.5, 'Per-dim data\nnot available',
                ha='center', va='center', transform=ax.transAxes, fontsize=9, color='gray')
        ax.set_title('(d) Per-dim variance $d=100$ isotropic')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return

    nhc_var = np.array(gaussian_results[key]['nhc_q_var'])
    ms_var  = np.array(gaussian_results[key]['ms_q_var'])
    x = np.arange(d)
    w = 0.35

    ax.bar(x - w/2, nhc_var, width=w, color=COLOR_NHC, alpha=0.6, label='NHC', linewidth=0)
    ax.bar(x + w/2, ms_var,  width=w, color=COLOR_MS,  alpha=0.6, label='NHCTail', linewidth=0)
    ax.axhline(1.0,  ls='-',  color='black', lw=0.8, alpha=0.6, label='Expected (kT=1)')
    ax.axhline(1.10, ls='--', color='gray',  lw=0.6, alpha=0.5)
    ax.axhline(0.90, ls='--', color='gray',  lw=0.6, alpha=0.5)

    ax.set_xlabel('Dimension index')
    ax.set_ylabel(r'var$(q_i)$')
    ax.set_title(r'(d) Per-dim var, isotropic Gaussian $d=100$')
    ax.set_xlim(-1, d)
    ax.set_xticks(np.arange(0, d+1, 20))
    ax.legend(frameon=False, fontsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def main():
    # Load results
    with open(f'{ORBIT_DIR}/results_gaussian.json') as f:
        gauss_res = json.load(f)
    with open(f'{ORBIT_DIR}/results_anisotropic.json') as f:
        aniso_res = json.load(f)

    # Try loading convergence data if available
    try:
        with open(f'{ORBIT_DIR}/results_convergence.json') as f:
            conv_res = json.load(f)
    except FileNotFoundError:
        conv_res = None

    fig = plt.figure(figsize=(7.0, 5.5))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.45, wspace=0.38,
                           left=0.10, right=0.97,
                           top=0.94, bottom=0.10)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_a(ax_a, gauss_res)
    panel_b(ax_b, aniso_res)
    panel_c(ax_c, aniso_res)
    panel_d(ax_d, gauss_res)

    fig.suptitle('High-Dimensional Validation: MultiScaleNHCTail vs NHC',
                 fontsize=10, fontweight='bold', y=0.99)

    out_path = f'{ORBIT_DIR}/highdim_figure.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {out_path}")

    out_pdf = f'{ORBIT_DIR}/highdim_figure.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"Figure saved to {out_pdf}")


if __name__ == '__main__':
    main()
