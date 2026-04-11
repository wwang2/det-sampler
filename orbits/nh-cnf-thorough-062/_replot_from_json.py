"""Cleanup replot: regenerate fig_training_stability.png and
fig_variance_scaling.png from their saved JSON results, applying the
layout/annotation fixes from the cleanup pass.

This script does NOT re-run any experiment; it reads the existing JSON
under ./results/ and writes new PNG/PDF under ./figures/.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, 'figures')
RESDIR = os.path.join(HERE, 'results')

mpl.rcParams.update({
    'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 130, 'savefig.dpi': 220, 'savefig.pad_inches': 0.2,
})


COLORS = {
    'NH exact':  '#1f77b4',
    'Hutch(1)':  '#ff7f0e',
    'Hutch(5)':  '#9467bd',
    'Hutch(20)': '#8c564b',
}
METHOD_ORDER = ['NH exact', 'Hutch(1)', 'Hutch(5)', 'Hutch(20)']


# =============================================================================
# fig_training_stability.png  (item 4: fontsize=11, constrained_layout, y=1.02)
# =============================================================================
def replot_training_stability():
    with open(os.path.join(RESDIR, 'e3_training_highd.json')) as f:
        data = json.load(f)
    loss_data = data['loss_variance_d10']
    grad_data = data['grad_noise_vs_d']
    DIMS = data['DIMS']

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5.2),
                                     constrained_layout=True)

    # --- (a) loss variance boxplot ---
    data_a = [loss_data[m] for m in METHOD_ORDER]
    bp = ax_a.boxplot(
        data_a, tick_labels=METHOD_ORDER, widths=0.55, patch_artist=True,
        medianprops=dict(color='black', lw=1.5),
        whiskerprops=dict(color='#555', lw=1.2),
        capprops=dict(color='#555', lw=1.2),
        flierprops=dict(marker='.', markerfacecolor='#888', markersize=4),
    )
    for patch, m in zip(bp['boxes'], METHOD_ORDER):
        patch.set_facecolor(COLORS[m]); patch.set_alpha(0.55)
        patch.set_edgecolor('black')
    ax_a.set_ylabel('reverse-KL loss (100 fresh MC draws)')
    ax_a.set_title('(a) Loss variance, $d=10$')
    ax_a.grid(True, axis='y', alpha=0.3)

    # annotate stds  (bumped to fontsize=11 per cleanup item 4)
    for i, m in enumerate(METHOD_ORDER):
        s = np.std(loss_data[m], ddof=1)
        ax_a.annotate(f'std={s:.2e}', xy=(i + 1, max(loss_data[m])),
                      xytext=(0, 10), textcoords='offset points',
                      ha='center', fontsize=11, color=COLORS[m])
    ax_a.set_xticklabels(METHOD_ORDER, rotation=0)
    # leave headroom for the annotations so they're not clipped
    lo, hi = ax_a.get_ylim()
    ax_a.set_ylim(lo, hi + 0.25 * (hi - lo))

    # --- (b) grad noise vs d ---
    DIMS_a = np.array(DIMS)
    for m in METHOD_ORDER:
        rel = np.array([grad_data[m][str(d)]['rel_noise'] for d in DIMS])
        ax_b.plot(DIMS_a, np.maximum(rel, 1e-16), '-o', color=COLORS[m],
                  lw=2, ms=6, label=m)
    ax_b.set_xscale('log')
    ax_b.set_yscale('log')
    ax_b.set_xlabel('dimension  $d$')
    ax_b.set_ylabel(r'$\|\mathrm{std}(\nabla_\theta L)\| \,/\, \|\mathrm{mean}(\nabla_\theta L)\|$')
    ax_b.set_title('(b) Gradient noise-to-signal vs dimension')
    ax_b.grid(True, alpha=0.3, which='both')
    ax_b.legend(loc='lower right', framealpha=0.95)

    # With constrained_layout=True, matplotlib reserves top space for the
    # suptitle automatically as long as it sits within y<=1.0. Placing it at
    # y=1.02 would put it outside the reserved region and collide with the
    # (a)/(b) subplot titles, so we use the default in-layout placement.
    # bbox_inches='tight' on savefig still prevents any clipping.
    fig.suptitle(
        'E3.3  NH-CNF training stability: exact divergence is deterministic at every $d$',
        fontsize=14,
    )

    out_png = os.path.join(FIGDIR, 'fig_training_stability.png')
    out_pdf = os.path.join(FIGDIR, 'fig_training_stability.pdf')
    fig.savefig(out_png, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print('saved', out_png)


# =============================================================================
# fig_variance_scaling.png  (item 5: "data variance dominated" annotations on
# iso + bimod; NOT on aniso which does separate the methods)
# =============================================================================
def replot_variance_scaling():
    with open(os.path.join(RESDIR, 'e3_variance.json')) as f:
        results = json.load(f)

    TARGETS = [
        ('iso',   'Isotropic Gaussian',   True),   # data-variance dominated
        ('aniso', 'Anisotropic Gaussian', False),  # methods separate
        ('bimod', 'Bimodal',              True),   # data-variance dominated
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True,
                             constrained_layout=True)

    for ax, (key, label, dominated) in zip(axes, TARGETS):
        dims_keys = sorted(
            next(iter(results[key].values())).keys(), key=lambda s: int(s)
        )
        DIMS = np.array([int(k) for k in dims_keys])
        for m in METHOD_ORDER:
            md = results[key][m]
            means = np.array([md[k][0] for k in dims_keys])
            stds  = np.array([md[k][1] for k in dims_keys])
            lo = np.maximum(means - stds, 1e-12)
            hi = means + stds
            ax.plot(DIMS, np.maximum(means, 1e-12), '-o',
                    color=COLORS[m], lw=1.8, ms=4.5, label=m)
            ax.fill_between(DIMS, lo, hi, alpha=0.18, color=COLORS[m])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('dimension d')
        ax.set_title(label)
        ax.grid(True, alpha=0.3, which='both')
        if dominated:
            # Traces go low-left -> high-right in the data-variance-dominated
            # panels, so upper-left is the only empty region. The legend lives
            # on the rightmost panel at lower-right, so there's no collision.
            ax.text(
                0.04, 0.96, 'data variance dominated',
                transform=ax.transAxes,
                ha='left', va='top', fontsize=11,
                color='#444', style='italic',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white', edgecolor='#888', alpha=0.9),
            )
    axes[0].set_ylabel('std[ log p(x) ]  across ODE trajectories')
    axes[-1].legend(loc='lower right', frameon=True, framealpha=0.95)
    # constrained_layout reserves space for the in-layout suptitle.
    fig.suptitle('E3.1  NH-CNF exact divergence vs Hutchinson — variance scaling',
                 fontsize=14)

    out_png = os.path.join(FIGDIR, 'fig_variance_scaling.png')
    out_pdf = os.path.join(FIGDIR, 'fig_variance_scaling.pdf')
    fig.savefig(out_png, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print('saved', out_png)


if __name__ == '__main__':
    replot_training_stability()
    replot_variance_scaling()
