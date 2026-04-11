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
    # Track NH exact values so we can annotate its (nearly invisible) line.
    nh_exact_rel = None
    for m in METHOD_ORDER:
        rel = np.array([grad_data[m][str(d)]['rel_noise'] for d in DIMS])
        if m == 'NH exact':
            nh_exact_rel = rel.copy()
        ax_b.plot(DIMS_a, np.maximum(rel, 1e-16), '-o', color=COLORS[m],
                  lw=2, ms=6, label=m)
    ax_b.set_xscale('log')
    ax_b.set_yscale('log')
    ax_b.set_xlabel('dimension  $d$')
    ax_b.set_ylabel(r'$\|\mathrm{std}(\nabla_\theta L)\| \,/\, \|\mathrm{mean}(\nabla_\theta L)\|$')
    ax_b.set_title('(b) Gradient noise-to-signal vs dimension')
    ax_b.grid(True, alpha=0.3, which='both')
    ax_b.legend(loc='center right', framealpha=0.95)

    # NH exact sits at the floating-point noise floor (~1e-15), which is
    # nearly invisible against the Hutchinson curves an order of ten-to-the
    # -thirteen above it. Force the y-axis down to 1e-17 so there is room
    # for an explicit annotation pointing at the NH exact line.
    ax_b.set_ylim(1e-17, ax_b.get_ylim()[1])
    if nh_exact_rel is not None:
        # Use the last NH exact point (highest dimension) as the anchor — it
        # is farthest from the crowded legend and the Hutchinson curves.
        x_anchor = DIMS_a[-1]
        y_anchor = max(float(nh_exact_rel[-1]), 1e-16)
        ax_b.annotate(
            r'NH exact $\approx 10^{-15}$' + '\n(machine precision,\nnearly invisible)',
            xy=(x_anchor, y_anchor),
            xytext=(0.60, 0.18), textcoords='axes fraction',
            ha='left', va='center', fontsize=10, color=COLORS['NH exact'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLORS['NH exact'], alpha=0.95),
            arrowprops=dict(arrowstyle='->', color=COLORS['NH exact'], lw=1.3),
        )

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

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True,
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
    # Put the shared legend on the anisotropic (middle) panel in the upper
    # left — that panel has no "data variance dominated" annotation and its
    # traces go from low-left up to the right, so the upper-left corner is
    # the one empty region. This keeps the legend inside the plot area,
    # preventing any collision with the x-axis tick labels that happens
    # when fig.legend is pushed below the panels.
    axes[1].legend(loc='upper left', frameon=True, framealpha=0.95,
                   fontsize=10)
    # constrained_layout reserves space for the in-layout suptitle.
    fig.suptitle('E3.1  NH-CNF exact divergence vs Hutchinson — variance scaling',
                 fontsize=14)

    out_png = os.path.join(FIGDIR, 'fig_variance_scaling.png')
    out_pdf = os.path.join(FIGDIR, 'fig_variance_scaling.pdf')
    fig.savefig(out_png, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print('saved', out_png)


# =============================================================================
# fig_walltime.png  (cleanup item 3: add a speedup-ratio panel so the
# "2.3x at d=1000" advantage is visible directly, not buried in a log plot)
# =============================================================================
def replot_walltime():
    with open(os.path.join(RESDIR, 'e3_walltime.json')) as f:
        res = json.load(f)

    DIMS = sorted(int(k) for k in res['exact'].keys())
    ds = np.array(DIMS)
    te = np.array([res['exact'][str(d)]  for d in DIMS]) * 1e3  # ms
    t1 = np.array([res['hutch1'][str(d)] for d in DIMS]) * 1e3
    t5 = np.array([res['hutch5'][str(d)] for d in DIMS]) * 1e3

    # Speedup ratios — how much faster is NH exact than Hutchinson(k)?
    sp1 = t1 / te
    sp5 = t5 / te

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.5, 4.8),
                                     constrained_layout=True)

    # --- (a) absolute wall clock ---
    ax_a.plot(ds, te, '-o', color='#1f77b4', lw=2, ms=6,
              label='NH-CNF (exact div)')
    ax_a.plot(ds, t1, '-s', color='#ff7f0e', lw=2, ms=6,
              label='FFJORD-style Hutchinson(1)')
    ax_a.plot(ds, t5, '-^', color='#9467bd', lw=2, ms=6,
              label='FFJORD-style Hutchinson(5)')
    ax_a.set_xscale('log')
    ax_a.set_yscale('log')
    ax_a.set_xlabel('dimension $d$')
    ax_a.set_ylabel('wall-clock per step (ms)  [batch=256]')
    ax_a.set_title('(a) Per-step cost')
    ax_a.grid(True, alpha=0.3, which='both')
    ax_a.legend(loc='upper left', frameon=True, framealpha=0.95)

    # --- (b) speedup ratio: Hutchinson / NH exact ---
    ax_b.plot(ds, sp1, '-s', color='#ff7f0e', lw=2, ms=6,
              label='Hutchinson(1) / NH exact')
    ax_b.plot(ds, sp5, '-^', color='#9467bd', lw=2, ms=6,
              label='Hutchinson(5) / NH exact')
    ax_b.axhline(1.0, color='#888', lw=1.0, ls='--')
    ax_b.set_xscale('log')
    ax_b.set_xlabel('dimension $d$')
    ax_b.set_ylabel('speedup  (Hutchinson / NH exact)')
    ax_b.set_title('(b) NH exact is strictly cheaper than any Hutchinson($k$)')
    ax_b.grid(True, alpha=0.3, which='both')
    ax_b.legend(loc='upper left', frameon=True, framealpha=0.95)

    # Headline numbers: speedup at the largest dimension. Place them on the
    # right side of the panel, away from the legend (upper-left) and the
    # y-axis. Extend the y-axis upper limit so there is headroom.
    d_last = ds[-1]
    ymin, ymax = ax_b.get_ylim()
    ax_b.set_ylim(0, max(ymax, sp5.max() * 1.20))
    ax_b.annotate(
        f'{sp5[-1]:.1f}x speedup at $d={d_last}$',
        xy=(d_last, sp5[-1]),
        xytext=(0.55, 0.92), textcoords='axes fraction',
        fontsize=11, color='#9467bd', ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                  edgecolor='#9467bd', alpha=0.9),
        arrowprops=dict(arrowstyle='->', color='#9467bd', lw=1.2),
    )
    ax_b.annotate(
        f'{sp1[-1]:.1f}x speedup at $d={d_last}$',
        xy=(d_last, sp1[-1]),
        xytext=(0.55, 0.55), textcoords='axes fraction',
        fontsize=11, color='#ff7f0e', ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                  edgecolor='#ff7f0e', alpha=0.9),
        arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1.2),
    )

    fig.suptitle(
        'E3.4  Wall-clock: NH-CNF (exact divergence) vs Hutchinson trace',
        fontsize=14,
    )
    out_png = os.path.join(FIGDIR, 'fig_walltime.png')
    out_pdf = os.path.join(FIGDIR, 'fig_walltime.pdf')
    fig.savefig(out_png, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print('saved', out_png, f'(speedup@d={d_last}: h1={sp1[-1]:.2f}x, h5={sp5[-1]:.2f}x)')


if __name__ == '__main__':
    replot_training_stability()
    replot_variance_scaling()
    replot_walltime()
