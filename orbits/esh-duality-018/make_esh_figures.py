"""ESH Duality: Consolidated Figure.

Produces figures/esh_duality_consolidated.png with panels:
  (a) Phase-space trajectories: ESH vs Log-Osc vs ESH-Thermo vs NHC on 1D HO
  (b) Friction function g(xi) comparison for all methods
  (c) Stationary distribution histograms (q and p marginals)
  (d) GMM 2D KL divergence bar chart

Loads results from comparison_results.json and trajectories.npz.
"""

import sys
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

WORKTREE = '/Users/wujiewang/code/det-sampler/.worktrees/esh-duality-018'
ORBIT_DIR = Path(WORKTREE) / 'orbits/esh-duality-018'
FIG_DIR = ORBIT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Color palette
COLORS = {
    'esh':        '#e74c3c',   # red — ESH (conservative, non-thermostat)
    'log_osc':    '#2980b9',   # blue — Log-Osc thermostat
    'esh_thermo': '#8e44ad',   # purple — ESH-inspired thermostat
    'nhc':        '#27ae60',   # green — NHC baseline
    'theory':     '#f39c12',   # orange — theory curves
}

LABELS = {
    'esh':        'ESH (conservative)',
    'log_osc':    'Log-Osc thermostat',
    'esh_thermo': 'ESH-Thermo (new)',
    'nhc':        'NHC baseline',
}


def load_data():
    """Load precomputed comparison results."""
    results_path = ORBIT_DIR / 'comparison_results.json'
    traj_path = ORBIT_DIR / 'trajectories.npz'

    if not results_path.exists():
        raise FileNotFoundError(f"Run make_esh_comparison.py first. Missing: {results_path}")
    if not traj_path.exists():
        raise FileNotFoundError(f"Run make_esh_comparison.py first. Missing: {traj_path}")

    with open(results_path) as f:
        results = json.load(f)

    traj = np.load(traj_path)
    return results, traj


def make_figure(results, traj):
    """Create consolidated 4-panel figure."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        'ESH Dynamics vs Generalized Thermostat Framework\n'
        'ESH is a conservative Hamiltonian system; ESH-Thermo is a new Master Theorem member',
        fontsize=13, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # ----------------------------------------------------------------
    # Panel (a): Phase-space trajectories on 1D HO
    # ----------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])

    n_show = 3000  # show first N steps
    for key, q_key, p_key, alpha, lw in [
        ('esh',        'q_esh',        'p_esh',        0.4, 0.7),
        ('log_osc',    'q_losc',       'p_losc',       0.4, 0.7),
        ('esh_thermo', 'q_esh_thermo', 'p_esh_thermo', 0.4, 0.7),
        ('nhc',        'q_nhc',        'p_nhc',        0.4, 0.7),
    ]:
        q = traj[q_key][:n_show]
        p = traj[p_key][:n_show]
        ax_a.plot(q, p, color=COLORS[key], alpha=alpha, linewidth=lw, label=LABELS[key])

    # Add canonical ellipse for reference
    theta = np.linspace(0, 2*np.pi, 200)
    ax_a.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1.5, alpha=0.5, label='Canonical 1σ ellipse')
    ax_a.plot(2*np.cos(theta), 2*np.sin(theta), 'k:', linewidth=1.0, alpha=0.3, label='2σ')

    ax_a.set_xlabel('q (position)', fontsize=11)
    ax_a.set_ylabel('p (momentum)', fontsize=11)
    ax_a.set_title('(a) Phase-Space Trajectories: 1D HO', fontsize=11, fontweight='bold')
    ax_a.legend(fontsize=8, loc='upper right')
    ax_a.set_xlim(-4, 4)
    ax_a.set_ylim(-5, 5)
    ax_a.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax_a.axvline(0, color='gray', linewidth=0.5, alpha=0.5)

    # Note on ESH
    ax_a.text(0.02, 0.02,
              'ESH: unit-speed motion\nonly |p| varies',
              transform=ax_a.transAxes, fontsize=7.5,
              color=COLORS['esh'], verticalalignment='bottom',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # ----------------------------------------------------------------
    # Panel (b): Friction function comparison
    # ----------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])

    friction = results['friction']
    xi_vals = np.array(friction['xi'])

    ax_b.plot(xi_vals, friction['g_nh'],         color='gray',           linewidth=2,   label='NH: g=xi (unbounded)', linestyle='--')
    ax_b.plot(xi_vals, friction['g_losc'],        color=COLORS['log_osc'],  linewidth=2.5, label='Log-Osc: 2ξ/(1+ξ²)')
    ax_b.plot(xi_vals, friction['g_tanh'],        color='#16a085',        linewidth=2,   label='Tanh: tanh(ξ)')
    ax_b.plot(xi_vals, friction['g_esh_thermo'],  color=COLORS['esh_thermo'], linewidth=2.5, label='ESH-Thermo: exp(2ξ)-1')

    # Clamp ESH-thermo for visibility
    g_esh = np.clip(np.array(friction['g_esh_thermo']), -5, 5)
    ax_b.plot(xi_vals, g_esh, color=COLORS['esh_thermo'], linewidth=2.5)

    ax_b.axhline(0, color='black', linewidth=1, alpha=0.7)
    ax_b.axvline(0, color='black', linewidth=0.5, alpha=0.5)

    # Indicate bounded vs unbounded
    ax_b.axhline(1, color=COLORS['log_osc'], linewidth=1, linestyle=':', alpha=0.7)
    ax_b.axhline(-1, color=COLORS['log_osc'], linewidth=1, linestyle=':', alpha=0.7)
    ax_b.text(2.7, 1.05, 'g=+1', fontsize=8, color=COLORS['log_osc'])
    ax_b.text(2.7, -1.2, 'g=-1', fontsize=8, color=COLORS['log_osc'])

    ax_b.set_xlabel('xi = log(|p|/sqrt(kT))', fontsize=11)
    ax_b.set_ylabel('g(xi) — friction coupling', fontsize=11)
    ax_b.set_title('(b) Friction Functions g(xi)', fontsize=11, fontweight='bold')
    ax_b.legend(fontsize=8.5, loc='upper left')
    ax_b.set_xlim(-3, 3)
    ax_b.set_ylim(-5, 8)

    ax_b.text(0.02, 0.92,
              'ESH has NO friction term\n(conservative dynamics)',
              transform=ax_b.transAxes, fontsize=8,
              color=COLORS['esh'], verticalalignment='top',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffeaa7', alpha=0.9))

    # ----------------------------------------------------------------
    # Panel (c): Stationary distribution histograms
    # ----------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])

    # q-marginal histograms
    burnin = 5000
    bins_q = np.linspace(-4, 4, 60)
    bins_p = np.linspace(-5, 5, 60)

    # Theory: N(0, 1)
    q_theory = np.linspace(-4, 4, 200)
    ax_c.plot(q_theory, np.exp(-q_theory**2/2) / np.sqrt(2*np.pi),
              'k-', linewidth=2.5, label='Canonical N(0,1)', zorder=10)

    # ESH - its q marginal is approximately correct but NOT exact due to improper p distribution
    q_esh_hist = traj['q_esh'][burnin:]
    ax_c.hist(q_esh_hist, bins=bins_q, density=True, alpha=0.4,
              color=COLORS['esh'], label=LABELS['esh'], histtype='stepfilled', edgecolor='none')

    # Log-Osc
    q_losc_hist = traj['q_losc'][burnin:]
    ax_c.hist(q_losc_hist, bins=bins_q, density=True, alpha=0.5,
              color=COLORS['log_osc'], label=LABELS['log_osc'], histtype='step', linewidth=2)

    # ESH-Thermo
    q_esh_t_hist = traj['q_esh_thermo'][burnin:]
    ax_c.hist(q_esh_t_hist, bins=bins_q, density=True, alpha=0.5,
              color=COLORS['esh_thermo'], label=LABELS['esh_thermo'], histtype='step', linewidth=2)

    # NHC
    q_nhc_hist = traj['q_nhc'][burnin:]
    ax_c.hist(q_nhc_hist, bins=bins_q, density=True, alpha=0.5,
              color=COLORS['nhc'], label=LABELS['nhc'], histtype='step', linewidth=2)

    ax_c.set_xlabel('q (position)', fontsize=11)
    ax_c.set_ylabel('Density', fontsize=11)
    ax_c.set_title('(c) Position Marginal P(q) — 1D HO', fontsize=11, fontweight='bold')
    ax_c.legend(fontsize=8, loc='upper right')
    ax_c.set_xlim(-4, 4)

    # Add KL values as text
    stats = results['ho_stats']
    kl_text = "\n".join([
        f"KL divergences (q):",
        f"  ESH:        {stats['esh']['kl_q']:.4f}" if stats['esh']['kl_q'] is not None else "  ESH: N/A",
        f"  Log-Osc:    {stats['log_osc']['kl_q']:.4f}" if stats['log_osc']['kl_q'] is not None else "  Log-Osc: N/A",
        f"  ESH-Thermo: {stats['esh_thermo']['kl_q']:.4f}" if stats['esh_thermo']['kl_q'] is not None else "  ESH-Thermo: N/A",
        f"  NHC:        {stats['nhc']['kl_q']:.4f}" if stats['nhc']['kl_q'] is not None else "  NHC: N/A",
    ])
    ax_c.text(0.02, 0.97, kl_text, transform=ax_c.transAxes, fontsize=7.5,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    # ----------------------------------------------------------------
    # Panel (d): GMM KL divergence bar chart
    # ----------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])

    gmm = results['gmm_kl']
    methods = ['NHC', 'Log-Osc', 'ESH-Thermo']
    kl_vals = [gmm.get(m, {}).get('kl', float('nan')) for m in methods]
    colors_bar = [COLORS['nhc'], COLORS['log_osc'], COLORS['esh_thermo']]

    valid = [(m, kl, c) for m, kl, c in zip(methods, kl_vals, colors_bar) if not np.isnan(kl)]
    if valid:
        m_valid, kl_valid, c_valid = zip(*valid)
        bars = ax_d.bar(range(len(m_valid)), kl_valid, color=c_valid,
                        edgecolor='black', linewidth=0.8, alpha=0.85, width=0.6)
        ax_d.set_xticks(range(len(m_valid)))
        ax_d.set_xticklabels(m_valid, fontsize=10)

        # Value labels on bars
        for i, (bar, kl) in enumerate(zip(bars, kl_valid)):
            ax_d.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                      f'{kl:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax_d.set_ylabel('KL Divergence (lower = better)', fontsize=11)
        ax_d.set_title('(d) GMM 2D KL Divergence Comparison', fontsize=11, fontweight='bold')
        ax_d.set_ylim(0, max(kl_valid) * 1.25)

        # Highlight best
        best_idx = np.argmin(kl_valid)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(2.5)
        ax_d.text(best_idx, kl_valid[best_idx] * 0.5, 'BEST',
                  ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    # Note about ESH
    ax_d.text(0.5, 0.97,
              'ESH (1D, conservative) not applicable\nto 2D GMM sampling comparison.\n'
              'ESH-Thermo IS a valid thermostat.',
              transform=ax_d.transAxes, fontsize=8.5,
              ha='center', va='top',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9fa', alpha=0.9))

    # ----------------------------------------------------------------
    # Annotate structural difference
    # ----------------------------------------------------------------
    fig.text(0.5, 0.005,
             'KEY FINDING: ESH is a conservative Hamiltonian system (dp/dt = -∇U·|p|), '
             'NOT a dissipative thermostat.\n'
             'ESH-Thermo uses V(ξ) = Q·(e^{2ξ}/2 − ξ), a NEW member of the Master Theorem family '
             '(not Log-Osc/NH/Tanh/Arctan).',
             ha='center', va='bottom', fontsize=9.5,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3cd', edgecolor='#f39c12', alpha=0.95))

    out_path = FIG_DIR / 'esh_duality_consolidated.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {out_path}")
    plt.close(fig)
    return out_path


def main():
    print("Loading results...")
    results, traj = load_data()
    print("Making figure...")
    out_path = make_figure(results, traj)
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
