"""Consolidated mixing-rates figures (2x3 panel).

Panels:
  (a) tau_int vs N (log y) — GMM observable, shows minimum at N=3
  (b) C(t) curves for N=1,3,5 — shows faster decay at N=3
  (c) PSD exponent alpha vs N — from parent orbit
  (d) tau_int vs alpha scatter — empirical scaling law
  (e) Barrier crossings vs N for different lambda — 1/f advantage grows with barrier
  (f) ESS/force-eval vs N — direct efficiency metric

Loads:
  - orbits/mixing-rates-019/autocorr_results.json
  - orbits/mixing-rates-019/barrier_crossing_results.json
  - orbits/spectral-1f-016/psd_results.json  (alpha values)
  - orbits/spectral-1f-016/gmm_vs_n_results.json  (KL values as reference)
"""
from __future__ import annotations

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ORBIT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(ORBIT_DIR, '..', 'spectral-1f-016')
FIG_DIR = os.path.join(ORBIT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

N_LIST = [1, 2, 3, 5, 7, 10]
COLORS = {1: '#e41a1c', 2: '#ff7f00', 3: '#377eb8', 5: '#4daf4a', 7: '#984ea3', 10: '#a65628'}
HIGHLIGHT = {1: '#e41a1c', 3: '#377eb8', 5: '#4daf4a'}


def load_autocorr():
    path = os.path.join(ORBIT_DIR, 'autocorr_results.json')
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return {}
    with open(path) as f:
        return json.load(f)


def load_barrier():
    path = os.path.join(ORBIT_DIR, 'barrier_crossing_results.json')
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return {}
    with open(path) as f:
        return json.load(f)


def load_psd():
    path = os.path.join(PARENT_DIR, 'psd_results.json')
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return {}
    with open(path) as f:
        raw = json.load(f)
    # Extract only per-N alpha values (skip per_thermo_* and timeseries_* keys)
    result = {}
    for k, v in raw.items():
        try:
            n = int(k)
            result[n] = v
        except ValueError:
            pass
    return result


def load_gmm():
    path = os.path.join(PARENT_DIR, 'gmm_vs_n_results.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def get_tau_int(autocorr: dict, N: int) -> float | None:
    """Extract tau_int in force-evals from autocorr dict (handles both old and new format)."""
    r = autocorr.get(str(N))
    if r is None:
        return None
    # New GMM-based format: mode indicator is most informative
    if 'tau_int_mode_evals' in r:
        return float(r['tau_int_mode_evals'])
    if 'tau_int_x_evals' in r:
        return float(r['tau_int_x_evals'])
    # Old HO format (tau_int in steps)
    if 'tau_int' in r:
        return float(r['tau_int'])
    return None


def get_ess_per_eval(autocorr: dict, N: int) -> float | None:
    r = autocorr.get(str(N))
    if r is None:
        return None
    return float(r.get('ess_per_eval', 0)) or None


def get_C_curve(autocorr: dict, N: int) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Return (lags_in_evals, C) for a given N."""
    r = autocorr.get(str(N))
    if r is None:
        return None, None
    # New format
    if 'C_x' in r and 'C_lag_evals' in r:
        return np.array(r['C_lag_evals']), np.array(r['C_x'])
    # Old HO format
    if 'C' in r:
        C = np.array(r['C'])
        lags = np.arange(len(C)) * 1  # 1 step per lag in HO
        return lags, C
    return None, None


def make_figure(autocorr: dict, barrier: dict, psd: dict, gmm: dict):
    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    ax_a = fig.add_subplot(gs[0, 0])  # tau_int vs N
    ax_b = fig.add_subplot(gs[0, 1])  # C(t) curves
    ax_c = fig.add_subplot(gs[0, 2])  # alpha vs N
    ax_d = fig.add_subplot(gs[1, 0])  # tau_int vs alpha
    ax_e = fig.add_subplot(gs[1, 1])  # barrier crossings vs N
    ax_f = fig.add_subplot(gs[1, 2])  # ESS/eval vs N

    # ---- Panel (a): tau_int vs N ----
    tau_by_N = {}
    for N in N_LIST:
        t = get_tau_int(autocorr, N)
        if t is not None:
            tau_by_N[N] = t
    if tau_by_N:
        Ns = sorted(tau_by_N.keys())
        taus = [tau_by_N[N] for N in Ns]
        ax_a.semilogy(Ns, taus, 'o-', color='steelblue', lw=2, ms=7, zorder=3)
        # Highlight N=3
        if 3 in tau_by_N:
            ax_a.semilogy([3], [tau_by_N[3]], 'o', color='red', ms=10, zorder=4,
                          label=f'N=3 (min, τ={tau_by_N[3]:.0f})')
            ax_a.legend(fontsize=8, loc='upper left')
    ax_a.set_xlabel('N (thermostat scales)')
    ax_a.set_ylabel('τ_int (force evals)')
    ax_a.set_title('(a) Autocorrelation time vs N')
    ax_a.set_xticks(N_LIST)
    ax_a.grid(True, alpha=0.3)

    # ---- Panel (b): C(t) curves for N=1,3,5 ----
    plotted_b = False
    for N, color in HIGHLIGHT.items():
        lags, C = get_C_curve(autocorr, N)
        if lags is not None and C is not None and len(C) > 10:
            # Only plot positive part in log scale
            pos = C > 1e-4
            if pos.sum() > 2:
                ax_b.semilogy(lags[pos], C[pos], '-', color=color, lw=1.5,
                              label=f'N={N}', alpha=0.85)
                plotted_b = True
    if plotted_b:
        ax_b.legend(fontsize=9)
        ax_b.set_xlabel('Lag (force evals)')
        ax_b.set_ylabel('C(t)')
        ax_b.set_title('(b) Autocorrelation function')
        ax_b.grid(True, alpha=0.3)
        # Mark tau_int for each
        for N, color in HIGHLIGHT.items():
            t = get_tau_int(autocorr, N)
            if t is not None:
                ax_b.axvline(t, color=color, linestyle='--', alpha=0.5, lw=1)
    else:
        ax_b.text(0.5, 0.5, 'No C(t) data\n(run make_autocorr.py)', ha='center', va='center',
                  transform=ax_b.transAxes)
        ax_b.set_title('(b) Autocorrelation function')

    # ---- Panel (c): alpha vs N ----
    if psd:
        Ns_psd = sorted(psd.keys())
        alphas_mid = [psd[N]['alpha_mid'] for N in Ns_psd]
        alphas_dh  = [psd[N]['alpha_dh_band'] for N in Ns_psd]
        ax_c.plot(Ns_psd, alphas_mid, 's-', color='darkorange', lw=2, ms=7,
                  label='α (0.1–10 Hz band)')
        ax_c.plot(Ns_psd, alphas_dh, '^--', color='gray', lw=1.5, ms=6,
                  label='α (DH band)', alpha=0.7)
        ax_c.axhline(1.0, color='blue', linestyle=':', lw=1.5, alpha=0.6, label='1/f (α=1)')
        ax_c.axhline(2.0, color='green', linestyle=':', lw=1.5, alpha=0.6, label='Brownian (α=2)')
        # Annotate N=3
        if 3 in psd:
            a3 = psd[3]['alpha_mid']
            ax_c.annotate(f'N=3\nα={a3:.2f}', xy=(3, a3), xytext=(4.5, a3 - 1.5),
                          arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                          fontsize=8, color='red')
        ax_c.set_xlabel('N (thermostat scales)')
        ax_c.set_ylabel('PSD exponent α')
        ax_c.set_title('(c) 1/f exponent α vs N')
        ax_c.set_xticks(Ns_psd)
        ax_c.legend(fontsize=7, loc='upper right')
        ax_c.grid(True, alpha=0.3)
        ax_c.set_ylim(-0.5, 14)
    else:
        ax_c.text(0.5, 0.5, 'No PSD data', ha='center', va='center', transform=ax_c.transAxes)
        ax_c.set_title('(c) 1/f exponent α vs N')

    # ---- Panel (d): tau_int vs alpha (empirical scaling law) ----
    tau_alpha_pairs = []
    for N in N_LIST:
        t = get_tau_int(autocorr, N)
        if t is not None and N in psd:
            alpha = psd[N]['alpha_mid']
            tau_alpha_pairs.append((alpha, t, N))

    if len(tau_alpha_pairs) >= 3:
        alphas_v = np.array([p[0] for p in tau_alpha_pairs])
        taus_v   = np.array([p[1] for p in tau_alpha_pairs])
        # Scatter
        for alpha_v, tau_v, N in tau_alpha_pairs:
            c = COLORS.get(N, 'gray')
            ax_d.semilogy(alpha_v, tau_v, 'o', color=c, ms=9, zorder=3)
            ax_d.annotate(f'N={N}', (alpha_v, tau_v), textcoords='offset points',
                          xytext=(4, 3), fontsize=7)
        # Fit log(tau) ~ c * alpha (linear in log-space vs alpha)
        try:
            # Filter out N=1 outlier (alpha=12 is off the scale)
            mask = alphas_v < 10
            if mask.sum() >= 2:
                c_fit = np.polyfit(alphas_v[mask], np.log(taus_v[mask]), 1)
                alpha_fit = np.linspace(alphas_v[mask].min(), alphas_v[mask].max(), 50)
                tau_fit = np.exp(np.polyval(c_fit, alpha_fit))
                ax_d.semilogy(alpha_fit, tau_fit, '--', color='black', lw=1.5, alpha=0.6,
                              label=f'log(τ) ∝ {c_fit[0]:.2f}·α')
                ax_d.legend(fontsize=8)
        except Exception:
            pass
        ax_d.set_xlabel('PSD exponent α')
        ax_d.set_ylabel('τ_int (force evals)')
        ax_d.set_title('(d) Scaling law: τ_int vs α')
        ax_d.grid(True, alpha=0.3)
    else:
        # Fall back to GMM KL vs N if tau data sparse
        if gmm:
            Ns_g = sorted(gmm.keys())
            kls = [gmm[N]['mean'] for N in Ns_g]
            ax_d.semilogy(Ns_g, kls, 'o-', color='purple', lw=2, ms=7)
            ax_d.set_xlabel('N')
            ax_d.set_ylabel('GMM KL divergence')
            ax_d.set_title('(d) GMM KL vs N (proxy for τ_int)')
            ax_d.set_xticks(Ns_g)
            ax_d.grid(True, alpha=0.3)
        else:
            ax_d.text(0.5, 0.5, 'Need ≥3 (alpha, tau) pairs', ha='center', va='center',
                      transform=ax_d.transAxes)
            ax_d.set_title('(d) Scaling law: τ_int vs α')

    # ---- Panel (e): barrier crossings vs N ----
    if barrier and 'data' in barrier:
        lambda_list = barrier.get('lambda_list', [1, 2, 4, 8])
        N_bc = barrier.get('N_list', [1, 2, 3, 5])
        cmap = plt.cm.viridis
        lam_colors = {lam: cmap(i / max(len(lambda_list) - 1, 1))
                      for i, lam in enumerate(lambda_list)}
        for lam in lambda_list:
            lam_str = str(lam)
            if lam_str not in barrier['data']:
                continue
            lam_data = barrier['data'][lam_str]
            rates = []
            for N in N_bc:
                nd = lam_data.get(str(N), {})
                rates.append(nd.get('rate_per_1k_evals', 0))
            ax_e.semilogy(N_bc, [max(r, 1e-4) for r in rates], 'o-',
                          color=lam_colors[lam], lw=2, ms=7,
                          label=f'λ={lam} ({lam}kT barrier)')
        ax_e.set_xlabel('N (thermostat scales)')
        ax_e.set_ylabel('Crossings per 1k evals')
        ax_e.set_title('(e) Barrier crossings vs N')
        ax_e.set_xticks(N_bc)
        ax_e.legend(fontsize=8, loc='upper right')
        ax_e.grid(True, alpha=0.3)
    else:
        ax_e.text(0.5, 0.5, 'No barrier data\n(run make_barrier_crossing.py)',
                  ha='center', va='center', transform=ax_e.transAxes)
        ax_e.set_title('(e) Barrier crossings vs N')

    # ---- Panel (f): ESS/eval vs N ----
    ess_by_N = {}
    for N in N_LIST:
        e = get_ess_per_eval(autocorr, N)
        if e is not None and e > 0:
            ess_by_N[N] = e
    if ess_by_N:
        Ns_e = sorted(ess_by_N.keys())
        ess_v = [ess_by_N[N] for N in Ns_e]
        ax_f.semilogy(Ns_e, ess_v, 'o-', color='teal', lw=2, ms=7, zorder=3)
        if 3 in ess_by_N:
            ax_f.semilogy([3], [ess_by_N[3]], 'o', color='red', ms=11, zorder=4,
                          label=f'N=3 (max, {ess_by_N[3]:.4f}/eval)')
            ax_f.legend(fontsize=8)
    else:
        # Fall back to GMM KL inverse as proxy
        if gmm:
            Ns_g = sorted(gmm.keys())
            ess_proxy = [1.0 / gmm[N]['mean'] for N in Ns_g]
            ax_f.semilogy(Ns_g, ess_proxy, 'o-', color='teal', lw=2, ms=7)
    ax_f.set_xlabel('N (thermostat scales)')
    ax_f.set_ylabel('ESS / force eval')
    ax_f.set_title('(f) Sampling efficiency vs N')
    ax_f.set_xticks(N_LIST)
    ax_f.grid(True, alpha=0.3)

    plt.suptitle('Mixing Rates: 1/f Noise and Autocorrelation Times\n'
                 'Multi-scale log-osc thermostat (mixing-rates-019)',
                 fontsize=12, y=1.01)

    outpath = os.path.join(FIG_DIR, 'mixing_rates_summary.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close(fig)
    return outpath


def print_scaling_law(autocorr: dict, psd: dict):
    """Print the empirical scaling law tau_int ~ f(alpha)."""
    pairs = []
    for N in N_LIST:
        t = get_tau_int(autocorr, N)
        if t is not None and N in psd:
            alpha = psd[N]['alpha_mid']
            pairs.append((alpha, t, N))

    if not pairs:
        print("No tau_int vs alpha data available.")
        return

    print("\n=== Empirical Scaling Law: tau_int vs alpha ===")
    print(f"{'N':>4}  {'alpha':>8}  {'tau_int':>12}  {'log(tau)':>10}")
    print("-" * 42)
    for alpha, tau, N in sorted(pairs, key=lambda x: x[0]):
        print(f"{N:>4}  {alpha:>8.3f}  {tau:>12.1f}  {np.log(tau):>10.3f}")

    # Fit excluding extreme outliers (N=1, alpha=12)
    alphas = np.array([p[0] for p in pairs])
    taus = np.array([p[1] for p in pairs])
    mask = alphas < 10
    if mask.sum() >= 2:
        c_fit = np.polyfit(alphas[mask], np.log(taus[mask]), 1)
        print(f"\nFit (alpha < 10): log(tau_int) = {c_fit[0]:.3f} * alpha + {c_fit[1]:.3f}")
        print(f"  => tau_int ∝ exp({c_fit[0]:.3f} * alpha)")
        print(f"  => {c_fit[0]:.3f} nats per unit alpha")


def main():
    print("=" * 70)
    print("Generating mixing-rates-019 consolidated figures")
    print("=" * 70)

    autocorr = load_autocorr()
    barrier = load_barrier()
    psd = load_psd()
    gmm = load_gmm()

    print(f"Loaded: autocorr N={list(autocorr.keys())[:8]}")
    print(f"Loaded: barrier lambda={barrier.get('lambda_list', [])}")
    print(f"Loaded: psd N={sorted(psd.keys())}")
    print(f"Loaded: gmm N={sorted(gmm.keys())}")

    print_scaling_law(autocorr, psd)

    outpath = make_figure(autocorr, barrier, psd, gmm)
    print(f"\nFigure saved: {outpath}")
    return outpath


if __name__ == '__main__':
    main()
