"""q-exponent-theory-041: Why Q_opt ~ omega^{-1.55} for the log-osc thermostat.

Key finding: The -1.55 exponent is a crossover between two regimes:
  Regime A (omega < 0.73): resonance matching, exponent ~ -2
  Regime B (omega > 0.73): thermostat cannot resonate, Q_opt ~ constant
"""

import numpy as np
from scipy import integrate, optimize, stats, special
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, json

OUTDIR = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(OUTDIR, "figures")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 14, "axes.titlesize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
})
COL_NHC = "#ff7f0e"
COL_LO  = "#2ca02c"
COL_TH  = "#d62728"
COL_RES = "#9467bd"
COL_B   = "#17becf"


# =====================================================================
# Analytics
# =====================================================================

def gprime_avg(Q):
    """<g'(xi)>_Q = (2Q-1)/(Q+1).  Valid for Q > 0.5."""
    if Q <= 0.5: return float('nan')
    return (2*Q - 1) / (Q + 1)

def omega_xi_sq(Q):
    if Q <= 0.5: return float('nan')
    return (2*Q - 1) / (Q*(Q + 1))

def Q_resonance(omega):
    """Solve omega_xi^2(Q) = omega^2."""
    a, b, c = omega**2, omega**2 - 2, 1.0
    disc = b**2 - 4*a*c
    if disc < 0: return np.nan
    q1 = (-b + np.sqrt(disc)) / (2*a)
    q2 = (-b - np.sqrt(disc)) / (2*a)
    cands = [q for q in [q1, q2] if q > 0.5]
    return max(cands) if cands else np.nan

def verify_gprime():
    def Z_Q(Q):
        return np.sqrt(np.pi) * special.gamma(Q - 0.5) / special.gamma(Q)
    print("Verifying <g'> = (2Q-1)/(Q+1):")
    for Q in [1.0, 2.0, 5.0, 10.0, 50.0]:
        z = Z_Q(Q)
        val, _ = integrate.quad(
            lambda xi: 2*(1-xi**2)/(1+xi**2)**2 * (1+xi**2)**(-Q), -50, 50, limit=200)
        num, ana = val/z, gprime_avg(Q)
        print(f"  Q={Q:5.1f}: num={num:.6f}  ana={ana:.6f}  OK={abs(num-ana)<1e-4}")


# =====================================================================
# Load parent orbit tau curves
# =====================================================================

def load_parent_curves():
    """Load tau(Q) curves from orbit #040 results."""
    path = os.path.join(OUTDIR, "..", "q-omega-mapping-040", "results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    curves = {}
    for om_str, qdict in d["logosc_curves"].items():
        om = float(om_str)
        Qs = sorted([float(q) for q in qdict.keys()])
        taus = [qdict[str(q)] for q in Qs]
        curves[om] = {"Qs": np.array(Qs), "taus": np.array(taus)}
    return curves


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 60)
    print("Q-exponent theory: log-osc thermostat")
    print("=" * 60)

    verify_gprime()

    Q_star = (1 + np.sqrt(3)) / 2
    omega_xi_max = np.sqrt(omega_xi_sq(Q_star))
    print(f"\nomega_xi_max = {omega_xi_max:.4f} at Q* = {Q_star:.4f}")

    # Empirical data
    emp_om = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    emp_q_lo = np.array([177.83, 17.78, 1.78, 0.1, 0.0316, 0.0562])
    emp_q_nhc = np.array([100.0, 10.0, 1.0, 0.1, 0.01, 0.001])

    # Regime classification
    print("\nREGIME CLASSIFICATION")
    print(f"{'omega':>8} {'Q_opt':>10} {'Q_res':>10} {'ratio':>8} {'regime':>12}")
    q_res_emp = np.array([Q_resonance(om) for om in emp_om])
    for i, om in enumerate(emp_om):
        qo, qr = emp_q_lo[i], q_res_emp[i]
        ratio = qo/qr if np.isfinite(qr) else np.nan
        regime = "A" if qo > 0.5 and np.isfinite(qr) else ("B" if qo < 0.5 else "crossover")
        print(f"{om:8.1f} {qo:10.4f} {qr:10.4f} {ratio:8.3f} {regime:>12}")

    # Fits by regime
    mask_A = emp_q_lo > 0.5
    mask_B = emp_q_lo < 0.5

    sl_A = it_A = r_A = np.nan
    if np.sum(mask_A) >= 2:
        sl_A, it_A, r_A, _, _ = stats.linregress(
            np.log(emp_om[mask_A]), np.log(emp_q_lo[mask_A]))
        print(f"\nRegime A: Q_opt ~ {np.exp(it_A):.3f} * omega^({sl_A:.3f}), R2={r_A**2:.4f}")

    sl_B = it_B = r_B = np.nan
    if np.sum(mask_B) >= 2:
        sl_B, it_B, r_B, _, _ = stats.linregress(
            np.log(emp_om[mask_B]), np.log(emp_q_lo[mask_B]))
        print(f"Regime B: Q_opt ~ {np.exp(it_B):.3f} * omega^({sl_B:.3f}), R2={r_B**2:.4f}")

    sl_f, it_f, r_f, _, _ = stats.linregress(np.log(emp_om), np.log(emp_q_lo))
    print(f"Full:     Q_opt ~ {np.exp(it_f):.3f} * omega^({sl_f:.3f}), R2={r_f**2:.4f}")

    # Resonance curve prediction
    om_res = np.logspace(-1.5, np.log10(omega_xi_max * 0.98), 50)
    q_res_th = np.array([Q_resonance(om) for om in om_res])
    valid_r = np.isfinite(q_res_th)
    sl_res, it_res, r_res, _, _ = stats.linregress(
        np.log(om_res[valid_r]), np.log(q_res_th[valid_r]))
    print(f"Resonance theory: Q ~ {np.exp(it_res):.3f} * omega^({sl_res:.3f}), R2={r_res**2:.4f}")

    # Load parent tau curves for the bimodal landscape figure
    parent_curves = load_parent_curves()

    # =====================================================================
    # FIGURE
    # =====================================================================
    print("\nGenerating figure...")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.subplots_adjust(hspace=0.32, wspace=0.30)

    # -- (a) omega_xi(Q) --
    ax = axes[0, 0]
    Q_arr = np.logspace(-0.15, 3, 500)
    wxi = np.sqrt(np.array([max(omega_xi_sq(Q), 0) if Q > 0.5 else 0 for Q in Q_arr]))
    ax.plot(Q_arr, wxi, 'k-', lw=2, label=r'$\omega_\xi(Q)$')
    ax.axhline(omega_xi_max, color='gray', ls=':', alpha=0.5)
    ax.text(500, omega_xi_max*1.05, f'$\\omega_{{\\xi,\\max}}={omega_xi_max:.3f}$',
            fontsize=10, color='gray', va='bottom')
    # Mark empirical resonance points
    for om, qo in zip(emp_om[:2], emp_q_lo[:2]):
        ax.plot(qo, om, 'o', color=COL_LO, ms=10, zorder=5, mec='k', mew=0.5)
        ax.annotate(f'$\\omega={om}$', xy=(qo, om), fontsize=9,
                    xytext=(qo*0.25, om*1.3),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
    # Shade no-resonance zone
    ylim_top = 3.0
    ax.fill_between([Q_arr[0], Q_arr[-1]], omega_xi_max, ylim_top,
                    alpha=0.08, color='red')
    ax.text(5, 1.3, 'No resonance\npossible', fontsize=10, color='red', alpha=0.6,
            ha='center')
    ax.set_xlabel('Q'); ax.set_ylabel(r'$\omega_\xi(Q)$')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_ylim(0.015, ylim_top)
    ax.set_title('(a) Thermostat natural frequency')
    ax.legend(fontsize=10, loc='lower left')

    # -- (b) Bimodal tau landscape from parent data --
    ax = axes[0, 1]
    if parent_curves:
        for om, col, ls in [(0.3, COL_LO, '-'), (1.0, '#1f77b4', '--'),
                             (10.0, COL_B, '-.')]:
            if om in parent_curves:
                c = parent_curves[om]
                ax.plot(c["Qs"], c["taus"], color=col, lw=2, ls=ls,
                        label=f'$\\omega={om}$')
                i_opt = np.argmin(c["taus"])
                ax.plot(c["Qs"][i_opt], c["taus"][i_opt], 'v', color=col,
                        ms=10, zorder=5, mec='k', mew=0.5)
    ax.axvspan(0.08, 0.5, alpha=0.08, color='red')
    ax.text(0.18, 3, 'Spike\nregion', fontsize=9, color='red', alpha=0.6,
            ha='center', va='bottom')
    ax.axvline(Q_star, color='gray', ls=':', alpha=0.3)
    ax.set_xlabel('Q'); ax.set_ylabel(r'$\tau_{\rm int}(q^2)$')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title('(b) Autocorrelation landscape (data from #040)')
    ax.legend(fontsize=9, loc='upper right')

    # -- (c) Q_opt(omega) two regimes --
    ax = axes[1, 0]
    ax.scatter(emp_om, emp_q_lo, s=100, zorder=5, color=COL_LO,
               marker='o', edgecolors='k', lw=0.5, label='Log-osc (#040)')
    ax.scatter(emp_om, emp_q_nhc, s=100, zorder=5, color=COL_NHC,
               marker='s', edgecolors='k', lw=0.5, label='NHC (#040)')

    om_fit = np.logspace(-1.5, 1.7, 100)
    ax.plot(om_fit, np.exp(it_f) * om_fit**sl_f, color=COL_LO, ls='--', alpha=0.4, lw=1,
            label=f'Naive fit: $\\omega^{{{sl_f:.2f}}}$')
    ax.plot(om_fit, 0.95 * om_fit**(-2.0), color=COL_NHC, ls='--', alpha=0.4, lw=1,
            label=r'NHC: $\omega^{-2}$')

    # Resonance theory
    ax.plot(om_res[valid_r], q_res_th[valid_r], color=COL_TH, lw=2.5,
            label=f'Resonance theory: $\\omega^{{{sl_res:.2f}}}$')

    # Shade regimes
    ax.axvline(omega_xi_max, color='gray', ls='--', alpha=0.3)
    ax.axvspan(0.02, omega_xi_max, alpha=0.04, color='blue')
    ax.axvspan(omega_xi_max, 60, alpha=0.04, color='red')
    ax.text(0.12, 0.002, 'Regime A\n(resonance)', fontsize=9, color='blue', alpha=0.7)
    ax.text(5, 0.002, 'Regime B\n(driven)', fontsize=9, color='red', alpha=0.7)

    ax.set_xlabel(r'$\omega$'); ax.set_ylabel(r'$Q_{\rm opt}$')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_ylim(5e-4, 1e4)
    ax.set_title(r'(c) $Q_{\rm opt}(\omega)$: two-regime structure')
    ax.legend(fontsize=7.5, loc='upper right', ncol=1)

    # -- (d) Exponent bar chart --
    ax = axes[1, 1]
    items = [
        ("Regime A\n(resonance)", sl_A, COL_TH),
        ("Regime B\n(driven)", sl_B, COL_B),
        ("Full range\n(crossover)", sl_f, COL_LO),
        ("NHC\n(exact)", -2.0, COL_NHC),
    ]
    names = [x[0] for x in items]
    slopes = [x[1] for x in items]
    colors = [x[2] for x in items]
    y_pos = np.arange(len(items))
    ax.barh(y_pos, slopes, color=colors, alpha=0.7, edgecolor='k', lw=0.5)
    for i, s in enumerate(slopes):
        xoff = -0.08 if s < -0.5 else 0.05
        ha = 'right' if s < -0.5 else 'left'
        ax.text(s + xoff, i, f'{s:.2f}', ha=ha, va='center', fontsize=11, fontweight='bold')
    ax.set_yticks(y_pos); ax.set_yticklabels(names, fontsize=10)
    ax.axvline(-1.55, color=COL_LO, ls='--', alpha=0.3)
    ax.axvline(-2.0, color=COL_NHC, ls='--', alpha=0.3)
    ax.set_xlabel(r'Exponent $\alpha$ in $Q_{\rm opt} \propto \omega^\alpha$')
    ax.set_title('(d) Exponents by regime')
    ax.set_xlim(-2.5, 0.5)

    fig_path = os.path.join(FIGDIR, "fig_theory_vs_empirical.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fig_path}")

    # ---- Save results ----
    summary = {
        "key_finding": (
            "The -1.55 exponent is a crossover artifact. "
            "Regime A (omega<0.73): resonance, exponent=-2.00. "
            "Regime B (omega>0.73): driven, exponent=-0.26. "
            "Single power-law fit across both gives -1.55."
        ),
        "formulas": {
            "gprime_avg": "(2Q-1)/(Q+1)",
            "omega_xi_sq": "(2Q-1)/(Q(Q+1))",
            "omega_xi_max": float(omega_xi_max),
            "Q_star": float(Q_star),
        },
        "regime_A_exponent": float(sl_A),
        "regime_B_exponent": float(sl_B),
        "full_range_exponent": float(sl_f),
        "resonance_exponent": float(sl_res),
        "empirical_exponent": -1.55,
        "nhc_exponent": -2.0,
    }
    with open(os.path.join(OUTDIR, "theory_results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\nCONCLUSIONS")
    print(f"  Regime A exponent: {sl_A:.2f} (resonance, R2={r_A**2:.3f})")
    print(f"  Regime B exponent: {sl_B:.2f} (driven, R2={r_B**2:.3f})")
    print(f"  Full range:        {sl_f:.2f} (crossover)")
    print(f"  Empirical:         -1.55")
    print(f"  NHC:               -2.00")


if __name__ == "__main__":
    main()
