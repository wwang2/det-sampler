#!/usr/bin/env python3
"""Figure 3: The Mechanism -- How Bounded Friction Breaks KAM Tori.

2x3 panel layout:
  (a) NHC trajectory on 1D HO in (q,p) -- thin elliptical torus
  (b) Log-Osc trajectory on 1D HO in (q,p) -- space-filling
  (c) Poincare section comparison (p vs xi at q=0 crossing)
  (d) g(xi) time series: NHC (growing) vs Log-Osc (bounded)
  (e) Kinetic energy time series: NHC (periodic) vs Log-Osc (chaotic)
  (f) Lyapunov exponent vs Q for NHC vs Log-Osc (with error bars)

Q=0.5, dt=0.005, 1M force evals.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from shared import (
    COLOR_NHC, COLOR_LO, FONTSIZE_LABEL, FONTSIZE_TICK, FONTSIZE_TITLE,
    FONTSIZE_ANNOT, DPI, g_func,
    LogOscThermostat, LogOscVelocityVerlet,
    run_trajectory,
)
from research.eval.potentials import HarmonicOscillator1D
from research.eval.baselines import NoseHooverChain
from research.eval.integrators import VelocityVerletThermostat, ThermostatState

N_EVALS = 1_000_000
Q = 0.5
DT = 0.005
SEED = 42


def run_with_xi(sampler_cls, integ_cls, n_evals=N_EVALS, seed=SEED, **kw):
    pot = HarmonicOscillator1D(omega=1.0)
    return run_trajectory(
        sampler_cls, integ_cls, pot, dt=DT, n_force_evals=n_evals,
        seed=seed, q0=np.array([1.0]), collect_xi=True, **kw
    )


def find_poincare_crossings(qs, ps, xis):
    """Find zero-crossings of q (Poincare section at q=0)."""
    p_cross = []
    xi_cross = []
    for i in range(1, len(qs)):
        if qs[i-1] < 0 and qs[i] >= 0:  # upward crossing
            # Linear interpolation
            frac = -qs[i-1] / (qs[i] - qs[i-1] + 1e-30)
            p_cross.append(ps[i-1] + frac * (ps[i] - ps[i-1]))
            xi_cross.append(xis[i-1] + frac * (xis[i] - xis[i-1]))
    return np.array(p_cross), np.array(xi_cross)


def estimate_lyapunov(sampler_cls, integ_cls, Q_val, n_evals=200_000,
                       seed=42, eps=1e-8, **kw):
    """Estimate max Lyapunov exponent via tangent-vector rescaling."""
    pot = HarmonicOscillator1D(omega=1.0)
    rng = np.random.default_rng(seed)

    dim = pot.dim
    dyn = sampler_cls(dim=dim, Q=Q_val, **kw)
    q0 = np.array([1.0])
    state = dyn.initial_state(q0, rng=rng)
    integ = integ_cls(dyn, pot, dt=DT)

    # Perturbed trajectory
    dyn2 = sampler_cls(dim=dim, Q=Q_val, **kw)
    state2 = ThermostatState(
        state.q + eps * rng.normal(size=dim),
        state.p + eps * rng.normal(size=dim),
        state.xi + eps * rng.normal(size=len(state.xi)),
        0
    )
    integ2 = integ_cls(dyn2, pot, dt=DT)

    lyap_sum = 0.0
    n_rescale = 0
    rescale_interval = 100

    step_count = 0
    while state.n_force_evals < n_evals:
        state = integ.step(state)
        state2 = integ2.step(state2)
        step_count += 1

        if np.any(np.isnan(state.q)) or np.any(np.isnan(state2.q)):
            break

        if step_count % rescale_interval == 0:
            delta = np.concatenate([
                state2.q - state.q,
                state2.p - state.p,
                state2.xi - state.xi
            ])
            d = np.linalg.norm(delta)
            if d > 0:
                lyap_sum += np.log(d / eps)
                n_rescale += 1
                # Rescale perturbation
                scale = eps / d
                state2 = ThermostatState(
                    state.q + scale * (state2.q - state.q),
                    state.p + scale * (state2.p - state.p),
                    state.xi + scale * (state2.xi - state.xi),
                    state2.n_force_evals
                )
                # Reset integrator cache
                integ2._cached_grad_U = None

    if n_rescale == 0:
        return 0.0
    t_total = step_count * DT
    return lyap_sum / t_total


def make_figure():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    # Run NHC and Log-Osc
    print("Running NHC on HO...")
    nhc_result = run_with_xi(NoseHooverChain, VelocityVerletThermostat,
                              chain_length=3, Q=Q)
    print("Running Log-Osc on HO...")
    lo_result = run_with_xi(LogOscThermostat, LogOscVelocityVerlet, Q=Q)

    nhc_q = nhc_result['q'].ravel()
    nhc_p = nhc_result['p'].ravel()
    nhc_xi = nhc_result['xi'][:, 0]
    lo_q = lo_result['q'].ravel()
    lo_p = lo_result['p'].ravel()
    lo_xi = lo_result['xi'][:, 0]

    # ── Panel (a): NHC phase portrait ──
    ax = axes[0, 0]
    thin = max(1, len(nhc_q) // 40000)
    ax.scatter(nhc_q[::thin], nhc_p[::thin], s=0.2, alpha=0.3, color=COLOR_NHC,
               rasterized=True)
    theta = np.linspace(0, 2*np.pi, 200)
    for sigma in [1, 2, 3]:
        ax.plot(sigma*np.cos(theta), sigma*np.sin(theta), '--', color='gray',
                alpha=0.5, lw=0.8)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect('equal')
    ax.set_xlabel(r'$q$', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'$p$', fontsize=FONTSIZE_LABEL)
    ax.set_title('NHC (M=3)', fontsize=FONTSIZE_TITLE - 2, color=COLOR_NHC)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.text(0.03, 0.95, panel_labels[0], transform=ax.transAxes,
            fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    # ── Panel (b): Log-Osc phase portrait ──
    ax = axes[0, 1]
    thin = max(1, len(lo_q) // 40000)
    ax.scatter(lo_q[::thin], lo_p[::thin], s=0.2, alpha=0.3, color=COLOR_LO,
               rasterized=True)
    for sigma in [1, 2, 3]:
        ax.plot(sigma*np.cos(theta), sigma*np.sin(theta), '--', color='gray',
                alpha=0.5, lw=0.8)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect('equal')
    ax.set_xlabel(r'$q$', fontsize=FONTSIZE_LABEL)
    ax.set_title('Log-Osc', fontsize=FONTSIZE_TITLE - 2, color=COLOR_LO)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.text(0.03, 0.95, panel_labels[1], transform=ax.transAxes,
            fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    # ── Panel (c): Poincare section ──
    ax = axes[0, 2]
    p_nhc, xi_nhc = find_poincare_crossings(nhc_q, nhc_p, nhc_xi)
    p_lo, xi_lo = find_poincare_crossings(lo_q, lo_p, lo_xi)

    if len(p_lo) > 0:
        ax.scatter(xi_lo, p_lo, s=2, alpha=0.5, color=COLOR_LO,
                   label='Log-Osc', rasterized=True, zorder=3)
    if len(p_nhc) > 0:
        # Clip NHC xi for visualization (shows structure near origin)
        mask_nhc = np.abs(xi_nhc) < 10
        ax.scatter(xi_nhc[mask_nhc], p_nhc[mask_nhc], s=2, alpha=0.5,
                   color=COLOR_NHC, label='NHC', rasterized=True, zorder=2)
    ax.set_xlabel(r'$\xi$', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'$p$ at $q=0$', fontsize=FONTSIZE_LABEL)
    ax.set_title('Poincar\u00e9 section', fontsize=FONTSIZE_TITLE - 2)
    ax.set_xlim(-8, 8)
    ax.legend(fontsize=FONTSIZE_ANNOT, markerscale=4)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.text(0.03, 0.95, panel_labels[2], transform=ax.transAxes,
            fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    # ── Panel (d): g(xi) time series ──
    ax = axes[1, 0]
    t_window = slice(0, 50000)
    t = np.arange(len(nhc_q))[t_window] * DT

    # For NHC, friction = xi[0] (unbounded)
    ax.plot(t, nhc_xi[t_window], color=COLOR_NHC, lw=0.4, alpha=0.7,
            label=r'NHC: $\xi_1(t)$')
    # For Log-Osc, friction = g(xi) (bounded)
    g_lo = g_func(lo_xi[t_window])
    ax.plot(t, g_lo, color=COLOR_LO, lw=0.4, alpha=0.7,
            label=r'Log-Osc: $g(\xi(t))$')
    ax.axhline(1, color=COLOR_LO, ls=':', lw=1, alpha=0.5)
    ax.axhline(-1, color=COLOR_LO, ls=':', lw=1, alpha=0.5)
    ax.set_xlabel('Time', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Friction signal', fontsize=FONTSIZE_LABEL)
    ax.legend(fontsize=FONTSIZE_ANNOT, loc='upper right')
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.text(0.03, 0.95, panel_labels[3], transform=ax.transAxes,
            fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    # ── Panel (e): Kinetic energy time series ──
    ax = axes[1, 1]
    KE_nhc = 0.5 * nhc_p[t_window]**2
    KE_lo = 0.5 * lo_p[t_window]**2
    ax.plot(t, KE_nhc, color=COLOR_NHC, lw=0.3, alpha=0.6, label='NHC')
    ax.plot(t, KE_lo, color=COLOR_LO, lw=0.3, alpha=0.6, label='Log-Osc')
    ax.axhline(0.5, color='gray', ls='--', lw=1, alpha=0.5, label=r'$\langle K \rangle = kT/2$')
    ax.set_xlabel('Time', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Kinetic energy', fontsize=FONTSIZE_LABEL)
    ax.legend(fontsize=FONTSIZE_ANNOT, loc='upper right')
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.text(0.03, 0.95, panel_labels[4], transform=ax.transAxes,
            fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    # ── Panel (f): Lyapunov exponent vs Q ──
    ax = axes[1, 2]
    Q_vals = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    seeds = [42, 123, 7]  # 3 seeds for error bars

    nhc_lyaps = []
    lo_lyaps = []
    for Qv in Q_vals:
        print(f"Computing Lyapunov for Q={Qv}...")
        nhc_l = [estimate_lyapunov(NoseHooverChain, VelocityVerletThermostat, Qv,
                                    n_evals=100_000, seed=s, chain_length=3)
                 for s in seeds]
        lo_l = [estimate_lyapunov(LogOscThermostat, LogOscVelocityVerlet, Qv,
                                   n_evals=100_000, seed=s)
                for s in seeds]
        nhc_lyaps.append(nhc_l)
        lo_lyaps.append(lo_l)

    nhc_lyaps = np.array(nhc_lyaps)
    lo_lyaps = np.array(lo_lyaps)

    ax.errorbar(Q_vals, nhc_lyaps.mean(axis=1), yerr=nhc_lyaps.std(axis=1),
                color=COLOR_NHC, marker='o', lw=2, capsize=4, label='NHC (M=3)')
    ax.errorbar(Q_vals, lo_lyaps.mean(axis=1), yerr=lo_lyaps.std(axis=1),
                color=COLOR_LO, marker='s', lw=2, capsize=4, label='Log-Osc')
    ax.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel(r'$Q$', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Max Lyapunov exponent', fontsize=FONTSIZE_LABEL)
    ax.set_xscale('log')
    ax.legend(fontsize=FONTSIZE_ANNOT)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.text(0.03, 0.95, panel_labels[5], transform=ax.transAxes,
            fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    fig.suptitle('Bounded Friction Breaks KAM Tori and Enables Ergodic Sampling',
                 fontsize=FONTSIZE_TITLE, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fig3_mechanism.png')
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == '__main__':
    make_figure()
