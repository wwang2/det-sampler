"""Task 4: Phase Space Analysis for ESH vs ESH+Thermostat.

For 1D HO, plots trajectories in (q, v) phase space for:
  - Pure ESH: conserves H_ESH level sets (ellipse-like curves)
  - ESH + thermostat: should fill phase space (hops between level sets)

Shows visually how the thermostat "hops" between ESH level sets.
"""

import sys
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

WORKTREE = '/Users/wujiewang/code/det-sampler/.worktrees/esh-hybrid-026'
sys.path.insert(0, WORKTREE)
sys.path.insert(0, str(Path(WORKTREE) / 'orbits/esh-hybrid-026'))

from research.eval.potentials import HarmonicOscillator1D
from research.eval.integrators import ThermostatState
from make_hybrid_sampler import ESHPlusThermostat, ESHHybridIntegrator

OUT_DIR = Path(WORKTREE) / 'orbits/esh-hybrid-026/figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

KT = 1.0


# ============================================================
# Pure ESH leapfrog (1D)
# ============================================================

def run_pure_esh(pot, q0, p0, dt, n_steps):
    """Run pure ESH leapfrog, return trajectory arrays."""
    q, p = np.array([q0], dtype=float), np.array([p0], dtype=float)
    qs, ps = [q[0]], [p[0]]
    H_eshs = [pot.energy(q) + np.log(abs(p[0]) + 1e-300)]

    grad_U = pot.gradient(q)

    for _ in range(n_steps):
        half_dt = 0.5 * dt
        p_norm = abs(p[0])
        if p_norm < 1e-300:
            p_norm = 1e-300

        # Half-step p (d=1, ||p||/d = |p|)
        p_half = p - half_dt * grad_U * p_norm
        p_half_norm = abs(p_half[0])
        if p_half_norm < 1e-300:
            p_half_norm = 1e-300

        # Full-step q
        q = q + dt * p_half / p_half_norm

        grad_U = pot.gradient(q)
        p = p_half - half_dt * grad_U * p_half_norm

        if np.any(np.isnan(q)) or np.any(np.isnan(p)):
            break
        qs.append(q[0])
        ps.append(p[0])
        H_eshs.append(pot.energy(q) + np.log(abs(p[0]) + 1e-300))

    return np.array(qs), np.array(ps), np.array(H_eshs)


# ============================================================
# ESH level sets (for visualization)
# ============================================================

def esh_level_set(pot, H_esh_val, q_range, n_points=500):
    """Compute |v| = exp(H_ESH - U(q)) on a q grid for a given H_ESH level."""
    qs = np.linspace(q_range[0], q_range[1], n_points)
    vs_pos = []
    vs_neg = []
    valid_qs = []

    for q_val in qs:
        U_val = pot.energy(np.array([q_val]))
        log_v = H_esh_val - U_val
        if log_v > -10:  # |v| = exp(log_v) > 0
            v_val = np.exp(log_v)
            valid_qs.append(q_val)
            vs_pos.append(v_val)
            vs_neg.append(-v_val)

    return np.array(valid_qs), np.array(vs_pos), np.array(vs_neg)


# ============================================================
# Main phase space analysis
# ============================================================

def main():
    print("=" * 60)
    print("PHASE SPACE ANALYSIS: ESH vs ESH+THERMOSTAT")
    print("=" * 60)

    pot = HarmonicOscillator1D(omega=1.0)

    # ---- Pure ESH trajectories from different initial conditions ----
    print("\n[1] Pure ESH trajectories (conserved H_ESH level sets)")
    dt_esh = 0.05
    n_steps_esh = 2000

    # Multiple starting points at different H_ESH values
    initial_conditions = [
        (1.0, 1.0),    # H_ESH = 0.5 + log(1) = 0.5
        (1.5, 0.5),    # H_ESH = 1.125 + log(0.5) ≈ 0.432
        (0.5, 2.0),    # H_ESH = 0.125 + log(2) ≈ 0.818
        (2.0, 0.3),    # H_ESH = 2.0 + log(0.3) ≈ 0.799
        (0.1, 3.0),    # H_ESH = 0.005 + log(3) ≈ 1.104
    ]

    esh_trajs = []
    for q0, p0 in initial_conditions:
        qs, ps, H_eshs = run_pure_esh(pot, q0, p0, dt_esh, n_steps_esh)
        H_val = H_eshs[0]
        print(f"  Start (q={q0}, p={p0}): H_ESH={H_val:.3f}, "
              f"H_std={np.std(H_eshs):.5f} (should be ~0)")
        esh_trajs.append((qs, ps, H_eshs, H_val))

    # ---- ESH + Thermostat trajectory ----
    print("\n[2] ESH + Thermostat trajectory (should fill phase space)")
    dyn = ESHPlusThermostat(dim=1, kT=KT, Qs=[0.1, 0.7, 10.0],
                             L_esh=10, M_thermo=100,
                             dt_esh=0.05, dt_thermo=0.01)
    integrator = ESHHybridIntegrator(dyn, pot, dt=0.01)

    q0_hybrid = np.array([1.0])
    state = dyn.initial_state(q0_hybrid, rng=np.random.default_rng(42))

    n_cycles = 300
    qs_hybrid, ps_hybrid, H_eshs_hybrid = [], [], []

    for cycle in range(n_cycles):
        state = integrator.step(state)
        if np.any(np.isnan(state.q)) or np.any(np.isnan(state.p)):
            print(f"  NaN at cycle {cycle}!")
            break
        qs_hybrid.append(state.q[0])
        ps_hybrid.append(state.p[0])
        H_eshs_hybrid.append(pot.energy(state.q) + np.log(abs(state.p[0]) + 1e-300))

    qs_hybrid = np.array(qs_hybrid)
    ps_hybrid = np.array(ps_hybrid)
    H_eshs_hybrid = np.array(H_eshs_hybrid)

    print(f"  Hybrid: {len(qs_hybrid)} cycles, {state.n_force_evals} force evals")
    print(f"  q: mean={np.mean(qs_hybrid):.3f}, std={np.std(qs_hybrid):.3f} (target ~1.0)")
    print(f"  p: mean={np.mean(ps_hybrid):.3f}, std={np.std(ps_hybrid):.3f} (target ~1.0)")
    print(f"  H_ESH: mean={np.mean(H_eshs_hybrid):.3f}, std={np.std(H_eshs_hybrid):.3f} (should vary)")

    # ---- Also run pure thermostat for comparison ----
    print("\n[3] Pure thermostat (NHCTail) for comparison")
    sys.path.insert(0, str(Path(WORKTREE) / 'orbits/multiscale-chain-009'))
    from solution import MultiScaleNHCTail, MultiScaleNHCTailVerlet

    dyn_thermo = MultiScaleNHCTail(dim=1, kT=KT, Qs=[0.1, 0.7, 10.0], chain_length=2)
    integrator_thermo = MultiScaleNHCTailVerlet(dyn_thermo, pot, dt=0.01, kT=KT)
    state_thermo = dyn_thermo.initial_state(np.array([1.0]), rng=np.random.default_rng(42))

    qs_thermo, ps_thermo = [], []
    for _ in range(30000):  # same approx force evals as hybrid
        state_thermo = integrator_thermo.step(state_thermo)
        if np.any(np.isnan(state_thermo.q)):
            break
        qs_thermo.append(state_thermo.q[0])
        ps_thermo.append(state_thermo.p[0])

    qs_thermo = np.array(qs_thermo)
    ps_thermo = np.array(ps_thermo)
    H_eshs_thermo = np.array([
        pot.energy(np.array([qs_thermo[i]])) + np.log(abs(ps_thermo[i]) + 1e-300)
        for i in range(len(qs_thermo))
    ])
    print(f"  Thermostat: {len(qs_thermo)} steps, {state_thermo.n_force_evals} force evals")

    # ---- Plot ----
    print("\n[4] Generating phase space plots...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Phase Space Analysis: ESH vs ESH+Thermostat (1D HO)",
                 fontsize=14, fontweight='bold')

    colors_esh = plt.cm.tab10(np.linspace(0, 0.5, len(esh_trajs)))
    q_range = (-4, 4)

    # ---- Panel 1: Pure ESH trajectories ----
    ax1 = axes[0, 0]
    for i, (qs, ps, H_eshs, H_val) in enumerate(esh_trajs):
        ax1.plot(qs, ps, '-', color=colors_esh[i], alpha=0.7, linewidth=1.0,
                 label=f'H_ESH={H_val:.2f}')
    ax1.set_xlabel('q')
    ax1.set_ylabel('v')
    ax1.set_title('Pure ESH (conserved level sets)')
    ax1.legend(fontsize=7)
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-5, 5)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='k', linewidth=0.5)
    ax1.axvline(0, color='k', linewidth=0.5)

    # ---- Panel 2: ESH level sets (theoretical) ----
    ax2 = axes[0, 1]
    H_vals = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    for H_val in H_vals:
        qs_ls, vs_pos, vs_neg = esh_level_set(pot, H_val, q_range)
        if len(qs_ls) > 0:
            ax2.plot(qs_ls, vs_pos, 'b-', alpha=0.6, linewidth=1.5)
            ax2.plot(qs_ls, vs_neg, 'b-', alpha=0.6, linewidth=1.5)
            ax2.annotate(f'H={H_val:.1f}',
                         xy=(qs_ls[-1], vs_pos[-1]),
                         fontsize=7, color='blue', alpha=0.7)
    ax2.set_xlabel('q')
    ax2.set_ylabel('v')
    ax2.set_title('ESH Level Sets (H_ESH = U(q) + log|v|)')
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-8, 8)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='k', linewidth=0.5)

    # ---- Panel 3: ESH Hybrid phase space ----
    ax3 = axes[0, 2]
    # Color by cycle index to show time evolution
    scatter = ax3.scatter(qs_hybrid, ps_hybrid,
                          c=np.arange(len(qs_hybrid)), cmap='viridis',
                          s=8, alpha=0.6)
    plt.colorbar(scatter, ax=ax3, label='Cycle index')
    ax3.set_xlabel('q')
    ax3.set_ylabel('v')
    ax3.set_title(f'ESH Hybrid (L=10 ESH + 100 thermo)\n{len(qs_hybrid)} cycles')
    ax3.set_xlim(-5, 5)
    ax3.set_ylim(-5, 5)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color='k', linewidth=0.5)
    ax3.axvline(0, color='k', linewidth=0.5)

    # ---- Panel 4: H_ESH over time ----
    ax4 = axes[1, 0]
    # Pure ESH: H_ESH should be conserved
    for i, (qs, ps, H_eshs, H_val) in enumerate(esh_trajs):
        ax4.plot(H_eshs, '-', color=colors_esh[i], alpha=0.7, linewidth=0.8)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('H_ESH = U(q) + log|v|')
    ax4.set_title('Pure ESH: H_ESH Conservation')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-3, 4)

    # ---- Panel 5: H_ESH for hybrid (should jump) ----
    ax5 = axes[1, 1]
    ax5.plot(H_eshs_hybrid, 'b-', alpha=0.7, linewidth=0.8, label='ESH hybrid')
    ax5.axhline(np.mean(H_eshs_hybrid), color='r', linestyle='--',
                linewidth=1.5, label=f'Mean={np.mean(H_eshs_hybrid):.2f}')
    ax5.set_xlabel('Cycle')
    ax5.set_ylabel('H_ESH = U(q) + log|v|')
    ax5.set_title('ESH Hybrid: H_ESH diffuses across level sets')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # ---- Panel 6: Marginal distribution comparison ----
    ax6 = axes[1, 2]
    q_grid = np.linspace(-5, 5, 200)
    p_true = np.exp(-0.5 * q_grid**2 / KT)
    p_true /= np.trapezoid(p_true, q_grid)

    # Hybrid histogram
    if len(qs_hybrid) > 50:
        ax6.hist(qs_hybrid, bins=40, density=True, alpha=0.6,
                 color='blue', label='ESH hybrid')
    # Pure thermostat histogram
    if len(qs_thermo) > 50:
        ax6.hist(qs_thermo, bins=40, density=True, alpha=0.4,
                 color='green', label='NHCTail')
    ax6.plot(q_grid, p_true, 'r-', linewidth=2, label='Target N(0,1)')
    ax6.set_xlabel('q')
    ax6.set_ylabel('Density')
    ax6.set_title('Position Marginal: Hybrid vs Target')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(-5, 5)

    plt.tight_layout()
    fig_path = OUT_DIR / 'phase_space_analysis.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # ---- Additional plot: trajectory comparison ----
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle("ESH Dynamics: Level Set Hopping Mechanism", fontsize=13)

    # Left: pure ESH stays on level set
    ax_left = axes2[0]
    qs0, ps0, H0, H_val0 = esh_trajs[0]
    ax_left.plot(qs0, ps0, 'b-', linewidth=1.5, alpha=0.8)
    ax_left.plot(qs0[0], ps0[0], 'go', markersize=10, label='Start')
    # Overlay the theoretical level set
    qs_ls, vs_pos, vs_neg = esh_level_set(pot, H_val0, (-4, 4))
    ax_left.plot(qs_ls, vs_pos, 'r--', linewidth=2, label=f'H_ESH={H_val0:.2f} (theory)')
    ax_left.plot(qs_ls, vs_neg, 'r--', linewidth=2)
    ax_left.set_xlabel('q', fontsize=12)
    ax_left.set_ylabel('v', fontsize=12)
    ax_left.set_title('Pure ESH: Trapped on H_ESH level set')
    ax_left.legend()
    ax_left.set_xlim(-4, 4)
    ax_left.set_ylim(-5, 5)
    ax_left.grid(True, alpha=0.3)

    # Right: hybrid hops between level sets
    ax_right = axes2[1]
    # Color segments by H_ESH value
    n_pts = len(qs_hybrid)
    H_min, H_max = H_eshs_hybrid.min(), H_eshs_hybrid.max()
    norm_H = (H_eshs_hybrid - H_min) / max(H_max - H_min, 1e-10)
    cmap = cm.coolwarm

    for i in range(n_pts - 1):
        color = cmap(norm_H[i])
        ax_right.plot([qs_hybrid[i], qs_hybrid[i+1]],
                      [ps_hybrid[i], ps_hybrid[i+1]],
                      '-', color=color, alpha=0.5, linewidth=1.0)

    # Overlay a few level sets as reference
    for H_val in [-0.5, 0.5, 1.5]:
        qs_ls, vs_pos, vs_neg = esh_level_set(pot, H_val, (-5, 5))
        if len(qs_ls) > 0:
            ax_right.plot(qs_ls, vs_pos, 'k--', alpha=0.3, linewidth=1.0)
            ax_right.plot(qs_ls, vs_neg, 'k--', alpha=0.3, linewidth=1.0)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=H_min, vmax=H_max))
    sm.set_array([])
    plt.colorbar(sm, ax=ax_right, label='H_ESH value')
    ax_right.set_xlabel('q', fontsize=12)
    ax_right.set_ylabel('v', fontsize=12)
    ax_right.set_title('ESH+Thermostat: Hops between level sets')
    ax_right.set_xlim(-5, 5)
    ax_right.set_ylim(-5, 5)
    ax_right.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2_path = OUT_DIR / 'level_set_hopping.png'
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig2_path}")

    # ---- Save summary stats ----
    summary = {
        'pure_esh': {
            'n_trajs': len(esh_trajs),
            'H_conservation': [float(np.std(t[2])) for t in esh_trajs],
            'H_values': [float(t[3]) for t in esh_trajs],
        },
        'esh_hybrid': {
            'n_cycles': len(qs_hybrid),
            'n_force_evals': int(state.n_force_evals),
            'q_mean': float(np.mean(qs_hybrid)),
            'q_std': float(np.std(qs_hybrid)),
            'p_mean': float(np.mean(ps_hybrid)),
            'p_std': float(np.std(ps_hybrid)),
            'H_esh_mean': float(np.mean(H_eshs_hybrid)),
            'H_esh_std': float(np.std(H_eshs_hybrid)),
        },
    }

    json_path = OUT_DIR / 'phase_space_stats.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Stats saved to {json_path}")

    print("\nPhase space analysis complete.")
    return summary


if __name__ == "__main__":
    main()
