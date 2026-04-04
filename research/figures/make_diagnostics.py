"""Generate visual diagnostic plots for baseline samplers.

Produces a single combined figure with 5 rows:
1. Phase space trajectories (1D HO): NH vs NHC vs true contours
2. 2D density comparison (double-well): true vs NH vs NHC
3. Trajectory traces (barrier crossing): NH and NHC x(t)
4. Energy distributions: double-well and 1D HO
5. Marginal distributions (1D HO): P(q) and P(p)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
sys.path.insert(0, "/Users/wujiewang/code/det-sampler")

from research.eval.potentials import HarmonicOscillator1D, DoubleWell2D
from research.eval.baselines import NoseHoover, NoseHooverChain
from research.eval.integrators import VelocityVerletThermostat, ThermostatState

FIGDIR = "/Users/wujiewang/code/det-sampler/research/figures"
kT = 1.0
mass = 1.0

NH_COLOR = '#1f77b4'
NHC_COLOR = '#ff7f0e'


def collect_trajectory(dynamics, potential, dt, n_steps, thin=1):
    """Run sampler and collect full trajectory."""
    rng = np.random.default_rng(42)
    q0 = rng.normal(0, 0.5, size=potential.dim)
    state = dynamics.initial_state(q0, rng=rng)
    integrator = VelocityVerletThermostat(dynamics, potential, dt, kT=kT, mass=mass)

    qs, ps, xis, energies = [], [], [], []
    for i in range(n_steps):
        state = integrator.step(state)
        if i % thin == 0:
            qs.append(state.q.copy())
            ps.append(state.p.copy())
            xis.append(state.xi.copy())
            e_kin = 0.5 * np.sum(state.p**2) / mass
            e_pot = potential.energy(state.q)
            energies.append(e_kin + e_pot)

    return np.array(qs), np.array(ps), np.array(xis), np.array(energies)


def true_density_2d(potential, xrange, yrange, n_grid=200, kT=1.0):
    """Compute true Boltzmann density on a grid."""
    x = np.linspace(*xrange, n_grid)
    y = np.linspace(*yrange, n_grid)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(n_grid):
        for j in range(n_grid):
            Z[i, j] = -potential.energy(np.array([X[i, j], Y[i, j]])) / kT
    Z -= np.max(Z)
    Z = np.exp(Z)
    Z /= np.sum(Z) * (x[1] - x[0]) * (y[1] - y[0])
    return X, Y, Z


# ============================================================
# Collect all data
# ============================================================
print("Collecting trajectories...")
pot_ho = HarmonicOscillator1D()
pot_dw = DoubleWell2D()

nh_1d = NoseHoover(dim=1, kT=1.0, Q=1.0)
nhc_1d = NoseHooverChain(dim=1, chain_length=3, kT=1.0, Q=1.0)
nh_2d = NoseHoover(dim=2, kT=1.0, Q=1.0)
nhc_2d = NoseHooverChain(dim=2, chain_length=3, kT=1.0, Q=1.0)

print("  1D HO: NH...")
qs_nh1, ps_nh1, _, en_nh1 = collect_trajectory(nh_1d, pot_ho, dt=0.01, n_steps=500_000, thin=10)
print("  1D HO: NHC...")
qs_nhc1, ps_nhc1, _, en_nhc1 = collect_trajectory(nhc_1d, pot_ho, dt=0.01, n_steps=500_000, thin=10)
print("  2D DW: NH...")
qs_nh2, ps_nh2, _, en_nh2 = collect_trajectory(nh_2d, pot_dw, dt=0.01, n_steps=500_000, thin=5)
print("  2D DW: NHC...")
qs_nhc2, ps_nhc2, _, en_nhc2 = collect_trajectory(nhc_2d, pot_dw, dt=0.01, n_steps=500_000, thin=5)

burnin = len(qs_nh2) // 10
X_true, Y_true, Z_true = true_density_2d(pot_dw, (-2.5, 2.5), (-3, 3), n_grid=200)

# ============================================================
# Build combined figure
# ============================================================
print("Building combined figure...")

fig = plt.figure(figsize=(18, 28))
gs = GridSpec(5, 3, figure=fig, hspace=0.35, wspace=0.3,
             height_ratios=[1, 1, 0.8, 0.8, 0.8])

# Row labels
row_labels = [
    'A. Phase Space Coverage (1D Harmonic Oscillator)',
    'B. Position Density (2D Double-Well)',
    'C. Barrier Crossing Trajectory (2D Double-Well)',
    'D. Energy Distributions',
    'E. Marginal Distributions (1D Harmonic Oscillator)',
]

# ============================================================
# Row A: Phase space (1D HO) — NH | NHC | overlay
# ============================================================
theta = np.linspace(0, 2 * np.pi, 200)

ax_a0 = fig.add_subplot(gs[0, 0])
ax_a0.scatter(qs_nh1[:, 0], ps_nh1[:, 0], s=0.2, alpha=0.2, c=NH_COLOR, rasterized=True)
for s in [1, 2, 3]:
    ax_a0.plot(s * np.cos(theta), s * np.sin(theta), 'k--', alpha=0.25, lw=0.8)
ax_a0.set_xlabel('q', fontsize=12)
ax_a0.set_ylabel('p', fontsize=12)
ax_a0.set_title('Nosé-Hoover', fontsize=13)
ax_a0.set_xlim(-4, 4); ax_a0.set_ylim(-4, 4); ax_a0.set_aspect('equal')

ax_a1 = fig.add_subplot(gs[0, 1])
ax_a1.scatter(qs_nhc1[:, 0], ps_nhc1[:, 0], s=0.2, alpha=0.2, c=NHC_COLOR, rasterized=True)
for s in [1, 2, 3]:
    ax_a1.plot(s * np.cos(theta), s * np.sin(theta), 'k--', alpha=0.25, lw=0.8)
ax_a1.set_xlabel('q', fontsize=12)
ax_a1.set_title('NHC (M=3)', fontsize=13)
ax_a1.set_xlim(-4, 4); ax_a1.set_ylim(-4, 4); ax_a1.set_aspect('equal')

ax_a2 = fig.add_subplot(gs[0, 2])
ax_a2.scatter(qs_nh1[::2, 0], ps_nh1[::2, 0], s=0.2, alpha=0.15, c=NH_COLOR, rasterized=True, label='NH')
ax_a2.scatter(qs_nhc1[::2, 0], ps_nhc1[::2, 0], s=0.2, alpha=0.15, c=NHC_COLOR, rasterized=True, label='NHC')
for s in [1, 2, 3]:
    ax_a2.plot(s * np.cos(theta), s * np.sin(theta), 'k--', alpha=0.25, lw=0.8)
ax_a2.set_xlabel('q', fontsize=12)
ax_a2.set_title('Overlay', fontsize=13)
ax_a2.set_xlim(-4, 4); ax_a2.set_ylim(-4, 4); ax_a2.set_aspect('equal')
ax_a2.legend(fontsize=10, markerscale=10, loc='upper right')

fig.text(0.02, 0.92, row_labels[0], fontsize=15, fontweight='bold', va='center')

# ============================================================
# Row B: 2D density — True | NH | NHC
# ============================================================
dw_range = [[-2.5, 2.5], [-3, 3]]

ax_b0 = fig.add_subplot(gs[1, 0])
ax_b0.contourf(X_true, Y_true, Z_true, levels=30, cmap='viridis')
ax_b0.set_xlabel('x', fontsize=12); ax_b0.set_ylabel('y', fontsize=12)
ax_b0.set_title('True Boltzmann', fontsize=13)
ax_b0.set_aspect('equal')

ax_b1 = fig.add_subplot(gs[1, 1])
ax_b1.hist2d(qs_nh2[burnin:, 0], qs_nh2[burnin:, 1], bins=80,
             range=dw_range, cmap='viridis', density=True)
ax_b1.set_xlabel('x', fontsize=12); ax_b1.set_ylabel('y', fontsize=12)
ax_b1.set_title('Nosé-Hoover', fontsize=13)
ax_b1.set_aspect('equal')

ax_b2 = fig.add_subplot(gs[1, 2])
ax_b2.hist2d(qs_nhc2[burnin:, 0], qs_nhc2[burnin:, 1], bins=80,
             range=dw_range, cmap='viridis', density=True)
ax_b2.set_xlabel('x', fontsize=12); ax_b2.set_ylabel('y', fontsize=12)
ax_b2.set_title('NHC (M=3)', fontsize=13)
ax_b2.set_aspect('equal')

fig.text(0.02, 0.735, row_labels[1], fontsize=15, fontweight='bold', va='center')

# ============================================================
# Row C: Barrier crossing trajectory — NH on top, NHC on bottom
# ============================================================
n_show = 50000
t = np.arange(n_show) * 0.01 * 5

ax_c0 = fig.add_subplot(gs[2, :2])
ax_c0.plot(t, qs_nh2[:n_show, 0], linewidth=0.3, color=NH_COLOR, alpha=0.7)
ax_c0.axhline(y=1.0, color='gray', ls='--', alpha=0.4)
ax_c0.axhline(y=-1.0, color='gray', ls='--', alpha=0.4)
ax_c0.axhline(y=0.0, color='red', ls=':', alpha=0.4)
ax_c0.set_ylabel('x(t)', fontsize=12)
ax_c0.set_title('Nosé-Hoover: x-trajectory', fontsize=13)
ax_c0.set_ylim(-3, 3)
ax_c0.set_xlabel('Time', fontsize=12)

ax_c1 = fig.add_subplot(gs[2, 2])
ax_c1.plot(t, qs_nhc2[:n_show, 0], linewidth=0.3, color=NHC_COLOR, alpha=0.7)
ax_c1.axhline(y=1.0, color='gray', ls='--', alpha=0.4)
ax_c1.axhline(y=-1.0, color='gray', ls='--', alpha=0.4)
ax_c1.axhline(y=0.0, color='red', ls=':', alpha=0.4)
ax_c1.set_ylabel('x(t)', fontsize=12)
ax_c1.set_title('NHC (M=3): x-trajectory', fontsize=13)
ax_c1.set_ylim(-3, 3)
ax_c1.set_xlabel('Time', fontsize=12)

fig.text(0.02, 0.555, row_labels[2], fontsize=15, fontweight='bold', va='center')

# ============================================================
# Row D: Energy distributions — double-well | 1D HO
# ============================================================
ax_d0 = fig.add_subplot(gs[3, 0])
ax_d0.hist(en_nh2[burnin:], bins=80, density=True, alpha=0.6, color=NH_COLOR, label='NH')
ax_d0.hist(en_nhc2[burnin:], bins=80, density=True, alpha=0.6, color=NHC_COLOR, label='NHC')
ax_d0.set_xlabel('H(q,p)', fontsize=12)
ax_d0.set_ylabel('Density', fontsize=12)
ax_d0.set_title('Double-Well Energy', fontsize=13)
ax_d0.legend(fontsize=10)

ax_d1 = fig.add_subplot(gs[3, 1])
burnin1 = len(en_nh1) // 10
ax_d1.hist(en_nh1[burnin1:], bins=80, density=True, alpha=0.6, color=NH_COLOR, label='NH')
ax_d1.hist(en_nhc1[burnin1:], bins=80, density=True, alpha=0.6, color=NHC_COLOR, label='NHC')
H_range = np.linspace(0, 8, 200)
P_H = np.exp(-H_range / kT)
P_H /= np.trapezoid(P_H, H_range)
ax_d1.plot(H_range, P_H, 'k--', lw=2, label='Analytical')
ax_d1.set_xlabel('H(q,p)', fontsize=12)
ax_d1.set_title('1D HO Energy', fontsize=13)
ax_d1.legend(fontsize=10)
ax_d1.set_xlim(0, 8)

# Summary text panel
ax_d2 = fig.add_subplot(gs[3, 2])
ax_d2.axis('off')
summary = (
    "Baseline Summary (10⁶ force evals)\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "                    NH        NHC(M=3)\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "DW KL:          0.037      0.029\n"
    "HO KL:          0.077      0.002\n"
    "ESS/force:   0.0031    0.0026\n"
    "Ergodicity:   0.54        0.92\n"
    "                  (FAIL)     (PASS)\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "Target: beat NHC on all metrics\n"
    "with novel thermostat designs"
)
ax_d2.text(0.05, 0.95, summary, transform=ax_d2.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

fig.text(0.02, 0.38, row_labels[3], fontsize=15, fontweight='bold', va='center')

# ============================================================
# Row E: Marginals (1D HO) — P(q) | P(p) | joint q-p correlation
# ============================================================
x_range = np.linspace(-4, 4, 200)
gaussian = np.exp(-x_range**2 / 2) / np.sqrt(2 * np.pi)

ax_e0 = fig.add_subplot(gs[4, 0])
ax_e0.hist(qs_nh1[:, 0], bins=80, density=True, alpha=0.5, color=NH_COLOR, label='NH')
ax_e0.hist(qs_nhc1[:, 0], bins=80, density=True, alpha=0.5, color=NHC_COLOR, label='NHC')
ax_e0.plot(x_range, gaussian, 'k--', lw=2, label='N(0,1)')
ax_e0.set_xlabel('q', fontsize=12)
ax_e0.set_ylabel('Density', fontsize=12)
ax_e0.set_title('Position P(q)', fontsize=13)
ax_e0.legend(fontsize=10)

ax_e1 = fig.add_subplot(gs[4, 1])
ax_e1.hist(ps_nh1[:, 0], bins=80, density=True, alpha=0.5, color=NH_COLOR, label='NH')
ax_e1.hist(ps_nhc1[:, 0], bins=80, density=True, alpha=0.5, color=NHC_COLOR, label='NHC')
ax_e1.plot(x_range, gaussian, 'k--', lw=2, label='N(0,1)')
ax_e1.set_xlabel('p', fontsize=12)
ax_e1.set_title('Momentum P(p)', fontsize=13)
ax_e1.legend(fontsize=10)

# q-p scatter (subsampled) showing correlation structure
ax_e2 = fig.add_subplot(gs[4, 2])
sub = slice(None, None, 5)
ax_e2.scatter(qs_nh1[sub, 0], ps_nh1[sub, 0], s=0.5, alpha=0.2, c=NH_COLOR, label='NH', rasterized=True)
ax_e2.scatter(qs_nhc1[sub, 0], ps_nhc1[sub, 0], s=0.5, alpha=0.2, c=NHC_COLOR, label='NHC', rasterized=True)
for s in [1, 2]:
    ax_e2.plot(s * np.cos(theta), s * np.sin(theta), 'k--', alpha=0.3, lw=0.8)
ax_e2.set_xlabel('q', fontsize=12)
ax_e2.set_ylabel('p', fontsize=12)
ax_e2.set_title('Joint (q, p)', fontsize=13)
ax_e2.set_xlim(-4, 4); ax_e2.set_ylim(-4, 4); ax_e2.set_aspect('equal')
ax_e2.legend(fontsize=10, markerscale=10, loc='upper right')

fig.text(0.02, 0.19, row_labels[4], fontsize=15, fontweight='bold', va='center')

# ============================================================
# Save
# ============================================================
plt.savefig(f'{FIGDIR}/baseline_diagnostics_combined.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("-> baseline_diagnostics_combined.png")
print("Done!")
