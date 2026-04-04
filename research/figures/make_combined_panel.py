"""Combined panel figure: all samplers, all benchmarks, all diagnostics.

Layout (6 rows x 3 cols):
  Row A: Phase space (1D HO) — NH | Log-Osc | NHC | Dual-Bath | SinhDrive-NHC
  Row B: 2D Double-Well density — True | NH | Log-Osc (winner)
  Row C: Barrier crossing trajectories — NH vs Log-Osc
  Row D: Stage 2 — GMM density | Rosenbrock density (Log-Osc)
  Row E: Nonlinear friction comparison — ergodicity bar chart | friction functions plot
  Row F: Summary leaderboard + metrics bar charts
"""

import sys, importlib.util, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

ROOT = Path("/Users/wujiewang/code/det-sampler")
sys.path.insert(0, str(ROOT))

from research.eval.potentials import (HarmonicOscillator1D, DoubleWell2D,
                                       GaussianMixture2D, Rosenbrock2D)
from research.eval.baselines import NoseHoover, NoseHooverChain
from research.eval.integrators import VelocityVerletThermostat

# Load log-osc
spec = importlib.util.spec_from_file_location('losc', str(ROOT / '.worktrees/log-osc-001/orbits/log-osc-001/solution.py'))
losc_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(losc_mod)
LogOscThermostat = losc_mod.LogOscThermostat
LogOscVelocityVerlet = losc_mod.LogOscVelocityVerlet

# Load nonlinear friction results
with open(str(ROOT / '.worktrees/general-nonlinear-004/orbits/general-nonlinear-004/results.json')) as f:
    nl_results = json.load(f)

FIGDIR = ROOT / "research/figures"
kT = 1.0; mass = 1.0

NH_COLOR = '#1f77b4'
NHC_COLOR = '#ff7f0e'
LOSC_COLOR = '#2ca02c'
DUAL_COLOR = '#d62728'
ESH_COLOR = '#9467bd'

def collect(dynamics, potential, dt, n_steps, thin=1, integrator_cls=VelocityVerletThermostat):
    rng = np.random.default_rng(42)
    q0 = rng.normal(0, 0.5, size=potential.dim)
    state = dynamics.initial_state(q0, rng=rng)
    integrator = integrator_cls(dynamics, potential, dt, kT=kT, mass=mass)
    qs, ps, energies = [], [], []
    for i in range(n_steps):
        state = integrator.step(state)
        if i % thin == 0:
            qs.append(state.q.copy())
            ps.append(state.p.copy())
            energies.append(0.5*np.sum(state.p**2)/mass + potential.energy(state.q))
    return np.array(qs), np.array(ps), np.array(energies)

def true_density_2d(potential, xr, yr, n=200):
    x = np.linspace(*xr, n); y = np.linspace(*yr, n)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[np.exp(-potential.energy(np.array([X[i,j], Y[i,j]]))/kT) for j in range(n)] for i in range(n)])
    Z /= np.sum(Z)*(x[1]-x[0])*(y[1]-y[0])
    return X, Y, Z

print("Collecting trajectories...")
pot_ho = HarmonicOscillator1D()
pot_dw = DoubleWell2D()
gmm = GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5)
ros = Rosenbrock2D(a=0.0, b=5.0)

N = 500_000; thin = 10

nh1 = NoseHoover(dim=1, kT=1.0, Q=1.0)
nhc1 = NoseHooverChain(dim=1, chain_length=3, kT=1.0, Q=1.0)
losc1 = LogOscThermostat(dim=1, Q=0.8)

print("  HO: NH, NHC, Log-Osc...")
qs_nh1, ps_nh1, _ = collect(nh1, pot_ho, 0.01, N, thin)
qs_nhc1, ps_nhc1, _ = collect(nhc1, pot_ho, 0.01, N, thin)
qs_losc1, ps_losc1, _ = collect(losc1, pot_ho, 0.01, N, thin, LogOscVelocityVerlet)

nh2 = NoseHoover(dim=2, kT=1.0, Q=1.0)
losc2 = LogOscThermostat(dim=2, Q=1.0)

print("  DW: NH, Log-Osc...")
qs_nh2, ps_nh2, _ = collect(nh2, pot_dw, 0.01, N, thin//2)
qs_losc2, ps_losc2, _ = collect(losc2, pot_dw, 0.035, N, thin//2, LogOscVelocityVerlet)

print("  Stage 2: GMM, Rosenbrock...")
losc_gmm = LogOscThermostat(dim=2, Q=0.5)
losc_ros = LogOscThermostat(dim=2, Q=2.0)
qs_gmm, _, _ = collect(losc_gmm, gmm, 0.01, N, thin//2, LogOscVelocityVerlet)
qs_ros, _, _ = collect(losc_ros, ros, 0.01, N, thin//2, LogOscVelocityVerlet)

print("Computing true densities...")
X_dw, Y_dw, Z_dw = true_density_2d(pot_dw, (-2.5,2.5), (-3,3), 150)
X_gmm, Y_gmm, Z_gmm = true_density_2d(gmm, (-5,5), (-5,5), 150)
X_ros, Y_ros, Z_ros = true_density_2d(ros, (-3,3), (-1,5), 150)

# ============================================================
# BUILD FIGURE
# ============================================================
print("Building figure...")
fig = plt.figure(figsize=(20, 32))
gs = GridSpec(6, 3, figure=fig, hspace=0.35, wspace=0.3,
             height_ratios=[1, 1, 0.7, 1, 0.8, 0.8])

theta = np.linspace(0, 2*np.pi, 200)
burnin = len(qs_nh2)//10

# === Row A: Phase Space 1D HO ===
for idx, (qs, ps, color, title) in enumerate([
    (qs_nh1, ps_nh1, NH_COLOR, 'Nosé-Hoover'),
    (qs_losc1, ps_losc1, LOSC_COLOR, 'Log-Osc (Q=0.8)'),
    (qs_nhc1, ps_nhc1, NHC_COLOR, 'NHC (M=3)'),
]):
    ax = fig.add_subplot(gs[0, idx])
    ax.scatter(qs[:,0], ps[:,0], s=0.2, alpha=0.2, c=color, rasterized=True)
    for s in [1,2,3]:
        ax.plot(s*np.cos(theta), s*np.sin(theta), 'k--', alpha=0.2, lw=0.8)
    ax.set_xlim(-4,4); ax.set_ylim(-4,4); ax.set_aspect('equal')
    ax.set_xlabel('q'); ax.set_ylabel('p')
    ax.set_title(title, fontsize=13)
fig.text(0.02, 0.935, 'A. Phase Space Coverage (1D Harmonic Oscillator)', fontsize=15, fontweight='bold')

# === Row B: 2D Double-Well Density ===
dw_range = [[-2.5,2.5],[-3,3]]

ax = fig.add_subplot(gs[1,0])
ax.contourf(X_dw, Y_dw, Z_dw, levels=30, cmap='viridis')
ax.set_title('True Boltzmann', fontsize=13); ax.set_aspect('equal')
ax.set_xlabel('x'); ax.set_ylabel('y')

ax = fig.add_subplot(gs[1,1])
ax.hist2d(qs_nh2[burnin:,0], qs_nh2[burnin:,1], bins=80, range=dw_range, cmap='viridis', density=True)
ax.set_title('Nosé-Hoover', fontsize=13); ax.set_aspect('equal')
ax.set_xlabel('x'); ax.set_ylabel('y')

ax = fig.add_subplot(gs[1,2])
ax.hist2d(qs_losc2[burnin:,0], qs_losc2[burnin:,1], bins=80, range=dw_range, cmap='viridis', density=True)
ax.set_title('Log-Osc (Q=1.0, dt=0.035)', fontsize=13); ax.set_aspect('equal')
ax.set_xlabel('x'); ax.set_ylabel('y')

fig.text(0.02, 0.77, 'B. Position Density (2D Double-Well)', fontsize=15, fontweight='bold')

# === Row C: Barrier Crossing Trajectory ===
n_show = 30000
t = np.arange(n_show) * 0.01 * (thin//2)

ax = fig.add_subplot(gs[2, :2])
ax.plot(t, qs_nh2[:n_show,0], lw=0.3, color=NH_COLOR, alpha=0.7, label='NH')
ax.plot(t, qs_losc2[:n_show,0], lw=0.3, color=LOSC_COLOR, alpha=0.7, label='Log-Osc')
ax.axhline(1, color='gray', ls='--', alpha=0.3); ax.axhline(-1, color='gray', ls='--', alpha=0.3)
ax.axhline(0, color='red', ls=':', alpha=0.3)
ax.set_ylabel('x(t)'); ax.set_xlabel('Time'); ax.set_ylim(-3,3)
ax.set_title('Double-Well: Barrier Crossing', fontsize=13)
ax.legend(fontsize=10, loc='upper right')

# Marginals comparison
ax = fig.add_subplot(gs[2, 2])
x_range = np.linspace(-4,4,200)
gaussian = np.exp(-x_range**2/2)/np.sqrt(2*np.pi)
ax.hist(qs_nh1[:,0], bins=60, density=True, alpha=0.4, color=NH_COLOR, label='NH')
ax.hist(qs_losc1[:,0], bins=60, density=True, alpha=0.4, color=LOSC_COLOR, label='Log-Osc')
ax.hist(qs_nhc1[:,0], bins=60, density=True, alpha=0.4, color=NHC_COLOR, label='NHC')
ax.plot(x_range, gaussian, 'k--', lw=2, label='N(0,1)')
ax.set_xlabel('q'); ax.set_ylabel('Density')
ax.set_title('HO Position Marginal', fontsize=13)
ax.legend(fontsize=9)

fig.text(0.02, 0.615, 'C. Trajectories & Marginals', fontsize=15, fontweight='bold')

# === Row D: Stage 2 Benchmarks ===
ax = fig.add_subplot(gs[3,0])
ax.contourf(X_gmm, Y_gmm, Z_gmm, levels=30, cmap='viridis')
ax.set_title('GMM True Density', fontsize=13); ax.set_aspect('equal')
ax.set_xlabel('x'); ax.set_ylabel('y')

ax = fig.add_subplot(gs[3,1])
ax.hist2d(qs_gmm[burnin:,0], qs_gmm[burnin:,1], bins=80, range=[[-5,5],[-5,5]], cmap='viridis', density=True)
ax.set_title('Log-Osc on GMM (Q=0.5)', fontsize=13); ax.set_aspect('equal')
ax.set_xlabel('x'); ax.set_ylabel('y')

ax = fig.add_subplot(gs[3,2])
ax.contourf(X_ros, Y_ros, Z_ros, levels=30, cmap='viridis')
ax.contour(X_ros, Y_ros, Z_ros, levels=10, colors='white', alpha=0.3, linewidths=0.5)
# Overlay Log-Osc samples
ax.scatter(qs_ros[burnin::5,0], qs_ros[burnin::5,1], s=0.3, alpha=0.3, c='red', rasterized=True)
ax.set_title('Rosenbrock: True + Log-Osc', fontsize=13); ax.set_aspect('equal')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_xlim(-3,3); ax.set_ylim(-1,5)

fig.text(0.02, 0.455, 'D. Stage 2 Benchmarks (Gaussian Mixture & Rosenbrock)', fontsize=15, fontweight='bold')

# === Row E: Nonlinear Friction Study ===
names = list(nl_results.keys())
erg_vals = [nl_results[n]['ho_erg'] for n in names]
dw_kl_vals = [nl_results[n]['dw_kl'] for n in names]
short_names = [n.replace('gaussian_damped_', 'gauss_').replace('standard_nh', 'NH (linear)').replace('_', ' ') for n in names]

ax = fig.add_subplot(gs[4, :2])
colors_bar = [LOSC_COLOR if 'log' in n else NH_COLOR if 'standard' in n else '#888888' for n in names]
bars = ax.bar(range(len(names)), erg_vals, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Ergodicity Score', fontsize=12)
ax.set_title('Nonlinear Friction: Ergodicity on 1D HO (Q=1.0)', fontsize=13)
ax.axhline(0.85, color='red', ls='--', alpha=0.5, label='Ergodic threshold')
ax.legend(fontsize=9)

# Friction function shapes
ax = fig.add_subplot(gs[4, 2])
xi = np.linspace(-5, 5, 300)
g_funcs = {
    'Log-osc': 2*xi/(1+xi**2),
    'Tanh(1)': np.tanh(xi),
    'Arctan': (2/np.pi)*np.arctan(xi),
    'Soft-clip': xi/np.sqrt(1+xi**2),
    'NH (linear)': xi,
}
for label, g_vals in g_funcs.items():
    color = LOSC_COLOR if 'Log' in label else NH_COLOR if 'NH' in label else '#888888'
    ls = '-' if 'Log' in label or 'NH' in label else '--'
    lw = 2 if 'Log' in label or 'NH' in label else 1
    ax.plot(xi, g_vals, color=color, ls=ls, lw=lw, label=label, alpha=0.8)
ax.set_xlabel('xi', fontsize=12); ax.set_ylabel('g(xi)', fontsize=12)
ax.set_title('Friction Function Shapes', fontsize=13)
ax.legend(fontsize=9); ax.set_ylim(-3,3)
ax.axhline(0, color='gray', alpha=0.3)

fig.text(0.02, 0.29, 'E. Systematic Nonlinear Friction Study', fontsize=15, fontweight='bold')

# === Row F: Summary Leaderboard ===
# Bar chart: ergodicity + KL for all samplers
samplers = ['NH', 'NHC(M=3)', 'Log-Osc', 'Dual-Bath', 'SD-NHC']
ho_erg = [0.54, 0.92, 0.944, 0.927, 0.949]
ho_kl = [0.077, 0.002, 0.023, 0.002, 0.001]
dw_kl = [0.037, 0.029, 0.010, 0.030, 0.029]
s_colors = [NH_COLOR, NHC_COLOR, LOSC_COLOR, DUAL_COLOR, ESH_COLOR]

ax = fig.add_subplot(gs[5, 0])
x = np.arange(len(samplers))
ax.bar(x, ho_erg, color=s_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.axhline(0.85, color='red', ls='--', alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels(samplers, rotation=30, ha='right', fontsize=10)
ax.set_ylabel('Ergodicity', fontsize=12); ax.set_title('1D HO Ergodicity', fontsize=13)
ax.set_ylim(0, 1.05)

ax = fig.add_subplot(gs[5, 1])
ax.bar(x, ho_kl, color=s_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.axhline(0.01, color='red', ls='--', alpha=0.5, label='Target')
ax.set_xticks(x); ax.set_xticklabels(samplers, rotation=30, ha='right', fontsize=10)
ax.set_ylabel('KL Divergence', fontsize=12); ax.set_title('1D HO KL', fontsize=13)
ax.legend(fontsize=9)

ax = fig.add_subplot(gs[5, 2])
ax.bar(x, dw_kl, color=s_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.axhline(0.01, color='red', ls='--', alpha=0.5, label='Target')
ax.set_xticks(x); ax.set_xticklabels(samplers, rotation=30, ha='right', fontsize=10)
ax.set_ylabel('KL Divergence', fontsize=12); ax.set_title('2D Double-Well KL', fontsize=13)
ax.legend(fontsize=9)

fig.text(0.02, 0.13, 'F. Leaderboard: All Samplers Compared', fontsize=15, fontweight='bold')

plt.savefig(str(FIGDIR / 'full_research_panel.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("-> full_research_panel.png")
print("Done!")
