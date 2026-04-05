"""3D Phase Space Flow visualization: NHC vs Log-Osc on 1D HO.

Shows trajectories in (q, p, xi) space and (q, p) projections.
NHC trajectory is confined to thin tubes/tori; Log-Osc fills a volume.

2x2 panel: (a) NHC 3D, (b) Log-Osc 3D, (c) NHC (q,p), (d) Log-Osc (q,p)
"""

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from research.eval.potentials import HarmonicOscillator1D
from research.eval.baselines import NoseHooverChain
from research.eval.integrators import VelocityVerletThermostat, ThermostatState

# ---- Inline Log-Osc thermostat (avoids import issues with hyphenated dir) ----
def g_func(xi_val):
    return 2.0 * xi_val / (1.0 + xi_val**2)

class LogOscThermostat:
    name = "log_osc"
    def __init__(self, dim, kT=1.0, mass=1.0, Q=1.0):
        self.dim = dim; self.kT = kT; self.mass = mass; self.Q = Q
    def initial_state(self, q0, rng=None):
        if rng is None: rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        return ThermostatState(q0.copy(), p0, np.array([0.0]), 0)
    def dqdt(self, state, grad_U): return state.p / self.mass
    def dpdt(self, state, grad_U): return -grad_U - g_func(state.xi[0]) * state.p
    def dxidt(self, state, grad_U):
        K = np.sum(state.p**2) / self.mass
        return np.array([(K - self.dim * self.kT) / self.Q])

class LogOscVerlet:
    """Custom Velocity Verlet for Log-Osc (uses g(xi) for friction)."""
    def __init__(self, dynamics, potential, dt, kT=1.0, mass=1.0):
        self.dynamics = dynamics; self.potential = potential
        self.dt = dt; self.kT = kT; self.mass = mass; self._cached_grad_U = None
    def step(self, state):
        q, p, xi, n_evals = state
        dt = self.dt; half_dt = 0.5 * dt
        if self._cached_grad_U is not None:
            grad_U = self._cached_grad_U
        else:
            grad_U = self.potential.gradient(q); n_evals += 1
        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot
        scale = np.clip(np.exp(-g_func(xi[0]) * half_dt), 1e-10, 1e10)
        p = p * scale; p = p - half_dt * grad_U
        q = q + dt * p / self.mass
        if np.any(np.isnan(q)) or np.any(np.isnan(p)):
            self._cached_grad_U = None; return ThermostatState(q, p, xi, n_evals)
        grad_U = self.potential.gradient(q); n_evals += 1
        p = p - half_dt * grad_U
        scale = np.clip(np.exp(-g_func(xi[0]) * half_dt), 1e-10, 1e10)
        p = p * scale
        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot
        self._cached_grad_U = grad_U
        return ThermostatState(q, p, xi, n_evals)

# ---- Style ----
COLOR_NHC = '#ff7f0e'
COLOR_LOGOSC = '#2ca02c'
LABEL_SIZE = 14; TICK_SIZE = 12; TITLE_SIZE = 15

# ---- Simulation ----
potential = HarmonicOscillator1D(omega=1.0)
kT = 1.0; mass = 1.0; dt = 0.01; n_steps = 500_000; seed = 42

def run_nhc(Q=1.0, M=3):
    nhc = NoseHooverChain(dim=1, chain_length=M, kT=kT, mass=mass, Q=Q)
    rng = np.random.default_rng(seed)
    state = nhc.initial_state(np.array([0.5]), rng=rng)
    integ = VelocityVerletThermostat(nhc, potential, dt=dt, kT=kT, mass=mass)
    qs, ps, xis = [], [], []
    for i in range(n_steps):
        state = integ.step(state)
        if i % 10 == 0:
            qs.append(state.q[0]); ps.append(state.p[0]); xis.append(state.xi[0])
    return np.array(qs), np.array(ps), np.array(xis)

def run_logosc(Q=1.0):
    lo = LogOscThermostat(dim=1, kT=kT, mass=mass, Q=Q)
    rng = np.random.default_rng(seed)
    state = lo.initial_state(np.array([0.5]), rng=rng)
    integ = LogOscVerlet(lo, potential, dt=dt, kT=kT, mass=mass)
    qs, ps, xis = [], [], []
    for i in range(n_steps):
        state = integ.step(state)
        if i % 10 == 0:
            qs.append(state.q[0]); ps.append(state.p[0]); xis.append(state.xi[0])
    return np.array(qs), np.array(ps), np.array(xis)

print("Running NHC(M=3, Q=1.0)...")
q_nhc, p_nhc, xi_nhc = run_nhc(Q=1.0, M=3)
print(f"  NHC xi range: [{xi_nhc.min():.2f}, {xi_nhc.max():.2f}]")

print("Running Log-Osc(Q=1.0)...")
q_lo, p_lo, xi_lo = run_logosc(Q=1.0)
print(f"  Log-Osc xi range: [{xi_lo.min():.2f}, {xi_lo.max():.2f}]")

# ---- Plotting ----
fig = plt.figure(figsize=(14, 12))

n_3d = 50000  # subsample for 3D clarity
alpha_3d = 0.08; lw_3d = 0.3

# Gaussian contour helper
theta = np.linspace(0, 2*np.pi, 300)

# (a) NHC 3D
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.plot(q_nhc[:n_3d], p_nhc[:n_3d], xi_nhc[:n_3d],
         color=COLOR_NHC, alpha=alpha_3d, lw=lw_3d, rasterized=True)
ax1.set_xlabel('q', fontsize=LABEL_SIZE, labelpad=8)
ax1.set_ylabel('p', fontsize=LABEL_SIZE, labelpad=8)
ax1.set_zlabel(r'$\xi_1$', fontsize=LABEL_SIZE, labelpad=8)
ax1.set_title('(a) NHC (M=3)', fontsize=TITLE_SIZE, pad=10)
ax1.view_init(elev=20, azim=40)
ax1.tick_params(labelsize=TICK_SIZE-2)

# (b) Log-Osc 3D
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.plot(q_lo[:n_3d], p_lo[:n_3d], xi_lo[:n_3d],
         color=COLOR_LOGOSC, alpha=alpha_3d, lw=lw_3d, rasterized=True)
ax2.set_xlabel('q', fontsize=LABEL_SIZE, labelpad=8)
ax2.set_ylabel('p', fontsize=LABEL_SIZE, labelpad=8)
ax2.set_zlabel(r'$\xi$', fontsize=LABEL_SIZE, labelpad=8)
ax2.set_title('(b) Log-Osc', fontsize=TITLE_SIZE, pad=10)
ax2.view_init(elev=20, azim=40)
ax2.tick_params(labelsize=TICK_SIZE-2)

# (c) NHC projection (q, p)
ax3 = fig.add_subplot(2, 2, 3)
ax3.scatter(q_nhc, p_nhc, s=0.05, alpha=0.05, color=COLOR_NHC, rasterized=True)
for sm in [1, 2, 3]:
    sq = sm * np.sqrt(kT); sp = sm * np.sqrt(mass * kT)
    ax3.plot(sq*np.cos(theta), sp*np.sin(theta), 'k--', lw=0.8, alpha=0.5,
             label=f'{sm}$\\sigma$' if sm == 1 else None)
ax3.set_xlabel('q', fontsize=LABEL_SIZE)
ax3.set_ylabel('p', fontsize=LABEL_SIZE)
ax3.set_title('(c) NHC (q, p) projection', fontsize=TITLE_SIZE)
ax3.set_aspect('equal'); ax3.set_xlim(-4, 4); ax3.set_ylim(-4, 4)
ax3.tick_params(labelsize=TICK_SIZE)
# Add annotation about confinement
ax3.annotate('Thin torus\n(KAM-like)', xy=(0.7, 0.7), xytext=(2.5, 3.2),
             fontsize=11, color=COLOR_NHC, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=COLOR_NHC, lw=1.5))

# (d) Log-Osc projection (q, p)
ax4 = fig.add_subplot(2, 2, 4)
ax4.scatter(q_lo, p_lo, s=0.05, alpha=0.05, color=COLOR_LOGOSC, rasterized=True)
for sm in [1, 2, 3]:
    sq = sm * np.sqrt(kT); sp = sm * np.sqrt(mass * kT)
    ax4.plot(sq*np.cos(theta), sp*np.sin(theta), 'k--', lw=0.8, alpha=0.5)
ax4.set_xlabel('q', fontsize=LABEL_SIZE)
ax4.set_ylabel('p', fontsize=LABEL_SIZE)
ax4.set_title('(d) Log-Osc (q, p) projection', fontsize=TITLE_SIZE)
ax4.set_aspect('equal'); ax4.set_xlim(-4, 4); ax4.set_ylim(-4, 4)
ax4.tick_params(labelsize=TICK_SIZE)
ax4.annotate('Volume-filling\n(ergodic)', xy=(0, 0), xytext=(2.5, 3.0),
             fontsize=11, color=COLOR_LOGOSC, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=COLOR_LOGOSC, lw=1.5))

plt.tight_layout(pad=2.0)
outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', 'phase_space_3d.png')
plt.savefig(outpath, dpi=300, bbox_inches='tight')
print(f"Saved: {outpath}")
plt.close()
