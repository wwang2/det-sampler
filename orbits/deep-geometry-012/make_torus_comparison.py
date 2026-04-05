"""Torus Geometry Comparison: NHC vs Log-Osc at different Q values.

2x3 grid: top row NHC, bottom row Log-Osc
Columns: Q=0.1, Q=0.5, Q=1.0
Each panel: (q,p) phase portrait colored by time (viridis)
Overlay true Gaussian contours (1,2,3 sigma)
"""

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research.eval.potentials import HarmonicOscillator1D
from research.eval.baselines import NoseHooverChain
from research.eval.integrators import VelocityVerletThermostat, ThermostatState

# ---- Inline Log-Osc ----
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

# ---- Parameters ----
LABEL_SIZE = 13; TICK_SIZE = 11; TITLE_SIZE = 14
potential = HarmonicOscillator1D(omega=1.0)
kT = 1.0; mass = 1.0; dt = 0.01; n_steps = 500_000; seed = 42
Q_values = [0.1, 0.5, 1.0]

def simulate(thermostat_cls, Q, is_logosc=False):
    if is_logosc:
        dyn = LogOscThermostat(dim=1, kT=kT, mass=mass, Q=Q)
        rng = np.random.default_rng(seed)
        state = dyn.initial_state(np.array([0.5]), rng=rng)
        integ = LogOscVerlet(dyn, potential, dt=dt, kT=kT, mass=mass)
    else:
        dyn = NoseHooverChain(dim=1, chain_length=3, kT=kT, mass=mass, Q=Q)
        rng = np.random.default_rng(seed)
        state = dyn.initial_state(np.array([0.5]), rng=rng)
        integ = VelocityVerletThermostat(dyn, potential, dt=dt, kT=kT, mass=mass)

    # Subsample every 20 steps
    n_save = n_steps // 20
    qs = np.empty(n_save); ps = np.empty(n_save)
    idx = 0
    for i in range(n_steps):
        state = integ.step(state)
        if i % 20 == 0 and idx < n_save:
            qs[idx] = state.q[0]; ps[idx] = state.p[0]
            idx += 1
    return qs[:idx], ps[:idx]

# ---- Run simulations ----
results = {}
for Q in Q_values:
    print(f"Running NHC(M=3, Q={Q})...")
    results[('nhc', Q)] = simulate(NoseHooverChain, Q, is_logosc=False)
    print(f"Running Log-Osc(Q={Q})...")
    results[('lo', Q)] = simulate(None, Q, is_logosc=True)

# ---- Plotting ----
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

theta = np.linspace(0, 2*np.pi, 300)

for col, Q in enumerate(Q_values):
    for row, (method, label) in enumerate([('nhc', 'NHC (M=3)'), ('lo', 'Log-Osc')]):
        ax = axes[row, col]
        qs, ps = results[(method, Q)]
        n_pts = len(qs)
        t_color = np.arange(n_pts) / n_pts  # time fraction for coloring

        ax.scatter(qs, ps, c=t_color, cmap='viridis', s=0.1, alpha=0.15, rasterized=True)

        # Gaussian contours
        for sm in [1, 2, 3]:
            sq = sm * np.sqrt(kT); sp = sm * np.sqrt(mass * kT)
            ax.plot(sq*np.cos(theta), sp*np.sin(theta), 'k--', lw=0.8, alpha=0.4)

        ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=TICK_SIZE)

        if row == 0:
            ax.set_title(f'Q = {Q}', fontsize=TITLE_SIZE, fontweight='bold')
        if row == 1:
            ax.set_xlabel('q', fontsize=LABEL_SIZE)
        if col == 0:
            ax.set_ylabel(f'{label}\np', fontsize=LABEL_SIZE)
        else:
            ax.set_ylabel('p', fontsize=LABEL_SIZE)

# Add colorbar
cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
sm_cb = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))
sm_cb.set_array([])
cbar = fig.colorbar(sm_cb, cax=cax)
cbar.set_label('Time fraction', fontsize=LABEL_SIZE)
cbar.ax.tick_params(labelsize=TICK_SIZE)

plt.subplots_adjust(left=0.07, right=0.90, top=0.94, bottom=0.06, wspace=0.25, hspace=0.15)
fig.suptitle('Phase space coverage: NHC vs Log-Osc at different thermostat masses Q\n(1D Harmonic Oscillator, 500k steps)',
             fontsize=TITLE_SIZE+1, y=0.99)

outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', 'torus_comparison.png')
plt.savefig(outpath, dpi=300, bbox_inches='tight')
print(f"Saved: {outpath}")
plt.close()
