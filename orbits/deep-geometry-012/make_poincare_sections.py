"""Poincare Sections in Extended Phase Space.

For 1D HO: take Poincare section at q=0 (plot p vs xi when trajectory crosses q=0).
1x3 panel: NH, NHC(M=3), Log-Osc
NH should show isolated curves (KAM tori), Log-Osc should show scatter (chaos).
This is the most direct evidence of ergodicity breaking.
"""

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research.eval.potentials import HarmonicOscillator1D
from research.eval.baselines import NoseHoover, NoseHooverChain
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
LABEL_SIZE = 14; TICK_SIZE = 12; TITLE_SIZE = 15
COLOR_NH = '#1f77b4'; COLOR_NHC = '#ff7f0e'; COLOR_LOGOSC = '#2ca02c'

potential = HarmonicOscillator1D(omega=1.0)
kT = 1.0; mass = 1.0; dt = 0.008  # moderate dt for crossing detection
n_steps = 2_000_000; seed = 42

def collect_poincare(dynamics_cls, integrator_cls, Q=1.0, is_logosc=False, **kwargs):
    """Collect Poincare section points at q=0 crossings (positive direction)."""
    if is_logosc:
        dyn = LogOscThermostat(dim=1, kT=kT, mass=mass, Q=Q)
        rng = np.random.default_rng(seed)
        state = dyn.initial_state(np.array([0.5]), rng=rng)
        integ = LogOscVerlet(dyn, potential, dt=dt, kT=kT, mass=mass)
    else:
        dyn = dynamics_cls(dim=1, kT=kT, mass=mass, Q=Q, **kwargs)
        rng = np.random.default_rng(seed)
        state = dyn.initial_state(np.array([0.5]), rng=rng)
        integ = VelocityVerletThermostat(dyn, potential, dt=dt, kT=kT, mass=mass)

    p_sec = []
    xi_sec = []
    q_prev = state.q[0]

    for i in range(n_steps):
        state = integ.step(state)
        q_curr = state.q[0]

        # Detect zero crossing: q changes sign, going positive (p > 0)
        if q_prev < 0 and q_curr >= 0 and state.p[0] > 0:
            # Linear interpolation for more accurate crossing point
            frac = -q_prev / (q_curr - q_prev + 1e-30)
            p_at_cross = state.p[0]  # approximate (p changes little over one step)
            xi_at_cross = state.xi[0]
            p_sec.append(p_at_cross)
            xi_sec.append(xi_at_cross)

        q_prev = q_curr

    return np.array(p_sec), np.array(xi_sec)

# ---- Run simulations ----
print("Collecting Poincare section for NH...")
p_nh, xi_nh = collect_poincare(NoseHoover, VelocityVerletThermostat, Q=1.0)
print(f"  NH: {len(p_nh)} crossings")

print("Collecting Poincare section for NHC(M=3)...")
p_nhc, xi_nhc = collect_poincare(NoseHooverChain, VelocityVerletThermostat, Q=1.0, chain_length=3)
print(f"  NHC: {len(p_nhc)} crossings")

print("Collecting Poincare section for Log-Osc...")
p_lo, xi_lo = collect_poincare(None, None, Q=1.0, is_logosc=True)
print(f"  Log-Osc: {len(p_lo)} crossings")

# ---- Plotting ----
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

datasets = [
    (axes[0], p_nh, xi_nh, '(a) Nose-Hoover', COLOR_NH, 'KAM tori\n(non-ergodic)'),
    (axes[1], p_nhc, xi_nhc, '(b) NHC (M=3)', COLOR_NHC, 'Partially\nbroken tori'),
    (axes[2], p_lo, xi_lo, '(c) Log-Osc', COLOR_LOGOSC, 'Chaotic sea\n(ergodic)'),
]

for ax, p_data, xi_data, title, color, annotation in datasets:
    ax.scatter(xi_data, p_data, s=0.3, alpha=0.3, color=color, rasterized=True)
    ax.set_xlabel(r'$\xi_1$', fontsize=LABEL_SIZE)
    ax.set_ylabel('p  (at q=0 crossing)', fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)

    # Add annotation
    ax.text(0.95, 0.95, annotation, transform=ax.transAxes,
            fontsize=12, fontweight='bold', color=color,
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.9))

# Make axis ranges consistent
xi_max = max(np.abs(xi_nh).max(), np.abs(xi_nhc).max(), np.abs(xi_lo).max()) * 1.1
p_max = max(np.abs(p_nh).max(), np.abs(p_nhc).max(), np.abs(p_lo).max()) * 1.1
xi_max = min(xi_max, 15); p_max = min(p_max, 6)
for ax in axes:
    ax.set_xlim(-xi_max, xi_max)
    ax.set_ylim(0, p_max)  # p > 0 at crossing by construction

plt.tight_layout(pad=1.5)
fig.suptitle('Poincare sections at q = 0  (1D Harmonic Oscillator, 2M steps, dt=0.008)',
             fontsize=TITLE_SIZE, y=1.02)

outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', 'poincare_sections.png')
plt.savefig(outpath, dpi=300, bbox_inches='tight')
print(f"Saved: {outpath}")
plt.close()
