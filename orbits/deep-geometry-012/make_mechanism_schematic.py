"""Bounded Friction Mechanism Schematic.

2x2 panel showing the feedback loop:
  (a) NH feedback diagram (text/arrows)
  (b) Log-Osc feedback diagram (text/arrows)
  (c) Time series of g(xi) for NH vs Log-Osc on 1D HO
  (d) Time series of kinetic energy for both
"""

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from research.eval.potentials import HarmonicOscillator1D
from research.eval.baselines import NoseHoover
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

# ---- Style ----
COLOR_NH = '#1f77b4'; COLOR_LOGOSC = '#2ca02c'
LABEL_SIZE = 14; TICK_SIZE = 12; TITLE_SIZE = 15

# ---- Simulate ----
potential = HarmonicOscillator1D(omega=1.0)
kT = 1.0; mass = 1.0; dt = 0.01; n_steps = 50_000; seed = 42

def simulate_timeseries(is_logosc=False, Q=1.0):
    if is_logosc:
        dyn = LogOscThermostat(dim=1, kT=kT, mass=mass, Q=Q)
        rng = np.random.default_rng(seed)
        state = dyn.initial_state(np.array([0.5]), rng=rng)
        integ = LogOscVerlet(dyn, potential, dt=dt, kT=kT, mass=mass)
    else:
        dyn = NoseHoover(dim=1, kT=kT, mass=mass, Q=Q)
        rng = np.random.default_rng(seed)
        state = dyn.initial_state(np.array([0.5]), rng=rng)
        integ = VelocityVerletThermostat(dyn, potential, dt=dt, kT=kT, mass=mass)

    times = []; gvals = []; ke_vals = []; xi_vals = []
    for i in range(n_steps):
        state = integ.step(state)
        if i % 5 == 0:
            t = i * dt
            times.append(t)
            xi_val = state.xi[0]
            xi_vals.append(xi_val)
            if is_logosc:
                gvals.append(g_func(xi_val))
            else:
                gvals.append(xi_val)  # NH friction IS xi
            ke_vals.append(0.5 * state.p[0]**2 / mass)
    return np.array(times), np.array(gvals), np.array(ke_vals), np.array(xi_vals)

print("Simulating NH time series...")
t_nh, g_nh, ke_nh, xi_nh = simulate_timeseries(is_logosc=False, Q=1.0)
print("Simulating Log-Osc time series...")
t_lo, g_lo, ke_lo, xi_lo = simulate_timeseries(is_logosc=True, Q=1.0)

# ---- Plotting ----
fig = plt.figure(figsize=(14, 11))

# ---------- (a) NH Feedback Diagram ----------
ax_a = fig.add_subplot(2, 2, 1)
ax_a.set_xlim(0, 10); ax_a.set_ylim(0, 10)
ax_a.set_aspect('equal')
ax_a.axis('off')
ax_a.set_title('(a) Nose-Hoover feedback loop', fontsize=TITLE_SIZE, pad=10)

# Draw feedback loop boxes
boxes_nh = [
    (2, 8, 'HOT\n$K > kT$'),
    (7, 8, r'$\xi$ grows' + '\n' + r'$\dot\xi > 0$'),
    (7, 3, 'Friction grows\n' + r'$g = \xi \to \infty$'),
    (2, 3, 'COLD\n$K < kT$'),
]
for x, y, txt in boxes_nh:
    ax_a.add_patch(mpatches.FancyBboxPatch((x-1.3, y-0.9), 2.6, 1.8,
                   boxstyle="round,pad=0.2", facecolor='#e6f0ff', edgecolor=COLOR_NH, lw=2))
    ax_a.text(x, y, txt, ha='center', va='center', fontsize=11, fontweight='bold')

# Arrows
arrow_kw = dict(arrowstyle='->', color=COLOR_NH, lw=2.5, connectionstyle='arc3,rad=0')
ax_a.annotate('', xy=(5.5, 8), xytext=(3.5, 8), arrowprops=arrow_kw)
ax_a.annotate('', xy=(7, 5), xytext=(7, 7), arrowprops=arrow_kw)
ax_a.annotate('', xy=(3.5, 3), xytext=(5.5, 3), arrowprops=arrow_kw)
ax_a.annotate('', xy=(2, 7), xytext=(2, 5), arrowprops=arrow_kw)

# Center label
ax_a.text(4.5, 5.5, 'RESONANT\nLOOP', ha='center', va='center', fontsize=13,
          color=COLOR_NH, fontweight='bold', fontstyle='italic')
ax_a.text(4.5, 4.5, '(creates KAM tori)', ha='center', va='center', fontsize=10,
          color='gray')

# ---------- (b) Log-Osc Feedback Diagram ----------
ax_b = fig.add_subplot(2, 2, 2)
ax_b.set_xlim(0, 10); ax_b.set_ylim(0, 10)
ax_b.set_aspect('equal')
ax_b.axis('off')
ax_b.set_title('(b) Log-Osc feedback loop', fontsize=TITLE_SIZE, pad=10)

boxes_lo = [
    (2, 8, 'HOT\n$K > kT$'),
    (7, 8, r'$\xi$ grows' + '\n' + r'$\dot\xi > 0$'),
    (7, 3, 'Friction SATURATES\n' + r'$g \to 1$ (bounded)'),
    (2, 3, 'Partial cooling\nxi keeps growing'),
]
for x, y, txt in boxes_lo:
    ax_b.add_patch(mpatches.FancyBboxPatch((x-1.3, y-0.9), 2.6, 1.8,
                   boxstyle="round,pad=0.2", facecolor='#e6ffe6', edgecolor=COLOR_LOGOSC, lw=2))
    ax_b.text(x, y, txt, ha='center', va='center', fontsize=10, fontweight='bold')

arrow_kw_lo = dict(arrowstyle='->', color=COLOR_LOGOSC, lw=2.5, connectionstyle='arc3,rad=0')
ax_b.annotate('', xy=(5.5, 8), xytext=(3.5, 8), arrowprops=arrow_kw_lo)
ax_b.annotate('', xy=(7, 5), xytext=(7, 7), arrowprops=arrow_kw_lo)
ax_b.annotate('', xy=(3.5, 3), xytext=(5.5, 3), arrowprops=arrow_kw_lo)
ax_b.annotate('', xy=(2, 7), xytext=(2, 5),
              arrowprops=dict(arrowstyle='->', color='red', lw=2.5, connectionstyle='arc3,rad=0'))

# Center label
ax_b.text(4.5, 5.5, 'BROKEN\nLOOP', ha='center', va='center', fontsize=13,
          color=COLOR_LOGOSC, fontweight='bold', fontstyle='italic')
ax_b.text(4.5, 4.5, '(escapes tori)', ha='center', va='center', fontsize=10,
          color='gray')
# Escape annotation
ax_b.text(1.0, 5.5, 'ESCAPE', ha='center', va='center', fontsize=12,
          color='red', fontweight='bold', rotation=90)

# ---------- (c) Friction time series ----------
ax_c = fig.add_subplot(2, 2, 3)
t_show = 200  # show first 200 time units
mask_nh = t_nh <= t_show; mask_lo = t_lo <= t_show

ax_c.plot(t_nh[mask_nh], g_nh[mask_nh], color=COLOR_NH, alpha=0.7, lw=0.5, label=r'NH: $g(\xi) = \xi$')
ax_c.plot(t_lo[mask_lo], g_lo[mask_lo], color=COLOR_LOGOSC, alpha=0.7, lw=0.5, label=r'Log-Osc: $g(\xi) = 2\xi/(1+\xi^2)$')
ax_c.axhline(1, color='gray', ls=':', lw=1, alpha=0.7)
ax_c.axhline(-1, color='gray', ls=':', lw=1, alpha=0.7)
ax_c.set_xlabel('Time', fontsize=LABEL_SIZE)
ax_c.set_ylabel(r'Friction $g(\xi)$', fontsize=LABEL_SIZE)
ax_c.set_title('(c) Friction coefficient time series', fontsize=TITLE_SIZE)
ax_c.legend(fontsize=11, loc='upper right')
ax_c.tick_params(labelsize=TICK_SIZE)

# Annotate bounded region
ax_c.text(0.03, 0.97, 'Bounded: |g| < 1', transform=ax_c.transAxes, fontsize=10,
          color=COLOR_LOGOSC, fontweight='bold', va='top')
ax_c.text(0.03, 0.88, 'Unbounded oscillations', transform=ax_c.transAxes, fontsize=10,
          color=COLOR_NH, fontweight='bold', va='top')

# ---------- (d) Kinetic energy time series ----------
ax_d = fig.add_subplot(2, 2, 4)
ax_d.plot(t_nh[mask_nh], ke_nh[mask_nh], color=COLOR_NH, alpha=0.5, lw=0.5, label='NH')
ax_d.plot(t_lo[mask_lo], ke_lo[mask_lo], color=COLOR_LOGOSC, alpha=0.5, lw=0.5, label='Log-Osc')
ax_d.axhline(0.5*kT, color='gray', ls='--', lw=1.5, alpha=0.7, label=r'$\langle K \rangle = kT/2$')
ax_d.set_xlabel('Time', fontsize=LABEL_SIZE)
ax_d.set_ylabel('Kinetic energy $K$', fontsize=LABEL_SIZE)
ax_d.set_title('(d) Kinetic energy time series', fontsize=TITLE_SIZE)
ax_d.legend(fontsize=11, loc='upper right')
ax_d.tick_params(labelsize=TICK_SIZE)
ax_d.set_ylim(0, max(ke_nh[mask_nh].max(), ke_lo[mask_lo].max()) * 1.1)

# Annotate
ax_d.text(0.03, 0.97, 'Periodic (non-ergodic)', transform=ax_d.transAxes, fontsize=10,
          color=COLOR_NH, fontweight='bold', va='top')
ax_d.text(0.03, 0.88, 'Irregular (chaotic)', transform=ax_d.transAxes, fontsize=10,
          color=COLOR_LOGOSC, fontweight='bold', va='top')

plt.tight_layout(pad=2.0)
outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', 'mechanism_schematic.png')
plt.savefig(outpath, dpi=300, bbox_inches='tight')
print(f"Saved: {outpath}")
plt.close()
