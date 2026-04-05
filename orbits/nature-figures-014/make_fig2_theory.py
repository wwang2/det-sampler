#!/usr/bin/env python3
"""Figure 2: The Theory -- Friction Function Design Space.

4x4 panel layout (rows x cols):
  Col 1: Quadratic/NH
  Col 2: Logarithmic/Log-Osc
  Col 3: Cosh
  Col 4: Arctan

  Row 1: V(xi) confining potentials
  Row 2: g(xi) friction functions with bounded/unbounded annotation
  Row 3: p(xi) ~ exp(-V/kT) marginal distributions
  Row 4: (q,p) phase portraits on 1D HO (500k evals each)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from shared import (
    FONTSIZE_LABEL, FONTSIZE_TICK, FONTSIZE_TITLE, FONTSIZE_ANNOT, DPI,
    COLOR_NHC, COLOR_LO, g_func,
)
from research.eval.potentials import HarmonicOscillator1D
from research.eval.integrators import ThermostatState

# Thermostat definitions: (name, color, V(xi), g(xi), Q_for_sim)
# Master theorem: g(xi) = V'(xi) / Q

THERMOSTATS = [
    {
        'name': 'Quadratic\n(Nose-Hoover)',
        'color': COLOR_NHC,
        'V': lambda xi, Q=1.0: Q * xi**2 / 2,
        'g': lambda xi: xi,  # V'(xi)/Q = xi
        'bounded': False,
        'Q': 1.0,
    },
    {
        'name': 'Logarithmic\n(Log-Osc)',
        'color': COLOR_LO,
        'V': lambda xi, Q=1.0: Q * np.log(1 + xi**2),
        'g': lambda xi: 2 * xi / (1 + xi**2),
        'bounded': True,
        'Q': 0.5,
    },
    {
        'name': 'Cosh',
        'color': '#d62728',
        'V': lambda xi, Q=1.0: Q * (np.cosh(xi) - 1),
        'g': lambda xi: np.sinh(xi),  # V'(xi)/Q = sinh(xi)
        'bounded': False,
        'Q': 1.0,
    },
    {
        'name': 'Arctan',
        'color': '#9467bd',
        'V': lambda xi, Q=1.0: Q * (xi * np.arctan(xi) - 0.5 * np.log(1 + xi**2)),
        'g': lambda xi: np.arctan(xi),  # V'(xi)/Q = arctan(xi)
        'bounded': True,
        'Q': 0.5,
    },
]


class GenericThermostat:
    """Generic thermostat with arbitrary friction function g(xi)."""
    def __init__(self, g_func, dim=1, kT=1.0, mass=1.0, Q=1.0):
        self._g = g_func
        self.dim = dim
        self.kT = kT
        self.mass = mass
        self.Q = Q
        self.name = "generic"

    def initial_state(self, q0, rng=None):
        if rng is None: rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        return ThermostatState(q0.copy(), p0, np.array([0.0]), 0)

    def dqdt(self, state, grad_U): return state.p / self.mass
    def dpdt(self, state, grad_U):
        return -grad_U - self._g(state.xi[0]) * state.p
    def dxidt(self, state, grad_U):
        K = np.sum(state.p**2) / self.mass
        return np.array([(K - self.dim * self.kT) / self.Q])


class GenericVerlet:
    def __init__(self, dynamics, potential, dt, kT=1.0, mass=1.0):
        self.dynamics = dynamics
        self.potential = potential
        self.dt = dt
        self._cached_grad_U = None

    def step(self, state):
        q, p, xi, n_evals = state
        dt = self.dt
        half_dt = 0.5 * dt

        if self._cached_grad_U is not None:
            grad_U = self._cached_grad_U
        else:
            grad_U = self.potential.gradient(q)
            n_evals += 1

        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        gxi = self.dynamics._g(xi[0])
        scale = np.clip(np.exp(-gxi * half_dt), 1e-10, 1e10)
        p = p * scale
        p = p - half_dt * grad_U

        q = q + dt * p / self.dynamics.mass

        if np.any(np.isnan(q)) or np.any(np.isnan(p)):
            self._cached_grad_U = None
            return ThermostatState(q, p, xi, n_evals)

        grad_U = self.potential.gradient(q)
        n_evals += 1

        p = p - half_dt * grad_U
        gxi = self.dynamics._g(xi[0])
        scale = np.clip(np.exp(-gxi * half_dt), 1e-10, 1e10)
        p = p * scale

        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        self._cached_grad_U = grad_U
        return ThermostatState(q, p, xi, n_evals)


def run_phase_portrait(g_func_val, Q, n_evals=500_000, seed=42):
    """Run thermostat on 1D HO and return (q, p) trajectory."""
    pot = HarmonicOscillator1D(omega=1.0)
    dyn = GenericThermostat(g_func_val, dim=1, Q=Q)
    rng = np.random.default_rng(seed)
    q0 = np.array([1.0])
    state = dyn.initial_state(q0, rng=rng)
    integrator = GenericVerlet(dyn, pot, dt=0.005)

    qs, ps = [], []
    while state.n_force_evals < n_evals:
        qs.append(state.q[0])
        ps.append(state.p[0])
        state = integrator.step(state)
        if np.any(np.isnan(state.q)):
            break

    return np.array(qs), np.array(ps)


def make_figure():
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    xi_range = np.linspace(-5, 5, 500)

    panel_idx = 0
    panel_labels = [chr(ord('a') + i) for i in range(16)]

    for col, thermo in enumerate(THERMOSTATS):
        color = thermo['color']

        # ── Row 1: V(xi) ──
        ax = axes[0, col]
        V_vals = thermo['V'](xi_range)
        ax.plot(xi_range, V_vals, color=color, lw=2.5)
        ax.set_ylabel(r'$V(\xi)$', fontsize=FONTSIZE_LABEL) if col == 0 else None
        ax.set_title(thermo['name'], fontsize=FONTSIZE_TITLE - 2, fontweight='bold')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-0.5, min(np.max(V_vals), 15))
        ax.tick_params(labelsize=FONTSIZE_TICK)
        ax.text(0.03, 0.93, f'({panel_labels[col]})', transform=ax.transAxes,
                fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

        # ── Row 2: g(xi) ──
        ax = axes[1, col]
        g_vals = np.array([thermo['g'](x) for x in xi_range])
        ax.plot(xi_range, g_vals, color=color, lw=2.5)
        ax.axhline(0, color='gray', ls='--', lw=0.5, alpha=0.5)
        if thermo['bounded']:
            ax.axhline(1, color='gray', ls=':', lw=0.8, alpha=0.5)
            ax.axhline(-1, color='gray', ls=':', lw=0.8, alpha=0.5)
            ax.text(0.97, 0.95, 'BOUNDED', transform=ax.transAxes,
                    fontsize=FONTSIZE_ANNOT, ha='right', va='top', color='green',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        else:
            ax.text(0.97, 0.95, 'UNBOUNDED', transform=ax.transAxes,
                    fontsize=FONTSIZE_ANNOT, ha='right', va='top', color='red',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        ax.set_ylabel(r'$g(\xi)$', fontsize=FONTSIZE_LABEL) if col == 0 else None
        ax.set_xlim(-5, 5)
        g_max = min(np.max(np.abs(g_vals)), 6)
        ax.set_ylim(-g_max * 1.2, g_max * 1.2)
        ax.tick_params(labelsize=FONTSIZE_TICK)
        ax.text(0.03, 0.93, f'({panel_labels[4 + col]})', transform=ax.transAxes,
                fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

        # ── Row 3: p(xi) ~ exp(-V/kT) ──
        ax = axes[2, col]
        log_p = -thermo['V'](xi_range)
        log_p -= np.max(log_p)
        p_xi = np.exp(log_p)
        p_xi /= np.trapezoid(p_xi, xi_range)
        ax.fill_between(xi_range, p_xi, alpha=0.3, color=color)
        ax.plot(xi_range, p_xi, color=color, lw=2)
        ax.set_ylabel(r'$p(\xi)$', fontsize=FONTSIZE_LABEL) if col == 0 else None
        ax.set_xlim(-5, 5)
        ax.set_ylim(0, None)
        ax.tick_params(labelsize=FONTSIZE_TICK)
        ax.text(0.03, 0.93, f'({panel_labels[8 + col]})', transform=ax.transAxes,
                fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

        # ── Row 4: Phase portrait on 1D HO ──
        ax = axes[3, col]
        print(f"Running phase portrait for {thermo['name'].replace(chr(10), ' ')}...")
        qs, ps = run_phase_portrait(thermo['g'], thermo['Q'])
        thin = max(1, len(qs) // 30000)
        ax.scatter(qs[::thin], ps[::thin], s=0.15, alpha=0.3, color=color,
                   rasterized=True)

        # Gaussian contours
        theta = np.linspace(0, 2 * np.pi, 200)
        for sigma in [1, 2, 3]:
            ax.plot(sigma * np.cos(theta), sigma * np.sin(theta),
                    '--', color='gray', alpha=0.5, lw=0.8)

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.set_xlabel(r'$q$', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel(r'$p$', fontsize=FONTSIZE_LABEL) if col == 0 else None
        ax.tick_params(labelsize=FONTSIZE_TICK)
        ax.text(0.03, 0.93, f'({panel_labels[12 + col]})', transform=ax.transAxes,
                fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    # Row labels on left
    row_labels = [r'Potential $V(\xi)$', r'Friction $g(\xi) = V^\prime(\xi)/Q$',
                  r'Marginal $p(\xi) \propto e^{-V(\xi)/kT}$',
                  'Phase portrait on 1D HO']
    for row, label in enumerate(row_labels):
        axes[row, 0].annotate(label, xy=(-0.35, 0.5), xycoords='axes fraction',
                              fontsize=FONTSIZE_LABEL - 1, ha='center', va='center',
                              rotation=90, fontweight='bold')

    fig.suptitle('Thermostat Design Space: Confining Potential Determines Friction Function',
                 fontsize=FONTSIZE_TITLE, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0.05, 0, 1, 0.97])

    out_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fig2_theory.png')
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == '__main__':
    make_figure()
