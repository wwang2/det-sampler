"""Systematic Exploration of Nonlinear Friction Functions for Deterministic Thermostats.

Tests a family of friction functions g(xi) in the generalized Nose-Hoover framework:
    dq/dt = p/m
    dp/dt = -dU/dq - g(xi)*p
    dxi/dt = (1/Q) * (sum p_i^2/m - dim*kT)

Each g(xi) = V'(xi) for some thermostat potential V(xi), giving extended Hamiltonian:
    H_ext = U(q) + p^2/(2m) + V(xi)
with invariant measure rho ~ exp(-H_ext/kT).

Friction families tested:
1. Log-osc (baseline): g(xi) = 2xi/(1+xi^2)
2. Tanh:               g(xi) = tanh(a*xi)
3. Arctan:             g(xi) = (2/pi)*arctan(xi)
4. Soft-clip:          g(xi) = xi/sqrt(1+xi^2)
5. Cubic saturation:   g(xi) = xi/(1+xi^2/3)
6. Gaussian-damped:    g(xi) = xi*exp(-xi^2/(2*s^2))
7. Standard NH:        g(xi) = xi

References:
    - Nose (1984), J. Chem. Phys. 81, 511 -- original Nose thermostat
    - Hoover (1985), Phys. Rev. A 31, 1695 -- Nose-Hoover formulation
    - Martyna et al. (1992), J. Chem. Phys. 97, 2635 -- Nose-Hoover chains
    - Builds on orbit/log-osc-001 (Issue #3) which showed bounded friction improves ergodicity
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from research.eval.integrators import ThermostatState


# ---------------------------------------------------------------------------
# Friction function definitions
# ---------------------------------------------------------------------------

def g_log_osc(xi):
    """Log-osc: g(xi) = 2xi/(1+xi^2). Bounded in [-1, 1]."""
    return 2.0 * xi / (1.0 + xi**2)


def g_tanh(xi, a=1.0):
    """Tanh: g(xi) = tanh(a*xi). Bounded in (-1, 1)."""
    return np.tanh(a * xi)


def g_arctan(xi):
    """Arctan: g(xi) = (2/pi)*arctan(xi). Bounded in (-1, 1)."""
    return (2.0 / np.pi) * np.arctan(xi)


def g_soft_clip(xi):
    """Soft-clip: g(xi) = xi/sqrt(1+xi^2). Bounded in (-1, 1)."""
    return xi / np.sqrt(1.0 + xi**2)


def g_cubic_sat(xi):
    """Cubic saturation: g(xi) = xi/(1+xi^2/3). Bounded in (-3*sqrt(3)/2, 3*sqrt(3)/2) ~ (-2.6, 2.6)."""
    return xi / (1.0 + xi**2 / 3.0)


def g_gaussian_damped(xi, s=1.0):
    """Gaussian-damped: g(xi) = xi*exp(-xi^2/(2*s^2)). Non-monotone, goes to 0 at large |xi|."""
    return xi * np.exp(-xi**2 / (2.0 * s**2))


def g_standard_nh(xi):
    """Standard Nose-Hoover: g(xi) = xi. Unbounded."""
    return xi


# ---------------------------------------------------------------------------
# Property catalog for each friction function
# ---------------------------------------------------------------------------

FRICTION_CATALOG = {
    "log_osc": {
        "g_func": g_log_osc,
        "label": "Log-osc",
        "bounded": True,
        "max_abs_g": 1.0,
        "g_prime_0": 2.0,  # g'(0) = 2
        "decay_rate": "1/xi",  # g(xi) ~ 2/xi for large xi
        "confining": True,  # V(xi) = log(1+xi^2) -> inf
    },
    "tanh_0.5": {
        "g_func": lambda xi: g_tanh(xi, a=0.5),
        "label": "Tanh(a=0.5)",
        "bounded": True,
        "max_abs_g": 1.0,
        "g_prime_0": 0.5,
        "decay_rate": "exp(-xi)",
        "confining": True,  # V = log(cosh(a*xi))/a -> |xi| for large xi
    },
    "tanh_1.0": {
        "g_func": lambda xi: g_tanh(xi, a=1.0),
        "label": "Tanh(a=1.0)",
        "bounded": True,
        "max_abs_g": 1.0,
        "g_prime_0": 1.0,
        "decay_rate": "exp(-xi)",
        "confining": True,
    },
    "tanh_2.0": {
        "g_func": lambda xi: g_tanh(xi, a=2.0),
        "label": "Tanh(a=2.0)",
        "bounded": True,
        "max_abs_g": 1.0,
        "g_prime_0": 2.0,
        "decay_rate": "exp(-xi)",
        "confining": True,
    },
    "tanh_5.0": {
        "g_func": lambda xi: g_tanh(xi, a=5.0),
        "label": "Tanh(a=5.0)",
        "bounded": True,
        "max_abs_g": 1.0,
        "g_prime_0": 5.0,
        "decay_rate": "exp(-xi)",
        "confining": True,
    },
    "arctan": {
        "g_func": g_arctan,
        "label": "Arctan",
        "bounded": True,
        "max_abs_g": 1.0,
        "g_prime_0": 2.0 / np.pi,  # ~0.637
        "decay_rate": "1/xi",
        "confining": True,  # V ~ xi*arctan(xi) -> pi/2 * |xi| for large xi
    },
    "soft_clip": {
        "g_func": g_soft_clip,
        "label": "Soft-clip",
        "bounded": True,
        "max_abs_g": 1.0,
        "g_prime_0": 1.0,
        "decay_rate": "1/xi^2",
        "confining": True,  # V = sqrt(1+xi^2)-1 -> |xi| for large xi
    },
    "cubic_sat": {
        "g_func": g_cubic_sat,
        "label": "Cubic-sat",
        "bounded": True,
        "max_abs_g": 3.0 * np.sqrt(3) / 2.0,  # ~2.598
        "g_prime_0": 1.0,
        "decay_rate": "1/xi",
        "confining": True,  # V = 1.5*log(1+xi^2/3) -> inf
    },
    "gaussian_damped_1.0": {
        "g_func": lambda xi: g_gaussian_damped(xi, s=1.0),
        "label": "Gauss(s=1.0)",
        "bounded": True,
        "max_abs_g": 1.0 / np.sqrt(np.e),  # ~0.607 at xi=s
        "g_prime_0": 1.0,
        "decay_rate": "xi*exp(-xi^2)",
        "confining": False,  # V = -s^2*exp(-xi^2/(2s^2)) -> 0
    },
    "gaussian_damped_2.0": {
        "g_func": lambda xi: g_gaussian_damped(xi, s=2.0),
        "label": "Gauss(s=2.0)",
        "bounded": True,
        "max_abs_g": 2.0 / np.sqrt(np.e),  # ~1.213
        "g_prime_0": 1.0,
        "decay_rate": "xi*exp(-xi^2)",
        "confining": False,
    },
    "standard_nh": {
        "g_func": g_standard_nh,
        "label": "Standard NH",
        "bounded": False,
        "max_abs_g": float('inf'),
        "g_prime_0": 1.0,
        "decay_rate": "none (grows linearly)",
        "confining": True,  # V = xi^2/2 -> inf
    },
}


# ---------------------------------------------------------------------------
# Generic Nonlinear Friction Thermostat
# ---------------------------------------------------------------------------

class NonlinearFrictionThermostat:
    """Generalized Nose-Hoover thermostat with nonlinear friction g(xi).

    Equations of motion:
        dq/dt = p/m
        dp/dt = -dU/dq - g(xi)*p
        dxi/dt = (1/Q) * (sum p_i^2/m - dim*kT)

    The friction function g(xi) determines the thermostat potential V(xi)
    where V'(xi) = g(xi).
    """

    def __init__(self, friction_key: str, dim: int, kT: float = 1.0,
                 mass: float = 1.0, Q: float = 1.0):
        self.friction_key = friction_key
        self.g_func = FRICTION_CATALOG[friction_key]["g_func"]
        self.name = f"nonlinear_{friction_key}"
        self.dim = dim
        self.kT = kT
        self.mass = mass
        self.Q = Q

    def initial_state(self, q0: np.ndarray, rng: np.random.Generator | None = None) -> ThermostatState:
        if rng is None:
            rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        xi0 = np.array([0.0])
        return ThermostatState(q0.copy(), p0, xi0, 0)

    def dqdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        return state.p / self.mass

    def dpdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        xi_val = state.xi[0]
        return -grad_U - self.g_func(xi_val) * state.p

    def dxidt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        kinetic = np.sum(state.p**2) / self.mass
        return np.array([(kinetic - self.dim * self.kT) / self.Q])


# ---------------------------------------------------------------------------
# Custom Velocity Verlet for nonlinear friction
# ---------------------------------------------------------------------------

class NonlinearFrictionVerlet:
    """Velocity Verlet integrator for generalized nonlinear friction thermostat.

    Uses exp(-g(xi)*dt/2) momentum rescaling. FSAL scheme, 1 force eval per step.

    Splitting:
      1. Half-step thermostat:  xi += (dt/2) * dxi/dt
      2. Half-step momenta:     p *= exp(-g(xi)*dt/2); p -= (dt/2)*grad_U
      3. Full-step positions:   q += dt * p/m
      4. Recompute forces
      5. Half-step momenta:     p -= (dt/2)*grad_U; p *= exp(-g(xi)*dt/2)
      6. Half-step thermostat:  xi += (dt/2) * dxi/dt
    """

    def __init__(self, dynamics, potential, dt: float, kT: float = 1.0, mass: float = 1.0):
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        self.dynamics = dynamics
        self.potential = potential
        self.dt = dt
        self.kT = kT
        self.mass = mass
        self._cached_grad_U = None

    def step(self, state: ThermostatState) -> ThermostatState:
        q, p, xi, n_evals = state
        dt = self.dt
        half_dt = 0.5 * dt
        g_func = self.dynamics.g_func

        # Get gradient (FSAL)
        if self._cached_grad_U is not None:
            grad_U = self._cached_grad_U
        else:
            grad_U = self.potential.gradient(q)
            n_evals += 1

        # Half-step thermostat
        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        # Half-step momenta: friction rescaling + kick
        gxi = g_func(xi[0])
        scale = np.exp(-gxi * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale
        p = p - half_dt * grad_U

        # Full-step positions
        q = q + dt * p / self.mass

        # NaN check
        if np.any(np.isnan(q)) or np.any(np.isnan(p)):
            self._cached_grad_U = None
            return ThermostatState(q, p, xi, n_evals)

        # Recompute forces
        grad_U = self.potential.gradient(q)
        n_evals += 1

        # Half-step momenta: kick + friction rescaling
        p = p - half_dt * grad_U
        gxi = g_func(xi[0])
        scale = np.exp(-gxi * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale

        # Half-step thermostat
        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        self._cached_grad_U = grad_U
        return ThermostatState(q, p, xi, n_evals)
