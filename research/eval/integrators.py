"""Numerical integration schemes for thermostat dynamics.

Each integrator takes a sampler's equations of motion and advances them one step.
The integrator is part of the deliverable — different samplers may need different schemes.
"""

import numpy as np
from typing import Protocol, NamedTuple


class ThermostatState(NamedTuple):
    """State of a thermostat sampler."""
    q: np.ndarray       # positions
    p: np.ndarray       # momenta
    xi: np.ndarray      # thermostat variables (can be empty)
    n_force_evals: int  # cumulative force evaluation count


class ThermostatDynamics(Protocol):
    """Protocol for thermostat dynamics: defines the equations of motion."""

    @property
    def name(self) -> str: ...

    def dqdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray: ...
    def dpdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray: ...
    def dxidt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray: ...


class VelocityVerletThermostat:
    """Velocity Verlet with FSAL for Nose-Hoover type thermostats.

    Splitting scheme (1 force eval per step after first):
      1. Half-step thermostat:  xi += (dt/2) * f(p, xi)
      2. Half-step momenta:     p *= exp(-xi * dt/2)  then  p += -(dt/2) * grad_U
      3. Full-step positions:   q += dt * p/m
      4. Recompute forces (cached via FSAL for next step)
      5. Half-step momenta:     p += -(dt/2) * grad_U  then  p *= exp(-xi * dt/2)
      6. Half-step thermostat:  xi += (dt/2) * f(p, xi)

    The momentum thermostat coupling uses the analytical exp(-xi*dt/2) rescaling
    from Martyna et al. (1996) to achieve proper time-reversibility.
    The force/friction parts are split so the palindromic structure is preserved.
    """

    def __init__(self, dynamics: ThermostatDynamics, potential, dt: float,
                 kT: float = 1.0, mass: float = 1.0):
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if kT <= 0:
            raise ValueError(f"kT must be positive, got {kT}")
        if mass <= 0:
            raise ValueError(f"mass must be positive, got {mass}")
        self.dynamics = dynamics
        self.potential = potential
        self.dt = dt
        self.kT = kT
        self.mass = mass
        self._cached_grad_U = None  # FSAL cache

    def step(self, state: ThermostatState) -> ThermostatState:
        q, p, xi, n_evals = state
        dt = self.dt
        half_dt = 0.5 * dt

        # Get gradient (use FSAL cache if available)
        if self._cached_grad_U is not None:
            grad_U = self._cached_grad_U
        else:
            grad_U = self.potential.gradient(q)
            n_evals += 1

        # Half-step thermostat variables
        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        # Half-step momenta: analytical friction + kick
        # p *= exp(-xi_1 * dt/2)  — thermostat friction via rescaling
        if len(xi) > 0:
            scale = np.exp(-xi[0] * half_dt)
            # Clamp to prevent overflow/underflow
            scale = np.clip(scale, 1e-10, 1e10)
            p = p * scale
        # Kick from potential force
        p = p - half_dt * grad_U

        # Full-step positions
        q = q + dt * p / self.mass

        # NaN check
        if np.any(np.isnan(q)) or np.any(np.isnan(p)):
            self._cached_grad_U = None
            return ThermostatState(q, p, xi, n_evals)

        # Recompute forces at new position (will be cached for next step)
        grad_U = self.potential.gradient(q)
        n_evals += 1

        # Half-step momenta: kick + analytical friction
        p = p - half_dt * grad_U
        if len(xi) > 0:
            scale = np.exp(-xi[0] * half_dt)
            scale = np.clip(scale, 1e-10, 1e10)
            p = p * scale

        # Half-step thermostat variables
        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        # Cache gradient for FSAL
        self._cached_grad_U = grad_U

        return ThermostatState(q, p, xi, n_evals)


class SymplecticEuler:
    """Symplectic Euler (leapfrog variant) — simplest baseline integrator.

    1. p += dt * dp/dt
    2. q += dt * dq/dt
    1 force eval per step. Not time-reversible.
    """

    def __init__(self, dynamics: ThermostatDynamics, potential, dt: float,
                 kT: float = 1.0, mass: float = 1.0):
        self.dynamics = dynamics
        self.potential = potential
        self.dt = dt
        self.kT = kT
        self.mass = mass

    def step(self, state: ThermostatState) -> ThermostatState:
        q, p, xi, n_evals = state
        dt = self.dt

        grad_U = self.potential.gradient(q)
        n_evals += 1

        # Update thermostat
        xi_dot = self.dynamics.dxidt(state, grad_U)
        xi = xi + dt * xi_dot

        # Update momenta
        p_dot = self.dynamics.dpdt(ThermostatState(q, p, xi, n_evals), grad_U)
        p = p + dt * p_dot

        # Update positions
        q_dot = self.dynamics.dqdt(ThermostatState(q, p, xi, n_evals), grad_U)
        q = q + dt * q_dot

        return ThermostatState(q, p, xi, n_evals)


class AdaptiveRK45:
    """RK4 integrator for thermostat dynamics.

    4 force evaluations per step. Useful for non-separable or stiff dynamics.
    """

    def __init__(self, dynamics: ThermostatDynamics, potential, dt: float,
                 kT: float = 1.0, mass: float = 1.0):
        self.dynamics = dynamics
        self.potential = potential
        self.dt = dt
        self.kT = kT
        self.mass = mass

    def _pack(self, q, p, xi):
        return np.concatenate([q, p, xi])

    def _unpack(self, y, dim_q, dim_xi):
        q = y[:dim_q]
        p = y[dim_q:2 * dim_q]
        xi = y[2 * dim_q:]
        return q, p, xi

    def _rhs(self, y, dim_q, dim_xi, n_evals):
        q, p, xi = self._unpack(y, dim_q, dim_xi)
        grad_U = self.potential.gradient(q)
        n_evals += 1
        state = ThermostatState(q, p, xi, n_evals)
        dq = self.dynamics.dqdt(state, grad_U)
        dp = self.dynamics.dpdt(state, grad_U)
        dxi = self.dynamics.dxidt(state, grad_U)
        return np.concatenate([dq, dp, dxi]), n_evals

    def step(self, state: ThermostatState) -> ThermostatState:
        q, p, xi, n_evals = state
        dim_q = len(q)
        dim_xi = len(xi)
        y = self._pack(q, p, xi)

        dt = self.dt
        k1, n_evals = self._rhs(y, dim_q, dim_xi, n_evals)
        k2, n_evals = self._rhs(y + 0.5 * dt * k1, dim_q, dim_xi, n_evals)
        k3, n_evals = self._rhs(y + 0.5 * dt * k2, dim_q, dim_xi, n_evals)
        k4, n_evals = self._rhs(y + dt * k3, dim_q, dim_xi, n_evals)

        y_new = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        q_new, p_new, xi_new = self._unpack(y_new, dim_q, dim_xi)

        return ThermostatState(q_new, p_new, xi_new, n_evals)
