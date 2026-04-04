"""Dual-Bath Thermostat: NHC(2) with Hamiltonian Rotation Enhancement.

A Nose-Hoover Chain of length 2 augmented with a measure-preserving
Hamiltonian rotation in the (xi, eta) thermostat subspace. The chain
coupling provides ergodicity improvement (as in standard NHC), while
the rotation creates additional mixing in thermostat space.

Equations of motion:
    dq/dt   = p / m
    dp/dt   = -dU/dq - xi * p
    dxi/dt  = (1/Q_xi) * (|p|^2/m - dim*kT) - eta*xi + alpha*sqrt(Q_eta/Q_xi)*eta
    deta/dt = (1/Q_eta) * (Q_xi*xi^2 - kT) - alpha*sqrt(Q_xi/Q_eta)*xi

Invariant measure:
    rho(q, p, xi, eta) ~ exp(-U(q)/kT - |p|^2/(2m*kT) - Q_xi*xi^2/(2*kT) - Q_eta*eta^2/(2*kT))

Verified symbolically (see verify_nhc_rotation.py).

The dynamics combine:
1. Standard NHC(2) chain coupling: -eta*xi in dxi/dt, Q_xi*xi^2 driving in deta/dt
2. Measure-preserving Hamiltonian rotation in (xi, eta) space

Physical interpretation:
- xi is the primary thermostat controlling kinetic temperature
- eta is the secondary thermostat that thermostatizes xi (chain coupling)
- The rotation adds a Hamiltonian flow in (xi, eta) that enhances mixing
- alpha=0 recovers exactly NHC(M=2)

References:
    - Nose (1984), Hoover (1985): original Nose-Hoover thermostat
    - Martyna et al. (1992): Nose-Hoover Chains
    - Patra & Bhattacharya (2015): dual thermostat ideas
    - Fukuda & Nakamura (2002): coupled Nose-Hoover equations
"""

import numpy as np
from research.eval.integrators import ThermostatState


class DualBathThermostat:
    """NHC(2) thermostat with Hamiltonian rotation enhancement.

    Combines the proven ergodicity of Nose-Hoover Chains with a
    measure-preserving rotation that creates additional mixing in
    the thermostat (xi, eta) subspace.
    """

    name = "dual_bath"

    def __init__(self, dim: int, kT: float = 1.0, mass: float = 1.0,
                 Q_xi: float = 1.0, Q_eta: float = 1.0, alpha: float = 1.0):
        self.dim = dim
        self.kT = kT
        self.mass = mass
        self.Q_xi = Q_xi
        self.Q_eta = Q_eta
        self.alpha = alpha
        # Precompute rotation coefficients
        self._rot_xi = alpha * np.sqrt(Q_eta / Q_xi)
        self._rot_eta = alpha * np.sqrt(Q_xi / Q_eta)

    def initial_state(self, q0: np.ndarray, rng: np.random.Generator | None = None) -> ThermostatState:
        if rng is None:
            rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        # xi[0] = xi (primary thermostat), xi[1] = eta (secondary)
        xi0 = np.array([0.0, 0.0])
        return ThermostatState(q0.copy(), p0, xi0, 0)

    def dqdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        return state.p / self.mass

    def dpdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        return -grad_U - state.xi[0] * state.p

    def dxidt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        xi_val = state.xi[0]
        eta_val = state.xi[1]
        kinetic = np.sum(state.p ** 2) / self.mass

        # NHC(2) chain coupling + Hamiltonian rotation
        dxi = ((kinetic - self.dim * self.kT) / self.Q_xi
               - eta_val * xi_val                    # NHC chain coupling
               + self._rot_xi * eta_val)             # rotation

        deta = ((self.Q_xi * xi_val**2 - self.kT) / self.Q_eta  # NHC chain driving
                - self._rot_eta * xi_val)             # rotation

        return np.array([dxi, deta])


class DualBathVelocityVerlet:
    """Velocity Verlet integrator for the dual-bath thermostat.

    Since only xi[0] couples to momentum, uses standard exp(-xi[0]*dt/2) rescaling.

    Splitting scheme (1 force eval per step via FSAL):
      1. Half-step thermostat:  (xi, eta) += (dt/2) * d(xi,eta)/dt
      2. Half-step momenta:     p *= exp(-xi*dt/2); p -= (dt/2)*grad_U
      3. Full-step positions:   q += dt * p/m
      4. Recompute forces
      5. Half-step momenta:     p -= (dt/2)*grad_U; p *= exp(-xi*dt/2)
      6. Half-step thermostat:  (xi, eta) += (dt/2) * d(xi,eta)/dt
    """

    def __init__(self, dynamics, potential, dt: float,
                 kT: float = 1.0, mass: float = 1.0):
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
        dyn = self.dynamics

        # Get gradient (FSAL cache)
        if self._cached_grad_U is not None:
            grad_U = self._cached_grad_U
        else:
            grad_U = self.potential.gradient(q)
            n_evals += 1

        # 1. Half-step thermostat variables
        xi_dot = dyn.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        # 2. Half-step momenta: analytical friction rescaling + force kick
        scale = np.exp(-xi[0] * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale
        p = p - half_dt * grad_U

        # 3. Full-step positions
        q = q + dt * p / self.mass

        # NaN check
        if np.any(np.isnan(q)) or np.any(np.isnan(p)):
            self._cached_grad_U = None
            return ThermostatState(q, p, xi, n_evals)

        # 4. Recompute forces
        grad_U = self.potential.gradient(q)
        n_evals += 1

        # 5. Half-step momenta: force kick + analytical friction rescaling
        p = p - half_dt * grad_U
        scale = np.exp(-xi[0] * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale

        # 6. Half-step thermostat variables
        xi_dot = dyn.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        # Cache for FSAL
        self._cached_grad_U = grad_U

        return ThermostatState(q, p, xi, n_evals)
