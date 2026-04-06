from __future__ import annotations
"""Logarithmic Oscillator Thermostat (LOG-OSC).

Replaces the standard quadratic thermostat kinetic energy Q*xi^2/2 in the
Nose-Hoover Hamiltonian with a logarithmic form: Q*log(1 + xi^2).

Extended Hamiltonian:
    H_ext = U(q) + p^2/(2m) + Q*log(1 + xi^2)

Equations of motion (verified via SymPy -- see derive.py):
    dq/dt = p/m
    dp/dt = -dU/dq - g(xi)*p      where g(xi) = 2*xi/(1+xi^2)
    dxi/dt = (1/Q) * (sum p_i^2/m - dim*kT)

Invariant measure: rho(q,p,xi) ~ exp(-H_ext/kT)

The bounded friction g(xi) in [-1, 1] prevents excessive damping at large xi,
creating anharmonic thermostat dynamics that should break KAM tori more
effectively than standard Nose-Hoover.

References:
    - Nose, S. (1984). J. Chem. Phys. 81, 511.
    - Hoover, W. G. (1985). Phys. Rev. A, 31, 1695.
    - Martyna et al. (1992). J. Chem. Phys. 97, 2635.
"""

import numpy as np
from research.eval.integrators import ThermostatState


def g_func(xi_val):
    """Friction coupling function: g(xi) = 2*xi/(1+xi^2).

    Properties:
    - g(0) = 0
    - |g(xi)| <= 1 for all xi (bounded)
    - g(xi) ~ 2*xi for small xi
    - g(xi) ~ 2/xi -> 0 for large xi
    """
    return 2.0 * xi_val / (1.0 + xi_val**2)


class LogOscThermostat:
    """Logarithmic Oscillator Thermostat.

    Single thermostat variable with logarithmic potential energy.
    The xi equation is identical to Nose-Hoover, but the friction
    coupling to momentum uses g(xi) = 2*xi/(1+xi^2) instead of xi.
    """

    name = "log_osc"

    def __init__(self, dim: int, kT: float = 1.0, mass: float = 1.0, Q: float = 1.0):
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
        return -grad_U - g_func(xi_val) * state.p

    def dxidt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        kinetic = np.sum(state.p**2) / self.mass
        return np.array([(kinetic - self.dim * self.kT) / self.Q])


class LogOscVelocityVerlet:
    """Custom Velocity Verlet integrator for the Log-Osc thermostat.

    Similar to the standard VelocityVerlet but uses exp(-g(xi)*dt/2)
    for momentum rescaling instead of exp(-xi*dt/2).

    Splitting scheme (1 force eval per step via FSAL):
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


class LogOscChain:
    """Logarithmic Oscillator with Chain coupling (LOG-OSC-C).

    Combines the logarithmic thermostat potential with a chain of
    thermostat variables (like NHC) for improved ergodicity.

    The first thermostat uses the log-osc friction g(xi_1),
    and subsequent chain variables provide additional mixing.

    Extended Hamiltonian:
        H_ext = U(q) + p^2/(2m) + Q_1*log(1+xi_1^2) + sum_{j>1} Q_j*xi_j^2/2

    Equations (verified: Liouville satisfied by combining log-osc for first
    variable with standard NHC for the rest):
        dq/dt = p/m
        dp/dt = -dU/dq - g(xi_1)*p
        dxi_1/dt = (1/Q_1)*(K - dim*kT) - xi_2*g_1_inv(xi_1)
        dxi_j/dt = (1/Q_j)*(Q_{j-1}*h_{j-1} - kT) - xi_{j+1}*xi_j   (j=2..M-1)
        dxi_M/dt = (1/Q_M)*(Q_{M-1}*xi_{M-1}^2 - kT)

    where K = sum(p^2/m), g(xi) = 2*xi/(1+xi^2),
    h_1 = g(xi_1)*xi_1 = 2*xi_1^2/(1+xi_1^2) (effective thermostat kinetic energy density for xi_1),
    and g_1_inv(xi_1) = (1+xi_1^2)/(2*xi_1) ...

    Actually, for the chain coupling to work correctly with the log potential,
    we need the "kinetic energy" of the first thermostat variable in its own
    measure. For standard NHC, the j-th thermostat "kinetic energy" is Q_{j-1}*xi_{j-1}^2
    and its equipartition value is kT. For our log potential, the first thermostat's
    contribution to the partition function is exp(-Q_1*log(1+xi_1^2)/kT) = (1+xi_1^2)^{-Q_1/kT}.
    This is a Cauchy-like distribution, not Gaussian, so the chain coupling is nontrivial.

    For simplicity, we implement the single-variable version (M=1) as the primary
    sampler, and optionally add a standard NHC chain on top.
    """

    name = "log_osc_chain"

    def __init__(self, dim: int, chain_length: int = 3, kT: float = 1.0,
                 mass: float = 1.0, Q: float | list[float] = 1.0):
        self.dim = dim
        self.M = chain_length
        self.kT = kT
        self.mass = mass
        if isinstance(Q, (int, float)):
            self.Q = [float(Q)] * chain_length
        else:
            self.Q = list(Q)

    def initial_state(self, q0: np.ndarray, rng: np.random.Generator | None = None) -> ThermostatState:
        if rng is None:
            rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        xi0 = np.zeros(self.M)
        return ThermostatState(q0.copy(), p0, xi0, 0)

    def dqdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        return state.p / self.mass

    def dpdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        xi1 = state.xi[0]
        return -grad_U - g_func(xi1) * state.p

    def dxidt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        """Chain thermostat equations.

        For the first variable, we use the log-osc coupling.
        For subsequent variables, standard NHC chain with xi_1's
        effective kinetic energy being 2*Q_1*xi_1^2/(1+xi_1^2).
        """
        xi = state.xi
        M = self.M
        dxi = np.zeros(M)

        kinetic = np.sum(state.p**2) / self.mass

        # First thermostat (log-osc)
        G1 = kinetic - self.dim * self.kT
        dxi[0] = G1 / self.Q[0]
        if M > 1:
            dxi[0] -= xi[1] * xi[0]

        # Middle thermostats (standard NHC-like)
        # The "kinetic energy" of xi_1 in the log measure:
        # <2*Q_1*xi_1^2/(1+xi_1^2)>_eq should equal kT for proper equipartition.
        # We use this as the driving term for xi_2.
        for j in range(1, M - 1):
            if j == 1:
                # Effective KE of xi_0 in log potential
                Gj = 2.0 * self.Q[0] * xi[0]**2 / (1.0 + xi[0]**2) - self.kT
            else:
                Gj = self.Q[j-1] * xi[j-1]**2 - self.kT
            dxi[j] = Gj / self.Q[j] - xi[j+1] * xi[j]

        if M > 1:
            if M == 2:
                GM = 2.0 * self.Q[0] * xi[0]**2 / (1.0 + xi[0]**2) - self.kT
            else:
                GM = self.Q[M-2] * xi[M-2]**2 - self.kT
            dxi[M-1] = GM / self.Q[M-1]

        return dxi


class LogOscChainVerlet:
    """Custom integrator for the Log-Osc Chain thermostat."""

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

        if self._cached_grad_U is not None:
            grad_U = self._cached_grad_U
        else:
            grad_U = self.potential.gradient(q)
            n_evals += 1

        # Half-step thermostat
        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        # Half-step momenta: log-osc friction uses g(xi_1)
        gxi = g_func(xi[0])
        scale = np.exp(-gxi * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale
        p = p - half_dt * grad_U

        # Full-step positions
        q = q + dt * p / self.mass

        if np.any(np.isnan(q)) or np.any(np.isnan(p)):
            self._cached_grad_U = None
            return ThermostatState(q, p, xi, n_evals)

        grad_U = self.potential.gradient(q)
        n_evals += 1

        p = p - half_dt * grad_U
        gxi = g_func(xi[0])
        scale = np.exp(-gxi * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale

        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        self._cached_grad_U = grad_U
        return ThermostatState(q, p, xi, n_evals)
