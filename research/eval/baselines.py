"""Baseline thermostat samplers: Nose-Hoover and Nose-Hoover Chain.

These serve as reference implementations to validate the evaluator
and establish baseline metrics to beat.
"""

from __future__ import annotations
import numpy as np
from .integrators import ThermostatState


class NoseHoover:
    """Single Nose-Hoover thermostat.

    Equations of motion:
        dq/dt = p / m
        dp/dt = -dU/dq - xi * p
        dxi/dt = (1/Q) * (p^2/m - dim * kT)

    Invariant measure: rho(q, p, xi) ~ exp(-U(q)/kT - p^2/(2m*kT) - Q*xi^2/(2*kT))
    Known non-ergodic for 1D harmonic oscillator.
    """

    name = "nose_hoover"

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
        return -grad_U - state.xi[0] * state.p

    def dxidt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        kinetic = np.sum(state.p ** 2) / self.mass
        return np.array([(kinetic - self.dim * self.kT) / self.Q])


class NoseHooverChain:
    """Nose-Hoover Chain thermostat (Martyna et al. 1992).

    Chain of M thermostat variables for improved ergodicity.

    Equations of motion:
        dq/dt = p / m
        dp/dt = -dU/dq - xi_1 * p
        dxi_1/dt = (1/Q_1) * (p^2/m - dim*kT) - xi_2 * xi_1
        dxi_j/dt = (1/Q_j) * (Q_{j-1}*xi_{j-1}^2 - kT) - xi_{j+1}*xi_j   (j=2..M-1)
        dxi_M/dt = (1/Q_M) * (Q_{M-1}*xi_{M-1}^2 - kT)
    """

    name = "nose_hoover_chain"

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
        return -grad_U - state.xi[0] * state.p

    def dxidt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        xi = state.xi
        M = self.M
        dxi = np.zeros(M)

        # First thermostat: driven by kinetic energy
        kinetic = np.sum(state.p ** 2) / self.mass
        G1 = kinetic - self.dim * self.kT
        dxi[0] = G1 / self.Q[0]
        if M > 1:
            dxi[0] -= xi[1] * xi[0]

        # Middle thermostats
        for j in range(1, M - 1):
            Gj = self.Q[j - 1] * xi[j - 1] ** 2 - self.kT
            dxi[j] = Gj / self.Q[j] - xi[j + 1] * xi[j]

        # Last thermostat
        if M > 1:
            GM = self.Q[M - 2] * xi[M - 2] ** 2 - self.kT
            dxi[M - 1] = GM / self.Q[M - 1]

        return dxi


BASELINE_SAMPLERS = {
    "nose_hoover": NoseHoover,
    "nose_hoover_chain": NoseHooverChain,
}
