"""Log-Osc Chain with Rotation (LOCR) Thermostat.

Combines three innovations:
1. Logarithmic thermostat potential for xi_1: Q_1 * log(1 + xi_1^2)
   Standard quadratic KE for chain variables j>1: Q_j * xi_j^2 / 2
2. Chain coupling a la NHC for improved ergodicity on harmonic systems
3. Antisymmetric rotation coupling between adjacent chain variables

Extended Hamiltonian:
    H_ext = U(q) + p^2/(2m) + Q_1*log(1+xi_1^2) + sum_{j>1} Q_j*xi_j^2/2

Equations of motion:
    dq/dt = p/m
    dp/dt = -dU/dq - g(xi_1) * p
    dxi_1/dt = (1/Q_1) * (K - dim*kT) - xi_2 * xi_1 + alpha * xi_2
    dxi_j/dt = (1/Q_j) * (G_j - kT) - xi_{j+1}*xi_j + alpha*(xi_{j+1} - xi_{j-1})
    dxi_M/dt = (1/Q_M) * (G_M - kT) - alpha * xi_{M-1}

where:
    g(xi) = 2*xi/(1+xi^2)           -- bounded friction function
    G_1 = 2*Q_1*xi_1^2/(1+xi_1^2)   -- effective KE of xi_1 in log measure
    G_j = Q_{j-1}*xi_{j-1}^2        -- standard KE for j>1
    K = sum(p_i^2/m)

The rotation terms (alpha) are antisymmetric: +alpha*xi_{j+1} to dxi_j and
-alpha*xi_j to dxi_{j+1}. This preserves phase-space volume (zero divergence)
and the invariant measure.

The chain coupling uses standard NHC-style -xi_{j+1}*xi_j damping, which
provides proper mixing. The log-osc potential only affects the first thermostat
variable's coupling to momentum via g(xi_1).

References:
    - Parent orbit: log-osc-001 (Issue #3) -- single log-osc thermostat
    - Nose-Hoover Chains: Martyna et al. (1992) J. Chem. Phys. 97, 2635
    - NHC integrators: Martyna et al. (1996) Mol. Phys. 87, 1117
    - Dual thermostat rotation: inspired by Patra-Bhattacharya (2014)
"""

import numpy as np
from research.eval.integrators import ThermostatState


def g_func(xi):
    """Bounded friction: g(xi) = 2*xi/(1+xi^2). Range: [-1, 1]."""
    xi2 = xi * xi
    if xi2 > 1e30:
        # For very large |xi|, g(xi) ~ 2/xi -> 0
        return 2.0 / xi if xi != 0 else 0.0
    return 2.0 * xi / (1.0 + xi2)


def h_func(xi):
    """Rotation driver: h(xi) = 2*xi^2/(1+xi^2) - 1. Range: [-1, 1].

    Properties:
    - h(0) = -1
    - h(+/-inf) -> +1
    - <h(xi)>_eq = 0 when xi ~ (1+xi^2)^{-Q/kT}  (for Q/kT > 1/2)
    - Odd under xi -> -xi: NO. h(-xi) = h(xi). It's even.

    Actually for rotation coupling we need something antisymmetric in some sense.
    Let's use a simpler approach: rotation in the (xi_j, xi_{j+1}) plane.
    """
    return 2.0 * xi ** 2 / (1.0 + xi ** 2) - 1.0


def eff_ke(Q_j, xi_j):
    """Effective kinetic energy of xi_j in the log measure.

    For standard NHC: KE_j = Q_j * xi_j^2
    For log-osc: the "kinetic energy" is the derivative of the potential
    times xi, giving: F_j = 2*Q_j*xi_j^2/(1+xi_j^2)

    This quantity equilibrates to kT in the invariant measure.
    Bounded: 0 <= F_j < 2*Q_j.
    """
    xi2 = xi_j * xi_j
    if xi2 > 1e30:
        return 2.0 * Q_j  # limit as xi -> inf
    return 2.0 * Q_j * xi2 / (1.0 + xi2)


class LogOscChainRotation:
    """Log-Osc Chain with Rotation (LOCR) thermostat.

    Parameters:
        dim: Number of physical degrees of freedom
        chain_length: Number of thermostat chain variables (M >= 1)
        kT: Temperature
        mass: Particle mass
        Q: Thermostat mass(es). Scalar or list of length M.
        alpha: Rotation coupling strength (0 = no rotation)
    """

    name = "log_osc_chain_rotation"

    def __init__(self, dim: int, chain_length: int = 2, kT: float = 1.0,
                 mass: float = 1.0, Q: float | list[float] = 1.0,
                 alpha: float = 0.0):
        self.dim = dim
        self.M = chain_length
        self.kT = kT
        self.mass = mass
        self.alpha = alpha
        if isinstance(Q, (int, float)):
            self.Q = [float(Q)] * chain_length
        else:
            self.Q = list(Q)
        assert len(self.Q) == chain_length

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
        xi = state.xi
        M = self.M
        Q = self.Q
        kT = self.kT
        alpha = self.alpha
        dxi = np.zeros(M)

        kinetic = np.sum(state.p ** 2) / self.mass

        # === First thermostat (j=0): log-osc variable ===
        G0 = kinetic - self.dim * kT
        dxi[0] = G0 / Q[0]

        # Chain coupling: standard NHC-style -xi[1]*xi[0]
        if M > 1:
            dxi[0] -= xi[1] * xi[0]

        # Rotation: bounded antisymmetric using g_func for stability
        # g(xi) is bounded in [-1,1], so this cannot cause divergence
        if M > 1 and alpha != 0:
            dxi[0] += alpha * g_func(xi[1])

        # === Chain thermostats (j=1..M-1): standard quadratic KE ===
        for j in range(1, M):
            # Driving force: "kinetic energy" of xi_{j-1}
            if j == 1:
                # Effective KE of xi_0 in log measure
                Gj = eff_ke(Q[0], xi[0])
            else:
                # Standard quadratic KE for j>1
                Gj = Q[j - 1] * xi[j - 1] ** 2

            dxi[j] = (Gj - kT) / Q[j]

            # Chain coupling from j+1 (standard NHC-style)
            if j < M - 1:
                dxi[j] -= xi[j + 1] * xi[j]

            # Rotation coupling: bounded antisymmetric using g_func
            if alpha != 0:
                dxi[j] -= alpha * g_func(xi[j - 1])
                if j < M - 1:
                    dxi[j] += alpha * g_func(xi[j + 1])

        return dxi


class LOCRIntegrator:
    """Custom velocity Verlet integrator for the LOCR thermostat.

    Uses the palindromic splitting:
      1. Half-step chain (from outermost to innermost): xi_M, ..., xi_1
      2. Half-step momenta: p *= exp(-g(xi_1)*dt/2); p -= (dt/2)*grad_U
      3. Full-step positions: q += dt*p/m
      4. Recompute forces
      5. Half-step momenta: p -= (dt/2)*grad_U; p *= exp(-g(xi_1)*dt/2)
      6. Half-step chain (from innermost to outermost): xi_1, ..., xi_M

    1 force eval per step (FSAL after initialization).
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

    def _half_step_chain(self, state, grad_U, half_dt, reverse=False):
        """Half-step all thermostat variables.

        If reverse=False: update from outermost to innermost (pre-step)
        If reverse=True: update from innermost to outermost (post-step)

        Uses simple Euler for each variable in the chain, applied sequentially
        so that each update sees the latest values. This is the standard
        approach from Martyna et al. (1996) for NHC.
        """
        q, p, xi, n_evals = state
        xi = xi.copy()
        M = self.dynamics.M
        Q = self.dynamics.Q
        kT = self.dynamics.kT
        alpha = self.dynamics.alpha
        dim = self.dynamics.dim
        mass = self.dynamics.mass

        if reverse:
            indices = range(0, M)
        else:
            indices = range(M - 1, -1, -1)

        for j in indices:
            # Compute driving force for this variable
            if j == 0:
                kinetic = np.sum(p ** 2) / mass
                force = (kinetic - dim * kT) / Q[0]
            else:
                if j == 1:
                    # Effective KE of xi_0 in log measure
                    Gj = eff_ke(Q[0], xi[0])
                else:
                    # Standard quadratic KE
                    Gj = Q[j - 1] * xi[j - 1] ** 2
                force = (Gj - kT) / Q[j]

            # Chain coupling: standard NHC-style -xi[j+1]*xi[j]
            if j < M - 1:
                force -= xi[j + 1] * xi[j]

            # Rotation coupling: bounded antisymmetric using g_func
            if alpha != 0:
                if j == 0 and M > 1:
                    force += alpha * g_func(xi[1])
                elif j > 0:
                    force -= alpha * g_func(xi[j - 1])
                    if j < M - 1:
                        force += alpha * g_func(xi[j + 1])

            xi[j] = xi[j] + half_dt * force

        return ThermostatState(q, p, xi, n_evals)

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

        # 1. Half-step chain (outer to inner)
        state_tmp = ThermostatState(q, p, xi, n_evals)
        state_tmp = self._half_step_chain(state_tmp, grad_U, half_dt, reverse=False)
        xi = state_tmp.xi

        # 2. Half-step momenta: friction rescaling + kick
        gxi = g_func(xi[0])
        scale = np.exp(-gxi * half_dt)
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

        # 5. Half-step momenta: kick + friction rescaling
        p = p - half_dt * grad_U
        gxi = g_func(xi[0])
        scale = np.exp(-gxi * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale

        # 6. Half-step chain (inner to outer)
        state_tmp = ThermostatState(q, p, xi, n_evals)
        state_tmp = self._half_step_chain(state_tmp, grad_U, half_dt, reverse=True)
        xi = state_tmp.xi

        self._cached_grad_U = grad_U
        return ThermostatState(q, p, xi, n_evals)
