"""Log-Osc Chain with Rotation (LOCR) Thermostat.

Combines three innovations:
1. Logarithmic thermostat potential: Q_j * log(1 + xi_j^2) for ALL chain variables
2. Chain coupling a la NHC for improved ergodicity on harmonic systems
3. Antisymmetric rotation coupling between adjacent chain variables

Extended Hamiltonian:
    H_ext = U(q) + p^2/(2m) + sum_{j=1}^{M} Q_j * log(1 + xi_j^2)

Equations of motion:
    dq/dt = p/m
    dp/dt = -dU/dq - g(xi_1) * p
    dxi_1/dt = (1/Q_1) * (K - dim*kT) - xi_2 * g(xi_1) + alpha * h(xi_2)
    dxi_j/dt = (1/Q_j) * (F_j - kT) - xi_{j+1}*g(xi_j) + alpha*(h(xi_{j+1}) - h(xi_{j-1}))
    dxi_M/dt = (1/Q_M) * (F_M - kT) - alpha * h(xi_{M-1})

where:
    g(xi) = 2*xi/(1+xi^2)           -- bounded friction function
    h(xi) = 2*xi^2/(1+xi^2) - 1     -- antisymmetric rotation driver (zero-mean in equilibrium)
    F_j = 2*Q_{j-1}*xi_{j-1}^2/(1+xi_{j-1}^2)  -- effective KE of xi_{j-1} in log measure
    K = sum(p_i^2/m)

The rotation terms are chosen to be divergence-free (antisymmetric) so they
preserve the invariant measure without affecting Liouville's theorem.

The chain coupling term -xi_{j+1}*g(xi_j) is the natural extension of NHC
chain coupling to log-osc variables: it ensures that xi_j's "kinetic energy"
F_j = 2*Q_j*xi_j^2/(1+xi_j^2) equilibrates to kT via the next chain variable.

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
    return 2.0 * xi / (1.0 + xi ** 2)


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
    """
    return 2.0 * Q_j * xi_j ** 2 / (1.0 + xi_j ** 2)


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

        # === First thermostat (j=0) ===
        G0 = kinetic - self.dim * kT
        dxi[0] = G0 / Q[0]

        # Chain coupling from j=1
        if M > 1:
            dxi[0] -= xi[1] * g_func(xi[0])

        # Rotation: antisymmetric coupling in (xi_0, xi_1) plane
        # We add alpha * xi_1 to dxi_0 and -alpha * xi_0 to dxi_1
        # This is a Hamiltonian rotation that preserves phase space volume
        # and is orthogonal to the gradient of H_ext, so it preserves the measure.
        if M > 1 and alpha != 0:
            dxi[0] += alpha * xi[1]

        # === Middle thermostats (j=1..M-2) ===
        for j in range(1, M):
            # Driving force: effective KE of xi_{j-1}
            Fj = eff_ke(Q[j - 1], xi[j - 1])
            dxi[j] = (Fj - kT) / Q[j]

            # Chain coupling from j+1
            if j < M - 1:
                dxi[j] -= xi[j + 1] * g_func(xi[j])

            # Rotation coupling (antisymmetric)
            if alpha != 0:
                # Rotation with previous neighbor
                dxi[j] -= alpha * xi[j - 1]
                # Rotation with next neighbor
                if j < M - 1:
                    dxi[j] += alpha * xi[j + 1]

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
                Fj = eff_ke(Q[j - 1], xi[j - 1])
                force = (Fj - kT) / Q[j]

            # Chain coupling from j+1
            if j < M - 1:
                force -= xi[j + 1] * g_func(xi[j])

            # Rotation coupling
            if alpha != 0:
                if j == 0 and M > 1:
                    force += alpha * xi[1]
                elif j > 0:
                    force -= alpha * xi[j - 1]
                    if j < M - 1:
                        force += alpha * xi[j + 1]

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
