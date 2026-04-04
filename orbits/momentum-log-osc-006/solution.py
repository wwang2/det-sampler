"""Momentum-Dependent Log-Osc Thermostat (MLOSC).

Cross-pollinates the Log-Osc thermostat (bounded friction g(xi) = 2xi/(1+xi^2))
with momentum-dependent friction strength, where the coupling f(x) depends on
the normalized kinetic energy x = K/(dim*kT).

=== Variant A: Adaptive Friction (MLOSC-A) ===

Extended Hamiltonian:
    H_ext = U(q) + K(p) + Q*log(1 + xi^2)

Equations of motion (derived and verified via SymPy for dim=1,2):
    dq/dt = p/m
    dp/dt = -dU/dq - g(xi) * f(x) * p
    dxi/dt = (1/Q) * F(K, dim, kT)

where:
    g(xi) = 2*xi/(1+xi^2)  [bounded log-osc friction]
    x = K/(dim*kT)          [normalized kinetic energy]
    f(x) = 1 + alpha*x      [amplified friction when KE is large]
    K = sum(p_i^2)/(2m)

    F(K) = A*S - dim*kT + C*S^2/kT
    where S = sum(p_i^2)/m = 2K
    A = 1 - (dim+2)*alpha/(2*dim)
    C = alpha/(2*dim)

For alpha=0, this reduces to the standard Log-Osc thermostat.

The invariant measure is rho ~ exp(-H_ext/kT), which is canonical in (q,p)
and Cauchy-like in xi. Verified via SymPy Liouville equation for dim=1,2.

Physical interpretation: When kinetic energy exceeds the target (K > dim*kT),
the friction is amplified by factor (1 + alpha*K/(dim*kT)), providing stronger
thermostating. This prevents large kinetic energy excursions and should improve
barrier crossing in multi-modal potentials.

=== Variant B: Rippled Thermostat Potential (MLOSC-B) ===

Extended Hamiltonian:
    H_ext = U(q) + K(p) + Q*V(xi)
    where V(xi) = log(1+xi^2) + epsilon*cos(omega_xi*xi)

Equations of motion:
    dq/dt = p/m
    dp/dt = -dU/dq - V'(xi)*p
    dxi/dt = (1/Q)*(sum p_i^2/m - dim*kT)

V'(xi) = 2*xi/(1+xi^2) - epsilon*omega_xi*sin(omega_xi*xi)

Invariant measure: rho ~ exp(-H_ext/kT) -- exact for ANY V(xi).
The cosine perturbation adds multiple local energy barriers in the thermostat
variable, creating more complex chaotic dynamics that should break KAM tori.

References:
    - Nose, S. (1984). J. Chem. Phys. 81, 511.
    - Hoover, W. G. (1985). Phys. Rev. A, 31, 1695.
    - Log-Osc parent orbit: orbits/log-osc-001/ (Issue #3)
    - ESH dynamics: Versteeg (2021), NeurIPS
    - KAM theorem: https://en.wikipedia.org/wiki/Kolmogorov-Arnold-Moser_theorem
"""

import numpy as np
from research.eval.integrators import ThermostatState


def g_func(xi_val):
    """Log-osc friction: g(xi) = 2*xi/(1+xi^2). Bounded in [-1, 1]."""
    return 2.0 * xi_val / (1.0 + xi_val**2)


# ---------------------------------------------------------------------------
# Variant A: Adaptive Friction (momentum-dependent coupling)
# ---------------------------------------------------------------------------

class MomentumLogOsc:
    """Momentum-dependent Log-Osc Thermostat (MLOSC-A).

    The friction on momentum is scaled by f(x) = 1 + alpha * K/(dim*kT),
    where K is the kinetic energy. When alpha > 0, friction is stronger
    when the system has excess kinetic energy.

    The xi equation is modified to maintain the exact invariant measure.
    """

    name = "momentum_log_osc"

    def __init__(self, dim: int, kT: float = 1.0, mass: float = 1.0,
                 Q: float = 1.0, alpha: float = 0.5):
        self.dim = dim
        self.kT = kT
        self.mass = mass
        self.Q = Q
        self.alpha = alpha
        # Precompute coefficients for xi equation
        # F = A * S + B * kT + C * S^2 / kT
        # where S = sum(p^2)/m = 2K
        self.A_coeff = 1.0 - (dim + 2) * alpha / (2.0 * dim)
        self.B_coeff = -float(dim)
        self.C_coeff = alpha / (2.0 * dim)

    def initial_state(self, q0: np.ndarray, rng: np.random.Generator | None = None) -> ThermostatState:
        if rng is None:
            rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        xi0 = np.array([0.0])
        return ThermostatState(q0.copy(), p0, xi0, 0)

    def _kinetic(self, p: np.ndarray) -> float:
        """Kinetic energy K = sum(p^2)/(2m)."""
        return np.sum(p**2) / (2.0 * self.mass)

    def _f_factor(self, p: np.ndarray) -> float:
        """Friction multiplier f(x) = 1 + alpha * K/(dim*kT)."""
        K = self._kinetic(p)
        x = K / (self.dim * self.kT)
        return 1.0 + self.alpha * x

    def dqdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        return state.p / self.mass

    def dpdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        xi_val = state.xi[0]
        f = self._f_factor(state.p)
        return -grad_U - g_func(xi_val) * f * state.p

    def dxidt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        """Modified xi equation to preserve invariant measure.

        F = A*S + B*kT + C*S^2/kT where S = sum(p^2)/m.
        """
        S = np.sum(state.p**2) / self.mass  # = 2K
        F = (self.A_coeff * S
             + self.B_coeff * self.kT
             + self.C_coeff * S**2 / self.kT)
        return np.array([F / self.Q])


class MomentumLogOscVerlet:
    """Custom Velocity Verlet integrator for MLOSC-A.

    Uses the exp(-g(xi)*f(K)*dt/2) rescaling for momentum, with the
    f factor evaluated at the current kinetic energy.

    Splitting: same as Log-Osc but with f(K) multiplier on the friction.
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
        f = self.dynamics._f_factor(p)
        scale = np.exp(-gxi * f * half_dt)
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
        f = self.dynamics._f_factor(p)
        scale = np.exp(-gxi * f * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale

        # Half-step thermostat
        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        self._cached_grad_U = grad_U
        return ThermostatState(q, p, xi, n_evals)


# ---------------------------------------------------------------------------
# Variant B: Rippled Thermostat Potential
# ---------------------------------------------------------------------------

class RippledLogOsc:
    """Rippled Log-Osc Thermostat (MLOSC-B).

    Uses V(xi) = log(1+xi^2) + epsilon*cos(omega_xi*xi) as the
    thermostat potential. The cosine perturbation creates multiple
    local energy barriers in the thermostat landscape, enhancing
    chaotic mixing.

    The friction is V'(xi) = 2*xi/(1+xi^2) - epsilon*omega_xi*sin(omega_xi*xi).
    """

    name = "rippled_log_osc"

    def __init__(self, dim: int, kT: float = 1.0, mass: float = 1.0,
                 Q: float = 1.0, epsilon: float = 0.3, omega_xi: float = 2.0):
        self.dim = dim
        self.kT = kT
        self.mass = mass
        self.Q = Q
        self.epsilon = epsilon
        self.omega_xi = omega_xi

    def _v_prime(self, xi_val: float) -> float:
        """V'(xi) = 2*xi/(1+xi^2) - epsilon*omega_xi*sin(omega_xi*xi)."""
        return (2.0 * xi_val / (1.0 + xi_val**2)
                - self.epsilon * self.omega_xi * np.sin(self.omega_xi * xi_val))

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
        return -grad_U - self._v_prime(xi_val) * state.p

    def dxidt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        kinetic = np.sum(state.p**2) / self.mass
        return np.array([(kinetic - self.dim * self.kT) / self.Q])


class RippledLogOscVerlet:
    """Custom Velocity Verlet integrator for MLOSC-B.

    Uses exp(-V'(xi)*dt/2) rescaling for momentum.
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

        if self._cached_grad_U is not None:
            grad_U = self._cached_grad_U
        else:
            grad_U = self.potential.gradient(q)
            n_evals += 1

        # Half-step thermostat
        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        # Half-step momenta: V'(xi) friction + kick
        vp = self.dynamics._v_prime(xi[0])
        scale = np.exp(-vp * half_dt)
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
        vp = self.dynamics._v_prime(xi[0])
        scale = np.exp(-vp * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale

        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        self._cached_grad_U = grad_U
        return ThermostatState(q, p, xi, n_evals)


# ---------------------------------------------------------------------------
# Variant C: Combined (momentum-dependent + rippled)
# ---------------------------------------------------------------------------

class CombinedMLogOsc:
    """Combined Momentum-Dependent Rippled Log-Osc (MLOSC-C).

    Combines both innovations:
    - Rippled thermostat potential V(xi) for chaotic thermostat dynamics
    - Momentum-dependent friction f(x) for adaptive coupling

    NOTE: This combination does NOT have an exact invariant measure because
    f(x) requires the log-osc specific H_ext, while the rippled V(xi) changes
    the H_ext. The momentum-dependent modification was derived specifically
    for V(xi) = log(1+xi^2).

    To have an exact measure with rippled V(xi), we would need to re-derive
    the F equation for the general V case. For now, we only use the rippled
    variant (MLOSC-B) or the adaptive variant (MLOSC-A) separately.
    """
    pass  # Not implemented -- using A and B separately
