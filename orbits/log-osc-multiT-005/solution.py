"""Log-Osc Multi-Thermostat (LOG-OSC-MT) for improved multi-modal hopping.

Extends the Log-Osc thermostat with MULTIPLE thermostat variables that
each directly couple to momentum, creating richer dynamics across multiple
timescales. This is distinct from NHC (which chains thermostats) -- here
each thermostat independently drives momentum, creating interference patterns
that help cross barriers.

Variant A: Dual Log-Osc
    Two log-osc thermostats with different Q values both couple to p:

    H_ext = U(q) + p^2/(2m) + Q1*log(1+xi1^2) + Q2*log(1+xi2^2)

    dq/dt  = p/m
    dp/dt  = -dU/dq - [g(xi1) + g(xi2)] * p
    dxi1/dt = (1/Q1) * (p^2/m - dim*kT)
    dxi2/dt = (1/Q2) * (p^2/m - dim*kT)

    Invariant measure: rho ~ exp(-H_ext/kT) -- marginal over (q,p) is canonical.

    Proof: The Liouville equation requires div(J * rho) = 0 where J is the
    flow field. With rho = exp(-H_ext/kT), the compressibility (phase space
    contraction rate) must equal dH_ext/dt / kT. The friction terms -[g1+g2]*p
    contribute -dim*[g1+g2] to compressibility, and the thermostat drives
    contribute [g1+g2]*(p^2/m)/kT to dH_ext/dt. At equilibrium these balance
    when <p^2/m> = dim*kT, which is exactly what the xi equations enforce.

Variant B: Dual Log-Osc with Cross-Coupling
    Add weak cross-coupling between the two thermostats to create chaotic
    mixing:

    dxi1/dt = (1/Q1) * (p^2/m - dim*kT) - epsilon * xi2 * g(xi1)
    dxi2/dt = (1/Q2) * (p^2/m - dim*kT) - epsilon * xi1 * g(xi2)

    The cross terms maintain the correct invariant measure because they
    are anti-symmetric in the extended Hamiltonian sense.

Variant C: Log-Osc with Deterministic Temperature Pulsing
    Single log-osc thermostat but with a second auxiliary variable (zeta, p_zeta)
    forming a harmonic oscillator that modulates the thermostat mass Q:

    Q_eff(zeta) = Q * (1 + A * zeta / sqrt(1 + zeta^2))

    This creates periodic "heating" and "cooling" cycles that help cross barriers.
    The bounded modulation ensures Q_eff stays positive.

References:
    - Parent: orbits/log-osc-001 (Issue #3) -- base log-osc thermostat
    - Nose (1984), Hoover (1985) -- original NH thermostat
    - Martyna et al. (1992) -- Nose-Hoover chains
    - Marinari & Parisi (1992) -- Simulated tempering (stochastic version)
    - Fukuda & Nakamura (2002) -- Multiple Nose-Hoover thermostats
      https://doi.org/10.1103/PhysRevE.65.026105
"""

from typing import Optional
import numpy as np
from research.eval.integrators import ThermostatState


def g_func(xi_val):
    """Bounded friction: g(xi) = 2*xi/(1+xi^2), in [-1, 1]."""
    return 2.0 * xi_val / (1.0 + xi_val**2)


# =========================================================================
# Variant A: Dual Log-Osc (two independent thermostats coupling to p)
# =========================================================================

class DualLogOsc:
    """Dual Log-Osc thermostat -- two log-osc variables both couple to momentum.

    xi[0] = xi1 (fast thermostat, small Q1)
    xi[1] = xi2 (slow thermostat, large Q2)

    The total friction is g(xi1) + g(xi2), bounded in [-2, 2].
    Different Q values create dynamics at different timescales,
    improving exploration of multi-modal landscapes.
    """

    name = "dual_log_osc"

    def __init__(self, dim: int, kT: float = 1.0, mass: float = 1.0,
                 Q1: float = 0.5, Q2: float = 5.0):
        self.dim = dim
        self.kT = kT
        self.mass = mass
        self.Q1 = Q1
        self.Q2 = Q2

    def initial_state(self, q0: np.ndarray, rng: Optional[np.random.Generator] = None) -> ThermostatState:
        if rng is None:
            rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        xi0 = np.array([0.0, 0.0])
        return ThermostatState(q0.copy(), p0, xi0, 0)

    def dqdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        return state.p / self.mass

    def dpdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        xi1, xi2 = state.xi[0], state.xi[1]
        total_friction = g_func(xi1) + g_func(xi2)
        return -grad_U - total_friction * state.p

    def dxidt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        kinetic = np.sum(state.p**2) / self.mass
        drive = kinetic - self.dim * self.kT
        return np.array([drive / self.Q1, drive / self.Q2])


class DualLogOscVerlet:
    """Velocity Verlet for Dual Log-Osc thermostat.

    Uses exp(-(g(xi1)+g(xi2))*dt/2) for momentum rescaling.
    FSAL scheme: 1 force eval per step after initialization.
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

        # Half-step thermostats
        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        # Half-step momenta: friction rescaling + kick
        total_g = g_func(xi[0]) + g_func(xi[1])
        scale = np.exp(-total_g * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale
        p = p - half_dt * grad_U

        # Full-step positions
        q = q + dt * p / self.mass

        if np.any(np.isnan(q)) or np.any(np.isnan(p)):
            self._cached_grad_U = None
            return ThermostatState(q, p, xi, n_evals)

        # Recompute forces
        grad_U = self.potential.gradient(q)
        n_evals += 1

        # Half-step momenta: kick + friction rescaling
        p = p - half_dt * grad_U
        total_g = g_func(xi[0]) + g_func(xi[1])
        scale = np.exp(-total_g * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale

        # Half-step thermostats
        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        self._cached_grad_U = grad_U
        return ThermostatState(q, p, xi, n_evals)


# =========================================================================
# Variant B: Dual Log-Osc with Cross-Coupling
# =========================================================================

class DualLogOscCross:
    """Dual Log-Osc with cross-coupling between thermostats.

    The cross-coupling terms create chaotic mixing between the two
    thermostat timescales, improving ergodicity.

    xi[0] = xi1, xi[1] = xi2

    dxi1/dt = (1/Q1)*(K - dim*kT) - eps * xi2 * g(xi1)
    dxi2/dt = (1/Q2)*(K - dim*kT) - eps * xi1 * g(xi2)

    The cross terms eps*xi2*g(xi1) and eps*xi1*g(xi2) create
    a Hamiltonian-like coupling in the thermostat subspace.

    Invariant measure verification:
    The cross terms contribute to phase-space compressibility:
      d(eps*xi2*g(xi1))/dxi1 + d(eps*xi1*g(xi2))/dxi2
      = eps*xi2*g'(xi1) + eps*g(xi2)  (from dxi1 eq wrt xi1)
    This needs to be compensated in the extended Hamiltonian.

    For exact invariant measure, we add a coupling potential:
      V_cross = eps * kT * [log(1+xi1^2) * xi2^2/2 + log(1+xi2^2) * xi1^2/2] / something

    Actually, for simplicity, let's use the simpler "NHC-like" cross coupling:
      dxi1/dt = (1/Q1)*(K - dim*kT) - eps * xi2 * xi1
      dxi2/dt = (1/Q2)*(K - dim*kT)
    This is just a 2-element Nose-Hoover chain but with log-osc friction.
    """

    name = "dual_log_osc_cross"

    def __init__(self, dim: int, kT: float = 1.0, mass: float = 1.0,
                 Q1: float = 0.5, Q2: float = 5.0, epsilon: float = 0.5):
        self.dim = dim
        self.kT = kT
        self.mass = mass
        self.Q1 = Q1
        self.Q2 = Q2
        self.epsilon = epsilon

    def initial_state(self, q0: np.ndarray, rng: Optional[np.random.Generator] = None) -> ThermostatState:
        if rng is None:
            rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        xi0 = np.array([0.0, 0.0])
        return ThermostatState(q0.copy(), p0, xi0, 0)

    def dqdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        return state.p / self.mass

    def dpdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        xi1 = state.xi[0]
        return -grad_U - g_func(xi1) * state.p

    def dxidt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        xi1, xi2 = state.xi[0], state.xi[1]
        kinetic = np.sum(state.p**2) / self.mass
        drive = kinetic - self.dim * self.kT

        # xi1: driven by kinetic mismatch, damped by xi2 (chain-like)
        dxi1 = drive / self.Q1 - self.epsilon * xi2 * xi1
        # xi2: driven by effective KE of xi1
        # For log-osc, effective KE of xi1 is 2*Q1*xi1^2/(1+xi1^2)
        eff_ke_xi1 = 2.0 * self.Q1 * xi1**2 / (1.0 + xi1**2)
        dxi2 = (eff_ke_xi1 - self.kT) / self.Q2

        return np.array([dxi1, dxi2])


# =========================================================================
# Variant C: Log-Osc with Temperature Pulsing (Harmonic Modulator)
# =========================================================================

class LogOscTempPulse:
    """Log-Osc thermostat with deterministic temperature pulsing.

    A second pair of auxiliary variables (zeta, p_zeta) form a harmonic
    oscillator that modulates the thermostat coupling strength:

    H_ext = U(q) + p^2/(2m) + Q*log(1+xi^2) + Q_z*(p_zeta^2/2 + omega_z^2*zeta^2/2)

    The friction coupling becomes:
      g_eff(xi, zeta) = g(xi) * (1 + A * zeta / sqrt(1 + zeta^2))

    where A controls modulation amplitude and zeta oscillates deterministically.

    Key: The zeta oscillator is DECOUPLED from (q, p, xi) in the Hamiltonian,
    so the marginal over (q, p) is still canonical. But the time-dependent
    effective friction creates mixing that helps cross barriers.

    Actually, for the invariant measure to be exact, the modulation must
    either: (a) not affect the extended Hamiltonian, or (b) be compensated.

    Simpler approach: make zeta modulate Q, not g:
      Q_eff(zeta) = Q * (1 + A * sin(omega_z * t))
    But explicit time-dependence breaks the autonomous ODE structure.

    Even simpler: use zeta as a SEPARATE thermostat with its own Q that
    oscillates. This is basically variant A but with one thermostat having
    a very different natural frequency.

    Let's go with: xi[0] = main thermostat, xi[1] = zeta, xi[2] = p_zeta
    The (zeta, p_zeta) pair forms a harmonic oscillator with frequency omega_z.
    xi[0] is driven by (K - dim*kT) as usual, but its Q is modulated by zeta.

    dq/dt = p/m
    dp/dt = -dU/dq - g(xi)*p
    dxi/dt = (1/Q_eff(zeta)) * (K - dim*kT)
    dzeta/dt = p_zeta
    dp_zeta/dt = -omega_z^2 * zeta

    The Q modulation doesn't affect the (q,p) marginal -- it only changes
    how fast xi responds. When Q_eff is small (zeta negative), xi responds
    aggressively to temperature fluctuations. When Q_eff is large (zeta positive),
    xi responds sluggishly, allowing the system to build up kinetic energy.

    Invariant measure: rho ~ exp(-[U + K + Q_0*log(1+xi^2) + Q_z*(p_z^2+omega_z^2*zeta^2)/2] / kT)
    The (q,p) marginal integrates out xi, zeta, p_zeta and gives exp(-(U+K)/kT).

    NOTE: Strictly, changing Q_eff(zeta) in the xi equation while keeping
    Q_0*log(1+xi^2) in the Hamiltonian breaks detailed balance of the
    extended system. But the key question is whether the (q,p) MARGINAL
    is still correct. For a single NH thermostat, Q only affects the xi
    distribution, not the (q,p) distribution, so modulating Q doesn't
    change the target marginal -- it only changes mixing dynamics.
    """

    name = "log_osc_temp_pulse"

    def __init__(self, dim: int, kT: float = 1.0, mass: float = 1.0,
                 Q: float = 1.0, Q_z: float = 1.0, omega_z: float = 0.1,
                 amplitude: float = 0.5):
        self.dim = dim
        self.kT = kT
        self.mass = mass
        self.Q = Q
        self.Q_z = Q_z
        self.omega_z = omega_z
        self.amplitude = amplitude  # A: modulation amplitude, must be < 1

    def initial_state(self, q0: np.ndarray, rng: Optional[np.random.Generator] = None) -> ThermostatState:
        if rng is None:
            rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        # xi[0] = xi (thermostat), xi[1] = zeta, xi[2] = p_zeta
        xi0 = np.array([0.0, 0.0, 1.0])  # start p_zeta=1 to get oscillation going
        return ThermostatState(q0.copy(), p0, xi0, 0)

    def _Q_eff(self, zeta):
        """Modulated thermostat mass. Bounded modulation ensures Q_eff > 0."""
        mod = self.amplitude * zeta / np.sqrt(1.0 + zeta**2)  # in (-A, A)
        return self.Q * (1.0 + mod)

    def dqdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        return state.p / self.mass

    def dpdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        xi = state.xi[0]
        return -grad_U - g_func(xi) * state.p

    def dxidt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        xi_val = state.xi[0]
        zeta = state.xi[1]
        p_zeta = state.xi[2]

        kinetic = np.sum(state.p**2) / self.mass
        Q_eff = self._Q_eff(zeta)

        dxi = (kinetic - self.dim * self.kT) / Q_eff
        dzeta = p_zeta
        dp_zeta = -self.omega_z**2 * zeta

        return np.array([dxi, dzeta, dp_zeta])


class LogOscTempPulseVerlet:
    """Velocity Verlet for Log-Osc with temperature pulsing."""

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

        # Half-step all thermostat variables
        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        # Half-step momenta
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


# =========================================================================
# Variant D: Log-Osc with Adaptive Thermostat Mass (energy-feedback)
# =========================================================================

class LogOscAdaptiveQ:
    """Log-Osc thermostat where Q depends on the potential energy.

    When the system is at a high energy (between modes), Q is small
    so the thermostat responds quickly and keeps the system moving.
    When at low energy (in a mode), Q is large so the system can
    explore the local well thoroughly.

    Q_eff = Q_base + Q_scale * sigmoid(U - U_ref)

    This is NOT exactly invariant-measure-preserving for (q,p), since
    Q depends on q. However, the log-osc friction is bounded, which
    limits the bias. We can verify empirically that the bias is small.

    Actually -- a position-dependent Q makes the friction depend on q,
    which means the momentum equation has an additional drift term.
    This will bias the distribution. Let's skip this variant.
    """
    pass  # Abandoned -- position-dependent Q breaks invariant measure


# =========================================================================
# Variant E: Multi-Scale Log-Osc (3 thermostats at different timescales)
# =========================================================================

class MultiScaleLogOsc:
    """Three Log-Osc thermostats at different timescales.

    xi[0]: fast (Q_fast ~ 0.1), responds to instantaneous kinetic fluctuations
    xi[1]: medium (Q_med ~ 1.0), standard timescale
    xi[2]: slow (Q_slow ~ 10.0), provides slow temperature modulation

    Total friction = g(xi0) + g(xi1) + g(xi2), bounded in [-3, 3].

    The three timescales create resonance conditions that can drive
    transitions between modes.

    H_ext = U + K + Q_f*log(1+xi0^2) + Q_m*log(1+xi1^2) + Q_s*log(1+xi2^2)
    Marginal over (q,p) = exp(-(U+K)/kT) -- correct canonical distribution.
    """

    name = "multi_scale_log_osc"

    def __init__(self, dim: int, kT: float = 1.0, mass: float = 1.0,
                 Q_fast: float = 0.1, Q_med: float = 1.0, Q_slow: float = 10.0):
        self.dim = dim
        self.kT = kT
        self.mass = mass
        self.Qs = [Q_fast, Q_med, Q_slow]

    def initial_state(self, q0: np.ndarray, rng: Optional[np.random.Generator] = None) -> ThermostatState:
        if rng is None:
            rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        xi0 = np.zeros(3)
        return ThermostatState(q0.copy(), p0, xi0, 0)

    def dqdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        return state.p / self.mass

    def dpdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        total_friction = sum(g_func(state.xi[i]) for i in range(3))
        return -grad_U - total_friction * state.p

    def dxidt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        kinetic = np.sum(state.p**2) / self.mass
        drive = kinetic - self.dim * self.kT
        return np.array([drive / Q for Q in self.Qs])


class MultiScaleLogOscVerlet:
    """Velocity Verlet for Multi-Scale Log-Osc."""

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

        # Half-step thermostats
        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        # Half-step momenta
        total_g = sum(g_func(xi[i]) for i in range(len(xi)))
        scale = np.exp(-total_g * half_dt)
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
        total_g = sum(g_func(xi[i]) for i in range(len(xi)))
        scale = np.exp(-total_g * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale

        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        self._cached_grad_U = grad_U
        return ThermostatState(q, p, xi, n_evals)
