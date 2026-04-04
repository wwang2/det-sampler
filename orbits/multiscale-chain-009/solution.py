"""Multi-Scale Log-Osc Chain (MSLOCR) -- combining multi-scale + chain thermostat.

Merges the two Round 3 winners:
- Multi-Scale Log-Osc (multiT-005): Best multi-modal (GMM KL=0.148)
- Log-Osc Chain (chain-002): Best unimodal (DW KL=0.007, HO KL=0.001)

KEY INSIGHT: For log-osc potential Q*log(1+xi^2), the extended distribution
p(xi) ~ (1+xi^2)^{-Q/kT} only normalizes when Q > kT/2. For Q <= kT/2,
the distribution is improper (heavy tails). This means:
  - Chain coupling ONLY works when Q > kT/2 (otherwise the chain variable
    tries to control a non-equilibrium variable, causing drift/instability)
  - Small Q values (e.g. 0.1) work fine as standalone multi-scale
    thermostats because g(xi) is bounded, so the (q,p) marginal is correct
    even though xi wanders

DESIGN: Combine multi-scale independent thermostats with optional chain
coupling ONLY on variables with Q > kT/2.

Architecture A': Multi-Scale + Selective Chain
===============================================
- N_chain thermostats with Q >= 0.7: each gets LOCR chain (length 2-3)
- N_free thermostats with small Q: standalone log-osc (no chain)
- All first variables couple to momentum: dp/dt = -dU/dq - [sum g(xi)] * p

Architecture B: Multi-Scale with NHC-tail chains
=================================================
- Use log-osc friction for the first variable in each chain
- Use STANDARD quadratic NH potential for all chain tail variables
  (Q_j*xi_j^2/2 instead of Q_j*log(1+xi_j^2))
- This avoids the normalization issue: quadratic potential always normalizes
- Chain coupling uses standard NHC: drive = Q_{j-1}*xi_{j-1}^2 - kT for j>1
  But for j=1->2: drive = 2*Q_1*xi_1^2/(1+xi_1^2) - kT (log-osc effective KE)

References:
  - Parent: orbits/log-osc-multiT-005 (Issue #8) -- multi-scale approach
  - Grandparent: orbits/log-osc-001 (Issue #3) -- base log-osc thermostat
  - Related: orbits/log-osc-chain-002 (Issue #5) -- LOCR chain approach
  - Martyna et al. (1992). Nose-Hoover chains. J. Chem. Phys. 97, 2635.
  - Fukuda & Nakamura (2002). Multiple Nose-Hoover thermostats. Phys. Rev. E 65, 026105.
"""

from typing import Optional
import numpy as np
from research.eval.integrators import ThermostatState


def g_func(xi_val):
    """Bounded friction: g(xi) = 2*xi/(1+xi^2), in [-1, 1]."""
    return 2.0 * xi_val / (1.0 + xi_val**2)


# =========================================================================
# Architecture B: Multi-Scale with NHC-tail chains
# =========================================================================

class MultiScaleNHCTail:
    """Multi-scale log-osc thermostats with standard NHC chain tails.

    Each scale k has:
      - xi_{k,0}: log-osc thermostat (friction g(xi) on momentum)
      - xi_{k,1}, ..., xi_{k,M-1}: standard NH chain variables (quadratic potential)

    The first variable uses log-osc friction g(xi_{k,0}) for bounded coupling.
    Chain tail variables use standard NHC equations:
      dxi_{k,1}/dt = (1/Q_{k,1}) * (2*Q_{k,0}*xi_{k,0}^2/(1+xi_{k,0}^2) - kT) - xi_{k,2}*xi_{k,1}
      dxi_{k,j}/dt = (1/Q_{k,j}) * (Q_{k,j-1}*xi_{k,j-1}^2 - kT) - xi_{k,j+1}*xi_{k,j}

    Extended Hamiltonian:
      H_ext = U(q) + K(p) + sum_k [Q_{k,0}*log(1+xi_{k,0}^2) + sum_{j>0} Q_{k,j}*xi_{k,j}^2/2]

    For the NHC tail to work correctly, we need the first variable's effective
    KE to have equilibrium value kT. For log-osc with Q > kT/2, this is satisfied:
      <2*Q*xi^2/(1+xi^2)> = kT  when Q > kT/2

    For Q <= kT/2, the chain tail is omitted (chain_length=1 effectively).

    xi layout: flat array [xi_{0,0}, xi_{0,1}, ..., xi_{0,M0-1}, xi_{1,0}, ...]
    """

    name = "multiscale_nhctail"

    def __init__(self, dim: int, kT: float = 1.0, mass: float = 1.0,
                 Qs: Optional[list] = None,
                 chain_length: int = 2,
                 chain_Q_multiplier: float = 1.0):
        """
        Args:
            Qs: Base Q values for each scale
            chain_length: Max chain length (only applied when Q > kT/2)
            chain_Q_multiplier: Q for chain tail = Q_base * multiplier
        """
        self.dim = dim
        self.kT = kT
        self.mass = mass
        self.Qs = Qs if Qs is not None else [0.1, 0.7, 10.0]
        self.n_scales = len(self.Qs)
        self.chain_Q_multiplier = chain_Q_multiplier

        # Determine chain lengths: only chain when Q > kT/2
        self.chain_lengths = []
        self.Q_matrix = []
        for Q_base in self.Qs:
            if Q_base > kT / 2.0:
                M = chain_length
            else:
                M = 1  # No chain for improper distributions
            self.chain_lengths.append(M)

            chain_Qs = [Q_base]
            for j in range(1, M):
                chain_Qs.append(Q_base * chain_Q_multiplier)
            self.Q_matrix.append(chain_Qs)

        # Build flat index mapping
        self._offsets = []  # offset[k] = start index for chain k
        total = 0
        for k in range(self.n_scales):
            self._offsets.append(total)
            total += self.chain_lengths[k]
        self.n_xi = total

    def _xi_idx(self, k, j):
        return self._offsets[k] + j

    def initial_state(self, q0: np.ndarray, rng: Optional[np.random.Generator] = None) -> ThermostatState:
        if rng is None:
            rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        xi0 = np.zeros(self.n_xi)
        return ThermostatState(q0.copy(), p0, xi0, 0)

    def dqdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        return state.p / self.mass

    def dpdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        total_friction = 0.0
        for k in range(self.n_scales):
            xi_k0 = state.xi[self._xi_idx(k, 0)]
            total_friction += g_func(xi_k0)
        return -grad_U - total_friction * state.p

    def dxidt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        xi = state.xi
        dxi = np.zeros(self.n_xi)
        kinetic = np.sum(state.p**2) / self.mass
        drive = kinetic - self.dim * self.kT

        for k in range(self.n_scales):
            M = self.chain_lengths[k]
            idx0 = self._xi_idx(k, 0)
            xi_k0 = xi[idx0]

            # First variable: driven by physical KE, damped by chain
            dxi[idx0] = drive / self.Q_matrix[k][0]
            if M > 1:
                xi_k1 = xi[self._xi_idx(k, 1)]
                dxi[idx0] -= xi_k1 * xi_k0

            # Chain tail variables
            for j in range(1, M):
                idx_j = self._xi_idx(k, j)
                xi_kj = xi[idx_j]

                if j == 1:
                    # Effective KE of log-osc first variable
                    prev_ke = 2.0 * self.Q_matrix[k][0] * xi_k0**2 / (1.0 + xi_k0**2)
                else:
                    # Standard NHC KE
                    prev_idx = self._xi_idx(k, j-1)
                    prev_ke = self.Q_matrix[k][j-1] * xi[prev_idx]**2

                Gj = prev_ke - self.kT
                dxi[idx_j] = Gj / self.Q_matrix[k][j]

                if j < M - 1:
                    next_xi = xi[self._xi_idx(k, j+1)]
                    dxi[idx_j] -= next_xi * xi_kj

        return dxi


class MultiScaleNHCTailVerlet:
    """Velocity Verlet for Multi-Scale NHC-Tail."""

    def __init__(self, dynamics, potential, dt: float, kT: float = 1.0, mass: float = 1.0):
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        self.dynamics = dynamics
        self.potential = potential
        self.dt = dt
        self.kT = kT
        self.mass = mass
        self._cached_grad_U = None

    def _total_friction(self, xi):
        total = 0.0
        for k in range(self.dynamics.n_scales):
            idx = self.dynamics._xi_idx(k, 0)
            total += g_func(xi[idx])
        return total

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
        total_g = self._total_friction(xi)
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
        total_g = self._total_friction(xi)
        scale = np.exp(-total_g * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale

        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        self._cached_grad_U = grad_U
        return ThermostatState(q, p, xi, n_evals)


# =========================================================================
# Architecture A': Multi-Scale pure (no chain) -- baseline from parent
# =========================================================================

class MultiScaleLogOsc:
    """N Log-Osc thermostats at different timescales (no chain coupling).

    Reproduced from parent orbit (multiT-005) for comparison.
    Each thermostat xi_k independently drives towards canonical temperature.

    dp/dt = -dU/dq - [sum_k g(xi_k)] * p
    dxi_k/dt = (1/Q_k) * (K - dim*kT)
    """

    name = "multi_scale_log_osc"

    def __init__(self, dim: int, kT: float = 1.0, mass: float = 1.0,
                 Qs: Optional[list] = None):
        self.dim = dim
        self.kT = kT
        self.mass = mass
        self.Qs = Qs if Qs is not None else [0.1, 0.7, 10.0]
        self.n_thermo = len(self.Qs)

    def initial_state(self, q0: np.ndarray, rng: Optional[np.random.Generator] = None) -> ThermostatState:
        if rng is None:
            rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        xi0 = np.zeros(self.n_thermo)
        return ThermostatState(q0.copy(), p0, xi0, 0)

    def dqdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        return state.p / self.mass

    def dpdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        total_friction = sum(g_func(state.xi[i]) for i in range(self.n_thermo))
        return -grad_U - total_friction * state.p

    def dxidt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        kinetic = np.sum(state.p**2) / self.mass
        drive = kinetic - self.dim * self.kT
        return np.array([drive / Q for Q in self.Qs])


class MultiScaleLogOscVerlet:
    """Velocity Verlet for Multi-Scale Log-Osc (no chain)."""

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

        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        total_g = sum(g_func(xi[i]) for i in range(len(xi)))
        scale = np.exp(-total_g * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale
        p = p - half_dt * grad_U

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


# =========================================================================
# Architecture C: Hierarchical LOCR (single chain, multi-scale Q)
# =========================================================================

class HierarchicalLOCR:
    """Single log-osc chain with hierarchically varying Q values.

    xi_1 -> xi_2 -> xi_3 -> ... -> xi_M
    Q values vary along chain to span multiple timescales.
    xi_1 uses log-osc friction on momentum; rest are standard NHC.

    CONSTRAINT: Q_1 > kT/2 for proper chain coupling.
    """

    name = "hierarchical_locr"

    def __init__(self, dim: int, kT: float = 1.0, mass: float = 1.0,
                 Qs: Optional[list] = None):
        self.dim = dim
        self.kT = kT
        self.mass = mass
        self.Qs = Qs if Qs is not None else [0.7, 1.0, 10.0]
        self.M = len(self.Qs)
        # Validate: first Q must be > kT/2
        if self.Qs[0] <= kT / 2.0:
            raise ValueError(f"First Q must be > kT/2 for log-osc chain normalization, got Q={self.Qs[0]}")

    def initial_state(self, q0: np.ndarray, rng: Optional[np.random.Generator] = None) -> ThermostatState:
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
        dxi = np.zeros(M)
        kinetic = np.sum(state.p**2) / self.mass

        # First variable: log-osc, driven by physical KE
        G1 = kinetic - self.dim * self.kT
        dxi[0] = G1 / self.Qs[0]
        if M > 1:
            dxi[0] -= xi[1] * xi[0]

        # Middle variables
        for j in range(1, M - 1):
            if j == 1:
                prev_ke = 2.0 * self.Qs[0] * xi[0]**2 / (1.0 + xi[0]**2)
            else:
                prev_ke = self.Qs[j-1] * xi[j-1]**2
            Gj = prev_ke - self.kT
            dxi[j] = Gj / self.Qs[j] - xi[j+1] * xi[j]

        # Last variable
        if M > 1:
            if M == 2:
                prev_ke = 2.0 * self.Qs[0] * xi[0]**2 / (1.0 + xi[0]**2)
            else:
                prev_ke = self.Qs[M-2] * xi[M-2]**2
            dxi[M-1] = (prev_ke - self.kT) / self.Qs[M-1]

        return dxi


class HierarchicalLOCRVerlet:
    """Velocity Verlet for Hierarchical LOCR."""

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

        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        gxi = g_func(xi[0])
        scale = np.exp(-gxi * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale
        p = p - half_dt * grad_U

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
# Architecture D: Hybrid -- Multi-Scale + Chain on main thermostat
# =========================================================================

class HybridMSLOCR:
    """Multi-scale with chain coupling on the main (medium-Q) thermostat.

    Structure:
      - Fast: xi_f (single log-osc, Q_f=small, no chain -- Q too small)
      - Medium chain: xi_m1 -> xi_m2 (LOCR chain, Q_m >= 0.7)
      - Slow: xi_s (single log-osc, Q_s=large, no chain)

    All first variables couple to momentum:
      dp/dt = -dU/dq - [g(xi_f) + g(xi_m1) + g(xi_s)] * p

    xi layout: [xi_f, xi_m1, xi_m2, xi_s]
    """

    name = "hybrid_mslocr"

    def __init__(self, dim: int, kT: float = 1.0, mass: float = 1.0,
                 Q_fast: float = 0.1, Q_med: float = 0.7,
                 Q_med_chain: float = 0.7, Q_slow: float = 10.0):
        self.dim = dim
        self.kT = kT
        self.mass = mass
        self.Q_fast = Q_fast
        self.Q_med = Q_med
        self.Q_med_chain = Q_med_chain
        self.Q_slow = Q_slow
        self.n_xi = 4  # [xi_f, xi_m1, xi_m2, xi_s]

    def initial_state(self, q0: np.ndarray, rng: Optional[np.random.Generator] = None) -> ThermostatState:
        if rng is None:
            rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        xi0 = np.zeros(self.n_xi)
        return ThermostatState(q0.copy(), p0, xi0, 0)

    def dqdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        return state.p / self.mass

    def dpdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        xi_f = state.xi[0]
        xi_m1 = state.xi[1]
        xi_s = state.xi[3]
        total_friction = g_func(xi_f) + g_func(xi_m1) + g_func(xi_s)
        return -grad_U - total_friction * state.p

    def dxidt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        xi = state.xi
        dxi = np.zeros(self.n_xi)
        kinetic = np.sum(state.p**2) / self.mass
        drive = kinetic - self.dim * self.kT

        # Fast: independent, no chain
        dxi[0] = drive / self.Q_fast

        # Medium chain: xi_m1 -> xi_m2
        dxi[1] = drive / self.Q_med - xi[2] * xi[1]
        # xi_m2 driven by effective KE of xi_m1 (log-osc form)
        eff_ke_m1 = 2.0 * self.Q_med * xi[1]**2 / (1.0 + xi[1]**2)
        dxi[2] = (eff_ke_m1 - self.kT) / self.Q_med_chain

        # Slow: independent, no chain
        dxi[3] = drive / self.Q_slow

        return dxi


class HybridMSLOCRVerlet:
    """Velocity Verlet for Hybrid MS-LOCR."""

    def __init__(self, dynamics, potential, dt: float, kT: float = 1.0, mass: float = 1.0):
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        self.dynamics = dynamics
        self.potential = potential
        self.dt = dt
        self.kT = kT
        self.mass = mass
        self._cached_grad_U = None

    def _total_friction(self, xi):
        return g_func(xi[0]) + g_func(xi[1]) + g_func(xi[3])

    def step(self, state: ThermostatState) -> ThermostatState:
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

        total_g = self._total_friction(xi)
        scale = np.exp(-total_g * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale
        p = p - half_dt * grad_U

        q = q + dt * p / self.mass

        if np.any(np.isnan(q)) or np.any(np.isnan(p)):
            self._cached_grad_U = None
            return ThermostatState(q, p, xi, n_evals)

        grad_U = self.potential.gradient(q)
        n_evals += 1

        p = p - half_dt * grad_U
        total_g = self._total_friction(xi)
        scale = np.exp(-total_g * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale

        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        self._cached_grad_U = grad_U
        return ThermostatState(q, p, xi, n_evals)


# =========================================================================
# Architecture E: Multi-Scale with 4+ thermostats (wider range)
# =========================================================================

class MultiScale4LogOsc:
    """4 Log-Osc thermostats spanning wider timescale range.

    Similar to parent MultiScaleLogOsc but with 4 thermostats for finer
    timescale coverage. No chain coupling (all standalone).

    Best of both worlds: wide range for GMM, medium Q for ergodicity.
    """

    name = "multiscale4_log_osc"

    def __init__(self, dim: int, kT: float = 1.0, mass: float = 1.0,
                 Qs: Optional[list] = None):
        self.dim = dim
        self.kT = kT
        self.mass = mass
        self.Qs = Qs if Qs is not None else [0.05, 0.3, 1.5, 10.0]
        self.n_thermo = len(self.Qs)

    def initial_state(self, q0: np.ndarray, rng: Optional[np.random.Generator] = None) -> ThermostatState:
        if rng is None:
            rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        xi0 = np.zeros(self.n_thermo)
        return ThermostatState(q0.copy(), p0, xi0, 0)

    def dqdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        return state.p / self.mass

    def dpdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        total_friction = sum(g_func(state.xi[i]) for i in range(self.n_thermo))
        return -grad_U - total_friction * state.p

    def dxidt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        kinetic = np.sum(state.p**2) / self.mass
        drive = kinetic - self.dim * self.kT
        return np.array([drive / Q for Q in self.Qs])


# Reuse MultiScaleLogOscVerlet for this (same integrator structure)
MultiScale4LogOscVerlet = MultiScaleLogOscVerlet
