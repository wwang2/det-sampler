"""Task 2: ESH + Thermostat Hybrid Sampler (Option C: Alternating).

Implements ESHPlusThermostat: alternates ESH leapfrog steps (fast local
exploration along H_ESH level sets) with log-osc thermostat steps
(global thermalization). Avoids the Liouville compatibility issue.

The key insight: ESH steps are (approximately) Hamiltonian moves that
explore level sets of H_ESH = U + log|v| without changing the canonical
marginal in q. Thermostat steps then thermalize the momentum, allowing
jumps between H_ESH level sets and ergodic coverage of phase space.
"""

import sys
import numpy as np
from pathlib import Path

WORKTREE = '/Users/wujiewang/code/det-sampler/.worktrees/esh-hybrid-026'
sys.path.insert(0, WORKTREE)

from research.eval.integrators import ThermostatState


def g_losc(xi_val):
    """Bounded log-osc friction: g(xi) = 2*xi/(1+xi^2) in [-1, 1]."""
    return 2.0 * xi_val / (1.0 + xi_val**2)


class ESHPlusThermostat:
    """ESH + Multi-scale Log-Osc Thermostat hybrid (Option C: Alternating).

    Each integration cycle consists of:
      1. L_esh ESH leapfrog steps  (conservative, fast local exploration)
      2. M_thermo thermostat steps  (dissipative, thermalizing)

    ESH dynamics (1D, generalized to d-dim):
      dx/dt = v / ||v||           (unit speed direction)
      dv/dt = -grad_U * ||v||/d  (force scaled by speed, d=2 for 2D)

    Thermostat: Multi-scale log-osc (NHCTail from multiscale-chain-009)
      dp/dt = -grad_U - g_total(xi) * p
      dxi_k/dt = (K - d*kT) / Q_k

    State layout: xi = [xi_fast, xi_med, xi_med_chain, xi_slow]
    (matches MultiScaleNHCTail with Qs=[0.1, 0.7, 10.0], chain_length=2)

    Force eval counting: ESH steps cost 1 force eval each (leapfrog).
    Thermostat steps cost 1 force eval each (Verlet). Total per cycle:
    L_esh + M_thermo force evals (FSAL reduces to ~L_esh + M_thermo - 1).
    """

    name = "esh_hybrid"

    def __init__(self, dim: int, kT: float = 1.0, mass: float = 1.0,
                 Qs: list = None, L_esh: int = 10, M_thermo: int = 100,
                 dt_esh: float = 0.05, dt_thermo: float = 0.03):
        self.dim = dim
        self.kT = kT
        self.mass = mass
        self.Qs = Qs if Qs is not None else [0.1, 0.7, 10.0]
        self.L_esh = L_esh
        self.M_thermo = M_thermo
        self.dt_esh = dt_esh
        self.dt_thermo = dt_thermo

        # Build NHCTail xi structure: same as MultiScaleNHCTail
        # Scale 0 (Q=0.1): standalone (Q < kT/2 = 0.5)
        # Scale 1 (Q=0.7): chain length 2 (Q > kT/2)
        # Scale 2 (Q=10): chain length 2 (Q > kT/2)
        self._offsets = []
        self._chain_lengths = []
        self._Q_matrix = []
        total = 0
        for Q_base in self.Qs:
            offset = total
            self._offsets.append(offset)
            if Q_base > kT / 2.0:
                M = 2
            else:
                M = 1
            self._chain_lengths.append(M)
            chain_Qs = [Q_base] + [Q_base] * (M - 1)
            self._Q_matrix.append(chain_Qs)
            total += M
        self.n_xi = total
        self.n_scales = len(self.Qs)

    def initial_state(self, q0: np.ndarray, rng=None) -> ThermostatState:
        if rng is None:
            rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        xi0 = np.zeros(self.n_xi)
        return ThermostatState(q0.copy(), p0, xi0, 0)

    def _xi_idx(self, k, j):
        return self._offsets[k] + j

    def _total_friction(self, xi: np.ndarray) -> float:
        """Sum of g(xi_{k,0}) over all scales."""
        return sum(g_losc(xi[self._xi_idx(k, 0)]) for k in range(self.n_scales))

    def _dxi_dt(self, p: np.ndarray, xi: np.ndarray) -> np.ndarray:
        """Thermostat variable derivatives (NHCTail structure)."""
        dxi = np.zeros(self.n_xi)
        kinetic = np.sum(p**2) / self.mass
        drive = kinetic - self.dim * self.kT

        for k in range(self.n_scales):
            M = self._chain_lengths[k]
            idx0 = self._xi_idx(k, 0)
            xi_k0 = xi[idx0]

            dxi[idx0] = drive / self._Q_matrix[k][0]
            if M > 1:
                xi_k1 = xi[self._xi_idx(k, 1)]
                dxi[idx0] -= xi_k1 * xi_k0

            for j in range(1, M):
                idx_j = self._xi_idx(k, j)
                xi_kj = xi[idx_j]
                if j == 1:
                    prev_ke = 2.0 * self._Q_matrix[k][0] * xi_k0**2 / (1.0 + xi_k0**2)
                else:
                    prev_idx = self._xi_idx(k, j-1)
                    prev_ke = self._Q_matrix[k][j-1] * xi[prev_idx]**2
                dxi[idx_j] = (prev_ke - self.kT) / self._Q_matrix[k][j]
                if j < M - 1:
                    next_xi = xi[self._xi_idx(k, j+1)]
                    dxi[idx_j] -= next_xi * xi_kj

        return dxi

    def esh_step(self, q: np.ndarray, p: np.ndarray, grad_U: np.ndarray,
                 dt: float) -> tuple:
        """Single ESH leapfrog step (does NOT consume a grad eval — caller provides grad_U).

        ESH leapfrog:
          p_half = p - (dt/2) * grad_U * ||p||/d
          q_new  = q + dt * p_half / ||p_half||
          [recompute grad_U at q_new — returned for FSAL]
          p_new  = p_half - (dt/2) * grad_U_new * ||p_half||/d

        Returns (q_new, p_new, grad_U_at_q_new).
        The caller is responsible for force eval counting.
        """
        half_dt = 0.5 * dt
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-300:
            p_norm = 1e-300

        # Half-step momentum: ESH force scaling by ||p||/d
        p_half = p - half_dt * grad_U * (p_norm / self.dim)
        p_half_norm = np.linalg.norm(p_half)
        if p_half_norm < 1e-300:
            p_half_norm = 1e-300

        # Full-step position: unit direction
        q_new = q + dt * p_half / p_half_norm

        return q_new, p_half, p_half_norm  # return p_half for second kick

    def thermo_step(self, q: np.ndarray, p: np.ndarray, xi: np.ndarray,
                    grad_U: np.ndarray, dt: float) -> tuple:
        """Single thermostat Verlet step (uses cached grad_U, returns new grad_U).

        Standard velocity Verlet with log-osc friction rescaling.
        Returns (q_new, p_new, xi_new, grad_U_new).
        """
        half_dt = 0.5 * dt

        # Half-step thermostats
        xi_dot = self._dxi_dt(p, xi)
        xi = xi + half_dt * xi_dot

        # Half-step momenta: friction rescaling + potential kick
        total_g = self._total_friction(xi)
        scale = np.exp(-total_g * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale
        p = p - half_dt * grad_U

        # Full-step positions
        q = q + dt * p / self.mass

        return q, p, xi, None  # None signals caller to recompute grad_U

    def thermo_step_second_half(self, q: np.ndarray, p: np.ndarray, xi: np.ndarray,
                                 grad_U: np.ndarray, dt: float) -> tuple:
        """Second half of thermostat Verlet step (given new grad_U)."""
        half_dt = 0.5 * dt

        p = p - half_dt * grad_U
        total_g = self._total_friction(xi)
        scale = np.exp(-total_g * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale

        xi_dot = self._dxi_dt(p, xi)
        xi = xi + half_dt * xi_dot

        return p, xi

    # Protocol methods for compatibility with VelocityVerletThermostat
    def dqdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        return state.p / self.mass

    def dpdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        total_g = self._total_friction(state.xi)
        return -grad_U - total_g * state.p

    def dxidt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        return self._dxi_dt(state.p, state.xi)


class ESHHybridIntegrator:
    """Integrator for ESH + Thermostat hybrid sampler.

    Each call to step() runs ONE full cycle:
      - L_esh ESH leapfrog steps
      - M_thermo thermostat Verlet steps

    Force eval cost per cycle: L_esh + M_thermo (with FSAL).
    """

    def __init__(self, dynamics: ESHPlusThermostat, potential, dt: float,
                 kT: float = 1.0, mass: float = 1.0):
        self.dynamics = dynamics
        self.potential = potential
        # dt here is unused (dynamics carries dt_esh, dt_thermo)
        self.dt = dt
        self.kT = kT
        self.mass = mass
        self._cached_grad_U = None

    def step(self, state: ThermostatState) -> ThermostatState:
        """Run one cycle: L_esh ESH steps + M_thermo thermostat steps."""
        q, p, xi, n_evals = state
        dyn = self.dynamics

        # Get initial gradient (use FSAL cache)
        if self._cached_grad_U is not None:
            grad_U = self._cached_grad_U
        else:
            grad_U = self.potential.gradient(q)
            n_evals += 1

        # ---- ESH phase: L_esh leapfrog steps ----
        for _ in range(dyn.L_esh):
            if np.any(np.isnan(q)) or np.any(np.isnan(p)):
                self._cached_grad_U = None
                return ThermostatState(q, p, xi, n_evals)

            p_norm = np.linalg.norm(p)
            if p_norm < 1e-300:
                p_norm = 1e-300
            half_dt = 0.5 * dyn.dt_esh

            # Half-step p (ESH force)
            p_half = p - half_dt * grad_U * (p_norm / dyn.dim)
            p_half_norm = np.linalg.norm(p_half)
            if p_half_norm < 1e-300:
                p_half_norm = 1e-300

            # Full-step q (unit direction)
            q = q + dyn.dt_esh * p_half / p_half_norm

            # Recompute gradient
            grad_U = self.potential.gradient(q)
            n_evals += 1

            # Second half-step p
            p = p_half - half_dt * grad_U * (p_half_norm / dyn.dim)

        # ---- Thermostat phase: M_thermo Verlet steps ----
        for _ in range(dyn.M_thermo):
            if np.any(np.isnan(q)) or np.any(np.isnan(p)):
                self._cached_grad_U = None
                return ThermostatState(q, p, xi, n_evals)

            half_dt = 0.5 * dyn.dt_thermo

            # Half-step thermostats
            xi_dot = dyn._dxi_dt(p, xi)
            xi = xi + half_dt * xi_dot

            # Half-step momenta: friction + kick
            total_g = dyn._total_friction(xi)
            scale = np.exp(-total_g * half_dt)
            scale = np.clip(scale, 1e-10, 1e10)
            p = p * scale
            p = p - half_dt * grad_U

            # Full-step positions
            q = q + dyn.dt_thermo * p / dyn.mass

            if np.any(np.isnan(q)) or np.any(np.isnan(p)):
                self._cached_grad_U = None
                return ThermostatState(q, p, xi, n_evals)

            # Recompute gradient (FSAL: reuse in next thermo step)
            grad_U = self.potential.gradient(q)
            n_evals += 1

            # Second half-step momenta + thermostats
            p = p - half_dt * grad_U
            total_g = dyn._total_friction(xi)
            scale = np.exp(-total_g * half_dt)
            scale = np.clip(scale, 1e-10, 1e10)
            p = p * scale

            xi_dot = dyn._dxi_dt(p, xi)
            xi = xi + half_dt * xi_dot

        self._cached_grad_U = grad_U
        return ThermostatState(q, p, xi, n_evals)


# ============================================================
# Quick sanity check
# ============================================================

def quick_sanity_check():
    """Verify the hybrid sampler runs without errors on 1D HO."""
    import sys
    sys.path.insert(0, WORKTREE)
    from research.eval.potentials import HarmonicOscillator1D, GaussianMixture2D
    from research.eval.evaluator import run_sampler, kl_divergence_histogram

    print("=" * 60)
    print("HYBRID SAMPLER: SANITY CHECK")
    print("=" * 60)

    kT = 1.0
    pot_ho = HarmonicOscillator1D(omega=1.0)

    # Test 1D
    dyn_1d = ESHPlusThermostat(dim=1, kT=kT, Qs=[0.1, 0.7, 10.0],
                                L_esh=5, M_thermo=50,
                                dt_esh=0.05, dt_thermo=0.01)
    state = dyn_1d.initial_state(np.array([1.0]), rng=np.random.default_rng(42))
    integrator = ESHHybridIntegrator(dyn_1d, pot_ho, dt=0.01)

    print("\n1D HO: running 200 cycles...")
    qs, ps = [], []
    for i in range(200):
        state = integrator.step(state)
        if np.any(np.isnan(state.q)) or np.any(np.isnan(state.p)):
            print(f"  NaN at cycle {i}!")
            break
        qs.append(state.q[0])
        ps.append(state.p[0])

    if qs:
        q_arr = np.array(qs)
        p_arr = np.array(ps)
        print(f"  Ran {len(qs)} cycles, {state.n_force_evals} force evals")
        print(f"  q: mean={np.mean(q_arr):.3f}, std={np.std(q_arr):.3f} (target: 0, 1)")
        print(f"  p: mean={np.mean(p_arr):.3f}, std={np.std(p_arr):.3f} (target: 0, 1)")
        kl = kl_divergence_histogram(q_arr[:, np.newaxis], pot_ho, kT=kT)
        print(f"  KL (1D HO, limited run): {kl:.4f}")

    # Test 2D GMM (quick)
    print("\n2D GMM: running 500 cycles (quick test)...")
    pot_gmm = GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5)
    dyn_2d = ESHPlusThermostat(dim=2, kT=kT, Qs=[0.1, 0.7, 10.0],
                                L_esh=10, M_thermo=100,
                                dt_esh=0.05, dt_thermo=0.03)
    state2 = dyn_2d.initial_state(np.array([0.5, 0.5]), rng=np.random.default_rng(42))
    integrator2 = ESHHybridIntegrator(dyn_2d, pot_gmm, dt=0.03)

    qs2 = []
    for _ in range(500):
        state2 = integrator2.step(state2)
        if np.any(np.isnan(state2.q)):
            break
        qs2.append(state2.q.copy())

    if qs2:
        q2_arr = np.array(qs2)
        print(f"  Ran {len(qs2)} cycles, {state2.n_force_evals} force evals")
        print(f"  q[0]: mean={np.mean(q2_arr[:,0]):.3f}, q[1]: mean={np.mean(q2_arr[:,1]):.3f}")
        print(f"  q range: [{q2_arr.min():.2f}, {q2_arr.max():.2f}]")
        # Check if we're exploring (should hit multiple modes at radius 3)
        radii = np.sqrt(q2_arr[:,0]**2 + q2_arr[:,1]**2)
        print(f"  Radii: mean={np.mean(radii):.2f}, max={np.max(radii):.2f} (modes at r=3)")

    print("\nSanity check complete.")
    return True


if __name__ == "__main__":
    quick_sanity_check()
