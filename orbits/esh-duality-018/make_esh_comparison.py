"""ESH vs Thermostat Framework: Numerical Comparison.

Implements ESH dynamics for 1D/2D and compares with our thermostat samplers.
Key comparisons:
1. Phase-space trajectories on 1D HO
2. Stationary distributions (histograms)
3. GMM KL divergence
4. ESH-inspired thermostat with V(xi) = Q*(exp(2*xi)/2 - xi)
"""

import sys
import numpy as np
import json
from pathlib import Path

WORKTREE = '/Users/wujiewang/code/det-sampler/.worktrees/esh-duality-018'
sys.path.insert(0, WORKTREE)

from research.eval.potentials import HarmonicOscillator1D, GaussianMixture2D
from research.eval.baselines import NoseHooverChain
from research.eval.integrators import ThermostatState
from research.eval.evaluator import run_sampler, kl_divergence_histogram


# ============================================================
# ESH Dynamics Implementation
# ============================================================

class ESHDynamics1D:
    """ESH dynamics for 1D (d=1).

    From Versteeg 2021 (arXiv:2111.02434):
      K(v) = (d/2)*log(||v||^2/d)
      dx/dt = v/||v|| = sign(v)   [unit speed]
      dv/dt = -grad_U(x) * ||v|| / d = -grad_U(x) * |v|  [d=1]

    This is a CONSERVATIVE (Hamiltonian) system, NOT a thermostat.
    The 'stationary' measure in x approaches exp(-U/kT) but the v-marginal
    is power-law (non-Gaussian), and the system is not ergodic in the
    usual thermostat sense.

    We implement it as a ThermostatState-compatible object for fair comparison
    using the evaluator harness, with xi = log(|p|/sqrt(kT)) as an auxiliary
    tracked variable.
    """
    name = "esh_1d"

    def __init__(self, kT: float = 1.0, mass: float = 1.0):
        self.kT = kT
        self.mass = mass
        self.dim = 1

    def initial_state(self, q0: np.ndarray, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
        # ESH starts with |p| = sqrt(kT) exactly (unit speed in thermal units)
        p0 = np.array([np.sqrt(self.kT)])
        xi0 = np.array([0.0])  # xi = log(|p|/sqrt(kT)) = 0
        return ThermostatState(q0.copy(), p0, xi0, 0)

    def dqdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        # dx/dt = sign(p) for d=1
        return np.sign(state.p)

    def dpdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        # dp/dt = -grad_U * |p|   for d=1
        return -grad_U * np.abs(state.p)

    def dxidt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        # xi = log(|p|/sqrt(kT)), dxi/dt = -grad_U * sign(p)  [tracking variable only]
        xi_dot = -grad_U * np.sign(state.p) / np.sqrt(self.kT)  # auxiliary
        return xi_dot


class ESHLeapfrog1D:
    """Leapfrog integrator for ESH in 1D.

    ESH leapfrog (from the paper):
      p_{n+1/2} = p_n - (dt/2) * grad_U(q_n) * |p_n|
      q_{n+1}   = q_n + dt * sign(p_{n+1/2})
      p_{n+1}   = p_{n+1/2} - (dt/2) * grad_U(q_{n+1}) * |p_{n+1/2}|
    """
    def __init__(self, dynamics, potential, dt: float, kT: float = 1.0, mass: float = 1.0):
        self.dynamics = dynamics
        self.potential = potential
        self.dt = dt
        self.kT = kT
        self.mass = mass
        self._cached_grad = None

    def step(self, state: ThermostatState) -> ThermostatState:
        q, p, xi, n_evals = state
        dt = self.dt
        half_dt = 0.5 * dt

        # Get gradient
        if self._cached_grad is not None:
            grad_U = self._cached_grad
        else:
            grad_U = self.potential.gradient(q)
            n_evals += 1

        # Half-step momentum: ESH force scaling
        p_half = p - half_dt * grad_U * np.abs(p)

        # Full-step position: unit velocity
        q_new = q + dt * np.sign(p_half)

        # New gradient
        grad_U_new = self.potential.gradient(q_new)
        n_evals += 1

        # Half-step momentum at new position
        p_new = p_half - half_dt * grad_U_new * np.abs(p_half)

        # Update xi auxiliary variable: xi = log(|p|/sqrt(kT))
        xi_new = np.log(np.abs(p_new) / np.sqrt(self.kT) + 1e-300)

        # Check stability
        if np.any(np.isnan(q_new)) or np.any(np.isnan(p_new)):
            self._cached_grad = None
            return ThermostatState(q_new, p_new, xi_new, n_evals)

        self._cached_grad = grad_U_new
        return ThermostatState(q_new, p_new, xi_new, n_evals)


# ============================================================
# ESH-inspired Thermostat: V(xi) = Q*(exp(2*xi)/2 - xi)
# ============================================================

class ESHThermostat:
    """ESH-inspired thermostat: uses V(xi) = Q*(exp(2*xi)/2 - xi).

    This is a PROPER thermostat in our Master Theorem framework.
    g(xi) = V'(xi)/Q = exp(2*xi) - 1
    dxi/dt = (p^2/kT - 1)/Q

    The friction g(xi) = exp(2*xi) - 1 has the same ZERO as NH (at xi=0),
    but grows exponentially for large positive xi (much stronger damping).
    For large negative xi (slow particles), g < 0 (heating).

    Note: g is UNBOUNDED (like NH), not bounded like Log-Osc.
    But it has the correct fixed point at |p| = sqrt(kT).
    """
    name = "esh_thermostat"

    def __init__(self, dim: int, kT: float = 1.0, mass: float = 1.0, Q: float = 1.0):
        self.dim = dim
        self.kT = kT
        self.mass = mass
        self.Q = Q

    def g_func(self, xi_val: float) -> float:
        """g(xi) = exp(2*xi) - 1"""
        return np.exp(2.0 * np.clip(xi_val, -10, 10)) - 1.0

    def initial_state(self, q0: np.ndarray, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        xi0 = np.array([0.0])
        return ThermostatState(q0.copy(), p0, xi0, 0)

    def dqdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        return state.p / self.mass

    def dpdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        xi_val = state.xi[0]
        return -grad_U - self.g_func(xi_val) * state.p

    def dxidt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        kinetic = np.sum(state.p ** 2) / self.mass
        return np.array([(kinetic - self.dim * self.kT) / self.Q])


class ESHThermostatVerlet:
    """Velocity Verlet for ESH thermostat.

    Uses exp(-g(xi)*dt/2) momentum rescaling like Log-Osc,
    but g(xi) = exp(2*xi) - 1 can be large — need clamping for stability.
    """
    def __init__(self, dynamics: ESHThermostat, potential, dt: float,
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

        if self._cached_grad_U is not None:
            grad_U = self._cached_grad_U
        else:
            grad_U = self.potential.gradient(q)
            n_evals += 1

        # Half-step thermostat
        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        # Half-step momenta: friction rescaling + kick
        gxi = self.dynamics.g_func(xi[0])
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
        gxi = self.dynamics.g_func(xi[0])
        scale = np.exp(-gxi * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale

        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        self._cached_grad_U = grad_U
        return ThermostatState(q, p, xi, n_evals)


# ============================================================
# Log-Osc Thermostat (from log-osc-001 orbit)
# ============================================================

def g_losc(xi_val):
    return 2.0 * xi_val / (1.0 + xi_val**2)


class LogOscThermostat:
    name = "log_osc"

    def __init__(self, dim: int, kT: float = 1.0, mass: float = 1.0, Q: float = 1.0):
        self.dim = dim
        self.kT = kT
        self.mass = mass
        self.Q = Q

    def initial_state(self, q0, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        return ThermostatState(q0.copy(), p0, np.array([0.0]), 0)

    def dqdt(self, state, grad_U):
        return state.p / self.mass

    def dpdt(self, state, grad_U):
        return -grad_U - g_losc(state.xi[0]) * state.p

    def dxidt(self, state, grad_U):
        kinetic = np.sum(state.p**2) / self.mass
        return np.array([(kinetic - self.dim * self.kT) / self.Q])


class LogOscVerlet:
    def __init__(self, dynamics, potential, dt, kT=1.0, mass=1.0):
        self.dynamics = dynamics
        self.potential = potential
        self.dt = dt
        self.kT = kT
        self.mass = mass
        self._cached_grad_U = None

    def step(self, state):
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

        gxi = g_losc(xi[0])
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
        gxi = g_losc(xi[0])
        scale = np.exp(-gxi * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale

        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        self._cached_grad_U = grad_U
        return ThermostatState(q, p, xi, n_evals)


# ============================================================
# Trajectory computation
# ============================================================

def run_trajectory(dynamics, integrator_cls, potential, dt, n_steps, q0=None):
    """Run a trajectory and collect (q, p, xi) history."""
    if q0 is None:
        q0 = np.array([1.0])
    rng = np.random.default_rng(42)
    state = dynamics.initial_state(q0, rng=rng)
    integrator = integrator_cls(dynamics, potential, dt)

    qs, ps, xis = [], [], []
    for _ in range(n_steps):
        state = integrator.step(state)
        if np.any(np.isnan(state.q)) or np.any(np.isnan(state.p)):
            break
        qs.append(state.q[0])
        ps.append(state.p[0])
        xis.append(state.xi[0])

    return np.array(qs), np.array(ps), np.array(xis)


# ============================================================
# Main comparison
# ============================================================

def main():
    print("=" * 70)
    print("ESH vs THERMOSTAT: NUMERICAL COMPARISON")
    print("=" * 70)

    kT = 1.0
    dt = 0.01
    n_traj_steps = 50000
    n_force_evals_gmm = 500_000
    rng = np.random.default_rng(42)

    pot_ho = HarmonicOscillator1D(omega=1.0)
    pot_gmm = GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5)

    results = {}

    # --------------------------------------------------------
    # 1. Phase-space trajectories on 1D HO
    # --------------------------------------------------------
    print("\n--- 1D HO Phase-Space Trajectories ---")

    esh1d = ESHDynamics1D(kT=kT)
    losc = LogOscThermostat(dim=1, kT=kT, Q=1.0)
    esh_thermo = ESHThermostat(dim=1, kT=kT, Q=1.0)
    nhc = NoseHooverChain(dim=1, chain_length=3, kT=kT, Q=1.0)

    q_esh, p_esh, xi_esh = run_trajectory(esh1d, ESHLeapfrog1D, pot_ho, dt=0.05, n_steps=n_traj_steps, q0=np.array([1.0]))
    q_losc, p_losc, xi_losc = run_trajectory(losc, LogOscVerlet, pot_ho, dt=dt, n_steps=n_traj_steps, q0=np.array([1.0]))
    q_esh_thermo, p_esh_thermo, xi_esh_thermo = run_trajectory(esh_thermo, ESHThermostatVerlet, pot_ho, dt=dt, n_steps=n_traj_steps, q0=np.array([1.0]))

    # Run NHC trajectory using our framework
    nhc_state = nhc.initial_state(np.array([1.0]), rng=np.random.default_rng(42))
    from research.eval.integrators import VelocityVerletThermostat
    nhc_integrator = VelocityVerletThermostat(nhc, pot_ho, dt=dt)
    q_nhc, p_nhc = [], []
    for _ in range(n_traj_steps):
        nhc_state = nhc_integrator.step(nhc_state)
        if not (np.isnan(nhc_state.q[0]) or np.isnan(nhc_state.p[0])):
            q_nhc.append(nhc_state.q[0])
            p_nhc.append(nhc_state.p[0])
    q_nhc, p_nhc = np.array(q_nhc), np.array(p_nhc)

    print(f"  ESH 1D:       {len(q_esh)} steps, q in [{q_esh.min():.2f}, {q_esh.max():.2f}], |p| in [{np.abs(p_esh).min():.3f}, {np.abs(p_esh).max():.3f}]")
    print(f"  Log-Osc:      {len(q_losc)} steps, q in [{q_losc.min():.2f}, {q_losc.max():.2f}], p in [{p_losc.min():.2f}, {p_losc.max():.2f}]")
    print(f"  ESH-Thermo:   {len(q_esh_thermo)} steps, q in [{q_esh_thermo.min():.2f}, {q_esh_thermo.max():.2f}], p in [{p_esh_thermo.min():.2f}, {p_esh_thermo.max():.2f}]")
    print(f"  NHC:          {len(q_nhc)} steps, q in [{q_nhc.min():.2f}, {q_nhc.max():.2f}], p in [{p_nhc.min():.2f}, {p_nhc.max():.2f}]")

    results['trajectories'] = {
        'esh': {'q': q_esh.tolist(), 'p': p_esh.tolist(), 'xi': xi_esh.tolist()},
        'log_osc': {'q': q_losc.tolist(), 'p': p_losc.tolist(), 'xi': xi_losc.tolist()},
        'esh_thermo': {'q': q_esh_thermo.tolist(), 'p': p_esh_thermo.tolist(), 'xi': xi_esh_thermo.tolist()},
        'nhc': {'q': q_nhc.tolist(), 'p': p_nhc.tolist()},
    }

    # --------------------------------------------------------
    # 2. Stationary distribution comparison
    # --------------------------------------------------------
    print("\n--- Stationary Distribution Analysis (1D HO) ---")

    # Theoretical marginals for kT=1, omega=1:
    sigma_q = 1.0  # sqrt(kT/omega^2)
    sigma_p = 1.0  # sqrt(m*kT)

    for name, q_arr, p_arr in [
        ("ESH", q_esh, p_esh),
        ("Log-Osc", q_losc, p_losc),
        ("ESH-Thermo", q_esh_thermo, p_esh_thermo),
        ("NHC", q_nhc, p_nhc),
    ]:
        if len(q_arr) < 100:
            continue
        q_arr_full = np.array(q_arr)
        p_arr_full = np.array(p_arr)

        var_q = np.var(q_arr_full)
        var_p = np.var(p_arr_full)
        mean_q = np.mean(q_arr_full)
        mean_p = np.mean(p_arr_full)

        # KL divergence in q
        kl_q = kl_divergence_histogram(q_arr_full[:, np.newaxis], pot_ho, kT=1.0)

        print(f"  {name:12s}: mean_q={mean_q:.3f}, var_q={var_q:.3f} (target=1.0), "
              f"var_p={var_p:.3f} (target=1.0), KL_q={kl_q:.4f}")

    results['ho_stats'] = {
        'esh': {'var_q': float(np.var(q_esh)), 'var_p': float(np.var(p_esh)),
                'kl_q': float(kl_divergence_histogram(q_esh[:, np.newaxis], pot_ho, kT=1.0)) if len(q_esh) > 100 else None},
        'log_osc': {'var_q': float(np.var(q_losc)), 'var_p': float(np.var(p_losc)),
                    'kl_q': float(kl_divergence_histogram(q_losc[:, np.newaxis], pot_ho, kT=1.0)) if len(q_losc) > 100 else None},
        'esh_thermo': {'var_q': float(np.var(q_esh_thermo)), 'var_p': float(np.var(p_esh_thermo)),
                       'kl_q': float(kl_divergence_histogram(q_esh_thermo[:, np.newaxis], pot_ho, kT=1.0)) if len(q_esh_thermo) > 100 else None},
        'nhc': {'var_q': float(np.var(q_nhc)), 'var_p': float(np.var(p_nhc)),
                'kl_q': float(kl_divergence_histogram(q_nhc[:, np.newaxis], pot_ho, kT=1.0)) if len(q_nhc) > 100 else None},
    }

    # --------------------------------------------------------
    # 3. GMM KL comparison
    # --------------------------------------------------------
    print("\n--- GMM 2D KL Divergence Comparison ---")

    # Note: ESH 1D doesn't apply to 2D GMM directly.
    # We compare ESH-Thermo (2D), Log-Osc (2D), NHC (2D)
    gmm_results = {}

    for sampler_name, dynamics_cls, integrator_cls_arg in [
        ("NHC", lambda: NoseHooverChain(dim=2, chain_length=3, kT=kT, Q=1.0), None),
        ("Log-Osc", lambda: LogOscThermostat(dim=2, kT=kT, Q=1.0), LogOscVerlet),
        ("ESH-Thermo", lambda: ESHThermostat(dim=2, kT=kT, Q=1.0), ESHThermostatVerlet),
    ]:
        dyn = dynamics_cls()
        result = run_sampler(
            dyn, pot_gmm,
            dt=0.01,
            n_force_evals=n_force_evals_gmm,
            kT=kT,
            rng=np.random.default_rng(42),
            integrator_cls=integrator_cls_arg,
        )
        kl = result['kl_divergence']
        ess = result['ess_metrics']['ess_per_force_eval'] if result['ess_metrics'] else None
        print(f"  {sampler_name:12s}: KL={kl:.4f}, ESS/eval={ess:.6f}" if ess else f"  {sampler_name:12s}: KL={kl:.4f}")
        gmm_results[sampler_name] = {'kl': kl, 'ess_per_eval': ess}

    results['gmm_kl'] = gmm_results

    # --------------------------------------------------------
    # 4. Friction function comparison
    # --------------------------------------------------------
    print("\n--- Friction Function g(xi) Comparison ---")

    xi_vals = np.linspace(-3, 3, 200)
    g_losc_vals = 2.0 * xi_vals / (1.0 + xi_vals**2)
    g_nh_vals = xi_vals
    g_esh_thermo_vals = np.exp(2.0 * np.clip(xi_vals, -5, 5)) - 1.0
    g_tanh_vals = np.tanh(xi_vals)

    print(f"  NH g(xi) range: [{g_nh_vals.min():.2f}, {g_nh_vals.max():.2f}] (UNBOUNDED)")
    print(f"  Log-Osc g(xi) range: [{g_losc_vals.min():.2f}, {g_losc_vals.max():.2f}] (BOUNDED in [-1, 1])")
    print(f"  Tanh g(xi) range: [{g_tanh_vals.min():.2f}, {g_tanh_vals.max():.2f}] (BOUNDED in [-1, 1])")
    print(f"  ESH-Thermo g(xi) range: [{g_esh_thermo_vals.min():.2f}, {g_esh_thermo_vals.max():.2f}] (UNBOUNDED, exp-growing)")

    results['friction'] = {
        'xi': xi_vals.tolist(),
        'g_losc': g_losc_vals.tolist(),
        'g_nh': g_nh_vals.tolist(),
        'g_esh_thermo': g_esh_thermo_vals.tolist(),
        'g_tanh': g_tanh_vals.tolist(),
    }

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    out_dir = Path(WORKTREE) / 'orbits/esh-duality-018'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'comparison_results.json'

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    # Save only summary stats (trajectories are too large)
    summary = {
        'ho_stats': results['ho_stats'],
        'gmm_kl': results['gmm_kl'],
        'friction': {k: v for k, v in results['friction'].items()},
    }
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")

    # Also save trajectory arrays separately
    np.savez(
        out_dir / 'trajectories.npz',
        q_esh=q_esh, p_esh=p_esh, xi_esh=xi_esh,
        q_losc=q_losc, p_losc=p_losc, xi_losc=xi_losc,
        q_esh_thermo=q_esh_thermo, p_esh_thermo=p_esh_thermo, xi_esh_thermo=xi_esh_thermo,
        q_nhc=q_nhc, p_nhc=p_nhc,
    )
    print(f"Trajectories saved to {out_dir}/trajectories.npz")

    return results


if __name__ == "__main__":
    results = main()
    print("\nDone.")
