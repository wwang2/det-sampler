"""Option C: Alternating ESH steps + 1/f thermostat steps.

ESH dynamics:
  dx/dt = sign(v)                [unit-speed position update]
  dv/dt = -dU/dx * |v|           [force scaled by speed]
  H_ESH = U(x) + log|v|         [conserved]

These are NOT a valid sampler alone (conservative).
Stochastic refreshment of v ~ N(0,1) makes it sample Boltzmann.

Option C replaces stochastic refreshment with our log-osc 1/f thermostat.
  - ESH steps: deterministic, fast local exploration along H_ESH level sets
  - Thermostat steps: inject/remove energy, provide global thermalization
  - Combined: ergodic by mixing argument (thermostat guarantees canonical marginal)

Benchmark: 2D GMM KL vs MultiScaleNHCTail champion (KL=0.054).
"""
from __future__ import annotations
import sys
import importlib.util
import numpy as np
import json
from pathlib import Path

WORKTREE = '/Users/wujiewang/code/det-sampler/.worktrees/esh-hybrid-026'
sys.path.insert(0, WORKTREE)

from research.eval.integrators import ThermostatState
from research.eval.potentials import GaussianMixture2D

spec = importlib.util.spec_from_file_location(
    'sol009', f'{WORKTREE}/orbits/multiscale-chain-009/solution.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
MultiScaleNHCTail = mod.MultiScaleNHCTail
MultiScaleNHCTailVerlet = mod.MultiScaleNHCTailVerlet

ORBIT_DIR = Path(f'{WORKTREE}/orbits/esh-hybrid-026')


# ─────────────────────────────────────────────────────────
# ESH integrator (split-step leapfrog for ESH + thermostat)
# ─────────────────────────────────────────────────────────

class ESHHybridIntegrator:
    """Option C: half-step thermostat / full-step ESH / half-step thermostat.

    ESH force: f_v = -dU/dx * |v|   (scaled by |v|)
    Thermostat: standard log-osc via MultiScaleNHCTail dynamics

    Note: ESH force uses |v| as mass-like factor. We implement:
      v_half = v + 0.5*dt*(-dU/dx * |v|)  (ESH force)
      x_new  = x + dt * sign(v_half)       (ESH position: unit speed)
      v_new  = v_half + 0.5*dt*(-dU/dx_new * |v_half|)
    Then a full thermostat step to update xi and apply friction to v.

    The thermostat step follows the multi-scale NHC tail dynamics but
    applied to the post-ESH velocity (thermostat sees actual |p|^2, not ESH scaled).
    """

    def __init__(self, dynamics, potential, dt: float, kT: float,
                 n_esh_per_thermo: int = 5):
        self.dynamics = dynamics
        self.potential = potential
        self.dt = dt
        self.kT = kT
        self.n_esh_per_thermo = n_esh_per_thermo
        self._grad_cache = {}

    def _grad(self, q: np.ndarray) -> np.ndarray:
        key = q.tobytes()
        if key not in self._grad_cache:
            self._grad_cache = {key: self.potential.gradient(q)}
        return self._grad_cache[key]

    def step(self, state: ThermostatState) -> ThermostatState:
        q, v, xi, n_evals = state.q, state.p, state.xi, state.n_force_evals
        dt = self.dt

        # ── N_ESH ESH leapfrog half-steps ──
        for _ in range(self.n_esh_per_thermo):
            grad_q = self._grad(q)
            n_evals += 1

            speed = np.abs(v)
            # Half-kick: ESH force scales by |v|
            v_half = v + 0.5 * dt * (-grad_q * speed)
            # Full drift: unit speed
            q_new = q + dt * np.sign(v_half)
            # Update gradient
            grad_new = self.potential.gradient(q_new)
            n_evals += 1
            speed_half = np.abs(v_half)
            v_new = v_half + 0.5 * dt * (-grad_new * speed_half)

            q, v = q_new, v_new

        # ── Thermostat step: update xi, apply friction to v ──
        # Use the NHC-tail thermostat equations directly
        # dxi_k/dt = (v^2 - kT) / Q_k  [for each parallel thermostat]
        # dv/dt = -g_total(xi_k) * v    [friction from all thermostats]
        g_total = self._thermostat_friction(xi)
        v_thermo = v * np.exp(-g_total * dt)

        # Update xi for each thermostat in parallel
        kinetic = np.sum(v**2)  # use pre-friction v for xi update
        xi_new = xi.copy()
        for k, Q_k in enumerate(self.dynamics.Qs_flat):
            xi_new[k] += dt * (kinetic - self.kT) / Q_k

        return ThermostatState(q, v_thermo, xi_new, n_evals)

    def _thermostat_friction(self, xi: np.ndarray) -> float:
        """Total friction coefficient from all parallel log-osc thermostats."""
        g = 0.0
        # Multi-scale: each thermostat contributes g_k(xi_k) = 2*xi_k / (1 + xi_k^2)
        # (bounded log-osc friction)
        for k in range(len(xi)):
            g += 2.0 * xi[k] / (1.0 + xi[k]**2)
        return g


class ESHHybridDynamics:
    """Minimal wrapper so we can call .initial_state() like other samplers."""
    name = "esh_hybrid"

    def __init__(self, dim: int, kT: float, Qs: list[float], chain_length: int = 1):
        self.dim = dim
        self.kT = kT
        self.Qs = Qs
        self.chain_length = chain_length
        # Flat Q list for parallel thermostats (N * chain_length total)
        self.Qs_flat = []
        for Q in Qs:
            for _ in range(chain_length):
                self.Qs_flat.append(Q)
        self.n_xi = len(self.Qs_flat)

    def initial_state(self, q0: np.ndarray, rng=None) -> ThermostatState:
        if rng is None:
            rng = np.random.default_rng(42)
        v0 = rng.normal(0, np.sqrt(self.kT), size=self.dim)
        xi0 = np.zeros(self.n_xi)
        return ThermostatState(q0.copy(), v0, xi0, 0)


# ─────────────────────────────────────────────────────────
# KL divergence evaluation on 2D GMM
# ─────────────────────────────────────────────────────────

def gmm_kl_divergence(q_samples: np.ndarray, potential: GaussianMixture) -> float:
    """KL(empirical || target) via kernel density estimate on a grid."""
    x_min, x_max = -6, 6
    n_grid = 80
    xx = np.linspace(x_min, x_max, n_grid)
    yy = np.linspace(x_min, x_max, n_grid)
    XY = np.stack(np.meshgrid(xx, yy), axis=-1).reshape(-1, 2)  # (n_grid^2, 2)

    # True distribution: exp(-U(q)/kT)
    log_true = np.array([-potential.energy(xy) / potential.kT for xy in XY])
    log_true -= np.max(log_true)
    p_true = np.exp(log_true)
    p_true /= p_true.sum()

    # Empirical distribution via KDE with bandwidth h
    n_samples = len(q_samples)
    h = 1.06 * np.std(q_samples) * n_samples**(-0.2)  # Silverman bandwidth
    h = max(h, 0.1)

    # Subsample for speed (use up to 20k samples)
    if n_samples > 20000:
        idx = np.random.choice(n_samples, 20000, replace=False)
        q_sub = q_samples[idx]
    else:
        q_sub = q_samples

    # KDE: p_emp(x) = (1/n) sum_i K_h(x - q_i)
    diff = XY[:, np.newaxis, :] - q_sub[np.newaxis, :, :]  # (grid, n, 2)
    sq_dist = np.sum(diff**2, axis=-1)                       # (grid, n)
    log_kde = -0.5 * sq_dist / h**2
    log_kde -= np.log(2 * np.pi * h**2)
    p_emp = np.mean(np.exp(log_kde), axis=1)
    p_emp = np.maximum(p_emp, 1e-300)
    p_emp /= p_emp.sum()

    # KL(emp || true)
    mask = p_emp > 1e-10
    kl = float(np.sum(p_emp[mask] * np.log(p_emp[mask] / p_true[mask])))
    return kl


def run_sampler_gmm(dynamics_or_integrator, potential, n_force_evals, dt, kT, seed=42,
                    integrator_cls=None, n_esh_per_thermo=5):
    """Run sampler, return q samples after burn-in."""
    rng = np.random.default_rng(seed)
    q0 = rng.normal(0, 1.0, size=potential.dim)
    state = dynamics_or_integrator.initial_state(q0, rng)

    if integrator_cls is not None:
        integrator = integrator_cls(dynamics_or_integrator, potential, dt=dt, kT=kT)
    else:
        # ESH hybrid
        integrator = ESHHybridIntegrator(dynamics_or_integrator, potential, dt=dt, kT=kT,
                                          n_esh_per_thermo=n_esh_per_thermo)

    burn_evals = n_force_evals // 5
    evals_done = 0
    while evals_done < burn_evals:
        state = integrator.step(state)
        evals_done = state.n_force_evals

    qs = []
    evals_done = 0
    while evals_done < n_force_evals:
        state = integrator.step(state)
        evals_done = state.n_force_evals
        if not np.any(np.isnan(state.q)):
            qs.append(state.q.copy())

    return np.array(qs)


def main():
    kT = 1.0
    n_force_evals = 500_000
    dt = 0.03

    potential = GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5)
    potential.kT = kT  # attach kT for KL computation

    print("=" * 70)
    print("BENCHMARK: ESH Hybrid vs MultiScaleNHCTail Champion on 2D GMM")
    print("=" * 70)
    print(f"n_force_evals = {n_force_evals:,}, dt = {dt}")
    print()

    results = {}

    # ── MultiScaleNHCTail champion ──
    print("Running MultiScaleNHCTail champion (Qs=[0.1, 0.7, 10.0])... ", end='', flush=True)
    ms_champion = MultiScaleNHCTail(dim=2, kT=kT, Qs=[0.1, 0.7, 10.0], chain_length=2)
    q_ms = run_sampler_gmm(ms_champion, potential, n_force_evals, dt, kT, seed=42,
                            integrator_cls=MultiScaleNHCTailVerlet)
    kl_ms = gmm_kl_divergence(q_ms, potential)
    print(f"KL = {kl_ms:.4f}")
    results['champion'] = {'kl': kl_ms, 'n_samples': len(q_ms),
                           'Qs': [0.1, 0.7, 10.0], 'type': 'MultiScaleNHCTail'}

    # ── ESH Hybrid with 1/f thermostat (Qs=[0.1, 0.7, 10.0]) ──
    for n_esh in [3, 5, 10]:
        name = f'ESH_hybrid_N{n_esh}'
        print(f"Running ESH Hybrid (n_esh_per_thermo={n_esh})... ", end='', flush=True)
        esh_dyn = ESHHybridDynamics(dim=2, kT=kT, Qs=[0.1, 0.7, 10.0], chain_length=2)
        q_esh = run_sampler_gmm(esh_dyn, potential, n_force_evals, dt * 0.5, kT, seed=42,
                                 n_esh_per_thermo=n_esh)
        kl_esh = gmm_kl_divergence(q_esh, potential)
        print(f"KL = {kl_esh:.4f}")
        results[name] = {'kl': kl_esh, 'n_samples': len(q_esh),
                         'n_esh_per_thermo': n_esh, 'type': 'ESH_hybrid'}

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Config':<30}  {'KL div':<10}  {'vs champion'}")
    print("-" * 55)
    champ_kl = results['champion']['kl']
    for name, r in results.items():
        delta = r['kl'] - champ_kl
        marker = " <-- BETTER" if delta < -0.005 else (" <-- similar" if abs(delta) < 0.01 else "")
        print(f"{name:<30}  {r['kl']:<10.4f}  {delta:+.4f}{marker}")

    print()
    print(f"Champion KL = {champ_kl:.4f}")
    best_esh_kl = min(r['kl'] for n, r in results.items() if 'ESH' in n)
    print(f"Best ESH hybrid KL = {best_esh_kl:.4f}")
    if best_esh_kl < champ_kl:
        print(f"RESULT: ESH hybrid BEATS champion by {champ_kl - best_esh_kl:.4f}")
    elif best_esh_kl < champ_kl * 1.5:
        print(f"RESULT: ESH hybrid is competitive (within 50% of champion)")
    else:
        print(f"RESULT: ESH hybrid underperforms champion — scale-free KE alone insufficient")

    out_path = ORBIT_DIR / 'esh_hybrid_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    return results


if __name__ == '__main__':
    main()
