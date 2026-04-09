"""Task 3: ESH Hybrid Benchmark vs NHC and NHCTail.

Compares on 2D GMM (5 modes) at 500k force evaluations, 3 seeds:
  - NHC(M=3): baseline
  - NHCTail(Qs=[0.1,0.7,10]): champion (GMM KL=0.054)
  - ESH pure + periodic velocity refreshment (every 100 steps)
  - ESH hybrid (Option C): alternating ESH+thermostat

Metric: GMM KL divergence at 500k force evaluations.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

WORKTREE = '/Users/wujiewang/code/det-sampler/.worktrees/esh-hybrid-026'
sys.path.insert(0, WORKTREE)

from research.eval.potentials import GaussianMixture2D
from research.eval.baselines import NoseHooverChain
from research.eval.evaluator import run_sampler, kl_divergence_histogram
from research.eval.integrators import ThermostatState, VelocityVerletThermostat

# Import from sibling orbit
sys.path.insert(0, str(Path(WORKTREE) / 'orbits/multiscale-chain-009'))
from solution import MultiScaleNHCTail, MultiScaleNHCTailVerlet

# Import our hybrid sampler
sys.path.insert(0, str(Path(WORKTREE) / 'orbits/esh-hybrid-026'))
from make_hybrid_sampler import ESHPlusThermostat, ESHHybridIntegrator

OUT_DIR = Path(WORKTREE) / 'orbits/esh-hybrid-026'
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_FORCE_EVALS = 500_000
KT = 1.0
SEEDS = [42, 123, 7]


# ============================================================
# Flat ESH Hybrid integrator: samples every THERMO step
# ============================================================

class ESHHybridFlatIntegrator:
    """ESH Hybrid that exposes individual thermostat steps for dense sampling.

    Internally maintains a cycle counter. Every thermo_per_esh thermostat
    steps, it runs L_esh ESH steps first (the 'ESH burst'). Then resumes
    thermostat steps. This way the evaluator sees one q-sample per force
    eval (like NHCTail), rather than one per cycle.

    Force eval cost:
      - ESH burst: L_esh evals  (amortized over thermo_per_esh steps)
      - Thermostat: 1 eval/step
      Total rate: (L_esh + thermo_per_esh) / thermo_per_esh evals per sample
      For L_esh=10, thermo_per_esh=100: 1.1 evals/sample vs 1.0 for NHCTail
    """

    def __init__(self, dynamics, potential, dt: float,
                 kT: float = 1.0, mass: float = 1.0,
                 L_esh: int = 10, thermo_per_esh: int = 100,
                 dt_esh: float = 0.05, dt_thermo: float = 0.03):
        self.dynamics = dynamics
        self.potential = potential
        self.dt = dt_thermo  # used by evaluator
        self.kT = kT
        self.mass = mass
        self.L_esh = L_esh
        self.thermo_per_esh = thermo_per_esh
        self.dt_esh = dt_esh
        self.dt_thermo = dt_thermo
        self._cached_grad_U = None
        self._thermo_step_count = 0  # steps since last ESH burst

    def _run_esh_burst(self, q, p, xi, n_evals, grad_U):
        """Run L_esh ESH leapfrog steps. Returns updated (q, p, xi, n_evals, grad_U)."""
        dyn = self.dynamics
        for _ in range(self.L_esh):
            if np.any(np.isnan(q)) or np.any(np.isnan(p)):
                return q, p, xi, n_evals, None

            p_norm = np.linalg.norm(p)
            if p_norm < 1e-300:
                p_norm = 1e-300
            half_dt = 0.5 * self.dt_esh

            p_half = p - half_dt * grad_U * (p_norm / dyn.dim)
            p_half_norm = np.linalg.norm(p_half)
            if p_half_norm < 1e-300:
                p_half_norm = 1e-300

            q = q + self.dt_esh * p_half / p_half_norm
            grad_U = self.potential.gradient(q)
            n_evals += 1
            p = p_half - half_dt * grad_U * (p_half_norm / dyn.dim)

        return q, p, xi, n_evals, grad_U

    def step(self, state: ThermostatState) -> ThermostatState:
        """One thermostat step. Runs ESH burst every thermo_per_esh steps."""
        q, p, xi, n_evals = state
        dyn = self.dynamics

        if self._cached_grad_U is not None:
            grad_U = self._cached_grad_U
        else:
            grad_U = self.potential.gradient(q)
            n_evals += 1

        # ESH burst every thermo_per_esh thermostat steps
        if self._thermo_step_count % self.thermo_per_esh == 0:
            q, p, xi, n_evals, grad_U_new = self._run_esh_burst(q, p, xi, n_evals, grad_U)
            if grad_U_new is not None:
                grad_U = grad_U_new
            else:
                self._cached_grad_U = None
                return ThermostatState(q, p, xi, n_evals)
            # After ESH burst, rescale ||p|| to canonical thermal value sqrt(d*kT).
            # ESH preserves log||p|| not ||p||^2/2 so kinetic energy can drift far
            # from d*kT, causing xi variables to explode over subsequent thermo steps.
            # Rescaling restores the canonical momentum magnitude while keeping the
            # direction (from the ESH exploration) intact.
            p_norm = np.linalg.norm(p)
            if p_norm > 1e-300:
                p_target = np.sqrt(dyn.dim * self.kT)
                p = p * (p_target / p_norm)

        # One thermostat Verlet step
        half_dt = 0.5 * self.dt_thermo

        xi_dot = dyn._dxi_dt(p, xi)
        xi = xi + half_dt * xi_dot

        total_g = dyn._total_friction(xi)
        scale = np.exp(-total_g * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale
        p = p - half_dt * grad_U

        q = q + self.dt_thermo * p / dyn.mass

        if np.any(np.isnan(q)) or np.any(np.isnan(p)):
            self._cached_grad_U = None
            return ThermostatState(q, p, xi, n_evals)

        grad_U = self.potential.gradient(q)
        n_evals += 1

        p = p - half_dt * grad_U
        total_g = dyn._total_friction(xi)
        scale = np.exp(-total_g * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale

        xi_dot = dyn._dxi_dt(p, xi)
        xi = xi + half_dt * xi_dot

        self._thermo_step_count += 1
        self._cached_grad_U = grad_U
        return ThermostatState(q, p, xi, n_evals)


# ============================================================
# ESH + periodic velocity refreshment
# ============================================================

class ESHWithRefresh:
    """ESH dynamics with periodic velocity refreshment.

    Runs ESH leapfrog steps; every refresh_interval steps, resample
    velocity from N(0, kT*I). This creates a simple stochastic ESH sampler.
    The refreshment breaks H_ESH level sets, enabling ergodic sampling.
    """
    name = "esh_refresh"

    def __init__(self, dim: int, kT: float = 1.0, refresh_interval: int = 100):
        self.dim = dim
        self.kT = kT
        self.refresh_interval = refresh_interval
        self._step_count = 0
        self._rng = np.random.default_rng(42)

    def initial_state(self, q0: np.ndarray, rng=None) -> ThermostatState:
        if rng is not None:
            self._rng = rng
        # Start with |p| = sqrt(kT) (ESH convention)
        p0 = self._rng.normal(0, np.sqrt(self.kT), size=self.dim)
        xi0 = np.array([0.0])  # auxiliary: log(||p||/sqrt(kT))
        return ThermostatState(q0.copy(), p0, xi0, 0)

    def dqdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        p_norm = np.linalg.norm(state.p)
        if p_norm < 1e-300:
            return np.zeros(self.dim)
        return state.p / p_norm

    def dpdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        p_norm = np.linalg.norm(state.p)
        if p_norm < 1e-300:
            return np.zeros(self.dim)
        return -grad_U * p_norm / self.dim

    def dxidt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        return np.array([0.0])


class ESHRefreshIntegrator:
    """ESH leapfrog with periodic velocity refreshment."""

    def __init__(self, dynamics: ESHWithRefresh, potential, dt: float,
                 kT: float = 1.0, mass: float = 1.0):
        self.dynamics = dynamics
        self.potential = potential
        self.dt = dt
        self.kT = kT
        self.mass = mass
        self._cached_grad_U = None
        self._step_count = 0

    def step(self, state: ThermostatState) -> ThermostatState:
        q, p, xi, n_evals = state
        dyn = self.dynamics
        dt = self.dt
        half_dt = 0.5 * dt

        if self._cached_grad_U is not None:
            grad_U = self._cached_grad_U
        else:
            grad_U = self.potential.gradient(q)
            n_evals += 1

        p_norm = np.linalg.norm(p)
        if p_norm < 1e-300:
            p_norm = 1e-300

        # ESH leapfrog half-step p
        p_half = p - half_dt * grad_U * (p_norm / dyn.dim)
        p_half_norm = np.linalg.norm(p_half)
        if p_half_norm < 1e-300:
            p_half_norm = 1e-300

        # Full-step q
        q = q + dt * p_half / p_half_norm

        if np.any(np.isnan(q)):
            self._cached_grad_U = None
            xi = np.log(np.abs(p) / np.sqrt(self.kT) + 1e-300)[:1]
            return ThermostatState(q, p, xi, n_evals)

        grad_U = self.potential.gradient(q)
        n_evals += 1

        # Second half-step p
        p = p_half - half_dt * grad_U * (p_half_norm / dyn.dim)

        self._step_count += 1

        # Periodic velocity refreshment
        if self._step_count % dyn.refresh_interval == 0:
            p = dyn._rng.normal(0, np.sqrt(self.kT), size=dyn.dim)

        xi = np.array([np.log(np.linalg.norm(p) / np.sqrt(self.kT) + 1e-300)])
        self._cached_grad_U = grad_U
        return ThermostatState(q, p, xi, n_evals)


# ============================================================
# Benchmark runner
# ============================================================

def run_benchmark_sampler(name, make_dyn_fn, make_integrator_fn, potential,
                          n_force_evals, kT, seeds):
    """Run a sampler over multiple seeds and return KL results."""
    results = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        q0 = rng.normal(0, 0.5, size=potential.dim)
        dyn = make_dyn_fn(seed=seed)
        integrator = make_integrator_fn(dyn, potential)

        state = dyn.initial_state(q0, rng=rng)

        all_q = []
        burnin_evals = int(n_force_evals * 0.1)
        t_start = time.time()

        while state.n_force_evals < n_force_evals:
            state = integrator.step(state)
            if np.any(np.isnan(state.q)) or np.any(np.isnan(state.p)):
                break
            if state.n_force_evals >= burnin_evals:
                all_q.append(state.q.copy())

        wall_time = time.time() - t_start

        if len(all_q) > 100:
            q_arr = np.array(all_q)
            kl = kl_divergence_histogram(q_arr, potential, kT=kT, n_bins=50)
            kl = max(0.0, kl)
        else:
            kl = float('inf')

        results.append({
            'seed': seed,
            'kl': float(kl),
            'n_samples': len(all_q),
            'n_force_evals': int(state.n_force_evals),
            'wall_seconds': wall_time,
        })
        print(f"    seed={seed}: KL={kl:.4f}, n_samples={len(all_q)}, "
              f"force_evals={state.n_force_evals}, t={wall_time:.1f}s")

    kls = [r['kl'] for r in results if r['kl'] < 1e10]
    summary = {
        'name': name,
        'seeds': results,
        'kl_mean': float(np.mean(kls)) if kls else float('inf'),
        'kl_std': float(np.std(kls)) if len(kls) > 1 else float('nan'),
        'kl_min': float(np.min(kls)) if kls else float('inf'),
    }
    return summary


def main():
    print("=" * 70)
    print("ESH HYBRID BENCHMARK: GMM 2D (5 modes)")
    print(f"Budget: {N_FORCE_EVALS:,} force evaluations, seeds: {SEEDS}")
    print("=" * 70)

    pot_gmm = GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5)

    all_results = {}

    # ---- 1. NHC(M=3): baseline ----
    print("\n[1/4] NHC(M=3): baseline")

    def make_nhc(seed=42):
        return NoseHooverChain(dim=2, chain_length=3, kT=KT, Q=1.0)

    def make_nhc_integrator(dyn, pot):
        return VelocityVerletThermostat(dyn, pot, dt=0.01, kT=KT)

    nhc_result = run_benchmark_sampler(
        "NHC_M3", make_nhc, make_nhc_integrator, pot_gmm,
        N_FORCE_EVALS, KT, SEEDS
    )
    all_results['nhc_m3'] = nhc_result
    print(f"  NHC(M=3): mean KL={nhc_result['kl_mean']:.4f} ± {nhc_result['kl_std']:.4f}")

    # ---- 2. NHCTail champion ----
    print("\n[2/4] NHCTail(Qs=[0.1,0.7,10]): champion")

    def make_nhctail(seed=42):
        return MultiScaleNHCTail(dim=2, kT=KT, Qs=[0.1, 0.7, 10.0], chain_length=2)

    def make_nhctail_integrator(dyn, pot):
        return MultiScaleNHCTailVerlet(dyn, pot, dt=0.03, kT=KT)

    nhctail_result = run_benchmark_sampler(
        "NHCTail", make_nhctail, make_nhctail_integrator, pot_gmm,
        N_FORCE_EVALS, KT, SEEDS
    )
    all_results['nhctail'] = nhctail_result
    print(f"  NHCTail: mean KL={nhctail_result['kl_mean']:.4f} ± {nhctail_result['kl_std']:.4f}")

    # ---- 3. ESH + periodic velocity refreshment ----
    print("\n[3/4] ESH + periodic velocity refreshment (every 100 steps)")

    def make_esh_refresh(seed=42):
        dyn = ESHWithRefresh(dim=2, kT=KT, refresh_interval=100)
        dyn._rng = np.random.default_rng(seed)
        return dyn

    def make_esh_refresh_integrator(dyn, pot):
        return ESHRefreshIntegrator(dyn, pot, dt=0.05, kT=KT)

    esh_refresh_result = run_benchmark_sampler(
        "ESH_refresh", make_esh_refresh, make_esh_refresh_integrator, pot_gmm,
        N_FORCE_EVALS, KT, SEEDS
    )
    all_results['esh_refresh'] = esh_refresh_result
    print(f"  ESH+refresh: mean KL={esh_refresh_result['kl_mean']:.4f} ± {esh_refresh_result['kl_std']:.4f}")

    # ---- 4. ESH Hybrid (Option C, FLAT: samples every thermo step) ----
    # Key fix: use ESHHybridFlatIntegrator so we collect ~1 sample per force eval.
    # The ESH burst runs every 100 thermostat steps (amortized: +10% overhead).
    print("\n[4/4] ESH Hybrid (Option C, flat: ESH burst every 100 thermo steps)")

    def make_esh_hybrid_dyn(seed=42):
        return ESHPlusThermostat(
            dim=2, kT=KT,
            Qs=[0.1, 0.7, 10.0],
            L_esh=10, M_thermo=100,
            dt_esh=0.05, dt_thermo=0.03,
        )

    def make_esh_hybrid_flat_integrator(dyn, pot):
        return ESHHybridFlatIntegrator(
            dyn, pot, dt=0.03, kT=KT,
            L_esh=10, thermo_per_esh=100,
            dt_esh=0.05, dt_thermo=0.03,
        )

    hybrid_result = run_benchmark_sampler(
        "ESH_hybrid_flat", make_esh_hybrid_dyn, make_esh_hybrid_flat_integrator, pot_gmm,
        N_FORCE_EVALS, KT, SEEDS
    )
    all_results['esh_hybrid'] = hybrid_result
    print(f"  ESH hybrid (flat): mean KL={hybrid_result['kl_mean']:.4f} ± {hybrid_result['kl_std']:.4f}")

    # ---- 4b. ESH Hybrid with larger ESH bursts (L=50 every 200 steps) ----
    print("\n[4b] ESH Hybrid (L=50 ESH burst every 200 thermo steps)")

    def make_esh_hybrid_dyn_v2(seed=42):
        return ESHPlusThermostat(
            dim=2, kT=KT,
            Qs=[0.1, 0.7, 10.0],
            L_esh=50, M_thermo=200,
            dt_esh=0.05, dt_thermo=0.03,
        )

    def make_esh_hybrid_flat_v2_integrator(dyn, pot):
        return ESHHybridFlatIntegrator(
            dyn, pot, dt=0.03, kT=KT,
            L_esh=50, thermo_per_esh=200,
            dt_esh=0.05, dt_thermo=0.03,
        )

    hybrid_v2_result = run_benchmark_sampler(
        "ESH_hybrid_L50", make_esh_hybrid_dyn_v2, make_esh_hybrid_flat_v2_integrator, pot_gmm,
        N_FORCE_EVALS, KT, SEEDS
    )
    all_results['esh_hybrid_L50'] = hybrid_v2_result
    print(f"  ESH hybrid (L=50): mean KL={hybrid_v2_result['kl_mean']:.4f} ± {hybrid_v2_result['kl_std']:.4f}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY (GMM KL, 500k force evals)")
    print("=" * 70)
    champion_kl = 0.054  # NHCTail at 1M evals (seed=42)

    rows = [
        ("NHC(M=3)",               all_results['nhc_m3']),
        ("NHCTail champion",        all_results['nhctail']),
        ("ESH+refresh",             all_results['esh_refresh']),
        ("ESH hybrid (L=10,flat)",  all_results['esh_hybrid']),
        ("ESH hybrid (L=50,flat)",  all_results['esh_hybrid_L50']),
    ]

    print(f"  {'Sampler':<26} {'KL mean':>10} {'KL std':>10} {'KL min':>10} {'n_samples':>12}")
    print("  " + "-" * 72)
    for label, res in rows:
        kl_m = res['kl_mean']
        kl_s = res['kl_std']
        kl_n = res['kl_min']
        n_s = int(np.mean([s['n_samples'] for s in res['seeds']]))
        flag = " <-- BEATS CHAMPION" if kl_m < champion_kl else ""
        print(f"  {label:<26} {kl_m:>10.4f} {kl_s:>10.4f} {kl_n:>10.4f} {n_s:>12}{flag}")

    print(f"\n  Reference: NHCTail champion at 1M evals = {champion_kl}")
    print(f"  Note: above results at 500k evals (half the budget)")

    # Determine primary metric (use best hybrid variant)
    hybrid_kl = min(all_results['esh_hybrid']['kl_mean'],
                    all_results['esh_hybrid_L50']['kl_mean'])
    if hybrid_kl < champion_kl:
        verdict = f"MAJOR RESULT: ESH hybrid ({hybrid_kl:.4f}) beats NHCTail champion ({champion_kl})"
    elif hybrid_kl < all_results['nhc_m3']['kl_mean']:
        verdict = f"PARTIAL RESULT: ESH hybrid ({hybrid_kl:.4f}) beats NHC baseline ({all_results['nhc_m3']['kl_mean']:.4f}), not champion"
    else:
        verdict = f"NEGATIVE RESULT: ESH hybrid ({hybrid_kl:.4f}) does not improve over NHC ({all_results['nhc_m3']['kl_mean']:.4f})"

    print(f"\n  {verdict}")

    # Save results
    out_path = OUT_DIR / 'benchmark_results.json'

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(out_path, 'w') as f:
        json.dump({
            'description': 'ESH Hybrid Benchmark: GMM 2D, 500k force evals',
            'n_force_evals': N_FORCE_EVALS,
            'seeds': SEEDS,
            'champion_reference': champion_kl,
            'results': all_results,
            'verdict': verdict,
        }, f, indent=2, default=convert)

    print(f"\nResults saved to {out_path}")
    return all_results


if __name__ == "__main__":
    results = main()
    print("\nBenchmark complete.")
