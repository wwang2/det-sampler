"""Task 2: Anisotropic Gaussian (d=20) with curvature spanning 3 orders of magnitude.

kappa_i = 10^(i/d * 3) for i=0,...,d-1  (curvatures from 1 to 1000)
U(q) = 0.5 * sum(kappa_i * q_i^2)

Tests multi-scale coverage across timescales. NHC with fixed Q handles one
timescale; multi-scale should cover the range.

Metric: fraction of dims with var(q_i) within 20% of kT/kappa_i
"""

from __future__ import annotations
import sys
import importlib.util
import numpy as np
import json

WORKTREE = '/Users/wujiewang/code/det-sampler/.worktrees/high-dim-022'
sys.path.insert(0, WORKTREE)

from research.eval.baselines import NoseHooverChain
from research.eval.integrators import VelocityVerletThermostat, ThermostatState

spec = importlib.util.spec_from_file_location(
    'sol009', f'{WORKTREE}/orbits/multiscale-chain-009/solution.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
MultiScaleNHCTail = mod.MultiScaleNHCTail
MultiScaleNHCTailVerlet = mod.MultiScaleNHCTailVerlet


class AnisotropicGaussian:
    """d-dimensional anisotropic Gaussian.

    kappa_i = 10^(i/d * 3)  for i=0,...,d-1
    U(q) = 0.5 * sum_i(kappa_i * q_i^2)
    Exact marginal: q_i ~ N(0, kT/kappa_i)
    """
    def __init__(self, d=20, kT=1.0):
        self.d = d
        self.dim = d
        self.kT = kT
        self.kappas = np.array([10.0 ** (i / d * 3.0) for i in range(d)])

    def energy(self, q):
        return 0.5 * np.dot(self.kappas, q**2)

    def gradient(self, q):
        return self.kappas * q


def run_sampler_full(dynamics, integrator_cls, potential, n_force_evals, dt, kT, seed=42):
    """Run sampler, return q and p samples after burn-in."""
    rng = np.random.default_rng(seed)
    q0 = rng.normal(0, 1.0 / np.sqrt(potential.kappas))  # init near truth
    state = dynamics.initial_state(q0, rng)

    integrator = integrator_cls(dynamics, potential, dt=dt, kT=kT)

    # Burn-in: 20% of budget (anisotropic needs more warm-up)
    burn_evals = n_force_evals // 5
    evals_done = 0
    while evals_done < burn_evals:
        state = integrator.step(state)
        evals_done = state.n_force_evals

    integrator._cached_grad_U = None
    qs, ps = [], []
    evals_done = 0
    while evals_done < n_force_evals:
        state = integrator.step(state)
        evals_done = state.n_force_evals
        if not np.any(np.isnan(state.q)):
            qs.append(state.q.copy())
            ps.append(state.p.copy())

    return np.array(qs), np.array(ps)


def ergodicity_score_anisotropic(q_samples, kappas, kT=1.0, tol=0.20):
    """Fraction of dims where var(q_i) is within tol of kT/kappa_i."""
    expected_var = kT / kappas
    observed_var = np.var(q_samples, axis=0)
    rel_err = np.abs(observed_var - expected_var) / expected_var
    good = rel_err < tol
    score = np.mean(good)
    return score, observed_var, expected_var, rel_err


def main():
    kT = 1.0
    n_force_evals = 1_000_000
    d = 20

    potential = AnisotropicGaussian(d=d, kT=kT)
    kappas = potential.kappas

    print("=" * 65)
    print("Task 2: Anisotropic Gaussian d=20, curvature ratio 1:1000")
    print("=" * 65)
    print(f"kappa range: {kappas[0]:.3f} to {kappas[-1]:.1f}")
    print(f"Force evals: {n_force_evals:,}, kT={kT}")
    print()

    # Choose dt small enough for stiffest dimension: omega_max = sqrt(kappa_max)
    # dt < 0.1/omega_max
    omega_max = np.sqrt(kappas[-1])  # ~31.6
    dt = min(0.02, 0.05 / omega_max)
    print(f"Using dt={dt:.5f} (omega_max={omega_max:.2f})")
    print()

    results = {}

    # --- NHC (M=3), Q tuned to middle timescale ---
    # Natural Q: mass * omega^2 related, use Q=1.0 (standard)
    nhc = NoseHooverChain(dim=d, chain_length=3, kT=kT, Q=1.0)
    q_nhc, p_nhc = run_sampler_full(nhc, VelocityVerletThermostat,
                                     potential, n_force_evals, dt, kT, seed=42)
    score_nhc, obs_nhc, exp_var, rel_nhc = ergodicity_score_anisotropic(q_nhc, kappas, kT)

    # --- MultiScaleNHCTail, Qs=[0.1,0.7,10.0], chain_length=2 ---
    ms = MultiScaleNHCTail(dim=d, kT=kT, Qs=[0.1, 0.7, 10.0], chain_length=2)
    q_ms, p_ms = run_sampler_full(ms, MultiScaleNHCTailVerlet,
                                   potential, n_force_evals, dt, kT, seed=42)
    score_ms, obs_ms, _, rel_ms = ergodicity_score_anisotropic(q_ms, kappas, kT)

    print(f"NHC(M=3):          ergodicity={score_nhc:.4f}  "
          f"(frac dims within 20% of truth)")
    print(f"MultiScaleNHCTail: ergodicity={score_ms:.4f}  "
          f"(frac dims within 20% of truth)")
    print(f"\nDelta (MS - NHC): {score_ms - score_nhc:+.4f}")
    print()

    # Per-dimension breakdown
    print("Per-dim relative error (first 5 and last 5):")
    print(f"  dim  kappa   NHC_relerr  MS_relerr  expected_var  nhc_var     ms_var")
    for i in list(range(5)) + list(range(d-5, d)):
        print(f"  {i:3d}  {kappas[i]:7.2f}  {rel_nhc[i]:.4f}      {rel_ms[i]:.4f}  "
              f"  {exp_var[i]:.5f}      {obs_nhc[i]:.5f}  {obs_ms[i]:.5f}")

    results = {
        'nhc_score': float(score_nhc),
        'ms_score': float(score_ms),
        'delta': float(score_ms - score_nhc),
        'kappas': kappas.tolist(),
        'expected_var': exp_var.tolist(),
        'nhc_obs_var': obs_nhc.tolist(),
        'ms_obs_var': obs_ms.tolist(),
        'nhc_rel_err': rel_nhc.tolist(),
        'ms_rel_err': rel_ms.tolist(),
        'n_samples_nhc': len(q_nhc),
        'n_samples_ms': len(q_ms),
        'dt': dt,
        'd': d,
    }

    out_path = f'{WORKTREE}/orbits/high-dim-022/results_anisotropic.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == '__main__':
    main()
