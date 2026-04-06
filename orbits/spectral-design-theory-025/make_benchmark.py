"""Benchmark: derived Q values vs champion on d=20 anisotropic Gaussian.

Theory (theory.md): for kappa in [kappa_min, kappa_max],
  Q_min = 1/sqrt(kappa_max),  Q_max = 1/sqrt(kappa_min)
  N=3 log-spaced in [Q_min, Q_max].

For d=20 anisotropic with kappa in [1, 1000]:
  Q_min = 1/sqrt(1000) ~ 0.0316
  Q_max = 1/sqrt(1) = 1.0
  Derived Q = [0.0316, 0.178, 1.0]

Champion (from search): Q = [0.1, 0.7, 10.0]

Metric: ergodicity_score = fraction of dims within 20% of true variance.
"""
from __future__ import annotations
import sys
import importlib.util
import numpy as np
import json
from pathlib import Path

WORKTREE = '/Users/wujiewang/code/det-sampler/.worktrees/spectral-design-theory-025'
sys.path.insert(0, WORKTREE)

from research.eval.baselines import NoseHooverChain
from research.eval.integrators import VelocityVerletThermostat, ThermostatState

spec = importlib.util.spec_from_file_location(
    'sol009', f'{WORKTREE}/orbits/multiscale-chain-009/solution.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
MultiScaleNHCTail = mod.MultiScaleNHCTail
MultiScaleNHCTailVerlet = mod.MultiScaleNHCTailVerlet

ORBIT_DIR = Path(f'{WORKTREE}/orbits/spectral-design-theory-025')


class AnisotropicGaussian:
    def __init__(self, d=20, kT=1.0):
        self.d = d
        self.dim = d
        self.kT = kT
        self.kappas = np.array([10.0 ** (i / d * 3.0) for i in range(d)])

    def energy(self, q):
        return 0.5 * np.dot(self.kappas, q**2)

    def gradient(self, q):
        return self.kappas * q


def run_sampler(dynamics, integrator_cls, potential, n_force_evals, dt, kT, seed=42):
    rng = np.random.default_rng(seed)
    q0 = rng.normal(0, 1.0 / np.sqrt(potential.kappas))
    state = dynamics.initial_state(q0, rng)
    integrator = integrator_cls(dynamics, potential, dt=dt, kT=kT)

    # 20% burn-in
    burn_evals = n_force_evals // 5
    evals_done = 0
    while evals_done < burn_evals:
        state = integrator.step(state)
        evals_done = state.n_force_evals

    integrator._cached_grad_U = None
    qs = []
    evals_done = 0
    while evals_done < n_force_evals:
        state = integrator.step(state)
        evals_done = state.n_force_evals
        if not np.any(np.isnan(state.q)):
            qs.append(state.q.copy())
    return np.array(qs)


def ergodicity_score(q_samples, kappas, kT=1.0, tol=0.20):
    expected_var = kT / kappas
    observed_var = np.var(q_samples, axis=0)
    rel_err = np.abs(observed_var - expected_var) / expected_var
    return float(np.mean(rel_err < tol)), observed_var, expected_var, rel_err


def main():
    kT = 1.0
    n_force_evals = 1_000_000
    d = 20

    potential = AnisotropicGaussian(d=d, kT=kT)
    kappas = potential.kappas
    kappa_min, kappa_max = kappas[0], kappas[-1]

    omega_max = np.sqrt(kappa_max)
    dt = min(0.02, 0.05 / omega_max)

    print("=" * 70)
    print("BENCHMARK: Derived Q-values vs Champion on d=20 Anisotropic Gaussian")
    print("=" * 70)
    print(f"kappa range: [{kappa_min:.3f}, {kappa_max:.1f}]")
    print(f"dt = {dt:.5f}, n_force_evals = {n_force_evals:,}")
    print()

    # ---- Theory-derived Q values ----
    Q_min_derived = 1.0 / np.sqrt(kappa_max)   # ~ 0.0316
    Q_max_derived = 1.0 / np.sqrt(kappa_min)   # = 1.0
    # N=3 log-spaced
    Qs_derived = np.exp(np.linspace(np.log(Q_min_derived), np.log(Q_max_derived), 3)).tolist()
    print(f"Derived Q (theory): Q_min={Q_min_derived:.4f}, Q_max={Q_max_derived:.4f}")
    print(f"  N=3 log-spaced: {[f'{q:.4f}' for q in Qs_derived]}")
    print()

    # ---- Champion Q values ----
    Qs_champion = [0.1, 0.7, 10.0]
    print(f"Champion Q (search): {Qs_champion}")
    print()

    configs = {
        'NHC_M3': {'type': 'NHC', 'Q': 1.0, 'chain_length': 3},
        'Derived': {'type': 'MS', 'Qs': Qs_derived, 'chain_length': 2},
        'Champion': {'type': 'MS', 'Qs': Qs_champion, 'chain_length': 2},
    }

    # Additional Q range tests
    for N in [2, 3, 4]:
        Qs_n = np.exp(np.linspace(np.log(Q_min_derived), np.log(Q_max_derived), N)).tolist()
        configs[f'Derived_N{N}'] = {'type': 'MS', 'Qs': Qs_n, 'chain_length': 2}

    # Also test a slightly extended range (Q_max = 3.0 instead of 1.0)
    for Q_max_ext in [3.0, 10.0]:
        Qs_ext = np.exp(np.linspace(np.log(Q_min_derived), np.log(Q_max_ext), 3)).tolist()
        configs[f'Extended_Qmax{int(Q_max_ext)}'] = {'type': 'MS', 'Qs': Qs_ext, 'chain_length': 2}

    results = {}
    for name, cfg in configs.items():
        print(f"Running {name}... ", end='', flush=True)
        if cfg['type'] == 'NHC':
            dynamics = NoseHooverChain(dim=d, chain_length=cfg['chain_length'],
                                       kT=kT, Q=cfg['Q'])
            integrator_cls = VelocityVerletThermostat
        else:
            dynamics = MultiScaleNHCTail(dim=d, kT=kT, Qs=cfg['Qs'],
                                          chain_length=cfg['chain_length'])
            integrator_cls = MultiScaleNHCTailVerlet

        q_samples = run_sampler(dynamics, integrator_cls, potential, n_force_evals, dt, kT, seed=42)
        score, obs_var, exp_var, rel_err = ergodicity_score(q_samples, kappas, kT)
        results[name] = {
            'config': cfg,
            'ergodicity_score': score,
            'obs_var': obs_var.tolist(),
            'rel_err': rel_err.tolist(),
            'n_samples': len(q_samples),
        }
        print(f"ergodicity={score:.4f}")

    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Config':<25}  {'Q values':<30}  {'Ergodicity'}")
    print("-" * 75)
    for name, r in results.items():
        cfg = r['config']
        if cfg['type'] == 'NHC':
            q_str = f"NHC Q={cfg['Q']}"
        else:
            q_str = f"MS Qs={[f'{q:.3f}' for q in cfg['Qs']]}"
        print(f"{name:<25}  {q_str:<30}  {r['ergodicity_score']:.4f}")

    print()
    derived_score = results['Derived']['ergodicity_score']
    champion_score = results['Champion']['ergodicity_score']
    nhc_score = results['NHC_M3']['ergodicity_score']
    print(f"NHC (baseline):   {nhc_score:.4f}")
    print(f"Derived Q:        {derived_score:.4f}  ({derived_score - nhc_score:+.4f} vs NHC)")
    print(f"Champion Q:       {champion_score:.4f}  ({champion_score - nhc_score:+.4f} vs NHC)")
    print(f"Derived vs Champion: {derived_score - champion_score:+.4f}")
    print()
    if derived_score >= champion_score - 0.02:
        print("RESULT: Derived Q matches or beats champion — theory validated!")
    else:
        print(f"RESULT: Champion wins by {champion_score - derived_score:.4f} — "
              f"champion's wider Q range helps (Q_max=10 vs Q_max=1)")

    # Save results
    results['kappas'] = kappas.tolist()
    results['expected_var'] = (kT / kappas).tolist()
    results['dt'] = dt
    results['n_force_evals'] = n_force_evals
    results['derived_Qs'] = Qs_derived
    results['champion_Qs'] = Qs_champion
    results['Q_min_derived'] = Q_min_derived
    results['Q_max_derived'] = Q_max_derived

    out_path = ORBIT_DIR / 'benchmark_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")

    return results


if __name__ == '__main__':
    main()
