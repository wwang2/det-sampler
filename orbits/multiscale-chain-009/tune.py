"""Parameter tuning for best architectures (NHCTail and Hierarchical)."""

import sys
import os
import json
import importlib.util
import numpy as np

WORKTREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WORKTREE)

from research.eval.evaluator import run_sampler
from research.eval.potentials import (
    HarmonicOscillator1D, DoubleWell2D, GaussianMixture2D, Rosenbrock2D
)

_spec = importlib.util.spec_from_file_location(
    "solution",
    os.path.join(WORKTREE, "orbits", "multiscale-chain-009", "solution.py"),
)
_sol = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sol)

N_FORCE_EVALS = 1_000_000
KT = 1.0


def quick_eval(dyn, integrator_cls, pot, dt, seed=42):
    rng = np.random.default_rng(seed)
    q0 = rng.normal(0, 0.5, size=pot.dim)
    r = run_sampler(dyn, pot, dt=dt, n_force_evals=N_FORCE_EVALS,
                    kT=KT, q0=q0, rng=rng, integrator_cls=integrator_cls)
    kl = r.get("kl_divergence", float("inf"))
    if kl is None:
        kl = float("inf")
    erg = r.get("ergodicity", {})
    erg_score = erg.get("score", None) if erg else None
    return kl, erg_score


# ============================================================
# Tune 1: DW dt for NHCTail
# ============================================================
print("=== Tune DW dt for NHCTail (Qs=[0.1, 0.7, 10.0], M=2) ===")
pot_dw = DoubleWell2D()
for dt in [0.02, 0.025, 0.03, 0.035, 0.04]:
    dyn = _sol.MultiScaleNHCTail(dim=2, kT=KT, Qs=[0.1, 0.7, 10.0],
                                   chain_length=2, chain_Q_multiplier=1.0)
    kl, _ = quick_eval(dyn, _sol.MultiScaleNHCTailVerlet, pot_dw, dt)
    print(f"  dt={dt}: DW KL={kl:.4f}")

# ============================================================
# Tune 2: GMM dt for NHCTail
# ============================================================
print("\n=== Tune GMM dt for NHCTail ===")
pot_gmm = GaussianMixture2D()
for dt in [0.02, 0.025, 0.03, 0.035, 0.04]:
    dyn = _sol.MultiScaleNHCTail(dim=2, kT=KT, Qs=[0.1, 0.7, 10.0],
                                   chain_length=2, chain_Q_multiplier=1.0)
    kl, _ = quick_eval(dyn, _sol.MultiScaleNHCTailVerlet, pot_gmm, dt)
    print(f"  dt={dt}: GMM KL={kl:.4f}")

# ============================================================
# Tune 3: Q values for NHCTail
# ============================================================
print("\n=== Tune Q values for NHCTail on GMM (dt=0.03) ===")
pot_gmm = GaussianMixture2D()
q_configs = [
    [0.1, 0.7, 10.0],
    [0.1, 0.7, 5.0],
    [0.1, 0.7, 15.0],
    [0.1, 1.0, 10.0],
    [0.1, 0.6, 10.0],
    [0.05, 0.7, 10.0],
    [0.2, 0.7, 10.0],
    [0.1, 0.7, 5.0, 20.0],  # 4 scales
]
for Qs in q_configs:
    dyn = _sol.MultiScaleNHCTail(dim=2, kT=KT, Qs=Qs,
                                   chain_length=2, chain_Q_multiplier=1.0)
    kl, _ = quick_eval(dyn, _sol.MultiScaleNHCTailVerlet, pot_gmm, dt=0.03)
    print(f"  Qs={Qs}: GMM KL={kl:.4f}")

# ============================================================
# Tune 4: Chain Q multiplier for NHCTail
# ============================================================
print("\n=== Tune chain_Q_multiplier for NHCTail on GMM ===")
for mult in [0.5, 1.0, 2.0, 5.0]:
    dyn = _sol.MultiScaleNHCTail(dim=2, kT=KT, Qs=[0.1, 0.7, 10.0],
                                   chain_length=2, chain_Q_multiplier=mult)
    kl, _ = quick_eval(dyn, _sol.MultiScaleNHCTailVerlet, pot_gmm, dt=0.03)
    print(f"  mult={mult}: GMM KL={kl:.4f}")

# ============================================================
# Tune 5: Chain length for NHCTail
# ============================================================
print("\n=== Tune chain_length for NHCTail on GMM ===")
for M in [2, 3, 4]:
    dyn = _sol.MultiScaleNHCTail(dim=2, kT=KT, Qs=[0.1, 0.7, 10.0],
                                   chain_length=M, chain_Q_multiplier=1.0)
    kl, _ = quick_eval(dyn, _sol.MultiScaleNHCTailVerlet, pot_gmm, dt=0.03)
    print(f"  M={M}: GMM KL={kl:.4f}")

# ============================================================
# Tune 6: Hierarchical Q values
# ============================================================
print("\n=== Tune Hierarchical LOCR Q values on GMM ===")
hier_configs = [
    [0.7, 1.0, 10.0],
    [0.7, 2.0, 10.0],
    [0.7, 1.0, 5.0],
    [0.7, 0.7, 10.0],
    [1.0, 1.0, 10.0],
    [0.6, 1.0, 10.0],
]
for Qs in hier_configs:
    dyn = _sol.HierarchicalLOCR(dim=2, kT=KT, Qs=Qs)
    kl, _ = quick_eval(dyn, _sol.HierarchicalLOCRVerlet, pot_gmm, dt=0.03)
    print(f"  Qs={Qs}: GMM KL={kl:.4f}")

# ============================================================
# Tune 7: Best config on HO (ergodicity check)
# ============================================================
print("\n=== Best configs on HO ergodicity ===")
pot_ho = HarmonicOscillator1D()
best_configs = [
    ("nhctail [0.1,0.7,10] M=2", lambda: _sol.MultiScaleNHCTail(dim=1, Qs=[0.1, 0.7, 10.0], chain_length=2), _sol.MultiScaleNHCTailVerlet),
    ("nhctail [0.1,0.7,10] M=3", lambda: _sol.MultiScaleNHCTail(dim=1, Qs=[0.1, 0.7, 10.0], chain_length=3), _sol.MultiScaleNHCTailVerlet),
    ("hier [0.7,1,10]", lambda: _sol.HierarchicalLOCR(dim=1, Qs=[0.7, 1.0, 10.0]), _sol.HierarchicalLOCRVerlet),
    ("hier [0.7,2,10]", lambda: _sol.HierarchicalLOCR(dim=1, Qs=[0.7, 2.0, 10.0]), _sol.HierarchicalLOCRVerlet),
]
for name, make_dyn, intcls in best_configs:
    dyn = make_dyn()
    kl, erg = quick_eval(dyn, intcls, pot_ho, dt=0.005)
    print(f"  {name}: HO KL={kl:.4f}, ergo={erg:.3f}" if erg else f"  {name}: HO KL={kl:.4f}")
