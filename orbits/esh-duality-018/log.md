---
strategy: esh-duality-018
status: complete
eval_version: eval-v1
metric: 1.0809
issue: 20
parent: spectral-1f-016
---

# ESH Dynamics as Special Case of Generalized Thermostat Framework

## Summary

ESH (Versteeg 2021, arXiv:2111.02434) is **NOT** a special case of our Master Theorem.
They are structurally distinct frameworks with different conservation laws, stationary
measures, and ergodicity mechanisms.

A new thermostat potential V(ξ) = Q·(exp(2ξ)/2 − ξ) — the "ESH-inspired thermostat" —
is identified as a valid new member of the Master Theorem family.

---

## Mathematical Duality Result

### Negative Result: ESH ≠ Our Thermostat

Three independent arguments show ESH cannot be embedded in our Master Theorem:

1. **p-equation structure:** ESH uses `dp/dt = -∇U·|p|` (force scaled by speed).
   Our framework uses `dp/dt = -∇U - g(ξ)·p` (force + friction). These are
   incompatible for generic U.

2. **ξ-equation:** Under ξ = log(|p|/√(kT)), ESH gives `dξ/dt = -∇U·sign(p)`
   (force-driven). Our framework gives `dξ/dt = (exp(2ξ)−1)/Q` (kinetic-energy-driven).
   Completely different.

3. **Conserved quantity:** ESH conserves `H_ESH = U(x) + (d/2)·log(‖p‖²/d)` exactly —
   it is a **conservative Hamiltonian system**. Our thermostats are **dissipative**.

### Stationary Measures

| System      | p-distribution        | Normalizable in 1D? |
|-------------|----------------------|---------------------|
| Our thermo  | Gaussian N(0, kT)    | Yes                 |
| ESH         | Power-law |p|^{-1}   | NO — improper!      |

ESH's "stationary measure" is improper in 1D (logarithmically divergent at |p|→0 and |p|→∞).

### New Contribution: ESH-Inspired Thermostat

Motivated by ESH's fixed point (|p|=√(kT)), we identify a new potential in our framework:

```
V_ESH(ξ) = Q·(exp(2ξ)/2 − ξ)
g_ESH(ξ) = exp(2ξ) − 1
```

Properties:
- g(0) = 0: equilibrium at |p|=√(kT) ✓
- g(ξ) > 0 for ξ > 0: damps fast particles ✓
- g(ξ) < 0 for ξ < 0: heats slow particles ✓
- P(ξ) ∝ exp(−V/kT) is normalizable ✓
- g is unbounded (unlike Log-Osc/Tanh/Arctan)

This is a new, previously uncatalogued member of the Master Theorem family.

---

## Numerical Results

### 1D Harmonic Oscillator Comparison

| Method      | var_q (target=1.0) | var_p (target=1.0) | KL_q    |
|-------------|--------------------|--------------------|---------|
| ESH (1D)    | 3.81               | ~0.000             | 493.67  |
| Log-Osc     | 0.710              | 1.000              | 0.1186  |
| ESH-Thermo  | 0.946              | 1.001              | 0.0310  |
| NHC         | 1.048              | 1.029              | 0.0161  |

**Key finding:** ESH 1D on the harmonic oscillator is catastrophically non-ergodic
(drifts to q≈31, KL=493). This confirms ESH is a conservative system that does NOT
sample the canonical distribution — it is a parameterized Hamiltonian flow, not a sampler.

ESH-Thermo (our new V(ξ)) achieves KL=0.031, competitive with NHC (0.016).

### GMM 2D KL Divergence (500k force evaluations)

| Method      | KL Divergence | ESS/eval   |
|-------------|---------------|------------|
| NHC         | 0.4996        | 0.000104   |
| ESH-Thermo  | 1.0809        | 0.000118   |
| Log-Osc     | 1.7911        | 0.000116   |

The metric in the front matter is the ESH-Thermo GMM KL = **1.0809**.

ESH-Thermo beats Log-Osc on GMM but not NHC. The exponentially-growing friction
may be too aggressive for multi-modal landscapes.

---

## ESH + Multi-Scale Assessment

ESH could in principle be extended multi-scale, but it is not straightforward:
- ESH is conservative, so "multi-scale" would require stochastic velocity refreshments
  (making it HMC-like, not deterministic)
- Alternatively, multiple ESH chains at different |p| levels could be run and
  combined — but this lacks the principled thermodynamic grounding of our chain approach
- NHCTail (from spectral-1f-016 parent orbit) is likely superior since it achieves
  multi-scale exploration within the dissipative thermostat framework

---

## Files

- `theory.md` — full mathematical derivation (6 sections)
- `make_esh_theory.py` — SymPy analysis with symbolic verification
- `make_esh_comparison.py` — numerical comparison: ESH, Log-Osc, ESH-Thermo, NHC
- `make_esh_figures.py` — consolidated 4-panel figure
- `figures/esh_duality_consolidated.png` — main result figure
- `comparison_results.json` — numerical results (KL, ESS, friction functions)
- `trajectories.npz` — phase-space trajectory data

---

## Conclusion

ESH is a **parallel, independent framework** — a reparameterized Hamiltonian dynamics
with logarithmic kinetic energy. It is NOT dual to, nor a special case of, the generalized
thermostat Master Theorem. The conceptual similarity (both use log-like momentum) is
superficial; the structural difference (conservative vs. dissipative) is fundamental.

The ESH-inspired thermostat with V(ξ) = Q·(exp(2ξ)/2 − ξ) is a novel, valid member
of the Master Theorem family, though it underperforms NHC on multi-modal targets.
