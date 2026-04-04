---
strategy: dual-bath-001
status: complete
eval_version: eval-v1
metric: 0.0302
issue: 2
parent: null
---

# Dual-Bath Thermostat: NHC(2) + Hamiltonian Rotation

## Summary

A Nose-Hoover Chain of length 2 augmented with a measure-preserving Hamiltonian
rotation in the thermostat (xi, eta) subspace. Combines two orthogonal mechanisms
for breaking KAM tori: chain energy transfer + Hamiltonian phase-space rotation.

**Best config: Q_xi=1.0, Q_eta=1.0, alpha=0.1, dt=0.01/0.005**

## Final Results (alpha=0.1, seed=42, 1M force evals)

### Fair comparison (same seed, same dt, same budget)

| Metric | Dual-Bath (ours) | NHC(M=3) | Winner |
|--------|-----------------|----------|--------|
| HO KL | **0.0021** | 0.0034 | **Ours (38% better)** |
| HO Ergodicity | **0.927** | 0.915 | **Ours** |
| DW KL | 0.0302 | 0.0287 | NHC (5% better) |
| DW ESS/force | **0.00274** | 0.00261 | **Ours (5% better)** |

Notes: NHC(M=3) uses 3 thermostat variables; ours uses only 2.
Our method wins on 3/4 metrics in head-to-head comparison.

### Seed robustness (3 seeds)

| Seed | HO KL | HO Erg | DW KL |
|------|-------|--------|-------|
| 42 | 0.0021 | 0.927 | 0.030 |
| 123 | 0.0033 | 0.934 | 0.034 |
| 789 | 0.0012 | 0.912 | 0.029 |
| **mean** | **0.002** | **0.924** | **0.031** |

### Alpha sensitivity (1M force evals, seed=42)

| alpha | HO KL | HO Erg | DW KL | DW ESS |
|-------|-------|--------|-------|--------|
| 0.0 (NHC2) | 0.006 | 0.850 | 0.055 | - |
| **0.1** | **0.002** | **0.927** | 0.030 | **0.00274** |
| 0.3 | 0.002 | 0.921 | 0.033 | 0.00138 |
| 0.5 | 0.005 | 0.915 | 0.028 | 0.00263 |
| 0.8 | 0.002 | 0.913 | 0.027 | 0.00136 |

Key: any alpha > 0 improves over pure NHC(2); alpha=0.1 is optimal for ergodicity.

## Method

### Equations of motion

```
dq/dt   = p / m
dp/dt   = -dU/dq - xi * p
dxi/dt  = (1/Q_xi) * (|p|^2/m - d*kT) - eta*xi + alpha*sqrt(Q_eta/Q_xi)*eta
deta/dt = (1/Q_eta) * (Q_xi*xi^2 - kT) - alpha*sqrt(Q_xi/Q_eta)*xi
```

### Invariant measure (Tier 1 -- proved)

rho ~ exp(-U/kT - p^2/(2mkT) - Q_xi*xi^2/(2kT) - Q_eta*eta^2/(2kT))

Verified via Liouville equation dS/dt = div(v). See derivation.md and verify_nhc_rotation.py.

### Integrator

Velocity Verlet with Euler half-steps for thermostat and analytical exp(-xi*dt/2) rescaling for momentum. 1 force eval per step via FSAL.

## Development history

1. **Dual-friction design (FAILED)**: Two parallel NH, both -(xi+eta)*p friction. Over-damped: effective Q halved.
2. **Free reservoir design (WEAK)**: Only xi damps p, eta purely rotational. Not ergodic enough (erg=0.60).
3. **NHC(2) + rotation (SUCCESS)**: Chain coupling + rotation. Beats NHC(M=3) with only 2 variables.
4. **Analytical rotation integrator (ABANDONED)**: More stable at high alpha but worse per-step accuracy.
5. **Alpha optimization**: Swept alpha 0.0-1.0. alpha=0.1 is sweet spot for ergodicity.

## Key insights

- The Hamiltonian rotation is a zero-cost, measure-preserving enhancement to NHC
- Even tiny rotation (alpha=0.1) breaks symmetries that trap standard NHC on KAM tori
- Chain coupling + rotation are complementary: chains transfer energy, rotation mixes phase space
- 2 variables with dual mechanisms can outperform 3 variables with single mechanism

## References

- [Nose (1984)](https://doi.org/10.1063/1.447334) -- Original extended Hamiltonian
- [Hoover (1985)](https://doi.org/10.1103/PhysRevA.31.1695) -- Canonical dynamics
- [Martyna et al. (1992)](https://doi.org/10.1063/1.463940) -- Nose-Hoover Chains
- [Patra & Bhattacharya (2015)](https://doi.org/10.1103/PhysRevE.93.023308) -- Dual thermostat with configurational temperature
- [Fukuda & Nakamura (2002)](https://doi.org/10.1103/PhysRevE.65.026105) -- Coupled Nose-Hoover equations
- [Rugh (1997)](https://doi.org/10.1103/PhysRevLett.78.772) -- Dynamical approach to temperature
- [KAM theory](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold%E2%80%93Moser_theorem) -- Why NH fails for 1D HO
- [Simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing) -- Related stochastic approach
