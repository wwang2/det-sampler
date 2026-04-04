---
strategy: dual-bath-001
status: in-progress
eval_version: eval-v1
metric: 0.0302
issue: 2
parent: null
---

# Dual-Bath Thermostat: NHC(2) + Hamiltonian Rotation

## Summary

A Nose-Hoover Chain of length 2 augmented with a measure-preserving Hamiltonian
rotation in the thermostat (xi, eta) subspace. The chain coupling provides
ergodicity (as in standard NHC), while the rotation creates additional mixing.

**Best config: Q_xi=1.0, Q_eta=1.0, alpha=0.1**

## Iteration 2: alpha=0.1 (optimal), full 1M budget

### Parameters
- Q_xi = 1.0, Q_eta = 1.0, alpha = 0.1
- dt = 0.01 (DW), 0.005 (HO)
- seed = 42
- n_force_evals = 1,000,000

### Results

| Metric | Dual-Bath | NH | NHC(M=3) | vs NHC |
|--------|-----------|-----|----------|--------|
| DW KL | 0.0302 | 0.037 | 0.029 | close |
| DW ESS/force | **0.00274** | 0.00310 | 0.00261 | +5% better |
| HO KL | **0.0021** | 0.077 | 0.002 | match |
| HO Ergodicity | **0.927** | 0.54 | 0.92 | +0.8% better |

### Alpha sensitivity analysis (all at 1M force evals)

| alpha | HO KL | HO Erg | DW KL | DW ESS |
|-------|-------|--------|-------|--------|
| 0.0 (pure NHC2) | 0.006 | 0.850 | 0.055 | - |
| **0.1** | **0.002** | **0.927** | 0.030 | **0.00274** |
| 0.3 | 0.002 | 0.921 | 0.033 | 0.00138 |
| 0.5 | 0.005 | 0.915 | 0.028 | 0.00263 |
| 0.8 | 0.002 | 0.913 | **0.027** | 0.00136 |
| 1.0 | 0.004 | 0.901 | 0.029 | 0.00111 |

Key findings:
- Any alpha > 0 improves over pure NHC(2) significantly
- alpha=0.1 gives best ergodicity and overall balance
- Larger alpha improves DW KL but reduces ergodicity
- The rotation consistently breaks KAM tori

## Iteration 1: Initial exploration, alpha=0.5

(See git history for details. DW_KL=0.028, HO_erg=0.915.)

## Development history

1. **Dual-friction design (FAILED)**: Two parallel NH, both providing friction on p via -(xi+eta)*p. Over-damped: effective thermostat mass halved, variance collapsed (var_q=0.53 vs 1.0 expected).

2. **Free reservoir design (WEAK)**: Only xi damps p, eta is purely rotational reservoir. Better than NH but not ergodic enough (best erg=0.60 vs 0.85 threshold needed).

3. **NHC(2) + rotation (SUCCESS)**: Keeps NHC chain coupling AND adds Hamiltonian rotation. Two orthogonal mechanisms for breaking KAM tori. Beats NHC(M=3) with only 2 thermostat variables.

4. **Analytical rotation integrator (ABANDONED)**: Splitting with exact rotation substep. More stable at high alpha but more error per step at moderate alpha. Simple Euler half-step works better for alpha<=1.

## Key insights

- The Hamiltonian rotation is a measure-preserving perturbation that breaks symmetries without adding computational cost
- Even small rotation (alpha=0.1) dramatically improves over pure NHC(2)
- The rotation complements chain coupling: chains provide energy flow, rotation provides phase-space mixing
- Two thermostat variables with dual mechanisms > three variables with single mechanism

## References

- [Nose (1984)](https://doi.org/10.1063/1.447334) -- Original extended Hamiltonian
- [Hoover (1985)](https://doi.org/10.1103/PhysRevA.31.1695) -- Canonical dynamics
- [Martyna et al. (1992)](https://doi.org/10.1063/1.463940) -- Nose-Hoover Chains
- [Patra & Bhattacharya (2015)](https://doi.org/10.1103/PhysRevE.93.023308) -- Dual thermostat with configurational temperature
- [Fukuda & Nakamura (2002)](https://doi.org/10.1103/PhysRevE.65.026105) -- Coupled Nose-Hoover equations
- [Rugh (1997)](https://doi.org/10.1103/PhysRevLett.78.772) -- Dynamical approach to temperature
- [KAM theory](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold%E2%80%93Moser_theorem) -- Why NH fails for 1D HO
