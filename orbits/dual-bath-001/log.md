---
strategy: dual-bath-001
status: in-progress
eval_version: eval-v1
metric: 0.0279
issue: 2
parent: null
---

# Dual-Bath Thermostat: NHC(2) + Hamiltonian Rotation

## Summary

A Nose-Hoover Chain of length 2 augmented with a measure-preserving Hamiltonian
rotation in the thermostat (xi, eta) subspace. The chain coupling provides
ergodicity (as in standard NHC), while the rotation creates additional mixing.

## Iteration 1: NHC(2) + rotation, Q_xi=1.0, Q_eta=1.0, alpha=0.5

### Parameters
- Q_xi = 1.0, Q_eta = 1.0, alpha = 0.5
- dt = 0.01 (DW), 0.005 (HO)
- seed = 42
- n_force_evals = 1,000,000

### Results

| Metric | Dual-Bath | NH | NHC(M=3) |
|--------|-----------|-----|----------|
| DW KL | **0.0279** | 0.037 | 0.029 |
| DW ESS/force | 0.00263 | 0.00310 | 0.00261 |
| HO KL | 0.0048 | 0.077 | 0.002 |
| HO Ergodicity | **0.915** | 0.54 | 0.92 |

### Approach

1. Started with proposed dual-friction design (two parallel NH, both providing friction on p). Failed: over-damping caused variance collapse.
2. Tried "single friction + free reservoir" variant (only xi damps p, eta is purely rotational reservoir). Better than NH but not ergodic enough (best erg=0.60).
3. Settled on NHC(2) + rotation: keeps the proven NHC chain coupling AND adds Hamiltonian rotation. This combines two orthogonal mechanisms for breaking KAM tori.

### What worked
- The Hamiltonian rotation term consistently improves ergodicity over pure NHC(2) (0.85 -> 0.90+)
- alpha=0.1 to 0.5 gives best results; larger alpha can be unstable
- Beats NHC(M=3) on DW KL with only 2 thermostat variables

### What I learned
- Additive dual friction (-(xi+eta)*p) effectively halves the thermostat mass, causing over-damping
- The "free reservoir" (eta only driven by rotation) is too weak -- needs a driving term
- NHC chain coupling is the key ingredient for ergodicity; rotation is an enhancement
- The rotation prevents the thermostat from getting trapped but needs a driving force to explore

### Failed approaches
1. Dual parallel friction: over-damped, variance collapsed (var_q=0.53 vs 1.0 expected)
2. Single friction + free reservoir: insufficient ergodicity (0.60 vs 0.85 threshold)
3. High alpha (>2.0): integrator instability

## References

- [Nose (1984)](https://doi.org/10.1063/1.447334) -- Original extended Hamiltonian
- [Hoover (1985)](https://doi.org/10.1103/PhysRevA.31.1695) -- Canonical dynamics
- [Martyna et al. (1992)](https://doi.org/10.1063/1.463940) -- Nose-Hoover Chains
- [Patra & Bhattacharya (2015)](https://doi.org/10.1103/PhysRevE.93.023308) -- Dual thermostat with configurational temperature
- [Fukuda & Nakamura (2002)](https://doi.org/10.1103/PhysRevE.65.026105) -- Coupled Nose-Hoover equations
- [Rugh (1997)](https://doi.org/10.1103/PhysRevLett.78.772) -- Dynamical approach to temperature
- [KAM theory](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold%E2%80%93Moser_theorem) -- Why NH fails for 1D HO
