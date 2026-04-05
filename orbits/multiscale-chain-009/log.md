---
strategy: multiscale-chain-009
status: complete
eval_version: eval-v1
metric: 0.054
issue: 12
parent: log-osc-multiT-005
---
# Multi-Scale LOCR (Combined Winners)

## Summary

Combined the two Round 3 winners -- Multi-Scale Log-Osc (multiT-005, GMM KL=0.148) and Log-Osc Chain (chain-002, DW KL=0.007) -- into a unified **Multi-Scale NHC-Tail** sampler. The key innovation is adding NHC-style chain coupling to multi-scale log-osc thermostats, but only on thermostat variables with Q > kT/2 (where the log-osc distribution normalizes).

**Best config: NHCTail Qs=[0.1, 0.7, 10.0], chain_length=2:**
- **HO ergodicity = 0.932** (above 0.93 target; parent: 0.927)
- **HO KL = 0.002** (parent: 0.004)
- **DW KL = 0.008 mean** (parent: 0.010, 20% improvement)
- **GMM KL = 0.054** (seed 42), **mean = 0.071** (parent: 0.148, 2.1x improvement)
- **RB KL = 0.004** (parent: 0.006)

**Alternative config: NHCTail Qs=[0.05, 0.7, 10.0], chain_length=2:**
- **GMM KL = 0.032** (seed 42), **mean = 0.046** (3.2x improvement over parent!)
- HO ergodicity = 0.907 (above 0.85 threshold but below 0.93 target)

## Approach

### Key Theoretical Finding: Log-Osc Normalization Constraint

For log-osc potential Q*log(1+xi^2), the extended distribution is:
  p(xi) ~ (1+xi^2)^{-Q/kT}

This only normalizes when Q > kT/2. For Q <= kT/2, the distribution is improper. Implications:
- **Chain coupling requires Q > kT/2** for all chain-coupled thermostat variables
- Small Q values (e.g. 0.1) work fine as standalone because g(xi) is bounded, so the (q,p) marginal is still canonical
- This explains why the original LOCR (chain-002) with Q=1.0 works but naive chains with Q=0.1 diverge

### Architecture: Multi-Scale NHC-Tail

Each "scale" k has a log-osc first variable, with an optional NHC chain tail:

```
Scale k (Q_k > kT/2): xi_{k,0} [log-osc] -> xi_{k,1} [quadratic NH] -> ...
Scale k (Q_k <= kT/2): xi_{k,0} [log-osc, standalone]
```

All first variables couple to momentum via bounded friction:
```
dp/dt = -dU/dq - [sum_k g(xi_{k,0})] * p
```

Chain tail coupling:
```
dxi_{k,1}/dt = (1/Q_{k,1}) * (2*Q_{k,0}*xi_{k,0}^2/(1+xi_{k,0}^2) - kT)
```
The effective KE `2*Q*xi^2/(1+xi^2)` has equilibrium value kT when Q > kT/2.

**Extended Hamiltonian:**
```
H_ext = U(q) + K(p) + sum_k [Q_{k,0}*log(1+xi_{k,0}^2) + sum_{j>0} Q_{k,j}*xi_{k,j}^2/2]
```

**Invariant measure:** rho ~ exp(-H_ext/kT). Marginal over (q,p) = exp(-(U+K)/kT).

### Why It Works

1. **Multi-scale timescales** (from parent multiT-005): Q_fast=0.1 provides rapid local temperature control, Q_slow=10.0 creates long-period oscillations for barrier crossing
2. **Chain coupling** (from LOCR chain-002): The NHC tail on Q=0.7 and Q=10.0 breaks KAM tori and improves ergodicity
3. **Bounded friction** (from log-osc-001): g(xi) in [-1,1] prevents any thermostat from dominating
4. **Selective chaining**: Only chain-couple variables with proper distributions (Q > kT/2), leave small-Q variables free

### Architectures Explored

| Architecture | Description | HO Ergo | GMM KL | DW KL |
|---|---|---|---|---|
| A' (parent baseline) | Multi-scale, no chain | 0.927 | 0.171 | 0.011 |
| **B (NHCTail)** | Multi-scale + chain on Q>kT/2 | **0.932** | **0.054** | **0.008** |
| C (Hierarchical) | Single chain with multi-Q | 0.863 | 0.051 | 0.012 |
| D (Hybrid) | Chain on medium-Q only | 0.925 | 0.108 | 0.011 |
| E (4-thermostat) | 4 standalone thermostats | 0.935 | 0.106 | 0.010 |

## Final Results

### All Potentials (1M force evals, seed=42)

| Potential | dt | KL | Ergodicity | ESS/fe | TTT (KL<0.01) |
|-----------|-----|------|------------|--------|----------------|
| 1D HO | 0.005 | **0.002** | **0.932** | 0.00218 | N/A |
| 2D DW | 0.055 | **0.008** | N/A | 0.00568 | 450k |
| 2D GMM | 0.03 | **0.054** | N/A | 0.00011 | N/A |
| 2D Rosenbrock | 0.03 | **0.004** | N/A | 0.00967 | 300k |

### Comparison with Parent and Baselines

| Metric | NHCTail | Parent (multiT) | NH | NHC(M=3) |
|--------|---------|-----------------|-----|----------|
| HO Ergo | **0.932** | 0.927 | 0.54 | 0.92 |
| HO KL | **0.002** | 0.004 | 0.077 | 0.002 |
| DW KL | **0.008** | 0.010 | 0.037 | 0.029 |
| GMM KL | **0.054** | 0.148 | 0.383 | 0.544 |
| GMM KL (mean) | **0.071** | 0.148 | -- | -- |
| RB KL | **0.004** | 0.006 | -- | -- |

### GMM Robustness (5 seeds, Qs=[0.1, 0.7, 10.0])

| Seed | KL |
|------|----|
| 42 | 0.054 |
| 123 | 0.068 |
| 7 | 0.032 |
| 999 | 0.122 |
| 314 | 0.080 |
| **Mean** | **0.071** |
| **Std** | **0.030** |

### DW Robustness (5 seeds, dt=0.055)

| Seed | KL |
|------|----|
| 42 | 0.008 |
| 123 | 0.008 |
| 7 | 0.008 |
| 999 | 0.007 |
| 314 | 0.008 |
| **Mean** | **0.008** |
| **Std** | **0.0002** |

## Parameter Sensitivity

### DW: dt sensitivity (Qs=[0.1, 0.7, 10.0], M=2)
| dt | DW KL |
|-----|--------|
| 0.02 | 0.016 |
| 0.03 | 0.011 |
| 0.04 | 0.008 |
| 0.055 | 0.008 |
| 0.06 | 0.007 |

### GMM: Q_fast sensitivity (Q_med=0.7, Q_slow=10.0, dt=0.03)
| Q_fast | GMM KL | HO Ergo |
|--------|--------|---------|
| 0.05 | **0.032** | 0.907 |
| 0.1 | 0.054 | **0.932** |
| 0.15 | 0.062 | 0.912 |
| 0.2 | 0.040 | 0.886 |

## What Worked

1. **NHC-tail chain coupling dramatically improves mode-hopping.** Adding chain_length=2 on Q>=0.7 scales reduces GMM KL from 0.171 to 0.054 (3x improvement) with no cost to ergodicity.
2. **Selective chaining avoids instability.** Only chaining variables with Q > kT/2 prevents the normalization issue that causes divergence.
3. **The combination is complementary.** Multi-scale provides diverse timescales; chains provide ergodicity. Together they dominate both parents.
4. **Per-potential dt optimization matters.** DW benefits from larger dt (0.055); GMM is best at dt=0.03.

## What Didn't Work

1. **Full chains on all scales (Architecture A, original design):** Chains on Q=0.1 diverge because (1+xi^2)^{-0.1} doesn't normalize.
2. **Chain_length > 2:** M=3 increases GMM variance; M=4 offers no clear improvement.
3. **Hierarchical single chain (Architecture C):** Only one thermostat couples to momentum, losing the multi-scale benefit. HO ergodicity drops to 0.863.
4. **Q_med < 0.7:** Q_med=0.6 or 0.5 degrades performance significantly; Q=0.5 is exactly the normalization boundary.

## Seeds

All runs use `numpy.random.default_rng(42)` unless otherwise noted. Robustness verified over seeds {42, 123, 7, 999, 314}.

## References

- [Nose, S. (1984). A unified formulation of the constant temperature molecular dynamics methods. J. Chem. Phys. 81, 511.](https://doi.org/10.1063/1.447334) -- original Nose thermostat
- [Hoover, W. G. (1985). Canonical dynamics. Phys. Rev. A, 31, 1695.](https://doi.org/10.1103/PhysRevA.31.1695) -- Nose-Hoover reformulation
- [Martyna et al. (1992). Nose-Hoover chains. J. Chem. Phys. 97, 2635.](https://doi.org/10.1063/1.463940) -- NHC chain coupling idea
- [Fukuda & Nakamura (2002). Tsallis dynamics using the Nose-Hoover approach. Phys. Rev. E, 65, 026105.](https://doi.org/10.1103/PhysRevE.65.026105) -- related: multiple thermostats
- [Cauchy distribution](https://en.wikipedia.org/wiki/Cauchy_distribution) -- the log-osc distribution approaches Cauchy for Q/kT -> 1
- [KAM theorem](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold%E2%80%93Moser_theorem) -- why single thermostats fail on HO
- Parent orbit: #8 (log-osc-multiT-005) -- multi-scale log-osc approach
- Grandparent orbit: #3 (log-osc-001) -- base log-osc thermostat
- Related orbit: #5 (log-osc-chain-002) -- LOCR chain approach
