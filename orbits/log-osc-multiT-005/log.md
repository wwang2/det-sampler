---
strategy: log-osc-multiT-005
status: in-progress
eval_version: eval-v1
metric: 0.1796
issue: 8
parent: log-osc-001
---

# Log-Osc Multi-Thermostat for Multi-Modal Hopping

## Summary

Extended the Log-Osc thermostat with multiple thermostat variables at different timescales. The **MultiScale** variant (3 log-osc thermostats: fast Q=0.1, medium Q=1.0, slow Q=10.0) achieves **KL=0.1796 on the 5-mode GMM**, a major improvement over the parent's 0.38 and below the target of 0.20.

## Approach

**Key insight:** Multiple independent log-osc thermostats with different Q values create friction at multiple timescales. The bounded friction g(xi) = 2*xi/(1+xi^2) ensures each thermostat contributes bounded coupling in [-1, 1], and the total friction is bounded in [-N, N] for N thermostats.

**Extended Hamiltonian (MultiScale, N=3):**
```
H_ext = U(q) + K(p) + Q_f*log(1+xi_f^2) + Q_m*log(1+xi_m^2) + Q_s*log(1+xi_s^2)
```

**Equations of motion:**
```
dq/dt = p/m
dp/dt = -dU/dq - [g(xi_f) + g(xi_m) + g(xi_s)] * p
dxi_k/dt = (1/Q_k) * (K - dim*kT)    for k in {f, m, s}
```

**Invariant measure:** rho ~ exp(-H_ext/kT). Marginal over (q,p) = exp(-(U+K)/kT) -- correct canonical distribution, since each thermostat variable integrates out independently.

**Why it helps mode-hopping:** The slow thermostat (Q_s=10) creates long-period oscillations in friction that periodically build up or drain kinetic energy. These slow cycles occasionally give the system enough kinetic energy to cross barriers between modes. The fast thermostat (Q_f=0.1) provides rapid local temperature control, while the medium thermostat (Q_m=1.0) handles intermediate fluctuations.

## Variants Tested

| Variant | Description | GMM KL | HO Ergo |
|---------|-------------|--------|---------|
| Parent (log-osc-001) | Single log-osc, Q=0.5 | 0.903 | 0.944 |
| DualLogOsc | 2 independent log-osc | 0.250-0.388 | 0.397-0.884 |
| DualLogOscCross | 2 with cross-coupling | inf (unstable) | inf |
| TempPulse | Q modulated by harmonic osc | 0.831-2.840 | 0.887 |
| **MultiScale** | **3 independent log-osc** | **0.180** | **0.727** |

## Iteration 1 Results

### GMM (5-mode, the target):
| Configuration | KL |
|---------------|-----|
| Parent (Q=0.5) | 0.903 |
| DualLogOsc(0.05, 1.0) | 0.250 |
| DualLogOsc(0.5, 5.0) | 0.308 |
| **MultiScale(0.1, 1.0, 10.0)** | **0.180** |
| MultiScale(0.05, 0.5, 5.0) | 0.876 |

### Stage 1: HO Ergodicity
| Configuration | KL | Ergodicity |
|---------------|-----|------------|
| Parent (Q=0.8) | 0.023 | 0.944 |
| DualLogOsc(0.1, 2.0) | 0.061 | 0.884 |
| MultiScale(0.1, 1.0, 10.0) | 0.020 | 0.727 |
| TempPulse(Q=0.8, wz=0.1, A=0.5) | 0.010 | 0.887 |

### Stage 1: Double Well
| Configuration | KL | ESS/fe |
|---------------|-----|--------|
| Parent (Q=1, dt=0.035) | 0.010 | 0.00219 |
| DualLogOsc(0.3, 3.0, dt=0.03) | 0.013 | 0.00570 |
| MultiScale(0.1, 1, 10, dt=0.02) | 0.016 | 0.00340 |

## What I Learned

1. **Multiple timescale thermostats dramatically improve multi-modal hopping.** The MultiScale approach reduces GMM KL from 0.38 to 0.18, well below the 0.20 target.
2. **Cross-coupling (variant B) is unstable** -- the feedback between thermostat variables creates divergent dynamics.
3. **TempPulse doesn't help GMM** -- modulating Q only changes the thermostat response speed, not the energy scale explored.
4. **Tradeoff: GMM vs HO ergodicity** -- multi-thermostats hurt HO ergodicity (0.73 vs 0.94). This may be a fundamental limitation of having multiple thermostats on a 1D system.

## Seeds

All runs use `numpy.random.default_rng(42)` via the evaluator default.

## References

- [Nose, S. (1984). A unified formulation of the constant temperature molecular dynamics methods. J. Chem. Phys. 81, 511.](https://doi.org/10.1063/1.447334)
- [Hoover, W. G. (1985). Canonical dynamics. Phys. Rev. A, 31, 1695.](https://doi.org/10.1103/PhysRevA.31.1695)
- [Martyna et al. (1992). Nose-Hoover chains. J. Chem. Phys. 97, 2635.](https://doi.org/10.1063/1.463940)
- [Fukuda & Nakamura (2002). Tsallis dynamics using the Nose-Hoover approach. Phys. Rev. E, 65, 026105.](https://doi.org/10.1103/PhysRevE.65.026105) -- multiple thermostats idea
- [Marinari & Parisi (1992). Simulated tempering. Europhys. Lett. 19, 451.](https://doi.org/10.1209/0295-5075/19/6/002) -- stochastic tempering inspiration
- [Simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing) -- temperature modulation for barrier crossing
- Parent orbit: #3 (log-osc-001) -- base log-osc thermostat with KL=0.010 on DW
