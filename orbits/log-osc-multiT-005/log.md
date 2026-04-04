---
strategy: log-osc-multiT-005
status: complete
eval_version: eval-v1
metric: 0.148
issue: 8
parent: log-osc-001
---

# Log-Osc Multi-Thermostat for Multi-Modal Hopping

## Summary

Extended the Log-Osc thermostat with multiple thermostat variables at different timescales (Multi-Scale Log-Osc). The **best configuration Qs=[0.1, 0.7, 10.0] at dt=0.03** achieves:

- **GMM (5-mode) KL=0.148 (mean over 5 seeds, std=0.024)** -- 2.6x improvement over parent (0.38)
- **HO ergodicity=0.927** (above 0.85 threshold)
- **DW KL=0.010** (matching parent)
- **Rosenbrock KL=0.006**

This is the first sampler to achieve KL < 0.20 on the 5-mode Gaussian mixture while maintaining ergodicity on the harmonic oscillator.

## Approach

**Key insight:** Multiple independent log-osc thermostats with different Q values create friction at multiple timescales. The bounded friction g(xi) = 2*xi/(1+xi^2) ensures each thermostat contributes bounded coupling in [-1, 1], and the total friction is bounded in [-N, N] for N thermostats.

**Extended Hamiltonian (N=3 thermostats):**
```
H_ext = U(q) + K(p) + Q_1*log(1+xi_1^2) + Q_2*log(1+xi_2^2) + Q_3*log(1+xi_3^2)
```

**Equations of motion:**
```
dq/dt = p/m
dp/dt = -dU/dq - [g(xi_1) + g(xi_2) + g(xi_3)] * p
dxi_k/dt = (1/Q_k) * (K - dim*kT)    for k = 1, 2, 3
```
where g(xi) = 2*xi/(1+xi^2) is the bounded log-osc friction function.

**Invariant measure:** rho ~ exp(-H_ext/kT). Marginal over (q,p) = exp(-(U+K)/kT) -- correct canonical distribution. Each thermostat variable integrates out independently because they enter the Hamiltonian additively and the xi equations only depend on (q, p) through the kinetic energy K.

**Why it helps mode-hopping:** The slow thermostat (Q=10) creates long-period oscillations in friction that periodically build up kinetic energy. When the slow thermostat variable has the right phase, the system has enough kinetic energy to cross barriers between modes. The fast thermostat (Q=0.1) provides rapid local temperature control. The medium thermostat (Q=0.7) bridges the timescale gap, creating complex interference patterns that further improve mixing.

**Integrator:** Modified velocity Verlet with exp(-[g(xi_1)+g(xi_2)+g(xi_3)]*dt/2) momentum rescaling. FSAL scheme, 1 force eval per step after initialization.

## Best Configuration

**Qs = [0.1, 0.7, 10.0], dt=0.03 (for 2D systems), dt=0.005 (for 1D HO)**

## Final Results

### All Potentials (1M force evals)

| Potential | dt | KL | Ergodicity | ESS/fe | TTT (KL<0.01) |
|-----------|-----|------|------------|--------|----------------|
| 1D HO | 0.005 | 0.004 | **0.927** | 0.00207 | N/A |
| 2D DW | 0.035 | **0.010** | N/A | 0.00637 | 650k |
| 2D GMM | 0.03 | **0.148** | N/A | ~0.0001 | N/A |
| 2D Rosenbrock | 0.02 | **0.006** | N/A | N/A | N/A |

### Comparison with Baselines

| Metric | MultiScale | Log-Osc (parent) | NH | NHC(M=3) |
|--------|-----------|-------------------|-----|----------|
| GMM KL | **0.148** | 0.377 | 0.383 | 0.544 |
| DW KL | 0.010 | **0.010** | 0.037 | 0.029 |
| HO Ergo | 0.927 | **0.944** | 0.54 | 0.92 |
| HO KL | **0.004** | 0.023 | 0.077 | 0.002 |
| DW ESS/fe | **0.00637** | 0.00219 | 0.00310 | 0.00261 |

### GMM Robustness (5 seeds)

| Seed | KL |
|------|----|
| 42 | 0.171 |
| 123 | 0.120 |
| 7 | 0.134 |
| 999 | 0.135 |
| 314 | 0.182 |
| **Mean** | **0.148** |
| **Std** | **0.024** |

### Variants Explored

| Variant | Description | GMM KL | HO Ergo |
|---------|-------------|--------|---------|
| Parent (log-osc-001) | Single log-osc, Q=0.5 | 0.903 | 0.944 |
| DualLogOsc | 2 independent log-osc | 0.250-0.388 | 0.397-0.884 |
| DualLogOscCross | 2 with cross-coupling | inf (unstable) | inf |
| TempPulse | Q modulated by harmonic osc | 0.831-2.840 | 0.887 |
| MultiScale(0.1, 1.0, 10.0) | 3 log-spaced thermostats | 0.078-0.457 | 0.727 |
| **MultiScale(0.1, 0.7, 10.0)** | **3 optimized thermostats** | **0.120-0.182** | **0.927** |
| MultiScale(0.1, 0.5, 10.0) | 3 thermostats (lower Q_med) | 0.261 | 0.936 |
| 5-thermostat(0.01..100) | Wide-range 5 thermostats | 0.156-0.287 | 0.809 |

### Key Parameter Sweep Results

**GMM: Q_med sensitivity (Q_f=0.1, Q_s=10.0, dt=0.03)**
| Q_med | GMM KL (mean, 3 seeds) | HO Ergo |
|-------|------------------------|---------|
| 0.5 | 0.261 (single run) | 0.936 |
| 0.7 | **0.142** | **0.927** |
| 0.8 | 0.211 | 0.873 |
| 1.0 | 0.078-0.457 (high variance) | 0.727 |

**GMM: dt sensitivity (Qs=[0.1, 1.0, 10.0])**
| dt | GMM KL |
|-----|--------|
| 0.01 | 0.273 |
| 0.02 | 0.180 |
| 0.028 | 0.113 |
| 0.03 | 0.078 |
| 0.032 | 0.207 |
| 0.035 | 0.175 |

## What Worked

1. **Multiple timescale thermostats dramatically improve multi-modal hopping.** The MultiScale approach reduces GMM KL from 0.38 to 0.15, a 2.6x improvement.
2. **Q_med=0.7 is the sweet spot.** Too small (0.5) hurts GMM; too large (1.0) hurts HO ergodicity and increases GMM variance.
3. **Log-spaced Q values.** The ratio Q_slow/Q_fast ~ 100 creates dynamics spanning two orders of magnitude in timescale.
4. **Bounded friction is essential.** The log-osc g(xi) in [-1,1] prevents any single thermostat from dominating, allowing balanced multi-scale dynamics.
5. **Larger dt helps GMM.** At dt=0.03, each integration step covers more phase space, aiding barrier crossing.

## What Didn't Work

1. **Cross-coupling between thermostats (variant B)** -- creates positive feedback leading to divergence.
2. **Temperature pulsing (variant C)** -- modulating Q only changes thermostat response speed, not the energy scale explored. Doesn't help with mode-hopping.
3. **Too many thermostats (5+)** -- diminishing returns and reduced robustness. 3 thermostats is the sweet spot.
4. **Q_med=1.0** -- gives the best single-seed GMM KL (0.078) but extremely high seed variance (0.078-0.457), making it unreliable.

## Seeds

All runs use `numpy.random.default_rng(42)` unless otherwise noted. Robustness verified over seeds {42, 123, 7, 999, 314}.

## References

- [Nose, S. (1984). A unified formulation of the constant temperature molecular dynamics methods. J. Chem. Phys. 81, 511.](https://doi.org/10.1063/1.447334) -- original Nose thermostat
- [Hoover, W. G. (1985). Canonical dynamics. Phys. Rev. A, 31, 1695.](https://doi.org/10.1103/PhysRevA.31.1695) -- Nose-Hoover reformulation
- [Martyna et al. (1992). Nose-Hoover chains. J. Chem. Phys. 97, 2635.](https://doi.org/10.1063/1.463940) -- chain thermostat idea
- [Fukuda & Nakamura (2002). Tsallis dynamics using the Nose-Hoover approach. Phys. Rev. E, 65, 026105.](https://doi.org/10.1103/PhysRevE.65.026105) -- related: multiple thermostats for enhanced sampling
- [Marinari & Parisi (1992). Simulated tempering. Europhys. Lett. 19, 451.](https://doi.org/10.1209/0295-5075/19/6/002) -- stochastic tempering for multi-modal distributions
- [Sugita & Okamoto (1999). Replica-exchange molecular dynamics. Chem. Phys. Lett. 314, 141.](https://doi.org/10.1016/S0009-2614(99)01123-9) -- related: temperature-based enhanced sampling
- [Simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing) -- temperature modulation for barrier crossing
- [KAM theorem](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold%E2%80%93Moser_theorem) -- why single thermostats fail on HO
- Parent orbit: #3 (log-osc-001) -- base log-osc thermostat
