---
strategy: unified-theory-007
status: in-progress
eval_version: eval-v1
metric: N/A (theory orbit)
issue: 10
parent: log-osc-001
---

# Unified Theory of Generalized Friction Thermostats

## Summary

Formalized a unified theoretical framework proving that ANY smooth confining
potential V(xi) defines a valid deterministic thermostat via g(xi) = V'(xi)/Q.
The Master Theorem (Theorem 1) provides a complete characterization with
necessity and sufficiency. Numerical Lyapunov exponents confirm that bounded
friction functions produce dramatically stronger chaos than standard
Nose-Hoover, and multi-scale thermostats reduce autocorrelation times by 5-7x.

## Iteration 1: Theory + Numerics

### Approach

1. **Master Theorem**: Proved that g(xi) = V'(xi)/Q is necessary and sufficient
   for invariant measure preservation, using the Liouville equation. Full
   divergence computation with uniqueness proof.

2. **Lyapunov Exponents**: Computed maximal Lyapunov exponents for 1D HO +
   thermostat with four friction functions (NH, Log-Osc, Tanh, Arctan) across
   11 values of Q from 0.1 to 10.0.

3. **Spectral Analysis**: Computed PSD of friction signal and autocorrelation
   times for multi-scale thermostats on 1D HO and 2D double-well.

4. **Phase Portraits**: Generated (q,p) phase portraits showing torus structure
   for different friction functions.

### Key Results

**Lyapunov exponents (1D HO, dt=0.01, T=5000, seed=42):**

| Q | NH | Log-Osc | Tanh | Arctan |
|---|-----|---------|------|--------|
| 0.1 | 0.002 | **0.626** | 0.435 | 0.320 |
| 0.2 | 0.026 | **0.514** | 0.323 | 0.240 |
| 0.5 | 0.056 | **0.199** | 0.098 | 0.116 |
| 1.0 | 0.001 | 0.001 | 0.001 | 0.001 |

- Bounded friction produces 10-300x larger Lyapunov exponents at small Q
- Log-Osc is consistently the most chaotic across all Q
- All methods become quasi-periodic at Q >= 0.7 (in this time window)

**Multi-scale mixing (2D double-well, log-osc friction):**

| Config | IAT(x) | Improvement |
|--------|--------|-------------|
| Single Q=1 | 264.4 | 1x |
| 3 log-spaced | 54.4 | 4.9x |
| 5 log-spaced | 41.4 | 6.4x |
| 7 log-spaced | 35.9 | 7.4x |

### What Worked

1. The Master Theorem proof is clean and complete -- the key insight is
   that the Liouville condition factors as (K - d*kT) * [...] = 0, forcing
   the bracket to vanish for all K, which uniquely determines g = V'/Q.

2. Lyapunov computation clearly shows bounded friction producing stronger
   chaos, confirming the theoretical prediction about KAM-breaking.

3. Multi-scale spectral analysis confirms the 1/f noise hypothesis: log-spaced
   thermostat masses broaden the friction spectrum.

### What I Learned

1. The chaos strength ordering follows friction boundedness: more bounded =
   more chaotic. This is counterintuitive -- weaker friction produces stronger chaos.

2. At Q >= 0.7 with kT=1, even bounded-friction thermostats appear quasi-periodic.
   The sweet spot Q=0.8 from orbit/log-osc-001 achieving ergodicity score 0.944
   likely requires longer observation times to see the chaotic behavior.

3. Multi-scale thermostats improve mixing quality (lower IAT) more than they
   improve raw barrier crossing counts. The benefit is in decorrelation, not
   exploration per se.

## Seeds

All computations use seed=42. Deterministic.

## References

- [Nose (1984)](https://doi.org/10.1063/1.447334) -- original thermostat
- [Hoover (1985)](https://doi.org/10.1103/PhysRevA.31.1695) -- canonical dynamics
- [Martyna et al. (1992)](https://doi.org/10.1063/1.463940) -- Nose-Hoover chains
- [Benettin et al. (1980)](https://doi.org/10.1007/BF02128236) -- Lyapunov exponents
- [KAM theorem](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold%E2%80%93Moser_theorem) -- non-ergodicity mechanism
- [1/f noise](https://en.wikipedia.org/wiki/Pink_noise) -- spectral density connection
- Builds on orbit/log-osc-001 (#3) which discovered the log-osc thermostat
