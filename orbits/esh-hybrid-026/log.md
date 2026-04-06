---
strategy: esh-hybrid-026
status: complete
eval_version: eval-v1
metric: 1.1047
issue: 26
parent: esh-duality-018
---

# ESH Hybrid: ESH Trajectory + 1/f Thermostat

Primary metric: KL(ESH hybrid) = **1.1047** on 2D GMM (500k evals, 3 seeds)
Champion KL = 0.054 (reference). **ESH hybrid fails — 20x worse than champion.**

## Liouville Analysis (make_liouville_analysis.py)

Naive combination: dx/dt = sign(v), dv/dt = -dU/dx*|v| - g(xi)*v, dxi/dt = (v²-kT)/Q

div(F) = df_x/dx + df_v/dv + df_xi/dxi
       = 0 + (-dU/dx*sign(v) - g(xi)) + 0
       = **-dU/dx*sign(v) - g(xi)  ≠ 0**

Numerically verified at 4 test points — all non-zero as predicted.

**Conclusion**: Naive hybrid has no simple stationary measure. No factored ansatz
rho(x,v)*h(xi) satisfies div(rho*F) = 0.

**Fix Option C (alternating)**: Keep ESH steps and thermostat steps separate.
- ESH steps: conservative, explore H_ESH = U+log|v| level sets
- Thermostat steps: thermalize to canonical measure
- No Liouville compatibility issue (each phase independently correct)

## Benchmark Results (benchmark_results.json, 3 seeds × 500k evals)

| Sampler             | KL mean ± std   | KL min |
|---------------------|-----------------|--------|
| NHC (M=3)           | 0.346 ± 0.170   | 0.105  |
| NHCTail (champion)  | 0.092 ± 0.025   | 0.058  |
| ESH + refresh       | 0.265 ± 0.012   | 0.256  |
| **ESH hybrid (C)**  | 1.105 ± 0.092   | 1.015  |
| ESH hybrid L5       | 1.651 ± 0.272   | 1.267  |

Champion reference: 0.054 (from multiscale-chain-009 with more evals).

## Why ESH Hybrid Fails

The ESH unit-speed position update **dx = dt * sign(v)** creates pathological dynamics:
- Particle moves at unit speed in fixed ±x, ±y directions
- Position trajectory becomes quasi-1D grid-like exploration
- For ring-shaped GMM (5 modes), this misses the circular geometry entirely
- More ESH steps per thermostat step makes it WORSE (L5 > L1)

Physical interpretation: ESH is designed for 1D potentials or well-separated modes
along coordinate axes. For multi-modal distributions with curved mode geometry,
the sign(v) discretization destroys the smooth trajectory needed for mode hopping.

ESH + stochastic refresh (KL=0.265) does better than pure NHC (0.346) — the ESH
dynamics provide some fast local exploration — but the thermostat beats both by 3x.

## Files

- `make_liouville_analysis.py` — symbolic + numerical divergence check
- `make_hybrid_sampler.py` — ESHPlusThermostat implementation (Option C)
- `make_esh_hybrid_benchmark.py` — full benchmark (3 configs × 3 seeds)
- `make_phase_space.py` — phase space analysis
- `benchmark_results.json` — full results with all seeds
- `esh_hybrid_results.json` — supplementary benchmark
- `figures/` — phase space plots, level set hopping

## Conclusion

**Dead-end**: ESH hybrid with log-osc thermostat does not improve over the champion.
The Liouville barrier (non-zero divergence) is patched by Option C, but the
fundamental issue is the ESH unit-speed dynamics which creates pathological
exploration patterns in multi-modal 2D distributions.

**Insight for paper**: ESH and our thermostats are fundamentally different mechanisms —
ESH is a conservative Hamiltonian system, our thermostats are dissipative with
provable canonical stationary measure. The "scale-free" analogy between them
(logarithmic KE vs 1/f friction) does not translate to improved sampling when combined.
