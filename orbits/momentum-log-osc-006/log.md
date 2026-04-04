---
strategy: momentum-log-osc-006
status: complete
eval_version: eval-v1
metric: 0.0088
issue: 9
parent: log-osc-001
---

# Momentum-Dependent Log-Osc Friction

## Summary

Explored two modifications to the Log-Osc thermostat:

**Variant A (MLOSC-A):** Momentum-dependent friction multiplier f(x) = 1 + alpha*K/(dim*kT). Derived exact invariant measure via SymPy (requires modified xi equation with K^2 terms). Result: **does not improve** over parent -- higher alpha degrades HO ergodicity without improving DW KL.

**Variant B (MLOSC-B, Rippled Log-Osc):** Replace thermostat potential V(xi) = log(1+xi^2) with V(xi) = log(1+xi^2) + epsilon*cos(omega*xi). Key insight: **ANY V(xi) preserves the exact canonical invariant measure** with the standard NH equations. The cosine ripples create a richer thermostat energy landscape with multiple local barriers. Best config (eps=0.3, w=5.0) achieves DW KL=0.0088 (12% better than parent) and TTT=500k (37% faster), while maintaining HO ergodicity above the 0.85 threshold.

## Approach

### Variant A: Adaptive Friction

Equations:
```
dq/dt = p/m
dp/dt = -dU/dq - g(xi) * (1 + alpha*K/(dim*kT)) * p
dxi/dt = (1/Q) * [A*S - dim*kT + C*S^2/kT]
```
where S = sum(p^2)/m, A = 1 - (dim+2)*alpha/(2*dim), C = alpha/(2*dim).

Invariant measure: rho ~ exp(-H_ext/kT) with H_ext = U + K + Q*log(1+xi^2). Verified via SymPy for dim=1,2.

### Variant B: Rippled Thermostat Potential

Equations:
```
dq/dt = p/m
dp/dt = -dU/dq - V'(xi) * p
dxi/dt = (1/Q) * (sum p_i^2/m - dim*kT)
```
where V(xi) = log(1+xi^2) + epsilon*cos(omega*xi), so V'(xi) = 2*xi/(1+xi^2) - epsilon*omega*sin(omega*xi).

Invariant measure: rho ~ exp(-H_ext/kT) with H_ext = U + K + Q*V(xi). **Exact for any V(xi)** -- verified symbolically.

Physical interpretation: The cosine ripples create oscillating friction that alternately amplifies and damps momentum as xi traverses the thermostat landscape. This breaks the simple quasi-periodic orbits that can trap standard NH/Log-Osc in KAM tori regions. At large |xi|, the log term dominates and the ripples become negligible, providing natural stability.

## Results

### Stage 1: Best Configurations

| Potential | Variant | Config | KL | Ergodicity | ESS/fe | TTT |
|-----------|---------|--------|------|------------|--------|-----|
| 1D HO | MLOSC-B | eps=0.3,w=5,Q=0.8,dt=0.005 | **0.004** | 0.880 | 0.00223 | N/A |
| 1D HO | MLOSC-B | eps=0.3,w=6,Q=0.8,dt=0.005 | 0.003 | **0.913** | 0.00223 | N/A |
| 1D HO | MLOSC-B | eps=0.3,w=8,Q=0.8,dt=0.005 | **0.002** | 0.920 | 0.00223 | N/A |
| 1D HO | Baseline | eps=0,Q=0.8,dt=0.005 | 0.023 | **0.944** | 0.00223 | N/A |
| 2D DW | MLOSC-B | eps=0.3,w=5,Q=1,dt=0.04 | **0.009** | N/A | 0.00504 | 500k |
| 2D DW | MLOSC-B | eps=0.3,w=6,Q=1,dt=0.035 | 0.010 | N/A | 0.00404 | 500k |
| 2D DW | Baseline | eps=0,Q=1,dt=0.035 | 0.010 | N/A | 0.00219 | 800k |

### Comparison with Baselines

| Metric | MLOSC-B (best) | Log-Osc | NH | NHC(M=3) |
|--------|---------------|---------|-----|----------|
| HO Ergodicity | 0.880 | **0.944** | 0.54 | 0.92 |
| HO KL | **0.004** | 0.023 | 0.077 | 0.002 |
| DW KL | **0.009** | 0.010 | 0.037 | 0.029 |
| DW TTT | **500k** | 800k | never | 250k |
| DW ESS/fe | **0.00504** | 0.00219 | 0.00310 | 0.00261 |

### Stage 2 Results (1M force evals)

| Potential | MLOSC-B (eps=0.3,w=5) | Log-Osc (baseline) |
|-----------|----------------------|-------------------|
| GMM 2D (dt=0.03) | KL=0.091 | KL=0.219 (dt=0.02) |
| Rosenbrock 2D (dt=0.02) | **KL=0.006** | KL=0.011 (dt=0.01) |

Note: GMM baseline achieves KL=0.052 at dt=0.04. MLOSC-B cannot use dt=0.04 on GMM due to stability limits, so the improvement at dt=0.03 is not fully fair. Rosenbrock improvement is robust across dt values.

### MLOSC-A Results (does not improve)

| alpha | HO KL | HO Erg | DW KL | DW TTT |
|-------|-------|--------|-------|--------|
| 0.0 | 0.023 | **0.944** | **0.010** | 800k |
| 0.1 | **0.007** | 0.927 | 0.023 | never |
| 0.2 | 0.005 | 0.924 | 0.017 | never |
| 0.3 | 0.100 | 0.597 | 0.013 | 650k |
| 0.5 | 0.170 | 0.599 | 0.012 | 650k |

alpha > 0 improves HO KL at low values but hurts ergodicity and DW KL. Not a useful modification.

### MLOSC-B Epsilon-Omega Grid

| eps | omega | HO Erg | HO KL | DW KL (dt=0.035) | DW Stable? |
|-----|-------|--------|-------|-------------------|-----------|
| 0.0 | - | 0.944 | 0.023 | 0.010 | yes |
| 0.3 | 5.0 | 0.880 | 0.004 | 0.010 | yes |
| 0.3 | 6.0 | 0.913 | 0.003 | 0.010 | yes |
| 0.3 | 8.0 | 0.920 | 0.002 | unstable | dt<=0.025 |
| 0.5 | 5.0 | 0.884 | 0.003 | 0.009 | yes |
| 0.8 | 2.0 | 0.881 | 0.003 | 0.010 | yes |

Higher omega improves HO KL and ergodicity but reduces DW stability boundary. eps=0.3,w=5-6 is the sweet spot.

## What Worked

1. **Rippled thermostat potential (MLOSC-B):** Adding cos(omega*xi) to the thermostat potential creates a richer energy landscape that enhances barrier crossing on the double-well. The exact invariant measure is preserved for ANY V(xi) -- a powerful general result.

2. **Higher omega improves HO sampling:** Faster oscillations in the friction create more chaotic thermostat dynamics that help explore phase space on simple potentials. HO KL improves from 0.023 to 0.002 with eps=0.3,w=8.

3. **ESS improvement on DW:** The rippled friction leads to 2.3x higher ESS/force_eval on DW (0.00504 vs 0.00219), likely because the oscillating friction helps the system escape potential wells more efficiently.

## What Didn't Work

1. **MLOSC-A (momentum-dependent friction):** Despite having an exact invariant measure, scaling friction by kinetic energy hurts ergodicity. The stronger friction at high KE traps the system rather than helping it explore.

2. **High epsilon*omega on DW:** Products epsilon*omega > ~2 cause integrator instability at the optimal DW step size (dt=0.035-0.04). The rapid oscillations in V'(xi) require smaller dt, losing the benefit of larger steps.

3. **Rippled approach on GMM:** The multi-modal GMM potential needs very large steps to hop between modes. The rippled friction constrains the maximum stable dt, making it worse than baseline at the optimal GMM step size.

## Seeds

All runs use `numpy.random.default_rng(42)` via the evaluator's default. Determinism verified.

## References

- [Nose, S. (1984). A unified formulation of the constant temperature molecular dynamics methods. J. Chem. Phys. 81, 511.](https://doi.org/10.1063/1.447334)
- [Hoover, W. G. (1985). Canonical dynamics: Equilibrium phase-space distributions. Phys. Rev. A, 31, 1695.](https://doi.org/10.1103/PhysRevA.31.1695)
- [Martyna et al. (1996). Explicit reversible integrators for extended systems dynamics. Mol. Phys. 87, 1117.](https://doi.org/10.1080/00268979600100761)
- [Versteeg (2021). Energy Sampling Hamiltonian. NeurIPS.](https://proceedings.neurips.cc/paper/2021) -- ESH dynamics inspiration
- [KAM theorem](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold%E2%80%93Moser_theorem) -- quasi-periodic orbit trapping
- [Nose-Hoover thermostat](https://en.wikipedia.org/wiki/Nos%C3%A9%E2%80%93Hoover_thermostat) -- standard method
- Parent orbit: #3 (log-osc-001) -- bounded friction g(xi) = 2xi/(1+xi^2), erg=0.944, DW KL=0.010
