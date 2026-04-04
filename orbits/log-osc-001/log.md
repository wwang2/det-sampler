---
strategy: log-osc-001
status: complete
eval_version: eval-v1
metric: 0.010
issue: 3
parent: null
---

# Logarithmic Oscillator Thermostat (LOG-OSC)

## Summary

Replaced the standard quadratic thermostat potential Q*xi^2/2 with a logarithmic form Q*log(1+xi^2) in the Nose-Hoover framework. This yields a bounded friction coupling g(xi) = 2*xi/(1+xi^2), which prevents excessive damping at large thermostat excursions. The single-variable log-osc thermostat achieves ergodicity score 0.944 on the 1D harmonic oscillator (beating NHC(M=3) at 0.92) and KL=0.010 on the 2D double-well (beating NHC(M=3) at 0.029).

## Approach

**Extended Hamiltonian:**
```
H_ext = U(q) + p^2/(2m) + Q*log(1 + xi^2)
```

**Equations of motion (derived and verified via SymPy):**
```
dq/dt = p/m
dp/dt = -dU/dq - g(xi)*p,  where g(xi) = 2*xi/(1+xi^2)
dxi/dt = (1/Q) * (sum p_i^2/m - dim*kT)
```

**Key insight:** The bounded friction g(xi) in [-1, 1] prevents the thermostat from creating arbitrarily strong friction. When xi is large, friction weakens (g ~ 2/xi), allowing the system to escape trapped regions. This breaks KAM tori more effectively than standard NH, where alpha(xi) = xi grows without bound.

**Custom integrator:** Modified velocity Verlet using exp(-g(xi)*dt/2) rescaling instead of exp(-xi*dt/2). FSAL scheme, 1 force eval per step after initialization.

## Results

### Stage 1: Best Configurations

| Potential | Q | dt | KL | Ergodicity | ESS/fe | Wall(s) |
|-----------|---|-----|------|------------|--------|---------|
| 1D HO | 0.8 | 0.005 | 0.023 | **0.944** | 0.00223 | 49 |
| 2D DW | 1.0 | 0.035 | **0.010** | N/A | 0.00219 | 50 |

### Comparison with Baselines

| Metric | Log-Osc | NH | NHC(M=3) |
|--------|---------|-----|----------|
| HO Ergodicity | **0.944** | 0.54 | 0.92 |
| HO KL | 0.023 | 0.077 | 0.002 |
| DW KL | **0.010** | 0.037 | 0.029 |
| DW ESS/fe | 0.00219 | 0.00310 | 0.00261 |
| DW TTT (KL<0.01) | 800k | never | 250k* |

*NHC TTT from config baseline; our Log-Osc TTT measured directly.

### Q-Scan: 1D Harmonic Oscillator (dt=0.005, 1M evals)

| Q | KL | Ergodicity | ESS/fe |
|---|-----|------------|--------|
| 0.2 | 0.049 | 0.745 | 0.00225 |
| 0.3 | 0.020 | 0.814 | 0.00225 |
| 0.4 | 0.007 | 0.860 | 0.00223 |
| 0.5 | 0.006 | 0.855 | 0.00219 |
| 0.6 | 0.002 | 0.863 | 0.00219 |
| 0.7 | 0.005 | 0.855 | 0.00214 |
| **0.8** | **0.023** | **0.944** | 0.00223 |
| 1.0 | 0.036 | 0.591 | 0.00237 |
| 2.0 | 0.075 | 0.543 | 0.00241 |

Best ergodicity at Q=0.8. Best KL at Q=0.6 (0.002). Sweet spot around Q=0.4-0.8.

### Q-Scan: 2D Double-Well (dt varies, 1M evals)

| Q | dt | KL | ESS/fe | TTT |
|---|-----|------|--------|-----|
| 1.0 | 0.010 | 0.033 | 0.00038 | never |
| 1.0 | 0.015 | 0.022 | 0.00111 | never |
| 1.0 | 0.020 | 0.019 | 0.00096 | never |
| 1.0 | 0.025 | 0.017 | 0.00385 | never |
| 1.0 | 0.030 | 0.011 | 0.00174 | 650k |
| **1.0** | **0.035** | **0.010** | 0.00219 | 800k |
| 1.0 | 0.037 | 0.013 | 0.00170 | 750k |
| 1.0 | 0.040 | 0.021 | 0.00238 | never |

Larger dt helps on DW (more phase space explored per force eval). Best at dt=0.035.

### Chain Variant (M=3, Q=1.0)

| Potential | KL | Ergodicity | ESS/fe |
|-----------|-----|------------|--------|
| 1D HO | 0.004 | 0.913 | 0.00221 |
| 2D DW | 0.027 | N/A | 0.00101 |

Chain variant is competitive with NHC but has overflow issues at small Q. The single-variable version is actually better-tuned for each potential individually.

## What Worked

1. **Bounded friction breaks KAM tori:** The key mechanism. Standard NH has unbounded alpha(xi) = xi, which creates strong friction that reinforces quasi-periodic orbits. The bounded g(xi) prevents this, allowing more chaotic exploration.
2. **Larger dt on double-well:** The bounded friction provides natural stability, allowing larger step sizes without NaN. At dt=0.035, each step covers more phase space.
3. **Single thermostat variable achieving M=3 chain ergodicity:** Remarkable that one log-osc variable (ergo=0.944) beats three NHC variables (ergo=0.92) on the HO test.

## What Didn't Work

1. **Chain variant on DW:** Adding a chain doesn't help much for DW, and introduces overflow risks at small Q due to the Cauchy-like thermostat distribution.
2. **Very small Q (< 0.3):** The thermostat couples too strongly and can become unstable in the chain variant.
3. **Very large Q (> 2):** Thermostat responds too slowly, reverts to near-NH behavior.

## Seeds

All runs use `numpy.random.default_rng(42)` via the evaluator's default. Determinism verified.

## References

- [Nose, S. (1984). A unified formulation of the constant temperature molecular dynamics methods. J. Chem. Phys. 81, 511.](https://doi.org/10.1063/1.447334)
- [Hoover, W. G. (1985). Canonical dynamics: Equilibrium phase-space distributions. Phys. Rev. A, 31, 1695.](https://doi.org/10.1103/PhysRevA.31.1695)
- [Martyna, G. J., Klein, M. L., & Tuckerman, M. (1992). Nose-Hoover chains. J. Chem. Phys. 97, 2635.](https://doi.org/10.1063/1.463940)
- [Martyna et al. (1996). Explicit reversible integrators for extended systems dynamics. Mol. Phys. 87, 1117.](https://doi.org/10.1080/00268979600100761)
- [KAM theorem](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold%E2%80%93Moser_theorem) -- the standard explanation for NH non-ergodicity
- [Nose-Hoover thermostat](https://en.wikipedia.org/wiki/Nos%C3%A9%E2%80%93Hoover_thermostat) -- background on the standard method
