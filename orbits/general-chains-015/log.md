---
strategy: general-chains-015
status: complete
eval_version: eval-v1
metric: 0.0018
issue: 17
parent: unified-theory-007
seed: 42
---

# Generalized NHC Chains via K_eff = xi * V'(xi)

## Result

Derived and implemented the Universal Chain Coupling Formula: K_eff(xi) = xi * V'(xi),
which generalizes NHC chains to arbitrary thermostat potentials. Machine-verified via
SymPy (Liouville equation for 1, 2, 3-level chains with general V). Benchmarked four
chain types on HO and DW with 1M force evals each.

| Chain | HO KL | HO Ergodicity | DW KL |
|-------|-------|---------------|-------|
| NHC(M=3) | 0.0087 | 0.889 | 0.029 |
| LogOscChain(M=3) | 0.0030 | 0.907 | 0.030 |
| TanhChain(M=3) | 0.0027 | 0.911 | 0.032 |
| ArctanChain(M=3) | 0.0018 | 0.894 | 0.031 |

Best HO KL: ArctanChain = 0.0018 (vs NHC baseline 0.0087, 4.8x improvement).
All chains pass ergodicity threshold (> 0.85).
DW KL: all chains competitive at ~0.03 (NHC baseline: 0.029).

## Approach

1. **Theory:** Proved that K_eff(xi) = xi * V'(xi) is the correct effective kinetic
   energy for chain coupling. Proved <K_eff> = kT via integration by parts.
   Derived generalized chain dynamics and proved Liouville invariance.

2. **Verification:** SymPy machine-checked div(rho*F) = 0 for:
   - Single thermostat with general V(xi)
   - 2-level generalized chain
   - 3-level generalized chain
   - Specific cases: NH, Log-Osc, Tanh, Arctan
   - Numerical K_eff equipartition for all 4 potentials

3. **Implementation:** GeneralizedChainThermostat class accepting arbitrary
   thermostat potentials. Custom Velocity Verlet integrator with analytical
   friction rescaling exp(-g_1*dt/2).

4. **Benchmarks:** 4 chain types x 2 potentials x 1M force evals.
   Seed = 42 throughout.

## What Happened

- All bounded-friction chains (Log-Osc, Tanh, Arctan) beat NHC on the
  harmonic oscillator KL metric, with Arctan being the best.
- Ergodicity scores are all above 0.85, with Tanh marginally highest (0.911).
- On the double well, performance is similar across all chains (~0.03 KL).
  The bounded friction doesn't help barrier crossing significantly.
- The K_eff formula correctly recovers NHC when V is quadratic.

## What I Learned

- Bounded friction helps ergodicity in small systems (HO) by preventing the
  thermostat from over-driving the momentum. The gentler coupling of Tanh/Arctan
  avoids quasi-periodic orbits that plague standard NHC.
- For barrier crossing (DW), all chains perform similarly because the bottleneck
  is the physical energy barrier, not the thermostat chain dynamics.
- Arctan gives the best HO KL likely because its K_eff grows sublinearly
  (xi*arctan(xi) ~ pi*|xi|/2 for large xi), providing the softest chain coupling.
- The K_eff = xi*V'(xi) formula is genuinely novel -- it provides a principled
  way to extend any thermostat to chains without ad-hoc guessing.

## References

- [Martyna et al. (1992)](https://doi.org/10.1063/1.463940) -- Nose-Hoover chains original paper
- [Martyna et al. (1996)](https://doi.org/10.1080/00268979600100761) -- Explicit reversible integrators
- [Watanabe & Kobayashi (2007)](https://doi.org/10.1103/PhysRevE.75.040102) -- Generalized Nose-Hoover thermostat
- [Nose-Hoover thermostat](https://en.wikipedia.org/wiki/Nos%C3%A9%E2%80%93Hoover_thermostat) -- Wikipedia background
- Parent orbit: unified-theory-007 (Master Theorem for g = V'/Q)
- Related: log-osc-001 (Log-Osc thermostat, bounded friction concept)
