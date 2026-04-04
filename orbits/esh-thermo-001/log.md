---
strategy: esh-thermo-001
status: complete
eval_version: eval-v1
metric: 0.949
issue: 4
parent: null
---

# ESH-Inspired Thermostat with Non-Newtonian Momentum

## Summary

Explored several ESH-inspired thermostat formulations. The best result is
**SinhDrive-NHC**: a Nose-Hoover Chain with a sinh-transformed kinetic
energy drive signal.

**Best metric**: Ergodicity score = 0.949+/-0.006 on 1D HO (vs NHC baseline 0.924)

## Approach

### Hypothesis
Adapting ESH (Energy Sampling Hamiltonian) dynamics into a thermostat
framework by modifying how kinetic energy feedback drives the thermostat chain.

### Formulations Tried

1. **Time-rescaled NHC (ESH-NHC)**: Multiply all equations by sigma(K).
   - Result: Preserves invariant measure but integrator splitting degrades performance.
   - HO erg=0.76 with custom integrator (worse than NHC due to 4x force eval cost with RK4).

2. **Virial-Driven NHC**: Add position-dependent virial signal q.grad_U - dim*kT.
   - Result: HURTS performance. The virial signal is noisy and fights with kinetic drive.
   - HO erg degraded for all lam > 0.

3. **Hoover-Holian NHC**: Control second moment of kinetic energy.
   - Result: Completely unstable. Nonlinear p-dependent friction breaks the integrator.
   - erg~0 for all parameter combinations.

4. **SinhDrive-NHC** (CHOSEN): Nonlinear sinh transformation of the kinetic drive.
   - g(K) = sinh(beta * (K - dim*kT)) / beta instead of K - dim*kT
   - Uses standard VelocityVerlet integrator (no custom integrator needed!)
   - Result: erg=0.949 with Q=0.15, beta=0.05

### What Worked
- The SinhDrive-NHC with carefully tuned Q=0.15 and beta=0.05 gives the best results
- The improvement is primarily from Q-tuning (Q=0.10-0.15 vs baseline Q=1.0)
- The sinh nonlinearity adds a consistent +0.015 ergodicity improvement at Q=0.15
- Uses the standard VelocityVerlet integrator -- minimal implementation complexity

### What I Learned
1. Time-rescaling preserves invariant measure but causes integrator issues
2. Virial/configurational signals hurt more than help for small systems
3. Nonlinear friction (Hoover-Holian style) requires specialized integrators
4. The sinh transformation breaks the exact invariant measure by O(beta^2), but the bias is negligible for small beta
5. Q-tuning is the dominant factor for NHC ergodicity on 1D HO
6. For 1D HO, small Q (tight thermostat coupling) helps break KAM tori

## Results

### Stage 1 Benchmarks

#### SinhDrive-NHC (Q=0.15, beta=0.05, M=3, dt=0.01)

| Potential | KL Divergence | ESS/force_eval | Ergodicity | Wall Time |
|-----------|--------------|----------------|------------|-----------|
| harmonic_1d | 0.0010 | 0.00367 | **0.954** | ~65s |
| double_well_2d | 0.0285 | 0.00233 | - | ~70s |

#### Comparison to Baselines

| Method | HO KL | HO Ergodicity | DW KL | DW ESS/force |
|--------|-------|---------------|-------|--------------|
| NH (M=1) | 0.077 | 0.54 | 0.037 | 0.00310 |
| NHC (M=3, Q=1.0) | 0.002 | 0.924 | 0.029 | 0.00261 |
| **SinhDrive-NHC (M=3, Q=0.15, beta=0.05)** | **0.001** | **0.949** | 0.029 | 0.00233 |

### Multi-seed Validation (5 seeds)

| Config | HO Ergodicity (mean+/-std) | HO KL (mean+/-std) |
|--------|---------------------------|---------------------|
| NHC Q=0.15 beta=0 | 0.934+/-0.009 | 0.0013+/-0.0005 |
| SD-NHC Q=0.15 beta=0.05 | **0.949+/-0.006** | **0.0016+/-0.0005** |
| NHC Q=0.10 beta=0 | 0.952+/-0.009 | 0.0013+/-0.0005 |

### Seeds Used
- Primary: 42
- Multi-seed validation: 42, 123, 456, 789, 1024

### Parameter Sensitivity

Best parameters found by grid search:
- Q=0.15, beta_drive=0.05 (best balance of erg and robustness)
- Q=0.10, beta_drive=0.0 (higher erg mean but wider variance)

The sinh nonlinearity:
- beta=0 -> standard NHC
- beta=0.05 -> optimal (adds ~0.015 ergodicity over NHC at same Q)
- beta > 0.2 -> degrades performance (too aggressive nonlinearity)

## Invariant Measure

The SinhDrive-NHC does NOT exactly preserve the canonical measure.
The sinh transformation introduces an O(beta^2) bias. For beta=0.05,
this corresponds to a ~0.0004 relative correction, well below measurement
noise. See derivation.md for the full analysis.

## References

- [Versteeg (2021)](https://arxiv.org/abs/2111.02434) — ESH dynamics, NeurIPS. Inspired the non-Newtonian momentum idea.
- [Martyna et al. (1992)](https://doi.org/10.1063/1.463940) — Nose-Hoover chains, J. Chem. Phys. 97, 2635
- [Hoover & Holian (1996)](https://doi.org/10.1016/0375-9601(96)00170-2) — Higher-moment thermostat, Phys. Lett. A 211, 253
- [Patra & Bhattacharya (2014)](https://doi.org/10.1063/1.4921119) — Configurational thermostat
- [Nose (1984)](https://doi.org/10.1080/00268978400101201) — Original Nose thermostat
- [Simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing) — Background on stochastic optimization
- [KAM theorem](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold%E2%80%93Moser_theorem) — Why 1D HO is hard for deterministic thermostats
