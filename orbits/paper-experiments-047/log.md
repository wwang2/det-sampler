---
strategy: paper-experiments-047
type: experiment
status: in-progress
eval_version: eval-v1
metric: null
issue: 47
parents:
  - orbit/friction-survey-045
  - orbit/comprehensive-bench-044
  - orbit/q-optimization-035
---

# Paper-Ready Final Experiments

## Glossary

- **NHC**: Nose-Hoover Chain thermostat (Martyna et al. 1992)
- **NH**: Single Nose-Hoover thermostat (Nose 1984, Hoover 1985)
- **GMM**: Gaussian Mixture Model (multi-modal target distribution)
- **TV**: Total Variation distance (0.5 * sum |p_i - q_i|)
- **IQR**: Interquartile Range (25th to 75th percentile)
- **BAOAB**: Splitting scheme for Langevin dynamics (Leimkuhler & Matthews 2015)
- **tau_int**: Integrated autocorrelation time of q^2

## Approach

Definitive paper experiments comparing the tanh parallel thermostat against NHC (M=3), single NH, and BAOAB Langevin baselines. Three experiments:

1. **E1 (Dimension Scaling)**: How does each method scale with dimension on (a) anisotropic Gaussian (kappa=100, tau_int metric) and (b) Gaussian mixture (5 modes, crossings/round-trips/TV)?
2. **E2 (Friction Validation)**: Compare three bounded friction functions -- tanh, arctan, log-osc -- all with N=5 parallel thermostats, tuned Q per friction. Tests g'>=0 hypothesis.
3. **E3 (Q Range Validation)**: Compare Q-setting strategies for the tanh thermostat: D*kT/w^2 formula, kT/w^2, fixed ranges [50,500], [10,1000], and dim-scaled [5D, 100D].

### Key Design Decisions

- **All baselines tuned per target per dimension** via parameter sweep (3 seeds on 100k tuning runs)
- **20 seeds** for final runs, reporting **median + IQR** error bars
- **GMM parameters**: radius=3.0, sigma=1.0 (validated to be tractable for deterministic thermostats)
- **Anisotropic Gaussian**: kappa=100, frequencies from 1 to 10

### Exploratory Finding: Q Range

The Q=D*kT/w^2 formula from orbit #044 gives Q values that are too small for the tanh thermostat. Exploratory tests showed:
- tanh needs Q >= 50 (preferably 100-1000) to work well
- This is because |g(xi)| = |tanh(xi)| <= 1, so the thermostat coupling saturates
- NHC uses g(xi) = xi (unbounded), so smaller Q gives faster coupling
- The effective Q scale for tanh is ~10x-100x larger than for NHC

When properly tuned, tanh parallel (N=5) matches or slightly beats NHC (M=3) on tau_int.

### Single-seed exploratory results (d=10, aniso kappa=100, 200k steps)

| Method | tau_int |
|--------|---------|
| tanh Q=50 (N=5, identical) | 2.6 |
| tanh Q=100 (N=5, identical) | 2.4 |
| tanh spread [10,1000] (N=5) | 2.4 |
| NHC M=3, Q=50 | 2.9 |
| NHC M=5, Q=50 | 2.9 |
| NH Q=50 | 2.3 |

### Dimension scaling (single seed, tuned)

| dim | tanh (best) | NHC3 (best) | NHC5 (best) | NH (best) |
|-----|------------|-------------|-------------|-----------|
| 2   | 3.8        | 35.0        | 31.4        | 3.8       |
| 5   | 2.3        | 3.1         | 10.8        | 2.8       |
| 10  | 2.5        | 3.0         | 2.7         | 2.5       |
| 20  | 5.5        | 5.1         | 5.1         | 5.6       |
| 50  | 6.2        | 4.9         | 5.0         | 6.1       |

Key: tanh and NH beat NHC at low dimensions (d=2,5); all comparable at high dim.

## Results

Pending -- full 20-seed experiment running.

## Prior Art & Novelty

### What is already known
- NHC thermostats: [Martyna et al. (1992)](https://doi.org/10.1063/1.463940)
- Bounded friction thermostats explored by [Samoletov et al. (2007)](https://doi.org/10.1007/s10955-007-9365-2)
- Parallel (multi-Q) thermostat idea: related to "massive thermostatting" in [Tobias et al. (1993)](https://doi.org/10.1021/j100120a038)

### What this orbit adds (if anything)
- Systematic comparison of tanh parallel thermostat vs NHC across dimensions 2-50
- Empirical validation that tanh achieves comparable mixing to NHC without chain coupling
- Evidence that the Q range (not Q formula) is what matters for bounded friction thermostats
- Comprehensive multi-seed statistics with proper baseline tuning

### Honest positioning
This orbit applies known bounded friction functions in a parallel thermostat configuration and benchmarks them against established methods. The tanh friction function is known; the parallel thermostat structure is a variant of massive thermostatting. The contribution is the systematic empirical comparison showing competitive performance without chain coupling, which simplifies implementation and analysis. No strong novelty claim.

## References

- Martyna et al. (1992) "Nose-Hoover chains" J. Chem. Phys. 97, 2635
- Samoletov et al. (2007) "Thermostats for slow configurational modes" J. Stat. Phys. 128, 1321
- Leimkuhler & Matthews (2015) "Molecular Dynamics" Springer
- Orbit #045 (friction-survey-045): tanh has no frequency ceiling (omega_max = 1.0)
- Orbit #035 (q-optimization-035): Q = D*kT/w^2 formula
