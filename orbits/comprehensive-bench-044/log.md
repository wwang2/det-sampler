---
strategy: comprehensive-bench-044
type: experiment
status: in-progress
eval_version: eval-v1
metric: null
issue: 44
parents:
  - orbit/q-optimization-035
  - orbit/q-omega-mapping-040
---

# Comprehensive All-Method Benchmark

## Glossary

- **LogOsc**: Log-oscillator thermostat with parallel multi-scale Q values
- **NHC**: Nose-Hoover Chain thermostat
- **NH-1**: Single Nose-Hoover thermostat
- **tau_int**: Integrated autocorrelation time (lower = better mixing)
- **Q**: Thermostat mass parameter
- **kappa**: Spring constant / stiffness
- **BAOAB**: Symmetric splitting integrator (B=kick, A=drift, O=thermostat)

## Approach

Six methods compared on six targets, with parameter tuning followed by final evaluation:

**Methods:**
1. LogOsc N=3: Our multi-scale parallel thermostat, Q_min=kT/kappa_max, Q_max=kT/kappa_min, log-uniform spacing
2. LogOsc N=5: Same range, 5 thermostat variables
3. NHC M=3: Nose-Hoover Chain, Q swept over {0.1, 1.0, 10.0, 31.6, 100.0}
4. NHC M=5: Same sweep
5. Underdamped Langevin (BAOAB): gamma swept over {0.1, 1.0, 3.0, 10.0}
6. NH single: Q swept over {0.1, 1.0, 10.0, 100.0}

**Targets:**
1. 1D harmonic (omega=1)
2. 2D double-well (barrier=1, y_stiffness=0.5)
3. 2D GMM 5-mode (radius=3, sigma=0.5)
4. 5D anisotropic Gaussian (kappa_ratio=100)
5. 10D anisotropic Gaussian (kappa_ratio=100)
6. 10D GMM 5-mode

**Protocol:**
- Tuning: 3 seeds x 100k force evals (160s)
- Final: 10 seeds x 400k force evals (723s)
- dt=0.03 for easy targets, dt=0.005 for stiff (anisotropic Gaussian)
- Pool(processes=10), total runtime: 885s

## Results (Run 1)

### Summary Table (tau_int for unimodal, mode crossings for multimodal)

| Target         | LogOsc-3 | LogOsc-5 | NHC-3 | NHC-5 | Langevin | NH-1 |
|----------------|----------|----------|-------|-------|----------|------|
| 1D Harmonic    | 7.7      | 7.8      | **5.6** | 5.8   | 16.2     | **3.4** |
| 2D DoubleWell  | 20.0     | 16.8     | 21.3  | 20.4  | 33.6     | **14.2** |
| 2D GMM (cross) | 35       | 60       | 30    | 37    | 27       | **182** |
| 5D AnisoGauss  | 1557     | 560      | 741   | **8.6** | 59.9   | **8.5** |
| 10D AnisoGauss | 1242     | 636      | **6.1** | 6.3  | 54.4     | **5.5** |
| 10D GMM (cross)| 7        | 310      | 21    | 32    | 22       | **570** |

**Bold** = best or near-best per target.

### Key Findings

1. **NH-1 is surprisingly competitive.** Plain Nose-Hoover wins or ties on 4/6 targets. This contradicts the common wisdom that NH is inferior. With tuned Q, NH-1 is excellent for high-D Gaussians and GMMs. Its best Q values (10-100) are much larger than the default Q=1.

2. **LogOsc struggles badly on high-D stiff systems.** On 5D and 10D anisotropic Gaussians, LogOsc is 70-225x worse than NH-1/NHC-5. The Q range [kT/kappa_max, kT/kappa_min] = [0.01, 1.0] is far too small. The optimal Q for these targets is 10-100, matching the dimension * kT scale.

3. **NHC-5 excels on anisotropic Gaussians.** With tuned Q_ref=31.6, NHC-5 achieves tau=8.6 on 5D (vs 560 for LogOsc-5). The longer chain helps ergodicity.

4. **Langevin is mediocre everywhere.** The stochastic baseline never wins. On 2D GMM it gets only 27 mode crossings (vs 182 for NH-1). Its optimal gamma varies widely by target.

5. **10D GMM is challenging for all methods.** NH-1 dominates (570 crossings) but LogOsc-5 does surprisingly well (310 crossings). NHC methods are poor (21-32 crossings).

### Best Tuning Parameters

| Target         | LogOsc-3 | LogOsc-5 | NHC-3  | NHC-5  | Langevin | NH-1   |
|----------------|----------|----------|--------|--------|----------|--------|
| 1D Harmonic    | auto     | auto     | Q=1.0  | Q=1.0  | g=1.0    | Q=1.0  |
| 2D DoubleWell  | auto     | auto     | Q=0.1  | Q=0.1  | g=1.0    | Q=0.1  |
| 2D GMM         | auto     | auto     | Q=10.0 | Q=0.1  | g=0.1    | Q=10.0 |
| 5D AnisoGauss  | auto     | auto     | Q=10.0 | Q=31.6 | g=3.0    | Q=10.0 |
| 10D AnisoGauss | auto     | auto     | Q=100  | Q=31.6 | g=3.0    | Q=100  |
| 10D GMM        | auto     | auto     | Q=1.0  | Q=0.1  | g=0.1    | Q=0.1  |

Note: LogOsc uses auto-computed Q range; NHC/NH are swept and tuned.

## What Happened

The LogOsc thermostat's automatic Q-range formula Q_min=kT/kappa_max, Q_max=kT/kappa_min produces values that are too small for high-D systems. For a 10D system with kappa_ratio=100, this gives Q in [0.01, 1.0], but the optimal Q is 10-100 (as found by NHC/NH tuning). The thermostat mass should scale with the number of degrees of freedom, not just individual spring constants.

## What I Learned

1. The Q-range prescription from q-optimization-035 breaks down badly in high dimensions. The formula was validated on 5D but with optimized (not auto) Q values.
2. NH single with tuned Q is an underappreciated baseline. It beats NHC on several targets.
3. For GMMs, deterministic thermostats (NH-1) dramatically outperform stochastic methods (Langevin) at mode hopping.
4. The optimal Q for NHC/NH scales roughly as Q ~ dim * kT / omega_typ^2, not Q ~ kT / kappa.

## Prior Art & Novelty

### What is already known
- NHC superiority over NH for ergodicity in small systems [Martyna et al. 1992]
- Q tuning importance documented in [Hoover 1985] and [Martyna et al. 1996]
- Underdamped Langevin BAOAB splitting [Leimkuhler & Matthews 2013]

### What this orbit adds
- Systematic head-to-head comparison of 6 methods on 6 targets with tuned parameters
- Evidence that NH-1 with tuned Q is surprisingly competitive (not commonly reported)
- Identification of LogOsc Q-range failure mode in high-D stiff systems

### Honest positioning
This is a benchmarking orbit, not a novelty claim. The primary contribution is empirical comparison data for the paper's benchmark section.

## References

- Martyna, G.J., Klein, M.L., Tuckerman, M. (1992). NHC thermostats. J. Chem. Phys. 97, 2635.
- Hoover, W.G. (1985). Canonical dynamics. Phys. Rev. A 31, 1695.
- Leimkuhler, B., Matthews, C. (2013). Rational construction of stochastic numerical methods. Appl. Math. Res. Express.

## Seeds & Reproducibility
- Tuning seeds: 1000, 1001, 1002
- Final seeds: 5000-5009
- GMM mode centers: default from GaussianMixture2D (5 modes, ring)
- 10D GMM centers: GaussianMixtureND(seed=0) random unit vectors
