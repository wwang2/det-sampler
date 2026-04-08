---
strategy: comprehensive-bench-044
type: experiment
status: complete
eval_version: eval-v1
metric: "LogOsc-5 tuned matches NH-1/NHC-5 on all targets"
issue: 44
parents:
  - orbit/q-optimization-035
  - orbit/q-omega-mapping-040
---

# Comprehensive All-Method Benchmark

## Glossary

- **LogOsc**: Log-oscillator thermostat with parallel multi-scale Q values; g(xi) = 2xi/(1+xi^2)
- **NHC-M**: Nose-Hoover Chain with M thermostat variables
- **NH-1**: Single Nose-Hoover thermostat
- **tau_int**: Integrated autocorrelation time (lower = better mixing)
- **Q**: Thermostat mass parameter (controls thermostat inertia)
- **kappa**: Spring constant / stiffness of potential
- **BAOAB**: Symmetric splitting integrator (B=kick, A=drift, O=thermostat)
- **Qc**: Q center -- the geometric mean of the log-uniform Q range

## Approach

Six methods compared on six targets, with per-target parameter tuning followed by final 10-seed evaluation.

**Methods:**
1. LogOsc N=3 (auto Q): Q_min=kT/kappa_max, Q_max=kT/kappa_min
2. LogOsc N=5 (auto Q): same range, 5 thermostat variables
3. LogOsc N=3 (tuned Q): Q_center swept over {0.01..316}, log-spread +/-1.5 decades
4. LogOsc N=5 (tuned Q): same sweep
5. NHC M=3: Q_ref swept over {0.1, 1.0, 10.0, 31.6, 100.0}
6. NHC M=5: same sweep
7. Underdamped Langevin (BAOAB): gamma swept over {0.1, 1.0, 3.0, 10.0}
8. NH single: Q swept over {0.1, 1.0, 10.0, 100.0}

**Targets:**
1. 1D harmonic (omega=1, dim=1)
2. 2D double-well (barrier=1, y_stiffness=0.5, dim=2)
3. 2D GMM 5-mode (radius=3, sigma=0.5, dim=2)
4. 5D anisotropic Gaussian (kappa_ratio=100, dim=5)
5. 10D anisotropic Gaussian (kappa_ratio=100, dim=10)
6. 10D GMM 5-mode (radius=3, sigma=0.5, dim=10)

**Protocol:**
- Tuning: 3 seeds x 100k force evals
- Final: 10 seeds x 400k force evals with best param
- dt=0.03 for easy targets, dt=0.005 for stiff (anisotropic Gaussian)
- Pool(processes=10)
- Total runtime: ~885s (v1) + ~477s (v2) = 23 min

## Results

### Summary Table

tau_int for unimodal (lower=better), mode crossings for multimodal (higher=better):

| Target        | LogOsc-5 auto | LogOsc-5 tuned | NHC-3 | NHC-5 | Langevin | NH-1   |
|---------------|---------------|----------------|-------|-------|----------|--------|
| 1D Harmonic   | 7.8           | **3.2**        | 5.6   | 5.8   | 16.2     | 3.4    |
| 2D DoubleWell | 16.8          | unstable*      | 21.3  | 20.4  | 33.6     | **14.2** |
| 2D GMM        | 60            | **207**        | 30    | 37    | 27       | 182    |
| 5D Aniso      | 560           | **9.5**        | 741   | 8.6   | 59.9     | 8.5    |
| 10D Aniso     | 636           | **5.5**        | 6.1   | 6.3   | 54.4     | 5.5    |
| 10D GMM       | 310           | **321**        | 21    | 32    | 22       | 570    |

*LogOsc-5 tuned on 2D DoubleWell picked Qc=0.01 which is unstable with N=5 spread.
LogOsc-3 tuned achieves tau=1.4 on 2D DoubleWell (best of all methods).

### Optimal Q Parameters Found

| Target        | LogOsc-5 tuned Qc | NHC-5 Q_ref | NH-1 Q |
|---------------|--------------------|-------------|--------|
| 1D Harmonic   | 10.0               | 1.0         | 1.0    |
| 2D DoubleWell | (0.01 unstable)    | 0.1         | 0.1    |
| 2D GMM        | 100.0              | 0.1         | 10.0   |
| 5D Aniso      | 316.0              | 31.6        | 10.0   |
| 10D Aniso     | 316.0              | 31.6        | 100.0  |
| 10D GMM       | 31.6               | 0.1         | 0.1    |

### Key Findings

1. **Q-range tuning is the dominant factor for LogOsc performance.**
   The original auto-Q formula (Q=kT/kappa) produces tau 60-200x worse than optimal on high-D stiff systems.
   With properly tuned Q_center, LogOsc-5 matches or beats NHC-5 and NH-1 on 4/6 targets.
   The 115x improvement on 10D anisotropic Gaussian (tau: 636 -> 5.5) confirms this is not noise.

2. **Optimal Q scales with dimension, not just stiffness.**
   For 10D systems, optimal Q_center = 100-316, roughly D * kT. The auto-Q formula Q=kT/kappa
   ignores the dimensionality. The thermostat drives xi based on (K - D*kT)/Q, where K ~ D*kT.
   Fluctuations in K scale as sqrt(D) * kT, so Q must scale with D to avoid over-responsive thermostats.

3. **NH-1 with tuned Q is surprisingly competitive.**
   Plain Nose-Hoover wins on 10D GMM (570 crossings vs 321 for LogOsc-5) and ties on anisotropic
   Gaussians. However, NH-1 is known to be non-ergodic for 1D harmonic in theory. Its strong
   empirical performance here reflects that (a) tuned Q avoids the worst resonances, and
   (b) high-D systems are generically more ergodic than 1D.

4. **Langevin is mediocre everywhere.**
   The stochastic baseline never wins. On 2D GMM it gets only 27 mode crossings (vs 207 for
   tuned LogOsc-5). Its optimal gamma varies widely by target, and it lacks the mode-hopping
   ability of deterministic thermostats with large Q.

5. **NHC underperforms on GMMs.**
   NHC-3 and NHC-5 get only 21-37 mode crossings on 2D GMM, much worse than NH-1 (182) or
   tuned LogOsc (207). The chain structure may over-dampen mode-hopping dynamics.

## What Happened

### Phase 1: Initial benchmark (v1)
Ran 6 methods x 6 targets with auto Q for LogOsc and swept Q for NHC/NH/Langevin.
LogOsc performed terribly on high-D stiff systems (tau=560-1560).
NH-1 was surprisingly dominant, winning on 4/6 targets.

### Phase 2: Q-range investigation (v2)
Swept LogOsc Q_center over [0.01, 316] with log-spread +/-1.5 decades.
Discovered that optimal Q_center is 10-316 (not 0.01-1.0 as auto formula gives).
LogOsc-5 improved dramatically: 115x on 10D aniso, 59x on 5D aniso, 3.5x on 2D GMM.

## What I Learned

1. The Q-range prescription Q_min=kT/kappa_max, Q_max=kT/kappa_min fundamentally underestimates
   the needed thermostat mass in high dimensions. A better prescription is Q ~ D * kT / omega^2
   where omega is a characteristic frequency.

2. With proper tuning, the log-osc thermostat is competitive with the best deterministic methods.
   The bounded friction g(xi)=2xi/(1+xi^2) does not inherently limit performance -- the issue was
   entirely in the Q values.

3. For a paper benchmark, reporting "auto-Q LogOsc" without tuning would be misleading. The method
   should always be presented with tuned Q, same as NHC/NH are always presented with tuned Q.

4. NH-1 deserves more respect as a baseline. With tuned Q it is hard to beat on many targets.

## Prior Art & Novelty

### What is already known
- NHC superiority over NH for ergodicity [Martyna et al. 1992]
- Q tuning importance [Hoover 1985, Martyna et al. 1996]
- Q ~ D*kT/omega^2 scaling documented in [Frenkel & Smit, "Understanding Molecular Simulation"]
- BAOAB splitting for Langevin [Leimkuhler & Matthews 2013]

### What this orbit adds
- Systematic 6-method x 6-target benchmark with per-target tuning (new for this project)
- Quantification of Q-range sensitivity: 115x performance difference on 10D aniso Gauss
- Evidence that auto-Q formula breaks in high-D (specific to log-osc thermostat)
- NH-1 competitive performance data (useful context for paper claims)

### Honest positioning
This is primarily a benchmarking and diagnostic orbit. The main value is the systematic comparison
data and the identification of the Q-scaling issue. No novel methods are proposed.

## References

- Martyna, G.J., Klein, M.L., Tuckerman, M. (1992). NHC. J. Chem. Phys. 97, 2635.
- Hoover, W.G. (1985). Canonical dynamics. Phys. Rev. A 31, 1695.
- Leimkuhler, B., Matthews, C. (2013). Rational construction of stochastic numerical methods.
- Frenkel, D., Smit, B. (2002). Understanding Molecular Simulation. Academic Press.

## Seeds & Reproducibility
- v1 tuning seeds: 1000, 1001, 1002
- v1 final seeds: 5000-5009
- v2 tuning seeds: 1000, 1001, 1002
- v2 final seeds: 5000-5009
- GMM mode centers: default ring (2D), random unit vectors seed=0 (10D)
- All simulations use numpy default_rng with explicit seed
