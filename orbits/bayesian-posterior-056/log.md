---
strategy: nh-cnf-bayesian-posterior
status: complete
eval_version: eval-v1
metric: 0.1708
issue: 56
parents:
  - orbit/diff-thermostat-055
---

# bayesian-posterior-056: NH thermostat as CNF for Bayesian posteriors

## Glossary

- **NH**: Nose-Hoover thermostat
- **NHC**: Nose-Hoover Chain thermostat
- **CNF**: Continuous Normalizing Flow
- **SGLD**: Stochastic Gradient Langevin Dynamics
- **BNN**: Bayesian Neural Network
- **KL**: Kullback-Leibler divergence
- **ESS**: Effective Sample Size
- **FFJORD**: Free-Form Jacobian of Reversible Dynamics (Grathwohl et al. 2019)
- **RK4**: Fourth-order Runge-Kutta integrator
- **BDL**: Bayesian Deep Learning
- **MAP**: Maximum A Posteriori estimate

## Approach

The NH thermostat ODE defines a deterministic flow on (q, p, xi) whose invariant
measure marginalizes to exp(-V(q)/kT) in the position variables. This makes it a
natural candidate for a CNF for Bayesian posterior sampling.

The key computational advantage: the divergence of the NH flow is analytically
known as div(f) = -d * g(xi), where d is the dimension and g is the thermostat
coupling function (e.g., tanh). This eliminates the need for Hutchinson trace
estimators used in FFJORD-style CNFs.

Three experiments:
1. E1: 2D banana distribution (toy posterior)
2. E2: BNN posterior for 1D sinusoidal regression (~481 params)
3. E3: 10D-500D Gaussian mixture (divergence speedup benchmark)

## Results

### E1: 2D Banana Distribution
| Sampler | KL divergence | Wall time |
|---------|--------------|-----------|
| NH-tanh multi-Q | 0.171 | 24.0s |
| SGLD | 0.066 | 5.0s |

NH-tanh with multi-scale Q=[0.1, 1, 10] underperforms SGLD by about 2.5x on
KL divergence. The NH sampler covers the banana shape but does not match the
density accurately -- likely due to the deterministic nature of the dynamics
getting partially trapped on certain trajectories. This is the well-known
ergodicity limitation of NH-type thermostats on curved distributions.

### E2: BNN Posterior (1D sinusoidal regression)
| Sampler | 95% CI Coverage | N samples |
|---------|----------------|-----------|
| NH-tanh multi-Q | 0.980 | 1000 |
| SGLD | 0.975 | 1000 |
| Deep Ensemble | 0.555 | 5 |

Strong result: both NH-tanh and SGLD achieve near-perfect calibration on the
BNN posterior, while the deep ensemble is severely underconfident (only 55.5%
coverage at the 95% level). This suggests the deterministic thermostat is a
viable tool for BDL -- it produces well-calibrated uncertainty estimates
competitive with SGLD but without stochastic noise.

### E3: Analytical Divergence Speedup (Refined)

**Why the NH divergence is analytically known.** The Nose-Hoover thermostat
dynamics on the extended phase space (q, p, xi) have a specific structure: the
momentum equation dp/dt = -grad_V(q) - g(xi)*p has a Jacobian with respect to p
that is block-diagonal: d(dp_i/dt)/dp_j = -g(xi) * delta_{ij}. The trace of
this Jacobian — which is the divergence of the flow in the momentum sector — is
therefore exactly -d * g(xi). For g = tanh, this gives div(f) = -d * tanh(xi).

This is not an approximation. It is the exact divergence, following from the
Liouville equation and the structure of the NH invariant measure. Standard CNFs
(e.g., FFJORD with a neural network vector field f_theta) have dense, unstructured
Jacobians where the trace must be estimated stochastically via the Hutchinson
estimator: div(f) ~ (1/n) sum_i v_i^T J v_i, with random probe vectors v_i.

**Quantitative results (10 repeats, 200 steps each, d in {2..500}):**

| d   | Analytical (ms) | Hutch(1) (ms) | Hutch(5) (ms) | Brute (ms) | H1/Ana | H5/Ana | BF/Ana |
|-----|-----------------|---------------|---------------|------------|--------|--------|--------|
| 2   | 0.22 +/- 0.00  | 0.27 +/- 0.01 | 0.29 +/- 0.00 | 0.28 +/- 0.01 | 1.24x | 1.30x | 1.25x |
| 5   | 0.22 +/- 0.01  | 0.29 +/- 0.02 | 0.29 +/- 0.01 | 0.29 +/- 0.01 | 1.31x | 1.28x | 1.31x |
| 10  | 0.22 +/- 0.01  | 0.27 +/- 0.00 | 0.28 +/- 0.00 | 0.31 +/- 0.01 | 1.24x | 1.29x | 1.42x |
| 20  | 0.22 +/- 0.01  | 0.28 +/- 0.01 | 0.29 +/- 0.01 | 0.34 +/- 0.02 | 1.27x | 1.34x | 1.54x |
| 50  | 0.23 +/- 0.01  | 0.28 +/- 0.01 | 0.29 +/- 0.01 | 0.43 +/- 0.02 | 1.19x | 1.27x | 1.88x |
| 100 | 0.22 +/- 0.00  | 0.28 +/- 0.01 | 0.30 +/- 0.00 | 0.55 +/- 0.02 | 1.29x | 1.34x | 2.48x |
| 200 | 0.22 +/- 0.01  | 0.29 +/- 0.01 | 0.32 +/- 0.00 | (too slow)     | 1.30x | 1.41x | N/A   |
| 500 | 0.24 +/- 0.01  | 0.30 +/- 0.00 | 0.35 +/- 0.01 | (too slow)     | 1.25x | 1.48x | N/A   |

**Variance and error analysis (100 draws per dimension):**

| d   | True div  | Hutch(1) var | Hutch(5) var | Hutch(1) rel err | Hutch(5) rel err |
|-----|-----------|-------------|-------------|------------------|------------------|
| 2   | -1.07     | 0.89        | 0.27        | 0.674            | 0.380            |
| 10  | -5.71     | 5.72        | 1.22        | 0.327            | 0.163            |
| 50  | -39.73    | 67.77       | 12.13       | 0.162            | 0.066            |
| 100 | 32.78     | 21.49       | 4.88        | 0.112            | 0.052            |
| 500 | 208.58    | 150.26      | 38.81       | 0.045            | 0.023            |

**Honest discussion of the wall-clock advantage.** The 1.2-1.5x wall-clock
speedup of analytical over Hutchinson(1) is modest. This is because the NH
system has a trivially diagonal Jacobian in the momentum block, so a single
Hutchinson vector already gives a reasonable trace estimate. The per-vector cost
in the Hutchinson estimator is dominated by a few vector operations (O(d)), not
by an expensive JVP through a neural network.

**The real advantage is zero variance.** The analytical divergence introduces no
noise into the log-density estimate. This matters in three scenarios:

1. **Training stability.** In CNF loss optimization, the log-density gradient
   depends on the divergence integral. Stochastic divergence estimates add noise
   to the gradient, requiring smaller learning rates or more samples.

2. **Long-trajectory accumulation.** Over T integration steps, the log-density
   change is the integral of div(f). Hutchinson errors compound: the variance of
   the accumulated divergence grows as T * Var(single step). With analytical
   divergence, the accumulated integral is exact regardless of trajectory length.

3. **Small batch sizes.** When computing log-likelihoods of individual samples
   (e.g., for importance weighting or evaluation), there is no batch averaging
   to smooth out Hutchinson noise. The analytical divergence gives exact
   per-sample log-densities.

At d=2, a single Hutchinson draw has relative error of 67% — the estimate is
essentially useless without averaging. At d=500 the relative error drops to 4.5%
per draw, but the variance is still 150. To achieve 1% relative accuracy with
Hutchinson at d=500, one would need approximately n = Var/(0.01 * |true_div|)^2
= 150/(0.01 * 209)^2 ~ 34 vectors per step. The analytical formula achieves
this for free.

**Figures:**
- `figures/e3_schematic.png` — Conceptual comparison: Standard CNF (Hutchinson) vs NH CNF (analytical)
- `figures/e3_analysis.png` — 4-panel quantitative analysis: cost scaling, variance, error, throughput

### Verification
Analytical divergence (-d*tanh(xi)) matches autograd Jacobian trace to 7.7e-8
absolute error. The formula is correct.

## What Worked
- MAP initialization for the BNN sampler (avoids starting in bad region)
- Velocity Verlet splitting with gradient clipping for high-d stability
- Multi-scale Q for parallel thermostats

## What Did Not Work
- NH-tanh on the banana distribution: KL=0.17 vs SGLD's 0.07
- Raw Euler integration in 481D: diverges immediately
- Single-Q NH on banana: same KL as multi-Q (the problem is ergodicity, not Q)

## What I Learned
- The analytical divergence wall-clock advantage is real but modest (~1.2-1.5x
  over Hutch(1)) because the NH Jacobian is trivially diagonal. The advantage
  grows with n_vec: Hutch(5) costs 1.3-1.5x analytical.
- The decisive advantage is zero variance: at d=2, a single Hutchinson draw
  has 67% relative error; even at d=500, 4.5% per draw. To achieve 1% accuracy
  at d=500, Hutchinson needs ~34 vectors per step. Analytical is free.
- This zero-variance property matters most for: (a) training stability in CNF
  losses, (b) long trajectory log-density accumulation where errors compound,
  (c) per-sample evaluations without batch averaging.
- NH-tanh is competitive with SGLD for BNN posterior calibration, suggesting
  deterministic thermostats are viable for BDL.
- The banana distribution exposes the fundamental ergodicity limitation of
  NH-type dynamics on curved, non-Gaussian posteriors.
