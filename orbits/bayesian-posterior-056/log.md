---
strategy: nh-cnf-bayesian-posterior
status: in-progress
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

### E3: Analytical Divergence Speedup
At d=10, the analytical divergence is 1.4x faster than Hutchinson(1), 2.7x
faster than Hutchinson(5), and 2.4x faster than brute-force Jacobian trace.

Dimension scaling: the Hutch(1)/Analytical ratio stays roughly constant at
1.3-1.5x from d=10 to d=500, while the brute-force ratio grows from 2.4x to
>13x. The modest Hutch(1) advantage is because the NH system has a trivially
diagonal Jacobian in the momentum block (d(dp/dt)/dp = -g(xi)*I), so one
Hutchinson vector suffices for a good estimate. The real advantage of analytical
divergence is zero variance (exact) vs the stochastic Hutchinson estimator.

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
- The analytical divergence advantage is real but modest (~1.4x) for the NH
  system because the Jacobian structure is trivially diagonal. The advantage
  would be larger for CNFs with non-diagonal dynamics.
- NH-tanh is competitive with SGLD for BNN posterior calibration, suggesting
  deterministic thermostats are viable for BDL.
- The banana distribution exposes the fundamental ergodicity limitation of
  NH-type dynamics on curved, non-Gaussian posteriors.
