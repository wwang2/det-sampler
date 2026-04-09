---
strategy: nh-cnf-bayesian-posterior
status: in-progress
eval_version: eval-v1
metric: null
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
2. E2: BNN posterior for 1D sinusoidal regression (~100 params)
3. E3: 10D Gaussian mixture (divergence speedup benchmark)

## Results

(pending)
