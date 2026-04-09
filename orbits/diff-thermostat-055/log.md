---
strategy: diff-thermostat-055
status: in-progress
eval_version: eval-v1
metric: null
issue: 55
parents:
  - orbit/learn-thermostat-050
---

## Glossary

- **NH**: Nose-Hoover (single thermostat variable)
- **NHC**: Nose-Hoover Chain (chain of thermostat variables)
- **MLP**: Multi-Layer Perceptron (small neural network)
- **ESS**: Effective Sample Size
- **F1**: The Q prescription from orbit #34: Q_min = 1/sqrt(kappa_max), Q_max = 1/sqrt(kappa_min), log-uniform spacing

## Approach

Use differentiable simulation (backpropagation through Nose-Hoover dynamics via PyTorch) to:

1. **E1**: Map the true optimal Q_eff(kappa, D) surface by learning log_Q on a grid of (D, kappa) cells.
2. **E2**: Learn g(xi) as an odd-symmetric neural network to test whether the shape of g matters beyond g'(0).
3. **E3**: Learn N independent Q_k values jointly to test whether log-uniform (1/f) spacing is optimal.

## Results

### E1: Q_eff surface
*(pending)*

### E2: Learned g(xi)
*(pending)*

### E3: Learned Q distribution
*(pending)*

## Prior Art & Novelty

### What is already known
- Differentiable molecular dynamics is well-established: [Schoenholz & Cubuk (2020)](https://arxiv.org/abs/1910.08681) — JAX MD
- Learning thermostat parameters via backprop: parent orbit/learn-thermostat-050 demonstrated Q learning
- Q_eff = Q/g'(0) framework: orbit #54

### What this orbit adds (if anything)
- Systematic mapping of Q_eff(kappa, D) across a grid, not just individual cases
- First test of whether g(xi) shape matters beyond g'(0) using learned neural g
- First test of whether log-uniform Q spacing is truly optimal via gradient-based learning

### Honest positioning
This orbit applies known differentiable simulation techniques to answer specific questions about thermostat parameter sensitivity. The methodology (backprop through NH dynamics) is not novel. The potential contributions are the empirical findings about Q scaling, g-shape degeneracy, and Q-distribution optimality.

## References

- [Schoenholz & Cubuk (2020)](https://arxiv.org/abs/1910.08681) — JAX MD, differentiable molecular dynamics
- orbit/learn-thermostat-050 — parent, demonstrated Q learning on anisotropic harmonic
- orbit #34 — F1 prescription: Q = 1/sqrt(kappa_max)
- orbit #44 — D*kT scaling for Q
- orbit #54 — Q_eff = Q/g'(0) framework
