---
strategy: diff-thermostat-055
status: complete
eval_version: eval-v1
metric: 0.628
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

Use differentiable simulation (backpropagation through Nose-Hoover dynamics via PyTorch) to answer three questions about thermostat parameter sensitivity:

1. **E1**: Map the true optimal Q_eff(kappa, D) surface by learning log_Q on a 4x3 grid.
2. **E2**: Learn g(xi) as an odd-symmetric neural network to test whether friction shape matters beyond g'(0).
3. **E3**: Learn N independent Q_k values jointly to test whether log-uniform (1/f) spacing is optimal.

All experiments use anisotropic harmonic potentials U(x) = 0.5 * sum_i kappa_i * x_i^2 with geometric kappa ladder kappa_i = kappa_max^(i/(D-1)). Loss function: 0.5 * variance_loss + 0.5 * |rho_1|.

Seeds: 42 (E1), [42, 123, 256, 789, 1024] (E2, E3).

## Results

### E1: Q_eff(kappa, D) surface

Grid: D in {2, 5, 10, 20}, kappa_max in {10, 100, 1000}. 12 cells total.

| D \ kappa_max | 10    | 100   | 1000  |
|---------------|-------|-------|-------|
| 2             | 1.494 | 0.621 | 0.789 |
| 5             | 0.712 | 1.134 | 1.950 |
| 10            | 1.178 | 0.597 | 2.123 |
| 20            | 0.918 | 0.520 | 2.159 |

**Q_learned stays O(1) across all cells (range 0.52 to 2.16).**

Comparison with analytical predictions:
- **F1 prescription (Q = 1/sqrt(kappa_max))**: predicts 0.316, 0.100, 0.032 for kappa=10/100/1000. Learned Q is 3-70x larger. F1 dramatically underestimates the optimal Q.
- **D*kT scaling (orbit #44)**: predicts 2, 5, 10, 20 for D=2/5/10/20. Learned Q is much smaller for D >= 5. D*kT overestimates.
- **Neither prescription matches.** The learned Q has weak dependence on D and kappa, staying near O(1). The slight increase at kappa=1000 (to ~2) may reflect the difficulty of coupling to very stiff modes.

Caveat: with only 2000 integration steps per epoch (n_steps reduced for runtime), the loss landscape is noisy. Losses are in the 0.5-1.2 range, suggesting the optimizer finds a broad, shallow minimum near Q ~ 1.

### E2: Learned g(xi) shape

D=10, kappa_max=100. 5 independent seeds training a 1-hidden-layer MLP (16 units) to parameterize g(xi).

| Seed | Q_learned | g'(0) | Final loss |
|------|-----------|-------|------------|
| 42   | 0.666     | 0.978 | 0.722      |
| 123  | 0.366     | 1.073 | 0.853      |
| 256  | 0.414     | 1.049 | 0.778      |
| 789  | 0.394     | 1.079 | 0.665      |
| 1024 | 0.660     | 0.979 | 0.734      |

**All 5 seeds converge to g(xi) approximately equal to xi** (the standard linear NH friction). g'(0) ranges from 0.978 to 1.079, tightly clustered near 1.0. The MLP corrections are negligible.

Interpretation:
- The shape of g(xi) does not matter much beyond g'(0) for this problem. The optimizer has freedom to learn tanh, sigmoid-like, or more exotic shapes, but converges to near-identity.
- This is consistent with orbit #54's Q_eff = Q/g'(0) framework: since g'(0) ~ 1, Q_eff ~ Q, and the effective thermostat mass is just Q itself.
- Two clusters emerge in Q: seeds 42/1024 converge to Q ~ 0.66, while seeds 123/256/789 converge to Q ~ 0.39. This suggests multiple local minima in the Q direction, but g'(0) is consistently ~1.0 in both basins.

### E3: Learned Q distribution for parallel thermostats

D=10, kappa_max=100. N in {3, 5} parallel thermostats, each with independent Q_k, 5 seeds each.

**Key finding: the Q_k ratios are preserved exactly from initialization.**

For N=3, initialized at [0.1, 1.0, 10.0] (log-uniform):
- All 5 seeds produce Qs with ratio ~1:10:100, just the overall scale changes.
- Example seed 42: [0.068, 0.678, 6.78] -- ratios 1:10:100 preserved.

For N=5, same pattern -- perfect log-uniform spacing maintained.

**This is a structural degeneracy, not evidence for log-uniform optimality.** All N thermostats couple identically to the physical system (all see the same v^2 - D*kT driving force), so their gradients w.r.t. log_Q_k are proportional. The optimizer can only shift the overall scale, not the relative spacing. The effective thermostat mass is 1/sum(1/Q_k), and only that aggregate matters.

Loss comparison: N=1 (0.628), N=3 (0.71 +/- 0.08), N=5 (0.71 +/- 0.07). **More parallel thermostats do not improve the loss.** This confirms that identical-coupling parallel thermostats are redundant -- for multi-scale coupling, the thermostats need to couple to *different* degrees of freedom (as in NHC chains or per-mode thermostats).

## What Worked
- Differentiable simulation via backprop through NH dynamics works reliably for learning Q.
- The MLP-parameterized g(xi) experiment cleanly demonstrates that friction shape is a secondary concern.
- Reducing n_steps to 1000-2000 gave noisy but convergent optimization.

## What Did Not Work
- Parallel identical-coupling thermostats are structurally degenerate -- cannot learn relative Q spacing.
- The absolute loss values (0.5-1.2) remain high, suggesting that single NH thermostats are inherently limited for multi-scale problems regardless of Q tuning.

## Key Takeaways

1. **Optimal Q is O(1)** for the combined variance+autocorrelation loss, with weak dependence on D and kappa. Neither 1/sqrt(kappa) nor D*kT captures this behavior.
2. **g(xi) shape is secondary** -- all learned g converge to near-identity with g'(0) ~ 1.
3. **Parallel identical-coupling thermostats are degenerate** -- cannot test 1/f optimality this way. A proper test requires per-mode or chain coupling.

## Prior Art & Novelty

### What is already known
- Differentiable molecular dynamics: [Schoenholz & Cubuk (2020)](https://arxiv.org/abs/1910.08681)
- Learning thermostat parameters via backprop: parent orbit/learn-thermostat-050
- Q_eff = Q/g'(0) framework: orbit #54

### What this orbit adds
- First systematic mapping of gradient-optimized Q_eff across a (D, kappa) grid, showing Q stays O(1)
- Demonstration that learned g(xi) stays near identity (g'(0) ~ 1), supporting Q_eff ~ Q
- Identification of the structural degeneracy in parallel identical-coupling thermostats

### Honest positioning
The methodology (backprop through NH) is established. The findings about Q ~ O(1) and g-shape insensitivity are empirical observations on the specific loss function used (variance + |rho_1|). Different loss functions (e.g., pure KL divergence, ESS) might yield different optimal Q values. The E3 degeneracy result is more about experimental design (identical coupling is degenerate) than about physics.

## References

- [Schoenholz & Cubuk (2020)](https://arxiv.org/abs/1910.08681) -- JAX MD, differentiable MD
- orbit/learn-thermostat-050 -- parent, demonstrated Q learning
- orbit #34 -- F1 prescription: Q = 1/sqrt(kappa_max)
- orbit #44 -- D*kT scaling for Q
- orbit #54 -- Q_eff = Q/g'(0) framework
