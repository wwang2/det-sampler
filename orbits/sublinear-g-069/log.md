---
strategy: sublinear-unbounded-g-arnold
type: experiment
status: complete
eval_version: eval-v1
metric: 1.169
issue: 69
parents:
  - orbit/gprime-ablation-052
---

# sublinear-g-069

## Glossary

- **NH**: Nose-Hoover thermostat
- **tau_int**: Integrated autocorrelation time (of q_d^2, the stiffest-mode observable)
- **kappa**: Condition number (ratio of max to min eigenvalue of the Hessian)
- **Q**: Thermostat mass parameter (controls coupling strength; smaller Q = stronger coupling)
- **g(xi)**: Friction function in generalized NH: dp/dt = -grad_U - g(xi)*p

## Hypothesis

g_new(xi) = xi * log(1+xi^2) / sqrt(1+xi^2) -- unbounded, odd, monotone, sublinear growth -- should outperform tanh on stiff targets (kappa=1000) where the NH frequency ceiling omega_max = 1 limits tanh. The reasoning: tanh is bounded (|g| <= 1), so when the system needs faster thermostat response than the ceiling allows, the dynamics get stuck on KAM tori. An unbounded g should allow arbitrarily fast thermostat response.

## Setup

- **Potential:** d=10 anisotropic Gaussian with log-spaced frequencies omega_i^2 in [1, kappa_max]
- **Friction functions compared:** tanh, log-osc (2xi/(1+xi^2)), linear (xi), sublinear (new)
- **Benchmark grid:** kappa in {10, 100, 1000}, Q in {0.3, 1, 3, 10, 30, 100}
- **Seeds:** 20 per (method, Q, kappa) combination
- **Integration:** BAOAB-style velocity-Verlet, dt=0.005, 200k steps
- **Observable:** q_d^2 (stiffest mode squared position)
- **Metric:** median tau_int across 20 seeds

## Results

**Headline: tau_tanh / tau_sublinear = 1.169 at kappa=100, Q=3.0**

This is at best a marginal 17% improvement at one Q value -- well within the range where noise could explain it, and reversed at other Q values.

### kappa=10 (mild stiffness)

| Method | Q=0.3 | Q=1 | Q=3 | Q=10 | Q=30 | Q=100 |
|--------|-------|-----|-----|------|------|-------|
| tanh | 95.4 | 93.1 | 71.6 | 64.8 | 64.4 | 64.5 |
| log-osc | 78.8 | 96.9 | 82.9 | 66.4 | 64.4 | 64.5 |
| linear | 87.2 | 89.9 | 68.9 | 64.7 | 64.5 | 64.5 |
| sublinear | 95.4 | 94.2 | 73.6 | 65.5 | 64.9 | 64.9 |

All methods nearly identical. The thermostat is essentially irrelevant at large Q (tau converges to ~64.5, the free-dynamics value).

### kappa=100 (moderate stiffness)

| Method | Q=0.3 | Q=1 | Q=3 | Q=10 | Q=30 | Q=100 |
|--------|-------|-----|-----|------|------|-------|
| tanh | 26.8 | **23.8** | 29.0 | 30.4 | 30.8 | 31.0 |
| log-osc | **20.0** | 22.7 | 22.6 | 29.7 | 30.8 | 30.9 |
| linear | 27.5 | 23.5 | 26.0 | 30.8 | 31.0 | 31.0 |
| sublinear | 26.0 | 24.1 | 24.8 | 31.1 | 32.3 | 131.0 |

Sublinear is ~1 unit better than tanh at Q=3 (24.8 vs 29.0), but catastrophically worse at Q=100 (131 vs 31). Log-osc is actually best at small Q.

### kappa=1000 (high stiffness) -- the key test

| Method | Q=0.3 | Q=1 | Q=3 | Q=10 | Q=30 | Q=100 |
|--------|-------|-----|-----|------|------|-------|
| tanh | 379 | 379 | 373 | 381 | 396 | 463 |
| log-osc | 6.3* | 6.3* | 6.3* | 7.1* | 477 | 405 |
| linear | inf | inf | inf | 256 | 275 | 396 |
| sublinear | inf | inf | inf | inf | 318 | ~500 |

*Log-osc tau~6 is almost certainly a non-ergodic artifact: the trajectory is trapped in one basin but q_d^2 decorrelates fast due to the natural oscillation frequency.

**The hypothesis is falsified.** At kappa=1000, sublinear is the worst-performing method:
- It fails to mix (tau=inf) for Q <= 10, while tanh gives finite tau~375 at all Q values
- Even at Q=30, sublinear (318) barely beats tanh (396), but tanh works at ALL Q values
- Linear is also bad but at least starts mixing at Q=10

## What Happened

The Arnold perspective predicted that unbounded g would lift the frequency ceiling. This prediction was wrong because it ignored a crucial mechanism: the exp(-g(xi)*dt/2) momentum rescaling in the BAOAB integrator.

When g(xi) is unbounded:
1. xi can drift to large values (especially at small Q where thermostat oscillations are fast)
2. exp(-g(xi)*dt/2) becomes either very large or very small
3. This causes violent momentum rescaling that either freezes the particle (p -> 0) or explodes it
4. The trajectory gets trapped in a non-ergodic orbit

When g(xi) is bounded (tanh):
1. The rescaling factor stays in [exp(-dt/2), exp(dt/2)] regardless of xi
2. This limits the thermostat's per-step influence, preventing catastrophic overshooting
3. The trajectory can still mix, just slowly (the frequency ceiling)

The frequency ceiling is not a deficiency of tanh -- it is a stability mechanism. The bounded range prevents the thermostat from destroying the trajectory. The real solution to the frequency ceiling is not to change g but to use chains (NHC) or multi-scale Q.

## What I Learned

1. **Unbounded friction is a liability, not an asset.** The frequency ceiling of tanh is actually stabilizing. Removing it causes worse non-ergodicity, not better mixing.

2. **The Arnold twist analysis is incomplete.** It correctly identifies that bounded g limits the twist rate, but it ignores that unlimited twist rate in the discrete integrator causes numerical instability. The continuous-time analysis does not capture the discretization pathology.

3. **Sublinear growth is not slow enough.** g_new grows as xi*log(xi) for large xi, which is still too fast for the exp(-g*dt) rescaling. A truly useful unbounded friction would need to grow slower than any power of xi -- essentially, it would need to be asymptotically bounded, which defeats the purpose.

4. **Log-osc's small tau at kappa=1000 is suspicious.** The constant tau~6.3 across all Q values at kappa=1000 strongly suggests non-ergodic trapping, not genuine fast mixing. The sign reversal in g'(xi) for |xi|>1 likely creates attracting fixed points that trap trajectories.

## Prior Art & Novelty

### What is already known
- The NH frequency ceiling for bounded friction functions is a well-known limitation [Martyna et al., 1996]
- NHC (Nose-Hoover Chains) is the standard remedy for NH non-ergodicity [Martyna et al., 1992]
- The connection between g' > 0 (monotonicity) and ergodicity is discussed in thermostat literature

### What this orbit adds
- Concrete numerical evidence that unbounded-but-sublinear friction is WORSE than bounded tanh
- Identification of the exp(-g*dt) rescaling as the mechanism that makes unbounded g pathological
- The insight that the frequency ceiling is a feature (stability), not a bug

### Honest positioning
This orbit tested a specific hypothesis from the Arnold/twist perspective on thermostat design. The hypothesis was plausible but wrong. The main value is the negative result: unbounded friction functions are a dead end for single-thermostat NH dynamics, and the frequency ceiling problem should be addressed through chains or multi-scale Q rather than through the friction function shape.

## Compute

[COMPUTE WARNING] Running locally -- Modal unavailable. All 20 seeds parallelized via multiprocessing.Pool.
Wall time: approximately 15 minutes for the main benchmark grid.

## References

- Martyna, G. J., Klein, M. L., & Tuckerman, M. (1992). Nose-Hoover chains. J. Chem. Phys.
- Martyna, G. J., Tuckerman, M. E., Tobias, D. J., & Klein, M. L. (1996). Explicit reversible integrators for extended systems dynamics. Mol. Phys.
- Parent orbit: orbit/gprime-ablation-052 (benchmark setup and g' > 0 criterion)
