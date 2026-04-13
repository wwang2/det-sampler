---
strategy: adaptive-annealing-via-nh-entropy-signal
status: complete
eval_version: eval-v1
metric: 5.2945
issue: 68
parents: []
root_rationale: "Fresh exploration: using NH bath entropy production rate as feedback signal for adaptive simulated annealing schedule"
---

## Glossary

- **NH**: Nose-Hoover thermostat dynamics
- **EMA**: Exponential Moving Average
- **KL**: Kullback-Leibler divergence
- **GMM**: Gaussian Mixture Model
- **BAOAB**: Splitting integrator (B=kick, A=drift, O=thermostat)

## Approach

Standard simulated annealing requires a hand-tuned cooling schedule T(t). NH dynamics produce
a natural equilibration signal -- the instantaneous bath heat rate:

    dsigma_bath/dt = tanh(xi) * |p|^2 / (d * T)

When this rate decays to near zero (tracked via EMA with tau=50 steps), the system has
equilibrated at the current temperature, and T can safely be lowered by a factor alpha=0.95.
This turns schedule design into feedback control: the system tells us when it is ready to cool.

We compare three methods on matched force-evaluation budgets:
1. **NH-Adaptive**: EMA of |dsigma_bath/dt| < 0.05 triggers T *= 0.95 cooling
2. **NH-Fixed**: geometric schedule T *= 0.95 every N_stage steps (budget matched to adaptive)
3. **Langevin-Fixed**: overdamped Langevin with same fixed schedule

Targets: 2D 4-Gaussian mixture (sep in {2,4,6}, sigma=0.5), 1D double well V(x)=(x^2-1)^2.
Annealing: T=5.0 -> T=1.0. Production: 8000 steps at T=1.0.
N_traj=30 independent trajectories per (method x target x seed).

## Results

### Key metric: fixed_KL / adaptive_KL = 5.29 at sep=4

The adaptive NH schedule achieves roughly 5x lower KL divergence than fixed geometric
cooling with the same NH integrator, at matched force-evaluation budget.

| Seed | NH-Adaptive KL | NH-Fixed KL | Langevin KL | Ratio (fixed/adaptive) | Wall time |
|------|----------------|-------------|-------------|------------------------|-----------|
| 42   | 0.1404         | 1.1193      | 0.0177      | 7.97                   | 112.5s    |
| 123  | 0.2261         | 1.0634      | 0.0355      | 4.70                   | 115.2s    |
| 7    | 0.2968         | 1.3297      | 0.0784      | 4.48                   | 112.7s    |
| **Mean** | **0.221 +/- 0.064** | **1.171 +/- 0.115** | **0.044 +/- 0.026** | **5.29** | |

### Mode coverage

All methods achieve 100% mode coverage (4/4 modes) across all separations when using
N_traj=30 independent trajectories. This is expected: each trajectory lands near one mode,
and 30 trajectories cover all four modes reliably.

### 1D Double Well KL

| Method | KL (mean +/- std) |
|--------|-------------------|
| NH-Adaptive | 0.0090 +/- 0.0048 |
| NH-Fixed | 0.0156 +/- 0.0044 |
| Langevin-Fixed | 0.0015 +/- 0.0004 |

### Interpretation

The adaptive schedule works because it waits for equilibration before cooling, avoiding
the "quench trap" where the fixed schedule forces cooling before the NH thermostat has
equilibrated. Panel (b) of the figure shows the adaptive schedule (green staircase)
reaching T=1.0 faster than the fixed linear schedule, because early high-T equilibration
is fast and the signal correctly detects this.

However, overdamped Langevin with fixed cooling still outperforms both NH variants.
This is expected: Langevin has intrinsic stochasticity that prevents mode trapping,
while NH dynamics are deterministic and can get stuck on invariant tori. The adaptive
signal helps NH dynamics significantly (5x improvement) but does not close the gap with
stochastic methods.

## What I Learned

1. The bath entropy signal dsigma_bath/dt is a reliable equilibration detector for NH dynamics.
2. Adaptive cooling driven by this signal provides approximately 5x KL improvement over fixed geometric cooling in NH dynamics, because it avoids premature cooling before equilibration.
3. The improvement is largest at intermediate mode separations (sep=4) where the cooling schedule matters most.
4. Despite the improvement, deterministic NH + adaptive annealing does not beat overdamped Langevin + fixed schedule, because the fundamental limitation is ergodicity, not schedule quality.
5. With N_traj=30, mode coverage is saturated for all methods -- the bottleneck is sample quality within each mode, not mode discovery.

## Prior Art & Novelty

### What is already known
- Adaptive simulated annealing schedules based on acceptance rate feedback are well-established (Ingber 1989, "Very Fast Simulated Re-Annealing")
- NH thermostats and their entropy production properties are classical (Nose 1984, Hoover 1985)
- Using thermostat signals for adaptive temperature control has been explored in replica exchange and parallel tempering contexts

### What this orbit adds (if anything)
- Specific formulation using tanh(xi)-based bath entropy rate as the feedback signal for cooling
- Quantitative comparison showing 5x KL improvement for NH dynamics with this adaptive rule vs fixed geometric schedule

### Honest positioning
This is an application of the well-known idea of feedback-controlled annealing to NH thermostat dynamics, using the natural bath entropy signal. The specific signal choice (EMA of |dsigma_bath/dt|) appears to work well but is not fundamentally novel. The main finding is that schedule optimization matters significantly for deterministic thermostats (5x improvement) but cannot overcome the fundamental ergodicity limitation that stochastic methods avoid.

## Compute

[COMPUTE WARNING] Running locally -- Modal unavailable.
Wall time per seed: approximately 113 seconds (3 seeds, parallelized).

## References

- Nose, S. (1984). "A unified formulation of the constant temperature molecular dynamics methods." J. Chem. Phys. 81, 511.
- Hoover, W.G. (1985). "Canonical dynamics: Equilibrium phase-space distributions." Phys. Rev. A 31, 1695.
- Ingber, L. (1989). "Very fast simulated re-annealing." Math. Comput. Model. 12, 967-973.
