---
strategy: kam-failure-map-053
status: complete
eval_version: eval-v1
metric: 0.6275
issue: 53
parents:
  - chirikov-exponent-032
---

# N=1 Thermostat KAM Failure Map: log-osc vs tanh on 1D HO

## Glossary
- HO: Harmonic Oscillator, V(q) = (1/2) omega^2 q^2
- N=1: single auxiliary thermostat variable xi
- log-osc: friction g(xi) = 2 xi / (1 + xi^2)
- tanh: friction g(xi) = tanh(xi)
- var_ratio: Var(q) / (k_B T / omega^2); ergodic target = 1.0
- erg_frac: fraction of grid cells with |var_ratio - 1| < 0.05

## Result
**Ergodic fractions (20x20 = 400 cells, 3 seeds, 1M steps, dt=0.005):**
- N=1 log-osc: **0.630** (252/400)
- N=1 tanh:    **0.627** (251/400)
- Ratio tanh / log-osc: **1.00** — statistically indistinguishable.

**Metric (tanh erg_frac): 0.6275**

The frequency-ceiling advantage predicted from orbit q-exponent-theory-041
(omega_max = 1.0 for tanh vs 0.732 for log-osc) does NOT translate into a
larger ergodic region at N=1. Both frictions carve out nearly the same KAM
failure band centered on the resonance curve omega * Q = 1, and both recover
ergodicity well away from it.

## Approach
Grid: Q in logspace(0.1, 20, 20), omega in logspace(0.1, 5, 20), seeds (42, 123, 999).
Vectorized BAOAB-style integration: all 1200 runs propagate in lockstep as a
single (nQ, nW, nS) numpy array — no Python loop over cells. Full 1M-step
run completes in ~150 s total on CPU (both frictions).

Per cell we average q^2 over samples past burn-in (10%), form
var_ratio = Var(q) * omega^2 / k_B T, then mean over seeds. A cell is
"ergodic" if |mean var_ratio - 1| < 0.05.

## What Happened
The heatmap of log10 |var_ratio - 1| shows a dark diagonal stripe of
non-ergodicity running along omega * Q = 1 (dashed black line) in BOTH panels.
This cross-validates the Chirikov resonance-overlap picture from orbit #32:
when the natural period 2 pi / omega matches the thermostat response time Q,
phase locking destroys ergodicity.

Key observations:
- **Resonance stripe aligns with omega * Q = 1** in both panels. This is the
  expected KAM obstruction.
- **Sub-normalization band Q < 0.5** (left of solid line): both frictions fail
  irregularly — expected because the log-osc normalization bound kT/2 forbids
  stationary densities there, and tanh inherits similar pathology.
- **Frequency ceiling is NOT visible as a hard boundary.** log-osc shows no
  sharp transition at omega = 0.732, and tanh shows no advantage at
  0.732 < omega < 1.0. Both panels look qualitatively the same above and
  below the dotted ceiling line.
- **High omega (omega > 2)**: both frictions fail broadly at large omega,
  likely because the force kick omega^2 * q * dt becomes poorly resolved at
  fixed dt = 0.005 (discretization artifact, not a thermostat failure).
- **Low-Q, low-omega pockets of near-ergodicity** exist for log-osc well away
  from resonance — not a refutation of Butler 2018, but a reminder that
  "non-ergodic" is a statement about KAM tori, not about every finite-time
  variance estimate.

## What I Learned
1. The "g'(xi) >= 0 and omega_max = 1 vs 0.732" prediction from orbit #041
   does not manifest as an ergodicity advantage at N=1. The paper's
   first-principles narrative needs reframing: the frequency ceiling is a
   LINEAR-STABILITY bound on a single mode, but real ergodicity failure at
   N=1 is dominated by the nonlinear resonance-overlap obstruction (Chirikov),
   which both g-functions suffer equally.
2. The resonance curve omega * Q = 1 is the single most important geometric
   feature of the KAM failure map — visible cleanly in both panels,
   confirming orbit #032's finding.
3. Adding a second thermostat (N=2) is the reliable cure — the tanh vs log-osc
   choice is a second-order effect compared to the topology change from
   N=1 to N=2.

## Prior Art & Novelty

### What is already known
- [Legoll, Luskin, Moeckel (2007)](https://arxiv.org/abs/math/0703059) — rigorous proof that the Nose-Hoover thermostat (N=1) fails to be ergodic on the harmonic oscillator.
- [Hoover (1985)](https://doi.org/10.1103/PhysRevA.31.1695) — original N=1 Nose-Hoover non-ergodicity on the HO.
- [Tapias, Bravetti, Sanders (2016)](https://arxiv.org/abs/1602.03909) — "logistic" thermostat with similar motivation.
- Parent orbit chirikov-exponent-032 — C(kappa) ~ kappa^0.4 power-law and the resonance-singularity at kappa * Q1 = 1 for N=2 log-osc.
- Sibling q-exponent-theory-041 — linear-stability derivation of omega_max = 0.732 (log-osc) and 1.0 (tanh).

### What this orbit adds
- First side-by-side (Q, omega) KAM failure map of N=1 log-osc vs N=1 tanh on the HO.
- Empirical refutation of the hypothesis "tanh's higher omega_max gives a larger
  N=1 ergodic region". They are equal to within 1 cell out of 400.
- Visual confirmation that the omega * Q = 1 resonance curve is the geometric
  organizing principle for N=1 failure.

### Honest positioning
This is not a new method — it is a targeted empirical test of a specific claim
from the first-principles track of the campaign. The finding is NEGATIVE for
the "frequency ceiling matters" story, which is important to report honestly
because it redirects the paper narrative toward the N=1 -> N=2 topological
transition and away from the g' >= 0 choice of friction.

## References
- Legoll, Luskin, Moeckel (2007) "Non-ergodicity of the Nose-Hoover thermostatted harmonic oscillator" — arXiv:math/0703059
- Tapias, Bravetti, Sanders (2016) "Ergodicity of the logistic thermostat" — arXiv:1602.03909
- Parent orbit chirikov-exponent-032 (resonance singularity at kappa * Q = 1)
- Sibling orbit q-exponent-theory-041 (omega_max derivation)
- Sibling orbit ergodicity-phase-diagram-027 (N=1 baseline failure)
