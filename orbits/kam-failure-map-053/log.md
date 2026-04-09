---
strategy: kam-failure-map-053
status: complete
eval_version: eval-v1
metric: 0.4300
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
- NH: Nose-Hoover
- var_ratio: Var(q) / (k_B T / omega^2); ergodic target = 1.0
- erg_frac (mean-first): fraction of grid cells where |mean_seeds(var_ratio) - 1| < 0.05
- erg_frac (majority-vote): fraction of cells where at least 6 of 10 seeds individually satisfy |var_ratio - 1| < 0.05
- SE: standard error of the mean

## Result

**Ten-seed sweep on a 20x20 (Q, omega) grid, 1M steps, dt=0.005, seeds
{42, 123, 999, 777, 2024, 31415, 271828, 61803, 1414, 1618}.**

Headline (both aggregations):

| Metric                              | log-osc | tanh   | log-osc - tanh |
|-------------------------------------|---------|--------|----------------|
| Mean-first erg_frac                 | 0.6800  | 0.7050 | -0.0250        |
| **Majority-vote erg_frac (>=6/10)** | 0.4725  | 0.4300 | **+0.0425**    |
| Restricted window 0.732 < w <= 1    |         |        |                |
|   - mean-first                      | 0.700   | 0.700  |  0.000         |
|   - majority-vote                   | 0.475   | 0.275  | +0.200         |
| Restricted window w <= 2            |         |        |                |
|   - mean-first                      | 0.590   | 0.613  | -0.023         |
|   - majority-vote                   | 0.330   | 0.267  | +0.063         |

**Per-seed ergodic fraction** (fraction of 400 cells per seed):

| seed    | log-osc | tanh  |
|---------|---------|-------|
| 42      | 0.5875  | 0.5375|
| 123     | 0.5475  | 0.5150|
| 999     | 0.5350  | 0.5075|
| 777     | 0.5500  | 0.5075|
| 2024    | 0.5400  | 0.4975|
| 31415   | 0.5225  | 0.5225|
| 271828  | 0.5375  | 0.5300|
| 61803   | 0.5200  | 0.5200|
| 1414    | 0.5500  | 0.4975|
| 1618    | 0.5500  | 0.5125|
| **mean**| **0.5540** | **0.5148** |
| **SE**  |  0.0056 |  0.0040 |

**Metric reported in frontmatter: tanh majority-vote erg_frac = 0.4300**
(more defensible than the mean-first aggregate, which is sensitive to a
single near-boundary seed flipping a cell in or out).

**Result:** With 10 seeds, log-osc and tanh have comparable N=1 ergodic
fractions on the 1D HO (log-osc majority-vote = 0.4725, tanh = 0.4300). The
direction **favors log-osc** but the magnitude is small (|Delta| < 0.06 on
the full grid; about 0.20 in the theoretically-critical window
0.732 < omega <= 1). The per-seed difference in mean-first fractions is
0.039 with combined SE ~= 0.007 (roughly 5 SE), so the sign is robust, but
the effect size is modest. The frequency-ceiling advantage predicted by
orbit #41 (omega_max = 1.0 for tanh vs 0.732 for log-osc) does NOT produce
a larger N=1 ergodic region for tanh; if anything, tanh is slightly worse
when majority-vote is used. **Do not use this orbit alone to redirect the
paper narrative** — the magnitude is small and other problem dimensions
(N-dependence, higher D) are untested here.

## Approach
Grid: Q in logspace(0.1, 20, 20), omega in logspace(0.1, 5, 20),
10 seeds. Integrator: **symmetric Trotter splitting (G-B-O-A-O-B-G)** for
the Nose-Hoover N=1 equations. This is a symmetric second-order operator
splitting for the Nose-Hoover equations (G = xi half-kick, B = momentum
force half-kick, O = momentum friction half-kick from exp(-g*dt/2),
A = position drift), not the BAOAB splitting used in Langevin dynamics
(there is no Ornstein-Uhlenbeck step here — the system is deterministic).
Half-step ordering is symmetric so the scheme is time-reversal symmetric
and second-order accurate.

All 4000 runs (20 Q x 20 omega x 10 seeds) propagate in lockstep as a
single (nQ, nW, nS) numpy array — no Python loop over cells. Full 1M-step
run completes in ~500 s total on CPU (both frictions). Per cell we average
q^2 over samples past burn-in (10%), form var_ratio = Var(q) * omega^2 / k_B T,
then apply the two aggregations above.

## What Happened
The heatmap of log10 |var_ratio - 1| in figure `kam_surface.png` shows a
dark diagonal stripe of non-ergodicity running along omega * Q = 1 (dashed
black line) in both panels (a) and (b). This cross-validates the Chirikov
resonance-overlap picture from orbit #32: when the natural period 2 pi / omega
matches the thermostat response time Q, phase locking destroys ergodicity.

Panel (c) shows log10 of the std-across-seeds for log-osc. High seed
disagreement (yellow) concentrates near the resonance curve and at low
Q -- precisely where the sampler is most sensitive to initial conditions,
as expected. Far from the resonance curve the std drops to ~1e-2, so cells
classified as ergodic there are stable against seed choice.

Key observations:
- **Resonance stripe aligns with omega * Q = 1** in both panels. This is
  the expected KAM obstruction.
- **Critical window 0.732 < omega <= 1** is where orbit #41 predicted tanh
  should have an advantage (tanh is linearly stable there, log-osc is not).
  On majority-vote, log-osc actually scores HIGHER in this window
  (0.475 vs 0.275). The linear-stability argument does not translate to
  nonlinear ergodicity.
- **Sub-normalization band Q < 0.5** (left of solid line): both frictions
  fail irregularly — expected because the log-osc normalization bound kT/2
  forbids stationary densities there.
- **High omega (omega > 2)**: both frictions fail broadly, likely a
  discretization artifact at fixed dt = 0.005. The w<=2 restricted window
  removes this effect and the log-osc vs tanh ranking is preserved (log-osc
  still slightly ahead on majority-vote).
- **Mean-first vs majority-vote flip sign.** Mean-first says tanh is ahead
  by 0.025; majority-vote says log-osc is ahead by 0.042. The mean-first
  aggregation is biased toward cells where seeds straddle the threshold
  but average near 1.0 — these are not robustly ergodic. Majority-vote is
  the more defensible metric because it requires per-seed ergodicity.

## What I Learned
1. The "g'(xi) >= 0 and omega_max = 1 (tanh) vs 0.732 (log-osc)" prediction
   from orbit #41 does not translate into a tanh ergodicity advantage at
   N=1. On the most defensible metric (majority-vote over 10 seeds),
   log-osc actually has a small but reproducible advantage. The frequency
   ceiling is a LINEAR-STABILITY bound on a single mode, but real ergodicity
   failure at N=1 is dominated by the nonlinear resonance-overlap obstruction
   (Chirikov), which both g-functions suffer — and log-osc's tighter
   coupling near resonance appears to slightly broaden the ergodic region.
2. **Aggregation choice matters.** With only 3 seeds (the original run),
   we saw the two methods as "indistinguishable" (0.630 vs 0.628). With 10
   seeds and majority-vote, a small but consistent log-osc advantage
   appears. The 3-seed conclusion was not wrong — it was under-powered.
3. The resonance curve omega * Q = 1 is the single most important
   geometric feature of the KAM failure map, confirming orbit #32.
4. Adding a second thermostat (N=2) remains the reliable cure -- the
   tanh vs log-osc choice is a second-order effect compared to the
   topology change from N=1 to N=2.

## Prior Art & Novelty

### What is already known
- [Legoll, Luskin, Moeckel (2007)](https://arxiv.org/abs/math/0703059) —
  rigorous proof that the Nose-Hoover thermostat (N=1) fails to be ergodic
  on the harmonic oscillator.
- [Hoover (1985)](https://doi.org/10.1103/PhysRevA.31.1695) — original
  N=1 Nose-Hoover non-ergodicity on the HO.
- [Tapias, Bravetti, Sanders (2016)](https://arxiv.org/abs/1602.03909) —
  "logistic"/tanh-style thermostat with similar motivation.
- Parent orbit chirikov-exponent-032 — C(kappa) ~ kappa^0.4 power law and
  the resonance singularity at kappa * Q1 = 1 for N=2 log-osc.
- Sibling q-exponent-theory-041 — linear-stability derivation of
  omega_max = 0.732 (log-osc) and 1.0 (tanh).

### What this orbit adds
- First side-by-side (Q, omega) KAM failure map of N=1 log-osc vs N=1 tanh
  on the HO, run at 10 seeds with both mean-first and majority-vote
  aggregations.
- Empirical evidence that the "tanh's higher omega_max gives a larger
  N=1 ergodic region" hypothesis is WRONG in the direction predicted.
  On majority-vote over 10 seeds, log-osc is slightly ahead
  (0.473 vs 0.430), and in the critical window 0.732 < omega <= 1 log-osc
  wins more clearly (0.475 vs 0.275).
- Visual confirmation that the omega * Q = 1 resonance curve is the
  geometric organizing principle for N=1 failure, plus a seed-disagreement
  map showing the std is concentrated along that same curve.

### Honest positioning
This is not a new method — it is a targeted empirical test of a specific
claim from the first-principles track of the campaign. The finding is
NEGATIVE for the "tanh's frequency ceiling buys larger N=1 ergodicity"
story. The effect size is small (|Delta| < 0.06 on the full grid) so the
orbit does not by itself justify redirecting the paper narrative away
from tanh; it does say that the g' >= 0 linear-stability argument is not
the right lens for N=1 ergodicity on the HO, and that the N=1 -> N=2
topological transition remains the more important axis.

## References
- Legoll, Luskin, Moeckel (2007) "Non-ergodicity of the Nose-Hoover thermostatted harmonic oscillator" — arXiv:math/0703059
- Hoover (1985) "Canonical dynamics: Equilibrium phase-space distributions" — Phys. Rev. A 31, 1695
- Tapias, Bravetti, Sanders (2016) "Ergodicity of one-dimensional oscillators with a signum thermostat" — arXiv:1602.03909
- Parent orbit chirikov-exponent-032 (resonance singularity at kappa * Q = 1)
- Sibling orbit q-exponent-theory-041 (omega_max derivation)
- Sibling orbit ergodicity-phase-diagram-027 (N=1 baseline failure)
