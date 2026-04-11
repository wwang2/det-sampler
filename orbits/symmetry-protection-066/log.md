---
strategy: symmetry-protection-diagnostic
type: experiment
status: complete
eval_version: eval-v1
metric: -0.028
issue: 66
parents:
  - orbit/triple-identity-064
spawn_reason: extend
---

# symmetry-protection-066: Detailed FT + Kraskov MI on orbit 064 data

## TL;DR

We tested whether `σ_bath − σ_exact` satisfies Crooks' Detailed Fluctuation
Theorem (H1) and whether `σ_bath ⊥ σ_hutch` via Kraskov k-NN MI (H4) on
orbit 064's 200-trajectory NH-tanh ensemble in a 2D double-well.

- **H1 result**: DFT slope = **-0.028** (95% CI **[-0.253, +0.233]**, Δt=5
  window, N=800 pooled increments). Target was +1.000. The CI excludes
  unity and cleanly brackets zero. **H1 is not confirmed.**
- **H4 result**: Kraskov MI(σ_bath−σ_exact ; σ_hutch−σ_exact) at t=25 is
  **-0.041 nats**, null mean -0.0004 ± 0.032, excess **-1.27 σ** (i.e.
  empirical MI sits *below* the shuffle-null mean, inside the noise band).
  Consistent with **statistical independence** — same verdict at all times
  t ∈ {5,10,15,20,25}, all |excess| < 2σ. Linear correlation stays ≤ 0.10.
- **Honest reinterpretation**: the DFT test was applied to the **estimator
  residual** `σ_bath − σ_exact`, not to an entropy-production variable.
  By construction this is the difference of two unbiased estimators of the
  same integral ∫tr(J)dt: it is zero-mean and near-symmetric. A slope of
  ~0 is *tautological* for any such noise and is independent of whether
  σ_bath is symmetry-protected. A correct test requires DFT on σ_bath (or
  σ_exact) *individually*, under a *non-equilibrium protocol* that
  generates nonzero mean entropy production (e.g., orbit 065's temperature
  quench).

## Results (raw)

### H1 — Crooks DFT on windowed Δ(σ_bath − σ_exact)

Primary test: non-overlapping windows of duration Δt starting at t ≥ 5,
pooled across all 200 trajectories. Weighted-LS slope of
log[P(+s)/P(-s)] vs s, 15 symmetric bins centered at 0, 1000-bootstrap
95% CI on slope.

| Δt (window) | N samples | slope    | 95% CI               | intercept |
|-------------|-----------|----------|----------------------|-----------|
| 2.0         | 2000      | +0.121   | [-0.036, +0.271]     | -0.134    |
| **5.0**     | **800**   | **-0.028** | **[-0.253, +0.233]** | +0.157    |
| 10.0        | 400       | -0.223   | [-0.554, +0.208]     | +0.378    |

Headline at Δt=5: **slope = -0.028 (95% CI [-0.25, +0.23])**. CI
excludes the target +1.000, contains 0. No bootstrap sample approaches
unit slope in the primary window.

Pointwise-at-t cross-checks (legacy view, NOT the primary test — these
use N=200 samples per time and are dominated by the residual startup
bias `mean(σ_bath − σ_exact) ≈ +1.0`):

| t    | mean Δ  | std Δ | slope   | 95% CI            |
|------|---------|-------|---------|-------------------|
| 5.0  | +0.917  | 1.41  | -2.360  | [-3.83, +0.31]    |
| 10.0 | +1.097  | 1.39  | -2.006  | [-3.25, -0.02]    |
| 15.0 | +0.965  | 1.10  | -2.160  | [-3.14, +0.07]    |
| 20.0 | +1.175  | 1.50  | -3.367  | [-4.88, -1.54]    |
| 25.0 | +0.968  | 1.30  | -0.072  | [-5.33, +2.38]    |

None approach +1 either; they are even more dominated by the offset of
the unsubtracted estimator residual mean and are not physically
meaningful DFT tests.

**Gaussianity check on Δt=5 pool (N=800):**
- skew = +0.037 (symmetric)
- excess kurtosis = **+5.40** (heavy-tailed)
- Shapiro-Wilk W = 0.929, p ≈ 5e-19 (strongly non-Gaussian)
- R² of weighted fit is meaningless at slope ≈ 0

The distribution is symmetric but leptokurtic, which is exactly what you
see for a noise residual with occasional large excursions. The symmetry
is what forces the DFT slope to ~0.

### H4 — Kraskov k-NN MI(σ_bath − σ_exact ; σ_hutch − σ_exact)

KSG estimator 1, k=5, Chebyshev metric, 500 permutation surrogates,
evaluated on (σ_bath − σ_exact, σ_hutch − σ_exact) at t = 25:

- Empirical MI: **-0.0408 nats**
- Null (permutation): mean = -0.0004, std = 0.0318
- Excess over null: **-1.27 σ** (empirical sits *below* null mean — noise)
- Gaussian entropy upper bound min(H_bath, H_hutch) = 1.68 nats
- Linear correlation at t=25: +0.044

Across times:

| t    | MI      | null mean ± std | excess (σ) | linear corr |
|------|---------|-----------------|------------|-------------|
| 5.0  | -0.0121 | -0.0006 ± 0.0341 | -0.34     | +0.041      |
| 10.0 | -0.0295 | -0.0076 ± 0.0292 | -0.75     | +0.055      |
| 15.0 | +0.0023 | +0.0064 ± 0.0374 | -0.11     | -0.101      |
| 20.0 | -0.0511 | -0.0049 ± 0.0273 | -1.69     | +0.042      |
| 25.0 | -0.0408 | +0.0030 ± 0.0346 | -1.27     | +0.044      |

All |excess| < 2 σ and all empirical MI values are indistinguishable
from (or below) the shuffle null. **Verdict: σ_bath and σ_hutch
(residuals) are statistically independent to within Kraskov sensitivity
at N=200.**

## Reinterpretation

### Why DFT-on-residual gives slope ~ 0 regardless

The Crooks DFT states `P[σ = +s] / P[σ = -s] = exp(β s)` where σ is the
**total entropy production** along a trajectory — a signed,
time-antisymmetric functional that accumulates irreversibility. Its
distribution is generically asymmetric in driven systems, with a biased
weight toward positive s that produces the famous unit slope (in units
where β=1).

`σ_bath − σ_exact` is **not** an entropy production. It is the
difference of two unbiased estimators of the same deterministic
integral `∫ tr(J) dt`: the bath-coupling readout (σ_bath, from the NH
thermostat equations) and the exact divergence accumulator (σ_exact,
computed from the Jacobian). By linearity of expectation its mean is
zero (up to transient startup bias — which we subtracted before
histogramming in the primary test), and in any regime where the two
estimators have roughly exchangeable noise, its distribution is
**symmetric about zero**. A symmetric distribution satisfies
`P(+s) = P(-s)` ⇒ `log[P(+s)/P(-s)] = 0` ⇒ any linear fit has **slope
0**, regardless of tail shape, regardless of whether σ_bath individually
has any symmetry-protected property.

Our measured slope (-0.028, 95% CI [-0.25, +0.23]) and our measured
skew (+0.037) are exactly what this tautology predicts. The test as
posed is vacuous.

### The correct test for the panel's hypothesis

The brainstorm panel's claim was that σ_bath's O(1) bounded variance is
a **consequence of** a fluctuation-theorem symmetry: that σ_bath
approximates the physical entropy production and satisfies Crooks DFT
individually, with variance bounded by linear-response / FDT arguments
around equilibrium.

The right test is:

1. Run a **non-equilibrium protocol** (e.g., orbit 065's sudden
   temperature quench, a time-dependent external force, or a driven
   work cycle) that drives the NH-tanh dynamics out of equilibrium and
   produces a nonzero mean σ_bath per trajectory.
2. Accumulate **σ_bath on individual trajectories** (not differences) —
   this is the physical heat-flow estimator.
3. Histogram σ_bath across trajectories and compute
   `log P(+s) / P(-s)`.
4. DFT predicts slope = β (inverse temperature of the bath).
5. Separately verify that `Var[σ_bath]` stays bounded even as the
   protocol takes the system arbitrarily far from equilibrium — the
   distinctive bounded-variance claim.

Additionally, the 2D double-well of orbit 064 sits in equilibrium
steady state, so both σ_bath and σ_exact have near-zero mean drift; no
nonequilibrium signal is available to fit anyway.

**Orbit 065 (tasaki-quench) is the natural substrate.** It already
generates a non-equilibrium temperature quench and computes σ_bath per
trajectory. A follow-up **orbit 067** can reuse 065's data the same way
066 reused 064's — pure post-processing, cheap.

### Kraskov MI (H4) — clean pass

At -0.041 nats and -1.27 σ excess over a 500-permutation null, with
|excess| < 2σ at every timestep we measured, σ_bath−σ_exact and
σ_hutch−σ_exact are **statistically independent** (at Kraskov
sensitivity for N=200 with k=5). This strengthens the
drop-in-replacement story: σ_hutch and σ_bath are uncorrelated
(linear ≤ 0.10) **and** independent (nonlinear: MI indistinguishable
from zero), so any variance reduction from switching σ_hutch → σ_bath
comes entirely from σ_bath's own bounded variance, **not** from a
correlation-based cancellation with σ_hutch.

Caveat: Kraskov MI with k=5 and N=200 has limited sensitivity to weak
nonlinear coupling. The Gaussian entropy upper bound is 1.68 nats, and
null std is ~0.032, so we can exclude MI above roughly 0.06 nats. This
is enough to rule out structural coupling but not enough to rule out
all conceivable nonlinear dependence.

## Implications for Paper 2

- **Drop the "symmetry-protected" framing for now.** It is not
  supported by the test as posed, and the corrected test (orbit 067)
  hasn't been run. Do not claim H1 in Paper 2.
- **Keep the empirical bounded-variance claim.** That is orbit 064's
  standalone result and is entirely unaffected by this null finding.
- **Keep the independence claim** (conditional on H4). It strengthens
  the drop-in-replacement pitch: σ_bath is not a control variate
  against σ_hutch — it's a primary estimator.
- **Queue orbit 067** (DFT on σ_bath under non-equilibrium protocol,
  reusing 065 data) as the correct follow-up. Cheap post-processing,
  correct physics.

## Methods

- **Data**: orbit 064 `ensemble_sigmas.npz`, 200 NH-tanh trajectories in
  a 2D double-well, 5001 timesteps over t ∈ [0, 25], dt = 0.005. Verified
  std_bath, std_hutch, ratio and correlation match orbit 064's
  `ensemble_summary.json` before use.
- **H1 (DFT) primary test**: non-overlapping windowed increments of
  (σ_bath − σ_exact) starting from t ≥ 5 (post pre-thermalization),
  window sizes Δt ∈ {2, 5, 10}, pooled across all 200 trajectories.
  Mean-centered each pool before histogramming. 15 bins, symmetric
  around 0, bin width from 95th percentile of |Δ|. Weighted LS slope on
  log-ratio pairs with weights ∝ sqrt(c+·c−/(c+ + c−)). Bootstrap 95%
  CI from 1000 resamples. Headline window Δt=5 (N=800).
- **H1 pointwise cross-check**: per-timestep σ_bath − σ_exact at
  t ∈ {5,10,15,20,25}, same binning/WLS. These are dominated by
  unsubtracted startup bias and are reported only as a sanity check.
- **H4 (Kraskov MI)**: KSG estimator 1 (arXiv:cond-mat/0305641), k=5,
  Chebyshev (L∞) metric in joint space via `scipy.spatial.cKDTree`.
  1e-10 Gaussian jitter on each marginal to break ties. Null from 500
  random permutations of y holding x fixed; 100 permutations per time
  for the cross-time table. 100 permutations per time for the
  by-time table.
- **Seed**: `np.random.default_rng(20260411)` shared across bootstrap
  and surrogates.

## Files

- `analysis.py`
- `results/dft_bootstrap.json`
- `results/mi_surrogate.json`
- `figures/fig_symmetry_protection.pdf`
- `figures/fig_symmetry_protection.png`

## Caveats

- **Heavy tails**: Δ(σ_bath − σ_exact) has excess kurtosis +5.4 and
  fails Shapiro-Wilk (p ≈ 5e-19). The symmetry of the distribution
  (skew +0.037) is what pins slope ≈ 0; the tails are non-Gaussian but
  symmetric. Bootstrap CIs absorb this.
- **Bootstrap sample size**: 1000 resamples, 800 pooled samples at
  Δt=5. CI widths are ~0.25 in slope units — fine for excluding unity
  but not for resolving small nonzero slopes.
- **Pooled independence**: pooling 4 non-overlapping Δt=5 windows per
  trajectory assumes weak autocorrelation between windows; at 2D
  double-well equilibrium this is the conservative assumption because
  hop events are rare at β used in orbit 064. A block-bootstrap would
  only widen CIs further.
- **Kraskov k=5, N=200**: can exclude MI above ~0.06 nats; cannot rule
  out weaker nonlinear coupling.
- **Equilibrium substrate**: the 2D double-well in orbit 064 is in
  equilibrium steady state, so σ_bath and σ_exact both have near-zero
  long-time mean drift. The correct DFT test requires a non-equilibrium
  protocol (orbit 065 provides this; orbit 067 is the queued test).
- **H1 test was on the wrong variable** — see `Reinterpretation` above.
  The measured null result is consistent with the tautological
  expectation and does not falsify (or support) symmetry protection.
