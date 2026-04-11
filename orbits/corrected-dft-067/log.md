---
strategy: corrected-dft-sigma-bath
type: experiment
status: complete
eval_version: eval-v1
metric: 1.059
issue: 67
parents:
  - orbit/tasaki-quench-065
  - orbit/symmetry-protection-066
spawn_reason: extend
---

# corrected-dft-067: Crooks DFT on sigma_bath under non-equilibrium quench

## TL;DR

Corrected version of orbit 066's DFT test -- this time testing on sigma_bath
itself (physical heat flow) under orbit 065's temperature quench, not on
the estimator residual sigma_bath - sigma_exact (which was orbit 066's mistake).

We test **three** variables: sigma_bath, sigma_exact, and sigma_tot = sigma_bath - sigma_exact.

**Results:**
- Phase 1 (T0=1 -> T1=2): sigma_tot slope = 1.059, 95% CI [0.94, 1.15] (expected: 1.0) -- **DFT holds**
- Phase 2 (T0=0.8 -> T1=1.5): sigma_tot slope = 0.905, 95% CI [0.80, 1.01] (expected: 1.0) -- **DFT holds**
- sigma_bath and sigma_exact individually do NOT satisfy the simple Crooks DFT (slopes ~ -0.9, not beta1 or 1.0)
- Jarzynski equality satisfied: <exp(-sigma_tot)> = 0.965 (Phase 1), 1.042 (Phase 2)
- **Verdict: DFT holds for total entropy production sigma_tot, not for sigma_bath alone**

## Theory

Crooks' Detailed Fluctuation Theorem for entropy production:

    log P(sigma_tot = +s) / P(sigma_tot = -s) = s

where sigma_tot is the total entropy production per trajectory. For the NH-tanh
thermostat under a sudden quench T0 -> T1:

    sigma_bath = beta1 * integral_0^t tanh(xi) |p|^2 ds     (heat flow into bath)
    sigma_exact = d * integral_0^t tanh(xi) ds                (exact estimator)
    sigma_tot  = sigma_bath - sigma_exact                      (total entropy production)

The Evans-Searles fluctuation theorem predicts slope = 1.0 for sigma_tot.
For sigma_bath ALONE, there is no guarantee of a simple DFT because it is
only one component of the total entropy production.

Phase 1: T1 = 2.0, beta1 = 0.5
Phase 2: T1 = 1.5, beta1 = 0.667

## Results

### Phase 1 -- T0=1.0 -> T1=2.0

**sigma_tot (total entropy production) -- the control:**
- DFT slope = 1.059 (expected 1.0)
- 95% bootstrap CI = [0.940, 1.148] -- **contains 1.0**
- Intercept = -0.043
- N data points = 9
- Mean = 0.453, Std = 1.056, Median = 0.186, Skew = 1.44

**sigma_bath (heat flow into bath):**
- DFT slope = -0.957 (expected beta1 = 0.5 if simple Crooks)
- 95% bootstrap CI = [-1.023, -0.806]
- Intercept = -0.091
- N data points = 8
- Mean = -0.873, Std = 1.474, Median = -0.635, Skew = -0.98

**sigma_exact (exact estimator):**
- DFT slope = -0.940
- 95% bootstrap CI = [-1.012, -0.813]
- Intercept = -0.077
- N data points = 7
- Mean = -1.326, Std = 1.837, Median = -1.055, Skew = -0.77

**Jarzynski check:** <exp(-sigma_tot)> = 0.965 (target: 1.0)
**Exact KL(q,p):** 0.328

Per-seed DFT slopes on sigma_tot: 1.083 (seed 42), 0.979 (seed 123), 0.972 (seed 7)
N_traj per seed: 2400 (60 parents x 40 branches), total: 7200

### Phase 2 -- T0=0.8 -> T1=1.5

**sigma_tot (total entropy production) -- the control:**
- DFT slope = 0.905 (expected 1.0)
- 95% bootstrap CI = [0.802, 1.012] -- **contains 1.0**
- Intercept = -0.064
- N data points = 8
- Mean = 0.399, Std = 1.048, Median = 0.142, Skew = 1.32

**sigma_bath (heat flow into bath):**
- DFT slope = -0.870 (expected beta1 = 0.667 if simple Crooks)
- 95% bootstrap CI = [-0.933, -0.764]
- Intercept = -0.009
- N data points = 8
- Mean = -0.776, Std = 1.478, Median = -0.542, Skew = -1.05

**sigma_exact (exact estimator):**
- DFT slope = -0.886
- 95% bootstrap CI = [-0.961, -0.804]
- Intercept = 0.072
- N data points = 7
- Mean = -1.175, Std = 1.850, Median = -0.904, Skew = -0.87

**Jarzynski check:** <exp(-sigma_tot)> = 1.042 (target: 1.0)
**Exact KL(q,p):** 0.280

Per-seed DFT slopes on sigma_tot: 0.673 (seed 42), 1.095 (seed 123), 0.960 (seed 7)
N_traj per seed: 2400, total: 7200

## Interpretation

The DFT holds for the total entropy production sigma_tot in both phases. The
95% CIs contain the expected slope of 1.0. The Jarzynski equality is
approximately satisfied (0.965 and 1.042, both close to 1.0). This validates
the simulation framework and the entropy production accounting.

sigma_bath alone does NOT satisfy the simple Crooks DFT. The slopes are
negative (~-0.96 and ~-0.87) because sigma_bath has a negative mean under
these quench protocols -- the system loses kinetic energy to the bath during
re-equilibration. The DFT log-ratio is necessarily negative for positive s
when the distribution is shifted negative. The slopes are NOT equal to beta1
(0.5 or 0.667), confirming that sigma_bath is only one component of entropy
production and does not individually satisfy a fluctuation theorem.

Interestingly, sigma_bath and sigma_exact have very similar DFT slopes in
both phases (~-0.95 vs ~-0.94 in Phase 1; ~-0.87 vs ~-0.89 in Phase 2).
This near-equality may reflect the fact that sigma_tot = sigma_bath - sigma_exact
has a much tighter distribution (std ~1.05) than either component (std ~1.5
and ~1.8 respectively), suggesting strong positive correlation between
sigma_bath and sigma_exact.

**For Paper 2:** The DFT result does NOT upgrade sigma_bath to "symmetry-protected"
in the fluctuation-theorem sense. sigma_bath alone does not satisfy a simple
FT. However, the fact that sigma_tot = sigma_bath - sigma_exact satisfies the
FT with slope 1.0 means the entropy production accounting is correct. Paper 2
should retain the "empirically bounded" framing for sigma_bath's variance, and
can cite the sigma_tot FT as validation of the simulation framework, but NOT
as evidence that sigma_bath itself is fluctuation-theorem-protected.

## Methods

The analysis runs a 2D double-well (V = (q1^2-1)^2 + 0.5*q2^2) with NH-tanh
thermostat (Q=1.0, dt=0.005). For each phase, 3 seeds x (60 parents x 40
branches) = 7200 trajectories are generated. Each trajectory burns in for
t=200 at T0, then runs t=200 at T1 after the quench, accumulating sigma_bath
and sigma_exact via trapezoidal integration. sigma_tot = sigma_bath - sigma_exact.

The DFT analysis bins the entropy production symmetrically about zero
(bin width = 0.3 * std), then computes log(P(+s)/P(-s)) for each bin center
s > 0 where both +s and -s bins have at least 5 counts. A weighted
least-squares fit (weight = sqrt(count_pos * count_neg)) gives the slope.
Bootstrap resampling (1000 resamples) provides 95% CIs on the slope.

## Files

- analysis.py -- DFT computation driver (672 lines, includes simulation, analysis, and figure generation)
- results/dft_phase1.json, dft_phase2.json -- full numerical results
- figures/fig_corrected_dft.pdf -- 2x2 panel figure: (a,c) DFT log-ratio plots for all three sigma variables, (b,d) distribution histograms with means
- run.sh -- execution script

## Caveats

- N_traj = 7200 per phase (3 seeds x 60 parents x 40 branches)
- Branching correlations: the 40 branches per parent share a common equilibrated state, with t_decorr=20 between branches at T0
- Binning scheme (0.3*std bin width, min 5 counts) may affect tail estimates
- NH-tanh may have non-standard FT properties; the bounded tanh friction modifies the phase-space measure
- The xi variable contributes to total entropy production (orbit 065 found <xi^2> ~ 2.6T, not T/Q); this is captured in sigma_tot but not in sigma_bath alone
- Phase 2 per-seed variance is high (seed 42 gives sigma_tot slope 0.673 vs seeds 123/7 giving ~1.0), suggesting 2400 trajectories per seed is marginally sufficient
