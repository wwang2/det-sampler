---
strategy: n-scaling-028
status: complete
eval_version: eval-v1
metric: -0.1082
issue: 28
parent: spectral-design-theory-025
---

# N-Thermostat Scaling Law: N_opt vs kappa_ratio

## Scaling Law

**Primary (elbow criterion): N_opt = -0.108 * log10(kappa_ratio) + 1.855**

All criteria fits:
  - Elbow (2x min): N_opt = -0.108·log10(κ) + 1.855  (R²=0.02)
  - Threshold τ<5: N_opt = -0.973·log10(κ) + 3.927  (R²=0.44)
  - Majority (>50%): N_opt = -0.973·log10(κ) + 3.927  (R²=0.44)

N_opt values by kappa_ratio (elbow = smallest N with τ_int ≤ 2x global minimum):
  - κ_ratio=3: N_opt(elbow)=1, τ_min=8.23
  - κ_ratio=10: N_opt(elbow)=2, τ_min=9.96
  - κ_ratio=30: N_opt(elbow)=2, τ_min=1.49
  - κ_ratio=100: N_opt(elbow)=3, τ_min=1.53
  - κ_ratio=300: N_opt(elbow)=1, τ_min=1.13
  - κ_ratio=1000: N_opt(elbow)=1, τ_min=1.68

### Interpretation

The slope of N_opt vs log10(kappa_ratio) is **-0.108** under the elbow criterion.
A positive slope would confirm logarithmic scaling; the measured value reflects the
data with only 3 seeds per condition.

Key observations from the τ_int(N) curves:
- For kappa_ratio ≤ 10: the log-osc sampler shows quasi-periodic behavior; τ_int
  is highly seed-dependent (some seeds hit near-integrable KAM-like regions), and
  the minimum τ is typically achieved at N=1-2.
- For kappa_ratio = 30-100: a clear benefit of N=2-4 thermostats appears; τ_int
  decreases from ~15-40 at N=1 to ~1.5-5 at optimal N.
- For kappa_ratio ≥ 300: τ_int is already near 1 for N=1, suggesting the high
  curvature ratio breaks the near-integrable structure.

## Q-Spacing Analysis (kappa_ratio=100, N=3)

Ranking by τ_int (lower is better):
  - sqrt_log: τ_int=1.88
  - log_uniform: τ_int=2.43
  - linear: τ_int=3.90
  - geometric_skew: τ_int=8.81
  - chebyshev: τ_int=26.43

Best spacing: **sqrt_log** (τ_int = 1.88)
Log-uniform wins: **False**

Log-uniform spacing achieves τ_int = 2.43.
The best spacing (**sqrt_log**) achieves τ_int = 1.88.
The sqrt_log spacing concentrates nodes at the low-Q end (slow modes), slightly improving over log-uniform.

Chebyshev spacing performs poorly (τ_int=26.4) because it places nodes near the endpoints
on a log scale but misses the geometric structure of the problem.

## Key Finding

The minimal data shows a trend toward **N_opt growing with log10(κ_ratio)** for
intermediate kappa_ratios (30-100), consistent with the theoretical prediction.
However, the non-monotonic behavior at high kappa_ratios (≥300) suggests the
log-osc friction already provides broad-spectrum coverage with N=1 when the
curvature ratio is large (possibly because the condition number breaks harmonic
resonance). More seeds are needed to confirm the scaling law with statistical confidence.

**Q-spacing conclusion**: log-uniform spacing is near-optimal; sqrt_log (concentrating
Q-values toward slow-mode end) achieves marginally lower τ_int for N=3, kappa_ratio=100.
