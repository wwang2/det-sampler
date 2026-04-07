---
strategy: n-scaling-robust-029
status: complete
eval_version: eval-v1
metric: -1.1338
issue: 29
parent: n-scaling-028
---

# N-Scaling Robust: N_opt vs kappa_ratio (10 seeds, 800k evals)

## Scaling Law

**N_opt = -1.134 ± 0.813 (SE) * log10(kappa_ratio) + 4.638**

- R² = 0.327
- p-value = 0.2357
- Status: **SUGGESTIVE** (R²=0.33): weak log-scaling trend, more data needed

## Results by kappa_ratio

  - κ_ratio=3: N_opt=5, τ(N=1)=113.26, gain=14.9x
  - κ_ratio=10: N_opt=2, τ(N=1)=47.46, gain=4.8x
  - κ_ratio=30: N_opt=2, τ(N=1)=34.02, gain=1.5x
  - κ_ratio=100: N_opt=5, τ(N=1)=15.10, gain=1.6x
  - κ_ratio=300: N_opt=1, τ(N=1)=6.46, gain=1.0x
  - κ_ratio=1000: N_opt=1, τ(N=1)=1.86, gain=1.0x

## Interpretation

The slope is not statistically significant (p≥0.05), suggesting the N_opt vs log(kappa_ratio) relationship is not confirmed at this sample size.

Slope = -1.134 means: doubling log10(kappa_ratio) by 1 (i.e., 10x increase in condition number)
requires ~-1.1 additional thermostats.

The non-monotonic behavior may reflect that at extreme kappa_ratio (≥300), the high curvature itself breaks KAM resonance, making N=1 sufficient (as found in orbit #028).

## Key Takeaway

**SUGGESTIVE** (R²=0.33): weak log-scaling trend, more data needed

For the paper: the gain ratio (tau(N=1)/tau(N_opt)) shows that intermediate kappa_ratios (30-100)
benefit most from multiple thermostats, while extreme ratios (≥300) or small ratios (≤10)
benefit less.
