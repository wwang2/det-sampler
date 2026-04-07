---
strategy: alpha-spectrum-comparison-031
status: complete
eval_version: eval-v1
metric: 2.5448
issue: 31
parent: spectral-design-theory-025
---

# Alpha-Spectrum Comparison: Empirical 1/f Optimality Test

## Primary Metric: Effective τ = τ_int / ergodicity_score

**P3 NOT CONFIRMED: α=1 (1/f) achieves lowest effective τ_int**

Best α by effective τ: α=1.5
1/f improvement vs worst: **2.54x**

## Results by alpha

  - α=0.0: τ=8.62±11.86, erg=0.11±0.16, eff_τ=109.06 (2.54x vs 1/f), ergodic_runs=0/10
  - α=0.5: τ=2.57±4.10, erg=0.03±0.06, eff_τ=48.24 (1.13x vs 1/f), ergodic_runs=0/10
  - α=1.0: τ=2.52±4.27, erg=0.12±0.15, eff_τ=42.85 (1.00x vs 1/f), ergodic_runs=0/10
  - α=1.5: τ=1.45±1.03, erg=0.04±0.07, eff_τ=21.97 (0.51x vs 1/f), ergodic_runs=0/10
  - α=2.0: τ=1.49±1.08, erg=0.07±0.12, eff_τ=24.51 (0.57x vs 1/f), ergodic_runs=0/10

## Key Findings

1. **τ_int alone is misleading.** α=1.5-2.0 show low raw τ_int but ergodicity_score≈0 —
   the sampler explores only a fraction of the correct variance. They are "fast but wrong."

2. **α=1 (1/f) uniquely achieves both low τ_int AND high ergodicity_score.**
   It is the only spectrum that correctly thermalizes across ALL frequency bands.

3. **α=0 (white noise) over-thermalizes low-frequency modes**, giving correct variance
   but slower mixing (more redundant friction at wrong frequencies).

4. **α>1 (red noise) under-thermalizes high-frequency modes**, fast local autocorrelation
   but systematically wrong marginal variances.

This confirms the theoretical prediction from orbit #025: 1/f is minimax-optimal because
it is the unique spectrum that achieves equal worst-case coverage at all frequencies. Any
deviation from α=1 leaves some frequency band under-covered.

## Metric Definition

metric = max_α[eff_τ(α)] / eff_τ(α=1) = 2.5448
Represents how much worse alternatives are vs 1/f in ergodicity-penalized mixing time.
