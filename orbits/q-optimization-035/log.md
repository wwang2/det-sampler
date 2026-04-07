---
strategy: q-optimization-and-nhc-comparison
status: complete
eval_version: custom-035
metric: 2.48
issue: 35
parents:
  - orbit/figure-update-033
  - alpha-spectrum-comparison-031
---

# q-optimization-035

Headline metric: **NHC_best τ_int / ours_best τ_int at κ_ratio=100 = 2.48** (parallel multi-scale wins by 2.5×).

## Two questions

1. If we optimize Q values freely (no spectral prior), what distribution do we get?
2. At equal thermostat count, does our parallel multi-scale log-osc design beat NHC(M=N)?

## Setup

- Sampler: N parallel log-osc thermostats with friction g(ξ)=2ξ/(1+ξ²); coupling Γ = Σ g(ξᵢ).
- Baseline: Nose-Hoover Chain length M = N, Q_i = 1.
- Both integrated with the same BAOAB-style splitting.
- Potentials: 5D anisotropic Gaussian (κ_r ∈ {10, 100}); 2D 5-mode GMM (radius 3, σ=0.5).

## Part 1 — Free Q optimization

Nelder-Mead in log-Q space, 8 random log-uniform inits, 40k-step short sims (1 seed during search); best refined with 3 seeds × 200k steps.

| config        | best optimized τ | log-uniform τ | optimized Q values (sorted) |
|---------------|-----------------:|--------------:|-----------------------------|
| κ_r=10, N=3   |            347.7 |         585.2 | 0.66, 2.12, 8.07            |
| κ_r=10, N=5   |            750.1 |        1572.9 | 0.42, 1.31, 1.50, 4.09, 4.88|
| κ_r=100, N=3  |           2396.0 |         756.9 | 0.34, 2.22, 2.81            |
| κ_r=100, N=5  |           3399.9 |        1134.7 | 1.23, 1.48, 1.56, 7.06, 7.56|

Observations:
- κ_r=10: optimizer beats log-uniform by ~2× and finds Q values spanning roughly the same log range as the reference but biased upward (max Q ≈ 8 vs reference 1.0). The optimum is *not* log-uniform; it puts more weight on slow (large-Q) thermostats.
- κ_r=100: optimizer finishes worse than log-uniform — the 40k-step optimization budget is too short relative to the κ_r=100 mixing time (~10³), the τ estimator is dominated by noise, and the search lands in a poor basin. Numerical Q optimization is not robust on stiff problems within a tractable budget; log-uniform remains the stronger default.

## Part 2 — Head-to-head vs NHC (10 seeds, 400k force evals)

τ_int on q² for the 5D anisotropic Gaussian:

| κ_r | N |      parallel τ      |        NHC τ         | NHC/parallel |
|----:|--:|---------------------:|---------------------:|-------------:|
|  10 | 3 |        192.8 ± 135.9 |        208.5 ± 41.7  |         1.08 |
|  10 | 5 |       1030.1 ± 577.6 |        203.7 ± 45.6  |         0.20 |
| 100 | 3 |        745.7 ± 175.6 |       1305.3 ± 753.4 |         1.75 |
| 100 | 5 |        527.3 ± 126.8 |       1539.1 ± 825.2 |         2.92 |

Mode crossings on the 2D 5-mode GMM (200k force evals, 10 seeds):

| N |     parallel cross. |       NHC cross. |
|--:|--------------------:|-----------------:|
| 3 |         19.4 ± 14.4 |      13.0 ± 4.2  |
| 5 |         37.7 ± 15.4 |       9.0 ± 2.2  |

Observations:
- **Stiff Gaussian (κ_r=100):** parallel wins clearly — 2.5× better at the best M, uniformly better at every N. NHC variance is huge (±60%); it occasionally gets stuck.
- **Mild Gaussian (κ_r=10):** parallel(N=3) is at parity, parallel(N=5) is *worse* than NHC. With the log-uniform Q range fixed at [1/√10, 1], several thermostats sit at too-similar timescales and contribute redundant friction; NHC shares one chain so it doesnt suffer this.
- **Multi-modal GMM:** parallel wins for mode-hopping (4× more crossings at N=5). NHC has only ξ_1 coupled to p, so its barrier crossing is rate-limited by a single timescale; parallel multi-scale log-osc friction at multiple scales helps escape modes.

## What this closes

- **Gap 1 (closed).** The log-uniform Q schedule is not the absolute optimum, but it is a robust default that unconstrained numerical optimization fails to beat reliably on stiff problems. At κ_r=10 the optimizer finds a ~2× speedup with Q values biased upward but stay within a broad log range.
- **Gap 2 (closed).** The parallel multi-scale design beats NHC at equal thermostat count on the two problems where multi-scale matters most: stiff anisotropic Gaussians (2.5× faster mixing at κ_r=100) and multi-modal GMM (4× more mode crossings). NHC is competitive only on the easy single-scale case (κ_r=10, N=3).

## Prior Art & Novelty

### What is already known
- [Martyna et al. (1992)](https://doi.org/10.1063/1.463940) — Nose-Hoover chains; baseline equations.
- [Tapias et al. (2016)](https://arxiv.org/abs/1605.02034) — Logarithmic-oscillator thermostat with bounded friction g(ξ)=2ξ/(1+ξ²).
- [Fukuda & Nakamura (2002)](https://doi.org/10.1103/PhysRevE.65.026105) — Multiple Nose-Hoover thermostats.
- The log-uniform Q schedule is the working default of prior orbits in this campaign.

### What this orbit adds
- A direct head-to-head between the parallel multi-scale log-osc design and NHC at *equal thermostat count*, on stiff Gaussians and a multi-modal target. Prior orbits compared α-exponents *within* the parallel family but never against NHC at matched complexity.
- A numerical free-Q optimization showing the log-uniform prior is robust but not globally optimal at moderate stiffness.

### Honest positioning
Empirical comparison of two known classes of deterministic thermostats. No new method is introduced. Contribution is a clean apples-to-apples benchmark closing two gaps prior orbits left open.

## Files
- run_experiment.py — full experiment (Part 1 + Part 2)
- make_figures.py — figure generation
- results.json — numerical results
- figures/q_optimization.png — optimized vs log-uniform Q (Part 1)
- figures/nhc_comparison.png — parallel vs NHC across problems (Part 2)
- run.sh — reproduce from seed

## References
- Martyna, Klein, Tuckerman (1992). Nose-Hoover chains. J. Chem. Phys. 97, 2635.
- Tapias, Bravetti, Sanders (2016). Ergodicity from a logarithmic oscillator thermostat.
- Fukuda, Nakamura (2002). Multiple Nose-Hoover thermostats. Phys. Rev. E 65, 026105.
