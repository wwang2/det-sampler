# Figures Needed for Paper Draft

Paper: "Bounded-Friction Thermostats for Canonical Sampling: 1/f Noise,
Generalized Chains, and Multi-Scale Ergodicity"

---

## Figure 1: Friction Function Gallery (Section 2.2)

**Filename:** `fig_friction_gallery.pdf`

**What it shows:** Three-panel comparison of friction functions g(xi):
- Panel A: g(xi) vs xi for NH (linear), log-osc, tanh (Tapias), and arctan.
  Annotate: "bounded", mark saturation at +/- 1/Q for log-osc.
- Panel B: Thermostat potentials V(xi) corresponding to each g(xi).
- Panel C: Thermostat marginal distributions rho(xi) ~ exp(-V(xi)/kT):
  Gaussian (NH), Cauchy (log-osc), sech^2 (tanh), Cauchy (arctan).

**Data source:** Analytic — no simulation required. All functions are closed-form.

**Key message:** Bounded friction functions saturate at finite values, while
NH grows without limit. This single design choice determines ergodicity behavior.

---

## Figure 2: Lyapunov Exponent Comparison (Section 2.2)

**Filename:** `fig_lyapunov_comparison.pdf`

**What it shows:** Maximal Lyapunov exponent lambda vs thermostat mass Q for
NH, log-osc, tanh, and arctan on the 1D harmonic oscillator.
- x-axis: Q from 0.1 to 2.0 (log scale)
- y-axis: lambda (linear scale)
- Four curves, one per friction function
- Annotate: "10-300x improvement" range for bounded vs NH at small Q
- Shaded region: Q in [0.3, 0.8] where log-osc achieves ergodicity > 0.85

**Data source:** `orbits/unified-theory-007/lyapunov_results.txt` and
`orbits/unified-theory-007/lyapunov_long_results.txt`

**Key message:** Bounded friction produces dramatically stronger chaos at
small Q, explaining the ergodicity improvement.

---

## Figure 3: Phase Portraits — KAM Tori vs Ergodic Orbits (Section 2.2)

**Filename:** `fig_phase_portraits.pdf`

**What it shows:** Side-by-side (q, p) phase portraits for 1D HO at Q=0.3:
- Left panel: NH — visible nested elliptical tori, sparse coverage
- Right panel: Log-Osc — space-filling, irregular orbit
- Both panels use same trajectory length and initial conditions

**Data source:** `orbits/unified-theory-007/figures/` (phase portrait figures
already generated in that orbit's run)

**Key message:** Visual confirmation that bounded friction breaks KAM confinement.
The most intuitive figure in the paper.

---

## Figure 4: Ergodicity Score vs Q (Section 4.3)

**Filename:** `fig_ergodicity_vs_Q.pdf`

**What it shows:** Ergodicity score (composite: KS + variance + coverage) vs Q
for log-osc and NH on the 1D harmonic oscillator.
- x-axis: Q from 0.1 to 5.0 (log scale preferred)
- y-axis: ergodicity score 0 to 1
- Two main curves: NH (red), Log-Osc (blue)
- Horizontal dashed line at 0.85 ("ergodic threshold")
- Annotate peak: log-osc 0.982 at Q=0.5; mark NH plateau at Q<=0.2

**Data source:** `orbits/unified-theory-007/` (coverage and ergodicity scan
described in iteration 3 of log.md); `orbits/log-osc-001/log.md` Q-scan table.

**Key message:** Log-osc exceeds the ergodic threshold across a wide Q range
(0.3–0.8), while NH collapses as soon as KAM tori form.

---

## Figure 5: Spectral Analysis — 1/f Noise (Section 3.2)

**Filename:** `fig_spectral_1f.pdf`

**What it shows:** Two-panel figure:
- Panel A: Power spectral density (PSD) of friction signal g_total(t) on
  log-log axes for N=1, 3, 5, 7 log-spaced thermostats. Show PSD curves
  with fitted power-law slopes alpha. Highlight N=3 curve (alpha=0.98).
- Panel B: Spectral exponent alpha and GMM KL vs N (number of thermostats).
  Dual y-axis: left axis alpha (0 to 2), right axis GMM KL (log scale).
  Mark the "1/f sweet spot" at N=3 with a vertical dashed line.

**Data source:** `orbits/spectral-1f-016/` (spectral exponent vs N results from
orbit log — alpha=0.98 for N=3, overshoot to alpha~2 for N>=5)

**Key message:** N=3 log-spaced thermostats produce near-perfect 1/f noise
(alpha=0.98) via the Dutta-Horn mechanism, coinciding with the best GMM
performance. The transition is sharp.

---

## Figure 6: Champion Sampler Benchmark (Section 4.1–4.2)

**Filename:** `fig_champion_benchmark.pdf`

**What it shows:** Three-panel comparison of sampling quality across benchmarks:
- Panel A: 1D HO — histogram of sampled q vs exact Gaussian; compare NH,
  NHC(M=3), Log-Osc, MultiScaleNHCTail
- Panel B: 2D DW — KL divergence convergence curves (KL vs force evaluations)
  for NH, NHC(M=3), Log-Osc, MultiScaleNHCTail
- Panel C: 2D GMM — scatter plot of samples in 2D for NHC(M=3) (stuck in 1-2
  modes) vs MultiScaleNHCTail (all 5 modes covered). Overlay mode centers.

**Data source:**
- HO and DW: `orbits/log-osc-001/log.md`, `orbits/unified-theory-007/`
- GMM: `orbits/multiscale-chain-009/log.md` (champion sampler results)

**Key message:** The champion sampler provides broad improvement across all
three benchmark types. The GMM panel is the most visually striking—the
contrast between stuck (NHC) and exploring (champion) is immediate.

---

## Figure 7: Generalized Chain Performance (Section 2.3)

**Filename:** `fig_generalized_chains.pdf`

**What it shows:** Bar chart of HO KL divergence for:
- NH (single), NHC(M=3), Log-Osc single, LogOscChain(M=3), ArctanChain(M=3)
- Log scale on y-axis; annotate "4.8x improvement" for ArctanChain vs NHC
- Error bars if multiple seeds available

**Data source:** `orbits/general-chains-015/log.md` (K_eff formula results;
ArctanChain(M=3) KL=0.0018 vs NHC KL=0.0087)

**Key message:** The K_eff generalized chain formula yields a family of
thermostats that uniformly outperform NHC, with ArctanChain achieving the
best single-mode accuracy reported in this work.

---

## Summary Table

| # | Filename                   | Section    | Data source                  | Priority |
|---|----------------------------|------------|------------------------------|----------|
| 1 | fig_friction_gallery.pdf   | 2.2        | Analytic                     | High     |
| 2 | fig_lyapunov_comparison.pdf| 2.2        | unified-theory-007           | High     |
| 3 | fig_phase_portraits.pdf    | 2.2        | unified-theory-007/figures/  | High     |
| 4 | fig_ergodicity_vs_Q.pdf    | 4.3        | unified-theory-007, log-osc-001 | High  |
| 5 | fig_spectral_1f.pdf        | 3.2        | spectral-1f-016              | High     |
| 6 | fig_champion_benchmark.pdf | 4.1–4.2    | log-osc-001, multiscale-009  | High     |
| 7 | fig_generalized_chains.pdf | 2.3        | general-chains-015           | Medium   |

All 7 figures are needed for the primary narrative. Figures 1–6 are essential;
Figure 7 can be moved to supplementary if page limits require.

**Note on data availability:** Orbits spectral-1f-016, general-chains-015, and
multiscale-chain-009 are not present in this worktree. Their key numbers are
recorded in the paper_draft.md from the orbit log descriptions provided in the
mission context. The actual figure-generation scripts and raw data live in
those orbit directories in the main research tree.
