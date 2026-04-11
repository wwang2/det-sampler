---
strategy: nh-cnf-paper
type: paper
status: complete
eval_version: eval-v1
metric: 0.376
issue: 58
parents:
  - nh-cnf-deep-057
  - bayesian-posterior-056
  - triple-identity-064
  - tasaki-quench-065
  - symmetry-protection-066
  - corrected-dft-067
---

# nh-cnf-paper-058: Paper 2 — Bounded-Variance CNF Divergence Estimation via Thermostat Dynamics

## Glossary

- **NH**: Nose-Hoover thermostat (plain, M=1 chain)
- **NHC**: Nose-Hoover Chain (M >= 2)
- **CNF**: Continuous Normalizing Flow
- **FFJORD**: Free-Form Jacobian of Reversible Dynamics (Grathwohl et al. 2019)
- **JVP**: Jacobian-Vector Product
- **DFT**: Detailed Fluctuation Theorem (Evans-Searles)
- **MI**: Mutual Information (Kraskov k-NN estimator)
- **KL**: Kullback-Leibler divergence
- **RK4**: Classical 4th-order Runge-Kutta integrator
- **DW**: Double-well potential V(q) = (q1^2-1)^2 + 0.5 q2^2

## Summary

Complete rewrite of paper.tex per the audit.md blueprint (commit 67b04d2). The paper pivots from "NH-CNF as a sampler with exact divergence" to "bounded-variance divergence estimation for CNFs via thermostat dynamics."

### What changed

**Title:** "Bounded-Variance Divergence Estimation for Continuous Normalizing Flows via Thermostat Dynamics"

**Line count:** 1280 -> 928 (27% reduction, net of cuts and additions)

**Sections rewritten from scratch (60%):**
- Abstract: new 200-word abstract focused on bounded variance, 10x ratio, Jarzynski, honest DFT falsification
- S1 Introduction: FFJORD variance problem -> sigma_bath as O(1) bounded replacement -> contributions
- S3 Thermostat estimators: formal definition of sigma_hutch and sigma_bath, unbiasedness via equipartition, variance analysis with orbit 064 numbers, independence (MI = -0.041 nats)
- S4 Non-equilibrium verification: temperature quench protocol, Phase 1 (Jar=0.965), Phase 2 (10x variance, Jar=1.042), Phase 3 (1D harmonic failure), Evans-Searles DFT (sigma_tot slope=1.059, sigma_bath slope=-0.957)
- S5 Variance diagnostic: 200-trajectory ensemble figure, mechanism explanation
- S6 Discussion + related work: Liu et al. 2025 as competitor, Ceriotti 2010, honest limitations

**Sections kept with relabeling (25%):**
- S2.1 CNF background: tightened
- S2.2 NH background: relabeled
- S2.3 Divergence identity: Theorem 1 downgraded to Lemma, cited Tuckerman Ch4 + Evans-Searles
- Phase-space figure (fig3_phase): relabeled caption to reference sigma_bath

**Sections cut (15%):**
- Algorithm 1 (NH-CNF sampling): artifact of sampler framing
- S3.3 Multi-scale Q: irrelevant to estimator pitch
- S3.4 Diffusion connection + Table 1: analogy, not load-bearing
- S4.1 E1 sample quality (6.1x ED): untuned ULA baseline
- S4.4 BNN posterior (98% calibration): over-cautious (1.00), orthogonal
- S4.5 Log-likelihood (KDE-on-samples): not marginal log p(q)
- S4.6 Dimension scaling (d>20 trapping): sampler question, out of scope
- App B Q_eff universality: belongs to Paper 1
- App D Frozen-momentum protocol: the tautology enabler

**Appendices kept:**
- App A: Full divergence derivation (shrunk)
- App B (was E): V_theta bias=False (compressed)

### Figures

4 figures, all with data provenance:
1. **fig1_concept.png** — Two computational paths (FFJORD vs NH-augmented). Relabeled caption.
2. **fig_ensemble_variance.png** — 3-panel ensemble diagnostic from orbit 064 (mean+std, sqrt(t) fits, scatter). Hero figure.
3. **fig3_phase.png** — Phase-space trajectory on 2D double-well. Relabeled caption to reference sigma_bath.
4. **fig_tasaki_quench.png** — 3-panel quench verification from orbit 065 (convergence, Jarzynski bars, harmonic failure).

9 figures deleted: fig2_density, fig4_schematic, fig5_loglik, fig6_scaling, fig7_bnn, fig_training_stability, fig4_divergence, fig_variance_scaling_new, fig_walltime.

### Key numbers (all from orbit data, no invented statistics)

| Quantity | Value | Source |
|----------|-------|--------|
| std_bath / std_hutch at t=25 | 0.376 | orbit 064, N=200 |
| std(sigma_bath - sigma_exact) | 1.303 | orbit 064 |
| std(sigma_hutch - sigma_exact) | 3.470 | orbit 064 |
| corr(bath, hutch) | 0.044 | orbit 064 |
| MI(bath, hutch) | -0.041 nats | orbit 066 |
| sqrt(t) R^2 for hutch | 0.988 | orbit 064 |
| sqrt(t) R^2 for bath | -3.22 | orbit 064 |
| Jarzynski Phase 1 (T0=1->T1=2) | 0.965 | orbit 065 |
| Jarzynski Phase 2 (T0=0.8->T1=1.5) | 1.042 | orbit 065 |
| std(Sigma) / std(Hutch) Phase 2 | 1.05 / 10.7 = 10x | orbit 065 |
| DFT slope sigma_tot | 1.059 CI [0.94, 1.15] | orbit 067 |
| DFT slope sigma_bath | -0.957 | orbit 067 |
| Jarzynski Phase 3 (1D harmonic) | 0.916 (8% violation) | orbit 065 |

### Honest disclosures

1. sigma_bath is NOT fluctuation-theorem-protected (orbit 067 falsified this). Bounded variance is from equipartition, not symmetry.
2. Plain NH is non-ergodic on harmonic potentials (Phase 3, 8% Jarzynski violation).
3. End-to-end FFJORD training with sigma_bath is NOT demonstrated -- explicitly stated as future work.
4. Ensemble size is N=200 (orbit 064) / N=2400 (orbit 065) -- finite sample sizes stated.
5. The naive control-variate combination does not work (r=0.044) -- sigma_bath is a replacement, not a supplement.

### TODO items

None -- all numbers filled from orbit data, all figures present.

## Prior Art & Novelty

### What is already known
- NH divergence = -d*g(xi) is classical (Tuckerman 2010, Evans & Searles 2002)
- Hutchinson trace estimator for CNFs (Grathwohl et al. 2019)
- Hutch++ variance reduction for FFJORD (Liu et al. 2025, arXiv:2502.18808)
- Evans-Searles fluctuation theorem applies to total entropy production
- NH non-ergodicity on harmonic oscillators (Legoll et al. 2009, Martyna et al. 1992)

### What this orbit adds
- Identification of sigma_bath as a bounded-variance estimator of the SAME integral that Hutchinson estimates
- Empirical characterization: O(1) saturation (R^2 = -3.22 vs sqrt(t)) vs O(sqrt(t)) random walk (R^2 = 0.988)
- Statistical independence measurement (MI = -0.041 nats, r = 0.044) ruling out control-variate combination
- Non-equilibrium verification via Jarzynski (3.5-4.2% accuracy)
- 10x variance ratio under temperature quench
- Honest falsification: DFT does NOT hold for sigma_bath alone (slope -0.957)

### Honest positioning
This paper applies known thermodynamic identities (equipartition, Jarzynski equality, Evans-Searles DFT) to the specific problem of divergence estimation in continuous normalizing flows. The bounded-variance property of sigma_bath is physically real and empirically verified, but its theoretical explanation (equipartition) is classical, not novel. The contribution is the connection to CNF training and the systematic variance characterization, not the underlying physics. Liu et al. 2025 is the direct competitor; our differentiator is the O(1) bound (physical) vs their improved constants (algorithmic).

## References

- Nose (1984). A unified formulation of the constant temperature molecular dynamics methods. J. Chem. Phys.
- Hoover (1985). Canonical dynamics: Equilibrium phase-space distributions. Phys. Rev. A.
- Martyna et al. (1992). Nose-Hoover chains. J. Chem. Phys.
- Chen et al. (2018). Neural ODEs. NeurIPS.
- Grathwohl et al. (2019). FFJORD. ICLR.
- Ceriotti et al. (2010). Colored-noise thermostats. JCTC.
- Tuckerman (2010). Statistical Mechanics: Theory and Molecular Simulation. Oxford.
- Evans & Searles (2002). The fluctuation theorem. Adv. Phys.
- Jarzynski (1997). Nonequilibrium equality for free energy differences. PRL.
- Gallavotti & Cohen (1995). Dynamical ensembles in nonequilibrium statistical mechanics. PRL.
- Legoll et al. (2009). Non-ergodicity of Nose-Hoover dynamics. Nonlinearity.
- Liu et al. (2025). Hutch++ for FFJORD. arXiv:2502.18808.
- Kraskov et al. (2004). Estimating mutual information. Phys. Rev. E.
- Lipman et al. (2023). Flow matching. ICLR.
- Onken et al. (2021). OT-Flow. AAAI.
- Ding et al. (2014). SGNHT. NeurIPS.
- Leimkuhler & Matthews (2013). Rational stochastic numerical methods. AMRX.

## Cleanup pass 2026-04-11

Applied the completion reviewer's NEEDS-CLEANUP fixes on commit 4f8ba50.
Only paper.tex and this log.md were touched; no figures, no scripts, no new scope.

### Blocking fixes applied

- **B1 (fig1_concept)** — Added LaTeX comment above `\includegraphics` line:
  `[TODO] Fig1 image shows old diffusion-table content; regenerate to match
  caption "Two paths to the divergence integral". Placeholder image preserved
  for layout.` No orbit-local script generates fig1_concept (only `run.sh`
  exists in orbits/nh-cnf-paper-058/), so option-2 (regenerate) was not
  available. Image left in place for layout.
- **B2 (fig3_phase caption/panel-(d) mismatch)** — Caption edit (option 1).
  Removed the "kinetic energy" claim and the "trajectory explores both wells"
  phrase. Panel (a) now reads "Configuration space $(q_1, q_2)$ colored by
  time." Panel (d) now reads "Bounded friction $g(\xi) = \tanh(\xi)$". The
  body paragraph above the figure was aligned to match.
- **B3 (author placeholder)** — Added `% [TODO: fill author info]` comment
  above the `\author{...}` block. Placeholder name/affiliation preserved.

### Cleanup fixes applied

- **C1 (RK4 first use)** — Expanded first occurrence at line ~399 to
  "fourth-order Runge-Kutta (RK4)". Second occurrence in a later figure
  caption left as "RK4".
- **C2 (Sigma vs sigma_tot)** — No-op. grep shows zero occurrences of
  `\Sigma` or "Σ" in paper.tex; the abstract and body both already use
  `\stot` = `\sigma_{\mathrm{tot}}`. Drift had already been resolved in
  the rewrite commit. Verified with
  `grep -cE 'Sigma|Σ' orbits/nh-cnf-paper-058/paper.tex` -> 0.
- **C3 (fig_tasaki_quench panel (b) 10x claim)** — Caption edit. Panel (b)
  description now explicitly says the $10\times$ variance ratio
  (std($\stot$) $= 1.05$ vs.\ std($\shutch$) $= 10.7$) is reported
  numerically in Table~\ref{tab:phase2} of Section~\ref{sec:phase2} rather
  than visualized in the panel. Caption no longer over-promises.

### Out of scope (not touched)

- End-to-end FFJORD training experiment (Fig new-2 from audit.md) — future
  work, already acknowledged in Discussion section. No [TODO] added.

### Remaining human TODOs (marked in paper.tex)

1. **Author info** — replace `Wujie Wang\thanks{...}` / `Affiliation
   placeholder` at lines 46-48 (see `% [TODO: fill author info]` comment).
2. **Regenerate fig1_concept.png** — current image still shows the old
   diffusion-model analogy; the caption now describes "Two paths to the
   divergence integral" (Rademacher vs. bath-heat). See
   `% [TODO] Fig1 image ...` comment above `\includegraphics` at line ~154.
