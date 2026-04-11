# Paper Audit Post-Debate Swarm Pivot

**Orbit:** `nh-cnf-paper-058`
**Date:** 2026-04-10
**Target artifact:** `orbits/nh-cnf-paper-058/paper.tex` (1280 lines, ~14 pages + appendix)
**Pivot source:** debate-swarm findings on orbits 062, 063, 064

---

## 0. TL;DR

The current paper is framed around "NH-CNF: a CNF with exact / zero-variance
divergence, beating FFJORD on sample quality, calibration, and wall-clock."
Three independent findings (orbits 062 E3.3a, 063, 064) show this pitch has
load-bearing problems:

1. The "zero loss variance" headline (Fig `fig_training_stability`) is a
   **tautology under frozen-momentum** — with `p_0` frozen, an analytic
   divergence integral **must** be deterministic; this is not an experiment.
2. The "test NLL comparison" of orbit 063 is actually measuring
   **entropy production offsets** of a biased extended-space density, not the
   marginal `log p(q)`. The 4-17 nat gap between the analytic NLL and the
   reported numbers is the tell.
3. The "triple/four-way identity" of orbit 064 is a **numerical consistency
   check**, not a research result. The pathwise piece is Liouville (classical);
   the on-average piece has `Corr(bath, hutch) = 0.044`, which **kills the
   naive control-variate narrative** unless variance reduction is recovered
   from `sigma_bath`'s own lower variance rather than from correlation.

The salvageable insight — and the right paper — is:

> **`sigma_bath = beta * int tanh(xi) |p|^2 dt` and `sigma_hutch = -int v^T J v dt`
> are both unbiased estimators of the same deterministic divergence integral,
> but `sigma_bath` has `O(1)` saturated variance (equipartition-pinned) while
> `sigma_hutch` has `O(sqrt(t))` random-walk variance. In thermostatted CNF
> training, `sigma_bath` can replace or augment `sigma_hutch` as a
> thermodynamic estimator of the trace integral, giving cleaner gradients
> without per-step Rademacher probes.**

The control-variate framing is the right *mechanical* pitch; the honest
*theoretical* framing is "thermodynamic variance reduction for CNF divergence
estimation." Both are consistent with orbit 064's bath/hutch numbers.

---

## 1. Section map (old -> new)

The current paper has 6 body sections + 5 appendices. Roughly **60% needs to
be rewritten**, **25% can be kept with relabeling**, and **15% should be
cut outright**. New sections are needed for the control-variate pitch.

| Old section | Lines | Status | Action |
|---|---|---|---|
| Abstract (L50-79) | 30 | **REWRITE** | New 200-word abstract built around the bath/hutch variance-reduction story (see S4). Drop "up to 6x better energy distance," "98% BNN calibration," "40-year-old thermostat equation," "first CNF with provably exact divergence from physical principles." |
| S1 Introduction (L82-160) | 79 | **REWRITE** | New framing: FFJORD training is variance-limited; `sigma_hutch` grows as `sqrt(t)`; an equipartition-based estimator with saturated variance is the cheap fix. Keep Figure 1 (concept) but relabel. Drop "zero-variance at machine precision" as a contribution. |
| S2.1 CNF background (L167-190) | 24 | **KEEP** | Still correct. Possibly tighten. |
| S2.2 NH background (L191-230) | 40 | **KEEP** | Factual. Relabel as "thermostat dynamics background." |
| S3.1 Exact divergence theorem (L237-311) | 75 | **REWRITE as "Section 2.3: Classical divergence identity"** | Downgrade Theorem 1 to a Lemma/Remark citing Tuckerman Ch.4 and Evans-Searles. Delete the "Physical interpretation" paragraph that calls it a "1D feedback signal driving dynamics toward target" — this is analogical flourish inconsistent with the pivot. Keep the 3-line trace calculation but as background, not a theorem. |
| S3.2 NH-CNF architecture (L313-366) | 54 | **CUT** | Algorithm 1 (NH-CNF sampling + density evaluation) is an artifact of the sampler framing and produces the biased extended-space density that orbit 063 exposed. Replace with Section 3: "Thermodynamic estimators of the CNF divergence integral" which defines `sigma_hutch` and `sigma_bath` side-by-side as estimators of the same `int tr(J) dt`. |
| S3.3 Multi-scale Q (L367-395) | 29 | **CUT** | "Thermostat noise schedule" is part of the sampler framing; irrelevant to the control-variate pitch. Move to orbit 057/062 archive. |
| S3.4 Connection to diffusion models (L397-452) + Table 1 | 56 | **CUT** | Table 1 was already demoted to "analogy" in Refinement 3. In the new paper the diffusion analogy plays no load-bearing role. Delete the section and the table; keep one sentence in related work. |
| S4.1 E1 2D sample quality (L465-521) + Table tab:e1 | 57 | **CUT** | The 6.1x/2.9x ED advantages on 2D targets are (a) against an *untuned* ULA baseline (known invalid from earlier audit) and (b) irrelevant to the control-variate pitch. Delete entirely. |
| S4.2 Exact divergence advantage (L523-655) | 133 | **REWRITE AS MAIN RESULT SECTION** | This is the only experimental real estate we keep. Drop "training stability — headline effect" (tautology). Keep the anisotropic variance-scaling panel (13x gap at d=200) but **relabel**: it is `std(log p)` across *trajectories*, and the new story is "Hutchinson variance grows with dim even in principle; equipartition estimator does not." Keep wall-clock figure but **relabel** as "cost of exact-trace vs Hutchinson on a structured vector field, used here as a cost-budget baseline for the control-variate experiments." Remove "2-6x speedup over FFJORD" as a contribution; it is a *sanity* measurement, not a headline. |
| S4.3 Phase-space mechanism (L657-682) | 26 | **RELABEL** | Keep Figure 3 (phase space on two moons) as a *pedagogical* figure showing what `xi(t)` and `tanh(xi(t))` look like, then note "this trajectory is the physical object whose `int tanh(xi)|p|^2 dt` we use as an unbiased estimator." Drop the "density-tracking signal comes for free" line. |
| S4.4 BNN posterior (L684-728) + Table tab:e2 | 45 | **CUT** | Orbit 062 re-ran BNN and NH-CNF was *over-cautious* (coverage=1.00 on 95% CI), not "98% calibrated." Moreover, BNN sampling quality is orthogonal to the control-variate pitch. Delete. |
| S4.5 Log-likelihood (L730-767) + Table tab:e7 | 38 | **CUT** | This used KDE-on-samples NLL (not the flow NLL), and orbit 063 showed the flow NLL is off by 4-17 nats because it is the extended-space density, not the marginal. Delete. |
| S4.6 Dimension scaling honest limits (L769-835) + Table tab:e6 | 67 | **RELABEL or CUT** | The d>20 KAM trapping is a property of NH-dynamics as a *sampler*. In the control-variate paper we are not pitching NH as a sampler at all — we are pitching it as a variance-reduction lens on FFJORD's divergence integral. Suggest **CUT** from body; retain 1 paragraph in the discussion ("this paper does not address NH as a sampling method; for a discussion of NH sampling limitations see Legoll 2009, Martyna 1992"). |
| S5 Related work (L838-879) | 42 | **REWRITE** | New paragraphs: (1) control variates in Monte Carlo / variance reduction in FFJORD training (cite Chen et al. 2019 "Residual Flows" for Russian roulette, Finlay et al. 2020 for RNODE regularizers, Onken 2021 for OT-Flow trace simplification), (2) fluctuation theorems / Jarzynski / bath entropy as control variates (cite Gallavotti-Cohen, Jarzynski 1997), (3) thermostats in ML only as pointer. Drop "augmented flows" and "score-based diffusion." |
| S6 Discussion / conclusion (L882-924) | 43 | **REWRITE** | New discussion: scope is thermostatted systems (energy-based models, MD, Bayesian posteriors with explicit likelihoods), limitations are (a) the estimator is only available when kinetic energy can be computed along the flow, (b) naive control-variate at finite t gives ~0 reduction due to `Corr(bath,hutch)=0.044` — the win comes from bath's bounded variance, not from correlation cancellation. |
| App A Full divergence derivation (L1044-1111) | 68 | **KEEP/SHRINK** | Move to a half-page "classical calculation" appendix. It is textbook content; keep for reviewer convenience. |
| App B Q_eff universality (L1114-1142) | 29 | **CUT** | Belongs to the sampler paper, not this one. |
| App C Experimental details (L1145-1201) | 57 | **REWRITE** | Reduce to FFJORD training setup + the two estimator implementations. |
| App D Frozen-momentum protocol (L1204-1228) | 25 | **CUT** | The protocol is what made the tautology look like a result. In the new paper, `p_0` is resampled every step (production training regime), not frozen. |
| App E V_theta bias=False (L1231-1278) | 48 | **KEEP** | Useful practical note; not load-bearing but helps reproducibility. Fine in appendix. |
| -- | -- | **ADD Section 3** | "Bath-heat and Hutchinson estimators of the divergence integral" — formal statement that `E[sigma_hutch] = E[sigma_bath] = int tr J dt` (the first from Hutchinson identity, the second from equipartition + `tr(J_NH) = -d*tanh(xi)`), with variance analysis: `Var[sigma_hutch] = O(t) * ||J||_F^2`, `Var[sigma_bath] = O(1)` bounded by momentum-shell fluctuations. |
| -- | -- | **ADD Section 4 (main experiment)** | "Control-variate FFJORD training on a standard benchmark." Minimum viable: train FFJORD on POWER / GAS / HEPMASS (or a 2D benchmark if compute is tight), with three divergence estimators: (a) `sigma_hutch` (baseline), (b) `sigma_bath` (drop-in replacement, requires NH-augmentation of the state), (c) `sigma_hutch - lambda * (sigma_bath - mean_bath)` control-variate combo. Metrics: training loss curves, gradient noise-to-signal, final test NLL (computed with a *clean* Hutchinson(k=1000) estimator for honesty), wall-clock. |
| -- | -- | **ADD Section 5 (variance diagnostic)** | Port orbit 064's ensemble figure: `std(sigma_bath)` saturates at O(1), `std(sigma_hutch)` grows as `C*sqrt(t)`, `Corr(bath, hutch) = 0.044`. This justifies why the drop-in replacement (not the naive control variate) is the right recipe. |

**Rewrite budget estimate (rough):**
- **Preserved with at most minor edits:** ~180 lines (S2.1 CNF background, S2.2 NH background, App A core, App E bias=False).
- **Relabeled/reframed (same figure, new caption + new surrounding text):** ~150 lines (Figure 3 phase-space, Figure `fig_variance_scaling`, Figure `fig_walltime`, related-work paragraph pointers).
- **Rewritten from scratch (same topic, new framing):** ~350 lines (abstract, intro, S3 divergence identity reframed, S4.2 variance section re-pitched, discussion).
- **Deleted outright:** ~450 lines (Algorithm 1, multi-scale Q, diffusion analogy + Table 1, S4.1 E1, S4.4 BNN, S4.5 NLL, S4.6 scaling, App B, App D).
- **New LaTeX required:** ~400 lines (Section 3 thermodynamic estimators, Section 4 control-variate experiment, Section 5 variance diagnostic).

**Net:** final paper length should be ~1050 LaTeX lines (vs 1280 current), with **~60% new or heavily rewritten content**. Plan on a 2-3 day rewrite sprint.

---

## 2. Figure map (old -> new)

Current figures in `orbits/nh-cnf-paper-058/figures/`:

| File | Current use | Status | Action |
|---|---|---|---|
| `fig1_concept.png` | Fig 1: diffusion-vs-NH concept | **RELABEL** | New caption: "FFJORD's Hutchinson trace estimator (left) vs the thermodynamic `sigma_bath` estimator (right), applied to the same divergence integral along an NH-augmented flow." Drop all "noise schedule <-> Q spectrum" language. |
| `fig2_density.png` | Fig 2: KDE contours on 4 2D targets | **CUT** | Belongs to the old sample-quality narrative; delete. |
| `fig3_phase.png` | Fig 3: phase-space on two-moons | **RELABEL** | New caption: "A single NH trajectory on a 2D double-well. Panel (d) shows `g(xi(t))` and `|p(t)|^2 - d`; their product `tanh(xi)*|p|^2` integrated over time is the `sigma_bath` estimator defined in Section 3." Cut "the signal that drives dynamics" framing. |
| `fig4_divergence.png` | Fig 4: 3-panel Hutchinson horizon | **RELABEL** | Still shows useful accumulation curves. New caption: "Hutchinson variance grows as `O(sqrt(T))` along an ODE trajectory. This motivates the saturated-variance alternative developed in Sections 3-4." Panel (a) "loss noise" is the frozen-momentum tautology — **crop it out or remove it from the composite**. Keep panels (b) `log p error vs T` and (c) `per-sample density variance vs d`. |
| `fig4_schematic.png` | Currently used as Fig 4 schematic in BNN section | **CUT** | BNN section is being deleted. |
| `fig5_loglik.png` | Fig 5: NLL comparison on 2D targets | **CUT** | Reports KDE-on-samples NLL on an untuned ULA baseline; delete. |
| `fig6_scaling.png` | Fig 6: 2-panel d-scaling on 5-mode GMM | **CUT** | Belongs to sampler pitch. |
| `fig7_bnn.png` | Fig 7: BNN posterior strips | **CUT** | BNN section being deleted; numbers were re-measured as "over-cautious" by orbit 062. |
| `fig_training_stability.png` | Headline fig for "zero variance" | **CUT** | Panel (a) is the frozen-momentum tautology. Panel (b) grad-noise ratio still trivially zero for NH exact because the estimator is literally the analytic formula. Remove both panels from the paper. |
| `fig_variance_scaling_new.png` | `std(log p)` vs d, 3 target families | **RELABEL + KEEP** | Re-caption: "Variance of the divergence-integral estimator `sigma_hutch(k=1,5,20)` across 100 ODE trajectories vs state dimension, measured on three Gaussian targets. The anisotropic case shows a 13x Hutchinson penalty at d=200; the bath-heat estimator (added to this figure in the rewrite, see NEEDS below) remains constant." **REQUIRES regeneration** to add `sigma_bath` as a fourth line — the current figure does not include it. |
| `fig_walltime.png` | Wall-clock vs d, d in [2, 1000] | **RELABEL** | Keep as "cost budget for the variance comparison," not as a headline. New caption: "Per-step cost of each divergence estimator. The bath-heat estimator sits at the same cost as NH exact (one `tanh(xi)*|p|^2` evaluation); Hutchinson requires JVPs." **May require regeneration** if `sigma_bath` is not already on the wall-clock plot. |

### Figures **NEEDED** for the new pitch (must exist before submission)

| New fig | Source | What it shows |
|---|---|---|
| **Fig new-1: Two estimators, one integral** | Adapt 064 `fig_triple_identity_ensemble.png` | 3 panels: (a) `sigma_exact`, `sigma_bath`, `sigma_hutch(k=1)` vs t on a single trajectory; (b) ensemble std (N=200 seeds) of each vs t, showing `std_bath = O(1)` saturates while `std_hutch = 0.68*sqrt(t)`; (c) joint scatter `(sigma_bath - sigma_exact)` vs `(sigma_hutch - sigma_exact)` at t=25 with correlation `r=0.044`. This is the *new hero figure.* |
| **Fig new-2: FFJORD training curves with control variate** | Required new experiment in orbit 065+ | 2 panels: (a) test NLL vs training iteration for FFJORD with (i) Hutchinson(k=1), (ii) bath-heat estimator, (iii) CV combo; (b) gradient noise-to-signal vs iteration for the same three variants. Target benchmark: **POWER or 2D GMM** (whichever fits compute budget). |
| **Fig new-3: Variance-reduction budget** | Required new experiment | 1 panel: final test NLL vs total wall-clock for the three estimators, showing Pareto front. This is the only honest way to present a "variance reduction with no accuracy loss" result. |
| **Fig new-4 (optional, if space):** Derivation schematic | New | Block diagram showing the two paths from "int tr(J) dt" to an estimator: (i) Hutchinson: Rademacher probe -> JVP -> trace estimate -> integrate; (ii) bath-heat: NH augmentation -> `tanh(xi)*|p|^2` -> integrate. Useful as a tutorial figure in the intro. |

---

## 3. Numerical claims that need to be removed or reframed

Every number currently in the abstract/intro/results, with defensibility verdict:

| Claim | Location | Verdict | Action |
|---|---|---|---|
| "strictly zero-variance (machine-precision) divergence estimation" | Abstract L62-64; Intro L111-116; S3.1 Remark L294-296 | **NOT DEFENSIBLE** | This is what orbit 062 E3.3a actually measures, but with frozen `p_0` it is tautological: any analytic formula is deterministic when its inputs are frozen. **REMOVE** from abstract, intro, and contribution list. Replace with: "For thermostatted systems, the trace integral admits a closed-form unbiased estimator (`sigma_bath`) with variance bounded by momentum-shell equipartition, in contrast to Hutchinson's `O(t)` random-walk variance." |
| "2-6x wall-clock speedup over FFJORD-style Hutchinson estimators at d=1000" | Abstract L64-66; S4.2 L612-618 | **VALID BUT REFRAMED** | Orbit 062 E3.4 measured it honestly (exact `-d*tanh(xi)` vs JVP-Hutchinson). The number stands. But it compares analytic-NH against JVP-Hutchinson on a *specific vector field* (NH-augmented flow), so the right phrasing is "the analytic trace is cheaper than a JVP on the same augmented dynamics." **REFRAME**: not a headline, move to cost-budget section of the control-variate experiment. Remove from abstract. |
| "13x variance reduction at d=200 anisotropic" | Abstract (implicit); S4.2 L583-589 | **VALID BUT REFRAMED** | Orbit 062 E3.1: at d=200 aniso, NH-exact std=10.1 vs Hutch(1) std=129.5 = 12.8x. **This is `std(log p)` across initial conditions, not per-draw**, and it reflects that a well-conditioned trace scales as `O(d)` while Hutchinson's relative error scales as `O(sqrt(d))` on a Frobenius-dominated Jacobian. Defensible as-is in the *variance* framing, but **re-label** as "cost of Hutchinson on anisotropic Jacobians" rather than as a generic NH-CNF advantage. Keep the number, rewrite the caption. |
| "up to 6x better energy distance on multimodal 2D targets" | Abstract L69-72; S4.1 | **NOT DEFENSIBLE** | Langevin baseline untuned, NFE mismatch, target is KDE-fitted so both samplers have the same target smoothing. **REMOVE** entirely. Not replaced with anything; the paper is no longer about sample quality. |
| "6.1x on two spirals, 2.9x on eight Gaussians" | S4.1 L483-484, tab:e1 | **NOT DEFENSIBLE** | Same. **REMOVE**. |
| "98% calibrated uncertainty on BNN posterior" | Abstract L72-74; S4.4 L700 | **NOT DEFENSIBLE** | Orbit 062 E2 measured NH-CNF coverage at **1.00 (over-cautious)** on all three UCI-like datasets, not 0.98, and the over-caution is a tempered-posterior artifact. **REMOVE** entirely from the paper. |
| "KAM trapping at d>20" | Abstract L74-76; S4.6 L778-785 | **VALID but scope-mismatched** | The measurement is real, but the paper is no longer a sampler paper. **REMOVE** from abstract. Optional 1-line mention in discussion as "we do not address NH as a sampler; see Legoll 2009 for sampling limitations." |
| "loss std = 0.133 / 0.060 / 0.028 for Hutch(k=1/5/20) at d=10" | S4.2 L541-548 | **DEFENSIBLE as Monte Carlo theorem check; NOT defensible as experiment** | The numbers are exactly `C/sqrt(k)` as MC theory predicts — which is why there is no experiment here. **REMOVE** these specific numbers from the paper; they belong in a footnote at most. |
| "NH exact ~1e-14 gradient noise at all d in {2,5,10,20,50}" | S4.2 L550-557 | **TAUTOLOGY** | The NH exact formula is an analytic scalar; its "noise" is numerical roundoff on `tanh(xi)*dt`. **REMOVE**. |
| "anisotropic NH exact std=10.1 vs Hutch(1) std=129.5 at d=200" | S4.2 L585-590 | **VALID** | Orbit 062 E3.1 real measurement. **KEEP** in rewritten Section 5 variance diagnostic, but re-label axes. |
| "per-step cost NH exact 1.5ms, Hutch(1) 3.5ms, Hutch(5) 10ms at d=1000" | S4.2 L612-614 | **VALID** | Orbit 062 E3.4 real measurement. **KEEP** in cost-budget section; not a headline. |
| "Two Moons ED 0.012" frontmatter metric (`metric: 0.012`) | `log.md` L6 | **NEEDS UPDATE** | Current eval metric is 2D two-moons energy distance. For the pivoted paper, this should become e.g. `test NLL on FFJORD + control variate` or `gradient variance reduction ratio vs baseline`. Update `log.md` frontmatter to reflect the new metric. |

### New numbers the pivoted paper MUST cite (from orbit 064)

- **Ensemble std ratio at t=25, N=200 trajectories**: `std(sigma_bath) / std(sigma_hutch) = 0.376` (bath has 2.66x lower variance).
- **Cross-correlation at t=25**: `Corr(sigma_bath - sigma_exact, sigma_hutch - sigma_exact) = 0.044` (essentially independent noises).
- **Variance growth fits**: `std_bath(t) ~ const` (equipartition-bounded), `std_hutch(t) = 0.681 * sqrt(t)` (R^2 = 0.988 on t>1). This is the *core numerical result of the new paper.*
- **From orbit 063 (reused)**: Hutch(1) test-NLL reporting std = 0.027 nats on moons, 0.069 nats on 10D aniso — but **reframed** as "estimator reporting noise" not as NLL headline.
- **From new required orbit 065+** (control-variate experiment): FFJORD test NLL with Hutch(1) vs FFJORD test NLL with `sigma_bath`. MUST SHOW: (i) final NLL no worse than baseline, (ii) training loss std reduced by X% (target: 30-60%), (iii) wall-clock comparable or better.

---

## 4. Draft abstract (new)

**Title (proposed):** *Thermodynamic Control Variates for Continuous Normalizing Flow Training*

**Alternative titles (ranked):**
1. **Thermodynamic Control Variates for Continuous Normalizing Flow Training** (most directly descriptive)
2. *Bath-Heat Estimators: A Bounded-Variance Alternative to Hutchinson's Trick in FFJORD*
3. *Equipartition Estimators for the Divergence Integral in Continuous Normalizing Flows*
4. *From Gallavotti to FFJORD: Thermodynamic Variance Reduction in Neural ODEs*

### Abstract (target 200 words)

> Continuous normalizing flows (CNFs) evaluate the change in log-density
> along a learned ODE by integrating the trace of its Jacobian. Since FFJORD,
> the standard estimator is Hutchinson's trick: a Rademacher-probe stochastic
> trace whose variance grows linearly in state dimension and accumulates as
> `O(sqrt(T))` over the integration window, degrading gradients in long or
> high-dimensional flows. We observe that for any continuous flow equipped
> with a Nose-Hoover-style thermostat, the *same* divergence integral admits
> a second unbiased estimator derived from bath-heat dissipation:
> `sigma_bath = beta * int tanh(xi) |p|^2 dt`. Equipartition (`<|p|^2> = dkT`)
> guarantees that this estimator has the same mean as Hutchinson's, but its
> variance is bounded by momentum-shell fluctuations and does *not* grow
> with trajectory length. On a 200-trajectory ensemble at `t=25`, the bath
> estimator has **2.66x lower** total variance than single-probe Hutchinson;
> the two noises are uncorrelated (`r=0.044`), ruling out naive control
> variates but enabling a clean drop-in replacement. We integrate this
> estimator into a thermostatted FFJORD training loop and show that on
> **[benchmark]** it recovers equivalent final test NLL with lower gradient
> variance and comparable wall-clock. This bridges Gallavotti's entropy
> production formula with modern generative modeling: CNF practitioners
> running Hutchinson have been unknowingly approximating a physical
> quantity that has a cheaper, lower-variance exact form. (198 words)

### Contributions (rewritten, replacing L118-141)

1. **Two estimators for one integral.** We identify the bath-heat integral `beta * int tanh(xi)|p|^2 dt` as an unbiased estimator of the FFJORD divergence integral on any thermostatted NH-augmented flow, and characterize its variance as equipartition-bounded (saturating in `t`), in contrast to Hutchinson's random-walk variance (Section 3).
2. **Ensemble variance measurement.** On a 200-seed ensemble at `t=25`, bath-heat variance is 2.66x lower than single-probe Hutchinson, with cross-correlation `r=0.044` -- meaning the improvement comes from the intrinsic structure of the bath estimator, not from noise cancellation (Section 5).
3. **FFJORD with a thermodynamic estimator.** We train FFJORD with the bath-heat estimator as a drop-in replacement for Hutchinson on **[benchmark]**, reaching equivalent final test NLL with reduced gradient variance and comparable wall-clock (Section 4).
4. **Scope and limitations.** The estimator requires an NH-augmented flow with a tracked momentum. It applies naturally to energy-based models, Bayesian posteriors with explicit likelihoods, and molecular systems; it does *not* apply to learned vector fields with no energy interpretation. We do not address NH as a stand-alone sampler (see Legoll 2009 for non-ergodicity results).

---

## 5. Minimum viable experimental set (MVP)

### Experiments that survive the pivot (from existing orbits)

| Orbit | Experiment | Keeps as | What changes |
|---|---|---|---|
| **062 E3.1** | `std(log p)` across trajectories vs d on 3 targets | **Section 5 variance diagnostic, Fig `fig_variance_scaling_new`** | Must regenerate to add `sigma_bath` as a fourth line. The 13x anisotropic ratio claim is preserved, re-framed as "Hutchinson variance on anisotropic Jacobians." |
| **062 E3.4** | Per-step wall-clock vs d | **Cost budget subsection of Section 4, Fig `fig_walltime`** | Add `sigma_bath` cost line (should be ~equal to NH exact, since both are `O(1)` per step). Not a headline. |
| **064 Refinement 1** | Ensemble variance of `sigma_exact`, `sigma_bath`, `sigma_hutch` on double-well | **Section 5 HERO FIGURE (new Fig 1)** | Use the ensemble numbers directly: `std_bath/std_hutch = 0.376`, `Corr=0.044`, `std_bath=O(1)` saturates while `std_hutch = 0.68*sqrt(t)`. This is the new paper's core quantitative finding. |
| **057 E5** | Phase-space visualization on two moons | **Optional pedagogical Figure 3** | Re-label as "Where the bath estimator comes from: friction `tanh(xi)` and kinetic energy `|p|^2` along a single NH trajectory." |

### Experiments that DO NOT survive the pivot

- **056 E2 BNN posterior (98% calibration).** Actually re-measured as over-cautious (coverage 1.00) by orbit 062, and orthogonal to the control-variate pitch. **Delete.**
- **057 E1 2D sample quality (6.1x / 2.9x ED).** Untuned ULA baseline, KDE-fitted potential, sampler framing. **Delete.**
- **057 E6 / 062 scaling to d=50.** Sampler question, out of scope. **Delete.**
- **062 E3.3 "training stability" panel (a) loss variance.** Frozen-momentum tautology. **Delete.**
- **063 test NLL reporting noise.** Useful as a rhetorical footnote, but the 4-17 nat offset exposes that the "NLL" being reported is not the marginal density. **Keep only as a one-sentence pointer in the experimental details appendix** ("for the variance measurements in Section 5 we reproduce the quadrature protocol of orbit 063").
- **064 pathwise identity `sigma_exact = sigma_lyap`.** Liouville's theorem, classical. Do not present as a result. **Delete from body; keep as a one-sentence consistency check in the appendix.**

### Experiments that MUST BE RUN before submission (gaps)

| ID | Description | Owner / blocking orbit |
|---|---|---|
| **NEW-1** | FFJORD training on a standard benchmark (POWER, GAS, HEPMASS, or 2D GMM) with three divergence estimators: Hutchinson(k=1), bath-heat, and a control-variate combo `sigma_hutch - lambda*(sigma_bath - mean_bath)`. **Metric:** test NLL (computed with Hutch(k=1000) at eval time for honesty), gradient noise-to-signal vs iteration, wall-clock per iter. | Open — suggest spawning **orbit 065 `ffjord-bath-cv`** |
| **NEW-2** | Variance ablation: `sigma_bath` variance vs temperature `kT`, thermostat mass `Q`, trajectory length `t`, and dimension `d`. This is the "equipartition works even out-of-equilibrium" sanity check. | Open — part of orbit 065 |
| **NEW-3** | Demonstration that the NH augmentation does not degrade FFJORD's expressiveness: train FFJORD on the same benchmark with and without NH augmentation (same vector field, just adding `(p, xi)` dims that integrate trivially), show final NLL matches. **This is the load-bearing sanity check.** If NH augmentation breaks FFJORD training, the whole pitch collapses. | Open — required for the main experiment to be credible |
| **NEW-4** | Correlation measurement on the *actual* FFJORD flow (not the double-well of orbit 064). Current `Corr=0.044` is measured on a 2D double-well with analytic NH dynamics. If on a trained FFJORD the correlation jumps to e.g. 0.5, the control-variate combo becomes viable; if it stays near 0, we use drop-in replacement. | Open — part of orbit 065 |

### What the experimental section should look like

```
Section 4: Control-variate FFJORD training
  4.1 Setup: NH-augmented FFJORD on [benchmark]
  4.2 Main result: training curves with three estimators (Fig new-2)
  4.3 Wall-clock Pareto (Fig new-3)

Section 5: Variance structure of the two estimators
  5.1 Ensemble measurement on double-well (Fig new-1, from orbit 064)
  5.2 Dimension scaling of Hutchinson variance (Fig `fig_variance_scaling_new`, from orbit 062 E3.1, re-labeled)
  5.3 Why the naive control variate fails: the Corr=0.044 story
```

No other experiments appear in the body. E2, E1 sample quality, E6 scaling,
E7 NLL tables, multi-scale Q, diffusion-analogy table -- all deleted.

---

## 6. Page budget for the rewrite

| Section | Current pages | Target pages | Action |
|---|---|---|---|
| Abstract + Intro (S1) | 1.5 | 1.0 | Rewrite, tighten |
| Background (S2 CNF + NH) | 1.0 | 1.0 | Keep |
| Classical divergence + two estimators (new S3) | -- | 1.5 | Write from scratch |
| Control-variate FFJORD experiment (new S4) | -- | 2.0 | Write from scratch (blocks on orbit 065) |
| Variance structure diagnostic (new S5) | -- | 1.5 | Port orbit 064 + 062 E3.1 |
| Related work | 0.5 | 0.5 | Rewrite |
| Discussion + limitations | 0.5 | 0.5 | Rewrite |
| **Body total** | **~7** | **~8** | +1 page |
| Appendix A: trace calculation | 1.0 | 0.5 | Shrink |
| Appendix B: Q_eff | 0.5 | 0 | Delete |
| Appendix C: exp details | 1.0 | 1.0 | Rewrite |
| Appendix D: frozen momentum | 0.5 | 0 | Delete |
| Appendix E: bias=False | 1.0 | 0.5 | Keep, compress |
| Appendix NEW: derivation of `sigma_bath` unbiasedness | -- | 0.5 | Write |
| **Appendix total** | **~4** | **~2.5** | -1.5 pages |
| **Grand total** | **~11** | **~10.5** | -0.5 pages |

Target venue consistent with revised scope: **JCTC** (existing memory note),
**ICML** (Workshop on ML+Physics), **AISTATS** if the control-variate framing
is prioritized. JCTC is the least-risk pick because the audience already
believes `sigma_bath = sigma_exact` on average.

---

## 7. Handoff checklist for the rewrite agent

Concrete, in order:

1. **Duplicate `paper.tex` as `paper_pivot.tex` and work in the copy** so the current draft survives as a reference.
2. **Update `log.md` frontmatter** — change `metric: 0.012` to a TBD marker for the new metric (gradient variance reduction ratio or final test NLL on FFJORD+CV).
3. **Delete figures** `fig2_density.png`, `fig4_schematic.png`, `fig5_loglik.png`, `fig6_scaling.png`, `fig7_bnn.png`, `fig_training_stability.png` from `figures/` after confirming they are not referenced in the new draft.
4. **Copy** `figures/fig_triple_identity_ensemble.png` from `/Users/wujiewang/code/det-sampler/.worktrees/triple-identity-064/orbits/triple-identity-064/figures/` into this orbit as `fig1_two_estimators.png` (new hero).
5. **Rewrite abstract** using Section 4 of this audit.
6. **Cut sections** per the section map: Algorithm 1, multi-scale Q, Table 1 diffusion correspondence, S4.1 E1, S4.4 BNN, S4.5 NLL, S4.6 scaling, App B, App D.
7. **Stub out new sections 3, 4, 5** with section headings and 1-paragraph descriptions each. This gives the orbit 065 agent a clear target to fill in.
8. **Post to Issue #59** a status comment indicating the paper is mid-pivot and which numerical claims have been retracted, so any citing work knows.
9. **Block** on orbit 065 before the experimental sections 4 can be fleshed out. The variance section 5 can be written now from existing 062 + 064 data.
10. **When orbit 065 lands**, integrate the FFJORD training curves, update the abstract's `[benchmark]` placeholder, and do a final pass on the discussion.

---

## 8. Risks and failure modes for the pivot

1. **NEW-3 risk (load-bearing):** If NH-augmented FFJORD trains *worse* than plain FFJORD on a real benchmark, the whole drop-in-replacement pitch collapses. Mitigation: run NEW-3 first on a small benchmark (POWER or 2D GMM) before committing to the full Section 4 experimental campaign.
2. **Correlation risk:** The `Corr = 0.044` was measured on a 2D double-well with analytic NH dynamics. On trained FFJORD with a learned `V_theta`, it could go either way. If it rises to ~0.5, the control-variate combo becomes the headline; if it stays near 0, drop-in replacement is the headline. Either way the variance reduction from `std_bath = O(1)` survives.
3. **Venue risk:** JCTC may not care about gradient variance reduction as a generative-modeling contribution; ICML may not care about thermodynamic derivations. Plan to write two versions of the intro — one for each audience — once the experiment lands.
4. **Priority claim risk:** "Gallavotti bath heat = FFJORD divergence integral" may already be known in the statistical-mechanics literature (it is essentially a rewriting of the fluctuation theorem). The paper should explicitly scope the contribution as "bringing this identity to the CNF training problem," citing Gallavotti-Cohen, Jarzynski, and Evans-Searles up front.
5. **Orbit 063 zombie:** If any reader remembers orbit 063's "4-17 nat offsets are entropy production, not NLL," they may misread the new paper as implying the bath estimator gives *density* rather than *divergence*. The new Section 3 MUST be explicit: bath estimates the *trace integral* (a scalar attached to the ODE), not the log-density (which is `log p_0 + int tr J dt` — only the *integral* is estimated stochastically).

---

*End of audit.*
