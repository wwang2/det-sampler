# Narrative Arc — det-sampler campaign (milestone, 2026-04-11, ultra-research)

The campaign has **bifurcated into two papers**, both building on the same body of orbits but with different theses, venues, and audiences.

---

## Paper 1 — "A frequency-ceiling design criterion for deterministic thermostats"

**Thesis (one sentence):** Requiring `g'(ξ) ≥ 0` in a Nosé-Hoover-type friction yields an explicit upper bound on the thermostat frequency, empirically explains the 536× τ_int gap between log-osc and tanh variants, and gives 7-19× efficiency gains over NHC(M=3) and tuned HMC on anisotropic and multimodal targets in 10 dimensions.

**Status:** publication-ready. Target venue: JCTC.

### Key evidence (unchanged from 2026-04-09 narrative)

- **orbit/q-exponent-theory-041** — exact ω_max = 0.732 for log-osc (resonance calculation).
- **orbit/friction-survey-045** — ω_max = 1.0 for tanh via the g'≥0 mechanism; two distinct criteria identified (raise g'(0) vs eliminate sign changes).
- **orbit/paper-experiments-047** — τ_int = 2.4 (tanh) vs 1287 (log-osc) at d=10 → 536× gap.
- **orbit/mode-hopping-042** — 7× multi-scale Q mode-hopping advantage specifically at d=10.
- **orbit/paper-final-049** — 19.5× ESS vs tuned HMC across 4 benchmarks (force-evaluation efficiency, no rejection waste).
- **orbit/gprime-ablation-052** — causal g'≥0 ablation (pending 1-2% readability fixes noted at orbit 052 cleanup_iterations=2).
- **orbit/kam-failure-map-053** — full N=1 KAM failure surface; per-seed/majority-vote shows log-osc > tanh by 11-24 cells (with seed-averaging caveat explicitly flagged in orbit notes).
- **orbit/forensic-qeff-054** — Q_eff mechanism confirmation.

### What remains for Paper 1
- (a) Clean the readability flags on orbits 052 and 053 (BAOAB mislabel, hardcoded abs paths, missing restricted-window analysis).
- (b) Cite Ceriotti 2010 for the g'≥0 idea's prior art (from `project_paper_narrative.md` memory).
- (c) Consider one real molecular benchmark (LJ-7) — optional, answers a reviewer question before they ask.

### Open questions (Paper 1)
- Does the tanh advantage survive on real molecular systems (LJ-7 or similar) or is it a synthetic-benchmark artifact?
- Is the N-scaling law `N_opt ~ log(κ_ratio)` real or a seed artifact? (R² = 0.33, suggestive but not significant.)

---

## Paper 2 — "Nosé-Hoover as a bounded-variance CNF divergence estimator"

**Thesis (one sentence):** The bath-heat integral `σ_bath = β·∫tanh(ξ)|p|² dt` from an NH-augmented continuous normalizing flow is an **unbiased, O(1)-variance-bounded** estimator of the CNF divergence integral `∫tr(J) dt`, in contrast to the Hutchinson stochastic estimator's `O(√t)` random-walk variance, giving a thermodynamic drop-in replacement for FFJORD's per-step Rademacher probe.

**Status:** under active construction; paper draft exists but audit (orbit 058) says 60% rewrite needed. Target venue: PRL, NeurIPS, or ICML.

### The core finding (orbit 064, 200-trajectory ensemble, 2D double-well, NH-tanh)

On a single 25-time-unit trajectory, four quantities agree pathwise:
- σ_exact (analytic Jacobian trace integral) ↔ σ_lyap (Benettin Lyapunov sum): rel err **2.08e-06** (numerical-precision limit of RK4).
- σ_bath ↔ σ_hutch: agree in mean, differ in variance structure.

Across 200 independent trajectories at `t=25`:
- **std(σ_bath − σ_exact) = 1.303**
- **std(σ_hutch − σ_exact) = 3.470**
- **ratio = 0.376** (bath is ~2.66× tighter than Hutchinson)
- **corr(bath, hutch) = 0.044** (essentially independent — kills the naive control-variate story)
- **σ_bath √t coefficient: 0.346 with R² = -3.22** (variance SATURATES, no √t growth)
- **σ_hutch √t coefficient: 0.681 with R² = 0.988** (clean random walk)

**The selling point is not variance reduction via correlation cancellation — it is variance reduction via a bounded-variance thermodynamic estimator.** The equipartition theorem pins `⟨|p|²⟩ = d·kT`, which bounds the fluctuations of `σ_bath` even as `t → ∞`. Hutchinson has no such bound.

### Evolution of the Paper 2 story arc

1. **orbit/nh-cnf-deep-057** — first framing: NH as a CNF that gives "exact divergence" (true, but tautologically true under frozen momentum).
2. **orbit/nh-cnf-paper-058** — paper draft using the "exact divergence → density estimator" pitch. Audit flagged 5 load-bearing broken claims: "zero loss variance" (tautology), "98% BNN calibration" (actually 1.00 — over-conservative), "6× ED advantage" (untuned ULA baseline), "test NLL off by 4-17 nats" (measuring entropy production, not marginal log p(q)), "triple identity" (just Liouville).
3. **orbit/exact-fep-059, orbit/logZ-estimation-060** — explored free-energy and log-Z interpretations; scored well but orthogonal to the control-variate pitch.
4. **orbit/nh-cnf-thorough-062** — exhaustive refinement of E1/E2/E3; revealed the extended-space vs marginal density confusion.
5. **orbit/nll-eval-noise-063** — showed Hutchinson trace has unacceptable NLL variance for the density-estimator framing.
6. **orbit/triple-identity-064** — the 200-trajectory ensemble above. The key pivot: σ_bath and σ_hutch are both unbiased estimators of the same integral, but σ_bath has O(1) saturated variance while σ_hutch has O(√t) growth. This is the salvageable research result from the whole NH-CNF arc.
7. **orbit/tasaki-quench-065** (in-flight) — verifies Tasaki's non-equilibrium KL identity `∫(σ_bath − σ_exact)dt = D_KL(ρ_{T₀}‖ρ_{T₁})` on a sudden temperature quench. Closed-form analytic target (`d·[log(T₁/T₀) − 1 + T₀/T₁]`). Selling point: at d=10 a kNN-KL estimator from Langevin samples cannot reach the target accuracy due to curse of dimensionality, but NH's deterministic exact trace gets there for free.

### The paper 058 rewrite plan (from audit.md, 294 lines, commit 67b04d2)

- **60% rewrite** (S1 intro, S3.1 exact divergence theorem → lemma, S4.2 experimental section relabeled, S5 related work, S6 discussion).
- **25% relabel** (S2.1, S2.2, S4.3 phase-space mechanism).
- **15% cut** (S3.2 NH-CNF architecture, S3.3 multi-scale Q, S3.4 diffusion connection, S4.1 2D sample quality, S4.4 BNN posterior, S4.5 log-likelihood, S4.6 dimension scaling).
- **New sections needed:**
  - Section 3: "Bath-heat and Hutchinson estimators of the divergence integral" — formal variance analysis.
  - Section 4: "Control-variate FFJORD training on standard benchmark" — train with σ_hutch vs σ_bath vs combo, report training curves and gradient SNR.
  - Section 5: "Variance diagnostic" — port orbit 064's ensemble figure.

### Competition (novelty audit)

**Liu, Du, Deng, Zhang 2025 (arXiv:2502.18808)** — applies Hutch++ to FFJORD for variance reduction on the divergence integral. This is the direct competitor for any "variance reduction for CNF training" pitch.

**Our differentiator must be the specific bounded-variance property**, not variance reduction in general. Hutch++ still has O(√t) variance growth (improved constants, same scaling). σ_bath has O(1) bounded variance (equipartition-pinned, not probabilistic). This is a qualitative distinction, not a constant-factor improvement — Paper 2 needs to land this point on page 1.

### Open questions (Paper 2)
- Does bath-as-drop-in actually accelerate FFJORD training wall-clock on a standard benchmark (POWER/GAS/HEPMASS), or is the variance reduction absorbed by FFJORD's other bottlenecks?
- Does the bounded-variance property survive if you use a chain-Nosé-Hoover (M≥2) — does the per-stage housekeeping correction preserve the O(1) bound? (Orbit 065 Phase 0 is supposed to derive this.)
- Can the Tasaki identity be turned into a non-equilibrium benchmark that Langevin provably cannot match at d=10? (Orbit 065 primary result.)
- Is there an analogous bounded-variance estimator for the **score function** (not just the trace) — i.e. can thermostat methods give O(1) stochastic-gradient noise reduction for score-matching training?

---

## Reconciliation with previous narrative

The 2026-04-09 narrative framed the campaign as a single story built on the g'≥0 criterion. That story is now Paper 1 and is largely frozen. The 2026-04-10 → 2026-04-11 orbits (057-065) explored a completely different question — whether NH dynamics can be cast as a CNF and whether its physical structure gives variance-reduced divergence estimates — and that line of work now constitutes Paper 2.

The initial NH-CNF framing (NH as a density estimator) was load-bearing wrong (extended-space vs marginal confusion in orbit 058/062/063). The pivot via orbit 064's 200-trajectory ensemble recovered a clean research result — bounded-variance vs random-walk trace estimation — that has a direct competitor (Liu et al. 2025) we must out-differentiate on the bounded-variance angle.

**Both papers reuse the NH-tanh simulator.** Paper 1 treats it as a sampler; Paper 2 treats it as a trace estimator for an externally-specified density-tracking problem. This is the same code with two different framings and two different audiences.

---

## Campaign-level open questions (spanning both papers)

- Are there deterministic thermostats with **O(1) bounded variance for higher-moment observables**, not just `|p|²`? (This could unify the two papers under a single design principle.)
- Is the g'≥0 criterion (Paper 1) related to the bounded-variance property (Paper 2) via a common Lyapunov argument? Current evidence: possibly — both trace back to dissipation-fluctuation bounds.
- What is the right **non-equilibrium benchmark suite** for deterministic vs stochastic samplers? Orbit 065's temperature quench is one candidate; transient barrier crossings are another.
