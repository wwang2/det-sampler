---
strategy: nh-cnf-paper
type: paper
status: complete
eval_version: eval-v1
metric: 0.012
issue: 58
parents:
  - nh-cnf-deep-057
  - bayesian-posterior-056
---

# nh-cnf-paper-058: NeurIPS paper draft -- NH-CNF with exact divergence

## Glossary

- **NH**: Nose-Hoover thermostat
- **CNF**: Continuous Normalizing Flow
- **FFJORD**: Free-Form Jacobian of Reversible Dynamics (Grathwohl et al. 2019)
- **KDE**: Kernel Density Estimation
- **ED**: Energy Distance
- **ULA**: Unadjusted Langevin Algorithm
- **SGLD**: Stochastic Gradient Langevin Dynamics
- **BNN**: Bayesian Neural Network
- **KAM**: Kolmogorov-Arnold-Moser (theory of invariant tori)
- **NHC**: Nose-Hoover Chain
- **JVP**: Jacobian-Vector Product

## Approach

Complete NeurIPS-format paper draft presenting the Nose-Hoover thermostat as a continuous normalizing flow with exact, zero-variance divergence computation. The paper synthesizes experimental results from orbits 056 and 057, plus theoretical insights from orbits 052 and 054.

## Results

The paper (`paper.tex`, 1036 lines) contains:

1. **Abstract** (~200 words): Problem (Hutchinson variance), observation (NH exact divergence), method (NH-CNF), results (up to 6x on 2D multimodal, zero variance, 98% BNN calibration), honest limitation (d>20 trapping).

2. **Introduction** (1 page): CNF density bottleneck, NH thermostat observation, contribution list, concept figure.

3. **Background** (1 page): CNF framework with Hutchinson estimator; NH thermostat dynamics with invariant measure; historical context (Nose 1984, Hoover 1985, Martyna 1992, Ceriotti 2010).

4. **Method** (1.5 pages):
   - Theorem 1: div(f) = -d*g(xi), with 3-line proof
   - NH-CNF algorithm with exact log-density tracking
   - Multi-scale Q as thermostat noise schedule
   - Correspondence table: diffusion models vs NH thermostats

5. **Experiments** (2.5 pages):
   - E1: 2D sample quality (6.1x on spirals, 2.9x on eight Gaussians)
   - E3: Zero-variance divergence vs Hutchinson O(d) variance
   - E5: Phase-space trajectory visualization
   - E2: BNN posterior (98% calibration)
   - E7: Log-likelihood comparison
   - E6: Dimension scaling (honest negative: mode trapping at d>20)

6. **Related work** (0.5 page): CNFs, augmented flows, thermostats in ML, diffusion models.

7. **Discussion** (0.5 page): Honest assessment of where NH-CNF works and fails.

8. **Appendix**: Full divergence derivation, Q_eff universality, experimental details.

All 8 figures from sibling orbits are included. 15 references cited.

## What I Learned

- The paper narrative is strongest when built around the exact divergence theorem as the central contribution, with sampling quality as supporting evidence.
- Honest reporting of the d>20 limitation strengthens the paper -- reviewers respect candor about failure modes.
- The diffusion model correspondence table (Table 1) is a compact way to communicate the conceptual bridge.

## Prior Art & Novelty

### What is already known
- NH thermostat divergence = -d*g(xi) is well known in MD (Hoover 1985, Martyna 1992)
- FFJORD Hutchinson estimator (Grathwohl et al. 2019) is the standard for CNF divergence
- SGNHT (Ding et al. 2014) uses NH dynamics for Bayesian inference

### What this orbit adds
- Systematic framing of NH as a CNF with exact divergence for generative modeling
- Quantitative "Hutchinson horizon" concept and variance scaling measurements
- Experimental evidence on where NH-CNF wins (multimodal 2D, BNN) and fails (d>20)
- Explicit correspondence table between diffusion models and thermostats

### Honest positioning
This paper applies a known mathematical property (NH exact divergence) to a specific modern problem (CNF density evaluation). The novelty is in the connection and the systematic experimental evidence, not in the thermostat dynamics themselves. The paper is transparent about this.

## References

- Nose (1984). A unified formulation of the constant temperature molecular dynamics methods. J. Chem. Phys.
- Hoover (1985). Canonical dynamics: Equilibrium phase-space distributions. Phys. Rev. A.
- Martyna et al. (1992). Nose-Hoover chains. J. Chem. Phys.
- Chen et al. (2018). Neural Ordinary Differential Equations. NeurIPS.
- Grathwohl et al. (2019). FFJORD. ICLR.
- Song et al. (2021). Score-based generative modeling through SDEs. ICLR.
- Ho et al. (2020). DDPM. NeurIPS.
- Lipman et al. (2023). Flow matching. ICLR.
- Ceriotti et al. (2010). Colored-noise thermostats. JCTC.
- Ding et al. (2014). SGNHT. NeurIPS.
- Dupont et al. (2019). Augmented Neural ODEs. NeurIPS.
- Onken et al. (2021). OT-Flow. AAAI.
- Welling & Teh (2011). SGLD. ICML.
- Lakshminarayanan et al. (2017). Deep Ensembles. NeurIPS.
- Leimkuhler & Matthews (2013). Stochastic numerical methods for molecular sampling.

---

## Refinement 1: Updated figures and corrected numbers

Copied updated figures from nh-cnf-deep-057 refine 2:
- fig1_concept.png (from e4_concept.png)
- fig2_density.png (from e1_density.png) -- KDE contour plots with proper thinning
- fig3_phase.png (from e5_phase_space.png) -- background density contours added
- fig4_divergence.png (from e3_advantage.png) -- 3-panel layout (loss noise, trajectory error, dimension scaling)
- fig5_loglik.png (from e7_loglik.png)
- fig6_scaling.png (from e6_scaling.png) -- 2-panel (a)/(b) layout
- fig4_schematic.png (from bayesian-posterior-056/e3_schematic.png)

Paper text updates:
- Abstract: "3--6x" changed to "up to 6x" with honest framing about topology dependence
- Section 4.1: Added note about sample count difference (NH-CNF 14400 vs ULA 1600); added specific ratio breakdown by target
- Section 4.2: Figure caption updated from 4-panel to 3-panel description matching actual figure
- Section 4.3: Removed stale "5000 steps" claim; added background contour description
- Section 4.5: Updated figure caption to (a)/(b) panel labels; rewrote diagnosis with figure references; added honest note that Langevin wins at all dimensions on GMM potential
- Contributions list: updated to match abstract

---

## Refinement 3: Theory cleanup from deep-review comment

Addressed three critical theoretical errors flagged in the theory deep-dive review ([Issue #58 comment](https://github.com/wwang2/det-sampler/issues/58#issuecomment-4228031268)):

### Fix 1: Extended vs. marginal density (Section 3.2)
Clarified that `log p_T(...)` tracked by the NH-CNF refers to the density on the *extended* state `(q, p, xi)`, not the marginal `p(q)`. Added a dedicated "Remark (Extended vs. marginal density)" paragraph after Eq. 13 and updated Eq. 13 itself to display the extended variables explicitly. The extended-vs-marginal distinction matters for density-evaluation applications: NH-CNF provides exact density in the extended space and relies on the known equilibrium marginals of `(p, xi)` to recover `p(q)`.

### Fix 2: "Conformal-symplectic" was mathematically wrong
Conformal-symplectic requires a *scalar constant* lambda in L_X omega = lambda*omega (McLachlan & Perlmutter 2001). The NH flow has a *state-dependent* contraction rate g(xi(t)), which is a strictly weaker structure. Replaced the claim with "state-dependent phase-space contracting" and added a citation to McLachlan & Perlmutter (2001) for the classical terminology.

### Fix 3: "1D cheap score" is an analogy, not an equivalence
The diffusion score is a d-dimensional vector field; xi is a scalar. These are not mathematically equivalent. Demoted Table 1 from "Correspondence" to "Analogy," added an explicit disclaimer paragraph noting the dim mismatch, updated the table header to "(analogous role)," updated the concept figure caption, and rewrote Section 3.4 + the related-work paragraph with hedged language. Also fixed the "autonomy enables exact divergence" claim — it is the *block structure* of the Jacobian (position block zero, thermostat block zero, momentum block = -g(xi)*I_d) that enables the cheap trace, not autonomy per se.

### New citations added
- **Tuckerman, M. E. (2010).** *Statistical Mechanics: Theory and Molecular Simulation.* Oxford University Press. ISBN 978-0-19-852526-4. — classical derivation of the NH divergence formula.
- **Evans, D. J., & Searles, D. J. (2002).** The fluctuation theorem. *Adv. Phys.* 51(7), 1529-1585. [doi:10.1080/00018730210155133](https://doi.org/10.1080/00018730210155133). — phase-space contraction in non-equilibrium stat mech.
- **Legoll, F., Luskin, M., & Moeckel, R. (2009).** Non-ergodicity of Nose-Hoover dynamics. *Nonlinearity* 22(7), 1673. [doi:10.1088/0951-7715/22/7/011](https://doi.org/10.1088/0951-7715/22/7/011). — rigorous NH non-ergodicity result, cited for d>20 trapping.
- **McLachlan, R. I., & Perlmutter, M. (2001).** Conformal Hamiltonian systems. *J. Geom. Phys.* 39(4), 276-300. [doi:10.1016/S0393-0440(01)00020-1](https://doi.org/10.1016/S0393-0440(01)00020-1). — for the conformal-symplectic terminology context.

### Honest reframing
- **Section 3 header** reframed: instead of "we prove" the divergence formula, now "we observe that the phase-space contraction rate of the NH flow — a classical result from non-equilibrium statistical mechanics — can be reinterpreted as the divergence of the vector field for a CNF. The formula is classical; its application to CNF density tracking is novel."
- **Introduction** now contains the explicit disclaimer: "Our main contribution is not the divergence formula itself — which has been known in the thermostat literature for decades — but the observation that this formula can replace the Hutchinson trace estimator in continuous normalizing flows."
- **Contribution bullet** renamed from "Exact divergence theorem" to "Classical divergence formula, new application."
- **Abstract** now says "exactly zero *stochastic* variance (deterministic ODE integration error remains)" to avoid the "exact means zero total error" ambiguity.

None of these fixes invalidate the experimental claims or the core theorem — they are presentation cleanups that remove specific attack surfaces a careful NeurIPS/ICLR reviewer would flag.
