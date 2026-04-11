---
strategy: nh-cnf-generative
type: experiment
status: complete
eval_version: eval-v1
metric: 0.050
issue: 58
parents:
  - bayesian-posterior-056
---

<!-- Refine 3 note: E1 metric improved from blurry KDE to proper potentials.
     Eight Gaussians ED went from 0.105 to 0.185 (worse with exact potential
     due to honest mode-hopping failure). Checkerboard from 0.007 to 0.001
     (better). Two Moons from 0.012 to 0.001 (better). Overall metric
     unchanged at 0.012 as it tracks the evaluator KL, not E1 energy distance. -->

# nh-cnf-deep-057: NH-CNF as generative model + annealed NH flow

## Glossary

- **NH**: Nose-Hoover thermostat
- **CNF**: Continuous Normalizing Flow
- **KDE**: Kernel Density Estimation
- **ED**: Energy Distance (sample quality metric)
- **NFE**: Number of Force Evaluations
- **FFJORD**: Free-Form Jacobian of Reversible Dynamics (Grathwohl et al. 2019)
- **ULA**: Unadjusted Langevin Algorithm
- **KAM**: Kolmogorov-Arnold-Moser (theory of invariant tori)

## Approach

The NH thermostat ODE has a structural property that makes it a natural continuous normalizing flow: its phase-space divergence is exactly `div(f) = -d * g(xi)`, where `g` is the damping function (here `tanh`). This gives O(1)-cost, zero-variance log-density tracking -- unlike the Hutchinson trace estimator used in FFJORD, which is O(d) and stochastic.

We tested this in four experiments:

1. **E1**: Sampling from 2D synthetic targets (two moons, two spirals, checkerboard, eight Gaussians) using NH-CNF with KDE-fitted potential, compared to ULA.
2. **E2**: Annealed NH flow with time-dependent Q(t) schedule, exploring the analogy to reverse diffusion.
3. **E3**: Quantifying the exact divergence advantage: zero variance vs Hutchinson, log-density error scaling, dimension dependence.
4. **E4**: Conceptual figure mapping diffusion models to NH thermostats.

## Results

### E1: NH-CNF density estimation (final, Refinement 3)

Both methods use the SAME potential, SAME total gradient evaluations (200k steps x 9 chains = 1.8M), and tuned hyperparameters (Q and eps selected by short preliminary runs). Eight Gaussians uses exact analytical potential; others use KDE(bw=0.1) + GridPotential.

| Target | NH-CNF ED | Langevin ED | Winner |
|--------|-----------|-------------|--------|
| Eight Gaussians | 0.185 | **0.074** | Langevin (2.5x) |
| Checkerboard | **0.001** | 0.003 | NH-CNF (2.7x) |
| Two Moons | 0.001 | **0.001** | Comparable |
| Two Spirals | 0.014 | **0.006** | Langevin (2.5x) |

**Mean NH-CNF ED: 0.050.** NH-CNF wins on checkerboard (periodic structure amenable to deterministic Hamiltonian traversal), loses on multimodal targets (Eight Gaussians, Two Spirals) where stochastic kicks aid mode hopping, and ties on smooth targets (Two Moons).

### E2: Annealed NH flow

The annealed Q schedule experiment showed mixed results:
- Cosine schedule was consistently best among NH schedules at moderate NFE (1000-4000)
- Linear schedule won at high NFE (8000)
- No schedule consistently beat the constant-Q baseline by a large margin
- At high NFE (8000), Langevin beat all NH schedules (ED 1.70 vs best NH 2.11)

**Honest assessment**: The annealed NH flow idea, while conceptually appealing as a deterministic analog of reverse diffusion, does not show a clear practical advantage over constant-Q NH or tuned Langevin on these 2D targets. The analogy to diffusion models is structural but the practical benefit requires further investigation (possibly with learned potentials rather than fixed KDE).

### E3: Exact divergence advantage (strongest result)

This is the clearest win for the NH-CNF framework:

**(a) Zero-variance log-density**: The NH exact divergence produces identical log-likelihood values across repeated evaluations (variance = 0). Hutchinson(1) has variance ~1.4e-4 and Hutchinson(5) ~2.2e-5 per evaluation.

**(b) Hutchinson horizon**: Over long trajectories (T steps), the NH-CNF log-density error is dominated purely by ODE integrator error. The Hutchinson estimator accumulates noise as O(sqrt(T)), creating a "Hutchinson horizon" beyond which the stochastic estimator's error dominates.

**(c) Dimension scaling**: NH-CNF density variance is exactly zero at all dimensions (d=2 to d=100). Hutchinson(1) variance grows roughly linearly with d, reaching 0.32 at d=100. Hutchinson(5) is 5x better but still grows. This O(d) variance scaling is the fundamental limitation of stochastic trace estimators.

### E4: Conceptual figure

Created a schematic mapping:
- Forward noising (diffusion) <-> NH with large Q (weak coupling)
- Reverse denoising (score network) <-> NH with small Q (strong coupling)  
- Noise schedule beta(t) <-> Q schedule Q(t)
- FFJORD pipeline (ODE -> Hutchinson trace -> stochastic log p) vs NH-CNF pipeline (ODE -> exact div -> deterministic log p)


### Prior E1 iterations (superseded)

The following E1 results used blurry KDE potentials (bw=0.35) and are superseded by the Refinement 3 results above. Kept for provenance.

<details>
<summary>Original E1 (blurry KDE, artifact-contaminated)</summary>

NH-CNF (warm-started from data, 10 chains, 8000 total steps) vs ULA (8000 steps, eps=0.005):

| Target | NH-CNF ED | Langevin ED | NH wins? |
|--------|-----------|-------------|----------|
| Two Moons | 0.039 | 0.035 | Comparable |
| Two Spirals | **0.029** | 0.227 | Yes (8x) |
| Checkerboard | **0.028** | 0.150 | Yes (5x) |
| Eight Gaussians | **0.272** | 2.199 | Yes (8x) |

**Caveat**: These results were inflated by blurry KDE potential giving NH an unfair advantage via warm-starting.

</details>

## What I Learned

1. **The exact divergence is the real story.** The sampling quality of NH-CNF vs Langevin depends heavily on the target and tuning. But the zero-variance log-density tracking is a mathematical fact that scales: O(1) cost, zero variance, any dimension.

2. **Single-chain deterministic samplers need warm starting.** Initializing NH far from the target support leads to long burn-in. Warm-starting from data points (or a proposal distribution) is essential.

3. **Multimodal targets favor deterministic dynamics.** On eight Gaussians, Langevin gets trapped while NH traverses modes via Hamiltonian trajectories. This is a genuine advantage of the deterministic approach.

4. **Annealed Q is not a silver bullet.** The time-dependent Q idea is elegant but does not clearly beat constant-Q in practice on simple targets.

## Prior Art & Novelty

### What is already known
- FFJORD (Grathwohl et al. 2019) introduced continuous normalizing flows with Hutchinson trace estimator
- Neural ODE (Chen et al. 2018) is the foundation for CNF-based generative models
- Nose-Hoover thermostat (Nose 1984, Hoover 1985) with exact divergence property has been known in MD community
- The connection between NH dynamics and normalizing flows was noted in orbit/bayesian-posterior-056

### What this orbit adds
- Systematic comparison of NH-CNF sampling vs Langevin on standard generative modeling benchmarks
- Quantitative measurement of the "Hutchinson horizon" -- the trajectory length at which Hutchinson noise dominates integrator error
- Dimension-scaling analysis showing O(d) variance growth for Hutchinson vs zero for NH-CNF
- Testing the annealed Q schedule as deterministic reverse diffusion (negative result: no clear win)

### Honest positioning
The exact divergence property of NH thermostats is well known in molecular dynamics. What is less explored is its value for generative modeling, where FFJORD-style CNFs currently rely on stochastic trace estimators. This orbit provides evidence that the NH-CNF approach has a genuine structural advantage for log-density evaluation, but the sampling quality advantage is target-dependent and requires warm-starting.

## References

- [Grathwohl et al. (2019)](https://arxiv.org/abs/1810.01367). FFJORD: Free-Form Continuous Dynamics for Scalable Reversible Generative Models. ICLR.
- [Chen et al. (2018)](https://arxiv.org/abs/1806.07366). Neural Ordinary Differential Equations. NeurIPS.
- [Nose (1984)](https://doi.org/10.1063/1.447334). A unified formulation of the constant temperature molecular dynamics methods. J. Chem. Phys.
- [Hoover (1985)](https://doi.org/10.1103/PhysRevA.31.1695). Canonical dynamics: Equilibrium phase-space distributions. Phys. Rev. A.
- [Song & Ermon (2019)](https://arxiv.org/abs/1907.05600). Generative Modeling by Estimating Gradients of the Data Distribution. NeurIPS.
- [Ceriotti, Bussi & Parrinello (2010)](https://doi.org/10.1021/ct900563s). Colored-Noise Thermostats a la Carte. J. Chem. Theory Comput.

---

## Refinement 1: Remade figures + new experiments

### E1 Remade: KDE contour density plots with proper thinning

The original E1 showed trajectory-like scatter plots. The remade version uses:
- Multi-scale Q = [0.1, 1.0, 10.0] with 3 chains per Q value (9 chains total)
- 100k steps per chain, 20% burn-in, thinning every 50th sample -> ~14k independent samples
- KDE contour plots (scipy.stats.gaussian_kde) with light scatter underneath
- 10k ground truth samples for comparison

| Target | NH-CNF ED | Langevin ED | NH wins? |
|--------|-----------|-------------|----------|
| Two Moons | **0.012** | 0.015 | Yes (1.3x) |
| Two Spirals | **0.005** | 0.027 | Yes (6.1x) |
| Checkerboard | **0.007** | 0.008 | Yes (1.1x) |
| Eight Gaussians | **0.105** | 0.308 | Yes (2.9x) |

The KDE contour plots now clearly show density structure comparable to ground truth. NH-CNF produces proper density coverage, not trajectory artifacts. The spirals and eight Gaussians show the clearest advantage: NH deterministic dynamics traverse between modes while Langevin's single chain gets trapped.

### E4 Remade: Simplified conceptual figure

Two clean panels:
- (a) Correspondence table: Diffusion Model <-> NH Thermostat mapping
- (b) Computational pipeline: FFJORD (stochastic trace) vs NH-CNF (exact divergence)

### E5 New: Phase-space trajectory visualization

The "wow" figure showing what makes NH unique. Target: two moons, 5000 steps.
- (a) Configuration space q1-q2 colored by time -- shows full 5000-step trajectory exploring both moon basins
- (b) Phase portrait q1-p1 -- shows Hamiltonian oscillation with thermostat damping
- (c) xi(t) time series -- thermostat variable fluctuating around zero
- (d) g(xi) = tanh(xi) -- the friction signal, which IS the exact divergence div/d

This illustrates the core mechanism: xi encodes a scalar summary of "how far from equilibrium" the system is, and feeds back as friction. This is the "cheap score" that replaces a learned score network.

### E6 New: Dimension scaling study (honest negative result)

Gaussian mixture (5 modes, ring layout) in d = 2, 5, 10, 20, 50. Using exact GMM potential (not KDE).

| d | NH-CNF ED | Langevin ED | NH modes | Lang modes |
|---|-----------|-------------|----------|------------|
| 2 | 0.215 | **0.012** | 5/5 | 5/5 |
| 5 | 0.447 | **0.323** | 5/5 | 5/5 |
| 10 | 0.291 | **0.071** | 5/5 | 5/5 |
| 20 | 0.275 | **0.048** | 1/5 | 5/5 |
| 50 | 0.241 | **0.018** | 0/5 | 0/5 |

**Honest assessment**: Langevin outperforms NH-CNF on energy distance at all dimensions tested with the GMM potential. The deterministic NH dynamics with fixed step size and Q values do not scale as well as Langevin's stochastic exploration in higher dimensions. At d >= 20, NH mode coverage degrades severely (1/5 modes at d=20, 0/5 at d=50), while Langevin maintains coverage until d=50 where both fail.

This suggests the NH-CNF advantage is specific to the 2D KDE-potential setting where warm-starting from data and multi-scale Q provide good initialization. In higher dimensions with known potentials, Langevin's noise-driven exploration is more robust. The NH advantage may require learned or adaptive Q schedules to extend to high-d.

### E7 New: Log-likelihood comparison

Test log-likelihood (KDE-fitted on sampler output, evaluated on held-out test data):

| Target | KDE (direct) | NH-CNF samples | Langevin samples |
|--------|-------------|----------------|------------------|
| Two Moons | **-1.47** | -1.83 | -1.92 |
| Two Spirals | **-3.28** | -3.31 | -3.38 |
| Eight Gaussians | **-3.44** | -3.43 | -4.48 |

NH-CNF samples produce better test log-likelihood than Langevin on all targets. The direct KDE baseline (using training data) is slightly better on Two Moons and Two Spirals, which is expected since it uses the original data directly. On Eight Gaussians, NH-CNF matches the direct KDE, while Langevin is substantially worse (mode collapse).

---

## Refinement 2: Figure fixes from PI review

All figures remade with fixes from PI review. Code in `experiment_refine2.py`.

### E1 Fix: Proper KDE bandwidth + scatter background

The over-smoothed KDE contours from refine 1 were caused by scipy's default bandwidth (Silverman's rule). Fixed by using explicit small bandwidths per target (0.05-0.10) and adding raw scatter points (alpha=0.05) underneath. Same bandwidth used for all three columns (ground truth, NH-CNF, Langevin) for fair comparison.

| Target | NH-CNF ED | Langevin ED | NH wins? |
|--------|-----------|-------------|----------|
| Two Moons | 0.012 | **0.010** | Comparable |
| Two Spirals | **0.005** | 0.014 | Yes (3.0x) |
| Checkerboard | **0.007** | 0.011 | Yes (1.5x) |
| Eight Gaussians | **0.105** | 0.112 | Yes (1.1x) |

The density plots now show clear structure: spiral arms, square checkerboard cells, and separated Gaussian modes are all visible.

### E3 Fix: 2x2 layout with new speedup panel

Remade as 2x2 layout (figsize 14x12) with larger panels:
- (a) Loss noise: clearer legend, thicker NH line, markers on Hutchinson
- (b) Hutchinson horizon: cleaner reference lines
- (c) Variance vs dimension: distinct markers and linestyles (squares, diamonds)
- (d) NEW: Speedup of NH-CNF over Hutchinson at target accuracy epsilon. Shows 10-1000x speedup at tight accuracy (eps=0.001) depending on dimension.

### E4 Fix: Larger text + boxed equation

- All text sizes increased ~30% (fontsize 15-17 for table entries, 22 for equation)
- More vertical spacing between rows (dy=1.8, up from 1.4)
- Key equation prominently boxed: nabla dot f = -d g(xi)
- Pipeline boxes wider with more padding

### E5 Fix: Extended trajectory + background density

- Panel (a): Full 5000 steps shown, with target density as gray background contours
- Panel (b): Phase portrait also shows 500 steps
- Panels (c,d): Full 5000-step time series showing clear oscillation patterns
- viridis colormap with thin connecting lines between trajectory points

### E6 Fix: Debugged + 2x2 layout

Root cause of contradictory data: the energy distance was computed correctly, but at d=50 Langevin samples were numerically valid and close to mode centers (low ED) while the mode-counting threshold was too strict. The rerun with NaN checks and log-sum-exp GMM potential gives consistent results.

2x2 layout:
- (a) ED vs d: Langevin outperforms NH-CNF at all dimensions
- (b) Mode coverage vs d: Langevin maintains 5/5 until d=50; NH degrades at d>=20
- (c) Samples at d=2: NH-CNF and Langevin scatter overlay, showing mode coverage comparison
- (d) Samples at d=50 projected: NH-CNF and Langevin scatter overlay, showing collapse at high dimension

| d | NH-CNF ED | Langevin ED | NH modes | Lang modes |
|---|-----------|-------------|----------|------------|
| 2 | 0.477 | **0.011** | 4/5 | 5/5 |
| 5 | 0.362 | **0.294** | 5/5 | 5/5 |
| 10 | 0.451 | **0.120** | 4/5 | 5/5 |
| 20 | 0.321 | **0.010** | 2/5 | 5/5 |
| 50 | 0.353 | **0.026** | 0/5 | 0/5 |

### E7 Fix: Relative comparison + checkerboard

Added second panel showing relative difference from KDE baseline: (NLL_method - NLL_KDE) / |NLL_KDE|. Added checkerboard target (previously missing).

| Target | KDE (direct) | NH-CNF | Langevin |
|--------|-------------|--------|----------|
| Two Moons | **-1.47** | -1.83 | -1.92 |
| Two Spirals | **-3.28** | -3.31 | -3.38 |
| Checkerboard | **-2.72** | -2.82 | -2.89 |
| Eight Gaussians | **-3.44** | -3.43 | -4.48 |

---

## Refinement 3: Root cause fix -- proper potentials for E1

### Diagnosis

The PI identified the root cause of blurry E1 samples: the KDE potential with bandwidth=0.35 is a blurry approximation of the true density. The NH sampler correctly samples from exp(-V_KDE), but V_KDE itself is blurry. No amount of chain length or tuning can fix this -- the sampler is faithfully reproducing the wrong target.

### Fix

Replace the blurry KDE potential with proper potentials for each target:

- **Eight Gaussians**: Exact analytical potential V(x) = -log (1/8) sum_k N(x; mu_k, sigma^2 I), using log-sum-exp for numerical stability. This is the correct potential with zero approximation error.
- **Checkerboard, Two Moons, Two Spirals**: KDE with small bandwidth (bw=0.1, down from 0.35). The key speed optimization: precompute grad_V on a fine 500x500 grid, then use bilinear interpolation during sampling. This converts O(N_data) per KDE eval to O(1) grid lookup.

We also tried:
- **Sigmoid checkerboard potential** (sharpness=20): Failed badly (ED=20.8). The steep gradients at square boundaries cause numerical instability.
- **MLP energy model** (denoising score matching): Failed because the MLP extrapolates poorly -- regions far from training data get low energy (high probability) instead of high energy (repulsive), causing samples to diverge.

### E1 Results (proper potentials)

Both methods use the SAME potential, SAME total gradient evaluations (200k steps x 9 chains = 1.8M), and tuned hyperparameters (Q and eps selected by short preliminary runs).

| Target | NH-CNF ED | Langevin ED | Winner |
|--------|-----------|-------------|--------|
| Eight Gaussians | 0.185 | **0.074** | Langevin (2.5x) |
| Checkerboard | **0.001** | 0.003 | NH-CNF (2.7x) |
| Two Moons | 0.001 | **0.001** | Comparable |
| Two Spirals | 0.014 | **0.006** | Langevin (2.5x) |

### E1 Potential diagnostic

New figure `e1_potential.png` shows the log-probability landscape for each potential, confirming they are well-calibrated before sampling:
- Eight Gaussians: 8 sharp Gaussian modes visible
- Checkerboard: clear alternating squares from KDE
- Two Moons: crescent shapes clearly resolved
- Two Spirals: spiral arms visible in the potential

### Honest assessment

With proper (non-blurry) potentials and fair comparison:
- **NH-CNF wins on checkerboard** (2.7x better ED). The deterministic Hamiltonian dynamics traverse between cells efficiently, while Langevin's random walk is less efficient at exploring the periodic structure.
- **Langevin wins on Eight Gaussians** (2.5x better ED). The exact GMM potential has deep, well-separated modes. Deterministic NH dynamics get trapped in a subset of modes -- without stochastic kicks, escaping requires the trajectory to have exactly the right kinetic energy. Langevin's noise provides the random kicks needed for mode hopping.
- **Comparable on Two Moons**. Both methods sample this smooth, connected distribution well.
- **Langevin wins on Two Spirals** (2.5x better ED). Similar reasoning to Eight Gaussians -- the spiral arms have non-trivial topology that deterministic trajectories don't explore as efficiently.

This corrects the overly optimistic E1 results from refine 1-2 (which showed NH winning on all targets) -- those results were an artifact of the blurry KDE potential, where NH's warm-starting from data gave it an unfair advantage over Langevin's random initialization.

### What changed from refine 2

- **Refine 2 showed NH winning on all 4 targets** because the KDE bw=0.35 potential was so blurry that both methods were sampling from the wrong distribution, and NH's warm-starting gave it an advantage.
- **Refine 3 with proper potentials** shows a more nuanced picture: NH wins on checkerboard (periodic structure), loses on multimodal targets (Eight Gaussians, Two Spirals), and ties on smooth targets (Two Moons).
- The **core NH-CNF contribution** (E3: exact divergence for zero-variance log-density tracking) remains unchanged and is the strongest result from this orbit.
