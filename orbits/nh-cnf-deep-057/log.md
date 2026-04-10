---
strategy: nh-cnf-generative
type: experiment
status: complete
eval_version: eval-v1
metric: 0.039
issue: 58
parents:
  - bayesian-posterior-056
---

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

### E1: NH-CNF density estimation

NH-CNF (warm-started from data, 10 chains, 8000 total steps) vs ULA (8000 steps, eps=0.005):

| Target | NH-CNF ED | Langevin ED | NH wins? |
|--------|-----------|-------------|----------|
| Two Moons | 0.039 | 0.035 | Comparable |
| Two Spirals | **0.029** | 0.227 | Yes (8x) |
| Checkerboard | **0.028** | 0.150 | Yes (5x) |
| Eight Gaussians | **0.272** | 2.199 | Yes (8x) |

The NH-CNF excels on multimodal targets where Langevin gets trapped in a subset of modes. The deterministic NH dynamics traverse between modes via Hamiltonian trajectories, while Langevin relies on noise to escape local minima -- which is slow for well-separated modes.

**Caveat**: The NH-CNF samples visually show trajectory structure (connected lines) due to deterministic dynamics. The energy distance metric properly captures coverage quality despite this visual artifact. With warm-starting from data points, the sampler explores the target support efficiently.

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

- Grathwohl et al. (2019). FFJORD: Free-Form Continuous Dynamics for Scalable Reversible Generative Models. ICLR.
- Chen et al. (2018). Neural Ordinary Differential Equations. NeurIPS.
- Nose (1984). A unified formulation of the constant temperature molecular dynamics methods.
- Hoover (1985). Canonical dynamics: Equilibrium phase-space distributions.
- Song & Ermon (2019). Generative Modeling by Estimating Gradients of the Data Distribution. NeurIPS.
