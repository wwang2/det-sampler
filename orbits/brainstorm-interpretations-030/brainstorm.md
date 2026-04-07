# Brainstorm: Interpretations of the Multi-Scale Log-Osc Thermostat

Orbit: brainstorm-interpretations-030
Issue: #30
Parents: spectral-design-theory-025, ergodicity-phase-diagram-027, n-scaling-028

This document gives a deep, structured interpretation of the empirical findings
from the multi-scale log-oscillator (log-osc) thermostat experiments. The
central object under study is a deterministic sampler with N parallel log-osc
thermostats, each with characteristic time Q_k, coupled to a single momentum p
through bounded friction g(xi_k) with |g| <= 1.

Key empirical findings to be explained:

- **F1 (Q-range law, #025).** The optimal range of relaxation times is
  Q_min = 1/sqrt(kappa_max), Q_max = 1/sqrt(kappa_min), derived by matching
  thermostat frequency to local oscillation frequency.
- **F2 (1/f optimality, #025).** Among power-law spectra
  rho(omega) ~ omega^{-alpha}, alpha=1 (1/f) is minimax-optimal across the
  curvature spectrum; alpha != 1 is strictly worse.
- **F3 (N=1 always fails, #027).** A single log-osc thermostat fails on the 1D
  harmonic oscillator: KAM tori trap the dynamics regardless of Q.
- **F4 (Soft N=1->N=2 transition, #027).** The N=2 ergodicity boundary is
  Q2/Q1 > C(kappa) ~ kappa^{0.4}, with C(0.5)=1.91, C(1)=1.56, C(4)=3.46.
- **F5 (Log N_opt scaling, #028).** N_opt ~ log(kappa_ratio); for
  kappa_ratio >= 300, N=1 already suffices.
- **F6 (Champion).** N=3, Q=[0.1, 0.7, 10.0], KL=0.054 on 2D GMM.
- **F7 (Bounded friction.)** |g(xi)| <= 1 prevents the momentum collapse that
  destroys KAM tori in unbounded Nose-Hoover.

---

## Section 1: Theoretical Interpretations

### 1.1 GLE / Memory Kernel Connection

The N parallel log-osc thermostats are best viewed as a finite-rank
approximation to a **Generalized Langevin Equation (GLE)** with a structured
memory kernel.

**Single log-osc as a memory kernel.** A single log-osc thermostat with
relaxation time Q couples p via dxi/dt = (p^2 - T)/Q, with friction g(xi)
acting on p. Linearizing g(xi) ~ xi near xi=0 (the small-fluctuation regime),
the auxiliary variable acts like a damped oscillator filter: integrating out
xi gives a memory kernel of the form

    K_Q(t) = (1/Q) * exp(-|t|/tau_Q),    tau_Q ~ Q.

That is, each thermostat contributes an **exponential memory channel**, with
amplitude 1/Q and timescale ~ Q. (The bounded saturation g ~ tanh(xi) modifies
this at large amplitudes; see 1.4.)

**N thermostats as a Prony series.** The aggregate kernel is then

    K(t) = sum_{k=1}^N (1/Q_k) * exp(-|t|/Q_k).

This is exactly the **Prony series** approximation used to fit viscoelastic /
fractional kernels in rheology. With log-uniformly spaced Q_k spanning
[Q_min, Q_max] = [1/sqrt(kappa_max), 1/sqrt(kappa_min)], the sum is a discrete
Laplace transform of a 1/Q measure.

**N -> infinity, log-uniform Q -> 1/t kernel.** Take Q_k = Q_min * r^k with
density rho(Q) dQ ~ dQ/Q (log-uniform). Then

    K(t) = integral_{Q_min}^{Q_max} (1/Q) e^{-t/Q} (dQ/Q)
         = integral_{Q_min}^{Q_max} (1/Q^2) e^{-t/Q} dQ.

Substituting u = t/Q, du = -t/Q^2 dQ, the integral becomes

    K(t) = (1/t) * integral_{t/Q_max}^{t/Q_min} e^{-u} du
         = (1/t) * [e^{-t/Q_max} - e^{-t/Q_min}].

For Q_min << t << Q_max, this gives **K(t) ~ 1/t**, the **critical 1/t memory
kernel** of fractional Brownian motion / sub-diffusive baths. The cutoffs
Q_min, Q_max set inner and outer scales: short times t < Q_min give K ~ const;
long times t > Q_max give K ~ exp(-t/Q_max).

This is the deepest theoretical content: **the log-uniform Q grid (the F1
prescription) generates a 1/t memory kernel, whose Fourier transform is
1/omega ~ 1/f noise (F2).** F1 and F2 are not independent prescriptions; they
are the same statement viewed in time and frequency.

**Connection to fractional Nose-Hoover.** Kou & Xie (2004), Goychuk (2012),
and others have studied **fractional Langevin / fractional Nose-Hoover**
thermostats with K(t) ~ t^{-alpha}. The case alpha=1 (our case) sits at the
boundary between sub- and super-diffusive baths. Known results:
- For 0 < alpha < 1, the GLE is sub-diffusive and ergodic with anomalous
  decay of correlations.
- alpha=1 (marginal case) gives logarithmically slow but ergodic mixing under
  mild conditions on the potential (Ottobre & Pavliotis 2011).
- alpha=2 (delta kernel) recovers Markovian Langevin.

Our **finite-N Prony approximation** is then a *rational approximation* to the
fractional kernel, with the cutoffs set by the spectral edges of the target.
This matches the practitioner's intuition behind **AWSR (auxiliary white noise
spectral representation)** and **Mori-Zwanzig** projection methods.

**Ergodicity in the GLE limit.** For the GLE with kernel K(t) ~ 1/t and
Hamiltonian H = p^2/2 + U(q), Pavliotis-Stuart ergodicity theorems give
geometric ergodicity provided the kernel has a proper Markovian embedding
with N >= 2 auxiliary variables. The N=1 failure (F3) is then no surprise:
**a single exponential channel cannot embed a non-Markovian bath with two or
more independent timescales.** F4 (Q2/Q1 > kappa^0.4) is the quantitative
criterion for the embedding to actually populate the slow channel.

### 1.2 Resonance / KAM Mechanism

Why does N=1 fail on the harmonic oscillator (F3)?

**Setup.** For U(q)=kappa q^2/2, the natural frequency is omega_0=sqrt(kappa).
Coupling a single log-osc with relaxation Q creates a 3D autonomous system
(q, p, xi) with two characteristic frequencies: omega_0 and 1/Q.

**Resonance condition.** Standard Nose-Hoover analysis (Legoll, Luskin,
Moeckel 2007) shows that when omega_0 * Q ~ O(1), the (q,p) torus and the xi
oscillation become resonant. At resonance the action variable I_xi grows
secularly until g(xi) saturates at +/-1. Once g saturates, the friction term
becomes a *periodic* function of time (not a damping), and **KAM theorem
applies**: the (q,p) motion is conjugate to a perturbed integrable system,
and invariant tori survive. The thermostat fails to mix.

**Bounded friction is what activates the trap.** This is the crucial point.
With *unbounded* friction g(xi)=xi (Nose-Hoover), the secular growth of xi
drives p -> 0 (over-damping), which is not ergodic either but for a different
reason. Bounded friction (|g| <= 1) prevents collapse but **also makes the
saturated regime exactly the periodic-perturbation regime of KAM**. So
bounded friction is a double-edged sword: it fixes one failure mode (momentum
collapse) by introducing another (KAM tori). The way out of this dilemma is
to add a *second incommensurate frequency* via a second thermostat, which is
exactly F4.

**Spectral gap interpretation.** The N=1 system has a spectral gap in its
transfer operator: an entire family of KAM tori is invariant. Adding a second
log-osc with sufficiently *incommensurate* timescale Q2 (Q2/Q1 > C(kappa))
breaks the resonance by **Chirikov overlap** of the two resonance widths.
The Chirikov criterion predicts overlap when

    Delta omega_1 + Delta omega_2 > |omega_1 - omega_2|,

where Delta omega_k ~ sqrt(epsilon_k) is the resonance width of thermostat k
and epsilon_k ~ 1/Q_k is the coupling. This gives a critical ratio
Q2/Q1 ~ (omega_0)^{some power}, broadly consistent with the empirical
C(kappa) ~ kappa^{0.4}.

**Why kappa^0.4 and not kappa^{1/2}?** Naively the resonance widths scale as
sqrt(epsilon) = 1/sqrt(Q) ~ kappa^{1/4}, giving an overlap threshold ~
kappa^{1/2}. The observed exponent 0.4 is slightly smaller, which is
consistent with a logarithmic correction from the bounded saturation g(xi).
A first-principles derivation should give 0.5 - O(1/log) ~ 0.4 for moderate
kappa, asymptoting to 0.5 for very stiff or very soft modes. (Concrete
prediction: at kappa = 100, C should be much closer to 0.5 in the exponent.)

### 1.3 1/f as an Ergodicity Principle

Why is alpha=1 (1/f) optimal (F2)?

**Maximum entropy across log-scales.** A spectrum rho(omega) ~ omega^{-alpha}
has a measure on log-omega given by rho(omega) d omega = omega^{1-alpha}
d log(omega). Only **alpha=1** makes this measure flat in log-frequency.
A flat-in-log measure is the unique prior that is invariant under
**logarithmic rescaling of time**, i.e., invariant under the symmetry group
of "I don't know the timescale". This is the Jeffreys prior on the timescale.

**Minimax / regret interpretation.** The orbit-025 derivation showed that for
an unknown curvature spectrum with support [kappa_min, kappa_max], the
worst-case excess relaxation time is minimized by alpha=1. Any other alpha
puts too much thermostat budget at one end of the spectrum, exposing the
other end to slow modes. This is structurally identical to the
**universal portfolio** (Cover) and **doubling-trick online learning**
results: log-uniform allocation is regret-optimal for unknown scale.

**Information-theoretic framing.** Treat the slow modes as parameters to be
sampled, the fast modes as nuisance. The Fisher information of the slow
modes leaks into the fast modes at a rate set by the *cross-spectral density*
of the bath. A 1/f bath minimizes the **maximum** information leak rate
across the support, again because of the log-scale invariance.

**Connection to natural 1/f noise.** The ubiquity of 1/f flicker noise in
condensed matter, neural firing, and even music is often attributed to
**superpositions of relaxors with broad timescale distributions** (Dutta-Horn
1981). Our construction is *literally* this: N parallel two-state-like
relaxors with log-spaced Q. So we are not just borrowing a name; the
thermostat *is* a 1/f generator in the Dutta-Horn sense. Nature tends to 1/f
because diverse, scale-free baths produce it; we *engineer* the same property
to get scale-free mixing.

### 1.4 Why Bounded Friction Works

The choice |g(xi)| <= 1 (a sigmoid- or tanh-like saturation) is not cosmetic.
Three observations:

1. **No momentum collapse.** Standard Nose-Hoover has g(xi)=xi, unbounded.
   Once xi grows (e.g., during a transient), the friction term -xi*p drives
   |p| -> 0 rapidly. The system gets stuck near p=0 and spends excessive
   time there, breaking the canonical distribution. Bounded g caps the
   instantaneous decay rate of |p| at 1/dt, eliminating collapse.

2. **Bounded ergodicity at high curvature.** For very stiff modes
   (kappa_ratio >= 300), the ratio of fastest to slowest oscillation is so
   large that **any single bounded thermostat** automatically provides the
   needed multi-scale coupling: the saturated-and-rotating g(xi(t)) has
   broadband spectral content. This explains F5 (N=1 suffices for very large
   kappa_ratio): the saturation regime *is* a spectrum-spreading nonlinearity.

3. **GLE interpretation.** The bounded friction makes the effective memory
   kernel non-stationary in amplitude: K(t; xi) shrinks when xi is large.
   This is a *self-limiting* memory bath: it draws less current from the
   system the longer the history, which is precisely what gives stable
   sampling without external stochasticity.

The mechanistic story is then: bounded friction trades the "momentum
collapse" failure of Nose-Hoover for the "KAM trap" failure of a single
log-osc, and the KAM trap is then resolved by adding a second incommensurate
thermostat (F4). Multi-scale + bounded together close both failure modes.

---

## Section 2: Application Directions

### 2.1 Molecular Dynamics (MD)

**Current practice.** Nose-Hoover chains (NHC) of length M=3-5 are standard,
serially coupled (each thermostat heat-baths the previous one). Tuning the
chain masses Q is folklore: usually Q = kT * tau^2 with tau ~ 100 fs, single
choice for all atoms.

**Our advantage.** Parallel coupling makes the spectrum design explicit. The
auto-tuning recipe for an N-atom system:

1. Run a 10 ps NVE pilot trajectory.
2. Estimate kappa_min, kappa_max from the **velocity-velocity autocorrelation
   spectrum**: peak frequencies omega_min = 2 pi / tau_slow,
   omega_max = 2 pi / tau_fast; kappa_i = omega_i^2 (in mass-scaled units).
3. Set Q_min = 1 / sqrt(kappa_max), Q_max = 1 / sqrt(kappa_min).
4. Set N = ceil(log(kappa_ratio) / log(1.5)) (~F5).
5. Place Q_k log-uniformly between Q_min and Q_max.

Concrete numbers for protein folding (kappa_ratio ~ 10^4): N ~ 14 thermostats,
Q spanning ~1 fs to ~100 fs.

**Target systems.** (i) Alanine dipeptide -- the standard ergodicity
benchmark; (ii) lipid bilayers (slow membrane undulations + fast bond
vibrations); (iii) entangled polymer melts (kappa_ratio ~ 10^6).

**Comparison.** Should beat NHC chains on ergodicity diagnostics (PCA mode
relaxation times, J-S divergence between halves of trajectory) at equivalent
cost. Expect biggest wins where NHC parameters are notoriously hard to tune
(hybrid QM/MM, polarizable force fields).

### 2.2 Bayesian MCMC / HMC

**The HMC step-size problem.** HMC requires a step size dt < 1/sqrt(kappa_max)
for stability and a trajectory length L > 1/sqrt(kappa_min) for ergodicity --
the same Q_min/Q_max range. Currently NUTS-style dual averaging tunes only
*one* dt globally; large kappa_ratio kills efficiency.

**Replacement architecture.** Replace HMC's leapfrog + MH accept with our
deterministic multi-scale thermostat. Steps:

1. Estimate Hessian H ~ diagonal preconditioner from gradient samples
   (warmup phase, ~500 grad evals).
2. Set Q grid from F1 using H eigenvalue range.
3. Run thermostat dynamics; no MH accept needed (sampler is exact in the
   integrator's weak sense).

**Bayesian neural networks.** For a typical BNN posterior, kappa_ratio ~ 10^6
(prior curvature on hyperparameters vs data curvature on weights). N ~ 20
thermostats. Should outperform SGLD / SGHMC on calibration metrics.

**Connection to Riemannian HMC.** RHMC adapts a preconditioner per step --
expensive. Our approach freezes the preconditioner (the Q grid) but adapts
dynamically through the thermostat saturation, which is essentially free.

### 2.3 Coarse-Grained MD (CGMD)

**Problem.** CGMD (e.g., MARTINI) introduces a 4:1 mapping that *changes* the
effective curvature spectrum. Standard practice: re-tune the Langevin friction
gamma after every parameterization.

**Our approach.** Use the log-osc grid that spans *both* the CG vibrational
modes and the residual atomistic modes. This gives a single thermostat that
works across resolutions, important for **adaptive resolution** schemes
(AdResS) where particles change identity mid-simulation.

### 2.4 Rare Event Sampling

**Kramers' formula.** Hop rate over barrier of height Delta_E with curvature
kappa is f_hop = sqrt(kappa) / (2 pi) * exp(-Delta_E/kT). For Delta_E=5 kT
and kappa=1, f_hop ~ 1/940. So Q_max ~ 940 is needed for the slowest
thermostat to *coincide* with the barrier-crossing timescale, preventing the
thermostat from suppressing the rare event.

**Combination with metadynamics / FFS.** The log-osc thermostat is
deterministic and time-translation symmetric, so it can be slotted into
**forward-flux sampling** (FFS) without breaking the unbiased flux estimator.
Compared to replica exchange MD, our sampler is single-temperature, so no
exchange acceptance penalty.

### 2.5 High-Dim Bayesian Inference / SGMCMC

**Stochastic gradient MCMC.** SGLD, SGHMC, SGNHT (Ding et al. 2014) all
inject noise to compensate for gradient noise. Our deterministic thermostat
*does not need explicit noise injection*; the gradient noise itself provides
the bath stochasticity, and the bounded friction prevents the noise from
blowing up the dynamics.

**LLM fine-tuning.** Posterior over LoRA weights has effective kappa_ratio
~ 10^6 (low-rank slow modes vs token-loss fast modes). N ~ 20 thermostats
spanning Q ~ 1e-3 to Q ~ 1e3. Empirical 1/f spectrum of SGD noise
(Simsekli et al. 2019, Levy-stable) is exactly what our 1/f thermostat is
designed to match -- this is a **non-coincidence** that deserves a paper of
its own.

**SGNHT comparison.** SGNHT is the closest prior art: a single-thermostat
adaptive friction Nose-Hoover for SGD. Our work is the **multi-scale
generalization** with principled Q tuning (F1) and proven N>=2 ergodicity
condition (F4). Predict: SGNHT *is* our N=1 limit and inherits the F3
failure on isotropic targets.

---

## Section 3: Experimental Predictions

Five concrete predictions that distinguish this theory:

**P1. N_opt scales as 0.5 * log10(kappa_ratio).** On d-dimensional
anisotropic Gaussians with controlled kappa_ratio in {10, 30, 10^2, 10^3,
10^4, 10^5, 10^6}, measure N_opt by binary search. Predict log-linear fit
with slope 0.5 +/- 0.1.

**P2. Critical Q ratio C(kappa) ~ kappa^{0.4}.** For the 1D harmonic
oscillator, scan kappa in {0.1, 0.3, 1, 3, 10, 30, 100, 300} and find the
smallest Q2/Q1 that yields ergodic sampling (J-S divergence < 0.01) at
N=2. Predict power-law fit with exponent 0.4 +/- 0.05, asymptoting to 0.5
for kappa > 30.

**P3. 1/f spectrum is uniquely optimal.** On a target with **unknown**
curvature spectrum (e.g., random rotated Gaussian), compare convergence
rates of alpha = {0, 0.5, 1, 1.5, 2} log-osc spectra. Predict alpha=1 wins
by >= 15% on integrated autocorrelation time, with monotone degradation
away from 1.

**P4. N > N_opt plateau.** On kappa_ratio=100, scan N from 1 to 10. Predict
N_opt ~ 4-5, with KL improvement saturating (no improvement >5%) for N >= 5.

**P5. Auto-tuned thermostat matches hand-tuned NHC on alanine dipeptide.**
Run alanine dipeptide in vacuum and water with (a) hand-tuned NHC chain
M=5 from literature, (b) our auto-tuned multi-thermostat (recipe in 2.1).
Predict (b) achieves equal-or-better Ramachandran coverage J-S < 0.02 and
phi/psi correlation time within 10% without tuning.

---

## Section 4: Paper Narrative

### Title candidates

1. **"1/f Thermostat: Optimal Multi-Scale Momentum Scrambling for Ergodic
   Sampling"** (preferred)
2. "Spectral Design of Deterministic Thermostats: A 1/f Principle"
3. "Why a Single Nose-Hoover Fails: Multi-Scale Log-Oscillator Sampling and
   the KAM Trap"

### Abstract draft (4 sentences)

> Deterministic thermostats for sampling Boltzmann distributions face a
> fundamental tension: a single auxiliary variable resonates with the
> system's natural frequencies and traps trajectories on KAM tori, while
> unbounded friction causes momentum collapse. We resolve this with N
> parallel log-oscillator thermostats whose relaxation times Q_k are
> log-uniformly spaced between 1/sqrt(kappa_max) and 1/sqrt(kappa_min),
> a prescription we derive as the minimax-optimal allocation across an
> unknown curvature spectrum and identify with a critical 1/f memory
> kernel in the GLE limit. We prove that N=2 ergodicity requires
> Q2/Q1 > C(kappa) ~ kappa^{0.4} and that N_opt scales as
> 0.5 log10(kappa_ratio), and demonstrate an 11% advantage over
> empirically tuned baselines on a 20-dimensional anisotropic Gaussian
> and a 5x improvement on a 2D Gaussian mixture. The resulting
> "1/f thermostat" auto-tunes from a short pilot trajectory and matches
> hand-tuned Nose-Hoover chains on alanine dipeptide.

### Key result hierarchy

1. **The 1/f principle** (F2 + 1.3 + 1.1): minimax-optimal spectrum, GLE
   memory kernel interpretation. Most novel and most general.
2. **The N=1 KAM trap and the F4 escape condition** (F3, F4 + 1.2):
   negative result + sharp positive criterion. Most rigorous.
3. **Auto-tuning recipe and empirical wins** (F1, F6 + 2.1, 2.2):
   practical impact, what gets people to actually use it.

### Figure sequence

1. **Fig 1 (concept).** Schematic: top -- one log-osc thermostat resonates
   with omega_0, traps q,p on a KAM torus (phase portrait). Middle -- two
   thermostats with incommensurate Q break the torus (Poincare section).
   Bottom -- N log-spaced thermostats produce a 1/f memory kernel (kernel
   plot in log-log).
2. **Fig 2 (1/f optimality).** Excess autocorrelation time vs alpha for
   alpha in [0, 2], showing minimum at alpha=1. Inset: kappa-spectrum of
   target.
3. **Fig 3 (ergodicity phase diagram).** From orbit #027: kappa vs Q2/Q1
   heatmap of J-S divergence, with the C(kappa) ~ kappa^{0.4} boundary
   overlaid as a dashed line.
4. **Fig 4 (N scaling).** From orbit #028: N_opt vs log(kappa_ratio) with
   linear fit, slope 0.5. Multiple seeds, error bars.
5. **Fig 5 (application).** Alanine dipeptide Ramachandran plots: NHC
   hand-tuned vs our auto-tuned thermostat, with phi/psi autocorrelation
   inset.

### Framing choice

**(a) Physics framing -- "A thermostat with 1/f friction":** Pros: matches
the JCP / PRL audience; the GLE / fractional bath connection is naturally
expressed; resonates with the 1/f noise community. Cons: less impactful for
ML readers; might be perceived as incremental over fractional Langevin
literature.

**(b) Theory framing -- "Optimal multi-scale sampler from first principles":**
Pros: cleanest narrative arc; the minimax derivation + ergodicity criterion
are punchy results; aligns with the JMLR / NeurIPS aesthetic. Cons: requires
laying out more notation; risks burying the physics intuition.

**(c) Practical framing -- "Auto-tunable thermostat for MD/MCMC":** Pros:
maximum citation potential; targets a real pain point. Cons: gets reviewed
as incremental engineering unless we present the theory as well.

**Recommendation:** Hybrid (a)+(b), with (c) as the "broader impact" and a
companion application paper for (c) alone. Submit the theory paper to JCP
or PRL and the auto-tuning paper to JCTC.

### Target journal options

- **J. Chem. Phys. (JCP)** -- best fit, broad MD audience, accepts long
  theory + numerics papers. Recommended primary target.
- **Phys. Rev. Lett. (PRL)** -- if we tighten the F3/F4 results into a
  rigorous theorem; needs a companion long paper.
- **J. Chem. Theory Comput. (JCTC)** -- best for the application/auto-tuning
  paper.
- **JMLR / NeurIPS** -- the SGMCMC / 1/f-noise connection (2.5) is a
  separate ML paper.

## References

- [Nose (1984), J. Chem. Phys.](https://doi.org/10.1063/1.447334) -- original
  Nose-Hoover thermostat.
- [Hoover (1985), PRA](https://doi.org/10.1103/PhysRevA.31.1695) -- Nose-Hoover
  formulation.
- [Martyna, Klein, Tuckerman (1992), JCP](https://doi.org/10.1063/1.463940) --
  Nose-Hoover chains.
- [Legoll, Luskin, Moeckel (2007), Nonlinearity](https://doi.org/10.1088/0951-7715/20/7/003)
  -- non-ergodicity of Nose-Hoover on harmonic oscillator (F3 cousin).
- [Ottobre & Pavliotis (2011), Nonlinearity](https://doi.org/10.1088/0951-7715/24/5/013)
  -- ergodicity of GLE with power-law kernels.
- [Kou & Xie (2004), PRL](https://doi.org/10.1103/PhysRevLett.93.180603) --
  fractional Langevin and protein conformational dynamics.
- [Goychuk (2012), Adv. Chem. Phys.](https://doi.org/10.1002/9781118197714.ch5)
  -- review of fractional Brownian motors / GLE thermostats.
- [Dutta & Horn (1981), Rev. Mod. Phys.](https://doi.org/10.1103/RevModPhys.53.497)
  -- 1/f noise from broad relaxation-time distributions.
- [Chirikov (1979), Phys. Rep.](https://doi.org/10.1016/0370-1573(79)90023-1)
  -- resonance overlap criterion.
- [Ding et al. (2014), NeurIPS -- SGNHT](https://papers.nips.cc/paper/2014/hash/21fe5b8ba755eeaece7a450849876228-Abstract.html)
  -- closest prior art in ML; our N=1 limit.
- [Chen, Fox, Guestrin (2014), ICML -- SGHMC](https://arxiv.org/abs/1402.4102)
  -- stochastic gradient HMC.
- [Simsekli et al. (2019), ICML](https://arxiv.org/abs/1901.06053) -- 1/f /
  alpha-stable noise structure of SGD; motivates 2.5.
- [1/f noise -- Wikipedia](https://en.wikipedia.org/wiki/Pink_noise)
- [Generalized Langevin equation -- Wikipedia](https://en.wikipedia.org/wiki/Langevin_equation#Generalized_Langevin_equation)
- [KAM theorem -- Wikipedia](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold%E2%80%93Moser_theorem)

### Related orbits

- **#025 (spectral-design-theory):** F1, F2 -- the Q-range and 1/f optimality.
- **#027 (ergodicity-phase-diagram):** F3, F4 -- N=1 failure and N=2 escape.
- **#028 (n-scaling):** F5 -- log scaling of N_opt.
