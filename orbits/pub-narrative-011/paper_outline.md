---
strategy: pub-narrative-011
status: in-progress
eval_version: eval-v1
metric: N/A (narrative orbit)
issue: 11
parent: unified-theory-007
---

# Any Potential, Any Friction: A Unified Framework for Deterministic Thermostat Sampling

## Paper Outline with Narrative Paragraphs

---

## 1. Introduction

> **Key insight:** The Nose-Hoover thermostat is not a single method but the simplest member of an infinite family -- and it happens to be one of the worst.

### Why deterministic thermostats?

Sampling the canonical ensemble is the central computational task of statistical mechanics. You have a system of particles with potential energy U(q), and you need configurations distributed as exp(-H/kT). The stochastic approach -- Langevin dynamics, Metropolis Monte Carlo -- works, but it injects artificial noise into what are fundamentally deterministic equations of motion. For decades, the alternative has been the Nose-Hoover thermostat: extend the system with a single friction variable xi that feeds energy in or out, steering the kinetic temperature toward kT. The equations are elegant, the physics is clear, and the method is deterministic. But it has a fatal flaw.

### The KAM problem

For small or stiff systems -- a single harmonic oscillator, a molecular vibration, a stiff bond -- the Nose-Hoover thermostat fails catastrophically. The trajectory does not explore phase space. Instead, it locks onto quasi-periodic orbits called KAM tori: closed curves in (q, p) space that the dynamics can never escape. The thermostat variable xi oscillates in sync with the physical system, creating a feedback loop that reinforces confinement rather than breaking it. This is not a numerical artifact. It is a structural property of the equations, predicted by KAM theory and observed in every simulation. Nose-Hoover chains (NHC) partially address this by coupling multiple thermostat variables, but the underlying mechanism -- unbounded linear friction -- remains. The chains add more knobs but do not change the shape of the knobs.

### What we do differently

We ask a different question: what if the friction function itself could be redesigned? We prove (Theorem 1) that for ANY smooth confining potential V(xi), the dynamics dq/dt = p/m, dp/dt = -grad U - V'(xi)p/Q, dxi/dt = (K - dkT)/Q preserve the canonical measure exactly. This is not an approximation. The Nose-Hoover thermostat is the special case V(xi) = Q xi^2/2. But choosing V differently -- a logarithmic potential, a hyperbolic cosine, an arctangent -- changes the friction function from linear to bounded, and bounded friction breaks KAM tori. The result is a design toolkit: pick a potential, get a thermostat, and the invariant measure is guaranteed by the theorem.

*Figure: Schematic 1 (fig_schematic_thermostat.png) -- The thermostat concept. Physical system coupled to a heat bath through a friction "coupling knob" xi. Standard NH: knob turns freely and over-tightens. Log-Osc: knob has a soft stop.*

---

## 2. The Generalized Friction Framework

> **Key insight:** The Liouville equation does not care what V(xi) is -- it only requires g(xi) = V'(xi)/Q. Every confining potential gives a valid thermostat.

### The master theorem in plain language

Here is the entire theoretical contribution in one paragraph. Take any smooth function V(xi) that goes to infinity as |xi| grows (a "confining potential"). Define the friction as g(xi) = V'(xi)/Q. Then the extended system -- Hamiltonian dynamics plus this friction, with xi driven by the temperature error (K - dkT)/Q -- samples the canonical distribution exactly. No approximation, no correction, no Metropolis accept/reject. The proof is a direct computation: the divergence of the flow plus the drift along the log-density equals zero, term by term, for any V. The cancellation is not accidental -- it is forced by the structure of the equations, where the xi-dynamics respond to kinetic energy while the p-dynamics respond to V'(xi). The two pieces interlock like gears.

### The design freedom this opens up

The Nose-Hoover thermostat uses V(xi) = Q xi^2/2, giving g(xi) = xi -- linear, unbounded friction. This is the "default" choice, and it is a poor one. But the theorem says we can choose V freely, subject only to smoothness and confinement. Logarithmic potentials give bounded friction. Hyperbolic potentials give saturation at large xi. The choice of V determines three things simultaneously: (1) the friction function g(xi) = V'(xi)/Q, which controls the coupling strength; (2) the thermostat marginal distribution p(xi) ~ exp(-V(xi)/kT), which determines the fluctuation statistics of friction; and (3) the nonlinear dynamics of the extended system, which determines whether KAM tori survive or break. This is a one-knob design space with three-dimensional consequences.

### A gallery of friction functions

The table of admissible thermostats (Corollary 1.1) is not just a list -- it is a design catalog. The Nose-Hoover quadratic V gives Gaussian-distributed xi with unbounded friction. The logarithmic V = Q log(1 + xi^2) gives Cauchy-distributed xi with bounded friction |g| <= 1. The tanh thermostat (V = Q log cosh xi) gives a sech^2-distributed xi. Each choice creates a different dynamical personality: how aggressively the thermostat grabs the system, how it releases, and what fluctuations the friction signal carries. The bounded-friction family, where |g(xi)| stays finite, turns out to be qualitatively different from the unbounded family -- and dramatically better for ergodicity.

*Figure: Schematic 2 (fig_schematic_friction.png) -- Friction function gallery. Side-by-side: g(xi) functions, V(xi) potentials, p(xi) marginal distributions. Annotate: "bounded", "Cauchy tails", "saturation".*

---

## 3. Why Bounded Friction Works

> **Key insight:** A thermostat that knows when to let go is better than one that grips harder and harder. Bounded friction prevents the feedback loop that creates KAM tori.

### The physical intuition

Think of a car's anti-lock braking system. When you slam the brakes, the wheels lock and the car skids -- you lose control. ABS prevents this by pulsing the brakes: grip, release, grip, release. The system never enters the locked state. Bounded friction does the same thing for a thermostat. In standard Nose-Hoover, when the system is too hot (K > dkT), xi grows, friction increases, the system slows down, now it is too cold, xi decreases, friction reverses, and the cycle continues. But because g(xi) = xi is unbounded, the friction can grow arbitrarily strong, locking the thermostat and physical system into a tight oscillation -- a KAM torus. With bounded friction g(xi) = 2xi/(1+xi^2), the maximum friction is 1, reached at xi = 1. Beyond that, the friction weakens even as xi grows. The thermostat lets go. The system escapes.

### Lyapunov evidence

The consequence is measurable. We compute the maximal Lyapunov exponent -- the exponential rate at which nearby trajectories diverge -- for the 1D harmonic oscillator coupled to each thermostat. At Q = 0.1, Nose-Hoover gives lambda = 0.002 (essentially zero -- quasi-periodic, trapped on a torus). Log-Osc gives lambda = 0.626 -- genuinely chaotic, with a 300-fold stronger divergence rate. The pattern holds across Q values from 0.1 to 0.5: bounded friction produces 10-300x larger Lyapunov exponents. This is the smoking gun. Bounded friction does not just "help" with ergodicity. It fundamentally changes the dynamical character of the extended system from quasi-periodic to chaotic.

### The two-regime theory

There is a subtlety. At moderate Q (around 0.5-0.8), both NH and Log-Osc have near-zero Lyapunov exponents -- both are quasi-periodic. Yet Log-Osc still achieves ergodicity scores above 0.94, while NH collapses to 0.54. How? The answer is torus deformation. Even when the dynamics are quasi-periodic, bounded friction deforms the shape of the invariant tori. NH tori are thin ellipses that leave most of phase space unvisited. Log-Osc tori are fat, irregular shapes that cover a much larger fraction of (q, p) space. Phase space coverage at Q = 0.5 is 0.96 for Log-Osc versus 0.34 for NH. The mechanism is different from chaos, but the outcome is the same: better sampling.

*Figure: Schematic 3 (fig_schematic_kam.png) -- KAM tori vs space-filling orbits. Left: NH with nested elliptical tori (trapped). Right: Log-Osc with deformed, space-filling orbits (ergodic).*

---

## 4. Multi-Scale Dynamics

> **Key insight:** A single thermostat oscillates at one frequency. Real systems have dynamics at every frequency. Log-spaced thermostat masses create a broadband friction signal that couples to all timescales.

### Listening to the system at every frequency

A single Nose-Hoover thermostat with mass Q oscillates at a characteristic frequency ~ sqrt(dkT/Q). If Q is small, the thermostat is fast -- good for tracking rapid fluctuations, but too jittery to push the system over energy barriers. If Q is large, the thermostat is slow -- good for barrier crossing, but too sluggish for local temperature control. No single Q can do both. The solution is the same as in audio engineering: use multiple oscillators at different frequencies and mix the signals. We couple M thermostat variables with log-spaced masses Q_j, so their natural frequencies span several decades. The combined friction signal is a superposition of fast, medium, and slow oscillations -- a broadband, approximately 1/f spectrum. This is the thermostat equivalent of white noise, but deterministic.

### Why log-spaced Q values

The choice of log-spacing is not arbitrary. If the Q values are linearly spaced, the frequency spectrum has gaps -- ranges where no thermostat is resonant with the system. Log-spacing ensures approximately equal coverage on a logarithmic frequency axis, which matches the self-similar structure of energy landscapes (barriers at every scale). With 3 log-spaced thermostats (Q = 0.1, 0.7, 10), the integrated autocorrelation time on the 2D double-well drops from 264 (single Q = 1) to 54 -- a 5x improvement. With 5 thermostats, it drops to 41 (6.4x). The returns diminish, but the first few oscillators buy enormous gains.

### Multi-modal hopping

The real payoff is on multi-modal distributions. A 5-mode Gaussian mixture in 2D is a brutal test: the sampler must hop between 5 separated wells. A single log-osc thermostat achieves KL = 0.38 -- it gets stuck in one or two modes. The multi-scale variant with 3 log-spaced Q values achieves KL = 0.054 -- a 7x improvement. The slow thermostat (Q = 10) provides the sustained push needed to climb over barriers, while the fast thermostat (Q = 0.1) provides local equilibration within each well. The medium thermostat (Q = 0.7) bridges the gap. Together, they create a friction signal that is both locally responsive and globally exploratory.

*Figure: Schematic 4 (fig_schematic_multiscale.png) -- Multi-scale thermostat diagram. Three oscillators at different sizes/speeds. Combined friction signal. System crossing a barrier.*

---

## 5. Results

> **Key insight:** The method works on everything from textbook test cases to 39-degree-of-freedom Lennard-Jones clusters, with consistent advantages over NH and NHC baselines.

### Progressive difficulty narrative

We validate on a progression of increasingly difficult systems, each designed to test a specific failure mode.

**1D Harmonic Oscillator (the litmus test).** This is the system that breaks Nose-Hoover. A single particle in a quadratic well -- the simplest possible test. NH achieves an ergodicity score of 0.54 (non-ergodic, trapped on KAM tori). NHC with 3 chain variables achieves 0.92 (marginally ergodic). A single Log-Osc thermostat achieves 0.944 -- better than a 3-variable chain, with a single variable. At the optimal Q = 0.5, the score reaches 0.982. One bounded-friction thermostat beats three unbounded ones.

**2D Double-Well (barrier crossing).** Two wells separated by a barrier of height 1.0. NH achieves KL = 0.037. NHC achieves 0.029. Log-Osc achieves 0.010 -- a 3.7x improvement over NH and reaching the "good sampler" threshold of 0.01. The bounded friction allows larger stable step sizes (dt = 0.035 vs typical dt = 0.01), which means more phase space explored per force evaluation.

**2D Gaussian Mixture (multi-modal hopping).** Five Gaussian modes arranged in a ring. This is where multi-scale dynamics shine. Single-Q log-osc: KL = 0.38. Multi-scale log-osc (3 thermostats, log-spaced Q): KL = 0.054 -- a 7x improvement. The slow thermostat enables barrier crossing that no single-frequency thermostat can match.

**Lennard-Jones Cluster (high-dimensional).** The ultimate test: 13 atoms in 3D, 39 degrees of freedom. Here the LOCR (Log-Osc Chain with selective tail thermostats) variant dominates, outperforming NHC by 40%. The advantage of bounded friction grows with dimension because there are more modes to couple, and the broadband friction signal has more frequencies to match.

*Figure: Schematic 5 (fig_schematic_progression.png) -- Discovery progression timeline. NH -> NHC -> Log-Osc -> Multi-Scale -> NHC-Tail.*

*Figure: Toy 1 (fig_toy_doublewell.png) -- 1D double-well with trajectory overlay and time series.*

*Figure: Toy 2 (fig_toy_harmonic.png) -- Harmonic oscillator phase portrait: NH torus vs Log-Osc space-filling.*

---

## 6. Discussion

> **Key insight:** The theorem is proven. The ergodicity is conjectured. The gap between the two is the most interesting open question.

### What is proven vs conjectured

The invariant measure result (Theorem 1) is rigorous: for any confining V(xi), the canonical distribution is stationary under the generalized friction dynamics. This is a mathematical theorem with a clean, machine-checked proof. What is NOT proven is ergodicity -- the property that the trajectory visits all of phase space, not just a subset. The Lyapunov exponent evidence strongly suggests that bounded friction produces ergodic dynamics for small to moderate Q, but a formal proof of ergodicity for nonlinear thermostat dynamics remains an open problem, just as it does for the original Nose-Hoover system. We have numerical evidence (Lyapunov exponents, phase space coverage, KS statistics) but not a theorem.

### Open questions

Several questions remain. First, is there an optimal V(xi) within the bounded-friction family? The logarithmic potential works well, but the tanh and arctan thermostats also break KAM tori -- is there a principle that selects one over the others? Second, the two-regime theory (chaos at small Q, torus deformation at moderate Q) needs formalization. What geometric property of the tori determines sampling quality? Third, the multi-scale construction is empirical. Is there an optimal distribution of Q values, analogous to optimal quadrature nodes? Information-theoretic arguments might answer this.

### Connection to Langevin dynamics

There is a deep connection to stochastic methods. Langevin dynamics add Gaussian noise to the equations of motion. Our generalized friction thermostats add deterministic, bounded, nonlinear coupling. In the limit of many multi-scale thermostat variables with log-spaced Q, the combined friction signal approximates colored noise with a 1/f spectrum. The deterministic system converges, in some functional sense, toward a stochastic one -- but without ever introducing randomness. This suggests a continuum between deterministic and stochastic sampling, parameterized by the number and distribution of thermostat variables. Understanding this continuum is perhaps the deepest implication of the generalized friction framework.

### Practical recommendations

For practitioners: use the Log-Osc thermostat (V = Q log(1+xi^2)) with Q between 0.3 and 0.8 for systems with fewer than ~10 degrees of freedom. For larger systems or multi-modal targets, use 3-5 log-spaced thermostat variables. For high-dimensional molecular systems (N > 10 atoms), use the LOCR chain variant with selective tail thermostats on the stiffest modes. The bounded friction provides natural numerical stability, allowing step sizes 2-3x larger than standard NH.

---

## Figure Plan Summary

| Figure | Type | Section | File |
|--------|------|---------|------|
| Thermostat concept | Schematic | 1 | fig_schematic_thermostat.png |
| Friction gallery | Schematic | 2 | fig_schematic_friction.png |
| KAM tori vs ergodic | Schematic | 3 | fig_schematic_kam.png |
| Multi-scale diagram | Schematic | 4 | fig_schematic_multiscale.png |
| Discovery progression | Schematic | 5 | fig_schematic_progression.png |
| Double-well + trajectory | Toy illustration | 5 | fig_toy_doublewell.png |
| HO phase portrait | Toy illustration | 5 | fig_toy_harmonic.png |

---

## References

- [Nose, S. (1984). A unified formulation of the constant temperature molecular dynamics methods. J. Chem. Phys. 81, 511.](https://doi.org/10.1063/1.447334)
- [Hoover, W. G. (1985). Canonical dynamics: Equilibrium phase-space distributions. Phys. Rev. A, 31, 1695.](https://doi.org/10.1103/PhysRevA.31.1695)
- [Martyna, G. J., Klein, M. L., & Tuckerman, M. (1992). Nose-Hoover chains. J. Chem. Phys. 97, 2635.](https://doi.org/10.1063/1.463940)
- [Martyna et al. (1996). Explicit reversible integrators for extended systems dynamics. Mol. Phys. 87, 1117.](https://doi.org/10.1080/00268979600100761)
- [Benettin et al. (1980). Lyapunov characteristic exponents for smooth dynamical systems. Meccanica, 15, 9-20.](https://doi.org/10.1007/BF02128236)
- [KAM theorem](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold%E2%80%93Moser_theorem) -- the standard explanation for NH non-ergodicity
- [Nose-Hoover thermostat](https://en.wikipedia.org/wiki/Nos%C3%A9%E2%80%93Hoover_thermostat) -- background
- [1/f noise](https://en.wikipedia.org/wiki/Pink_noise) -- spectral density connection for multi-scale thermostats
- Builds on #3 (log-osc-001) which discovered the log-osc thermostat
- Builds on #10 (unified-theory-007) which proved the master theorem and computed Lyapunov exponents
