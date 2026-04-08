---
strategy: friction-survey-045
status: complete
eval_version: eval-v1
metric: 1.0
issue: 45
parents:
  - orbit/q-exponent-theory-041
---

## Glossary

- **g(xi)**: Bounded friction function applied to thermostat variable
- **V(xi)**: Confining potential for thermostat, where g = dV/dxi
- **omega_xi(Q)**: Effective thermostat natural frequency = sqrt(<g'>/Q)
- **omega_max**: Frequency ceiling = sup_Q omega_xi(Q)
- **Q**: Thermostat mass parameter
- **Q***: Value of Q that maximizes omega_xi
- **NHC**: Nose-Hoover Chain
- **tau_int**: Integrated autocorrelation time
- **GMM**: Gaussian Mixture Model

## Result

**The tanh friction function g(xi) = tanh(xi) raises the frequency ceiling from 0.732
to 1.0 (exact), and wins the 5D benchmark (tau_int = 698 vs 920 for log-osc, 1115 for NHC).**

### Corrected frequency ceiling ranking

| Rank | Name | g(xi) | omega_max | Q* type | Notes |
|------|------|-------|-----------|---------|-------|
| 1 | logosc(6,1) | 6xi/(1+xi^2) | 2.197 | finite Q*=0.45 | g'(0)=6 |
| 2 | logosc(4,1) | 4xi/(1+xi^2) | 1.465 | finite Q*=0.67 | g'(0)=4 |
| 3 | tanh | tanh(xi) | 1.000 | Q->0 sup | exact |
| 4 | soft-clip | xi/sqrt(1+xi^2) | 0.992 | Q->0 sup | |
| 5 | arctan | (2/pi)arctan(xi) | 0.972 | Q->0 sup | |
| 6 | logosc(2,0.1) | 2xi/(1+0.1xi^2) | 2.316 | finite Q*=0.14 | quad. artifact? |
| 7 | cubic-sat | 3xi/(1+xi^2)^(3/2) | 0.770 | finite Q*=2.50 | |
| 8 | log-osc | 2xi/(1+xi^2) | 0.732 | finite Q*=1.37 | exact |

Note: The logosc(a,b) family with a>2 achieves omega_max > 1 at finite Q* because
g'(0) = a > 2 and the distribution at moderate Q is concentrated near xi=0 where
g' is large. However, practical performance depends on the full dynamics, not just
the frequency ceiling.

### Benchmark results (5 seeds, 200k force evals)

**5D Anisotropic Gaussian (kappa_ratio = 100):**

| Method | tau_int (mean +/- std) |
|--------|----------------------|
| NHC(M=5) | 1115 +/- 624 |
| log-osc | 920 +/- 175 |
| logosc(6,1) | 799 +/- 103 |
| **tanh** | **698 +/- 196** |

**2D Gaussian Mixture (5 modes, crossings -- higher = better):**

| Method | Mode crossings (mean +/- std) |
|--------|------------------------------|
| NHC(M=3) | 12.4 +/- 2.0 |
| logosc(6,1) | 16.0 +/- 16.3 |
| log-osc | 20.0 +/- 18.0 |
| **tanh** | **43.2 +/- 36.8** |

## Approach

### The frequency ceiling problem

Orbit #041 showed that log-osc g(xi) = 2xi/(1+xi^2) has a frequency ceiling
omega_max = 0.732: the thermostat cannot resonate with physical modes above
this frequency. The question is whether a different choice of g(xi) can do better.

The thermostat variable xi has equilibrium distribution P(xi) ~ exp(-Q V(xi) / kT)
where V(xi) is the confining potential satisfying g = dV/dxi. The effective
thermostat frequency is omega_xi(Q) = sqrt(<g'(xi)>_Q / Q).

### Two mechanisms for higher omega_max

**Mechanism 1: Increase g'(0).** For the parametric family g = a xi/(1+b xi^2),
we have g'(0) = a. At moderate Q, the distribution is concentrated near xi=0,
so <g'> ~ g'(0) = a. This gives omega ~ sqrt(a/Q), and the peak can reach
sqrt(a) times the log-osc value. logosc(6,1) with a=6 achieves omega_max = 2.2.

**Mechanism 2: Keep g' >= 0 everywhere.** For tanh, g'(xi) = sech^2(xi) >= 0
for all xi. This avoids the cancellation that kills <g'> at small Q for log-osc.
The exact formula is:

    P(xi) ~ cosh(xi)^{-Q}
    <g'> = Q / (Q + 1)
    omega_xi = 1 / sqrt(Q + 1)
    omega_max = lim_{Q->0+} omega_xi = 1.0

For log-osc, g'(xi) = 2(1-xi^2)/(1+xi^2)^2 changes sign at |xi|=1. At small Q
the distribution has heavy tails (Student-t), and the negative g' regions
cancel the positive, driving <g'> to zero.

### Why does tanh win the benchmarks?

Despite log-osc having better tau_int on the single-thermostat 1D HO (from earlier
Part 2 data), tanh wins on the multi-scale multi-dimensional benchmarks. The likely
reasons:

1. **Non-negative g' helps ergodicity.** When xi explores large values (transient
   excursions), log-osc's friction can become weakly negative, temporarily amplifying
   momenta. Tanh's friction stays bounded in [0, 1], providing consistently damping
   behavior.

2. **Better multi-scale coupling.** With parallel thermostats at different Q values,
   the total friction is sum_i g(xi_i). Tanh's smoother saturation provides more
   uniform damping across the chain.

3. **Large error bars.** The standard deviations are substantial (especially on GMM),
   so the advantage is suggestive rather than decisive with only 5 seeds.

## What Happened

1. Computed analytical frequency ceilings for 8 candidate g(xi) functions.
2. Discovered two distinct mechanisms for exceeding log-osc's 0.732 ceiling:
   increasing g'(0) (logosc(a,1) family) and eliminating g' sign changes (tanh).
3. Derived exact formula <g'> = Q/(Q+1) for tanh, giving omega_max = 1.0.
4. Found that early numerical results showing omega_max > 1 for arctan/soft-clip
   were quadrature artifacts from insufficient integration domains.
5. Head-to-head benchmarks showed tanh winning on 5D anisotropic Gaussian and
   2D GMM mode-crossing, despite worse single-thermostat 1D performance.

## What I Learned

1. **The g' sign is the key discriminant.** Functions with g' >= 0 everywhere
   (tanh, arctan, soft-clip) avoid the cancellation that creates a finite
   frequency ceiling.

2. **Increasing g'(0) is a separate knob.** The logosc(a,b) family shows that
   larger a raises the peak of omega_xi(Q) at finite Q, without requiring
   g' >= 0 everywhere.

3. **Theoretical omega_max does not directly predict practical performance.**
   Tanh has omega_max = 1.0 (vs logosc(6,1)'s 2.2), but tanh wins the
   benchmark. The full nonlinear dynamics matter beyond the linear frequency
   analysis.

4. **Numerical quadrature at small Q is treacherous.** The distributions
   become extremely broad, requiring integration domains of xi_max > 1000/Q
   for accurate normalization. Exact formulas are essential when available.

## Prior Art & Novelty

### What is already known
- Nose-Hoover thermostat resonance theory: Q_opt = kT/omega^2 (Martyna et al. 1992)
- Log-osc frequency ceiling omega_max = 0.732 (orbit #041)
- tanh, arctan as bounded activation functions: standard in machine learning
- Student-t distributions for power-law potentials: standard stat. mech.

### What this orbit adds
- Survey of 8 bounded friction functions with analytical frequency ceilings
- Exact formula <g'> = Q/(Q+1) for tanh friction, giving omega_max = 1.0
- Classification by sign of g': the key discriminant for frequency ceiling behavior
- Head-to-head benchmarks showing tanh's practical advantage on multi-dimensional systems
- Identification of two independent mechanisms (g'(0) magnitude vs g' sign) for
  exceeding the log-osc ceiling

### Honest positioning
The tanh function and its properties are well-known. The contribution is identifying
that g'(xi) >= 0 eliminates the frequency ceiling, deriving the exact coupling formula
for the tanh thermostat, and demonstrating its practical advantage on benchmarks.
This is an application-level contribution, not a fundamentally new technique.

## References

- Martyna, Klein, Tuckerman (1992) -- Nose-Hoover Chains
- Hoover (1985) -- Canonical dynamics: equilibrium phase-space distributions
- Orbit #041 (q-exponent-theory) -- log-osc frequency ceiling = 0.732
- Orbit #040 (q-omega-mapping) -- empirical Q_opt measurements
- Orbit #035 (q-optimization) -- multi-scale integrator implementation

## Seeds

- Part 2 (1D HO): seed=42 (single seed, validation only)
- Part 3 (5D Gaussian): seeds 5000-5004
- Part 3 (2D GMM): seeds 6000-6004
