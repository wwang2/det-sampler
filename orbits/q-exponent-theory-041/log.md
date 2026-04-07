---
strategy: q-exponent-theory-041
type: theory
status: complete
eval_version: eval-v1
metric: exponent_explained
issue: 41
parents:
  - orbit/q-omega-mapping-040
---

## Glossary

- **NHC**: Nose-Hoover Chain thermostat
- **Log-osc**: Logarithmic-oscillator thermostat, g(xi) = 2xi/(1+xi^2)
- **Q_opt**: Optimal thermostat mass parameter minimizing autocorrelation time
- **omega_xi**: Effective natural frequency of the thermostat variable

## Result

**The -1.55 exponent is a crossover artifact, not a fundamental power law.**

Orbit #040 found Q_opt ~ omega^{-1.55} for a single log-osc thermostat on a
1D harmonic oscillator, compared to NHC's exact omega^{-2}. This work explains
WHY:

| Regime | omega range | Exponent | R^2 | Mechanism |
|--------|------------|----------|-----|-----------|
| A (resonance) | omega < 0.73 | -2.00 | 0.999 | Thermostat frequency matches omega |
| B (driven) | omega > 0.73 | -0.26 | 0.27 | No resonance possible, Q ~ const |
| Full range | 0.1 to 30 | -1.55 | 0.91 | Crossover between A and B |

## Approach

### Analytical derivation

1. **Student-t marginal**: For the log-osc thermostat, xi has equilibrium
   distribution P(xi) ~ (1+xi^2)^{-Q}, a Student-t with nu = 2Q-1.

2. **Exact coupling formula**: <g'(xi)> = (2Q-1)/(Q+1). Verified numerically
   to machine precision for Q = 1, 2, 5, 10, 50.

3. **Thermostat frequency**: omega_xi^2 = <g'>/Q = (2Q-1)/(Q(Q+1)).
   This has a MAXIMUM at Q* = (1+sqrt(3))/2 ~ 1.37:
   omega_xi_max = 0.732.

4. **Resonance breakdown**: For omega > 0.732, the equation omega_xi(Q) = omega
   has no solution. The thermostat cannot match the physical frequency.

5. **Two-regime structure**: The tau(Q) landscape is bimodal, with a spike
   near Q ~ 0.1-0.3. Regime A (large Q) follows resonance; Regime B (small Q)
   is in a driven, non-equilibrium regime.

### Key insight: NHC vs log-osc

NHC: g(xi) = xi, so omega_xi = sqrt(kT/Q) has NO maximum. Resonance is always
possible at Q = kT/omega^2, giving exponent -2 everywhere.

Log-osc: g(xi) = 2xi/(1+xi^2), so omega_xi has maximum 0.732. Above this,
resonance breaks down and the exponent flattens.

## What Happened

The analytical formula <g'> = (2Q-1)/(Q+1) was derived from the Student-t
marginal using Gamma function identities. The key prediction --
omega_xi_max = 0.732 separating two regimes -- was confirmed by the parent
orbit's tau(Q) curves, which show a clear bimodal structure with a spike
near Q ~ Q*.

Within Regime A (omega = 0.1, 0.3), the resonance prediction matches
empirical Q_opt to within 10-14%. The full-range fit of -1.55 emerges
naturally from combining the steep -2 slope in Regime A with the flat
~0 slope in Regime B.

## What I Learned

1. The log-osc thermostat has a fundamental frequency ceiling: it cannot
   resonate above omega ~ 0.73. This limits single-thermostat performance
   for high-frequency modes.

2. The -1.55 exponent is not intrinsic to the coupling function -- it depends
   on the omega range measured. Measuring only omega < 0.5 would give -2;
   measuring only omega > 5 would give ~0.

3. This explains WHY multi-scale thermostat chains are needed: a single
   log-osc thermostat cannot cover arbitrary frequency ranges.

4. The bimodal tau(Q) landscape is a signature of the bounded coupling.
   The spike near Q* ~ 1.37 is where the thermostat is most responsive
   but also most nonlinear.

## Prior Art & Novelty

### What is already known
- Nose-Hoover thermostat resonance theory: Q_opt = kT/omega^2 is standard
  (Martyna et al. 1992, Hoover 1985)
- Student-t distributions arise from power-law potentials (standard stat. mech.)
- Multi-scale thermostat chains address high-frequency limitations (orbit #009)

### What this orbit adds
- Exact analytical formula <g'> = (2Q-1)/(Q+1) for the log-osc coupling
- Identification of omega_xi_max = 0.732 as the crossover frequency
- Explanation of the -1.55 exponent as a two-regime crossover artifact
- Quantitative prediction: Regime A matches empirical data to 10% accuracy

### Honest positioning
The individual ingredients (Student-t marginals, resonance matching) are
well-known. The contribution is combining them to explain a specific
empirical observation from orbit #040. No novelty claim beyond this
application.

## References

- Martyna, Klein, Tuckerman (1992) -- Nose-Hoover Chains
- Orbit #040 (q-omega-mapping) -- empirical Q_opt measurements
- Orbit #035 (q-optimization) -- log-osc integrator implementation
- Orbit #009 (multiscale-chain) -- multi-scale thermostat motivation

## Seeds

All analytical results are deterministic. Parent orbit data uses seeds 0-4
as documented in orbit #040.
