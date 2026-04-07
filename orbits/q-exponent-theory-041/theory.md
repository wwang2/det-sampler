# Why Q_opt ~ omega^{-1.55} for the Log-Osc Thermostat

## Summary

The empirical finding from orbit #040 is that the optimal thermostat mass
for the log-osc sampler scales as Q_opt ~ 2.34 * omega^{-1.55} on a 1D
harmonic oscillator with frequency omega, compared to NHC's Q_opt ~ omega^{-2}.

**Key result:** The -1.55 exponent is not a fundamental power law. It arises
from a crossover between two regimes:

- **Regime A** (omega < omega_xi_max ~ 0.73): The thermostat can resonate with
  the physical oscillator. Q_opt follows the resonance condition, giving
  Q ~ 2/omega^2 (exponent ~ -2), matching NHC.

- **Regime B** (omega > omega_xi_max): The resonance condition has no solution.
  Q_opt collapses to a small, weakly omega-dependent value.

A single power-law fit across both regimes yields the intermediate exponent -1.55.

## Derivation

### Step 1: The Student-t marginal distribution

For a single log-osc thermostat with mass Q on a system at temperature kT=1:

    dxi/dt = (p^2 - kT) / Q

The thermostat potential is V(xi) = log(1 + xi^2), giving the extended
Hamiltonian contribution Q * V(xi). The marginal distribution of xi at
equilibrium is:

    P(xi) ~ exp(-Q * log(1 + xi^2)) = (1 + xi^2)^{-Q}

This is a (generalized) Student-t distribution with parameter nu = 2Q - 1.
Crucially, it is **normalizable only when Q > 1/2**. For Q < 1/2, the
distribution is not integrable and the system cannot reach thermal equilibrium.

### Step 2: The coupling strength <g'(xi)>

The friction coupling g(xi) = 2xi/(1+xi^2) has derivative:

    g'(xi) = 2(1 - xi^2) / (1 + xi^2)^2

which equals V''(xi). The thermal average is:

    <g'(xi)>_Q = (2Q - 1) / (Q + 1)

**Derivation:** Write g'(xi) = 4/(1+xi^2)^2 - 2/(1+xi^2), then use:

    <(1+xi^2)^{-k}>_Q = Z(Q+k)/Z(Q)

where Z(Q) = sqrt(pi) Gamma(Q-1/2)/Gamma(Q). Applying Gamma recurrences:

    <(1+xi^2)^{-1}> = (Q - 1/2) / Q
    <(1+xi^2)^{-2}> = (Q + 1/2)(Q - 1/2) / ((Q + 1) Q)

Therefore:

    <g'> = 4(Q+1/2)(Q-1/2)/((Q+1)Q) - 2(Q-1/2)/Q
         = (Q-1/2)/Q * [4(Q+1/2)/(Q+1) - 2]
         = (Q-1/2)/Q * 2Q/(Q+1)
         = (2Q - 1) / (Q + 1)

This formula is verified numerically to machine precision.

**Key properties:**
- <g'> = 0 at Q = 1/2 (thermostat decouples)
- <g'> -> 2 as Q -> infinity (saturation at maximum coupling)
- <g'> increases monotonically with Q

For NHC: g(xi) = xi, so g' = 1 for all xi, and <g'> = 1 regardless of Q.
This constant coupling is the fundamental difference.

### Step 3: The effective thermostat frequency

The thermostat variable xi oscillates (in a linearized sense) with frequency:

    omega_xi^2 = <V''(xi)> / Q = <g'(xi)> / Q = (2Q - 1) / (Q(Q + 1))

**Crucially, omega_xi has a maximum:**

    d(omega_xi^2)/dQ = 0  =>  Q* = (1 + sqrt(3)) / 2 ~ 1.366

    omega_xi_max = sqrt(omega_xi^2(Q*)) = sqrt((sqrt(3)-1) / (sqrt(3)(3+sqrt(3))/2))
                 ~ 0.732

For NHC: omega_xi = sqrt(kT/Q) = 1/sqrt(Q), which decreases monotonically
and has NO maximum -- it can reach any frequency by choosing Q small enough.

### Step 4: The resonance condition and its breakdown

Optimal thermostat performance requires the thermostat to respond at the
physical frequency omega. Setting omega_xi(Q) = omega:

    (2Q - 1) / (Q(Q + 1)) = omega^2

This is a quadratic in Q:

    omega^2 Q^2 + (omega^2 - 2) Q + 1 = 0

with solutions:

    Q = [(2 - omega^2) +/- sqrt((omega^2 - 2)^2 - 4 omega^2)] / (2 omega^2)

**The discriminant vanishes when omega = omega_xi_max ~ 0.73.**

- For omega < omega_xi_max: two real solutions exist. The larger root is the
  physically relevant one (Q > 0.5). This is Regime A.

- For omega > omega_xi_max: no real solution. The thermostat CANNOT match the
  physical frequency. This is Regime B.

### Step 5: Regime A (omega < omega_xi_max)

In this regime, Q_opt is determined by the resonance condition. For large Q
(small omega):

    omega_xi^2 ~ 2/Q  =>  Q_resonance ~ 2/omega^2

This gives exponent -2, identical to NHC.

The correction from the exact formula (2Q-1)/(Q(Q+1)) vs 2/Q makes Q_resonance
slightly smaller than 2/omega^2. Empirically, Q_opt/Q_resonance ~ 0.89 for
omega = 0.1, 0.3, 1.0 -- excellent agreement.

### Step 6: Regime B (omega > omega_xi_max)

When resonance is impossible, the thermostat operates in an overdamped/driven
regime. For small Q (< 0.5):

- The equilibrium distribution P(xi) is NOT normalizable
- xi undergoes large excursions driven by (p^2 - kT)/Q
- g(xi) = 2xi/(1+xi^2) saturates at |g| <= 1

In this regime, the thermostat provides a bounded, time-varying friction
that depends weakly on Q. The optimal Q is determined by a balance between
response speed (small Q = fast xi dynamics) and stability (not too small).

The empirical Q_opt in this regime is nearly constant (Q ~ 0.03-0.06),
with a very shallow omega dependence.

### Step 7: The crossover exponent

Fitting Q_opt ~ c * omega^alpha across the FULL range [0.1, 30]:

- Three points in Regime A: Q = 178, 17.8, 1.78 (omega = 0.1, 0.3, 1.0)
  These follow omega^{-2}.

- One point in the crossover: Q = 0.1 (omega = 3.0)

- Two points in Regime B: Q = 0.032, 0.056 (omega = 10, 30)
  These are nearly flat.

A single power-law fit blends the steep -2 slope at low omega with the
nearly flat slope at high omega, yielding an intermediate exponent of -1.55.

The R^2 = 0.906 (not close to 1.0) reflects that a single power law is a
poor model for this two-regime system.

## Comparison with NHC

| Property | NHC | Log-osc |
|----------|-----|---------|
| g(xi) | xi (linear) | 2xi/(1+xi^2) (saturating) |
| V(xi) | xi^2/2 (quadratic) | log(1+xi^2) (logarithmic) |
| <g'> | 1 (constant) | (2Q-1)/(Q+1) (Q-dependent) |
| omega_xi | 1/sqrt(Q) (no max) | sqrt((2Q-1)/(Q(Q+1))) (max at Q* ~ 1.37) |
| omega_xi_max | infinity | 0.732 |
| Q_opt scaling | omega^{-2} (exact) | omega^{-2} for omega < 0.73, then breakdown |
| Full-range fit | omega^{-2.00}, R^2=1.000 | omega^{-1.55}, R^2=0.906 |

## Key Takeaway

The -1.55 exponent is an artifact of fitting a two-regime system with a
single power law. The underlying physics is:

1. The log-osc thermostat's effective frequency omega_xi(Q) has a MAXIMUM
   of ~0.73, unlike NHC which can resonate at any frequency.

2. For omega below this maximum, log-osc behaves like NHC: Q_opt ~ 1/omega^2.

3. For omega above this maximum, resonance is impossible and Q_opt drops
   to a small, weakly omega-dependent value.

4. This implies that for high-frequency modes, a SINGLE log-osc thermostat
   is fundamentally limited. Multi-scale chains or other modifications are
   needed to recover NHC-like performance at high omega.

## References

- Martyna et al. (1992) -- Nose-Hoover Chains
- Orbit #040 (q-omega-mapping) -- empirical Q_opt measurements
- Orbit #035 (q-optimization) -- log-osc integrator implementation
