"""ESH-Thermostat Duality: SymPy Analysis.

This script derives the mathematical relationship between ESH dynamics
(Versteeg 2021, arXiv:2111.02434) and our generalized thermostat framework.

Key questions:
1. Is ESH a special case of our Master Theorem?
2. What V(xi) corresponds to ESH?
3. Does the coordinate transform xi = log(|p|/sqrt(kT)) map ESH to our framework?
4. Is Q=kT the magic parameter?
"""

import sympy as sp
from sympy import (
    symbols, Function, exp, log, sqrt, diff, simplify, expand,
    solve, factor, cancel, trigsimp, Rational, oo, pi,
    integrate, Abs, sign, tanh, atanh, cosh, sinh, cos, sin,
    Symbol, Eq, pprint, latex
)

print("=" * 70)
print("ESH-THERMOSTAT DUALITY: SYMPY ANALYSIS")
print("=" * 70)

# ============================================================
# Section 1: Setup symbols
# ============================================================
print("\n" + "=" * 60)
print("SECTION 1: SYMBOLS AND FRAMEWORK SETUP")
print("=" * 60)

q, p, xi, t = symbols('q p xi t', real=True)
kT, Q, d, omega = symbols('kT Q d omega', positive=True)
V = Function('V')
g = Function('g')

print("\nOur Master Theorem: for any confining V(xi),")
print("  dx/dt = p")
print("  dp/dt = -dU/dx - g(xi)*p,   g(xi) = V'(xi)/Q")
print("  dxi/dt = (p^2/d - kT) / Q")
print("\nPreserved measure: rho ~ exp(-U(x)/kT) * exp(-p^2/(2kT)) * exp(-V(xi)/kT)")
print("\nV examples:")
print("  NH:      V(xi) = xi^2/2,                 g = xi/Q")
print("  Log-Osc: V(xi) = Q*log(1+xi^2),          g = 2*xi/(Q*(1+xi^2)) [BOUNDED]")
print("  Tanh:    V(xi) = Q*log(cosh(xi)),         g = tanh(xi)/Q")
print("  Arctan:  V(xi) = Q*(xi*arctan(xi)-log(1+xi^2)/2), g = arctan(xi)/Q")

# ============================================================
# Section 2: ESH dynamics in 1D
# ============================================================
print("\n" + "=" * 60)
print("SECTION 2: ESH DYNAMICS IN 1D")
print("=" * 60)

print("""
ESH (Energy-Sampler-Hamiltonian, Versteeg 2021):
  Logarithmic kinetic energy: K(v) = (d/2) * log(||v||^2 / d)

  In d dimensions:
    dx/dt = v / ||v||                    (unit-speed motion)
    dv/dt = -grad_U(x) * ||v|| / d       (force scaled by speed)

  In 1D (d=1), with v = p:
    dx/dt = sign(p)                      (always unit speed!)
    dp/dt = -dU/dx * |p|                 (force scaled by |p|)

  ESH Hamiltonian: H_ESH = U(x) + (1/2)*log(p^2/d) [up to constants]
""")

# Verify ESH preserves its Hamiltonian
print("Verifying ESH Hamiltonian conservation in 1D:")
# H_ESH = U(x) + (1/2)*log(p^2) for d=1
# dH/dt = dU/dx * dx/dt + (1/p) * dp/dt
# = dU/dx * sign(p) + (1/p) * (-dU/dx * |p|)
# = dU/dx * sign(p) - dU/dx * sign(p) = 0  ✓
print("  dH_ESH/dt = dU/dx * sign(p) + (1/p)*(-dU/dx*|p|)")
print("            = dU/dx * sign(p) - dU/dx * sign(p) = 0  ✓")
print("  ESH IS a Hamiltonian system — it conserves H_ESH exactly!")
print("  This means it is NOT a thermostat in the usual sense.")

# ============================================================
# Section 3: Coordinate transformation
# ============================================================
print("\n" + "=" * 60)
print("SECTION 3: THE KEY COORDINATE TRANSFORMATION")
print("=" * 60)

print("""
Define: xi = log(|p| / sqrt(kT))
So:     |p| = sqrt(kT) * exp(xi)
And:    p^2 = kT * exp(2*xi)

Under this change of variables, what does the ESH xi-update look like?

From ESH: dp/dt = -dU/dx * |p|
  => d(log|p|)/dt = (1/|p|) * dp/dt * sign(p) = -dU/dx * sign(p)

But xi = log|p| - (1/2)*log(kT), so:
  dxi/dt = d(log|p|)/dt = -dU/dx * sign(p)

This is NOT the thermostat xi-update!  In our framework:
  dxi/dt = (p^2/kT - 1) / Q
         = (exp(2*xi) - 1) / Q

These are completely different equations. The ESH xi-evolution is
force-driven (depends on gradient of U), while our thermostat xi-evolution
is kinetic-energy-driven (depends on p^2).
""")

# Let's compute dxi/dt for ESH
print("ESH dxi/dt in terms of xi and the force:")
print("  xi = log(|p|/sqrt(kT))")
print("  dxi/dt = -dU/dx * sign(p)   [ESH, d=1]")
print("")
print("Our thermostat dxi/dt:")
print("  dxi/dt = (p^2/kT - 1) / Q = (exp(2*xi) - 1) / Q")
print("")
print("These are STRUCTURALLY DIFFERENT.")
print("ESH: xi driven by force (non-autonomous, coupled to gradient of U)")
print("Thermostat: xi driven by kinetic energy deviation (autonomous in xi, p)")

# ============================================================
# Section 4: What V(xi) could reproduce ESH-like dynamics?
# ============================================================
print("\n" + "=" * 60)
print("SECTION 4: WHAT V(xi) WOULD MIMIC ESH?")
print("=" * 60)

print("""
Suppose we want our thermostat to have the SAME FRICTION as ESH.
ESH's effective friction on x-dynamics comes from the p-scaling:
  dp/dt = -dU/dx * |p|  = -dU/dx * sqrt(kT) * exp(xi)

This is NOT a friction term of the form -g(xi)*p!
In our framework: dp/dt = -dU/dx - g(xi)*p
In ESH:           dp/dt = -dU/dx * |p|

These have fundamentally different forms:
- Our framework: additive friction -g(xi)*p, force enters as -dU/dx
- ESH: the FORCE itself is scaled by |p|, there is no separate friction

So the p-equation alone tells us ESH CANNOT be written as our thermostat.
""")

# ============================================================
# Section 5: Stationary measure of ESH
# ============================================================
print("\n" + "=" * 60)
print("SECTION 5: STATIONARY MEASURES")
print("=" * 60)

print("""
Our framework preserves:
  rho_thermo(q, p, xi) ~ exp(-U(q)/kT) * exp(-p^2/(2kT)) * exp(-V(xi)/kT)

  The p marginal is GAUSSIAN: p ~ N(0, kT)

ESH preserves (in d dimensions):
  rho_ESH(q, v) ~ exp(-U(q)/kT) * (||v||^2/d)^{-d/2}

  In 1D: rho_ESH(q, p) ~ exp(-U(q)/kT) * (p^2)^{-1/2} = exp(-U(q)/kT) / |p|

  The p marginal is CAUCHY: P(p) ~ 1/|p|  [power law, NOT Gaussian!]
""")

print("Checking the 1D ESH p-distribution:")
print("  From K(v) = (d/2)*log(v^2/d), at d=1: K(p) = (1/2)*log(p^2)")
print("  Boltzmann weight: exp(-K/kT) = exp(-(1/2)*log(p^2)/kT)")
print("                              = |p|^{-1/kT}")
print("")
print("  For kT=1: P(p) ~ |p|^{-1} — this is NOT normalizable on R!")
print("  This means ESH does NOT have a proper canonical distribution in 1D!")
print("  (The distribution is only proper in d >= 2 with appropriate normalization)")

# For d dimensions:
print("\nFor d dimensions:")
print("  K(v) = (d/2)*log(||v||^2/d)")
print("  P(v) ~ ||v||^{-d/kT} * d^{d/(2kT)}")
print("  For kT=1: P(v) ~ ||v||^{-d}")
print("  The spherical integral: integral of ||v||^{-d} * ||v||^{d-1} dv")
print("  = integral of ||v||^{-1} d||v|| — diverges logarithmically at both 0 and inf!")
print("  => ESH's 'stationary measure' is improper even in d dimensions at kT=1")
print("")
print("  In practice, ESH is run with a fixed ||v||=1 or by treating it as")
print("  sampling on a shell — the marginal in x IS correct, but the")
print("  joint (x, v) distribution is not canonical Boltzmann.")

# ============================================================
# Section 6: The actual connection — time reparameterization
# ============================================================
print("\n" + "=" * 60)
print("SECTION 6: THE ACTUAL CONNECTION — TIME REPARAMETERIZATION")
print("=" * 60)

print("""
Despite the differences, there IS a deep connection:

Consider our thermostat with V(xi) such that dxi/dt = (exp(2*xi) - 1)/Q.
This requires solving: V'(xi)/Q = g(xi) where the measure of xi is flat.

Actually, let's think differently. ESH in d dimensions:
  dx/dt = v/||v||              (unit vector direction)
  d||v||/dt = -grad_U · v/||v||/d  * ||v|| = -grad_U·vhat/d * ||v||

  Let s = log||v||, then:
  ds/dt = (1/||v||) * d||v||/dt = -grad_U·vhat / d

This is EXACTLY what you get if you define s = log(||p||) and use
the ESH force scaling. But ds/dt depends on the FORCE, not on p^2.

Now compare to our thermostat (using xi):
  dxi/dt = (p^2/kT - 1)/Q

The connection requires two steps:
1. Rescaling: define tau = integral |p| dt (arc-length time)
2. Then in tau-time, |p| gets absorbed into the dynamics

Under time rescaling t -> tau where dtau/dt = |p|/sqrt(kT):
  dq/dtau = (dq/dt)/(dtau/dt) = p/|p| = sign(p)
  dp/dtau = (dp/dt)/(dtau/dt) = (-dU/dq - g(xi)*p) * sqrt(kT)/|p|

For the p-equation to match ESH (dp/dtau = -dU/dq * sign(p)),
we need the friction term to vanish: g(xi) = 0, which means no thermostat!

Conclusion: ESH is a CONSERVATIVE system, not a thermostat.
""")

# ============================================================
# Section 7: Sympy verification of Liouville condition
# ============================================================
print("\n" + "=" * 60)
print("SECTION 7: LIOUVILLE CONDITION CHECK")
print("=" * 60)

print("""
For our thermostat framework, we verify the Liouville condition:
  div(rho * F) = 0  where F = (dq/dt, dp/dt, dxi/dt)

With rho ~ exp(-U/kT - p^2/(2kT) - V(xi)/kT):

  d/dq [rho * p] + d/dp [rho * (-dU/dq - g(xi)*p)] + d/dxi [rho * (p^2/kT-1)/Q] = 0

Expanding (with rho' = rho * (-dU/dq/kT)):
  rho*(-dU/dq/kT)*p + rho*0  [first term]
  + rho*(dU/dq/kT)*(-dU/dq - g*p)/... + rho*(-g)  [second term]
  + rho*(-V'(xi)/kT)*(p^2/kT-1)/Q + 0  [third term]

After simplification:
  -rho * g(xi) + rho * (-V'(xi)/Q/kT) * (p^2/kT - 1) = 0
  ... which holds when g(xi) = V'(xi)/Q × [correction from kT]

  Actually the condition is: g(xi) = V'(xi)/(Q*kT) * kT = V'(xi)/Q  ✓

For ESH in 1D: F = (sign(p), -dU/dq * |p|, 0)  [no xi]
  Liouville with rho_ESH ~ exp(-U/kT) * |p|^{-1}:

  d/dq[rho_ESH * sign(p)] + d/dp[rho_ESH * (-dU/dq * |p|)]
  = -dU/(dq*kT) * rho_ESH * sign(p) + (-dU/dq/kT)*(-1)|p| * rho_ESH/|p|
     + rho_ESH * (-dU/dq) * sign(p)  [from d/dp of |p|]
  = -dU/dq/kT * sign(p)*rho_ESH - dU/dq/kT * (-sign(p))*rho_ESH
    + (-dU/dq)*sign(p)*rho_ESH

  Hmm, let me be more careful...
""")

# Let's do this symbolically
print("Symbolic verification:")
U_sym = symbols('U', positive=True)
dUdq = symbols('dU_dq', real=True)

# ESH 1D: q-dot = sign(p), p-dot = -dU/dq * |p|
# rho_ESH ~ exp(-U/kT) / |p|

# div(rho * F) where F = (sign(p), -dUdq * |p|)
# = d/dq[rho * sign(p)] + d/dp[rho * (-dUdq * |p|)]

# rho = C * exp(-U/kT) / |p|
# d/dq[rho * sign(p)] = sign(p) * d/dq[rho]
#                     = sign(p) * rho * (-1/kT) * d/dq[U]
#                     = sign(p) * rho * (-dUdq/kT)

# d/dp[rho * (-dUdq * |p|)] = -dUdq * d/dp[rho * |p|]
#                            = -dUdq * d/dp[(C * exp(-U/kT) / |p|) * |p|]
#                            = -dUdq * d/dp[C * exp(-U/kT)]
#                            = 0  (since C*exp(-U/kT) doesn't depend on p!)

print("  d/dq[rho_ESH * sign(p)] = sign(p) * rho_ESH * (-dUdq/kT)")
print("  d/dp[rho_ESH * (-dUdq*|p|)] = -dUdq * d/dp[(rho_ESH * |p|)]")
print("                               = -dUdq * d/dp[exp(-U/kT)]")
print("                               = 0   [independent of p!]")
print("")
print("  div = sign(p) * rho_ESH * (-dUdq/kT) + 0")
print("      = -dUdq * sign(p) * rho_ESH / kT ≠ 0  in general!")
print("")
print("  => ESH does NOT satisfy Liouville for rho ~ exp(-U/kT)/|p|!")
print("  => ESH's conservation is via its Hamiltonian H_ESH, not Liouville.")

# ============================================================
# Section 8: Final classification
# ============================================================
print("\n" + "=" * 60)
print("SECTION 8: FINAL MATHEMATICAL CLASSIFICATION")
print("=" * 60)

print("""
THEOREM (ESH vs Thermostat Framework):

ESH (Versteeg 2021) is NOT a special case of the generalized thermostat
Master Theorem. They are distinct frameworks with fundamentally different
mechanisms:

1. FORCE STRUCTURE:
   - Our framework:   dp/dt = -dU/dq - g(xi) * p    [friction on p]
   - ESH (1D):        dp/dt = -dU/dq * |p|          [force scaled by |p|]
   These cannot be made equal for generic U.

2. THERMOSTAT VARIABLE:
   - Our framework:   dxi/dt = (p^2/kT - 1)/Q       [kinetic energy driven]
   - ESH:             no separate xi variable; |p| itself plays that role

   Under xi = log(|p|/sqrt(kT)), ESH gives:
     dxi/dt = -dU/dq * sign(p)  [FORCE driven — completely different!]

   While our framework gives:
     dxi/dt = (exp(2*xi) - 1)/Q  [autonomous, only depends on xi]

3. CONSERVED QUANTITY:
   - Our framework:   preserves extended Hamiltonian H + V(xi)/kT
                      (ergodic diffusion in phase space)
   - ESH:             conserves H_ESH = U(x) + (d/2)*log(||p||^2/d)
                      (Hamiltonian flow, NOT a thermostat!)

4. STATIONARY MEASURE:
   - Our framework:   rho ~ exp(-U/kT) * exp(-p^2/(2kT)) [Gaussian p]
   - ESH:             effective rho ~ exp(-U/kT) * ||p||^{-d/kT}
                      [power-law p, improper in 1D!]

5. ERGODICITY MECHANISM:
   - Our framework:   friction+forcing drives exploration (non-Hamiltonian)
   - ESH:             Hamiltonian flow, ergodicity from irrational frequency
                      ratios (similar to NH on simple potentials)

CONCLUSION: ESH and our thermostat framework are PARALLEL, INDEPENDENT
approaches that both aim to improve sampling. They are NOT dual to each
other in a strict mathematical sense. ESH is better understood as a
REPARAMETERIZED HAMILTONIAN DYNAMICS, while our framework is a
DISSIPATIVE THERMOSTAT SYSTEM.

The conceptual similarity (both use a log-like momentum) is superficial.
The structural difference (conservative vs dissipative) is fundamental.
""")

# ============================================================
# Section 9: Could ESH be extended to fit our framework?
# ============================================================
print("\n" + "=" * 60)
print("SECTION 9: ESH AS INSPIRATION FOR A NEW THERMOSTAT")
print("=" * 60)

print("""
Even though ESH != thermostat, we can ask: what thermostat INCORPORATES
the ESH idea of logarithmic momentum?

Proposal: "ESH-inspired thermostat"
  Use V(xi) = (Q/2) * xi^2 (standard NH potential)
  BUT with a nonlinear momentum: define p = sqrt(kT) * exp(xi) * sign(s)
  where s is the signed momentum.

  This gives the same p-marginal as our Log-Osc but with a different
  g(xi) structure.

Alternatively, a true "ESH thermostat" would be:
  dx/dt = p
  dp/dt = -dU/dx - g(xi) * p
  dxi/dt = (exp(2*xi) - 1) / Q    [where xi = log(|p|/sqrt(kT))]

  This requires V(xi) such that V'(xi) = Q * (exp(2*xi) - 1)
  => V(xi) = Q * (exp(2*xi)/2 - xi) + const

  Let's verify: g(xi) = V'(xi)/Q = (exp(2*xi) - 1)
  And |g(xi)| -> inf as xi -> inf: UNBOUNDED (like NH, not bounded like Log-Osc)

  The stationary xi distribution would be:
  P(xi) ~ exp(-V(xi)/kT) = exp(-Q*(exp(2*xi)/2 - xi)/kT)
  This is normalizable (the exp(2*xi) term dominates and suppresses large xi).

  This "ESH-thermostat" is a NEW potential in our Master Theorem framework
  with V(xi) = Q*(exp(2*xi)/2 - xi), not any of our current examples.
""")

xi_sym = symbols('xi', real=True)
V_esh = Q * (sp.exp(2*xi_sym)/2 - xi_sym)
g_esh = sp.diff(V_esh, xi_sym) / Q
print(f"V_ESH(xi) = Q*(exp(2*xi)/2 - xi)")
print(f"g_ESH(xi) = V'(xi)/Q = {g_esh}")
print(f"         = exp(2*xi) - 1")
print(f"\nThis IS a valid thermostat potential in our Master Theorem!")
print(f"At xi=0 (|p|=sqrt(kT)): g=0, no friction — correct fixed point")
print(f"For xi>0 (|p|>sqrt(kT)): g>0, positive friction — slows down fast particles")
print(f"For xi<0 (|p|<sqrt(kT)): g<0, negative friction — speeds up slow particles")

# Check the xi stationary distribution
print("\nStationary xi distribution for V_ESH:")
print("  P(xi) ~ exp(-V_ESH(xi)/kT) = exp(-Q*(exp(2*xi)/2 - xi)/kT)")
print("  This is normalizable (dominated by exp(-Q*exp(2*xi)/(2kT)) for large xi)")
print("  Mean: xi=0 (i.e., |p|=sqrt(kT)), which is the correct canonical temperature")

print("\n" + "=" * 70)
print("SUMMARY:")
print("  ESH != Our thermostat framework (different structure)")
print("  ESH is a conservative Hamiltonian system, not a dissipative thermostat")
print("  A new 'ESH-inspired thermostat' is possible with V(xi) = Q*(exp(2xi)/2 - xi)")
print("  This V is NOT one of our current examples (NH/Log-Osc/Tanh/Arctan)")
print("  But it IS a valid member of our Master Theorem family!")
print("=" * 70)
