"""Symbolic derivation and verification of the logarithmic oscillator thermostat.

We want to find equations of motion for (q, p, xi) such that:
1. The extended Hamiltonian H_ext = U(q) + p^2/(2m) + Q*log(1 + xi^2)
   defines the invariant measure rho ~ exp(-H_ext / kT).
2. The flow preserves this measure: div(rho * v) = 0,
   equivalently: div(v) + v . grad(log rho) = 0  (Liouville equation).

Approach: The standard Nose-Hoover equations come from requiring this
divergence condition with the quadratic thermostat potential Q*xi^2/2.
We generalize by replacing g(xi) = d/dxi [Q*xi^2/2] / Q = xi
with g(xi) = d/dxi [Q*log(1+xi^2)] / Q = 2*xi/(1+xi^2).

The key insight: the NH equations can be written in a generalized form:
    dq/dt = p/m
    dp/dt = -dU/dq - alpha(xi) * p
    dxi/dt = (1/Q) * beta(xi) * (p^2/m - dim*kT)

For the standard NH: alpha(xi) = xi, beta(xi) = 1
For the log oscillator, we need to find alpha(xi) and beta(xi).

The Liouville condition for the invariant measure exp(-H_ext/kT) is:
    sum_i d(v_i)/d(x_i) = sum_i v_i * d(H_ext)/d(x_i) / kT

Let's verify this symbolically.
"""

import sympy as sp

# Symbols
q, p, xi, kT, m, Q_param = sp.symbols('q p xi kT m Q', positive=True)
dim_sym = sp.Symbol('N', positive=True, integer=True)  # number of DOF

# Thermostat potential: Q * log(1 + xi^2)
V_therm = Q_param * sp.log(1 + xi**2)
dV_dxi = sp.diff(V_therm, xi)  # = 2*Q*xi / (1 + xi^2)

print("Thermostat potential V(xi) =", V_therm)
print("dV/dxi =", dV_dxi)
print()

# Extended Hamiltonian (1D for simplicity, generalize by replacing p^2/m -> dim*kT in thermostat eqn)
H_ext = p**2 / (2*m) + Q_param * sp.log(1 + xi**2)  # U(q) term handled generically

# The invariant density: rho ~ exp(-H_ext / kT)
# log(rho) = -H_ext/kT + const
# d(log rho)/dp = -p/(m*kT)
# d(log rho)/dxi = -dV_dxi/kT = -2*Q*xi/((1+xi^2)*kT)

dlogrho_dp = -p / (m * kT)
dlogrho_dxi = -dV_dxi / kT

print("d(log rho)/dp =", dlogrho_dp)
print("d(log rho)/dxi =", dlogrho_dxi)
print()

# ============================================================
# ATTEMPT 1: Direct generalization
# ============================================================
# dq/dt = p/m
# dp/dt = -dU/dq - alpha(xi) * p
# dxi/dt = beta(xi) * (p^2/m - dim*kT) / Q
#
# We need: div(v) + v . grad(log rho) = 0
# where the gradient is w.r.t. (q, p, xi)

print("=" * 60)
print("ATTEMPT 1: General alpha(xi), beta(xi)")
print("=" * 60)

alpha = sp.Function('alpha')(xi)
beta = sp.Function('beta')(xi)

# Flow components (1D case, N=1):
v_q = p / m
v_p = -alpha * p  # force term cancels in divergence since d(-dU/dq)/dp = 0
v_xi = beta * (p**2 / m - dim_sym * kT) / Q_param

# Divergence components
div_q = sp.diff(v_q, q)   # = 0 (no q dependence in p/m)
# d(v_p)/dp: d(-alpha*p)/dp = -alpha  (alpha depends on xi, not p)
div_p = -alpha
# d(v_xi)/dxi
div_xi = sp.diff(v_xi, xi)

print("div_q =", div_q)
print("div_p =", div_p)
print("div_xi =", div_xi)
print()

# v . grad(log rho):
# v_q * d(log rho)/dq  = (p/m) * (-dU/dq / kT)  -- we don't track U explicitly
# v_p * d(log rho)/dp  = (-alpha*p) * (-p/(m*kT)) = alpha * p^2 / (m*kT)
# BUT we also need to account for the -dU/dq term in dp/dt:
#   Full v_p = -dU/dq - alpha*p
#   v_p * d(log rho)/dp = (-dU/dq - alpha*p) * (-p/(m*kT))
#                       = p*dU/dq/(m*kT) + alpha*p^2/(m*kT)
# And v_q * d(log rho)/dq = (p/m) * (-dU/dq/kT) = -p*dU/dq/(m*kT)
# These cancel! So:
# v_q * dlogrho_dq + v_p * dlogrho_dp = alpha * p^2 / (m * kT)

# Plus the thermostat part:
# v_xi * d(log rho)/dxi

vdot_logrho_phys = alpha * p**2 / (m * kT)  # from (q,p) sector
vdot_logrho_xi = v_xi * dlogrho_dxi

print("v.grad(log rho) from (q,p) sector:", vdot_logrho_phys)
print("v.grad(log rho) from xi sector:", sp.simplify(vdot_logrho_xi))
print()

# Full Liouville condition: div(v) + v.grad(log rho) = 0
# For N-dimensional system, the (q,p) divergence gives:
#   div_p contributes: -N * alpha  (sum over N DOF, each gives -alpha)
#   v.grad(log rho) from physical: alpha * (sum p_i^2/m) / kT = alpha * K / kT
#                                  where K = sum p_i^2 / m

# In 1D: K = p^2/m
# Liouville = -alpha + alpha*p^2/(m*kT) + div_xi + v_xi * dlogrho_dxi = 0
# For general N dimensions:
# Liouville = -N*alpha + alpha*(K/kT) + d(v_xi)/dxi + v_xi * dlogrho_dxi = 0

# Let's compute the xi-sector terms:
# v_xi = beta * (K - N*kT) / Q
# d(v_xi)/dxi = beta' * (K - N*kT) / Q   (since K doesn't depend on xi)
# v_xi * dlogrho_dxi = beta * (K - N*kT) / Q * (-2*Q*xi / ((1+xi^2)*kT))
#                     = -2*beta*xi*(K - N*kT) / ((1+xi^2)*kT)

# So full Liouville:
# -N*alpha + alpha*K/kT + beta'*(K-N*kT)/Q - 2*beta*xi*(K-N*kT)/((1+xi^2)*kT) = 0

# Rearrange:
# alpha * (K/kT - N) + (K-N*kT) * [beta'/Q - 2*beta*xi/((1+xi^2)*kT)] = 0
# alpha * (K - N*kT)/kT + (K-N*kT) * [beta'/Q - 2*beta*xi/((1+xi^2)*kT)] = 0
# (K - N*kT) * [alpha/kT + beta'/Q - 2*beta*xi/((1+xi^2)*kT)] = 0

# This must hold for ALL values of K (i.e., for all p), so the bracket must be zero:
# alpha/kT + beta'/Q - 2*beta*xi/((1+xi^2)*kT) = 0

print("Liouville condition (must hold for all p):")
print("alpha/kT + beta'/Q - 2*beta*xi/((1+xi^2)*kT) = 0")
print()

# ============================================================
# SOLUTION: Choose beta(xi) = 1 (simplest)
# ============================================================
print("=" * 60)
print("SOLUTION 1: beta(xi) = 1  (constant)")
print("=" * 60)
# Then beta' = 0, and:
# alpha/kT = 2*xi/((1+xi^2)*kT)
# => alpha(xi) = 2*xi/(1+xi^2)
alpha_sol1 = 2*xi / (1 + xi**2)
print(f"alpha(xi) = {alpha_sol1}")
print("Equations of motion:")
print("  dq/dt = p/m")
print("  dp/dt = -dU/dq - [2*xi/(1+xi^2)] * p")
print("  dxi/dt = (1/Q) * (p^2/m - N*kT)")
print()

# Verify the Liouville condition:
beta_val = sp.Integer(1)
beta_prime = sp.Integer(0)
alpha_val = 2*xi / (1 + xi**2)
check1 = alpha_val/kT + beta_prime/Q_param - 2*beta_val*xi/((1+xi**2)*kT)
print(f"Liouville check: {sp.simplify(check1)} = 0  ✓")
print()

# ============================================================
# SOLUTION: Choose alpha(xi) = xi (like standard NH)
# ============================================================
print("=" * 60)
print("SOLUTION 2: alpha(xi) = xi  (NH-like friction)")
print("=" * 60)
# Then: xi/kT + beta'/Q = 2*beta*xi/((1+xi^2)*kT)
# => beta' = Q * [2*beta*xi/((1+xi^2)*kT) - xi/kT]
# => beta' = (Q*xi/kT) * [2*beta/(1+xi^2) - 1]
# This is a 1st order ODE for beta(xi). Not as clean.
print("Leads to ODE for beta -- less elegant, skip.")
print()

# ============================================================
# SOLUTION: Choose alpha = g(xi) = dV/dxi / Q, beta = g(xi)
# ============================================================
print("=" * 60)
print("SOLUTION 3: alpha = beta = g(xi) = 2*xi/(1+xi^2)")
print("=" * 60)
g = 2*xi / (1 + xi**2)
g_prime = sp.diff(g, xi)
print(f"g(xi) = {g}")
print(f"g'(xi) = {sp.simplify(g_prime)}")

check3 = g/kT + g_prime/Q_param - 2*g*xi/((1+xi**2)*kT)
check3_simplified = sp.simplify(check3)
print(f"Liouville check: {check3_simplified}")
# This equals g'(xi)/Q, which is NOT zero in general. So this doesn't work unless g'=0.
print("This does NOT satisfy Liouville unless g'(xi)/Q = 0. Rejected.")
print()

# ============================================================
# FINAL: Solution 1 is the cleanest and correct
# ============================================================
print("=" * 60)
print("FINAL VERIFIED EQUATIONS (Solution 1)")
print("=" * 60)
print()
print("Extended Hamiltonian:")
print("  H_ext = U(q) + p^2/(2m) + Q*log(1 + xi^2)")
print()
print("Equations of motion:")
print("  dq/dt = p/m")
print("  dp/dt = -dU/dq - g(xi) * p,  where g(xi) = 2*xi/(1+xi^2)")
print("  dxi/dt = (1/Q) * (sum_i p_i^2/m_i - N*kT)")
print()
print("Invariant measure: rho(q,p,xi) ~ exp(-H_ext/kT)")
print()
print("Key properties:")
print("  - g(xi) is bounded: |g(xi)| <= 1 for all xi")
print("  - g(xi) ~ 2*xi for small xi (like 2x the NH coupling)")
print("  - g(xi) ~ 2/xi for large xi (friction vanishes -- anharmonic!)")
print("  - xi equation is IDENTICAL to NH (no change)")
print("  - Only the friction coupling alpha(xi) changes from xi to g(xi)")
print()

# Verify g(xi) properties
print("g(xi) properties:")
print(f"  g(0) = {g.subs(xi, 0)}")
print(f"  g(1) = {g.subs(xi, 1)}")
print(f"  g(oo) = {sp.limit(g, xi, sp.oo)}")
print(f"  max of |g|: at xi=1, g(1) = {g.subs(xi, 1)} = 1")

# Critical points
g_prime_simplified = sp.simplify(g_prime)
print(f"  g'(xi) = {g_prime_simplified}")
crits = sp.solve(g_prime_simplified, xi)
print(f"  Critical points: xi = {crits}")
print(f"  g(1) = {g.subs(xi, 1)}, g(-1) = {g.subs(xi, -1)}")
print()
print("The bounded friction g(xi) prevents the thermostat from creating")
print("excessively strong friction at large xi, which should help break")
print("KAM tori by allowing the system to explore more freely.")

# ============================================================
# Integrator consideration
# ============================================================
print()
print("=" * 60)
print("INTEGRATOR NOTES")
print("=" * 60)
print()
print("The VelocityVerlet integrator uses exp(-xi[0]*dt/2) rescaling.")
print("For our dynamics, the friction is g(xi)*p, not xi*p.")
print("The analytical momentum rescaling should be exp(-g(xi)*dt/2).")
print("Since g(xi) = 2*xi/(1+xi^2), we need a custom integrator that")
print("uses exp(-g(xi)*dt/2) instead of exp(-xi*dt/2).")
print()
print("However, a simpler approach: we can use the default VelocityVerlet")
print("if we define our dpdt to include -g(xi)*p and let the integrator")
print("handle it. But the default integrator does exp(-xi[0]*dt/2) hardcoded.")
print("We need a custom integrator.")
