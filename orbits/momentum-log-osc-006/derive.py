"""Derivation of invariant measure for momentum-dependent log-osc thermostat.

We verify the Liouville equation condition for the extended system.

For a general thermostat dynamics with state (q, p, xi):
    dq_i/dt = v_i(q, p, xi)
    dp_i/dt = f_i(q, p, xi)
    dxi/dt  = h(q, p, xi)

The invariant measure rho(q,p,xi) satisfies the continuity equation:
    sum_i d(rho*v_i)/dq_i + sum_i d(rho*f_i)/dp_i + d(rho*h)/dxi = 0

If rho = Z^{-1} * exp(-H_ext/kT), this becomes:
    rho * [kappa + (1/kT) * dH_ext/dt] = 0

where kappa = div(v) is the phase-space compressibility (divergence of the flow)
and dH_ext/dt is the time derivative of H_ext along the flow.

So the condition is: kappa = -(1/kT) * dH_ext/dt

=== Approach B: Adaptive friction strength ===

Equations:
    dq_i/dt = p_i/m
    dp_i/dt = -dU/dq_i - g(xi) * f(K/(dim*kT)) * p_i
    dxi/dt  = (1/Q) * F(K, xi)

where:
    K = sum(p_i^2) / (2m)
    g(xi) = 2*xi / (1 + xi^2)  [log-osc friction]
    f(x) = some function of normalized kinetic energy

We need to find F(K, xi) such that the invariant measure is canonical in (q,p).

Let's compute kappa (phase-space compressibility):
    d(dq_i/dt)/dq_i = 0  (for all i)
    d(dp_i/dt)/dp_i = -g(xi)*f(K/(dim*kT)) - g(xi)*f'(K/(dim*kT)) * (p_i^2/(m*dim*kT))
                     ... wait, need chain rule through K

Actually d(dp_i/dt)/dp_i needs:
    dp_i/dt = -dU/dq_i - g(xi) * f(K/(dim*kT)) * p_i

    d/dp_i [-g(xi)*f(K/(dim*kT))*p_i]
    = -g(xi) * [f(K/(dim*kT)) + p_i * f'(K/(dim*kT)) * d(K/(dim*kT))/dp_i]
    = -g(xi) * [f(K/(dim*kT)) + p_i * f'(K/(dim*kT)) * p_i/(m*dim*kT)]
    = -g(xi) * [f(K/(dim*kT)) + f'(K/(dim*kT)) * p_i^2/(m*dim*kT)]

Summing over all i:
    sum_i d(dp_i/dt)/dp_i = -g(xi) * [dim*f(K/(dim*kT)) + f'(K/(dim*kT)) * sum(p_i^2)/(m*dim*kT)]
    = -g(xi) * [dim*f(K/(dim*kT)) + f'(K/(dim*kT)) * 2K/(dim*kT)]

And d(dxi/dt)/dxi = (1/Q) * dF/dxi.

So: kappa = -g(xi) * [dim*f(x) + 2*x*f'(x)] + (1/Q)*dF/dxi
where x = K/(dim*kT).

Now compute dH_ext/dt where H_ext = U(q) + K(p) + Q*log(1+xi^2):

dU/dt = sum dU/dq_i * dq_i/dt = sum grad_U_i * p_i/m
dK/dt = sum p_i/m * dp_i/dt = sum p_i/m * [-dU/dq_i - g(xi)*f(x)*p_i]
      = -sum grad_U_i * p_i/m - g(xi)*f(x) * sum(p_i^2)/m
      = -dU/dt - g(xi)*f(x)*2K/m ... wait

Actually K = sum(p_i^2)/(2m), so dK/dt = sum(p_i*dp_i/dt)/m
= sum p_i/m * [-dU/dq_i - g(xi)*f(x)*p_i]
= -sum grad_U_i * p_i/m - g(xi)*f(x)*sum(p_i^2)/m
= -dU/dt - 2*K*g(xi)*f(x)/m  ... no

sum(p_i^2)/m = 2K. So:
dK/dt = -dU/dt - g(xi)*f(x)*2K

dH_xi/dt = d[Q*log(1+xi^2)]/dt = Q * 2*xi/(1+xi^2) * dxi/dt = g(xi)*Q*dxi/dt
         = g(xi) * F(K, xi)

Total: dH_ext/dt = dU/dt + dK/dt + dH_xi/dt
     = dU/dt + (-dU/dt - 2K*g(xi)*f(x)) + g(xi)*F(K,xi)
     = -2K*g(xi)*f(x) + g(xi)*F(K,xi)
     = g(xi) * [F(K,xi) - 2K*f(x)]

Condition: kappa = -(1/kT) * dH_ext/dt

-g(xi)*[dim*f(x) + 2x*f'(x)] + (1/Q)*dF/dxi = -(1/kT)*g(xi)*[F - 2Kf(x)]

Note 2K = 2*dim*kT*x, so 2Kf(x) = 2*dim*kT*x*f(x).

-(1/kT)*g(xi)*[F - 2*dim*kT*x*f(x)] = -(g(xi)/kT)*F + g(xi)*2*dim*x*f(x)

So the condition becomes:
-g(xi)*dim*f(x) - g(xi)*2x*f'(x) + (1/Q)*dF/dxi = -(g(xi)/kT)*F + 2*dim*x*g(xi)*f(x)

Rearranging:
(1/Q)*dF/dxi = -(g(xi)/kT)*F + 2*dim*x*g(xi)*f(x) + g(xi)*dim*f(x) + g(xi)*2x*f'(x)
(1/Q)*dF/dxi = g(xi)*[-(1/kT)*F + dim*f(x)*(1 + 2x) + 2x*f'(x)]

This is a PDE in F(K, xi). If we try the ansatz F(K, xi) = A(K) (independent of xi):
(1/Q)*0 = g(xi) * [-(1/kT)*A(K) + dim*f(x)*(1+2x) + 2x*f'(x)]

This requires: A(K) = kT*[dim*f(x)*(1+2x) + 2x*f'(x)]

For f(x) = 1 (standard log-osc): A = kT*[dim*(1+2x) + 0] = kT*dim*(1+2x) = dim*kT + 2*dim*kT*x = dim*kT + 2K
So dxi/dt = (1/Q)*(dim*kT + 2K) -- this is NOT the standard log-osc equation!

Wait, let me recheck. For standard log-osc, f(x) = 1:
    dp/dt = -dU/dq - g(xi)*p
    dxi/dt = (1/Q)*(K - dim*kT)  [from the original paper]

Let me recheck the Liouville condition for f=1.
kappa = -g(xi)*dim  [since f'=0, f=1]
dH_ext/dt = g(xi)*[F - 2K]

Condition: -g(xi)*dim = -(g(xi)/kT)*[F - 2K]
=> dim = (1/kT)*(F - 2K)
=> F = dim*kT + 2K

But the original log-osc uses F = K - dim*kT = (sum p^2/m) - dim*kT = 2K - dim*kT.

Hmm, that gives F - 2K = -dim*kT, so -(g(xi)/kT)*(F-2K) = -(g(xi)/kT)*(-dim*kT) = g(xi)*dim.
And kappa = -g(xi)*dim.
So condition: -g(xi)*dim = -g(xi)*dim. YES! It checks out.

Wait, I made an error above. Let me redo:
kappa + (1/kT)*dH_ext/dt = 0
kappa = -g(xi)*dim
dH_ext/dt = g(xi)*(F - 2K*1) = g(xi)*(F - 2K)
(1/kT)*dH_ext/dt = (g(xi)/kT)*(F - 2K)

kappa + (1/kT)*dH_ext/dt = -g(xi)*dim + (g(xi)/kT)*(F - 2K) = 0
=> (F - 2K)/kT = dim
=> F = 2K + dim*kT

But the original uses F = (sum p^2/m) - dim*kT.
sum(p^2/m) = sum(p^2)/(m) and K = sum(p^2)/(2m), so sum(p^2/m) = 2K.
So F = 2K - dim*kT.

Then: -dim + (1/kT)*(2K - dim*kT - 2K) = -dim + (1/kT)*(-dim*kT) = -dim - dim = -2*dim != 0

PROBLEM! Let me recheck more carefully...
"""

import sympy as sp

# Let's verify the standard log-osc invariant measure with SymPy
# We work in 1D for simplicity, then generalize

# Variables
q, p, xi = sp.symbols('q p xi', real=True)
m, kT, Q, omega = sp.symbols('m kT Q omega', positive=True)
dim = sp.Symbol('dim', positive=True, integer=True)

# Potential (generic -- we just need grad_U)
U = sp.Function('U')(q)
grad_U = sp.diff(U, q)

# Kinetic energy
K_expr = p**2 / (2*m)

# g function
g = 2*xi / (1 + xi**2)

# Extended Hamiltonian
H_ext = U + K_expr + Q * sp.log(1 + xi**2)

# Invariant density
rho = sp.exp(-H_ext / kT)

# Equations of motion (standard log-osc, 1D)
dqdt = p / m
dpdt = -grad_U - g * p
dxidt_expr = (p**2/m - kT) / Q  # = (2K - kT) / Q for dim=1

# Liouville equation: div(rho * v) = 0
# = d(rho*dqdt)/dq + d(rho*dpdt)/dp + d(rho*dxidt)/dxi

term_q = sp.diff(rho * dqdt, q)
term_p = sp.diff(rho * dpdt, p)
term_xi = sp.diff(rho * dxidt_expr, xi)

liouville = sp.simplify(term_q + term_p + term_xi)
print("=== Standard Log-Osc (1D) ===")
print(f"Liouville residual (should be 0): {liouville}")

# Factor out rho
liouville_over_rho = sp.simplify(liouville / rho)
print(f"Residual / rho: {liouville_over_rho}")

print("\n=== Now try with f(x) multiplier ===")
# dp/dt = -grad_U - g(xi) * f(K/(dim*kT)) * p
# For 1D: x = p^2/(2*m*kT)
x = p**2 / (2*m*kT)

# Try f(x) = sqrt(x) first
# f_x = sp.sqrt(x)
# Try f(x) = x (linear in normalized KE)
# f_x = x

# General f
f = sp.Function('f')
f_x = f(x)

dpdt_mod = -grad_U - g * f_x * p

# We need to find dxidt such that Liouville = 0
# Let dxidt = (1/Q) * F(p, xi)
F_func = sp.Function('F')
F_val = F_func(p, xi)
dxidt_mod = F_val / Q

term_q_mod = sp.diff(rho * dqdt, q)
term_p_mod = sp.diff(rho * dpdt_mod, p)
term_xi_mod = sp.diff(rho * dxidt_mod, xi)

liouville_mod = term_q_mod + term_p_mod + term_xi_mod

# Simplify
liouville_mod_simplified = sp.simplify(liouville_mod / rho)
print(f"Modified Liouville / rho: {liouville_mod_simplified}")

# Let's try specific f and specific F to see what works
print("\n=== Trying specific forms ===")

# f(x) = 1 + alpha*x, F = A*p^2/m + B*kT  (affine in K)
alpha_param = sp.Symbol('alpha', real=True)
A_param, B_param = sp.symbols('A B', real=True)

f_specific = 1 + alpha_param * x
F_specific = A_param * p**2/m + B_param * kT

dpdt_spec = -grad_U - g * f_specific * p
dxidt_spec = F_specific / Q

term_q_s = sp.diff(rho * dqdt, q)
term_p_s = sp.diff(rho * dpdt_spec, p)
term_xi_s = sp.diff(rho * dxidt_spec, xi)

liouville_spec = sp.simplify((term_q_s + term_p_s + term_xi_s) / rho)
print(f"Liouville/rho with f=1+alpha*x, F=A*p^2/m+B*kT:")
print(f"  {liouville_spec}")

# Collect terms by powers of p
liouville_expanded = sp.expand(liouville_spec)
print(f"\nExpanded: {liouville_expanded}")

# Try to solve for A, B in terms of alpha
# Group by powers of p
liouville_poly = sp.Poly(liouville_expanded, p)
print(f"\nAs polynomial in p:")
for monom, coeff in liouville_poly.as_dict().items():
    print(f"  p^{monom}: {sp.simplify(coeff)}")
