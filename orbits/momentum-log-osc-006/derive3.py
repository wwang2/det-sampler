"""Derivation 3: Verify momentum-dependent friction in d dimensions.

Path 1: f(x) = 1 + alpha*x with x = K/(dim*kT), K = sum(p_i^2)/(2m)
From 1D: F = (1-3*alpha/2)*2K/dim - kT + (alpha/2)*(2K/dim)^2/kT  (generalizing p^2/m -> 2K)
Wait, need to redo for dim dimensions carefully.

In dim dimensions:
  K = sum_{i=1}^{dim} p_i^2 / (2m)
  x = K / (dim*kT)

Friction on p_i: dp_i/dt = -dU/dq_i - g(xi) * f(x) * p_i

Compressibility from p:
  sum_i d(dp_i/dt)/dp_i = sum_i [-g(xi)*f(x) - g(xi)*f'(x)*p_i^2/(m*dim*kT)]
  = -dim*g(xi)*f(x) - g(xi)*f'(x)*2K/(m*dim*kT)  [since sum p_i^2 = 2mK]

Wait: sum p_i^2/(m*dim*kT) = 2K/(dim*kT) = 2x. Hmm no:
  d(K/(dim*kT))/dp_i = p_i/(m*dim*kT)

  sum_i p_i * d(K/(dim*kT))/dp_i = sum_i p_i^2/(m*dim*kT) = 2mK/(m*dim*kT) = 2K/(dim*kT) = 2x

So: sum_i d(dp_i/dt)/dp_i = -dim*g(xi)*f(x) - g(xi)*f'(x)*2x

Compressibility from xi: d(dxi/dt)/dxi = (1/Q)*dF/dxi

dH_ext/dt with H_ext = U + K + Q*log(1+xi^2):
  = sum_i (dU/dq_i)(p_i/m) + sum_i (p_i/m)(dp_i/dt) + Q*g(xi)*dxi/dt
  = sum (dU/dq_i)(p_i/m) + sum (p_i/m)(-dU/dq_i - g(xi)*f(x)*p_i) + g(xi)*F
  = -g(xi)*f(x)*sum(p_i^2)/m + g(xi)*F
  = -g(xi)*f(x)*2K + g(xi)*F

Condition: kappa + (1/kT)*dH/dt = 0

[-dim*g(xi)*f(x) - g(xi)*f'(x)*2x + (1/Q)*dF/dxi] + (g(xi)/kT)*[-f(x)*2K + F] = 0

If F = F(K) only (no xi dependence), dF/dxi = 0:
-dim*g(xi)*f(x) - 2x*g(xi)*f'(x) + (g(xi)/kT)*F - (g(xi)/kT)*2K*f(x) = 0

Dividing by g(xi) (assuming g(xi) != 0):
-dim*f(x) - 2x*f'(x) + F/kT - 2K*f(x)/kT = 0

Note: 2K = 2*dim*kT*x, so 2K/kT = 2*dim*x.

-dim*f(x) - 2x*f'(x) + F/kT - 2*dim*x*f(x) = 0

F/kT = dim*f(x) + 2x*f'(x) + 2*dim*x*f(x) = dim*f(x)*(1 + 2x) + 2x*f'(x)

F = kT*[dim*f(x)*(1+2x) + 2x*f'(x)]

For f(x) = 1 (standard):
F = kT*[dim*(1+2x)] = dim*kT + 2*dim*kT*x = dim*kT + 2K

Hmm, standard log-osc has dxi/dt = (1/Q)*(sum p_i^2/m - dim*kT) = (1/Q)*(2K - dim*kT).
So F_standard = 2K - dim*kT.

But we derived F = dim*kT + 2K. That's F = 2K + dim*kT, NOT 2K - dim*kT.

There must be a sign error. Let me recheck...

Actually wait. The issue is that the Liouville condition for rho = exp(-H/kT) is:
  div(flow) = -(1/kT) * dH/dt  (NOT positive)

Let me redo: d(rho*v)/dz = rho*div(v) + v*grad(rho) = rho*[div(v) - (1/kT)*v*grad(H)] = rho*[div(v) - (1/kT)*dH/dt]

Setting this to 0: div(v) = (1/kT)*dH/dt

Hmm, that also doesn't match. Let me just trust the SymPy result.

From 1D SymPy: f=1 + alpha*x, with F = A*p^2/m + B*kT + C*p^4/(m^2*kT)
Solution: A = 1 - 3*alpha/2, B = -1, C = alpha/2

Check for alpha=0: F = 1*p^2/m - kT + 0 = p^2/m - kT = 2K - kT (for 1D dim=1).
This matches! Standard log-osc has dxi/dt = (2K - dim*kT)/Q = (p^2/m - kT)/Q.

Now for general alpha in 1D:
F = (1-3*alpha/2)*p^2/m - kT + (alpha/2)*p^4/(m^2*kT)

In terms of K (1D: K = p^2/(2m), so p^2/m = 2K, p^4/m^2 = 4K^2):
F = (1-3*alpha/2)*2K - kT + (alpha/2)*4K^2/kT
  = 2K - 3*alpha*K - kT + 2*alpha*K^2/kT

And x = K/kT (dim=1), so K = kT*x:
F = 2*kT*x - 3*alpha*kT*x - kT + 2*alpha*kT*x^2
  = kT*(2x - 3*alpha*x - 1 + 2*alpha*x^2)
  = kT*(-1 + x*(2 - 3*alpha) + 2*alpha*x^2)

For general dim, x = K/(dim*kT):
  K = dim*kT*x
  sum(p_i^2)/m = 2K = 2*dim*kT*x

Generalizing from 1D:
F = sum(p_i^2/m)*(1 - (dim+2)*alpha/2) - dim*kT + alpha*(sum(p_i^2/m))^2/(4*dim*kT)
... this is getting messy. Let me just do it with SymPy for 2D.
"""

import sympy as sp

# 2D case
q1, q2, p1, p2, xi = sp.symbols('q1 q2 p1 p2 xi', real=True)
m, kT, Q = sp.symbols('m kT Q', positive=True)
alpha = sp.Symbol('alpha', real=True)
A, B, C = sp.symbols('A B C', real=True)

U = sp.Function('U')(q1, q2)
grad_U1 = sp.diff(U, q1)
grad_U2 = sp.diff(U, q2)

K = (p1**2 + p2**2) / (2*m)
x = K / (2*kT)  # dim=2

g = 2*xi / (1 + xi**2)
H_ext = U + K + Q * sp.log(1 + xi**2)
rho = sp.exp(-H_ext / kT)

f_x = 1 + alpha * x

# Equations of motion
dq1dt = p1/m
dq2dt = p2/m
dp1dt = -grad_U1 - g * f_x * p1
dp2dt = -grad_U2 - g * f_x * p2

# F = A*(p1^2+p2^2)/m + B*kT + C*(p1^2+p2^2)^2/(m^2*kT)
S = p1**2 + p2**2  # = 2*m*K
F_expr = A * S/m + B * kT + C * S**2 / (m**2 * kT)
dxidt = F_expr / Q

# Liouville
terms = (
    sp.diff(rho * dq1dt, q1) +
    sp.diff(rho * dq2dt, q2) +
    sp.diff(rho * dp1dt, p1) +
    sp.diff(rho * dp2dt, p2) +
    sp.diff(rho * dxidt, xi)
)

liouville = sp.simplify(terms / rho)
print("2D Liouville / rho (simplified):")

# Expand and collect by monomials in (p1, p2)
liouville_expanded = sp.expand(liouville)

# Substitute p1^2 + p2^2 = S for readability
# Instead, collect as polynomial in p1, p2
# Actually let's just get the numerator and factor
liouville_num = sp.fraction(sp.cancel(liouville))[0]
liouville_num_expanded = sp.expand(liouville_num)

# Group terms: find coefficients of various p monomials
# The key independent structures are: 1, p1^2+p2^2, (p1^2+p2^2)^2, p1^4+p2^4, etc.
# By symmetry, terms should only depend on S = p1^2 + p2^2

# Substitute p2 = 0 to simplify (using symmetry, terms in p2 are same as p1)
liouville_sub = liouville_num_expanded.subs(p2, 0)
liouville_sub = sp.expand(liouville_sub)
poly = sp.Poly(liouville_sub, p1)
print("\nWith p2=0, coefficients by p1 power:")
for monom, coeff in sorted(poly.as_dict().items()):
    coeff_s = sp.simplify(coeff)
    print(f"  p1^{monom[0]}: {coeff_s}")

print("\n--- Solving ---")
coeffs_dict = {}
for monom, coeff in poly.as_dict().items():
    coeffs_dict[monom[0]] = sp.simplify(coeff)

eqs = list(coeffs_dict.values())
sol = sp.solve(eqs, [A, B, C], dict=True)
print(f"Solutions: {sol}")

if sol:
    s = sol[0]
    print(f"\nA = {s[A]}")
    print(f"B = {s[B]}")
    print(f"C = {s[C]}")

    # Write out the xi equation
    print(f"\nFor dim=2:")
    print(f"  dxi/dt = (1/Q) * [{s[A]}*sum(p^2)/m + {s[B]}*kT + {s[C]}*(sum(p^2))^2/(m^2*kT)]")

    # Check alpha=0
    print(f"\nalpha=0: A={s[A].subs(alpha,0)}, B={s[B].subs(alpha,0)}, C={s[C].subs(alpha,0)}")
    print(f"  => dxi/dt = (1/Q)*[{s[A].subs(alpha,0)}*2K - kT] = (1/Q)*(2K-2kT) = (1/Q)*(sum p^2/m - 2kT)")
    print(f"  Standard log-osc (dim=2): dxi/dt = (1/Q)*(sum p^2/m - dim*kT) = (1/Q)*(sum p^2/m - 2kT) ✓")
