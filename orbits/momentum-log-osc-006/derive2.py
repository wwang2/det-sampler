"""Derivation attempt 2: Try higher-order F to compensate f(x) = 1 + alpha*x."""

import sympy as sp

q, p, xi = sp.symbols('q p xi', real=True)
m, kT, Q = sp.symbols('m kT Q', positive=True)
alpha_param = sp.Symbol('alpha', real=True)
A, B, C = sp.symbols('A B C', real=True)

U = sp.Function('U')(q)
grad_U = sp.diff(U, q)

K_expr = p**2 / (2*m)
x = K_expr / kT  # = p^2/(2*m*kT) -- normalized KE for dim=1

g = 2*xi / (1 + xi**2)
H_ext = U + K_expr + Q * sp.log(1 + xi**2)
rho = sp.exp(-H_ext / kT)

dqdt = p / m

# f(x) = 1 + alpha*x
f_x = 1 + alpha_param * x

# F with p^4 term
F_specific = A * p**2/m + B * kT + C * p**4 / (m**2 * kT)

dpdt = -grad_U - g * f_x * p
dxidt = F_specific / Q

term_q = sp.diff(rho * dqdt, q)
term_p = sp.diff(rho * dpdt, p)
term_xi = sp.diff(rho * dxidt, xi)

liouville = sp.simplify((term_q + term_p + term_xi) / rho)
print("Liouville/rho:")

# Collect as polynomial in p
liouville_num = sp.fraction(sp.cancel(liouville))[0]
poly = sp.Poly(sp.expand(liouville_num), p)
print("\nCoefficients by power of p:")
for monom, coeff in sorted(poly.as_dict().items()):
    coeff_s = sp.simplify(coeff)
    print(f"  p^{monom[0]}: {coeff_s}")

print("\n--- Solving for A, B, C ---")
coeffs = {}
for monom, coeff in poly.as_dict().items():
    coeffs[monom[0]] = sp.simplify(coeff)

# Set each coefficient to 0
eqs = [coeffs[k] for k in sorted(coeffs.keys())]
print(f"Equations: {eqs}")
sol = sp.solve(eqs, [A, B, C, alpha_param], dict=True)
print(f"Solutions: {sol}")

# Also try: what if we allow alpha to be determined?
# Or: try a different form. f(x) = 1/(1+beta*x) ?
print("\n\n=== Alternative: f(x) = 1/(1 + beta*x) ===")
beta = sp.Symbol('beta', positive=True)
f_alt = 1 / (1 + beta * x)

# F_alt can be general -- let's see what shape is needed
# Start with F = A2*p^2/m + B2*kT
A2, B2 = sp.symbols('A2 B2', real=True)
F_alt = A2 * p**2/m + B2 * kT

dpdt_alt = -grad_U - g * f_alt * p
dxidt_alt = F_alt / Q

term_q_a = sp.diff(rho * dqdt, q)
term_p_a = sp.diff(rho * dpdt_alt, p)
term_xi_a = sp.diff(rho * dxidt_alt, xi)

liouville_alt = sp.simplify((term_q_a + term_p_a + term_xi_a) / rho)

# Collect
liouville_alt_num = sp.fraction(sp.cancel(liouville_alt))[0]
# Need to rationalize -- multiply by denominator of f_alt
expr = sp.cancel(liouville_alt * (1 + beta*x)**2 * (1 + xi**2))
expr_expanded = sp.expand(expr)
poly_alt = sp.Poly(expr_expanded, p)
print("\nCoefficients by power of p (after clearing denominators):")
for monom, coeff in sorted(poly_alt.as_dict().items()):
    coeff_s = sp.simplify(coeff)
    print(f"  p^{monom[0]}: {coeff_s}")


print("\n\n=== Alternative: multiplicative thermostat potential ===")
print("What if H_ext = U + K + Q*V(xi) where V is chosen to match?")
print("Try: dq/dt = p/m, dp/dt = -dU/dq - V'(xi)*p, dxi/dt = (K - kT)/Q")
print("The friction is V'(xi) instead of g(xi) = 2xi/(1+xi^2)")
print("Standard NH: V(xi) = xi^2/2 => V'=xi, OK")
print("Log-osc: V(xi) = log(1+xi^2) => V'=2xi/(1+xi^2), OK")
print()
print("Key: ANY V(xi) gives an exact invariant measure if we use:")
print("  dp/dt = -dU/dq - V'(xi)*p")
print("  dxi/dt = (K - dim*kT/2)/Q ... wait, let me verify this for general V")

# General V(xi)
V = sp.Function('V')(xi)
V_prime = sp.diff(V, xi)

H_gen = U + K_expr + Q * V
rho_gen = sp.exp(-H_gen / kT)

dqdt_gen = p/m
dpdt_gen = -grad_U - V_prime * p
# Standard NH-like xi equation
dxidt_gen = (p**2/m - kT) / Q  # dim=1

term_q_g = sp.diff(rho_gen * dqdt_gen, q)
term_p_g = sp.diff(rho_gen * dpdt_gen, p)
term_xi_g = sp.diff(rho_gen * dxidt_gen, xi)

liouville_gen = sp.simplify((term_q_g + term_p_g + term_xi_g) / rho_gen)
print(f"\nGeneral V(xi) Liouville/rho: {liouville_gen}")

print("\n=== What if friction has BOTH xi and K dependence through V? ===")
print("Try: H_ext = U + K + Q*W(xi, K)")
print("where W couples xi and K explicitly")
print("Then dp/dt needs extra terms from dH_ext/dp...")

# Let's try the cleanest approach:
# H_ext = U + K + Q*log(1 + xi^2) + lambda*xi^2*K  (cross term!)
lam = sp.Symbol('lambda', real=True)
H_cross = U + K_expr + Q * sp.log(1 + xi**2) + lam * xi**2 * K_expr
rho_cross = sp.exp(-H_cross / kT)

# Hamilton's equations:
# dq/dt = dH/dp = p/m + lam*xi^2*p/m = p*(1+lam*xi^2)/m
# dp/dt = -dH/dq = -dU/dq
# dxi/dt = ?? -- need to design this

dqdt_cross = p * (1 + lam*xi**2) / m
dpdt_cross = -grad_U  # pure Hamiltonian, no thermostat friction yet

# But wait, this is Hamiltonian dynamics -- it conserves H_cross, not thermostat!
# We need to ADD thermostat friction. Let's use the Nose-Hoover approach:
# dp/dt = -dH_cross/dq - alpha(xi)*p = -dU/dq - alpha(xi)*p
# dxi/dt = ...

# Actually for a general extended Hamiltonian, the thermostat structure is more nuanced.
# Let me take a step back and try the simplest possible momentum-dependent modification.

print("\n\n=== SIMPLEST APPROACH: Two thermostat variables ===")
print("xi_1 with log-osc potential, xi_2 with quadratic potential")
print("xi_1 couples to p normally, xi_2 provides additional mixing")
print("This is just the chain variant -- already tried by parent.")

print("\n=== APPROACH: Kinetic-energy-dependent thermostat mass Q(K) ===")
print("dxi/dt = (K - dim*kT) / Q(K)")
print("This changes the xi dynamics but NOT the momentum equation.")
print("Since friction is still g(xi)*p, the compressibility from p is -dim*g(xi).")
print("The xi compressibility is d[(K-dim*kT)/(Q(K))]/dxi = 0 (no xi in numerator or Q(K)).")
print("So kappa = -dim*g(xi) + 0 = -dim*g(xi) -- same as standard log-osc!")
print("And dH_ext/dt must also be the same for this to work...")
print("But Q(K) changes how FAST xi evolves, not the equilibrium distribution.")
print()
print("Extended Hamiltonian is still H_ext = U + K + Q_0*log(1+xi^2)")
print("The Q in the dxi/dt equation is Q(K), but the Q in H_ext is Q_0.")
print("This BREAKS the connection between equations and Hamiltonian!")
print("So Q(K) in dxi/dt does NOT preserve the invariant measure.")

print("\n\n=== VERIFIED APPROACH: General V(xi) with standard equations ===")
print("The key finding: ANY thermostat potential V(xi) works if:")
print("  dp/dt = -dU/dq - V'(xi)*p")
print("  dxi/dt = (K - dim*kT) / Q")
print("(Note: NOT (K-dim*kT/2)/Q -- it's K = sum(p^2/m)/2 vs sum(p^2)/m)")
print()
print("The standard NH uses V(xi) = xi^2/2 => V'=xi")
print("Log-osc uses V(xi) = log(1+xi^2) => V'=2xi/(1+xi^2)")
print()
print("NEW IDEA: Use V(xi) with MULTIPLE local minima to enhance mixing!")
print("e.g. V(xi) = -cos(xi) + xi^2/(2*sigma^2)")
print("or V(xi) = log(1+xi^2) + epsilon*cos(omega_xi*xi)")

# Test: V(xi) = log(1+xi^2) + eps*cos(w*xi)
eps_param, w_param = sp.symbols('epsilon omega_xi', real=True, positive=True)
V_mixed = sp.log(1 + xi**2) + eps_param * sp.cos(w_param * xi)
V_mixed_prime = sp.diff(V_mixed, xi)

H_mixed = U + K_expr + Q * V_mixed
rho_mixed = sp.exp(-H_mixed / kT)

dpdt_mixed = -grad_U - V_mixed_prime * p
dxidt_mixed = (p**2/m - kT) / Q

t_q = sp.diff(rho_mixed * dqdt, q)
t_p = sp.diff(rho_mixed * dpdt_mixed, p)
t_xi = sp.diff(rho_mixed * dxidt_mixed, xi)

liouville_mixed = sp.simplify((t_q + t_p + t_xi) / rho_mixed)
print(f"\nMixed V(xi) Liouville/rho: {liouville_mixed}")
print("=> 0 means ANY V(xi) preserves the invariant measure with standard equations!")
