"""Symbolic verification of the Master Theorem (Theorem 1).

Uses SymPy to verify the Liouville equation for general V(xi):
    div(F) + F . grad(log rho) = 0

for the dynamics:
    dq/dt = p/m
    dp/dt = -dU/dq - V'(xi)/Q * p
    dxi/dt = (1/Q)(p^2/m - kT)  [1D case]

with rho ~ exp(-(U + p^2/(2m) + V(xi))/kT).

This provides a machine-checked proof of Theorem 1.
"""

import sympy as sp


def verify_general_theorem():
    """Verify the Master Theorem for general V(xi)."""
    print("="*70)
    print("SYMBOLIC VERIFICATION: Master Theorem (Theorem 1)")
    print("="*70)

    # Symbols
    q, p, xi, kT, m, Q = sp.symbols('q p xi kT m Q', positive=True)
    N = sp.Symbol('N', positive=True, integer=True)  # number of DOF

    # General thermostat potential V(xi)
    V = sp.Function('V')
    U = sp.Function('U')

    # Extended Hamiltonian
    H_ext = U(q) + p**2 / (2*m) + V(xi)

    # Target density: rho ~ exp(-H_ext / kT)
    log_rho = -H_ext / kT

    # Friction function (the CLAIMED form)
    g = sp.diff(V(xi), xi) / Q  # g(xi) = V'(xi)/Q

    print(f"\nV(xi) = V(xi)  [general]")
    print(f"g(xi) = V'(xi)/Q = {g}")

    # ============================================================
    # 1D case (d=1) for clarity, then generalize
    # ============================================================
    print("\n--- 1D Case ---")

    # Vector field F = (dq/dt, dp/dt, dxi/dt)
    F_q = p / m
    F_p = -sp.diff(U(q), q) - g * p
    F_xi = (p**2 / m - kT) / Q

    print(f"F_q = {F_q}")
    print(f"F_p = {F_p}")
    print(f"F_xi = {F_xi}")

    # Divergence
    div_q = sp.diff(F_q, q)
    div_p = sp.diff(F_p, p)
    div_xi = sp.diff(F_xi, xi)
    div_F = div_q + div_p + div_xi

    div_F_simplified = sp.simplify(div_F)
    print(f"\ndiv(F) = {div_F_simplified}")

    # F . grad(log rho)
    grad_log_rho_q = sp.diff(log_rho, q)
    grad_log_rho_p = sp.diff(log_rho, p)
    grad_log_rho_xi = sp.diff(log_rho, xi)

    Fdot = F_q * grad_log_rho_q + F_p * grad_log_rho_p + F_xi * grad_log_rho_xi

    Fdot_simplified = sp.simplify(sp.expand(Fdot))
    print(f"F.grad(log rho) = {Fdot_simplified}")

    # Liouville condition
    liouville = sp.simplify(sp.expand(div_F + Fdot))
    print(f"\nLiouville = div(F) + F.grad(log rho) = {liouville}")

    if liouville == 0:
        print("  >>> VERIFIED: Liouville equation satisfied for GENERAL V(xi)")
    else:
        print(f"  >>> WARNING: non-zero result: {liouville}")
        # Try harder simplification
        liouville2 = sp.trigsimp(sp.cancel(liouville))
        print(f"  >>> After further simplification: {liouville2}")

    # ============================================================
    # Verify specific cases
    # ============================================================
    print("\n" + "="*70)
    print("SPECIFIC CASES")
    print("="*70)

    cases = {
        "NH (quadratic)": Q * xi**2 / 2,
        "Log-Osc": Q * sp.log(1 + xi**2),
        "Tanh": Q * sp.log(sp.cosh(xi)),
        "Arctan": Q * (xi * sp.atan(xi) - sp.log(1 + xi**2) / 2),
    }

    for name, V_specific in cases.items():
        g_specific = sp.diff(V_specific, xi) / Q
        g_specific_simplified = sp.simplify(g_specific)

        # Check Liouville with this specific V
        F_p_s = -sp.diff(U(q), q) - g_specific * p
        F_xi_s = (p**2 / m - kT) / Q

        div_s = sp.diff(F_p_s, p) + sp.diff(F_xi_s, xi)
        grad_rho_q = -sp.diff(U(q), q) / kT
        grad_rho_p = -p / (m * kT)
        grad_rho_xi = -sp.diff(V_specific, xi) / kT

        Fdot_s = (p/m) * grad_rho_q + F_p_s * grad_rho_p + F_xi_s * grad_rho_xi
        liouville_s = sp.simplify(sp.expand(div_s + Fdot_s))

        status = "PASS" if liouville_s == 0 else f"FAIL ({liouville_s})"
        print(f"\n{name}:")
        print(f"  V(xi) = {V_specific}")
        print(f"  g(xi) = {g_specific_simplified}")
        print(f"  Liouville check: {status}")

    # ============================================================
    # Uniqueness: show g must equal V'/Q
    # ============================================================
    print("\n" + "="*70)
    print("UNIQUENESS PROOF")
    print("="*70)

    alpha = sp.Function('alpha')  # general friction

    F_p_gen = -sp.diff(U(q), q) - alpha(xi) * p
    div_gen = sp.diff(F_p_gen, p)  # = -alpha(xi)
    Fdot_gen = ((p/m) * (-sp.diff(U(q), q)/kT) +
                F_p_gen * (-p/(m*kT)) +
                (p**2/m - kT)/Q * (-sp.diff(V(xi), xi)/kT))

    liouville_gen = sp.simplify(sp.expand(div_gen + Fdot_gen))
    print(f"\nLiouville with general alpha(xi):")
    print(f"  = {liouville_gen}")

    # This must be zero for all p. Collect in p^2 and p^0:
    collected = sp.collect(sp.expand(liouville_gen), p)
    print(f"\n  Collected by p: {collected}")

    # The coefficient of p^2 must vanish, and the constant term must vanish
    coeff_p2 = liouville_gen.coeff(p, 2)
    coeff_p0 = liouville_gen.coeff(p, 0)
    print(f"\n  Coefficient of p^2: {sp.simplify(coeff_p2)}")
    print(f"  Coefficient of p^0: {sp.simplify(coeff_p0)}")

    # From p^2 coefficient = 0:
    # alpha(xi)/(m*kT) - V'(xi)/(Q*m*kT) = 0
    # => alpha(xi) = V'(xi)/Q
    print(f"\n  From p^2 coeff = 0: alpha(xi) = V'(xi)/Q  (uniqueness!)")

    # Verify p^0 coefficient also vanishes when alpha = V'/Q:
    p0_check = coeff_p0.subs(alpha(xi), sp.diff(V(xi), xi) / Q)
    p0_simplified = sp.simplify(p0_check)
    print(f"  p^0 coeff with alpha=V'/Q: {p0_simplified}")
    if p0_simplified == 0:
        print("  >>> CONFIRMED: Both conditions satisfied. Uniqueness proved.")


def verify_multiscale():
    """Verify invariant measure for multi-scale thermostat (2 variables)."""
    print("\n\n" + "="*70)
    print("MULTI-SCALE THEOREM (Proposition 4.1)")
    print("="*70)

    q, p, kT, m = sp.symbols('q p kT m', positive=True)
    xi1, xi2, Q1, Q2 = sp.symbols('xi1 xi2 Q1 Q2', positive=True)

    V1 = sp.Function('V1')
    V2 = sp.Function('V2')
    U = sp.Function('U')

    g1 = sp.diff(V1(xi1), xi1) / Q1
    g2 = sp.diff(V2(xi2), xi2) / Q2
    G = g1 + g2  # total friction

    # H_ext = U + p^2/(2m) + V1(xi1) + V2(xi2)
    log_rho = -(U(q) + p**2/(2*m) + V1(xi1) + V2(xi2)) / kT

    # Vector field (1D physical system)
    F_q = p / m
    F_p = -sp.diff(U(q), q) - G * p
    F_xi1 = (p**2/m - kT) / Q1
    F_xi2 = (p**2/m - kT) / Q2

    # Divergence
    div_F = (sp.diff(F_q, q) + sp.diff(F_p, p) +
             sp.diff(F_xi1, xi1) + sp.diff(F_xi2, xi2))

    # F . grad(log rho)
    Fdot = (F_q * sp.diff(log_rho, q) + F_p * sp.diff(log_rho, p) +
            F_xi1 * sp.diff(log_rho, xi1) + F_xi2 * sp.diff(log_rho, xi2))

    liouville = sp.simplify(sp.expand(div_F + Fdot))
    print(f"\nLiouville = {liouville}")
    if liouville == 0:
        print(">>> VERIFIED: Multi-scale thermostat preserves invariant measure")
    else:
        print(f">>> Non-zero: {liouville}")


if __name__ == "__main__":
    verify_general_theorem()
    verify_multiscale()
