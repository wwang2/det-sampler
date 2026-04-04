"""Verify invariant measure condition for dual-bath thermostat candidates.

The Liouville equation for a flow with velocity field v = (dq/dt, dp/dt, dxi/dt, deta/dt):
    d(rho)/dt + rho * div(v) = 0

If rho = exp(-S), then:
    dS/dt = div(v)

where dS/dt = sum_i (dS/dx_i) * (dx_i/dt).

We check: dS/dt - div(v) = 0 identically.
"""

import sympy as sp

# Symbols
q, p, xi, eta = sp.symbols('q p xi eta')
kT, m, Q_xi, Q_eta = sp.symbols('kT m Q_xi Q_eta', positive=True)
dim = sp.Symbol('d', positive=True, integer=True)

# U(q) is an arbitrary potential -- keep it symbolic
U = sp.Function('U')(q)
dUdq = sp.diff(U, q)  # gradient

# Target invariant measure:
# rho ~ exp(-S) where S = U/kT + p^2/(2*m*kT) + Q_xi*xi^2/(2*kT) + Q_eta*eta^2/(2*kT)
S = U/kT + p**2/(2*m*kT) + Q_xi*xi**2/(2*kT) + Q_eta*eta**2/(2*kT)

dS_dq = sp.diff(S, q)  # = dUdq / kT
dS_dp = sp.diff(S, p)  # = p / (m*kT)
dS_dxi = sp.diff(S, xi)  # = Q_xi*xi / kT
dS_deta = sp.diff(S, eta)  # = Q_eta*eta / kT

print("Partial derivatives of S:")
print(f"  dS/dq = {dS_dq}")
print(f"  dS/dp = {dS_dp}")
print(f"  dS/dxi = {dS_dxi}")
print(f"  dS/deta = {dS_deta}")
print()


def check_invariant(name, dqdt, dpdt, dxidt, detadt):
    """Check if dS/dt = div(v) for given dynamics."""
    print(f"=== {name} ===")

    # dS/dt = dS/dq * dq/dt + dS/dp * dp/dt + dS/dxi * dxi/dt + dS/deta * deta/dt
    dS_dt = dS_dq * dqdt + dS_dp * dpdt + dS_dxi * dxidt + dS_deta * detadt

    # div(v) = d(dqdt)/dq + d(dpdt)/dp + d(dxidt)/dxi + d(detadt)/deta
    div_v = sp.diff(dqdt, q) + sp.diff(dpdt, p) + sp.diff(dxidt, xi) + sp.diff(detadt, eta)

    # Check: dS/dt - div(v) should be 0
    residual = sp.simplify(dS_dt - div_v)

    print(f"  dS/dt = {sp.simplify(dS_dt)}")
    print(f"  div(v) = {sp.simplify(div_v)}")
    print(f"  dS/dt - div(v) = {residual}")
    print(f"  INVARIANT: {'YES' if residual == 0 else 'NO'}")
    print()
    return residual == 0


# ============================================================
# Candidate 1: Two independent NH thermostats on p
# dq/dt = p/m
# dp/dt = -dU/dq - (xi + eta) * p
# dxi/dt = (1/Q_xi) * (p^2/m - dim*kT)
# deta/dt = (1/Q_eta) * (p^2/m - dim*kT)
# ============================================================
# Note: for 1D, dim=1, p^2/m -> p^2/m (scalar)
# We use scalar p for verification

check_invariant(
    "Candidate 1: additive dual NH (both kinetic)",
    dqdt = p/m,
    dpdt = -dUdq - (xi + eta) * p,
    dxidt = (p**2/m - kT) / Q_xi,  # using dim=1 for simplicity
    detadt = (p**2/m - kT) / Q_eta,
)


# ============================================================
# Candidate 2: Cross-coupled dual bath
# dq/dt = p/m
# dp/dt = -dU/dq - xi * p
# dxi/dt = (1/Q_xi) * (p^2/m - kT) - eta * xi
# deta/dt = (1/Q_eta) * (Q_xi * xi^2 - kT)
# This is basically NHC(M=2) structure but we want to verify
# ============================================================

check_invariant(
    "Candidate 2: NHC(M=2) structure",
    dqdt = p/m,
    dpdt = -dUdq - xi * p,
    dxidt = (p**2/m - kT) / Q_xi - eta * xi,
    detadt = (Q_xi * xi**2 - kT) / Q_eta,
)


# ============================================================
# Candidate 3: Dual coupling - xi friction on p, eta modifies force
# dq/dt = p/m
# dp/dt = -(1 + eta) * dU/dq - xi * p
# dxi/dt = (1/Q_xi) * (p^2/m - kT)
# deta/dt = (1/Q_eta) * (something involving configurational temp)
#
# For this we need to figure out what deta/dt should be.
# If dp/dt = -(1+eta)*dUdq - xi*p, then:
#   d(dpdt)/dp = -xi  (contribution to div)
#   dS/dp * dpdt = (p/(m*kT)) * (-(1+eta)*dUdq - xi*p)
# We need dS/deta * detadt to compensate extra terms.
# ============================================================

# Let's try: deta/dt = (1/Q_eta) * (p*dUdq/m + kT*d2Udq2/? ...)
# Actually, let's be systematic. We need:
# dS/dt = div(v)
#
# Standard NH part gives: dS/dq*p/m + dS/dp*(-dUdq - xi*p) + dS/dxi*(KE-kT)/Q_xi = -xi (div from -xi*p)
# The extra eta*dUdq in dp/dt gives: dS/dp * (-eta*dUdq) = -(p/(m*kT)) * eta * dUdq
# No extra div contribution from eta*dUdq (no p dependence in eta*dUdq)
# So we need: dS/deta * detadt = -(p/(m*kT)) * eta * dUdq
# i.e., (Q_eta*eta/kT) * detadt = -(p*eta*dUdq)/(m*kT)
# => detadt = -p*dUdq / (m*Q_eta)
# But this depends on p and dUdq -- let's check if div adds terms

# Let me just define it and check
check_invariant(
    "Candidate 3: eta modifies force, detadt = -p*dUdq/(m*Q_eta)",
    dqdt = p/m,
    dpdt = -(1 + eta) * dUdq - xi * p,
    dxidt = (p**2/m - kT) / Q_xi,
    detadt = -p * dUdq / (m * Q_eta),
)


# ============================================================
# Candidate 4: Additive friction with cross-coupling for mixing
# dq/dt = p/m
# dp/dt = -dU/dq - (xi + eta) * p
# dxi/dt = (1/Q_xi) * (p^2/m - kT) - eta * xi
# deta/dt = (1/Q_eta) * (p^2/m - kT) + xi * eta  (antisymmetric coupling)
#
# Note the antisymmetric xi<->eta coupling: -eta*xi and +xi*eta
# This creates a rotational flow in (xi, eta) space which aids mixing
# ============================================================

check_invariant(
    "Candidate 4: Additive friction + antisymmetric cross-coupling",
    dqdt = p/m,
    dpdt = -dUdq - (xi + eta) * p,
    dxidt = (p**2/m - kT) / Q_xi - eta * xi,
    detadt = (p**2/m - kT) / Q_eta + xi * eta,
)

# Let me also check a variant where the cross-coupling is different
# to ensure volume preservation in (xi, eta) subspace

# ============================================================
# Candidate 5: Two baths, one kinetic one configurational
# Using Rugh's configurational temperature: kT = <|grad U|^2> / <lap U>
# For 1D: kT = (dU/dq)^2 / (d2U/dq2)
#
# dq/dt = p/m
# dp/dt = -dU/dq - xi * p
# dxi/dt = (1/Q_xi) * (p^2/m - kT) - eta * xi
# deta/dt = (1/Q_eta) * (Q_xi * xi^2 - kT)
#
# This is just NHC(2) again. Let's try something truly novel:
# Two PARALLEL chains rather than serial
# ============================================================

# ============================================================
# Candidate 6: Parallel dual bath with Hamiltonian cross-coupling
# dq/dt = p/m
# dp/dt = -dU/dq - xi * p - eta * p  = -dU/dq - (xi+eta)*p
# dxi/dt = (1/Q_xi) * (p^2/m - kT) + alpha * eta   (Hamiltonian rotation)
# deta/dt = (1/Q_eta) * (p^2/m - kT) - alpha * xi   (Hamiltonian rotation)
#
# The alpha*(eta, -xi) rotation is divergence-free and creates
# a Hamiltonian flow in (xi, eta) space that promotes mixing.
# ============================================================

alpha = sp.Symbol('alpha', real=True)

check_invariant(
    "Candidate 6: Parallel dual + Hamiltonian rotation in (xi,eta)",
    dqdt = p/m,
    dpdt = -dUdq - (xi + eta) * p,
    dxidt = (p**2/m - kT) / Q_xi + alpha * eta,
    detadt = (p**2/m - kT) / Q_eta - alpha * xi,
)

# ============================================================
# Candidate 7: Like 6 but with the rotation terms using Q-weighted vars
# to ensure consistency with the measure
#
# The rotation should be symplectic w.r.t. the thermostat "energy"
# H_th = Q_xi*xi^2/2 + Q_eta*eta^2/2
# Hamiltonian rotation: dxi/dt += alpha * dH_th/deta / Q_xi = alpha * eta
# deta/dt += -alpha * dH_th/dxi / Q_eta = -alpha * xi
# Wait, that's not quite right. For a Hamiltonian-like rotation:
# dxi/dt += alpha * eta * Q_eta / Q_xi  (to preserve H_th)
# deta/dt += -alpha * xi * Q_xi / Q_eta
# Actually no. For divergence-free rotation in the measure exp(-Q_xi*xi^2/(2kT) - Q_eta*eta^2/(2kT)),
# we need the rotation to preserve this.
#
# Let's define u = sqrt(Q_xi)*xi, v = sqrt(Q_eta)*eta.
# Standard rotation in (u,v): du/dt = alpha*v, dv/dt = -alpha*u
# dxi/dt = alpha*sqrt(Q_eta/Q_xi)*eta
# deta/dt = -alpha*sqrt(Q_xi/Q_eta)*xi
#
# Check: dS/dxi * alpha*sqrt(Q_eta/Q_xi)*eta + dS/deta * (-alpha*sqrt(Q_xi/Q_eta)*xi)
# = Q_xi*xi/kT * alpha*sqrt(Q_eta/Q_xi)*eta - Q_eta*eta/kT * alpha*sqrt(Q_xi/Q_eta)*xi
# = alpha*xi*eta/kT * [sqrt(Q_xi*Q_eta) - sqrt(Q_xi*Q_eta)] = 0  ✓
#
# div contribution: d/dxi[alpha*sqrt(Q_eta/Q_xi)*eta] + d/deta[-alpha*sqrt(Q_xi/Q_eta)*xi] = 0 ✓
# ============================================================

Q_xi_sqrt = sp.sqrt(Q_xi)
Q_eta_sqrt = sp.sqrt(Q_eta)
r = sp.sqrt(Q_eta / Q_xi)

check_invariant(
    "Candidate 7: Parallel dual + measure-preserving rotation",
    dqdt = p/m,
    dpdt = -dUdq - (xi + eta) * p,
    dxidt = (p**2/m - kT) / Q_xi + alpha * sp.sqrt(Q_eta/Q_xi) * eta,
    detadt = (p**2/m - kT) / Q_eta - alpha * sp.sqrt(Q_xi/Q_eta) * xi,
)

print("\n\n====== SUMMARY ======")
print("Candidates that preserve canonical measure are marked YES above.")
print("Key insight: Candidate 7 combines two parallel kinetic thermostats")
print("with a measure-preserving Hamiltonian rotation for enhanced mixing.")
