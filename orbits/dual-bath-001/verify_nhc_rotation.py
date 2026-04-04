"""Check if NHC(M=2) + rotation preserves the invariant measure."""
import sympy as sp

q, p, xi, eta = sp.symbols('q p xi eta')
kT, m, Q_xi, Q_eta, alpha = sp.symbols('kT m Q_xi Q_eta alpha', positive=True)
U = sp.Function('U')(q)
dUdq = sp.diff(U, q)

# NHC(M=2) invariant measure:
# rho ~ exp(-U/kT - p^2/(2m*kT) - Q_xi*xi^2/(2*kT) - Q_eta*eta^2/(2*kT))
S = U/kT + p**2/(2*m*kT) + Q_xi*xi**2/(2*kT) + Q_eta*eta**2/(2*kT)

dS_dq = sp.diff(S, q)
dS_dp = sp.diff(S, p)
dS_dxi = sp.diff(S, xi)
dS_deta = sp.diff(S, eta)


def check(name, dqdt, dpdt, dxidt, detadt):
    dS_dt = dS_dq*dqdt + dS_dp*dpdt + dS_dxi*dxidt + dS_deta*detadt
    div_v = sp.diff(dqdt, q) + sp.diff(dpdt, p) + sp.diff(dxidt, xi) + sp.diff(detadt, eta)
    residual = sp.simplify(dS_dt - div_v)
    print(f"{name}: residual = {residual}  {'PASS' if residual == 0 else 'FAIL'}")
    return residual == 0


# NHC(M=2) + rotation in (xi, eta)
# Standard NHC(M=2): dp/dt = -dU/dq - xi*p, dxi/dt = (KE-kT)/Q_xi - eta*xi, deta/dt = (Q_xi*xi^2 - kT)/Q_eta
# Add measure-preserving rotation: +alpha*sqrt(Q_eta/Q_xi)*eta to dxi, -alpha*sqrt(Q_xi/Q_eta)*xi to deta
KE = p**2/m

check("NHC(M=2) standard",
      dqdt=p/m,
      dpdt=-dUdq - xi*p,
      dxidt=(KE - kT)/Q_xi - eta*xi,
      detadt=(Q_xi*xi**2 - kT)/Q_eta)

check("NHC(M=2) + rotation",
      dqdt=p/m,
      dpdt=-dUdq - xi*p,
      dxidt=(KE - kT)/Q_xi - eta*xi + alpha*sp.sqrt(Q_eta/Q_xi)*eta,
      detadt=(Q_xi*xi**2 - kT)/Q_eta - alpha*sp.sqrt(Q_xi/Q_eta)*xi)

# Hmm the NHC chain coupling -eta*xi in dxi/dt creates div contribution
# Let me also try: parallel with only xi coupling to p
# dp/dt = -dU/dq - xi*p  (only xi provides friction)
# dxi/dt = (KE - kT)/Q_xi + alpha*sqrt(Q_eta/Q_xi)*eta
# deta/dt = ??? - alpha*sqrt(Q_xi/Q_eta)*xi
# For deta/dt, eta has no direct coupling to p, so we need something to drive it.
# Option: deta/dt = (KE - kT)/Q_eta - alpha*sqrt(Q_xi/Q_eta)*xi
# This means eta responds to kinetic energy but doesn't directly damp p.
# Its effect is purely through the rotation influencing xi.

check("Single friction + kinetic eta + rotation",
      dqdt=p/m,
      dpdt=-dUdq - xi*p,
      dxidt=(KE - kT)/Q_xi + alpha*sp.sqrt(Q_eta/Q_xi)*eta,
      detadt=(KE - kT)/Q_eta - alpha*sp.sqrt(Q_xi/Q_eta)*xi)

# Another option: eta is driven by xi's energy (like NHC)
check("Single friction + chain eta + rotation",
      dqdt=p/m,
      dpdt=-dUdq - xi*p,
      dxidt=(KE - kT)/Q_xi + alpha*sp.sqrt(Q_eta/Q_xi)*eta,
      detadt=(Q_xi*xi**2 - kT)/Q_eta - alpha*sp.sqrt(Q_xi/Q_eta)*xi)

# What about eta driven by a mix?
# Let me try something where eta acts as a "reserve" bath
# that exchanges energy with xi through the rotation
check("Single friction + free eta + rotation",
      dqdt=p/m,
      dpdt=-dUdq - xi*p,
      dxidt=(KE - kT)/Q_xi + alpha*sp.sqrt(Q_eta/Q_xi)*eta,
      detadt=-alpha*sp.sqrt(Q_xi/Q_eta)*xi)

# Pure rotation for eta -- only the rotation drives it
# Check: dS/deta * detadt = Q_eta*eta/kT * (-alpha*sqrt(Q_xi/Q_eta)*xi)
# = -alpha*sqrt(Q_xi*Q_eta)*xi*eta/kT
# And dS/dxi * rotation_part = Q_xi*xi/kT * alpha*sqrt(Q_eta/Q_xi)*eta
# = alpha*sqrt(Q_xi*Q_eta)*xi*eta/kT
# These cancel! Good.
# div: d(-alpha*sqrt(Q_xi/Q_eta)*xi)/deta = 0, d(alpha*sqrt(Q_eta/Q_xi)*eta)/dxi = 0
# So no extra div terms. The only div is still from -xi*p -> -xi (for dim=1)
# And dS/dt from the non-rotation parts: same as NH.
# This should work!
