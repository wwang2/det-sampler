"""Debug the spectral gap computation — inspect eigenvalue structure."""

import numpy as np
from spectral_gap import build_liouville_matrix, g_tanh, g_losc
from scipy import linalg


def inspect_spectrum(N, alpha=1.0, Q=1.0, omega=1.0, kT=1.0, g_func=None, label=""):
    """Print the eigenvalue spectrum near zero."""
    if g_func is None:
        g_func = g_tanh(alpha)
    L = build_liouville_matrix(N, omega=omega, kT=kT, Q=Q, g_func=g_func)
    eigs = linalg.eigvals(L)

    # Sort by |Re(λ)|
    idx_sorted = np.argsort(np.abs(eigs.real))
    eigs_sorted = eigs[idx_sorted]

    print(f"\n{'='*60}")
    print(f"{label} N={N}, α={alpha}, Q={Q}")
    print(f"Matrix size: {L.shape[0]}x{L.shape[0]}")
    print(f"{'='*60}")

    # Check: is there a zero eigenvalue?
    print(f"\n20 eigenvalues nearest to Re(λ)=0:")
    print(f"  {'Re(λ)':>12s}  {'Im(λ)':>12s}  {'|λ|':>12s}")
    for i in range(min(20, len(eigs_sorted))):
        e = eigs_sorted[i]
        print(f"  {e.real:12.6e}  {e.imag:12.6e}  {abs(e):12.6e}")

    # Count eigenvalues by sign of Re(λ)
    n_neg = np.sum(eigs.real < -1e-8)
    n_zero = np.sum(np.abs(eigs.real) < 1e-8)
    n_pos = np.sum(eigs.real > 1e-8)
    print(f"\nRe(λ) < -1e-8: {n_neg}")
    print(f"|Re(λ)| < 1e-8: {n_zero}")
    print(f"Re(λ) > 1e-8: {n_pos}")

    # The spectral gap should be from eigenvalues with Re(λ) < 0
    neg_real = eigs[eigs.real < -1e-8]
    if len(neg_real) > 0:
        gap_candidates = np.abs(neg_real.real)
        gap = np.min(gap_candidates)
        print(f"\nSpectral gap (min |Re(λ)| for Re<0): {gap:.6e}")

    # Also check: what fraction of eigenvalues are purely imaginary?
    purely_imag = np.sum((np.abs(eigs.real) < 1e-8) & (np.abs(eigs.imag) > 1e-8))
    print(f"Purely imaginary eigenvalues (|Re|<1e-8, |Im|>1e-8): {purely_imag}")

    return eigs


# Quick sanity check: for g(ξ) = αξ (linear, i.e. standard NH), the matrix
# should have known properties
print("\n" + "=" * 60)
print("SANITY CHECK: Linear g(ξ) = αξ (standard Nose-Hoover)")
print("=" * 60)

# For standard NH, g(ξ) = ξ (α=1, linear coupling)
# The 1D HO NH system is known to have purely imaginary eigenvalues (non-ergodic!)
g_linear = lambda xi: xi
inspect_spectrum(8, g_func=g_linear, Q=1.0, label="Linear g(ξ)=ξ")

# For tanh, it should break the KAM tori and introduce real parts
print("\n" + "=" * 60)
print("TANH DAMPING")
print("=" * 60)
inspect_spectrum(8, alpha=1.0, Q=1.0, label="tanh(ξ)")
inspect_spectrum(10, alpha=1.0, Q=1.0, label="tanh(ξ)")

# Check the structure of the matrix itself
print("\n" + "=" * 60)
print("MATRIX STRUCTURE CHECK")
print("=" * 60)
L = build_liouville_matrix(6, g_func=g_tanh(1.0), Q=1.0)
print(f"\nL shape: {L.shape}")
print(f"L is real: {np.allclose(L.imag, 0) if np.iscomplexobj(L) else True}")
print(f"Trace of L: {np.trace(L):.6e}")
print(f"||L - L^T|| / ||L||: {np.linalg.norm(L - L.T) / np.linalg.norm(L):.6e}")
print(f"L has non-zero diagonal: {np.any(np.abs(np.diag(L)) > 1e-10)}")

# Check: does the constant function (n=0,l=0,k=0) have eigenvalue 0?
# i.e., does L @ e_0 = 0?
e0 = np.zeros(6**3)
e0[0] = 1.0
Le0 = L @ e0
print(f"\n||L @ ψ_{{0,0,0}}||: {np.linalg.norm(Le0):.6e}")
print(f"L @ ψ_{{0,0,0}} (first 10): {Le0[:10]}")

# The stationary distribution in the orthonormal Hermite basis should be ψ_{0,0,0}
# (the constant function, since the measure is already the Gaussian weight)
# So L @ e_0 should be zero. If not, there's a bug.

# Also verify: for the Hamiltonian part alone (no thermostat), the operator should be
# antisymmetric (purely imaginary eigenvalues)
print("\n" + "=" * 60)
print("HAMILTONIAN PART ONLY (g=0, Q=inf equivalent)")
print("=" * 60)
g_zero = lambda xi: 0 * xi
L_ham = build_liouville_matrix(8, g_func=g_zero, Q=1.0)
eigs_ham = linalg.eigvals(L_ham)
print(f"Max |Re(λ)| for Hamiltonian part: {np.max(np.abs(eigs_ham.real)):.6e}")
print(f"This should be ~0 (antisymmetric operator)")
