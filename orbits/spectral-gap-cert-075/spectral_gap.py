"""Spectral gap of the Nose-Hoover Liouville operator via Hermite basis expansion.

Computes the eigenspectrum of the NH Liouville operator L for the 1D harmonic oscillator
in the {He_n(q/σ_q) · He_l(p/σ_p) · He_k(ξ/σ_ξ)} basis.

NH generator for 1D HO (U(q) = ω²q²/2, mass m=1):
  L = p·∂/∂q - (ω²q + g(ξ)·p)·∂/∂p + (p² - kT)/Q · ∂/∂ξ

Invariant measure: μ ∝ exp(-ω²q²/(2kT) - p²/(2kT) - Qξ²/(2kT))
"""

import numpy as np
from scipy import linalg
from scipy.special import roots_hermite
import itertools


def build_liouville_matrix(N, omega=1.0, kT=1.0, Q=1.0, g_func=None, g_prime_0=None):
    """Build the N³×N³ matrix representation of the NH Liouville operator.

    Parameters
    ----------
    N : int
        Truncation order for each variable (total basis size = N³)
    omega : float
        Harmonic oscillator frequency
    kT : float
        Temperature
    Q : float
        Thermostat mass
    g_func : callable
        Damping function g(ξ). If None, uses tanh(α·ξ) with α = g_prime_0.
    g_prime_0 : float
        g'(0) value. Used to construct default g_func if g_func is None.

    Returns
    -------
    L : ndarray, shape (N³, N³)
        Matrix representation of the Liouville operator
    """
    if g_func is None and g_prime_0 is None:
        raise ValueError("Must provide either g_func or g_prime_0")
    if g_func is None:
        alpha = g_prime_0
        g_func = lambda xi: np.tanh(alpha * xi)

    # Standard deviations for the Gaussian measure
    sigma_q = np.sqrt(kT / omega**2)
    sigma_p = np.sqrt(kT)
    sigma_xi = np.sqrt(kT / Q)

    # Total basis size
    ntot = N**3

    # Index mapping: (n, l, k) -> linear index
    def idx(n, l, k):
        return n * N * N + l * N + k

    # Pre-compute g(ξ) matrix elements in the Hermite basis via Gauss-Hermite quadrature
    # ⟨He_k | g(σ_ξ · x) | He_k'⟩ where the inner product is w.r.t. standard Gaussian
    # Use enough quadrature points for accuracy
    n_quad = max(4 * N, 40)
    xi_nodes, xi_weights = roots_hermite(n_quad)
    # The roots_hermite returns nodes/weights for ∫ f(x) exp(-x²) dx
    # We need ∫ He_k(x) g(σ_ξ x) He_k'(x) (1/√(2π)) exp(-x²/2) dx
    # Convert: if nodes are for exp(-x²), scale to exp(-x²/2) by x -> x*√2
    # Actually, let's use our own quadrature for the standard normal measure

    # Gauss-Hermite for standard normal: ∫ f(x) (1/√(2π)) exp(-x²/2) dx
    # We use the "probabilist's" Hermite quadrature
    # scipy roots_hermite gives physicist's: ∫ f(x) exp(-x²) dx
    # Convert: x_prob = x_phys * √2, w_prob = w_phys * √2 / √(2π) * exp(x_phys²) * exp(-x_prob²/2)
    # Simpler: just evaluate directly

    # Build Hermite polynomial values at quadrature points (probabilist's)
    # He_0(x) = 1, He_1(x) = x, He_n(x) = x·He_{n-1}(x) - (n-1)·He_{n-2}(x)

    # Use physicist's Hermite from scipy, then convert
    # Or just build our own quadrature on a fine grid

    # Actually, let's use Gauss-Hermite properly.
    # scipy.special.roots_hermite gives physicist's Hermite: ∫ f(x) exp(-x²) dx ≈ Σ w_i f(x_i)
    # We want ∫ f(x) (1/√(2π)) exp(-x²/2) dx
    # Substitution: x = t√2, dx = √2 dt
    # ∫ f(t√2) (1/√(2π)) exp(-t²) √2 dt = (1/√π) ∫ f(t√2) exp(-t²) dt
    # So: ∫ f(x) N(0,1) dx ≈ (1/√π) Σ w_i f(x_i √2)

    x_phys, w_phys = roots_hermite(n_quad)
    x_normal = x_phys * np.sqrt(2)  # nodes for standard normal
    w_normal = w_phys / np.sqrt(np.pi)  # weights for standard normal

    # Build probabilist's Hermite polynomials He_n(x) at quadrature nodes
    # He_0 = 1, He_1 = x, He_n = x He_{n-1} - (n-1) He_{n-2}
    He_vals = np.zeros((N, n_quad))  # He_n(x_normal) for n=0..N-1
    He_vals[0, :] = 1.0
    if N > 1:
        He_vals[1, :] = x_normal
    for n in range(2, N):
        He_vals[n, :] = x_normal * He_vals[n - 1, :] - (n - 1) * He_vals[n - 2, :]

    # Normalization: ⟨He_n, He_m⟩ = n! δ_{nm} under standard normal
    # Use normalized basis: ψ_n(x) = He_n(x) / √(n!)
    factorials = np.ones(N)
    for n in range(1, N):
        factorials[n] = factorials[n - 1] * n
    sqrt_fact = np.sqrt(factorials)

    # Normalized Hermite values: ψ_n(x) = He_n(x) / √(n!)
    psi_vals = He_vals / sqrt_fact[:, np.newaxis]

    # g(ξ) matrix elements in normalized basis:
    # G_{k,k'} = ⟨ψ_k | g(σ_ξ · x) | ψ_k'⟩ = Σ_i w_i ψ_k(x_i) g(σ_ξ x_i) ψ_k'(x_i)
    g_at_nodes = g_func(sigma_xi * x_normal)
    G_mat = np.zeros((N, N))
    for k in range(N):
        for kp in range(N):
            G_mat[k, kp] = np.sum(w_normal * psi_vals[k, :] * g_at_nodes * psi_vals[kp, :])

    # Now build the full Liouville matrix
    L = np.zeros((ntot, ntot))

    # Precompute coupling coefficients for normalized basis:
    # x · ψ_n(x) = √(n+1) ψ_{n+1}(x) + √n ψ_{n-1}(x)
    # ∂/∂x ψ_n(x) = √n ψ_{n-1}(x)   [since d/dx He_n = n He_{n-1}, and normalization]
    # x² in normalized basis: ⟨ψ_l | x² | ψ_l'⟩
    # x² ψ_l = x(√(l+1) ψ_{l+1} + √l ψ_{l-1})
    #         = √(l+1)(√(l+2) ψ_{l+2} + √(l+1) ψ_l) + √l(√l ψ_l + √(l-1) ψ_{l-2})
    #         = √((l+1)(l+2)) ψ_{l+2} + (2l+1) ψ_l + √(l(l-1)) ψ_{l-2}

    for n in range(N):
        for l in range(N):
            for k in range(N):
                row = idx(n, l, k)

                # --- Term 1: p · ∂/∂q ---
                # p = σ_p · x_p, ∂/∂q = (1/σ_q) · ∂/∂(x_q)
                # In normalized basis: (σ_p/σ_q) · x_p · ∂_{x_q}
                # x_p · ψ_l(x_p) = √(l+1) ψ_{l+1} + √l ψ_{l-1}
                # ∂_{x_q} ψ_n(x_q) = √n ψ_{n-1}(x_q)
                # Couples: (n,l,k) <- (n-1, l+1, k) with coeff (σ_p/σ_q)·√(l+1)·... wait
                # L acts on basis function, so L ψ_{n,l,k} produces terms in other basis functions
                # But we want the matrix element: L_{row, col} = coefficient of ψ_row in L ψ_col
                # Actually, for non-self-adjoint operators, we need to be careful.
                #
                # Let's think of it as: L_{ij} where L |j⟩ = Σ_i L_{ij} |i⟩
                # Then eigenvalues of the matrix L give the spectrum.
                #
                # L ψ_{n',l',k'} has component along ψ_{n,l,k}:
                # We compute L_{(n,l,k),(n',l',k')} = ⟨ψ_{n,l,k} | L | ψ_{n',l',k'}⟩
                # under the L² inner product with Gaussian weight (which is the invariant measure)
                #
                # BUT: L is not self-adjoint under this inner product. We need the matrix
                # representation in the basis, not the Galerkin projection.
                # For non-self-adjoint operators, the correct approach is:
                # L ψ_{n',l',k'} = Σ_{n,l,k} M_{(n,l,k),(n',l',k')} ψ_{n,l,k}
                # This is the same as the Galerkin matrix since {ψ} is orthonormal.
                pass  # We'll fill below more efficiently

    # More efficient: iterate over columns (n', l', k') and compute non-zero entries
    L = np.zeros((ntot, ntot))

    for np_ in range(N):
        for lp in range(N):
            for kp in range(N):
                col = idx(np_, lp, kp)

                # === Term 1: p · ∂/∂q ===
                # = σ_p · (x_p) · (1/σ_q) · ∂/∂(x_q)
                # Acting on ψ_{np_}(x_q) ψ_{lp}(x_p) ψ_{kp}(x_ξ):
                # (σ_p/σ_q) · [√(lp+1) ψ_{lp+1} + √lp ψ_{lp-1}] · √np_ ψ_{np_-1} · ψ_{kp}
                coeff_pq = sigma_p / sigma_q
                if np_ >= 1:
                    sqrt_np = np.sqrt(np_)
                    # Term: lp+1 channel
                    if lp + 1 < N:
                        row = idx(np_ - 1, lp + 1, kp)
                        L[row, col] += coeff_pq * np.sqrt(lp + 1) * sqrt_np
                    # Term: lp-1 channel
                    if lp >= 1:
                        row = idx(np_ - 1, lp - 1, kp)
                        L[row, col] += coeff_pq * np.sqrt(lp) * sqrt_np

                # === Term 2: -ω²q · ∂/∂p ===
                # = -ω² · σ_q · (x_q) · (1/σ_p) · ∂/∂(x_p)
                # Acting: -ω²(σ_q/σ_p) · [√(np_+1) ψ_{np_+1} + √np_ ψ_{np_-1}] · √lp ψ_{lp-1} · ψ_{kp}
                coeff_qp = -omega**2 * sigma_q / sigma_p
                if lp >= 1:
                    sqrt_lp = np.sqrt(lp)
                    if np_ + 1 < N:
                        row = idx(np_ + 1, lp - 1, kp)
                        L[row, col] += coeff_qp * np.sqrt(np_ + 1) * sqrt_lp
                    if np_ >= 1:
                        row = idx(np_ - 1, lp - 1, kp)
                        L[row, col] += coeff_qp * np.sqrt(np_) * sqrt_lp

                # === Term 3: -g(ξ) · p · ∂/∂p ===
                # = -g(ξ) · (p d/dp in x_p basis)
                # p · ∂/∂p ψ_l(x_p) = x_p · ∂/∂(x_p) ψ_l = x_p · √l ψ_{l-1}
                #   = √l [√l ψ_l + √(l-1) ψ_{l-2}]  ... wait
                # Actually: p ∂/∂p = σ_p x_p · (1/σ_p) ∂/∂x_p = x_p ∂/∂x_p
                # x_p ∂/∂x_p ψ_l = x_p √l ψ_{l-1} = √l [√l ψ_l + √(l-1) ψ_{l-2}]
                # = l ψ_l + √(l(l-1)) ψ_{l-2}
                #
                # But this isn't right either. Let me redo:
                # x · ψ_{l-1}(x) = √l ψ_l(x) + √(l-1) ψ_{l-2}(x)
                # So x · ∂_x ψ_l = √l · x · ψ_{l-1} = √l [√l ψ_l + √(l-1) ψ_{l-2}]
                #                 = l ψ_l + √(l(l-1)) ψ_{l-2}
                #
                # Combined with g(ξ) coupling (G matrix in k space):
                # -g(ξ) · (x_p ∂_{x_p}) ψ_{np_, lp, kp}
                # = - Σ_{k} G_{k, kp} · [l' ψ_{np_, lp, k} + √(lp(lp-1)) ψ_{np_, lp-2, k}]

                for k in range(N):
                    g_kk = G_mat[k, kp]
                    if abs(g_kk) < 1e-15:
                        continue

                    # Diagonal in l: -g · l' ψ_{np_, lp, k}
                    row = idx(np_, lp, k)
                    L[row, col] += -g_kk * lp

                    # Off-diagonal: -g · √(lp(lp-1)) ψ_{np_, lp-2, k}
                    if lp >= 2:
                        row = idx(np_, lp - 2, k)
                        L[row, col] += -g_kk * np.sqrt(lp * (lp - 1))

                # === Term 4: (p² - kT)/Q · ∂/∂ξ ===
                # p² = kT · x_p², and kT cancels:
                # (kT(x_p² - 1))/Q · (1/σ_ξ) ∂/∂(x_ξ)
                # = (kT/(Q σ_ξ)) · (x_p² - 1) · ∂_{x_ξ}
                #
                # x_p² ψ_lp = √((lp+1)(lp+2)) ψ_{lp+2} + (2lp+1) ψ_lp + √(lp(lp-1)) ψ_{lp-2}
                # (x_p² - 1) ψ_lp = √((lp+1)(lp+2)) ψ_{lp+2} + 2lp ψ_lp + √(lp(lp-1)) ψ_{lp-2}
                #
                # ∂_{x_ξ} ψ_kp = √kp ψ_{kp-1}

                coeff_therm = kT / (Q * sigma_xi)
                if kp >= 1:
                    sqrt_kp = np.sqrt(kp)

                    # lp+2 channel
                    if lp + 2 < N:
                        row = idx(np_, lp + 2, kp - 1)
                        L[row, col] += coeff_therm * np.sqrt((lp + 1) * (lp + 2)) * sqrt_kp

                    # lp channel (diagonal in l)
                    row = idx(np_, lp, kp - 1)
                    L[row, col] += coeff_therm * 2 * lp * sqrt_kp

                    # lp-2 channel
                    if lp >= 2:
                        row = idx(np_, lp - 2, kp - 1)
                        L[row, col] += coeff_therm * np.sqrt(lp * (lp - 1)) * sqrt_kp

    return L


def compute_spectral_gap(L):
    """Compute the spectral gap from the Liouville matrix.

    The spectral gap is the smallest |Re(λ)| among eigenvalues with Re(λ) < 0.
    (The zero eigenvalue corresponds to the stationary distribution.)

    Returns
    -------
    gap : float
        The spectral gap (positive number)
    eigenvalues : ndarray
        All eigenvalues
    """
    eigenvalues = linalg.eigvals(L)

    # The stationary distribution has eigenvalue 0
    # The spectral gap is min |Re(λ)| for Re(λ) < 0
    real_parts = eigenvalues.real

    # Filter eigenvalues with Re(λ) < -threshold (to skip the zero eigenvalue)
    threshold = 1e-6
    negative_real = real_parts[real_parts < -threshold]

    if len(negative_real) == 0:
        # No decaying modes found — might be purely imaginary (non-ergodic)
        return 0.0, eigenvalues

    gap = np.min(np.abs(negative_real))
    return gap, eigenvalues


def g_tanh(alpha):
    """Return tanh damping function with g'(0) = alpha."""
    return lambda xi: np.tanh(alpha * xi)


def g_losc(xi):
    """Log-oscillator damping: g(ξ) = 2ξ/(1+ξ²), g'(0)=2, g(∞)=0."""
    return 2 * xi / (1 + xi**2)


def sweep_spectral_gap(N=12, omega=1.0, kT=1.0):
    """Sweep α and Q, compute spectral gap for each."""

    alphas = [0.25, 0.5, 0.75, 1.0, np.sqrt(2), 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    Qs = [0.1, 0.3, 1.0, 3.0, 10.0]

    results = {}

    for Q in Qs:
        print(f"\n=== Q = {Q} ===")
        print(f"  Kac prediction: α_opt = √(2ω²/Q) = {np.sqrt(2*omega**2/Q):.4f}")

        for alpha in alphas:
            L = build_liouville_matrix(N, omega=omega, kT=kT, Q=Q,
                                       g_func=g_tanh(alpha))
            gap, eigs = compute_spectral_gap(L)
            results[('tanh', alpha, Q)] = (gap, eigs)
            print(f"  tanh(α={alpha:.3f}): λ₂ = {gap:.6f}")

        # Log-oscillator
        L = build_liouville_matrix(N, omega=omega, kT=kT, Q=Q, g_func=g_losc)
        gap, eigs = compute_spectral_gap(L)
        results[('losc', 2.0, Q)] = (gap, eigs)
        print(f"  log-osc (g'(0)=2):  λ₂ = {gap:.6f}")

    return results, alphas, Qs


def run_nh_simulation(omega=1.0, kT=1.0, Q=1.0, g_func=None, alpha=1.0,
                      dt=0.01, n_steps=200000, n_skip=10):
    """Run NH simulation and compute empirical autocorrelation time.

    Returns τ_int for the position observable.
    """
    if g_func is None:
        g_func = g_tanh(alpha)

    sigma_q = np.sqrt(kT / omega**2)

    # Initialize
    q = np.random.randn() * sigma_q
    p = np.random.randn() * np.sqrt(kT)
    xi = np.random.randn() * np.sqrt(kT / Q)

    # Collect samples
    n_samples = n_steps // n_skip
    q_samples = np.zeros(n_samples)

    for step in range(n_steps):
        # Velocity Verlet with NH thermostat (simplified)
        # Half-step ξ
        xi += 0.5 * dt * (p**2 - kT) / Q

        # Half-step p (friction + force)
        p *= np.exp(-g_func(xi) * 0.5 * dt)
        p -= 0.5 * dt * omega**2 * q

        # Full-step q
        q += dt * p

        # Force at new position
        # Half-step p
        p -= 0.5 * dt * omega**2 * q
        p *= np.exp(-g_func(xi) * 0.5 * dt)

        # Half-step ξ
        xi += 0.5 * dt * (p**2 - kT) / Q

        if step % n_skip == 0:
            q_samples[step // n_skip] = q

    # Compute autocorrelation time
    tau_int = compute_autocorrelation_time(q_samples)

    return tau_int, q_samples


def compute_autocorrelation_time(x, max_lag=None):
    """Compute integrated autocorrelation time using initial positive sequence estimator."""
    x = x - np.mean(x)
    n = len(x)
    if max_lag is None:
        max_lag = n // 4

    var = np.var(x)
    if var < 1e-15:
        return 1.0

    # FFT-based autocorrelation
    fft_x = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(fft_x * np.conj(fft_x)).real[:n] / (n * var)

    # Integrated autocorrelation time with automatic windowing
    tau = 0.5  # start with C(0)/2
    for t in range(1, max_lag):
        if acf[t] < 0:
            break
        tau += acf[t]

    return tau


if __name__ == "__main__":
    import time

    print("=" * 70)
    print("SPECTRAL GAP OF NH LIOUVILLE OPERATOR — HERMITE BASIS")
    print("=" * 70)

    # Step 1: Convergence check with basis size
    print("\n--- Convergence check: N = 8, 10, 12, 14 for tanh(ξ), Q=1 ---")
    for N in [8, 10, 12, 14]:
        t0 = time.time()
        L = build_liouville_matrix(N, g_func=g_tanh(1.0), Q=1.0)
        gap, _ = compute_spectral_gap(L)
        elapsed = time.time() - t0
        print(f"  N={N:2d} (dim={N**3:5d}): λ₂ = {gap:.6f}  ({elapsed:.2f}s)")

    # Step 2: Full parameter sweep
    print("\n--- Full parameter sweep ---")
    results, alphas, Qs = sweep_spectral_gap(N=12)

    # Step 3: Find optimal α for each Q
    print("\n\n--- Optimal α per Q ---")
    print(f"{'Q':>6s}  {'α_opt(tanh)':>12s}  {'λ₂_opt':>10s}  {'Kac pred':>10s}  {'λ₂_losc':>10s}  {'ratio losc/tanh':>16s}")
    print("-" * 80)

    for Q in Qs:
        gaps_tanh = [(alpha, results[('tanh', alpha, Q)][0]) for alpha in alphas]
        best_alpha, best_gap = max(gaps_tanh, key=lambda x: x[1])

        losc_gap = results[('losc', 2.0, Q)][0]
        kac_pred = np.sqrt(2.0 / Q)

        # Find tanh gap at g'(0)=2 for fair comparison
        tanh2_gap = results[('tanh', 2.0, Q)][0]
        ratio = losc_gap / tanh2_gap if tanh2_gap > 0 else float('inf')

        print(f"{Q:6.1f}  {best_alpha:12.4f}  {best_gap:10.6f}  {kac_pred:10.4f}  {losc_gap:10.6f}  {ratio:16.4f}")

    # Step 4: Empirical validation
    print("\n\n--- Empirical validation: τ_int vs 1/λ₂ ---")
    print(f"{'α':>6s}  {'Q':>5s}  {'λ₂':>10s}  {'1/λ₂':>10s}  {'τ_int(emp)':>12s}  {'ratio':>8s}")
    print("-" * 65)

    np.random.seed(42)
    test_cases = [(1.0, 1.0), (np.sqrt(2), 1.0), (2.0, 1.0), (2.0, 0.3), (1.0, 3.0)]
    for alpha, Q in test_cases:
        gap = results[('tanh', alpha, Q)][0]
        inv_gap = 1.0 / gap if gap > 0 else float('inf')
        tau, _ = run_nh_simulation(alpha=alpha, Q=Q, n_steps=500000)
        ratio = tau / inv_gap if inv_gap < float('inf') else 0
        print(f"{alpha:6.3f}  {Q:5.1f}  {gap:10.6f}  {inv_gap:10.4f}  {tau:12.4f}  {ratio:8.4f}")

    # Also validate log-osc empirically
    for Q in [0.3, 1.0]:
        gap = results[('losc', 2.0, Q)][0]
        inv_gap = 1.0 / gap if gap > 0 else float('inf')
        tau, _ = run_nh_simulation(g_func=g_losc, Q=Q, n_steps=500000)
        tanh2_gap = results[('tanh', 2.0, Q)][0]
        print(f"{'losc':>6s}  {Q:5.1f}  {gap:10.6f}  {inv_gap:10.4f}  {tau:12.4f}  {1/gap if gap>0 else 0:8.4f}  (tanh2: 1/λ₂={1/tanh2_gap:.4f})")
