"""Spectral gap of the NH Liouville operator — v2: correct symmetrized formulation.

The NH dynamics for 1D HO:
  dq/dt = p
  dp/dt = -omega^2 q - g(xi) p
  dxi/dt = (p^2 - kT) / Q

Invariant density: rho(q,p,xi) = Z^{-1} exp(-omega^2 q^2/(2kT) - p^2/(2kT) - Q xi^2/(2kT))

The Fokker-Planck (forward Kolmogorov) operator acts on densities:
  L_FP rho = -div(v * rho)
where v = (dq/dt, dp/dt, dxi/dt).

The Koopman (backward Kolmogorov) operator acts on observables:
  L_K f = v . grad(f)

For computing autocorrelation decay, we want the Koopman operator.
In the weighted L^2(mu) space where mu is the invariant measure,
the Koopman operator becomes:

  L_K f = p dq f - (omega^2 q + g(xi) p) dp f + (p^2 - kT)/Q dxi f

Working in the variable x = q/sigma_q, y = p/sigma_p, z = xi/sigma_xi,
and using the orthonormal Hermite basis psi_n(x) = He_n(x)/sqrt(n!),
the matrix elements are computed by expanding L_K psi_{n,l,k}.

Key insight: L_K is NOT symmetric in L^2(mu). It splits as:
  L_K = A + D
where A (Hamiltonian + thermostat coupling) is antisymmetric and D (friction) is
negative semi-definite. The eigenvalues should all have Re(lambda) <= 0.

The friction part is: -g(xi) p dp f
In variables: -g(sigma_xi z) * sigma_p y * (1/sigma_p) d/dy f = -g(sigma_xi z) * y dy f

Now y dy psi_l = y sqrt(l) psi_{l-1} = sqrt(l)[sqrt(l) psi_l + sqrt(l-1) psi_{l-2}]
              = l psi_l + sqrt(l(l-1)) psi_{l-2}

But there's also a compensating term from the divergence of the flow.
The Koopman operator L_K in L^2(mu) satisfies:
  <f, L_K h>_mu = -<L_K^* f, h>_mu  ... no, more complex.

Actually: L_K is the generator of the Markov semigroup. For the NH system,
  L_K = L_ham + L_therm + L_fric
where
  L_ham = p dq - omega^2 q dp    (antisymmetric in L^2(mu))
  L_therm = (p^2 - kT)/Q dxi    (antisymmetric in L^2(mu))
  L_fric = -g(xi) p dp           (NOT antisymmetric)

The friction term L_fric = -g(xi) p dp is not antisymmetric because:
  <f, g(xi) p dp h>_mu != -<g(xi) p dp f, h>_mu

Integration by parts in L^2(mu) for the p variable:
  <f, p dp h>_mu = int f * p * (dh/dp) * rho dp dq dxi
  = -int h * d/dp(f * p * rho) dp ...
  = -int h * [dp(f) * p * rho + f * rho + f * p * (-p/kT) * rho] dp
  = -int h * [p dp(f) * rho + f * rho - f * p^2/kT * rho] dp
  = -<p dp(f), h>_mu - <f, h>_mu + <f p^2/kT, h>_mu

So: <f, p dp h>_mu = -<p dp f, h>_mu - <f, h>_mu + (1/kT)<f*p^2, h>_mu

This means p dp is NOT antisymmetric. In fact, in the Hermite basis:
  (p dp)^dagger = -p dp - 1 + p^2/kT

The key point: the Koopman operator eigenvalues CAN have both signs of Re(lambda)
in a finite truncation. The correct approach is to compute the eigenvalues and
look at the ones governing decay of physical observables.

Actually, let me reconsider the whole approach. The issue is more fundamental.
The NH system is NOT a gradient system — it's a deterministic ODE with no noise.
The "spectral gap" concept is subtle for such systems.

For deterministic dynamics, the decay of correlations is governed by:
  C(t) = <f(x(t)) g(x(0))>_mu = <f, e^{L_K t} g>_mu

The rate of decay is determined by the eigenvalues of L_K with largest (least negative)
real part. For the system to be mixing, we need all non-zero eigenvalues to have
Re(lambda) < 0.

For standard NH on 1D HO, ALL eigenvalues are purely imaginary — consistent with
our diagnostic showing 512 eigenvalues with |Re| < 1e-8 for linear g.

For tanh NH, some eigenvalues acquire negative real parts — the system becomes
partially mixing. BUT the truncation also creates spurious positive Re parts.

The correct fix: use the SYMMETRIZED operator. Define:
  H = (L_K + L_K^dagger) / 2  (symmetric part)
  A = (L_K - L_K^dagger) / 2  (antisymmetric part)

The decay rate is governed by H. For a properly truncated operator, the eigenvalues
of L_K should come in conjugate pairs and should satisfy Re(lambda) <= 0 as N -> inf.

In practice, for a finite truncation, let's:
1. Build L_K correctly
2. Check that L_K + L_K^T has non-positive eigenvalues (this would confirm dissipation)
3. Use the eigenvalues of L_K directly, but identify the spectral gap as the
   smallest |Re(lambda)| among eigenvalues with Re(lambda) < 0

ALTERNATIVE APPROACH: Use the CORRELATION FUNCTION method directly.
Instead of diagonalizing L, compute C_qq(t) = <q(t) q(0)>_mu by propagating
the observable q forward in time using the matrix exponential:
  C(t) = e_q^T exp(L t) e_q
where e_q is the basis vector for the q observable.

This sidesteps the eigenvalue identification problem and directly gives the
decay rate from the autocorrelation function.
"""

import numpy as np
from scipy import linalg
from scipy.special import roots_hermite
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def build_koopman_matrix(N, omega=1.0, kT=1.0, Q=1.0, g_func=None):
    """Build the Koopman operator matrix in the orthonormal Hermite basis.

    The Koopman operator is L_K f = v . grad(f) where v is the NH velocity field.
    We represent it in L^2(mu) where mu is the invariant Gaussian measure.

    Returns L such that d/dt <psi_i(t)> = sum_j L_{ij} <psi_j(t)>
    i.e., L_{ij} = <psi_i | L_K | psi_j>_mu

    Actually, since psi are orthonormal in L^2(mu), the matrix element is:
    L_{ij} = <psi_i, L_K psi_j>_mu = integral psi_i * (L_K psi_j) d mu
    """
    sigma_q = np.sqrt(kT / omega**2)
    sigma_p = np.sqrt(kT)
    sigma_xi = np.sqrt(kT / Q)

    ntot = N**3

    def idx(n, l, k):
        return n * N * N + l * N + k

    # Gauss-Hermite quadrature for g(xi) matrix elements
    n_quad = max(4 * N, 60)
    x_phys, w_phys = roots_hermite(n_quad)
    x_normal = x_phys * np.sqrt(2)
    w_normal = w_phys / np.sqrt(np.pi)

    # Probabilist's Hermite polynomials (normalized)
    He = np.zeros((N, n_quad))
    He[0, :] = 1.0
    if N > 1:
        He[1, :] = x_normal
    for n in range(2, N):
        He[n, :] = x_normal * He[n-1, :] - (n-1) * He[n-2, :]

    factorials = np.ones(N)
    for n in range(1, N):
        factorials[n] = factorials[n-1] * n
    sqrt_fact = np.sqrt(factorials)

    psi = He / sqrt_fact[:, np.newaxis]

    # g(xi) matrix: G_{k,k'} = <psi_k | g(sigma_xi * x) | psi_k'>
    g_vals = g_func(sigma_xi * x_normal)
    G = np.zeros((N, N))
    for k in range(N):
        for kp in range(N):
            G[k, kp] = np.sum(w_normal * psi[k] * g_vals * psi[kp])

    # Build the matrix
    L = np.zeros((ntot, ntot))

    for np_ in range(N):
        for lp in range(N):
            for kp in range(N):
                col = idx(np_, lp, kp)

                # Term 1: p d/dq = sigma_p * y * (1/sigma_q) d/dx
                # On psi_{np_}(x): d/dx psi_{np_} = sqrt(np_) psi_{np_-1}
                # On psi_{lp}(y): y * psi_{lp} = sqrt(lp+1) psi_{lp+1} + sqrt(lp) psi_{lp-1}
                c1 = sigma_p / sigma_q
                if np_ >= 1:
                    snp = np.sqrt(np_)
                    if lp + 1 < N:
                        L[idx(np_-1, lp+1, kp), col] += c1 * np.sqrt(lp+1) * snp
                    if lp >= 1:
                        L[idx(np_-1, lp-1, kp), col] += c1 * np.sqrt(lp) * snp

                # Term 2: -omega^2 q d/dp = -omega^2 sigma_q x * (1/sigma_p) d/dy
                # On psi_{lp}(y): d/dy psi_{lp} = sqrt(lp) psi_{lp-1}
                # On psi_{np_}(x): x psi_{np_} = sqrt(np_+1) psi_{np_+1} + sqrt(np_) psi_{np_-1}
                c2 = -omega**2 * sigma_q / sigma_p
                if lp >= 1:
                    slp = np.sqrt(lp)
                    if np_ + 1 < N:
                        L[idx(np_+1, lp-1, kp), col] += c2 * np.sqrt(np_+1) * slp
                    if np_ >= 1:
                        L[idx(np_-1, lp-1, kp), col] += c2 * np.sqrt(np_) * slp

                # Term 3: -g(xi) p d/dp = -g(sigma_xi z) * y d/dy
                # y d/dy psi_l = sqrt(l) y psi_{l-1} = sqrt(l)[sqrt(l) psi_l + sqrt(l-1) psi_{l-2}]
                #              = l psi_l + sqrt(l(l-1)) psi_{l-2}
                for k in range(N):
                    gkk = G[k, kp]
                    if abs(gkk) < 1e-15:
                        continue
                    # -g * l' * psi_{np_, lp, k}
                    L[idx(np_, lp, k), col] += -gkk * lp
                    # -g * sqrt(lp(lp-1)) * psi_{np_, lp-2, k}
                    if lp >= 2:
                        L[idx(np_, lp-2, k), col] += -gkk * np.sqrt(lp * (lp-1))

                # Term 4: (p^2 - kT)/Q d/dxi = (kT(y^2 - 1))/Q * (1/sigma_xi) d/dz
                # y^2 psi_l = sqrt((l+1)(l+2)) psi_{l+2} + (2l+1) psi_l + sqrt(l(l-1)) psi_{l-2}
                # (y^2 - 1) psi_l = sqrt((l+1)(l+2)) psi_{l+2} + 2l psi_l + sqrt(l(l-1)) psi_{l-2}
                # d/dz psi_k = sqrt(k) psi_{k-1}
                c4 = kT / (Q * sigma_xi)
                if kp >= 1:
                    skp = np.sqrt(kp)
                    if lp + 2 < N:
                        L[idx(np_, lp+2, kp-1), col] += c4 * np.sqrt((lp+1)*(lp+2)) * skp
                    L[idx(np_, lp, kp-1), col] += c4 * 2 * lp * skp
                    if lp >= 2:
                        L[idx(np_, lp-2, kp-1), col] += c4 * np.sqrt(lp*(lp-1)) * skp

    return L


def build_adjoint_matrix(N, omega=1.0, kT=1.0, Q=1.0, g_func=None):
    """Build L_K^dagger in L^2(mu).

    L_K^dagger = -L_K - div(v) where div(v) = -g(xi) + 0 + 0 for NH.
    Wait, let's compute properly.

    div(v) = d(p)/dq + d(-omega^2 q - g(xi)p)/dp + d((p^2-kT)/Q)/dxi
           = 0 + (-g(xi)) + 0 = -g(xi)

    The Fokker-Planck operator is: L_FP rho = -div(v rho) = -v.grad(rho) - div(v) rho
    So L_FP = -L_K + g(xi)    [as multiplication operator]

    In L^2(mu), the adjoint of L_K satisfies:
    <f, L_K g>_mu = <L_K^dag f, g>_mu

    Integration by parts gives:
    L_K^dag = -L_K - div(v) = -L_K + g(xi)

    Wait, this isn't right either. The adjoint in L^2(mu) is different from L^2(dx).

    For the invariant measure mu, L_FP^* mu = 0, which means:
    <1, L_K f>_mu = integral L_K f d mu = integral div(v * f * rho) dx ...

    Let me just compute L_K^dag numerically as L.T (since the basis is orthonormal
    in L^2(mu), the adjoint is just the matrix transpose).
    """
    # Since the basis is orthonormal in L^2(mu), L_K^dag matrix = L^T
    return None  # Just use L.T


def correlation_decay(L, obs_idx, t_max=50, n_points=500):
    """Compute C(t) = <obs(t) obs(0)>_mu by matrix exponential.

    obs_idx: index of the observable in the basis (e.g., q -> psi_{1,0,0})

    C(t) = e_{obs}^T exp(L*t) e_{obs}

    For efficiency, diagonalize L first: L = V D V^{-1}
    Then exp(Lt) = V exp(Dt) V^{-1}
    """
    eigs, V = linalg.eig(L)
    Vinv = linalg.inv(V)

    # Coefficients
    c_left = Vinv @ np.eye(L.shape[0])[:, obs_idx]  # V^{-1} e_obs
    c_right = V[obs_idx, :]  # e_obs^T V

    ts = np.linspace(0, t_max, n_points)
    C = np.zeros(n_points)

    for i, t in enumerate(ts):
        # C(t) = e_obs^T V diag(exp(eig*t)) V^{-1} e_obs
        #       = sum_k c_right[k] * exp(eig[k]*t) * c_left[k]
        C[i] = np.real(np.sum(c_right * np.exp(eigs * t) * c_left))

    return ts, C


def extract_decay_rate(ts, C, C0_frac=0.5):
    """Extract the decay rate from a correlation function.

    Finds the time at which |C(t)| drops to C0_frac * C(0) and estimates the rate.
    Also fits an exponential envelope.
    """
    C0 = C[0]
    if abs(C0) < 1e-15:
        return 0.0

    # Use the envelope: take |C(t)|
    absC = np.abs(C)

    # Find crossings of C0_frac * C0
    target = C0_frac * abs(C0)
    crossings = np.where(absC < target)[0]
    if len(crossings) == 0:
        return 0.0

    t_half = ts[crossings[0]]
    if t_half < 1e-10:
        return 0.0

    rate = -np.log(C0_frac) / t_half
    return rate


def g_tanh(alpha):
    return lambda xi: np.tanh(alpha * xi)

def g_losc(xi):
    return 2 * xi / (1 + xi**2)

def g_linear(xi):
    return xi


def compute_spectral_gap_from_eigenvalues(L):
    """Compute spectral gap as smallest |Re(lambda)| for Re(lambda) < 0."""
    eigs = linalg.eigvals(L)
    neg = eigs[eigs.real < -1e-8]
    if len(neg) == 0:
        return 0.0, eigs
    gap = np.min(np.abs(neg.real))
    return gap, eigs


def compute_spectral_gap_from_symmetric_part(L):
    """Compute the spectral gap from the symmetric part S = (L + L^T)/2.

    The symmetric part governs the energy dissipation rate:
    d/dt ||f||^2 = 2 <f, L f> = 2 <f, S f>

    The spectral gap of S (largest non-zero eigenvalue, since S <= 0) gives
    the rate of L^2 decay.
    """
    S = 0.5 * (L + L.T)
    eigs_S = linalg.eigvalsh(S)  # S is symmetric, use eigvalsh

    # S should be negative semi-definite for a dissipative system
    # The spectral gap is the largest (least negative) non-zero eigenvalue
    neg_eigs = eigs_S[eigs_S < -1e-8]
    if len(neg_eigs) == 0:
        return 0.0, eigs_S

    gap = np.min(np.abs(neg_eigs))  # smallest |eigenvalue| = least dissipative mode
    return gap, eigs_S


def run_nh_simulation(omega=1.0, kT=1.0, Q=1.0, g_func=None, alpha=1.0,
                      dt=0.005, n_steps=500000, n_skip=10):
    """Run NH simulation, return autocorrelation time for q."""
    if g_func is None:
        g_func = g_tanh(alpha)

    sigma_q = np.sqrt(kT / omega**2)
    q = np.random.randn() * sigma_q
    p = np.random.randn() * np.sqrt(kT)
    xi = np.random.randn() * np.sqrt(kT / Q)

    n_samples = n_steps // n_skip
    q_samples = np.zeros(n_samples)

    for step in range(n_steps):
        # Velocity Verlet NH
        xi += 0.5 * dt * (p**2 - kT) / Q
        p *= np.exp(-g_func(xi) * 0.5 * dt)
        p -= 0.5 * dt * omega**2 * q
        q += dt * p
        p -= 0.5 * dt * omega**2 * q
        p *= np.exp(-g_func(xi) * 0.5 * dt)
        xi += 0.5 * dt * (p**2 - kT) / Q

        if step % n_skip == 0:
            q_samples[step // n_skip] = q

    tau = compute_tau_int(q_samples)
    return tau, q_samples


def compute_tau_int(x, max_lag=None):
    """Integrated autocorrelation time via FFT."""
    x = x - np.mean(x)
    n = len(x)
    if max_lag is None:
        max_lag = n // 4
    var = np.var(x)
    if var < 1e-15:
        return 1.0
    fft_x = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(fft_x * np.conj(fft_x)).real[:n] / (n * var)
    tau = 0.5
    for t in range(1, max_lag):
        if acf[t] < 0:
            break
        tau += acf[t]
    return tau


def main():
    print("=" * 70)
    print("SPECTRAL GAP v2 — KOOPMAN OPERATOR + CORRELATION DECAY")
    print("=" * 70)

    # ===== Step 1: Verify antisymmetry of Hamiltonian + thermostat parts =====
    print("\n--- Antisymmetry check ---")
    N = 10

    # Hamiltonian only (g=0)
    g_zero = lambda xi: 0 * xi
    L_ham = build_koopman_matrix(N, g_func=g_zero, Q=1.0)
    S_ham = 0.5 * (L_ham + L_ham.T)
    print(f"||S_ham|| = {np.linalg.norm(S_ham):.6e}  (should be ~0 for antisymmetric Hamiltonian)")

    # Actually, the thermostat term (p^2-kT)/Q d/dxi is also present with g=0
    # Let's check: is (p^2-kT)/Q d/dxi antisymmetric?
    # <f, (p^2-kT) dxi g>_mu = integral f (p^2-kT) dg/dxi rho dq dp dxi
    # Integrate by parts in xi: = -integral g d/dxi[f (p^2-kT) rho] dq dp dxi
    # = -integral g [(p^2-kT) df/dxi rho + f(p^2-kT)(-Qxi/kT)rho] dq dp dxi
    # = -<(p^2-kT) dxi f, g>_mu + integral f g (p^2-kT) Q xi/kT rho dq dp dxi
    # The second term: <f g (p^2-kT) Q xi/kT>_mu
    # For the thermostat term T = (p^2-kT)/(Q) * dxi:
    # <f, T g>_mu = -<T f, g>_mu + <f g (p^2-kT) xi/kT>_mu
    # So T is NOT antisymmetric! There's a correction term.

    # Hmm, let me just check the symmetric part of L for different g functions.

    # Full operator with tanh
    for alpha in [0.5, 1.0, 2.0]:
        L = build_koopman_matrix(N, g_func=g_tanh(alpha), Q=1.0)
        S = 0.5 * (L + L.T)
        eigs_S = linalg.eigvalsh(S)
        print(f"  tanh(α={alpha}): S eigenvalues: min={eigs_S[0]:.4f}, max={eigs_S[-1]:.4f}, "
              f"#pos={np.sum(eigs_S > 1e-8)}, #neg={np.sum(eigs_S < -1e-8)}")

    # ===== Step 2: Use CORRELATION DECAY approach =====
    print("\n--- Correlation decay approach ---")
    print("Computing C_qq(t) = <q(t) q(0)>_mu via matrix exponential")

    # The q observable in the orthonormal basis is psi_{1,0,0} (He_1 = x, normalized)
    N = 12

    def idx(n, l, k):
        return n * N * N + l * N + k

    q_idx = idx(1, 0, 0)  # q/sigma_q observable
    p_idx = idx(0, 1, 0)  # p/sigma_p observable

    alphas = [0.25, 0.5, 0.75, 1.0, np.sqrt(2), 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    Qs = [0.1, 0.3, 1.0, 3.0, 10.0]

    results = {}

    for Q in Qs:
        print(f"\n=== Q = {Q} ===")
        print(f"  Kac prediction: α_opt = √(2/Q) = {np.sqrt(2.0/Q):.4f}")

        for alpha in alphas:
            L = build_koopman_matrix(N, Q=Q, g_func=g_tanh(alpha))

            # Method 1: eigenvalue spectral gap
            gap_eig, eigs = compute_spectral_gap_from_eigenvalues(L)

            # Method 2: symmetric part spectral gap
            gap_sym, _ = compute_spectral_gap_from_symmetric_part(L)

            # Method 3: correlation decay rate
            ts, Cq = correlation_decay(L, q_idx, t_max=100, n_points=1000)
            rate_q = extract_decay_rate(ts, Cq, C0_frac=1/np.e)

            results[('tanh', alpha, Q)] = {
                'gap_eig': gap_eig, 'gap_sym': gap_sym, 'rate_q': rate_q,
                'eigs': eigs
            }
            print(f"  tanh(α={alpha:5.3f}): gap_eig={gap_eig:.4e}, gap_sym={gap_sym:.4e}, rate_q={rate_q:.4e}")

        # Log-oscillator
        L = build_koopman_matrix(N, Q=Q, g_func=g_losc)
        gap_eig, eigs = compute_spectral_gap_from_eigenvalues(L)
        gap_sym, _ = compute_spectral_gap_from_symmetric_part(L)
        ts, Cq = correlation_decay(L, q_idx, t_max=100, n_points=1000)
        rate_q = extract_decay_rate(ts, Cq, C0_frac=1/np.e)
        results[('losc', 2.0, Q)] = {
            'gap_eig': gap_eig, 'gap_sym': gap_sym, 'rate_q': rate_q,
            'eigs': eigs
        }
        print(f"  log-osc(g'=2):     gap_eig={gap_eig:.4e}, gap_sym={gap_sym:.4e}, rate_q={rate_q:.4e}")

    # ===== Step 3: Summary table with correlation-based rates =====
    print("\n\n" + "=" * 90)
    print("SUMMARY: Correlation decay rate (rate_q) vs α for each Q")
    print("=" * 90)
    print(f"{'Q':>6s}", end="")
    for alpha in alphas:
        print(f"  α={alpha:5.2f}", end="")
    print(f"  {'losc':>8s}  {'α_opt':>6s}  {'Kac':>6s}")
    print("-" * (8 + 10 * len(alphas) + 25))

    for Q in Qs:
        print(f"{Q:6.1f}", end="")
        rates = []
        for alpha in alphas:
            r = results[('tanh', alpha, Q)]['rate_q']
            rates.append((alpha, r))
            print(f"  {r:8.4f}", end="")
        losc_r = results[('losc', 2.0, Q)]['rate_q']
        best_alpha = max(rates, key=lambda x: x[1])
        kac = np.sqrt(2.0 / Q)
        print(f"  {losc_r:8.4f}  {best_alpha[0]:6.3f}  {kac:6.3f}")

    # ===== Step 4: Empirical validation =====
    print("\n\n--- Empirical validation ---")
    np.random.seed(42)
    print(f"{'type':>6s}  {'α':>5s}  {'Q':>5s}  {'rate_q(theory)':>14s}  {'1/τ_int(emp)':>14s}  {'ratio':>8s}")
    print("-" * 65)

    for alpha, Q in [(1.0, 1.0), (np.sqrt(2), 1.0), (2.0, 1.0), (2.0, 0.3), (1.0, 0.1)]:
        rate = results[('tanh', alpha, Q)]['rate_q']
        tau, _ = run_nh_simulation(alpha=alpha, Q=Q, dt=0.005, n_steps=1000000)
        dt_sample = 0.005 * 10  # n_skip=10
        tau_real = tau * dt_sample  # convert to real time
        emp_rate = 1.0 / tau_real if tau_real > 0 else 0
        ratio = rate / emp_rate if emp_rate > 0 else 0
        print(f"{'tanh':>6s}  {alpha:5.3f}  {Q:5.1f}  {rate:14.4e}  {emp_rate:14.4e}  {ratio:8.4f}")

    for Q in [0.3, 1.0]:
        rate = results[('losc', 2.0, Q)]['rate_q']
        tau, _ = run_nh_simulation(g_func=g_losc, Q=Q, dt=0.005, n_steps=1000000)
        dt_sample = 0.005 * 10
        tau_real = tau * dt_sample
        emp_rate = 1.0 / tau_real if tau_real > 0 else 0
        ratio = rate / emp_rate if emp_rate > 0 else 0
        print(f"{'losc':>6s}  {'2.0':>5s}  {Q:5.1f}  {rate:14.4e}  {emp_rate:14.4e}  {ratio:8.4f}")

    # ===== Step 5: Key comparison: losc vs tanh at g'(0)=2 =====
    print("\n\n--- Log-osc vs tanh(2ξ): spectral gap ratio ---")
    print(f"{'Q':>6s}  {'rate_losc':>10s}  {'rate_tanh2':>10s}  {'ratio':>8s}")
    for Q in Qs:
        r_losc = results[('losc', 2.0, Q)]['rate_q']
        r_tanh = results[('tanh', 2.0, Q)]['rate_q']
        ratio = r_losc / r_tanh if r_tanh > 0 else float('inf')
        print(f"{Q:6.1f}  {r_losc:10.4e}  {r_tanh:10.4e}  {ratio:8.4f}")

    # ===== Step 6: Generate figures =====
    generate_figures(results, alphas, Qs)

    return results, alphas, Qs


def generate_figures(results, alphas, Qs):
    """Generate publication-quality figures."""

    fig_dir = "/Users/wujiewang/code/det-sampler/.worktrees/spectral-gap-cert-075/orbits/spectral-gap-cert-075/figures"
    import os
    os.makedirs(fig_dir, exist_ok=True)

    # Figure 1: rate_q vs α for each Q
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(Qs)))

    for i, Q in enumerate(Qs):
        rates = [results[('tanh', a, Q)]['rate_q'] for a in alphas]
        ax.plot(alphas, rates, 'o-', color=colors[i], label=f'Q={Q}', markersize=4)
        # Mark Kac prediction
        kac = np.sqrt(2.0 / Q)
        ax.axvline(kac, color=colors[i], linestyle=':', alpha=0.5)
        # Mark log-osc
        losc_rate = results[('losc', 2.0, Q)]['rate_q']
        ax.plot(2.0, losc_rate, 's', color=colors[i], markersize=8,
                markeredgecolor='red', markeredgewidth=1.5)

    ax.set_xlabel(r"$\alpha$ = g'(0)", fontsize=12)
    ax.set_ylabel(r"Correlation decay rate $\gamma_q$", fontsize=12)
    ax.set_title("Spectral gap vs damping strength (1D HO, ω=1, kT=1)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/rate_vs_alpha.png", dpi=150)
    print(f"\nSaved: {fig_dir}/rate_vs_alpha.png")
    plt.close()

    # Figure 2: Eigenvalue spectrum in complex plane for select cases
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cases = [
        (('tanh', 1.0, 1.0), r'tanh($\xi$), Q=1'),
        (('tanh', 2.0, 1.0), r'tanh($2\xi$), Q=1'),
        (('losc', 2.0, 1.0), r'log-osc, Q=1'),
    ]
    for ax, (key, title) in zip(axes, cases):
        eigs = results[key]['eigs']
        ax.scatter(eigs.real, eigs.imag, s=3, alpha=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_xlabel('Re(λ)')
        ax.set_ylabel('Im(λ)')
        ax.set_title(title)
        ax.set_xlim(-5, 5)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Eigenvalue spectrum of Koopman operator (N=12)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/eigenvalue_spectrum.png", dpi=150)
    print(f"Saved: {fig_dir}/eigenvalue_spectrum.png")
    plt.close()

    # Figure 3: Correlation function decay for select cases
    N = 12
    def idx_fn(n, l, k): return n*N*N + l*N + k
    q_idx = idx_fn(1, 0, 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel a: vary alpha at Q=1
    ax = axes[0]
    for alpha in [0.5, 1.0, np.sqrt(2), 2.0, 4.0]:
        L = build_koopman_matrix(N, Q=1.0, g_func=g_tanh(alpha))
        ts, Cq = correlation_decay(L, q_idx, t_max=80, n_points=500)
        ax.plot(ts, Cq / Cq[0], label=f'α={alpha:.2f}')
    # log-osc
    L = build_koopman_matrix(N, Q=1.0, g_func=g_losc)
    ts, Cq = correlation_decay(L, q_idx, t_max=80, n_points=500)
    ax.plot(ts, Cq / Cq[0], 'k--', linewidth=2, label='log-osc')
    ax.set_xlabel('t')
    ax.set_ylabel(r'$C_{qq}(t) / C_{qq}(0)$')
    ax.set_title('Q = 1.0')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.2, 1.2)

    # Panel b: vary Q at alpha=2
    ax = axes[1]
    for Q in [0.1, 0.3, 1.0, 3.0]:
        L = build_koopman_matrix(N, Q=Q, g_func=g_tanh(2.0))
        ts, Cq = correlation_decay(L, q_idx, t_max=80, n_points=500)
        ax.plot(ts, Cq / Cq[0], label=f'Q={Q}')
    ax.set_xlabel('t')
    ax.set_ylabel(r'$C_{qq}(t) / C_{qq}(0)$')
    ax.set_title(r'$\alpha$ = 2.0 (tanh)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.2, 1.2)

    fig.suptitle("Autocorrelation function decay (1D HO)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/correlation_decay.png", dpi=150)
    print(f"Saved: {fig_dir}/correlation_decay.png")
    plt.close()


if __name__ == "__main__":
    results, alphas, Qs = main()
