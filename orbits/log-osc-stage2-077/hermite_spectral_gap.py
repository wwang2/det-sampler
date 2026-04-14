"""Part B: Hermite spectral gap computation for NH Liouville operator.

For 1D HO with various g(xi) friction functions, build the Liouville operator
in a Hermite polynomial basis and compute the spectral gap.

The NH generator on observables in L^2(mu):
  L f = p df/dq - omega^2 q df/dp - g(xi) p df/dp + (p^2-kT)/Q df/dxi

Orthonormal basis of L^2(mu):
  psi_{n,l,k} = He_n(q/s_q) He_l(p/s_p) He_k(xi/s_xi) / sqrt(n! l! k!)

Coupling rules:
  x * psi_n = sqrt(n+1)*psi_{n+1} + sqrt(n)*psi_{n-1}
  d/dx psi_n = sqrt(n)*psi_{n-1}  (times 1/sigma)

The spectral gap = min |Re(lambda)| over non-zero eigenvalues.
Note: NH dynamics have time-reversal symmetry, so eigenvalues come in +/- Re pairs.
"""

import numpy as np
from math import factorial, sqrt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import time
import json

ORBIT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ORBIT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def gauss_hermite_quadrature(n_points=80):
    """Gauss-Hermite quadrature for integrals against exp(-x^2/2)."""
    pts_phys, wts_phys = np.polynomial.hermite.hermgauss(n_points)
    pts = pts_phys * np.sqrt(2)
    wts = wts_phys * np.sqrt(2)
    return pts, wts


def hermite_values(n_max, x):
    """Probabilist's Hermite polynomials He_0..He_{n_max} at points x."""
    x = np.asarray(x, dtype=float)
    H = np.zeros((n_max + 1, len(x)))
    H[0] = 1.0
    if n_max >= 1:
        H[1] = x
    for n in range(2, n_max + 1):
        H[n] = x * H[n - 1] - (n - 1) * H[n - 2]
    return H


def compute_g_overlap(g_func, N, sigma_xi, n_quad=80):
    """Overlap matrix G[k',k] = <psi_{k'} | g(sigma_xi * x) | psi_k> in orthonormal Hermite basis."""
    pts, wts = gauss_hermite_quadrature(n_quad)
    He = hermite_values(N - 1, pts)
    g_vals = np.array([g_func(sigma_xi * x) for x in pts])
    G_raw = np.zeros((N, N))
    for i in range(len(pts)):
        G_raw += wts[i] * np.outer(He[:, i], He[:, i]) * g_vals[i]
    norms = np.array([sqrt(factorial(k)) for k in range(N)])
    G = G_raw / (np.outer(norms, norms) * sqrt(2 * np.pi))
    return G


def build_liouville_dense(N, omega, kT, Q, g_func, n_quad=80):
    """Build the Liouville operator L as a dense matrix.

    L[target, source] = coefficient of psi_target in L * psi_source.
    """
    dim = N ** 3
    s_q = sqrt(kT) / omega
    s_p = sqrt(kT)
    s_xi = sqrt(kT / Q)
    G = compute_g_overlap(g_func, N, s_xi, n_quad)

    L = np.zeros((dim, dim))

    def idx(n, l, k):
        return n * N * N + l * N + k

    for n in range(N):
        for l in range(N):
            for k in range(N):
                src = idx(n, l, k)

                # T1: p * d/dq
                if n >= 1:
                    c1 = (s_p / s_q) * sqrt(n)
                    if l + 1 < N:
                        L[idx(n-1, l+1, k), src] += c1 * sqrt(l+1)
                    if l >= 1:
                        L[idx(n-1, l-1, k), src] += c1 * sqrt(l)

                # T2: -omega^2 * q * d/dp
                if l >= 1:
                    c2 = -omega**2 * (s_q / s_p) * sqrt(l)
                    if n + 1 < N:
                        L[idx(n+1, l-1, k), src] += c2 * sqrt(n+1)
                    if n >= 1:
                        L[idx(n-1, l-1, k), src] += c2 * sqrt(n)

                # T3: -g(xi) * p * d/dp
                # p * d/dp psi_l = l*psi_l + sqrt(l(l-1))*psi_{l-2}
                # g(xi) * psi_k = sum_{k'} G[k',k] * psi_{k'}
                if l >= 1:
                    for kp in range(N):
                        gkk = G[kp, k]
                        if abs(gkk) < 1e-18:
                            continue
                        L[idx(n, l, kp), src] += -l * gkk
                        if l >= 2:
                            L[idx(n, l-2, kp), src] += -sqrt(l*(l-1)) * gkk

                # T4: (p^2-kT)/Q * d/dxi
                # (p^2-kT) = kT * He_2(p/s_p) in orthonorm:
                # He_2 * psi_l = sqrt((l+1)(l+2))*psi_{l+2} + 2l*psi_l + sqrt(l(l-1))*psi_{l-2}
                # d/dxi psi_k = (1/s_xi)*sqrt(k)*psi_{k-1}
                if k >= 1:
                    c4 = (kT / (Q * s_xi)) * sqrt(k)
                    if l + 2 < N:
                        L[idx(n, l+2, k-1), src] += c4 * sqrt((l+1)*(l+2))
                    L[idx(n, l, k-1), src] += c4 * 2.0 * l
                    if l >= 2:
                        L[idx(n, l-2, k-1), src] += c4 * sqrt(l*(l-1))

    return L


def spectral_gap(L):
    """Spectral gap = min |Re(lambda)| over non-zero eigenvalues."""
    evals = np.linalg.eigvals(L)
    re = np.abs(np.real(evals))
    re_nonzero = re[re > 1e-6]
    if len(re_nonzero) == 0:
        return 0.0, evals
    gap = float(np.min(re_nonzero))

    # Also find the negative eigenvalue closest to 0 (dissipative gap)
    re_neg = [-np.real(e) for e in evals if np.real(e) < -1e-6]
    neg_gap = min(re_neg) if re_neg else 0.0

    return gap, evals


def main():
    print("=" * 80)
    print("PART B: Hermite Spectral Gap Computation")
    print("=" * 80)

    # Use N=8 (dim=512) — dense eigensolve is fast and well-converged
    N = 8
    omega = 1.0
    kT = 1.0

    alphas = [0.5, 1.0, np.sqrt(2), 2.0, 3.0, 4.0]
    Qs = [0.1, 1.0, 10.0]

    def g_tanh(alpha):
        return lambda xi: np.tanh(alpha * xi)

    def g_losc(xi):
        return 2.0 * xi / (1.0 + xi * xi)

    print(f"\nBasis size: N={N} per dimension, total dim={N**3}")

    # Convergence check
    print("\nConvergence check (Q=1, tanh(sqrt(2)*xi)):")
    for Ntest in [4, 6, 8, 10]:
        L = build_liouville_dense(Ntest, omega, kT, 1.0, g_tanh(np.sqrt(2)))
        gap, _ = spectral_gap(L)
        print(f"  N={Ntest:2d} (dim={Ntest**3:5d}): gap = {gap:.6f}")

    all_results = {}

    for Q in Qs:
        kac = np.sqrt(2.0 * omega**2 / Q)
        print(f"\n{'='*60}")
        print(f"Q = {Q}")
        print(f"  Kac prediction: g'(0)_opt = sqrt(2*omega^2/Q) = {kac:.3f}")
        print(f"{'='*60}")

        results_Q = {}

        for alpha in alphas:
            print(f"\n  tanh(alpha={alpha:.3f})...", end=" ", flush=True)
            t0 = time.time()
            L = build_liouville_dense(N, omega, kT, Q, g_tanh(alpha))
            gap, evals = spectral_gap(L)
            elapsed = time.time() - t0

            # Find lambda closest to 0 with negative real part
            neg_evals = [e for e in evals if np.real(e) < -1e-6]
            neg_gap = min(-np.real(e) for e in neg_evals) if neg_evals else 0.0

            results_Q[f"tanh_{alpha:.3f}"] = {
                "spectral_gap": gap,
                "neg_gap": neg_gap,
                "alpha": alpha,
            }
            print(f"gap = {gap:.6f}  neg_gap = {neg_gap:.6f}  ({elapsed:.1f}s)")
            sys.stdout.flush()

        print(f"\n  log-osc...", end=" ", flush=True)
        t0 = time.time()
        L = build_liouville_dense(N, omega, kT, Q, g_losc)
        gap, evals = spectral_gap(L)
        elapsed = time.time() - t0
        neg_evals = [e for e in evals if np.real(e) < -1e-6]
        neg_gap = min(-np.real(e) for e in neg_evals) if neg_evals else 0.0
        results_Q["losc"] = {
            "spectral_gap": gap,
            "neg_gap": neg_gap,
        }
        print(f"gap = {gap:.6f}  neg_gap = {neg_gap:.6f}  ({elapsed:.1f}s)")
        sys.stdout.flush()

        all_results[Q] = results_Q

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("\n\n" + "=" * 100)
    print(f"SUMMARY: Spectral gap min|Re(lambda)|, N={N}")
    print("=" * 100)

    print(f"\n{'Method':<20}", end="")
    for Q in Qs:
        print(f"  Q={Q:<8}", end="")
    print()
    print("-" * (20 + 12 * len(Qs)))

    for alpha in alphas:
        key = f"tanh_{alpha:.3f}"
        print(f"tanh(a={alpha:.3f})   ", end="")
        for Q in Qs:
            gap = all_results[Q][key]["spectral_gap"]
            print(f"  {gap:<10.6f}", end="")
        print()

    print(f"{'log-osc':<20}", end="")
    for Q in Qs:
        gap = all_results[Q]["losc"]["spectral_gap"]
        print(f"  {gap:<10.6f}", end="")
    print()

    print(f"\n{'Kac optimal g(0)':<20}", end="")
    for Q in Qs:
        print(f"  {np.sqrt(2*omega**2/Q):<10.3f}", end="")
    print()

    # Ratios
    print("\n\nSpectral gap ratio log-osc / tanh(alpha=1):")
    for Q in Qs:
        gap_losc = all_results[Q]["losc"]["spectral_gap"]
        gap_tanh1 = all_results[Q]["tanh_1.000"]["spectral_gap"]
        ratio = gap_losc / gap_tanh1 if gap_tanh1 > 1e-10 else float('nan')
        print(f"  Q={Q}: {ratio:.3f}")

    print("\nSpectral gap ratio log-osc / best_tanh:")
    for Q in Qs:
        gap_losc = all_results[Q]["losc"]["spectral_gap"]
        best_gap = max(all_results[Q][f"tanh_{alpha:.3f}"]["spectral_gap"] for alpha in alphas)
        best_alpha = max(alphas, key=lambda a: all_results[Q][f"tanh_{a:.3f}"]["spectral_gap"])
        ratio = gap_losc / best_gap if best_gap > 1e-10 else float('nan')
        print(f"  Q={Q}: {ratio:.3f} (best tanh alpha={best_alpha:.3f})")

    # Figures
    generate_figures(all_results, alphas, Qs, omega, N)

    # Save
    out_path = os.path.join(ORBIT_DIR, "results_spectral.json")
    # Convert for JSON
    save_results = {}
    for Q in all_results:
        save_results[str(Q)] = {}
        for key, val in all_results[Q].items():
            save_results[str(Q)][key] = {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                          for k, v in val.items()}
    with open(out_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return all_results


def generate_figures(all_results, alphas, Qs, omega, N):
    """Generate spectral gap plots."""

    # Figure 1: gap vs alpha for each Q
    fig, axes = plt.subplots(1, len(Qs), figsize=(5 * len(Qs), 5))
    if len(Qs) == 1:
        axes = [axes]

    for ax, Q in zip(axes, Qs):
        gaps_tanh = [all_results[Q][f"tanh_{alpha:.3f}"]["spectral_gap"] for alpha in alphas]
        gap_losc = all_results[Q]["losc"]["spectral_gap"]
        kac_opt = np.sqrt(2 * omega**2 / Q)

        ax.plot(alphas, gaps_tanh, 'bo-', ms=7, lw=1.5, label='tanh($\\alpha\\xi$)')
        ax.axhline(gap_losc, color='red', ls='--', lw=2,
                   label=f'log-osc (gap={gap_losc:.4f})')
        ax.axvline(kac_opt, color='green', ls=':', alpha=0.6,
                   label=f'Kac $\\sqrt{{2/Q}}$={kac_opt:.2f}')

        ax.set_xlabel("$\\alpha$ = g'(0)", fontsize=12)
        ax.set_ylabel("Spectral gap min$|\\mathrm{Re}(\\lambda)|$", fontsize=12)
        ax.set_title(f"Q = {Q}", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Hermite Spectral Gap: 1D HO, N={N} basis", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "spectral_gap_vs_alpha.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: figures/spectral_gap_vs_alpha.png")

    # Figure 2: heatmap + comparison bar
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    gap_matrix = np.zeros((len(alphas), len(Qs)))
    for i, alpha in enumerate(alphas):
        for j, Q in enumerate(Qs):
            gap_matrix[i, j] = all_results[Q][f"tanh_{alpha:.3f}"]["spectral_gap"]

    im = ax.imshow(gap_matrix, aspect='auto', cmap='viridis', origin='lower')
    ax.set_xticks(range(len(Qs)))
    ax.set_xticklabels([str(Q) for Q in Qs])
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels([f"{a:.2f}" for a in alphas])
    ax.set_xlabel("Q", fontsize=12)
    ax.set_ylabel("alpha", fontsize=12)
    ax.set_title("Spectral gap: tanh(alpha*xi)", fontsize=13)
    plt.colorbar(im, ax=ax, label="gap")
    for i in range(len(alphas)):
        for j in range(len(Qs)):
            ax.text(j, i, f"{gap_matrix[i,j]:.4f}", ha='center', va='center',
                    fontsize=7, color='white' if gap_matrix[i,j] < np.median(gap_matrix) else 'black')

    ax = axes[1]
    losc_gaps = [all_results[Q]["losc"]["spectral_gap"] for Q in Qs]
    best_tanh_gaps = [max(all_results[Q][f"tanh_{alpha:.3f}"]["spectral_gap"]
                         for alpha in alphas) for Q in Qs]
    x = np.arange(len(Qs))
    width = 0.35
    ax.bar(x - width/2, best_tanh_gaps, width, label='Best tanh', color='#1f77b4')
    ax.bar(x + width/2, losc_gaps, width, label='Log-osc', color='#d62728')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q={Q}" for Q in Qs])
    ax.set_ylabel("Spectral gap", fontsize=12)
    ax.set_title("Best tanh vs log-osc", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "spectral_gap_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: figures/spectral_gap_heatmap.png")


if __name__ == "__main__":
    main()
