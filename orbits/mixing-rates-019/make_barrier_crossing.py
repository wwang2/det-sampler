"""Barrier crossing analysis: 1/f noise advantage grows with barrier height.

Uses 1D double-well U(q) = lambda*(q^2-1)^2 with lambda=1,2,4,8.
Barrier heights: lambda kT (at q=0, minima at q=+-1).
Counts crossings through q=0 for N=1,2,3,5 thermostat scales.

Prediction 3: The 1/f (N=3, alpha~1) advantage over Brownian (N=5, alpha~2)
grows with barrier height, because Brownian noise lacks low-freq power.
"""
from __future__ import annotations

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def g_func(xi_val: float) -> float:
    return 2.0 * xi_val / (1.0 + xi_val**2)


def get_log_spaced_Qs(n_scales: int, Q_min: float = 0.01, Q_max: float = 1000.0) -> list[float]:
    if n_scales == 1:
        return [float(np.sqrt(Q_min * Q_max))]
    return list(np.logspace(np.log10(Q_min), np.log10(Q_max), n_scales))


def dw_energy(q: float, lam: float) -> float:
    return lam * (q**2 - 1.0)**2


def dw_grad(q: float, lam: float) -> float:
    return 4.0 * lam * q * (q**2 - 1.0)


def count_barrier_crossings(Qs: list[float], lam: float, n_evals: int = 200_000,
                             dt: float = 0.01, seed: int = 42) -> tuple[int, list[float]]:
    """Run 1D double-well dynamics, count q=0 crossings.

    Returns (n_crossings, q_traj_subsampled).
    """
    dim = 1
    kT = 1.0
    mass = 1.0
    n_thermo = len(Qs)

    rng = np.random.default_rng(seed)
    q = 1.0 + 0.1 * rng.standard_normal()   # start near right minimum
    p = rng.normal(0, np.sqrt(mass * kT))
    xi = np.zeros(n_thermo)

    half_dt = 0.5 * dt
    grad_U = dw_grad(q, lam)
    eval_count = 1

    crossings = 0
    prev_sign = np.sign(q)
    q_traj = []
    record_every = 10

    step = 0
    while eval_count < n_evals:
        kinetic = p**2 / mass
        drive = kinetic - dim * kT
        xi_dot = np.array([drive / Q for Q in Qs])
        xi = xi + half_dt * xi_dot

        total_g = sum(g_func(xi[k]) for k in range(n_thermo))
        scale = float(np.clip(np.exp(-total_g * half_dt), 1e-10, 1e10))
        p = p * scale - half_dt * grad_U

        q = q + dt * p / mass

        if np.isnan(q) or np.isnan(p):
            break

        grad_U = dw_grad(q, lam)
        eval_count += 1

        p = p - half_dt * grad_U
        total_g = sum(g_func(xi[k]) for k in range(n_thermo))
        scale = float(np.clip(np.exp(-total_g * half_dt), 1e-10, 1e10))
        p = p * scale

        kinetic = p**2 / mass
        drive = kinetic - dim * kT
        xi_dot = np.array([drive / Q for Q in Qs])
        xi = xi + half_dt * xi_dot

        # Count zero crossings
        curr_sign = np.sign(q)
        if curr_sign != prev_sign and curr_sign != 0:
            crossings += 1
        prev_sign = curr_sign if curr_sign != 0 else prev_sign

        step += 1
        if step % record_every == 0:
            q_traj.append(float(q))

    return crossings, q_traj


def main():
    print("=" * 70)
    print("Barrier Crossing Analysis: 1/f advantage vs barrier height")
    print("=" * 70)

    # lambda values: barrier height = lambda kT (since U(0) - U(1) = lambda)
    lambda_list = [1, 2, 4, 8]
    N_list = [1, 2, 3, 5]
    # Use more evals for high barriers so we see enough crossings
    n_evals_by_lambda = {1: 500_000, 2: 1_000_000, 4: 2_000_000, 8: 4_000_000}
    dt = 0.01
    seeds = [42, 123, 7]

    results: dict = {
        'lambda_list': lambda_list,
        'N_list': N_list,
        'n_evals_by_lambda': n_evals_by_lambda,
        'data': {}
    }

    for lam in lambda_list:
        n_evals = n_evals_by_lambda[lam]
        print(f"\n--- Barrier height = {lam} kT (lambda={lam}, n_evals={n_evals:,}) ---")
        results['data'][str(lam)] = {}
        for N in N_list:
            Qs = get_log_spaced_Qs(N)
            crossing_counts = []
            for seed in seeds:
                n_cross, _ = count_barrier_crossings(Qs, lam=lam, n_evals=n_evals,
                                                     dt=dt, seed=seed)
                rate = n_cross / (n_evals / 1000)  # crossings per 1000 force evals
                crossing_counts.append(n_cross)
                print(f"  N={N}, seed={seed}: {n_cross} crossings "
                      f"({rate:.3f}/1k evals)")

            mean_c = float(np.mean(crossing_counts))
            std_c = float(np.std(crossing_counts))
            rate_mean = mean_c / (n_evals / 1000)
            print(f"  N={N}: mean={mean_c:.1f} +/- {std_c:.1f} "
                  f"({rate_mean:.3f}/1k evals)")
            results['data'][str(lam)][str(N)] = {
                'N': N, 'lambda': lam,
                'Qs': Qs,
                'n_evals': n_evals,
                'crossings': crossing_counts,
                'mean': mean_c,
                'std': std_c,
                'rate_per_1k_evals': rate_mean,
            }

    # Print summary table
    print("\n" + "=" * 70)
    print("Summary: crossing rate (per 1k evals)")
    header = f"{'lambda':>8}  " + "  ".join(f"N={N:>2}" for N in N_list)
    print(header)
    print("-" * 60)
    for lam in lambda_list:
        row = f"{lam:>8}  "
        for N in N_list:
            r = results['data'][str(lam)][str(N)]
            row += f"{r['rate_per_1k_evals']:>7.2f}  "
        print(row)

    # Check prediction: N=3/N=1 advantage grows with lambda
    print("\nN=3 / N=1 crossing ratio vs lambda:")
    for lam in lambda_list:
        r3 = results['data'][str(lam)]['3']['rate_per_1k_evals']
        r1 = results['data'][str(lam)]['1']['rate_per_1k_evals']
        ratio = r3 / r1 if r1 > 0 else float('inf')
        results['data'][str(lam)]['ratio_N3_N1'] = ratio
        print(f"  lambda={lam}: N=3/N=1 = {ratio:.2f}x")

    print("\nN=3 / N=5 crossing ratio vs lambda:")
    for lam in lambda_list:
        r3 = results['data'][str(lam)]['3']['rate_per_1k_evals']
        r5 = results['data'][str(lam)]['5']['rate_per_1k_evals']
        ratio = r3 / r5 if r5 > 0 else float('inf')
        results['data'][str(lam)]['ratio_N3_N5'] = ratio
        print(f"  lambda={lam}: N=3/N=5 = {ratio:.2f}x")

    # Save
    outdir = os.path.dirname(os.path.abspath(__file__))
    outpath = os.path.join(outdir, 'barrier_crossing_results.json')
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outpath}")
    return results


if __name__ == '__main__':
    main()
