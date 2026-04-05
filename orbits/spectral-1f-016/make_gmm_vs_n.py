"""GMM KL divergence vs number of scales N.

Run multi-scale log-osc with N=1,2,3,4,5,7,10 scales on GMM,
measure KL divergence with 1M force evals each.
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from research.eval.potentials import GaussianMixture2D


def g_func(xi_val):
    return 2.0 * xi_val / (1.0 + xi_val**2)


def get_log_spaced_Qs(n_scales, Q_min=0.01, Q_max=1000.0):
    if n_scales == 1:
        return [np.sqrt(Q_min * Q_max)]
    return list(np.logspace(np.log10(Q_min), np.log10(Q_max), n_scales))


def run_gmm(Qs, n_evals=1_000_000, dt=0.03, seed=42, kT=1.0):
    potential = GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5)
    dim = 2
    mass = 1.0
    n_thermo = len(Qs)

    rng = np.random.default_rng(seed)
    q = np.array([3.0, 0.0])
    p = rng.normal(0, np.sqrt(mass * kT), size=dim)
    xi = np.zeros(n_thermo)

    half_dt = 0.5 * dt
    grad_U = potential.gradient(q)
    n_eval_count = 1
    positions = []
    record_every = 5

    step = 0
    while n_eval_count < n_evals:
        kinetic = np.sum(p**2) / mass
        drive = kinetic - dim * kT
        xi_dot = np.array([drive / Q for Q in Qs])
        xi = xi + half_dt * xi_dot

        total_g = sum(g_func(xi[k]) for k in range(n_thermo))
        scale = np.clip(np.exp(-total_g * half_dt), 1e-10, 1e10)
        p = p * scale
        p = p - half_dt * grad_U
        q = q + dt * p / mass

        if np.any(np.isnan(q)) or np.any(np.isnan(p)):
            break

        grad_U = potential.gradient(q)
        n_eval_count += 1

        p = p - half_dt * grad_U
        total_g = sum(g_func(xi[k]) for k in range(n_thermo))
        scale = np.clip(np.exp(-total_g * half_dt), 1e-10, 1e10)
        p = p * scale

        kinetic = np.sum(p**2) / mass
        drive = kinetic - dim * kT
        xi_dot = np.array([drive / Q for Q in Qs])
        xi = xi + half_dt * xi_dot

        step += 1
        if step % record_every == 0:
            positions.append(q.copy())

    return np.array(positions)


def estimate_kl_gmm(positions, kT=1.0, n_bins=50):
    potential = GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5)
    if len(positions) < 100:
        return 999.0

    burn = len(positions) // 10
    pos = positions[burn:]

    x_range = (-5, 5)
    y_range = (-5, 5)
    H_emp, xedges, yedges = np.histogram2d(
        pos[:, 0], pos[:, 1], bins=n_bins, range=[x_range, y_range], density=True
    )

    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    xx, yy = np.meshgrid(xc, yc, indexing='ij')

    H_true = np.zeros_like(H_emp)
    for i in range(n_bins):
        for j in range(n_bins):
            q = np.array([xx[i, j], yy[i, j]])
            H_true[i, j] = np.exp(-potential.energy(q) / kT)

    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    H_true = H_true / (np.sum(H_true) * dx * dy)

    mask = (H_emp > 0) & (H_true > 0)
    kl = np.sum(H_emp[mask] * np.log(H_emp[mask] / H_true[mask])) * dx * dy
    return max(kl, 0.0)


def main():
    print("=" * 60)
    print("GMM KL vs N_scales")
    print("=" * 60)

    N_list = [1, 2, 3, 5, 7, 10]
    seeds = [42, 123, 7]
    n_evals = 1_000_000

    results = {}
    for N in N_list:
        Qs = get_log_spaced_Qs(N)
        print(f"\nN={N}, Qs={[f'{q:.3g}' for q in Qs]}")
        kls = []
        for seed in seeds:
            pos = run_gmm(Qs, n_evals=n_evals, dt=0.03, seed=seed)
            kl = estimate_kl_gmm(pos)
            kls.append(kl)
            print(f"  seed={seed}: KL={kl:.4f}")

        results[N] = {
            'Qs': Qs,
            'kls': kls,
            'mean': float(np.mean(kls)),
            'std': float(np.std(kls)),
        }
        print(f"  Mean: {np.mean(kls):.4f} +/- {np.std(kls):.4f}")

    outdir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(outdir, 'gmm_vs_n_results.json'), 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for N in N_list:
        r = results[N]
        print(f"  N={N:2d}: KL={r['mean']:.4f}+/-{r['std']:.4f}")


if __name__ == '__main__':
    main()
