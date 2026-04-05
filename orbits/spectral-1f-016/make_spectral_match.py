"""Spectral Matching: Tune Q values to maximize friction PSD at barrier-crossing frequency.

Estimate the GMM barrier-crossing timescale from the inter-mode distance and barrier height.
Choose Q values to place Lorentzian peaks at that frequency.
Compare spectral-matched vs log-spaced on GMM (5 seeds, 2M evals).

Reference: Dutta & Horn, Rev. Mod. Phys. 53, 497 (1981).
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from research.eval.potentials import GaussianMixture2D
from research.eval.integrators import ThermostatState


def g_func(xi_val):
    return 2.0 * xi_val / (1.0 + xi_val**2)


def run_gmm_multiscale(Qs, n_evals=2_000_000, dt=0.03, seed=42, kT=1.0):
    """Run multi-scale log-osc on GMM, return positions for KL estimation."""
    potential = GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5)
    dim = 2
    mass = 1.0
    n_thermo = len(Qs)

    rng = np.random.default_rng(seed)
    q = np.array([3.0, 0.0])  # start near first mode
    p = rng.normal(0, np.sqrt(mass * kT), size=dim)
    xi = np.zeros(n_thermo)

    half_dt = 0.5 * dt
    grad_U = potential.gradient(q)
    n_eval_count = 1

    positions = []
    record_every = 10

    step = 0
    while n_eval_count < n_evals:
        # Velocity Verlet
        kinetic = np.sum(p**2) / mass
        drive = kinetic - dim * kT
        xi_dot = np.array([drive / Q for Q in Qs])
        xi = xi + half_dt * xi_dot

        total_g = sum(g_func(xi[k]) for k in range(n_thermo))
        scale = np.exp(-total_g * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale
        p = p - half_dt * grad_U

        q = q + dt * p / mass

        if np.any(np.isnan(q)) or np.any(np.isnan(p)):
            break

        grad_U = potential.gradient(q)
        n_eval_count += 1

        p = p - half_dt * grad_U
        total_g = sum(g_func(xi[k]) for k in range(n_thermo))
        scale = np.exp(-total_g * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale

        kinetic = np.sum(p**2) / mass
        drive = kinetic - dim * kT
        xi_dot = np.array([drive / Q for Q in Qs])
        xi = xi + half_dt * xi_dot

        step += 1
        if step % record_every == 0:
            positions.append(q.copy())

    return np.array(positions)


def estimate_kl_gmm(positions, potential, kT=1.0, n_bins=50):
    """Estimate KL divergence for GMM using histogram method."""
    if len(positions) < 100:
        return 999.0

    # Burn-in: discard first 10%
    burn = len(positions) // 10
    pos = positions[burn:]

    # 2D histogram
    x_range = (-5, 5)
    y_range = (-5, 5)
    H_emp, xedges, yedges = np.histogram2d(
        pos[:, 0], pos[:, 1], bins=n_bins, range=[x_range, y_range], density=True
    )

    # True density on same grid
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    xx, yy = np.meshgrid(xc, yc, indexing='ij')

    H_true = np.zeros_like(H_emp)
    for i in range(n_bins):
        for j in range(n_bins):
            q = np.array([xx[i, j], yy[i, j]])
            H_true[i, j] = np.exp(-potential.energy(q) / kT)

    # Normalize
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    H_true = H_true / (np.sum(H_true) * dx * dy)

    # KL divergence
    mask = (H_emp > 0) & (H_true > 0)
    kl = np.sum(H_emp[mask] * np.log(H_emp[mask] / H_true[mask])) * dx * dy
    return max(kl, 0.0)


def main():
    print("=" * 70)
    print("Spectral Matching: GMM barrier-crossing frequency")
    print("=" * 70)

    potential = GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5)

    # Estimate barrier-crossing timescale
    # Inter-mode distance: modes on circle of radius 3, angular separation = 2*pi/5
    # Distance between adjacent modes: 2 * 3 * sin(pi/5) ~ 3.53
    # Barrier height: energy at midpoint between two modes
    # The saddle between two adjacent modes is approximately at the midpoint
    mode1 = potential.centers[0]
    mode2 = potential.centers[1]
    midpoint = 0.5 * (mode1 + mode2)
    E_saddle = potential.energy(midpoint)
    E_mode = potential.energy(mode1)
    barrier = E_saddle - E_mode
    print(f"  Inter-mode distance: {np.linalg.norm(mode2 - mode1):.3f}")
    print(f"  Barrier height: {barrier:.3f} kT")
    print(f"  Kramers rate ~ exp(-barrier) ~ {np.exp(-barrier):.4f}")

    # Barrier crossing frequency estimate
    # Kramers theory: rate ~ (omega_well * omega_barrier / (2*pi*gamma)) * exp(-barrier/kT)
    # For simplicity, the crossing timescale is tau_cross ~ exp(barrier)
    # The frequency to match is f_cross ~ 1/tau_cross
    tau_cross = np.exp(barrier)  # in simulation time units
    f_cross = 1.0 / tau_cross
    print(f"  Estimated crossing frequency: {f_cross:.4f} Hz")
    print(f"  Estimated crossing timescale: {tau_cross:.1f} time units")

    # Spectral-matched Q: place Lorentzian corner at f_cross
    # Corner frequency of thermostat k: f_k ~ 1/sqrt(Q_k)
    # For Lorentzian peak at f_cross: Q_matched ~ 1/f_cross^2
    Q_matched_center = 1.0 / f_cross**2
    print(f"  Q for barrier-crossing match: {Q_matched_center:.2f}")

    # Spectral-matched config: 3 thermostats centered on barrier frequency
    # One at the crossing frequency, one faster, one slower
    Q_spectral = [Q_matched_center / 10.0, Q_matched_center, Q_matched_center * 10.0]
    Q_spectral = [max(0.01, q) for q in Q_spectral]  # floor

    # Log-spaced baseline (champion from multiscale-chain-009)
    Q_logspaced = [0.1, 0.7, 10.0]

    # Wide log-spaced
    Q_wide = [0.01, 3.162, 1000.0]

    configs = {
        'spectral_matched': Q_spectral,
        'log_spaced_champion': Q_logspaced,
        'wide_log_spaced': Q_wide,
    }

    seeds = [42, 123, 7, 999, 314]
    n_evals = 2_000_000
    dt = 0.03

    results = {}
    for name, Qs in configs.items():
        print(f"\n  Config: {name}, Qs={[f'{q:.3f}' for q in Qs]}")
        kls = []
        for seed in seeds:
            pos = run_gmm_multiscale(Qs, n_evals=n_evals, dt=dt, seed=seed)
            kl = estimate_kl_gmm(pos, potential)
            kls.append(kl)
            print(f"    seed={seed}: KL={kl:.4f} ({len(pos)} samples)")

        results[name] = {
            'Qs': Qs,
            'kls': kls,
            'mean_kl': float(np.mean(kls)),
            'std_kl': float(np.std(kls)),
        }
        print(f"    Mean KL: {np.mean(kls):.4f} +/- {np.std(kls):.4f}")

    # Save results
    outdir = os.path.dirname(os.path.abspath(__file__))
    outpath = os.path.join(outdir, 'spectral_match_results.json')
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outpath}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary: Spectral Matching vs Log-Spaced")
    print("=" * 70)
    for name, r in results.items():
        print(f"  {name:25s}: KL={r['mean_kl']:.4f}+/-{r['std_kl']:.4f}  "
              f"Qs={[f'{q:.3f}' for q in r['Qs']]}")

    return results


if __name__ == '__main__':
    main()
