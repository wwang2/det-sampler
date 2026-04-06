"""Autocorrelation time analysis for multi-scale log-osc thermostats.

Uses the 2D Gaussian Mixture (5 modes on a ring) as the test potential.
The observable is the x-coordinate (slow mode: measures mode-hopping between +-x modes).
Computes integrated autocorrelation time tau_int as a function of N.

Key insight: on multi-modal potentials, tau_int tracks barrier-crossing timescale.
1/f noise (N=3) has power at ALL frequencies including the slow barrier-crossing mode.
"""
from __future__ import annotations

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from research.eval.potentials import GaussianMixture2D, HarmonicOscillator1D


def g_func(xi_val: float) -> float:
    return 2.0 * xi_val / (1.0 + xi_val**2)


def get_log_spaced_Qs(n_scales: int, Q_min: float = 0.01, Q_max: float = 1000.0) -> list[float]:
    if n_scales == 1:
        return [float(np.sqrt(Q_min * Q_max))]
    return list(np.logspace(np.log10(Q_min), np.log10(Q_max), n_scales))


def run_gmm_trajectory(Qs: list[float], n_evals: int = 2_000_000,
                       dt: float = 0.03, seed: int = 42) -> np.ndarray:
    """Run multi-scale log-osc on 2D GMM, return x-positions (every 5 steps)."""
    potential = GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5)
    dim = 2
    kT = 1.0
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
        kinetic = float(np.sum(p**2) / mass)
        drive = kinetic - dim * kT
        xi_dot = np.array([drive / Q for Q in Qs])
        xi = xi + half_dt * xi_dot

        total_g = sum(g_func(xi[k]) for k in range(n_thermo))
        scale = float(np.clip(np.exp(-total_g * half_dt), 1e-10, 1e10))
        p = p * scale
        p = p - half_dt * grad_U
        q = q + dt * p / mass

        if np.any(np.isnan(q)) or np.any(np.isnan(p)):
            print(f"    NaN detected at step {step}, breaking")
            break

        grad_U = potential.gradient(q)
        n_eval_count += 1

        p = p - half_dt * grad_U
        total_g = sum(g_func(xi[k]) for k in range(n_thermo))
        scale = float(np.clip(np.exp(-total_g * half_dt), 1e-10, 1e10))
        p = p * scale

        kinetic = float(np.sum(p**2) / mass)
        drive = kinetic - dim * kT
        xi_dot = np.array([drive / Q for Q in Qs])
        xi = xi + half_dt * xi_dot

        step += 1
        if step % record_every == 0:
            positions.append(q.copy())

    return np.array(positions)


def run_ho_trajectory(Qs: list[float], n_steps: int = 2_000_000,
                      dt: float = 0.005, seed: int = 42) -> np.ndarray:
    """Run multi-scale log-osc on 1D HO, return q trajectory."""
    potential = HarmonicOscillator1D(omega=1.0)
    dim = 1
    kT = 1.0
    mass = 1.0
    n_thermo = len(Qs)

    rng = np.random.default_rng(seed)
    q = np.array([0.5])
    p = rng.normal(0, np.sqrt(mass * kT), size=dim)
    xi = np.zeros(n_thermo)

    q_traj = np.zeros(n_steps)
    half_dt = 0.5 * dt
    grad_U = potential.gradient(q)

    for step in range(n_steps):
        q_traj[step] = q[0]

        kinetic = float(np.sum(p**2) / mass)
        drive = kinetic - dim * kT
        xi_dot = np.array([drive / Q for Q in Qs])
        xi = xi + half_dt * xi_dot

        total_g = sum(g_func(xi[k]) for k in range(n_thermo))
        scale = max(1e-10, min(1e10, float(np.exp(-total_g * half_dt))))
        p = p * scale
        p = p - half_dt * grad_U

        q = q + dt * p / mass
        grad_U = potential.gradient(q)

        p = p - half_dt * grad_U
        total_g = sum(g_func(xi[k]) for k in range(n_thermo))
        scale = max(1e-10, min(1e10, float(np.exp(-total_g * half_dt))))
        p = p * scale

        kinetic = float(np.sum(p**2) / mass)
        drive = kinetic - dim * kT
        xi_dot = np.array([drive / Q for Q in Qs])
        xi = xi + half_dt * xi_dot

    return q_traj


def compute_autocorr(obs: np.ndarray, max_lag: int = 5000) -> np.ndarray:
    """Normalized autocorrelation C(t) for lags 0..max_lag via FFT."""
    x = obs - np.mean(obs)
    var = float(np.mean(x**2))
    if var < 1e-30:
        return np.ones(max_lag + 1)
    n = len(x)
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    xfft = np.fft.rfft(x, n=nfft)
    acf_raw = np.fft.irfft(xfft * np.conj(xfft))[:max_lag + 1].real
    counts = np.arange(n, n - max_lag - 1, -1, dtype=float)
    C = acf_raw / counts / var
    return C


def integrated_autocorr_time(C: np.ndarray) -> float:
    """tau_int = 0.5 + sum_{t=1}^{T_cut} C(t), T_cut = first zero-crossing."""
    T_cut = len(C) - 1
    for t in range(1, len(C)):
        if C[t] < 0:
            T_cut = t - 1
            break
    tau_int = 0.5 + float(np.sum(C[1:T_cut + 1]))
    return max(tau_int, 0.5)


def mode_indicator(positions: np.ndarray) -> np.ndarray:
    """Return mode index (0-4) for each position in 5-mode GMM on ring."""
    # 5 modes at angles 0, 72, 144, 216, 288 deg, radius=3
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    centers = 3.0 * np.column_stack([np.cos(angles), np.sin(angles)])  # (5, 2)
    # Assign each position to nearest mode
    diffs = positions[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (T, 5, 2)
    dists = np.sqrt(np.sum(diffs**2, axis=2))  # (T, 5)
    return np.argmin(dists, axis=1)  # (T,)


def main():
    print("=" * 70)
    print("Autocorrelation Time Analysis: Multi-scale log-osc on GMM")
    print("=" * 70)

    # GMM trajectory settings (match parent orbit make_gmm_vs_n.py)
    dt_gmm = 0.03
    n_evals_gmm = 2_000_000
    max_lag_gmm = 10000  # in units of recorded steps (every 5 evals)

    # HO trajectory settings
    dt_ho = 0.005
    n_steps_ho = 2_000_000
    max_lag_ho = 5000

    seed = 42
    N_list = [1, 2, 3, 5, 7, 10]
    highlight_Ns = [1, 3, 5]

    results: dict = {}

    # --- GMM analysis ---
    print("\n--- GMM (5-mode ring, primary) ---")
    gmm_tau = {}
    for N in N_list:
        Qs = get_log_spaced_Qs(N)
        print(f"\nN={N}, Qs={[f'{q:.4g}' for q in Qs]}")
        print(f"  Running {n_evals_gmm:,} force evals on GMM...")
        pos = run_gmm_trajectory(Qs, n_evals=n_evals_gmm, dt=dt_gmm, seed=seed)
        print(f"  Collected {len(pos)} positions")

        if len(pos) < 200:
            print("  ERROR: too few positions")
            continue

        # Use mode indicator as the primary slow observable (captures inter-mode hopping)
        modes = mode_indicator(pos).astype(float)
        # Also use x-coordinate and radius as cross-checks
        obs_x = pos[:, 0]
        obs_r = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)

        max_lag_use = min(max_lag_gmm, len(pos) // 5)
        C_mode = compute_autocorr(modes, max_lag=max_lag_use)
        C_x = compute_autocorr(obs_x, max_lag=max_lag_use)
        C_r = compute_autocorr(obs_r, max_lag=max_lag_use)

        tau_mode = integrated_autocorr_time(C_mode)
        tau_x = integrated_autocorr_time(C_x)
        tau_r = integrated_autocorr_time(C_r)

        # tau_int in units of force evaluations (recorded every 5 evals)
        tau_mode_evals = tau_mode * 5
        tau_x_evals = tau_x * 5
        tau_r_evals = tau_r * 5

        # ESS based on mode autocorr (best measure of global mixing)
        n_recorded = len(pos)
        ess_mode = n_recorded / (2 * tau_mode + 1)
        ess_per_eval = ess_mode / n_evals_gmm * 5

        # Count actual mode transitions
        mode_ints = mode_indicator(pos)
        n_mode_hops = int(np.sum(mode_ints[1:] != mode_ints[:-1]))
        hop_rate = n_mode_hops / (n_evals_gmm / 1000)

        print(f"  tau_int(mode) = {tau_mode:.1f} rec-steps = {tau_mode_evals:.0f} force-evals")
        print(f"  tau_int(x)    = {tau_x:.1f} rec-steps = {tau_x_evals:.0f} force-evals")
        print(f"  tau_int(r)    = {tau_r:.1f} rec-steps")
        print(f"  mode hops     = {n_mode_hops} ({hop_rate:.2f}/1k evals)")
        print(f"  ESS(mode)/force-eval = {ess_per_eval:.5f}")

        gmm_tau[N] = tau_mode_evals
        entry: dict = {
            'N': N,
            'Qs': Qs,
            'tau_int_mode_evals': float(tau_mode_evals),
            'tau_int_x_evals': float(tau_x_evals),
            'tau_int_r_evals': float(tau_r_evals),
            'ess_per_eval': float(ess_per_eval),
            'n_mode_hops': n_mode_hops,
            'mode_hop_rate_per_1k': float(hop_rate),
        }
        if N in highlight_Ns:
            # Subsample C for storage
            sub = max(1, len(C_mode) // 2000)
            entry['C_x'] = C_mode[::sub].tolist()
            entry['C_lag_evals'] = (np.arange(0, len(C_mode), sub) * 5).tolist()
        results[N] = entry

    # --- HO analysis (ergodicity test) ---
    print("\n--- HO (1D harmonic, ergodicity check) ---")
    ho_tau = {}
    for N in N_list:
        Qs = get_log_spaced_Qs(N)
        print(f"\nN={N}")
        q_traj = run_ho_trajectory(Qs, n_steps=n_steps_ho, dt=dt_ho, seed=seed)
        C = compute_autocorr(q_traj, max_lag=max_lag_ho)
        tau = integrated_autocorr_time(C)
        ho_tau[N] = tau
        print(f"  tau_int(HO) = {tau:.1f} steps")
        if N in results:
            results[N]['tau_int_ho'] = float(tau)
            if N in highlight_Ns:
                sub = max(1, len(C) // 2000)
                results[N]['C_ho'] = C[::sub].tolist()

    # Summary
    print("\n" + "=" * 70)
    print("Summary: tau_int vs N (GMM x-observable, in force-evals)")
    print(f"{'N':>4}  {'tau_int_gmm':>14}  {'tau_int_ho':>12}  {'ESS/eval':>10}")
    print("-" * 50)
    for N in N_list:
        if N not in results:
            continue
        r = results[N]
        print(f"{N:>4}  {r['tau_int_x_evals']:>14.1f}  {r.get('tau_int_ho', 0):>12.1f}  {r['ess_per_eval']:>10.5f}")

    # Pull alpha values from parent orbit PSD results
    psd_path = os.path.join(os.path.dirname(__file__), '..', 'spectral-1f-016', 'psd_results.json')
    alpha_by_N: dict[int, float] = {}
    if os.path.exists(psd_path):
        with open(psd_path) as f:
            psd_data = json.load(f)
        for N in N_list:
            key = str(N)
            if key in psd_data:
                alpha_by_N[N] = psd_data[key]['alpha_mid']
                if N in results:
                    results[N]['alpha'] = alpha_by_N[N]
        print("\nalpha vs N (from parent orbit):")
        for N in N_list:
            if N in alpha_by_N:
                print(f"  N={N}: alpha={alpha_by_N[N]:.3f}")

    # tau_int vs alpha summary
    if alpha_by_N:
        print("\ntau_int(mode, GMM) vs alpha:")
        pairs = []
        for N in N_list:
            if N in results and N in alpha_by_N:
                # Use mode-based tau as the primary metric
                tau_v = results[N].get('tau_int_mode_evals',
                         results[N].get('tau_int_x_evals', 0))
                alpha_v = alpha_by_N[N]
                pairs.append((alpha_v, tau_v, N))
                hops = results[N].get('mode_hop_rate_per_1k', 0)
                print(f"  alpha={alpha_v:.2f}  tau_int={tau_v:.0f}  hops/1k={hops:.2f}  (N={N})")
        results['tau_vs_alpha'] = [{'alpha': a, 'tau_int': t, 'N': n} for a, t, n in pairs]

    # Save
    outdir = os.path.dirname(os.path.abspath(__file__))
    outpath = os.path.join(outdir, 'autocorr_results.json')
    results_str = {}
    for k, v in results.items():
        results_str[str(k)] = v
    with open(outpath, 'w') as f:
        json.dump(results_str, f)
    print(f"\nResults saved to {outpath}")
    return results


if __name__ == '__main__':
    main()
