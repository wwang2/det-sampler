"""Spectral analysis of multi-scale thermostat friction signals.

Computes the power spectral density of the total friction signal G(t)
for multi-scale thermostats with different numbers of thermostat variables
and different spacing strategies (linear vs logarithmic).

Also computes the autocorrelation time of position for the 2D double-well
potential under single-scale and multi-scale thermostats.

Reference:
    Van der Ziel, A. (1950). On the noise spectra of semi-conductor noise
    and of flicker effect. Physica 16, 359-372.
    https://doi.org/10.1016/S0065-2539(08)60768-4
"""

import numpy as np
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Multi-scale thermostat dynamics (1D HO)
# ============================================================

def g_log_osc(xi):
    """Log-osc friction: g(xi) = 2*xi/(1+xi^2)."""
    return 2.0 * xi / (1.0 + xi**2)


def simulate_multiscale_1d_ho(Q_list, omega=1.0, kT=1.0, m=1.0, dt=0.005,
                                n_steps=500_000, seed=42):
    """Simulate 1D HO with multiple independent log-osc thermostats.

    Each thermostat j has mass Q_j and variable xi_j.
    Total friction: G(t) = sum_j g(xi_j(t)).

    Returns: (times, q_traj, G_traj, xi_trajs)
    """
    rng = np.random.default_rng(seed)
    N_thermo = len(Q_list)

    q = rng.normal(0, np.sqrt(kT / omega**2))
    p = rng.normal(0, np.sqrt(m * kT))
    xi = np.zeros(N_thermo)

    # Storage (subsample)
    subsample = 10
    n_store = n_steps // subsample
    q_traj = np.zeros(n_store)
    G_traj = np.zeros(n_store)
    xi_trajs = np.zeros((N_thermo, n_store))

    store_idx = 0
    for step in range(n_steps):
        # RK4 integration
        def rhs_full(state):
            q_s, p_s = state[0], state[1]
            xi_s = state[2:]
            G = np.sum(g_log_osc(xi_s))
            dq = p_s / m
            dp = -omega**2 * q_s - G * p_s
            K = p_s**2 / m
            dxi = (K - kT) / np.array(Q_list)  # d=1
            return np.concatenate([[dq, dp], dxi])

        state = np.concatenate([[q, p], xi])
        k1 = rhs_full(state)
        k2 = rhs_full(state + 0.5 * dt * k1)
        k3 = rhs_full(state + 0.5 * dt * k2)
        k4 = rhs_full(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        q, p = state[0], state[1]
        xi = state[2:]

        if step % subsample == 0 and store_idx < n_store:
            q_traj[store_idx] = q
            G_traj[store_idx] = np.sum(g_log_osc(xi))
            xi_trajs[:, store_idx] = xi
            store_idx += 1

    times = np.arange(n_store) * subsample * dt
    return times, q_traj, G_traj, xi_trajs


def compute_psd(signal, dt_eff):
    """Compute one-sided power spectral density using FFT."""
    n = len(signal)
    signal_centered = signal - np.mean(signal)
    fft_vals = np.fft.rfft(signal_centered)
    psd = (2.0 * dt_eff / n) * np.abs(fft_vals)**2
    freqs = np.fft.rfftfreq(n, d=dt_eff)
    return freqs, psd


def compute_iat(signal, max_lag=None):
    """Compute integrated autocorrelation time."""
    n = len(signal)
    if max_lag is None:
        max_lag = n // 10
    signal_centered = signal - np.mean(signal)
    var = np.var(signal)
    if var < 1e-15:
        return np.inf

    tau = 1.0
    for lag in range(1, max_lag):
        acf = np.mean(signal_centered[:n-lag] * signal_centered[lag:]) / var
        if acf < 0.05:
            break
        tau += 2.0 * acf
    return tau


# ============================================================
# Double-well simulation for mixing comparison
# ============================================================

def simulate_multiscale_double_well(Q_list, kT=1.0, m=1.0, dt=0.01,
                                      n_steps=1_000_000, seed=42):
    """Simulate 2D double-well with multi-scale log-osc thermostat.

    U(x,y) = (x^2 - 1)^2 + 0.5*y^2

    Returns: (times, x_traj, barrier_crossings)
    """
    rng = np.random.default_rng(seed)
    N_thermo = len(Q_list)
    dim = 2

    q = np.array([1.0, 0.0])  # start in one well
    p = rng.normal(0, np.sqrt(m * kT), size=dim)
    xi = np.zeros(N_thermo)

    subsample = 10
    n_store = n_steps // subsample
    x_traj = np.zeros(n_store)

    store_idx = 0
    for step in range(n_steps):
        def rhs_dw(state):
            q_s = state[:dim]
            p_s = state[dim:2*dim]
            xi_s = state[2*dim:]
            x, y = q_s
            grad_U = np.array([4.0 * x * (x**2 - 1), 2.0 * 0.5 * y])
            G = np.sum(g_log_osc(xi_s))
            dq = p_s / m
            dp = -grad_U - G * p_s
            K = np.sum(p_s**2) / m
            dxi = (K - dim * kT) / np.array(Q_list)
            return np.concatenate([dq, dp, dxi])

        state = np.concatenate([q, p, xi])
        k1 = rhs_dw(state)
        k2 = rhs_dw(state + 0.5 * dt * k1)
        k3 = rhs_dw(state + 0.5 * dt * k2)
        k4 = rhs_dw(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        q = state[:dim]
        p = state[dim:2*dim]
        xi = state[2*dim:]

        if step % subsample == 0 and store_idx < n_store:
            x_traj[store_idx] = q[0]
            store_idx += 1

    # Count barrier crossings (x changes sign)
    crossings = np.sum(np.diff(np.sign(x_traj)) != 0)

    times = np.arange(n_store) * subsample * dt
    return times, x_traj, crossings


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(out_dir, "figures")

    kT = 1.0
    seed = 42

    # ============================================================
    # Part 1: Spectral density of friction signal (1D HO)
    # ============================================================
    print("="*60)
    print("Part 1: Spectral density of multi-scale friction")
    print("="*60)

    configs = {
        "Single (Q=1)": [1.0],
        "3 log-spaced": [0.1, 1.0, 10.0],
        "3 lin-spaced": [1.0, 5.5, 10.0],
        "5 log-spaced": list(np.logspace(-1, 1, 5)),
        "7 log-spaced": list(np.logspace(-1, 1.5, 7)),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))

    for i, (name, Q_list) in enumerate(configs.items()):
        print(f"\n  Config: {name}, Q = {[f'{q:.2f}' for q in Q_list]}")
        times, q_traj, G_traj, xi_trajs = simulate_multiscale_1d_ho(
            Q_list, dt=0.005, n_steps=500_000, seed=seed
        )
        dt_eff = times[1] - times[0]

        # PSD
        freqs, psd = compute_psd(G_traj, dt_eff)
        # Smooth PSD with log-binning
        mask = freqs > 0
        log_f = np.log10(freqs[mask])
        log_p = np.log10(psd[mask] + 1e-30)
        n_bins = 50
        bins = np.linspace(log_f.min(), log_f.max(), n_bins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_means = np.zeros(n_bins)
        for b in range(n_bins):
            in_bin = (log_f >= bins[b]) & (log_f < bins[b+1])
            if np.any(in_bin):
                bin_means[b] = np.mean(log_p[in_bin])
            else:
                bin_means[b] = np.nan

        valid = np.isfinite(bin_means)
        axes[0].plot(10**bin_centers[valid], 10**bin_means[valid],
                     color=colors[i], label=name, linewidth=2, alpha=0.8)

        # IAT of position
        iat = compute_iat(q_traj)
        print(f"    IAT(q) = {iat:.1f}")

    # Reference 1/f line
    f_ref = np.logspace(-2, 1, 100)
    axes[0].plot(f_ref, 0.1 / f_ref, "k--", alpha=0.4, label=r"$1/f$ reference")

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Frequency", fontsize=14)
    axes[0].set_ylabel("PSD of friction signal G(t)", fontsize=14)
    axes[0].set_title("Spectral Density: Multi-Scale Thermostat", fontsize=14)
    axes[0].legend(fontsize=10, loc="upper right")
    axes[0].tick_params(labelsize=12)
    axes[0].grid(True, alpha=0.3)

    # Part 2: IAT comparison
    print("\n" + "="*60)
    print("Part 2: Barrier crossing comparison (2D double-well)")
    print("="*60)

    dw_configs = {
        "Single Q=1": [1.0],
        "3 log": [0.1, 1.0, 10.0],
        "5 log": list(np.logspace(-1, 1, 5)),
        "7 log": list(np.logspace(-1, 1.5, 7)),
        "3 lin": [1.0, 5.5, 10.0],
    }

    names_dw = []
    crossings_dw = []
    iats_dw = []

    for name, Q_list in dw_configs.items():
        print(f"\n  Config: {name}")
        times, x_traj, crossings = simulate_multiscale_double_well(
            Q_list, dt=0.01, n_steps=500_000, seed=seed
        )
        iat = compute_iat(x_traj)
        print(f"    Barrier crossings: {crossings}")
        print(f"    IAT(x): {iat:.1f}")
        names_dw.append(name)
        crossings_dw.append(crossings)
        iats_dw.append(iat)

    x_pos = np.arange(len(names_dw))
    axes[1].bar(x_pos, crossings_dw, color=plt.cm.tab10(np.linspace(0, 0.5, len(names_dw))),
                edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(names_dw, fontsize=11, rotation=15)
    axes[1].set_ylabel("Barrier crossings (x=0)", fontsize=14)
    axes[1].set_title("Multi-Scale Thermostat: Double-Well Mixing", fontsize=14)
    axes[1].tick_params(labelsize=12)
    axes[1].grid(True, alpha=0.3, axis="y")

    fig_path = os.path.join(fig_dir, "spectral_multiscale.png")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved figure: {fig_path}")

    # ============================================================
    # Part 3: Phase portraits for different thermostats
    # ============================================================
    print("\n" + "="*60)
    print("Part 3: Phase portraits (1D HO)")
    print("="*60)

    # Define thermostat friction functions for phase portraits
    phase_pots = [
        {"name": "NH (quadratic)", "color": "#1f77b4",
         "g": lambda xi, Q: xi / Q, "dV": lambda xi: xi},
        {"name": "Log-Osc", "color": "#2ca02c",
         "g": lambda xi, Q: 2.0*xi/((1.0+xi**2)*Q), "dV": lambda xi: 2.0*xi/(1.0+xi**2)},
        {"name": "Tanh", "color": "#d62728",
         "g": lambda xi, Q: np.tanh(xi)/Q, "dV": lambda xi: np.tanh(xi)},
        {"name": "Arctan", "color": "#9467bd",
         "g": lambda xi, Q: np.arctan(xi)/Q, "dV": lambda xi: np.arctan(xi)},
    ]

    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))
    Q_phase = 1.0
    dt_phase = 0.005
    n_steps_phase = 500_000

    for idx, pot in enumerate(phase_pots):
        ax = axes3[idx // 2, idx % 2]
        rng = np.random.default_rng(seed)
        q_val = rng.normal(0, np.sqrt(kT))
        p_val = rng.normal(0, np.sqrt(kT))
        xi_val = 0.0

        q_arr = np.zeros(n_steps_phase // 10)
        p_arr = np.zeros(n_steps_phase // 10)

        g_func = pot["g"]

        for step in range(n_steps_phase):
            state = np.array([q_val, p_val, xi_val])

            # RK4
            def rhs_phase(s, _g=g_func, _Q=Q_phase, _kT=kT):
                q_s, p_s, xi_s = s
                gv = _g(xi_s, _Q)
                return np.array([p_s, -q_s - gv * p_s, (p_s**2 - _kT) / _Q])

            k1 = rhs_phase(state)
            k2 = rhs_phase(state + 0.5 * dt_phase * k1)
            k3 = rhs_phase(state + 0.5 * dt_phase * k2)
            k4 = rhs_phase(state + dt_phase * k3)
            state = state + (dt_phase / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            q_val, p_val, xi_val = state

            if step % 10 == 0:
                si = step // 10
                q_arr[si] = q_val
                p_arr[si] = p_val

        ax.scatter(q_arr[::5], p_arr[::5], s=0.1, alpha=0.3, color=pot["color"])

        # Expected Gaussian contours
        theta = np.linspace(0, 2*np.pi, 200)
        for sigma_mult in [1, 2, 3]:
            ax.plot(sigma_mult * np.cos(theta), sigma_mult * np.sin(theta),
                    "k-", alpha=0.3, linewidth=0.5)

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.set_xlabel("q", fontsize=12)
        ax.set_ylabel("p", fontsize=12)
        ax.set_title(f"{pot['name']} (Q={Q_phase})", fontsize=14)
        ax.tick_params(labelsize=11)

    fig3_path = os.path.join(fig_dir, "phase_portraits.png")
    fig3.tight_layout()
    fig3.savefig(fig3_path, dpi=150)
    plt.close(fig3)
    print(f"Saved figure: {fig3_path}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nDouble-well barrier crossings:")
    for name, xc, iat in zip(names_dw, crossings_dw, iats_dw):
        print(f"  {name:15s}: crossings={xc:5d}, IAT={iat:.1f}")


if __name__ == "__main__":
    main()
