"""Lyapunov exponent computation for generalized friction thermostats.

Computes the maximal Lyapunov exponent for the 1D harmonic oscillator
coupled to thermostats with different friction functions V'(xi)/Q.

Method: Benettin et al. (1980) algorithm -- evolve trajectory + tangent
vector, periodically renormalize, accumulate log of stretching.

Reference:
    Benettin, G. et al. (1980). Lyapunov characteristic exponents for
    smooth dynamical systems. Meccanica 15, 9-20.
    https://doi.org/10.1007/BF02128236
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
# Thermostat potential definitions: V(xi), V'(xi), V''(xi)
# All normalized so that g(xi) = V'(xi) when Q=1 (i.e. V'(xi)/Q).
# ============================================================

class ThermostatPotential:
    """Base class for thermostat potentials V(xi)."""
    name: str
    color: str

    def dV(self, xi):
        """V'(xi) -- the friction when Q=1."""
        raise NotImplementedError

    def d2V(self, xi):
        """V''(xi) -- needed for tangent dynamics."""
        raise NotImplementedError


class QuadraticPotential(ThermostatPotential):
    """V(xi) = xi^2/2  (standard Nose-Hoover)."""
    name = "NH (quadratic)"
    color = "#1f77b4"
    def dV(self, xi): return xi
    def d2V(self, xi): return np.ones_like(xi) if hasattr(xi, '__len__') else 1.0


class LogOscPotential(ThermostatPotential):
    """V(xi) = log(1 + xi^2)."""
    name = "Log-Osc"
    color = "#2ca02c"
    def dV(self, xi): return 2.0 * xi / (1.0 + xi**2)
    def d2V(self, xi): return 2.0 * (1.0 - xi**2) / (1.0 + xi**2)**2


class TanhPotential(ThermostatPotential):
    """V(xi) = log(cosh(xi))."""
    name = "Tanh"
    color = "#d62728"
    def dV(self, xi): return np.tanh(xi)
    def d2V(self, xi): return 1.0 / np.cosh(xi)**2


class ArctanPotential(ThermostatPotential):
    """V(xi) = xi*arctan(xi) - log(1+xi^2)/2."""
    name = "Arctan"
    color = "#9467bd"
    def dV(self, xi): return np.arctan(xi)
    def d2V(self, xi): return 1.0 / (1.0 + xi**2)


ALL_POTENTIALS = [
    QuadraticPotential(),
    LogOscPotential(),
    TanhPotential(),
    ArctanPotential(),
]


# ============================================================
# Vectorized RK4 with tangent dynamics for 1D HO
# ============================================================

def compute_lyapunov(thermo_pot, Q, omega=1.0, kT=1.0, m=1.0,
                     dt=0.01, total_time=5000.0, renorm_interval=20,
                     seed=42):
    """Compute the maximal Lyapunov exponent using Benettin's algorithm.

    Uses a combined state+tangent RK4 step. Renormalize every `renorm_interval`
    steps to prevent overflow.

    Returns: (lambda_max, lambda_trace)
    """
    rng = np.random.default_rng(seed)

    n_total = int(total_time / dt)
    burnin = n_total // 10

    # Initial conditions
    q = rng.normal(0, np.sqrt(kT / omega**2))
    p = rng.normal(0, np.sqrt(m * kT))
    xi = 0.0

    # Combined RHS: state = (q, p, xi), tangent = (dq, dp, dxi)
    def rhs_combined(q, p, xi, dq, dp, dxi):
        g = thermo_pot.dV(xi) / Q
        dg = thermo_pot.d2V(xi) / Q

        # Trajectory
        fq = p / m
        fp = -omega**2 * q - g * p
        fxi = (p**2 / m - kT) / Q

        # Tangent (Jacobian applied to tangent vector)
        tq = dp / m
        tp = -omega**2 * dq - g * dp - dg * p * dxi
        txi = 2.0 * p * dp / (m * Q)

        return fq, fp, fxi, tq, tp, txi

    def rk4_step(q, p, xi, dq, dp, dxi):
        k1 = rhs_combined(q, p, xi, dq, dp, dxi)
        h = dt * 0.5
        k2 = rhs_combined(q+h*k1[0], p+h*k1[1], xi+h*k1[2],
                           dq+h*k1[3], dp+h*k1[4], dxi+h*k1[5])
        k3 = rhs_combined(q+h*k2[0], p+h*k2[1], xi+h*k2[2],
                           dq+h*k2[3], dp+h*k2[4], dxi+h*k2[5])
        h2 = dt
        k4 = rhs_combined(q+h2*k3[0], p+h2*k3[1], xi+h2*k3[2],
                           dq+h2*k3[3], dp+h2*k3[4], dxi+h2*k3[5])

        s = dt / 6.0
        q  += s * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        p  += s * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        xi += s * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        dq  += s * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
        dp  += s * (k1[4] + 2*k2[4] + 2*k3[4] + k4[4])
        dxi += s * (k1[5] + 2*k2[5] + 2*k3[5] + k4[5])

        return q, p, xi, dq, dp, dxi

    # Burn-in (trajectory only, no tangent tracking)
    dq, dp_t, dxi_t = 0.0, 0.0, 0.0
    for _ in range(burnin):
        q, p, xi, dq, dp_t, dxi_t = rk4_step(q, p, xi, dq, dp_t, dxi_t)
        if not (np.isfinite(q) and np.isfinite(p) and np.isfinite(xi)):
            return np.nan, np.array([np.nan])

    # Reset tangent vector
    tv = rng.normal(size=3)
    nrm = np.sqrt(tv[0]**2 + tv[1]**2 + tv[2]**2)
    dq, dp_t, dxi_t = tv[0]/nrm, tv[1]/nrm, tv[2]/nrm

    # Main Lyapunov computation
    n_main = n_total - burnin
    n_renorm = n_main // renorm_interval
    log_stretching = np.zeros(n_renorm)

    for i in range(n_renorm):
        for _ in range(renorm_interval):
            q, p, xi, dq, dp_t, dxi_t = rk4_step(q, p, xi, dq, dp_t, dxi_t)

        # Check finiteness
        if not (np.isfinite(q) and np.isfinite(p) and np.isfinite(xi)):
            return np.nan, np.full(n_renorm, np.nan)

        nrm = np.sqrt(dq**2 + dp_t**2 + dxi_t**2)
        if nrm < 1e-300 or not np.isfinite(nrm):
            return np.nan, np.full(n_renorm, np.nan)

        log_stretching[i] = np.log(nrm)
        dq /= nrm
        dp_t /= nrm
        dxi_t /= nrm

    # Cumulative average
    cumsum = np.cumsum(log_stretching)
    times = np.arange(1, n_renorm + 1) * renorm_interval * dt
    lambda_trace = cumsum / times

    return lambda_trace[-1], lambda_trace


# ============================================================
# Main: scan Q values for each thermostat potential
# ============================================================

def main():
    omega = 1.0
    kT = 1.0
    m = 1.0
    dt = 0.01
    total_time = 5000.0  # moderate run for each Q
    seed = 42

    Q_values = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])

    out_dir = os.path.dirname(os.path.abspath(__file__))

    results = {}
    for pot in ALL_POTENTIALS:
        print(f"\n{'='*60}")
        print(f"Thermostat: {pot.name}")
        print(f"{'='*60}")
        lambdas = []
        for Q in Q_values:
            lam, trace = compute_lyapunov(
                pot, Q, omega=omega, kT=kT, m=m,
                dt=dt, total_time=total_time, seed=seed
            )
            lambdas.append(lam)
            status = f"{lam:.4f}" if np.isfinite(lam) else "FAILED"
            print(f"  Q={Q:5.1f}  lambda_max={status}")
        results[pot.name] = np.array(lambdas)

    # Save numerical results
    results_path = os.path.join(out_dir, "lyapunov_results.txt")
    with open(results_path, "w") as f:
        f.write("# Maximal Lyapunov exponent for 1D HO + thermostat\n")
        f.write(f"# omega={omega}, kT={kT}, m={m}, dt={dt}, total_time={total_time}, seed={seed}\n")
        f.write(f"# Q values: {list(Q_values)}\n\n")
        for name, lams in results.items():
            f.write(f"{name}:\n")
            for Q, lam in zip(Q_values, lams):
                f.write(f"  Q={Q:.1f}: lambda_max={lam:.6f}\n")
            f.write("\n")

    # ============================================================
    # Plot 1: Lyapunov exponent vs Q
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 6))

    for pot in ALL_POTENTIALS:
        lams = results[pot.name]
        mask = np.isfinite(lams)
        ax.plot(Q_values[mask], lams[mask], "o-", color=pot.color,
                label=pot.name, linewidth=2, markersize=6)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label=r"$\lambda_{\max}=0$ (non-ergodic)")
    ax.set_xlabel("Thermostat mass Q", fontsize=14)
    ax.set_ylabel(r"$\lambda_{\max}$ (maximal Lyapunov exponent)", fontsize=14)
    ax.set_title("Ergodicity: Lyapunov Exponents for 1D HO + Thermostat", fontsize=16)
    ax.set_xscale("log")
    ax.legend(fontsize=12, loc="upper right")
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)

    fig_path = os.path.join(out_dir, "figures", "lyapunov_vs_Q.png")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved figure: {fig_path}")

    # ============================================================
    # Plot 2: Convergence trace + friction functions
    # ============================================================
    Q_example = 1.0
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

    for pot in ALL_POTENTIALS:
        _, trace = compute_lyapunov(
            pot, Q_example, omega=omega, kT=kT, m=m,
            dt=dt, total_time=total_time, seed=seed
        )
        n_renorm = len(trace)
        times = np.arange(1, n_renorm + 1) * 20 * dt
        mask = np.isfinite(trace)
        if np.any(mask):
            axes[0].plot(times[mask], trace[mask], color=pot.color,
                         label=pot.name, linewidth=1.5, alpha=0.8)

    axes[0].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Time", fontsize=14)
    axes[0].set_ylabel(r"Running $\lambda_{\max}$", fontsize=14)
    axes[0].set_title(f"Lyapunov Convergence (Q={Q_example})", fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].tick_params(labelsize=12)
    axes[0].grid(True, alpha=0.3)

    # Plot friction functions
    xi_range = np.linspace(-5, 5, 500)
    for pot in ALL_POTENTIALS:
        g_vals = pot.dV(xi_range)
        axes[1].plot(xi_range, g_vals, color=pot.color, label=pot.name,
                     linewidth=2)

    axes[1].set_xlabel(r"$\xi$", fontsize=14)
    axes[1].set_ylabel(r"$g(\xi) = V'(\xi)$ (Q=1)", fontsize=14)
    axes[1].set_title("Friction Functions", fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].tick_params(labelsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-3, 3)

    fig2_path = os.path.join(out_dir, "figures", "lyapunov_convergence.png")
    fig2.tight_layout()
    fig2.savefig(fig2_path, dpi=150)
    plt.close(fig2)
    print(f"Saved figure: {fig2_path}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*60)
    print("SUMMARY: Fraction of Q values with lambda_max > 0.01")
    print("="*60)
    for pot in ALL_POTENTIALS:
        lams = results[pot.name]
        valid = np.isfinite(lams)
        positive = lams[valid] > 0.01
        frac = np.sum(positive) / np.sum(valid) if np.sum(valid) > 0 else 0
        print(f"  {pot.name:20s}: {frac:.0%} ({np.sum(positive)}/{np.sum(valid)})")


if __name__ == "__main__":
    main()
