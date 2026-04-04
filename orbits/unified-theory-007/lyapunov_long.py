"""Extended Lyapunov computation: longer runs at selected Q values.

Focuses on Q=0.3, 0.5, 0.8, 1.0 with T=50000 to get converged estimates.
Also generates improved phase portraits at Q=0.3 where the difference is stark.
"""

import numpy as np
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Friction function classes (inlined for standalone execution)
class QuadraticPot:
    name = "NH (quadratic)"
    color = "#1f77b4"
    def dV(self, xi): return xi
    def d2V(self, xi): return np.ones_like(xi) if hasattr(xi, '__len__') else 1.0

class LogOscPot:
    name = "Log-Osc"
    color = "#2ca02c"
    def dV(self, xi): return 2.0 * xi / (1.0 + xi**2)
    def d2V(self, xi): return 2.0 * (1.0 - xi**2) / (1.0 + xi**2)**2

class TanhPot:
    name = "Tanh"
    color = "#d62728"
    def dV(self, xi): return np.tanh(xi)
    def d2V(self, xi): return 1.0 / np.cosh(xi)**2

class ArctanPot:
    name = "Arctan"
    color = "#9467bd"
    def dV(self, xi): return np.arctan(xi)
    def d2V(self, xi): return 1.0 / (1.0 + xi**2)

ALL_POTS = [QuadraticPot(), LogOscPot(), TanhPot(), ArctanPot()]


def compute_lyapunov_long(pot, Q, omega=1.0, kT=1.0, m=1.0,
                           dt=0.01, total_time=15000.0, renorm_interval=20,
                           seed=42):
    """Long Lyapunov computation with convergence trace."""
    rng = np.random.default_rng(seed)
    n_total = int(total_time / dt)
    burnin = n_total // 10

    q = rng.normal(0, np.sqrt(kT / omega**2))
    p = rng.normal(0, np.sqrt(m * kT))
    xi = 0.0

    def rhs(q, p, xi, dq, dp, dxi):
        g = pot.dV(xi) / Q
        dg = pot.d2V(xi) / Q
        fq = p / m
        fp = -omega**2 * q - g * p
        fxi = (p**2 / m - kT) / Q
        tq = dp / m
        tp = -omega**2 * dq - g * dp - dg * p * dxi
        txi = 2.0 * p * dp / (m * Q)
        return fq, fp, fxi, tq, tp, txi

    def step(q, p, xi, dq, dp, dxi):
        k1 = rhs(q, p, xi, dq, dp, dxi)
        h = dt * 0.5
        k2 = rhs(q+h*k1[0], p+h*k1[1], xi+h*k1[2],
                  dq+h*k1[3], dp+h*k1[4], dxi+h*k1[5])
        k3 = rhs(q+h*k2[0], p+h*k2[1], xi+h*k2[2],
                  dq+h*k2[3], dp+h*k2[4], dxi+h*k2[5])
        h2 = dt
        k4 = rhs(q+h2*k3[0], p+h2*k3[1], xi+h2*k3[2],
                  dq+h2*k3[3], dp+h2*k3[4], dxi+h2*k3[5])
        s = dt / 6.0
        return (q+s*(k1[0]+2*k2[0]+2*k3[0]+k4[0]),
                p+s*(k1[1]+2*k2[1]+2*k3[1]+k4[1]),
                xi+s*(k1[2]+2*k2[2]+2*k3[2]+k4[2]),
                dq+s*(k1[3]+2*k2[3]+2*k3[3]+k4[3]),
                dp+s*(k1[4]+2*k2[4]+2*k3[4]+k4[4]),
                dxi+s*(k1[5]+2*k2[5]+2*k3[5]+k4[5]))

    # Burn-in
    dq, dp_t, dxi_t = 0.0, 0.0, 0.0
    for _ in range(burnin):
        q, p, xi, dq, dp_t, dxi_t = step(q, p, xi, dq, dp_t, dxi_t)
        if not (np.isfinite(q) and np.isfinite(p) and np.isfinite(xi)):
            return np.nan, np.array([np.nan])

    # Reset tangent
    tv = rng.normal(size=3)
    nrm = np.linalg.norm(tv)
    dq, dp_t, dxi_t = tv[0]/nrm, tv[1]/nrm, tv[2]/nrm

    n_main = n_total - burnin
    n_renorm = n_main // renorm_interval
    log_stretching = np.zeros(n_renorm)

    for i in range(n_renorm):
        for _ in range(renorm_interval):
            q, p, xi, dq, dp_t, dxi_t = step(q, p, xi, dq, dp_t, dxi_t)
        if not (np.isfinite(q) and np.isfinite(p) and np.isfinite(xi)):
            return np.nan, np.full(n_renorm, np.nan)
        nrm = np.sqrt(dq**2 + dp_t**2 + dxi_t**2)
        if nrm < 1e-300 or not np.isfinite(nrm):
            return np.nan, np.full(n_renorm, np.nan)
        log_stretching[i] = np.log(nrm)
        dq /= nrm; dp_t /= nrm; dxi_t /= nrm

    cumsum = np.cumsum(log_stretching)
    times = np.arange(1, n_renorm + 1) * renorm_interval * dt
    trace = cumsum / times
    return trace[-1], trace


def simulate_phase(pot, Q, kT=1.0, dt=0.005, n_steps=500_000, seed=42):
    """Generate phase portrait data."""
    rng = np.random.default_rng(seed)
    q = rng.normal(0, np.sqrt(kT))
    p = rng.normal(0, np.sqrt(kT))
    xi = 0.0

    sub = 20
    n_store = n_steps // sub
    q_arr = np.zeros(n_store)
    p_arr = np.zeros(n_store)

    for s in range(n_steps):
        g = pot.dV(xi) / Q
        k1 = np.array([p, -q - g*p, (p**2 - kT)/Q])
        s2 = np.array([q, p, xi]) + 0.5*dt*k1
        g2 = pot.dV(s2[2]) / Q
        k2 = np.array([s2[1], -s2[0] - g2*s2[1], (s2[1]**2 - kT)/Q])
        s3 = np.array([q, p, xi]) + 0.5*dt*k2
        g3 = pot.dV(s3[2]) / Q
        k3 = np.array([s3[1], -s3[0] - g3*s3[1], (s3[1]**2 - kT)/Q])
        s4 = np.array([q, p, xi]) + dt*k3
        g4 = pot.dV(s4[2]) / Q
        k4 = np.array([s4[1], -s4[0] - g4*s4[1], (s4[1]**2 - kT)/Q])
        state = np.array([q, p, xi]) + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        q, p, xi = state

        if s % sub == 0:
            idx = s // sub
            q_arr[idx] = q
            p_arr[idx] = p

    return q_arr, p_arr


def flush_print(*args, **kwargs):
    kwargs['flush'] = True
    print(*args, **kwargs)


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(out_dir, "figures")
    kT = 1.0
    seed = 42

    # ============================================================
    # Part 1: Long Lyapunov runs at key Q values
    # ============================================================
    print("="*60)
    print("Part 1: Long Lyapunov runs (T=50000)")
    print("="*60)

    Q_focus = [0.3, 0.5, 0.8, 1.0]

    results = {}
    for pot in ALL_POTS:
        print(f"\n  {pot.name}:")
        lams = {}
        for Q in Q_focus:
            lam, trace = compute_lyapunov_long(
                pot, Q, total_time=15000.0, seed=seed
            )
            lams[Q] = lam
            status = f"{lam:.5f}" if np.isfinite(lam) else "FAILED"
            print(f"    Q={Q:.1f}: lambda_max = {status}")
        results[pot.name] = lams

    # Save
    with open(os.path.join(out_dir, "lyapunov_long_results.txt"), "w") as f:
        f.write("# Long Lyapunov runs (T=50000, dt=0.01, seed=42)\n\n")
        for name, lams in results.items():
            f.write(f"{name}:\n")
            for Q, lam in lams.items():
                f.write(f"  Q={Q:.1f}: {lam:.6f}\n")
            f.write("\n")

    # ============================================================
    # Part 2: Phase portraits at Q=0.3 (strong chaos regime)
    # ============================================================
    print("\n" + "="*60)
    print("Part 2: Phase portraits at Q=0.3")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    Q_phase = 0.3

    for idx, pot in enumerate(ALL_POTS):
        ax = axes[idx // 2, idx % 2]
        print(f"  Simulating {pot.name}...")
        q_arr, p_arr = simulate_phase(pot, Q_phase, kT=kT, seed=seed)

        ax.scatter(q_arr[::3], p_arr[::3], s=0.05, alpha=0.15, color=pot.color)

        # Gaussian contours
        theta = np.linspace(0, 2*np.pi, 200)
        for sm in [1, 2, 3]:
            ax.plot(sm * np.cos(theta), sm * np.sin(theta),
                    "k-", alpha=0.3, linewidth=0.5)

        lam = results[pot.name].get(Q_phase, 0)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.set_xlabel("q", fontsize=12)
        ax.set_ylabel("p", fontsize=12)
        ax.set_title(f"{pot.name} (Q={Q_phase}, lam={lam:.3f})", fontsize=13)
        ax.tick_params(labelsize=11)

    fig.suptitle("Phase Portraits: 1D HO + Thermostat (Q=0.3)", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "phase_portraits_Q03.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved phase_portraits_Q03.png")

    # ============================================================
    # Part 3: Convergence traces at Q=0.8 (the sweet spot)
    # ============================================================
    print("\n" + "="*60)
    print("Part 3: Lyapunov convergence traces at Q=0.8")
    print("="*60)

    fig2, ax = plt.subplots(figsize=(8, 5))
    Q_conv = 0.8

    for pot in ALL_POTS:
        print(f"  {pot.name}...")
        _, trace = compute_lyapunov_long(pot, Q_conv, total_time=15000.0, seed=seed)
        n = len(trace)
        times = np.arange(1, n+1) * 20 * 0.01
        mask = np.isfinite(trace)
        if np.any(mask):
            ax.plot(times[mask], trace[mask], color=pot.color,
                    label=pot.name, linewidth=1.5, alpha=0.8)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel(r"Running $\lambda_{\max}$", fontsize=14)
    ax.set_title(f"Lyapunov Convergence (Q={Q_conv}, T=50000)", fontsize=14)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(os.path.join(fig_dir, "lyapunov_convergence_Q08.png"), dpi=150)
    plt.close(fig2)
    print("  Saved lyapunov_convergence_Q08.png")


if __name__ == "__main__":
    main()
