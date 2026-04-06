"""Phase diagram for N=2 parallel log-osc thermostats on 1D harmonic oscillator.

Tasks:
  1. Verify N=1 failure (baseline)
  2. Scan (Q1, Q2/Q1) for kappa=1.0, 0.5, 4.0
  3. Find critical curve
  4. Generate heatmap figure
"""

import json
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Add repo root to path
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

ORBIT_DIR = Path(__file__).resolve().parent

# ──────────────────────────────────────────────
# Physics helpers
# ──────────────────────────────────────────────

def g_func(xi):
    """Log-osc friction: g(xi) = 2*xi / (1 + xi^2)."""
    return 2.0 * xi / (1.0 + xi**2)


# ──────────────────────────────────────────────
# N=1 integrator
# ──────────────────────────────────────────────

def run_N1(kappa=1.0, kT=1.0, Q=1.0, dt=0.02, n_evals=1_000_000, seed=42):
    """Single log-osc thermostat on 1D HO. Returns var(q)/kT."""
    rng = np.random.default_rng(seed)
    q = np.array([rng.normal(0, np.sqrt(kT / kappa))])
    p = np.array([rng.normal(0, np.sqrt(kT))])
    xi = np.array([0.0])

    half_dt = 0.5 * dt
    q_samples = []
    grad_U = np.array([kappa * q[0]])
    n_force = 1

    while n_force < n_evals:
        # Half-step xi
        K = p[0]**2
        xi[0] += half_dt * (K - kT) / Q

        # Half-step p: friction + kick
        g = g_func(xi[0])
        p *= np.exp(-g * half_dt)
        p -= half_dt * grad_U

        # Full-step q
        q += dt * p

        # New force
        grad_U = np.array([kappa * q[0]])
        n_force += 1

        # Half-step p: kick + friction
        p -= half_dt * grad_U
        g = g_func(xi[0])
        p *= np.exp(-g * half_dt)

        # Half-step xi
        K = p[0]**2
        xi[0] += half_dt * (K - kT) / Q

        q_samples.append(float(q[0]))

    q_arr = np.array(q_samples)
    # Use latter half for statistics
    half = len(q_arr) // 2
    var_q = np.var(q_arr[half:])
    return var_q / (kT / kappa)


# ──────────────────────────────────────────────
# N=2 parallel integrator
# ──────────────────────────────────────────────

def run_N2(kappa=1.0, kT=1.0, Q1=1.0, Q2=2.0, dt=0.02, n_evals=300_000, seed=42):
    """Two PARALLEL log-osc thermostats on 1D HO.

    dp/dt = -kappa*q - [g(xi1) + g(xi2)] * p
    dxi1/dt = (p^2 - kT) / Q1
    dxi2/dt = (p^2 - kT) / Q2

    Returns var(q) / (kT/kappa)  [should be ~1.0 if ergodic]
    """
    rng = np.random.default_rng(seed)
    q = float(rng.normal(0, np.sqrt(kT / kappa)))
    p = float(rng.normal(0, np.sqrt(kT)))
    xi1 = 0.0
    xi2 = 0.0

    half_dt = 0.5 * dt
    q_samples = []
    grad_U = kappa * q
    n_force = 1

    while n_force < n_evals:
        # Half-step thermostats
        K = p * p
        xi1 += half_dt * (K - kT) / Q1
        xi2 += half_dt * (K - kT) / Q2

        # Half-step p: combined friction + kick
        g_total = g_func(xi1) + g_func(xi2)
        p *= np.exp(-g_total * half_dt)
        p -= half_dt * grad_U

        # Full-step q
        q += dt * p

        # New force
        grad_U = kappa * q
        n_force += 1

        # Half-step p: kick + friction
        p -= half_dt * grad_U
        g_total = g_func(xi1) + g_func(xi2)
        p *= np.exp(-g_total * half_dt)

        # Half-step thermostats
        K = p * p
        xi1 += half_dt * (K - kT) / Q1
        xi2 += half_dt * (K - kT) / Q2

        q_samples.append(q)

    q_arr = np.array(q_samples)
    # Use latter half for statistics
    half = len(q_arr) // 2
    var_q = float(np.var(q_arr[half:]))
    return var_q / (kT / kappa)


# ──────────────────────────────────────────────
# Task 1: Baseline N=1 failure
# ──────────────────────────────────────────────

def task1_baseline():
    print("\n=== Task 1: N=1 Baseline ===")
    kT = 1.0
    kappa = 1.0
    results = {}
    for Q in [0.2, 0.5, 0.8, 1.0, 2.0, 5.0]:
        vr = run_N1(kappa=kappa, kT=kT, Q=Q, dt=0.02, n_evals=1_000_000)
        deviation = abs(vr - 1.0)
        status = "ERGODIC" if deviation < 0.05 else "NON-ERGODIC"
        print(f"  Q={Q:.1f}: var_ratio={vr:.4f}  deviation={deviation:.4f}  [{status}]")
        results[str(Q)] = {"var_ratio": vr, "deviation": deviation}
    return results


# ──────────────────────────────────────────────
# Task 2: Phase diagram scan
# ──────────────────────────────────────────────

def task2_phase_diagram(kappa=1.0, kT=1.0, dt=0.02, n_evals=300_000, seed=42):
    print(f"\n=== Task 2: Phase diagram kappa={kappa} ===")
    Q1_vals = np.logspace(np.log10(0.05), np.log10(5.0), 12)
    ratio_vals = np.logspace(np.log10(1.1), np.log10(200), 20)

    results = []
    total = len(Q1_vals) * len(ratio_vals)
    t0 = time.time()

    for i, Q1 in enumerate(Q1_vals):
        for j, ratio in enumerate(ratio_vals):
            Q2 = Q1 * ratio
            vr = run_N2(kappa=kappa, kT=kT, Q1=Q1, Q2=Q2,
                        dt=dt, n_evals=n_evals, seed=seed)
            ergodic = abs(vr - 1.0) < 0.05
            results.append({
                "Q1": float(Q1),
                "Q2": float(Q2),
                "ratio": float(ratio),
                "var_ratio": float(vr),
                "ergodic": bool(ergodic),
            })
        done = (i + 1) * len(ratio_vals)
        elapsed = time.time() - t0
        eta = elapsed / done * (total - done)
        print(f"  Q1={Q1:.3f} ({done}/{total}) elapsed={elapsed:.0f}s ETA={eta:.0f}s")

    return results


# ──────────────────────────────────────────────
# Task 3: Critical curve analysis
# ──────────────────────────────────────────────

def task3_critical_curve(all_results):
    print("\n=== Task 3: Critical Curve ===")
    analysis = {}

    for kappa_str, results in all_results.items():
        kappa = float(kappa_str)
        # Find boundary: for each Q1, find minimum ratio that gives ergodic
        Q1_unique = sorted(set(r["Q1"] for r in results))
        boundary = []
        for Q1 in Q1_unique:
            rows = sorted([r for r in results if abs(r["Q1"] - Q1) < 1e-9],
                          key=lambda x: x["ratio"])
            # Find smallest ratio where ergodic
            crit_ratio = None
            for r in rows:
                if r["ergodic"]:
                    crit_ratio = r["ratio"]
                    break
            boundary.append({"Q1": Q1, "crit_ratio": crit_ratio})

        # Filter valid boundary points
        valid = [(b["Q1"], b["crit_ratio"]) for b in boundary if b["crit_ratio"] is not None]
        if len(valid) >= 3:
            log_Q1 = np.log(np.array([v[0] for v in valid]))
            log_r = np.log(np.array([v[1] for v in valid]))
            # Fit: log(crit_ratio) = a * log(Q1) + b
            coeffs = np.polyfit(log_Q1, log_r, 1)
            slope, intercept = coeffs
            crit_at_Q1_1 = np.exp(intercept)  # ratio at Q1=1.0
            print(f"  kappa={kappa}: slope={slope:.3f}, crit_ratio(Q1=1)={crit_at_Q1_1:.2f}")
            print(f"  Fit: log(crit_ratio) = {slope:.3f}*log(Q1) + {intercept:.3f}")
            analysis[kappa_str] = {
                "boundary": [{"Q1": v[0], "crit_ratio": v[1]} for v in valid],
                "slope": float(slope),
                "intercept": float(intercept),
                "crit_ratio_at_Q1_1": float(crit_at_Q1_1),
            }
        else:
            print(f"  kappa={kappa}: insufficient boundary points")
            analysis[kappa_str] = {"boundary": [], "note": "insufficient data"}

    return analysis


# ──────────────────────────────────────────────
# Task 4: Heatmap figure
# ──────────────────────────────────────────────

def task4_figure(all_results, analysis):
    print("\n=== Task 4: Figure ===")
    kappas = sorted(all_results.keys(), key=float)
    n_kappa = len(kappas)

    fig, axes = plt.subplots(1, n_kappa, figsize=(6 * n_kappa, 5), squeeze=False)

    Q1_vals = np.logspace(np.log10(0.05), np.log10(5.0), 12)
    ratio_vals = np.logspace(np.log10(1.1), np.log10(200), 20)
    log_Q1 = np.log10(Q1_vals)
    log_r = np.log10(ratio_vals)

    for col, kappa_str in enumerate(kappas):
        kappa = float(kappa_str)
        results = all_results[kappa_str]
        ax = axes[0][col]

        # Build 2D grid (rows=Q1, cols=ratio)
        grid = np.zeros((len(Q1_vals), len(ratio_vals)))
        for r in results:
            i = np.argmin(np.abs(Q1_vals - r["Q1"]))
            j = np.argmin(np.abs(ratio_vals - r["ratio"]))
            grid[i, j] = r["var_ratio"]

        # Clip for display
        vmin, vmax = 0.5, 2.0
        grid_clipped = np.clip(grid, vmin, vmax)

        im = ax.pcolormesh(log_r, log_Q1, grid_clipped,
                           cmap="RdYlGn_r", vmin=vmin, vmax=vmax,
                           shading="auto")
        plt.colorbar(im, ax=ax, label="var(q)/(kT/κ)")

        # Overlay ergodic/non-ergodic boundary
        if kappa_str in analysis and "boundary" in analysis[kappa_str]:
            bdy = analysis[kappa_str]["boundary"]
            if bdy:
                bx = np.log10([b["crit_ratio"] for b in bdy])
                by = np.log10([b["Q1"] for b in bdy])
                ax.plot(bx, by, "b--o", lw=2, ms=5, label="boundary")
                ax.legend(fontsize=8)

        ax.set_xlabel("log₁₀(Q₂/Q₁)")
        ax.set_ylabel("log₁₀(Q₁)")
        ax.set_title(f"κ={kappa}  [N=2 parallel log-osc]")

        # Add contour at var_ratio=1.05 (ergodic boundary)
        try:
            cs = ax.contour(log_r, log_Q1, grid, levels=[1.05, 0.95],
                            colors=["blue", "blue"], linewidths=1.5, linestyles="--")
            ax.clabel(cs, fmt="%.2f", fontsize=7)
        except Exception:
            pass

    fig.suptitle("Ergodicity Phase Diagram: N=2 Parallel Log-Osc Thermostats\n"
                 "var_ratio ≈ 1.0 (green) = ergodic; deviation (red) = non-ergodic",
                 fontsize=12)
    plt.tight_layout()
    out = ORBIT_DIR / "figures" / "phase_diagram.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    out_dir = ORBIT_DIR

    # Task 1
    n1_results = task1_baseline()

    # Task 2 — three kappa values
    all_results = {}
    for kappa in [1.0, 0.5, 4.0]:
        res = task2_phase_diagram(kappa=kappa)
        all_results[str(kappa)] = res

    # Save raw data
    data = {
        "n1_baseline": n1_results,
        "phase_diagram": all_results,
    }
    out_json = out_dir / "phase_diagram_results.json"
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved results to {out_json}")

    # Task 3
    analysis = task3_critical_curve(all_results)
    analysis_path = out_dir / "critical_curve_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved analysis to {analysis_path}")

    # Task 4
    task4_figure(all_results, analysis)

    # Print summary
    print("\n=== SUMMARY ===")
    for kappa_str, info in analysis.items():
        if "crit_ratio_at_Q1_1" in info:
            crit = info["crit_ratio_at_Q1_1"]
            slope = info["slope"]
            print(f"  kappa={kappa_str}: crit_ratio(Q1=1)={crit:.2f}, slope={slope:.3f}")

    # Return crit ratio for kappa=1.0
    if "1.0" in analysis and "crit_ratio_at_Q1_1" in analysis["1.0"]:
        return analysis["1.0"]["crit_ratio_at_Q1_1"]
    return None


if __name__ == "__main__":
    crit = main()
    print(f"\nPrimary metric (crit_Q_ratio, kappa=1, Q1=1): {crit}")
