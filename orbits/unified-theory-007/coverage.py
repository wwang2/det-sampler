"""Phase space coverage analysis for generalized friction thermostats.

Measures what fraction of the expected (q,p) phase space is visited
by each thermostat type, as a function of Q. This directly connects
to the ergodicity score metric (20x20 grid coverage component).

Also computes KS statistics for marginal distributions.
"""

import numpy as np
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def g_nh(xi):
    return xi

def g_log_osc(xi):
    return 2.0 * xi / (1.0 + xi**2)

def g_tanh(xi):
    return np.tanh(xi)

def g_arctan(xi):
    return np.arctan(xi)


FRICTION_FUNCS = [
    ("NH", g_nh, "#1f77b4"),
    ("Log-Osc", g_log_osc, "#2ca02c"),
    ("Tanh", g_tanh, "#d62728"),
    ("Arctan", g_arctan, "#9467bd"),
]


def simulate_ho(g_func, Q, kT=1.0, omega=1.0, m=1.0, dt=0.005,
                n_steps=1_000_000, seed=42):
    """Simulate 1D HO with given friction function. Return (q, p) arrays."""
    rng = np.random.default_rng(seed)
    q = rng.normal(0, np.sqrt(kT / omega**2))
    p = rng.normal(0, np.sqrt(m * kT))
    xi = 0.0

    burnin = n_steps // 10
    n_record = n_steps - burnin
    sub = 5
    n_store = n_record // sub
    q_arr = np.zeros(n_store)
    p_arr = np.zeros(n_store)

    for step in range(n_steps):
        # RK4
        def rhs(s):
            qq, pp, xx = s
            g = g_func(xx) / Q  # friction = V'(xi)/Q, and we define g_func as V'
            # Wait -- need to be careful. For NH, V(xi) = Q*xi^2/2, V' = Q*xi,
            # so g = V'/Q = xi. For log-osc, V = Q*log(1+xi^2), V' = 2Q*xi/(1+xi^2),
            # g = 2*xi/(1+xi^2). So g_func should return V'(xi), and we divide by Q here.
            # But g_nh returns xi = V'/(Q) already... Let me fix this.
            # Actually, in the thermostat equations, friction = g(xi) where g = V'/Q.
            # g_nh(xi) = xi, which equals V'(xi)/Q = Q*xi/Q = xi. Correct.
            # g_log_osc(xi) = 2xi/(1+xi^2), which equals V'(xi)/Q = 2Q*xi/((1+xi^2)*Q). Correct.
            # So g_func already returns V'/Q. Don't divide by Q again!
            g = g_func(xx)
            return np.array([pp/m, -omega**2*qq - g*pp, (pp**2/m - kT)/Q])

        s = np.array([q, p, xi])
        k1 = rhs(s)
        k2 = rhs(s + 0.5*dt*k1)
        k3 = rhs(s + 0.5*dt*k2)
        k4 = rhs(s + dt*k3)
        s = s + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        q, p, xi = s

        if step >= burnin and (step - burnin) % sub == 0:
            idx = (step - burnin) // sub
            if idx < n_store:
                q_arr[idx] = q
                p_arr[idx] = p

    return q_arr, p_arr


def compute_coverage(q_arr, p_arr, kT=1.0, n_grid=20):
    """Compute phase space coverage on n_grid x n_grid grid."""
    # Grid spans +/- 3 sigma
    sigma_q = np.sqrt(kT)
    sigma_p = np.sqrt(kT)
    q_edges = np.linspace(-3*sigma_q, 3*sigma_q, n_grid + 1)
    p_edges = np.linspace(-3*sigma_p, 3*sigma_p, n_grid + 1)

    H, _, _ = np.histogram2d(q_arr, p_arr, bins=[q_edges, p_edges])
    visited = np.sum(H > 0)
    total = n_grid * n_grid
    return visited / total


def compute_ks(q_arr, p_arr, kT=1.0):
    """Compute KS statistics for q and p marginals."""
    sigma_q = np.sqrt(kT)
    sigma_p = np.sqrt(kT)

    ks_q, _ = stats.kstest(q_arr, 'norm', args=(0, sigma_q))
    ks_p, _ = stats.kstest(p_arr, 'norm', args=(0, sigma_p))

    return ks_q, ks_p


def compute_ergodicity_score(q_arr, p_arr, kT=1.0):
    """Compute the ergodicity score as defined in eval config."""
    ks_q, ks_p = compute_ks(q_arr, p_arr, kT)
    coverage = compute_coverage(q_arr, p_arr, kT)

    var_q = np.var(q_arr)
    var_p = np.var(p_arr)
    var_err_q = abs(var_q - kT) / kT
    var_err_p = abs(var_p - kT) / kT

    ks_comp = 1.0 - max(ks_q, ks_p)
    var_comp = 1.0 - max(var_err_q, var_err_p)
    var_comp = max(var_comp, 0.0)

    # Geometric mean
    score = (ks_comp * var_comp * coverage) ** (1.0/3.0)
    return score, ks_comp, var_comp, coverage


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(out_dir, "figures")
    kT = 1.0
    seed = 42

    Q_values = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]

    print("="*70, flush=True)
    print("Phase Space Coverage & Ergodicity Score vs Q", flush=True)
    print("="*70, flush=True)

    results = {}
    for name, g_func, color in FRICTION_FUNCS:
        print(f"\n{name}:", flush=True)
        scores = []
        coverages = []
        ks_comps = []
        var_comps = []
        for Q in Q_values:
            q_arr, p_arr = simulate_ho(g_func, Q, kT=kT, n_steps=1_000_000, seed=seed)
            score, ks_c, var_c, cov = compute_ergodicity_score(q_arr, p_arr, kT)
            scores.append(score)
            coverages.append(cov)
            ks_comps.append(ks_c)
            var_comps.append(var_c)
            print(f"  Q={Q:4.1f}: score={score:.3f} (KS={ks_c:.3f}, var={var_c:.3f}, cov={cov:.3f})", flush=True)
        results[name] = {
            "scores": np.array(scores),
            "coverages": np.array(coverages),
            "ks_comps": np.array(ks_comps),
            "var_comps": np.array(var_comps),
            "color": color,
        }

    # ============================================================
    # Plot: Ergodicity score and components vs Q
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Score
    for name, data in results.items():
        axes[0].plot(Q_values, data["scores"], "o-", color=data["color"],
                     label=name, linewidth=2, markersize=5)
    axes[0].axhline(0.85, color="gray", linestyle="--", alpha=0.7, label="Ergodic threshold")
    axes[0].set_xlabel("Thermostat mass Q", fontsize=14)
    axes[0].set_ylabel("Ergodicity Score", fontsize=14)
    axes[0].set_title("Ergodicity Score vs Q", fontsize=14)
    axes[0].set_xscale("log")
    axes[0].legend(fontsize=10)
    axes[0].tick_params(labelsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    # Coverage
    for name, data in results.items():
        axes[1].plot(Q_values, data["coverages"], "o-", color=data["color"],
                     label=name, linewidth=2, markersize=5)
    axes[1].set_xlabel("Thermostat mass Q", fontsize=14)
    axes[1].set_ylabel("Phase Space Coverage", fontsize=14)
    axes[1].set_title("Coverage (20x20 grid)", fontsize=14)
    axes[1].set_xscale("log")
    axes[1].legend(fontsize=10)
    axes[1].tick_params(labelsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    # KS component
    for name, data in results.items():
        axes[2].plot(Q_values, data["ks_comps"], "o-", color=data["color"],
                     label=name, linewidth=2, markersize=5)
    axes[2].set_xlabel("Thermostat mass Q", fontsize=14)
    axes[2].set_ylabel("1 - max(KS_q, KS_p)", fontsize=14)
    axes[2].set_title("Distribution Match (KS)", fontsize=14)
    axes[2].set_xscale("log")
    axes[2].legend(fontsize=10)
    axes[2].tick_params(labelsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1)

    fig_path = os.path.join(fig_dir, "ergodicity_vs_Q.png")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {fig_path}", flush=True)

    # Save numerical results
    with open(os.path.join(out_dir, "coverage_results.txt"), "w") as f:
        f.write("# Ergodicity score components vs Q (1D HO, 1M steps, seed=42)\n\n")
        for name, data in results.items():
            f.write(f"{name}:\n")
            for i, Q in enumerate(Q_values):
                f.write(f"  Q={Q:.1f}: score={data['scores'][i]:.4f} "
                        f"cov={data['coverages'][i]:.4f} "
                        f"ks={data['ks_comps'][i]:.4f} "
                        f"var={data['var_comps'][i]:.4f}\n")
            f.write("\n")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
