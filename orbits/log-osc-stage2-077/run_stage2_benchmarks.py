"""Part A: Stage 2 benchmarks for log-osc NH friction.

Tests log-osc NH on:
  1. 2D Gaussian mixture (5 modes, radius=3, sigma=0.5)
  2. Rosenbrock banana (a=0, b=5)

Compares:
  - NH tanh(xi), best Q (baseline)
  - NH log-osc g(xi)=2xi/(1+xi^2), Q in {0.05, 0.1, 0.3, 1.0}
  - NHC(M=3) tanh, Q=1 (campaign baseline)

Metrics: KL divergence, ESS/force-eval, tau_int
Seeds: 3 per condition.
"""

import numpy as np
import json
import time
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from research.eval.potentials import GaussianMixture2D, Rosenbrock2D
from research.eval.integrators import ThermostatState
from research.eval.evaluator import kl_divergence_histogram, autocorrelation_time

ORBIT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ORBIT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Dynamics classes
# ---------------------------------------------------------------------------

class NHTanh:
    """NH with g(xi) = tanh(xi). Standard bounded friction."""

    def __init__(self, dim, kT=1.0, Q=1.0):
        self.dim = dim
        self.kT = kT
        self.Q = Q
        self.name = f"NH_tanh_Q{Q}"

    def initial_state(self, q0, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.kT), size=self.dim)
        xi0 = np.array([0.0])
        return ThermostatState(q0.copy(), p0, xi0, 0)

    def dqdt(self, state, grad_U):
        return state.p

    def dpdt(self, state, grad_U):
        g = np.tanh(state.xi[0])
        return -grad_U - g * state.p

    def dxidt(self, state, grad_U):
        KE = np.sum(state.p ** 2)
        return np.array([(KE - self.dim * self.kT) / self.Q])


class NHLogOsc:
    """NH with log-osc g(xi) = 2*xi / (1 + xi^2). Self-limiting friction."""

    def __init__(self, dim, kT=1.0, Q=1.0):
        self.dim = dim
        self.kT = kT
        self.Q = Q
        self.name = f"NH_losc_Q{Q}"

    def initial_state(self, q0, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.kT), size=self.dim)
        xi0 = np.array([0.0])
        return ThermostatState(q0.copy(), p0, xi0, 0)

    def dqdt(self, state, grad_U):
        return state.p

    def dpdt(self, state, grad_U):
        xi = state.xi[0]
        g = 2.0 * xi / (1.0 + xi * xi)
        return -grad_U - g * state.p

    def dxidt(self, state, grad_U):
        KE = np.sum(state.p ** 2)
        return np.array([(KE - self.dim * self.kT) / self.Q])


class NHC3:
    """NHC(M=3) with tanh friction on first variable. Campaign baseline."""

    def __init__(self, dim, kT=1.0, Q=1.0):
        self.dim = dim
        self.kT = kT
        self.Q = Q
        self.M = 3
        self.Qs = [Q] * 3
        self.name = f"NHC3_tanh_Q{Q}"

    def initial_state(self, q0, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.kT), size=self.dim)
        xi0 = np.zeros(self.M)
        return ThermostatState(q0.copy(), p0, xi0, 0)

    def dqdt(self, state, grad_U):
        return state.p

    def dpdt(self, state, grad_U):
        g = np.tanh(state.xi[0])
        return -grad_U - g * state.p

    def dxidt(self, state, grad_U):
        xi = state.xi
        M = self.M
        dxi = np.zeros(M)

        KE = np.sum(state.p ** 2)
        G1 = KE - self.dim * self.kT
        dxi[0] = G1 / self.Qs[0]
        if M > 1:
            dxi[0] -= xi[1] * xi[0]

        for j in range(1, M - 1):
            Gj = self.Qs[j-1] * xi[j-1]**2 - self.kT
            dxi[j] = Gj / self.Qs[j] - xi[j+1] * xi[j]

        if M > 1:
            GM = self.Qs[M-2] * xi[M-2]**2 - self.kT
            dxi[M-1] = GM / self.Qs[M-1]

        return dxi


# ---------------------------------------------------------------------------
# Custom Verlet integrator that correctly uses g(xi) from dynamics
# ---------------------------------------------------------------------------

def _compute_g(dynamics, xi_val):
    """Extract the friction function g(xi) from the dynamics object."""
    if isinstance(dynamics, NHLogOsc):
        return 2.0 * xi_val / (1.0 + xi_val * xi_val)
    elif isinstance(dynamics, NHTanh):
        return np.tanh(xi_val)
    elif isinstance(dynamics, NHC3):
        return np.tanh(xi_val)
    else:
        return xi_val  # fallback: linear NH


def run_nh_trajectory(dynamics, potential, dt, n_force_evals, kT, seed,
                      burnin_frac=0.1, thin=10):
    """Run NH-type dynamics with custom Verlet that uses g(xi) properly.

    Splitting scheme (1 force eval per step):
      1. Half-step xi
      2. Half-step p: exp(-g(xi)*dt/2) rescaling + kick
      3. Full-step q
      4. Force eval
      5. Half-step p: kick + exp(-g(xi)*dt/2) rescaling
      6. Half-step xi
    """
    rng = np.random.default_rng(seed)
    q0 = rng.normal(0, 0.5, size=potential.dim)
    state = dynamics.initial_state(q0, rng=rng)

    burnin_evals = int(n_force_evals * burnin_frac)
    n_evals = 0

    q, p, xi = state.q.copy(), state.p.copy(), state.xi.copy()
    dim = dynamics.dim
    half_dt = 0.5 * dt
    Q_val = dynamics.Q
    is_nhc = isinstance(dynamics, NHC3)
    is_losc = isinstance(dynamics, NHLogOsc)

    # Pre-allocate output arrays (estimate max samples)
    max_samples = (n_force_evals - burnin_evals) // thin + 100
    q_out = np.zeros((max_samples, potential.dim))
    p_out = np.zeros((max_samples, potential.dim))
    out_idx = 0

    # Initial force eval
    grad_U = potential.gradient(q)
    n_evals += 1
    step_count = 0

    while n_evals < n_force_evals:
        # Half-step thermostat variables
        if is_nhc:
            dxi = dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
            xi = xi + half_dt * dxi
        else:
            KE = np.dot(p, p)
            xi[0] += half_dt * (KE - dim * kT) / Q_val

        # Compute g(xi)
        xv = xi[0]
        if is_losc:
            g_val = 2.0 * xv / (1.0 + xv * xv)
        else:
            g_val = np.tanh(xv)

        # Half-step momenta: friction rescaling + kick
        s = np.exp(-g_val * half_dt)
        if s > 1e10: s = 1e10
        elif s < 1e-10: s = 1e-10
        p *= s
        p -= half_dt * grad_U

        # Full-step positions
        q += dt * p

        # Force eval at new position
        grad_U = potential.gradient(q)
        n_evals += 1

        # Half-step momenta: kick + friction rescaling
        p -= half_dt * grad_U
        xv = xi[0]
        if is_losc:
            g_val = 2.0 * xv / (1.0 + xv * xv)
        else:
            g_val = np.tanh(xv)
        s = np.exp(-g_val * half_dt)
        if s > 1e10: s = 1e10
        elif s < 1e-10: s = 1e-10
        p *= s

        # Half-step thermostat variables
        if is_nhc:
            dxi = dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
            xi = xi + half_dt * dxi
        else:
            KE = np.dot(p, p)
            xi[0] += half_dt * (KE - dim * kT) / Q_val

        step_count += 1

        # NaN check (infrequent)
        if step_count % 10000 == 0:
            if np.any(np.isnan(q)) or np.any(np.isnan(p)):
                return None, None, n_evals

        if n_evals >= burnin_evals and step_count % thin == 0:
            if out_idx < max_samples:
                q_out[out_idx] = q
                p_out[out_idx] = p
                out_idx += 1

    if out_idx == 0:
        return None, None, n_evals
    return q_out[:out_idx].copy(), p_out[:out_idx].copy(), n_evals


def evaluate_condition(dynamics, potential, dt, n_force_evals, kT, seeds):
    """Run multiple seeds and aggregate metrics."""
    kls = []
    taus = []
    ess_per_fe = []

    for seed in seeds:
        q_samples, p_samples, actual_fe = run_nh_trajectory(
            dynamics, potential, dt, n_force_evals, kT, seed
        )

        if q_samples is None or len(q_samples) < 200:
            kls.append(float('inf'))
            taus.append(float('inf'))
            ess_per_fe.append(0.0)
            continue

        kl = kl_divergence_histogram(q_samples, potential, kT, n_bins=80)
        kl = max(0.0, kl)
        tau = autocorrelation_time(q_samples)
        n = len(q_samples)
        ess = n / tau
        ess_fe = ess / max(actual_fe, 1)

        kls.append(kl)
        taus.append(tau)
        ess_per_fe.append(ess_fe)

    return {
        "kl_mean": float(np.mean(kls)),
        "kl_std": float(np.std(kls)),
        "kl_min": float(np.min(kls)),
        "tau_mean": float(np.mean(taus)),
        "tau_std": float(np.std(taus)),
        "ess_per_fe_mean": float(np.mean(ess_per_fe)),
        "n_seeds": len(seeds),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("PART A: Stage 2 Benchmarks — log-osc NH vs tanh NH vs NHC(M=3)")
    print("=" * 80)

    kT = 1.0
    dt = 0.005
    n_force_evals = 500_000
    seeds = [42, 123, 777]
    Q_sweep = [0.05, 0.1, 0.3, 1.0]

    potentials = [
        ("gaussian_mixture_2d", GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5)),
        ("rosenbrock_2d", Rosenbrock2D(a=0.0, b=5.0)),
    ]

    all_results = {}

    for pot_name, potential in potentials:
        print(f"\n{'='*60}")
        print(f"Potential: {pot_name}")
        print(f"{'='*60}")

        results_for_pot = {}

        # --- NHC(M=3) baseline ---
        print(f"\n  NHC(M=3), Q=1.0 (campaign baseline)...")
        sys.stdout.flush()
        nhc = NHC3(dim=potential.dim, kT=kT, Q=1.0)
        t0 = time.time()
        r = evaluate_condition(nhc, potential, dt, n_force_evals, kT, seeds)
        elapsed = time.time() - t0
        r["wall_sec"] = elapsed
        results_for_pot["NHC3_Q1.0"] = r
        print(f"    KL = {r['kl_mean']:.4f} +/- {r['kl_std']:.4f}  "
              f"tau = {r['tau_mean']:.1f}  ESS/fe = {r['ess_per_fe_mean']:.6f}  "
              f"({elapsed:.1f}s)")
        sys.stdout.flush()

        # --- NH tanh, sweep Q ---
        for Q in Q_sweep:
            print(f"\n  NH tanh, Q={Q}...")
            sys.stdout.flush()
            nh = NHTanh(dim=potential.dim, kT=kT, Q=Q)
            t0 = time.time()
            r = evaluate_condition(nh, potential, dt, n_force_evals, kT, seeds)
            elapsed = time.time() - t0
            r["wall_sec"] = elapsed
            results_for_pot[f"NH_tanh_Q{Q}"] = r
            print(f"    KL = {r['kl_mean']:.4f} +/- {r['kl_std']:.4f}  "
                  f"tau = {r['tau_mean']:.1f}  ESS/fe = {r['ess_per_fe_mean']:.6f}  "
                  f"({elapsed:.1f}s)")
            sys.stdout.flush()

        # --- NH log-osc, sweep Q ---
        for Q in Q_sweep:
            print(f"\n  NH log-osc, Q={Q}...")
            sys.stdout.flush()
            nh = NHLogOsc(dim=potential.dim, kT=kT, Q=Q)
            t0 = time.time()
            r = evaluate_condition(nh, potential, dt, n_force_evals, kT, seeds)
            elapsed = time.time() - t0
            r["wall_sec"] = elapsed
            results_for_pot[f"NH_losc_Q{Q}"] = r
            print(f"    KL = {r['kl_mean']:.4f} +/- {r['kl_std']:.4f}  "
                  f"tau = {r['tau_mean']:.1f}  ESS/fe = {r['ess_per_fe_mean']:.6f}  "
                  f"({elapsed:.1f}s)")
            sys.stdout.flush()

        all_results[pot_name] = results_for_pot

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print("\n\n" + "=" * 100)
    print("SUMMARY TABLES")
    print("=" * 100)

    for pot_name in all_results:
        res = all_results[pot_name]
        print(f"\n--- {pot_name} ---")
        print(f"{'Method':<25} {'KL (mean +/- std)':<25} {'tau_int':<12} {'ESS/fe':<12}")
        print("-" * 74)

        for key in sorted(res.keys()):
            r = res[key]
            print(f"{key:<25} {r['kl_mean']:.4f} +/- {r['kl_std']:.4f}       "
                  f"{r['tau_mean']:<12.1f} {r['ess_per_fe_mean']:<12.6f}")

    # ---------------------------------------------------------------------------
    # Compute metric
    # ---------------------------------------------------------------------------
    print("\n\n" + "=" * 100)
    print("METRIC COMPUTATION")
    print("=" * 100)

    for pot_name in all_results:
        res = all_results[pot_name]
        nhc_kl = res["NHC3_Q1.0"]["kl_mean"]

        # Find best log-osc KL
        best_losc_kl = float('inf')
        best_losc_key = None
        for key in res:
            if "losc" in key:
                if res[key]["kl_mean"] < best_losc_kl:
                    best_losc_kl = res[key]["kl_mean"]
                    best_losc_key = key

        # Find best tanh KL
        best_tanh_kl = float('inf')
        best_tanh_key = None
        for key in res:
            if "tanh" in key and "NHC" not in key:
                if res[key]["kl_mean"] < best_tanh_kl:
                    best_tanh_kl = res[key]["kl_mean"]
                    best_tanh_key = key

        print(f"\n  {pot_name}:")
        print(f"    NHC(M=3) baseline KL: {nhc_kl:.4f}")
        print(f"    Best log-osc: {best_losc_key} -> KL = {best_losc_kl:.4f}")
        print(f"    Best tanh:    {best_tanh_key} -> KL = {best_tanh_kl:.4f}")
        if nhc_kl > 0:
            print(f"    metric (KL_losc_best / KL_nhc): {best_losc_kl / nhc_kl:.3f}")
            print(f"    metric (KL_tanh_best / KL_nhc): {best_tanh_kl / nhc_kl:.3f}")

    # ---------------------------------------------------------------------------
    # Figures
    # ---------------------------------------------------------------------------
    generate_figures(all_results, Q_sweep)

    # Save results
    out_path = os.path.join(ORBIT_DIR, "results_stage2.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return all_results


def generate_figures(all_results, Q_sweep):
    """Generate publication-quality comparison figures."""

    for pot_name, res in all_results.items():
        # Figure: KL vs Q for each method
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Panel A: KL divergence
        ax = axes[0]
        tanh_kls = [res[f"NH_tanh_Q{Q}"]["kl_mean"] for Q in Q_sweep]
        tanh_errs = [res[f"NH_tanh_Q{Q}"]["kl_std"] for Q in Q_sweep]
        losc_kls = [res[f"NH_losc_Q{Q}"]["kl_mean"] for Q in Q_sweep]
        losc_errs = [res[f"NH_losc_Q{Q}"]["kl_std"] for Q in Q_sweep]
        nhc_kl = res["NHC3_Q1.0"]["kl_mean"]

        ax.errorbar(Q_sweep, tanh_kls, yerr=tanh_errs, marker='s', ms=7, lw=1.5,
                     capsize=3, label='NH tanh($\\xi$)', color='#1f77b4')
        ax.errorbar(Q_sweep, losc_kls, yerr=losc_errs, marker='o', ms=7, lw=1.5,
                     capsize=3, label='NH log-osc', color='#d62728')
        ax.axhline(nhc_kl, color='#2ca02c', ls='--', lw=2, label=f'NHC(M=3) KL={nhc_kl:.4f}')

        ax.set_xlabel('Q (thermostat mass)', fontsize=12)
        ax.set_ylabel('KL divergence', fontsize=12)
        ax.set_title(f'{pot_name}: KL divergence', fontsize=13)
        ax.set_xscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Panel B: tau_int
        ax = axes[1]
        tanh_taus = [res[f"NH_tanh_Q{Q}"]["tau_mean"] for Q in Q_sweep]
        losc_taus = [res[f"NH_losc_Q{Q}"]["tau_mean"] for Q in Q_sweep]
        nhc_tau = res["NHC3_Q1.0"]["tau_mean"]

        ax.plot(Q_sweep, tanh_taus, 's-', ms=7, lw=1.5, label='NH tanh($\\xi$)', color='#1f77b4')
        ax.plot(Q_sweep, losc_taus, 'o-', ms=7, lw=1.5, label='NH log-osc', color='#d62728')
        ax.axhline(nhc_tau, color='#2ca02c', ls='--', lw=2, label=f'NHC(M=3) tau={nhc_tau:.1f}')

        ax.set_xlabel('Q (thermostat mass)', fontsize=12)
        ax.set_ylabel('$\\tau_{\\mathrm{int}}$', fontsize=12)
        ax.set_title(f'{pot_name}: autocorrelation time', fontsize=13)
        ax.set_xscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"stage2_{pot_name}.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: figures/stage2_{pot_name}.png")

    # Combined bar chart: best KL per method per potential
    fig, ax = plt.subplots(figsize=(10, 5))
    pots = list(all_results.keys())
    x = np.arange(len(pots))
    width = 0.25

    for i, pot_name in enumerate(pots):
        res = all_results[pot_name]
        nhc_kl = res["NHC3_Q1.0"]["kl_mean"]
        best_tanh = min(res[f"NH_tanh_Q{Q}"]["kl_mean"] for Q in Q_sweep)
        best_losc = min(res[f"NH_losc_Q{Q}"]["kl_mean"] for Q in Q_sweep)

        if i == 0:
            ax.bar(x[i] - width, nhc_kl, width, color='#2ca02c', label='NHC(M=3)')
            ax.bar(x[i], best_tanh, width, color='#1f77b4', label='NH tanh (best Q)')
            ax.bar(x[i] + width, best_losc, width, color='#d62728', label='NH log-osc (best Q)')
        else:
            ax.bar(x[i] - width, nhc_kl, width, color='#2ca02c')
            ax.bar(x[i], best_tanh, width, color='#1f77b4')
            ax.bar(x[i] + width, best_losc, width, color='#d62728')

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('_', '\n') for p in pots], fontsize=10)
    ax.set_ylabel('KL divergence (lower is better)', fontsize=12)
    ax.set_title('Stage 2 Benchmarks: Best KL by Method', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "stage2_summary_bar.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: figures/stage2_summary_bar.png")


if __name__ == "__main__":
    main()
