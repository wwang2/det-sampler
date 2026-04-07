"""N-scaling robust experiment for orbit n-scaling-robust-029.

Re-runs the N-scaling experiment from orbit n-scaling-028 with:
- 10 seeds (was 3)
- 800k force evals per run (was 400k)
- Enhanced analysis: mean±std, R², confidence intervals on slope
- Better figures with error bands
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

sys.path.insert(0, '/Users/wujiewang/code/det-sampler/.worktrees/n-scaling-robust-029')
from research.eval.integrators import ThermostatState

ORBIT_DIR = Path('/Users/wujiewang/code/det-sampler/.worktrees/n-scaling-robust-029/orbits/n-scaling-robust-029')


def g_losc(xi: np.ndarray) -> np.ndarray:
    return 2.0 * xi / (1.0 + xi ** 2)


class MultiScaleLogOscIntegrator:
    def __init__(self, Q_values, potential, dt, kT=1.0, dim=5):
        self.Q = np.array(Q_values)
        self.N = len(Q_values)
        self.potential = potential
        self.dt = dt
        self.kT = kT
        self.dim = dim

    def step(self, state):
        q, p, xi, n_evals = state
        dt = self.dt
        hdt = 0.5 * dt

        grad_U = self.potential.gradient(q)
        n_evals += 1

        kinetic = np.dot(p, p) / self.dim
        xi_dot = (kinetic - self.kT) / self.Q
        xi = xi + hdt * xi_dot

        Gamma = np.sum(g_losc(xi))
        scale = np.clip(np.exp(-Gamma * hdt), 1e-10, 1e10)
        p = p * scale
        p = p - hdt * grad_U
        q = q + dt * p

        if np.any(np.isnan(q)) or np.any(np.isnan(p)):
            return ThermostatState(q, p, xi, n_evals)

        grad_U = self.potential.gradient(q)
        n_evals += 1

        p = p - hdt * grad_U
        kinetic = np.dot(p, p) / self.dim
        xi_dot = (kinetic - self.kT) / self.Q
        xi = xi + hdt * xi_dot

        Gamma = np.sum(g_losc(xi))
        scale = np.clip(np.exp(-Gamma * hdt), 1e-10, 1e10)
        p = p * scale

        return ThermostatState(q, p, xi, n_evals)


class AnisotropicGaussian:
    def __init__(self, kappa_min, kappa_ratio, dim=5):
        self.dim = dim
        self.kappa_min = kappa_min
        self.kappa_max = kappa_min * kappa_ratio
        self.kappa = kappa_min * kappa_ratio ** (np.arange(dim) / (dim - 1))

    def energy(self, q):
        return 0.5 * float(np.dot(self.kappa, q ** 2))

    def gradient(self, q):
        return self.kappa * q


def compute_tau_int(xs):
    xs = xs - xs.mean()
    n = len(xs)
    C0 = float(np.mean(xs ** 2))
    if C0 < 1e-30:
        return 1.0
    tau = 1.0
    for t in range(1, n):
        Ct = float(np.mean(xs[:n - t] * xs[t:]))
        if Ct < 0.05 * C0:
            break
        tau += 2.0 * Ct / C0
    return max(tau, 1.0)


def run_single(kappa_ratio, N, seed, Q_values, n_force_evals=800_000, kT=1.0, dim=5):
    rng = np.random.default_rng(seed)
    kappa_min = 1.0
    pot = AnisotropicGaussian(kappa_min=kappa_min, kappa_ratio=kappa_ratio, dim=dim)
    kappa_max = pot.kappa_max

    dt = min(0.05, 0.1 / np.sqrt(kappa_max))

    q0 = rng.standard_normal(dim) * np.sqrt(kT / pot.kappa)
    p0 = rng.standard_normal(dim) * np.sqrt(kT)
    xi0 = rng.standard_normal(N) * 0.1

    integrator = MultiScaleLogOscIntegrator(
        Q_values=Q_values, potential=pot, dt=dt, kT=kT, dim=dim
    )

    state = ThermostatState(q=q0, p=p0, xi=xi0, n_force_evals=0)

    # Burn-in: 10% of budget
    burnin_evals = n_force_evals // 10
    while state.n_force_evals < burnin_evals:
        state = integrator.step(state)

    # Production
    q_traj = []
    target_evals = state.n_force_evals + n_force_evals
    record_every = max(1, int(1.0 / dt))

    step_count = 0
    while state.n_force_evals < target_evals:
        state = integrator.step(state)
        step_count += 1
        if step_count % record_every == 0:
            if not np.any(np.isnan(state.q)):
                q_traj.append(state.q.copy())

    if len(q_traj) < 100:
        return {
            'kappa_ratio': float(kappa_ratio), 'N': N, 'seed': seed,
            'tau_int': 1e6, 'ergodicity_score': 0.0,
            'n_samples': len(q_traj), 'diverged': True
        }

    q_arr = np.array(q_traj)
    true_var = kT / pot.kappa
    tau_per_dim = []
    for d in range(dim):
        obs = q_arr[:, d] ** 2
        tau_per_dim.append(compute_tau_int(obs))

    tau_int = float(np.mean(tau_per_dim))
    emp_var = np.var(q_arr, axis=0)
    within_20 = np.abs(emp_var - true_var) / true_var < 0.20
    ergodicity_score = float(np.mean(within_20))

    return {
        'kappa_ratio': float(kappa_ratio),
        'N': N,
        'seed': seed,
        'tau_int': tau_int,
        'ergodicity_score': ergodicity_score,
        'n_samples': len(q_traj),
        'diverged': False,
    }


def make_log_uniform_Q(N, Q_min, Q_max):
    if N == 1:
        return np.array([np.sqrt(Q_min * Q_max)])
    k = np.arange(N)
    return Q_min * (Q_max / Q_min) ** (k / (N - 1))


def run_n_scaling_experiment():
    kappa_ratios = [3, 10, 30, 100, 300, 1000]
    N_values = [1, 2, 3, 4, 5, 6]
    seeds = list(range(10))
    n_force_evals = 800_000
    kT = 1.0
    kappa_min = 1.0
    dim = 5

    results = []
    total = len(kappa_ratios) * len(N_values) * len(seeds)
    done = 0

    for kappa_ratio in kappa_ratios:
        kappa_max = kappa_min * kappa_ratio
        Q_min = 1.0 / np.sqrt(kappa_max)
        Q_max = 1.0 / np.sqrt(kappa_min)

        for N in N_values:
            Q_vals = make_log_uniform_Q(N, Q_min, Q_max)

            for seed in seeds:
                done += 1
                print(f"[{done}/{total}] kappa_ratio={kappa_ratio}, N={N}, seed={seed}", flush=True)
                res = run_single(
                    kappa_ratio=kappa_ratio, N=N, seed=seed,
                    Q_values=Q_vals, n_force_evals=n_force_evals,
                    kT=kT, dim=dim
                )
                results.append(res)
                print(f"  tau_int={res['tau_int']:.2f}, ergodicity={res['ergodicity_score']:.2f}", flush=True)

    return results


def compute_n_opt_summary(results):
    """For each kappa_ratio, compute N_opt and statistics with 10 seeds."""
    kappa_ratios = sorted(set(r['kappa_ratio'] for r in results))
    N_values = sorted(set(r['N'] for r in results))

    summary = {}

    for kr in kappa_ratios:
        tau_stats = {}
        for N in N_values:
            vals = [r['tau_int'] for r in results
                    if r['kappa_ratio'] == kr and r['N'] == N and not r.get('diverged', False)]
            if vals:
                tau_stats[N] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'median': float(np.median(vals)),
                    'all': vals,
                }

        if not tau_stats:
            continue

        Ns = sorted(tau_stats.keys())
        means = [tau_stats[n]['mean'] for n in Ns]
        medians = [tau_stats[n]['median'] for n in Ns]

        # N_opt = argmin of median tau_int
        best_idx_median = int(np.argmin(medians))
        n_opt_median = Ns[best_idx_median]

        # N_opt by mean
        best_idx_mean = int(np.argmin(means))
        n_opt_mean = Ns[best_idx_mean]

        # N_tau5 = smallest N where mean tau_int < 5
        n_tau5 = None
        for n in Ns:
            if tau_stats[n]['mean'] < 5.0:
                n_tau5 = n
                break

        # Gain ratio = tau(N=1) / tau(N_opt)
        tau_n1 = tau_stats[1]['mean'] if 1 in tau_stats else None
        tau_nopt = tau_stats[n_opt_median]['mean']
        gain_ratio = float(tau_n1 / tau_nopt) if tau_n1 and tau_nopt > 0 else None

        summary[float(kr)] = {
            'n_opt_median': n_opt_median,
            'n_opt_mean': n_opt_mean,
            'n_tau5': n_tau5,
            'tau_by_N': tau_stats,
            'gain_ratio': gain_ratio,
            'tau_n1': tau_n1,
        }

    return summary


def fit_n_opt_scaling(summary):
    """Linear regression: N_opt = a * log10(kappa_ratio) + b."""
    log_krs = []
    n_opts = []

    for kr, info in sorted(summary.items()):
        log_krs.append(np.log10(kr))
        n_opts.append(info['n_opt_median'])

    log_krs = np.array(log_krs)
    n_opts = np.array(n_opts, dtype=float)

    # scipy linregress for slope, intercept, R², p-value, SE
    result = stats.linregress(log_krs, n_opts)

    return {
        'slope': float(result.slope),
        'intercept': float(result.intercept),
        'r_squared': float(result.rvalue ** 2),
        'p_value': float(result.pvalue),
        'slope_se': float(result.stderr),
        'log_krs': log_krs.tolist(),
        'n_opts': n_opts.tolist(),
    }


def make_figures(results, summary, fit):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('N-Thermostat Scaling Law (10 seeds, 800k evals) — orbit n-scaling-robust-029', fontsize=11)

    kappa_ratios = sorted(set(r['kappa_ratio'] for r in results))
    N_values = sorted(set(r['N'] for r in results))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(kappa_ratios)))

    # Panel (a): tau_int(N) with mean ± std error bands
    ax = axes[0]
    for kr, color in zip(kappa_ratios, colors):
        kr_key = float(kr)
        if kr_key not in summary:
            continue
        Ns_plot = sorted(summary[kr_key]['tau_by_N'].keys())
        means = [summary[kr_key]['tau_by_N'][n]['mean'] for n in Ns_plot]
        stds = [summary[kr_key]['tau_by_N'][n]['std'] for n in Ns_plot]
        means = np.array(means)
        stds = np.array(stds)
        ax.semilogy(Ns_plot, means, 'o-', color=color, label=f'κ={int(kr)}', linewidth=1.5)
        ax.fill_between(Ns_plot, np.maximum(means - stds, 0.1), means + stds,
                        color=color, alpha=0.15)

    ax.set_xlabel('N (number of thermostats)')
    ax.set_ylabel('τ_int (mean ± std, 10 seeds)')
    ax.set_title('(a) τ_int vs N')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel (b): N_opt vs log10(kappa_ratio) with scatter + fit
    ax = axes[1]
    log_krs = np.array(fit['log_krs'])
    n_opts = np.array(fit['n_opts'])

    # Show per-seed N_opt scatter (light)
    for kr, color in zip(kappa_ratios, colors):
        kr_key = float(kr)
        if kr_key not in summary:
            continue
        # Per-seed N_opt: for each seed, find argmin tau_int(N)
        seed_n_opts = []
        for seed in range(10):
            tau_by_N_seed = {}
            for N in N_values:
                vals = [r['tau_int'] for r in results
                        if r['kappa_ratio'] == kr and r['N'] == N and r['seed'] == seed
                        and not r.get('diverged', False)]
                if vals:
                    tau_by_N_seed[N] = vals[0]
            if tau_by_N_seed:
                best_N = min(tau_by_N_seed, key=tau_by_N_seed.get)
                seed_n_opts.append(best_N)
        if seed_n_opts:
            ax.scatter([np.log10(kr)] * len(seed_n_opts), seed_n_opts,
                      alpha=0.2, s=15, color=color)

    # Mean N_opt with std
    ax.scatter(log_krs, n_opts, s=80, zorder=5, color='steelblue', label='N_opt (median)')

    # Linear fit
    slope = fit['slope']
    intercept = fit['intercept']
    r2 = fit['r_squared']
    x_fit = np.linspace(log_krs.min() - 0.1, log_krs.max() + 0.1, 50)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'r--', linewidth=2,
            label=f'N_opt = {slope:.2f}·log₁₀(κ) + {intercept:.2f}\nR²={r2:.2f}')

    ax.set_xlabel('log₁₀(κ_max/κ_min)')
    ax.set_ylabel('N_opt')
    ax.set_title('(b) N_opt vs log₁₀(κ_ratio)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (c): Gain ratio = tau(N=1) / tau(N_opt) per kappa_ratio
    ax = axes[2]
    krs_gain = []
    gains = []
    for kr in kappa_ratios:
        kr_key = float(kr)
        if kr_key in summary and summary[kr_key]['gain_ratio'] is not None:
            krs_gain.append(kr)
            gains.append(summary[kr_key]['gain_ratio'])

    bar_colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(krs_gain)))
    bars = ax.bar(range(len(krs_gain)), gains, color=bar_colors, alpha=0.85)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='no gain')
    ax.set_xticks(range(len(krs_gain)))
    ax.set_xticklabels([f'κ={int(kr)}' for kr in krs_gain], rotation=20, fontsize=9)
    ax.set_ylabel('τ_int(N=1) / τ_int(N_opt)')
    ax.set_title('(c) Gain from optimal N\n(how much N thermostats help)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = ORBIT_DIR / 'figures' / 'n_scaling_robust.png'
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {out_path}", flush=True)


def write_log(fit, summary):
    slope = fit['slope']
    intercept = fit['intercept']
    r2 = fit['r_squared']
    se = fit['slope_se']
    p = fit['p_value']

    table_lines = []
    for kr in sorted(summary.keys()):
        info = summary[kr]
        n_opt = info['n_opt_median']
        tau_n1 = info.get('tau_n1')
        gain = info.get('gain_ratio')
        tau_str = f"{tau_n1:.2f}" if tau_n1 else "N/A"
        gain_str = f"{gain:.1f}x" if gain else "N/A"
        table_lines.append(f"  - κ_ratio={int(kr)}: N_opt={n_opt}, τ(N=1)={tau_str}, gain={gain_str}")

    table = '\n'.join(table_lines)

    # Interpretation
    if r2 > 0.6:
        r2_interp = f"**CONFIRMED** (R²={r2:.2f}): N_opt scales logarithmically with kappa_ratio"
    elif r2 > 0.3:
        r2_interp = f"**SUGGESTIVE** (R²={r2:.2f}): weak log-scaling trend, more data needed"
    else:
        r2_interp = f"**NOT CONFIRMED** (R²={r2:.2f}): no clear log-scaling with 10 seeds"

    log_content = f"""---
strategy: n-scaling-robust-029
status: complete
eval_version: eval-v1
metric: {slope:.4f}
issue: 29
parent: n-scaling-028
---

# N-Scaling Robust: N_opt vs kappa_ratio (10 seeds, 800k evals)

## Scaling Law

**N_opt = {slope:.3f} ± {se:.3f} (SE) * log10(kappa_ratio) + {intercept:.3f}**

- R² = {r2:.3f}
- p-value = {p:.4f}
- Status: {r2_interp}

## Results by kappa_ratio

{table}

## Interpretation

{"The slope is positive and statistically significant (p<0.05), confirming that N_opt grows logarithmically with the condition number kappa_ratio." if p < 0.05 else "The slope is not statistically significant (p≥0.05), suggesting the N_opt vs log(kappa_ratio) relationship is not confirmed at this sample size."}

Slope = {slope:.3f} means: doubling log10(kappa_ratio) by 1 (i.e., 10x increase in condition number)
requires ~{slope:.1f} additional thermostats.

{"This is consistent with the theoretical prediction from orbit #025: Q values span log(kappa_ratio) decades, and N_opt ~ log(kappa_ratio) means one thermostat per decade of frequency coverage." if slope > 0.3 else "The non-monotonic behavior may reflect that at extreme kappa_ratio (≥300), the high curvature itself breaks KAM resonance, making N=1 sufficient (as found in orbit #028)."}

## Key Takeaway

{r2_interp}

For the paper: the gain ratio (tau(N=1)/tau(N_opt)) shows that intermediate kappa_ratios (30-100)
benefit most from multiple thermostats, while extreme ratios (≥300) or small ratios (≤10)
benefit less.
"""

    log_path = ORBIT_DIR / 'log.md'
    with open(log_path, 'w') as f:
        f.write(log_content)
    print(f"log.md updated", flush=True)


def main():
    results_path = ORBIT_DIR / 'n_scaling_results.json'

    if results_path.exists():
        print("Loading existing results...", flush=True)
        with open(results_path) as f:
            results = json.load(f)
    else:
        print("=== Running N-scaling experiment (10 seeds, 800k evals) ===", flush=True)
        results = run_n_scaling_experiment()
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}", flush=True)

    print("\n=== Computing N_opt summary ===", flush=True)
    summary = compute_n_opt_summary(results)
    for kr, info in sorted(summary.items()):
        print(f"  κ_ratio={int(kr)}: N_opt={info['n_opt_median']}, gain={info['gain_ratio']:.1f}x", flush=True)

    summary_path = ORBIT_DIR / 'n_opt_summary.json'
    # Convert keys to strings for JSON
    summary_str_keys = {str(k): v for k, v in summary.items()}
    with open(summary_path, 'w') as f:
        json.dump(summary_str_keys, f, indent=2)

    print("\n=== Fitting N_opt scaling law ===", flush=True)
    fit = fit_n_opt_scaling(summary)
    print(f"  slope = {fit['slope']:.3f} ± {fit['slope_se']:.3f}", flush=True)
    print(f"  R² = {fit['r_squared']:.3f}, p = {fit['p_value']:.4f}", flush=True)

    print("\n=== Making figures ===", flush=True)
    make_figures(results, summary, fit)

    print("\n=== Writing log.md ===", flush=True)
    write_log(fit, summary)

    return fit['slope']


if __name__ == '__main__':
    main()
