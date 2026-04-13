"""
gprime-optimization-073: What is the optimal g'(0) for NH friction?

Orbit 072 showed log-osc (g'(0)=2) is 37% faster than tanh (g'(0)=1) at matched Q.
This orbit finds the optimal α in the family g_α(ξ) = tanh(α·ξ) [g'(0)=α, g(∞)=1].

Family:
  α=0.25 — gentle slope, slow saturation
  α=0.5  — half slope
  α=1.0  — standard tanh
  α=2.0  — steep tanh, approximates log-osc near origin
  α=4.0  — very steep, fast saturation
  α=8.0  — approaches hard clip at ±1

Setup: d=10, κ=100, T=1.0, Q-sweep, N=300k steps, 3 seeds.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import time, os, json

mpl.rcParams.update({
    'font.size': 13, 'axes.titlesize': 14, 'axes.labelsize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.pad_inches': 0.2,
    'axes.spines.top': False, 'axes.spines.right': False,
})

ORBIT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ORBIT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Friction family: g_α(ξ) = tanh(α·ξ)
# ---------------------------------------------------------------------------

def make_g_alpha(alpha):
    def g(xi):
        return np.tanh(alpha * xi)
    g.__name__ = f'tanh_{alpha:.2f}'
    return g

# Also include log-osc for direct comparison
def g_losc(xi):
    return 2.0 * xi / (1.0 + xi**2)

ALPHA_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
FRICTION_FUNCS = {f'alpha={a}': make_g_alpha(a) for a in ALPHA_VALUES}
FRICTION_FUNCS['log-osc'] = g_losc  # reference from orbit 072

COLORS = plt.cm.plasma(np.linspace(0.1, 0.9, len(ALPHA_VALUES)))
COLOR_MAP = {f'alpha={a}': COLORS[i] for i, a in enumerate(ALPHA_VALUES)}
COLOR_MAP['log-osc'] = '#d62728'

# ---------------------------------------------------------------------------
# Potential: d=10 anisotropic Gaussian
# ---------------------------------------------------------------------------

def make_omega2(dim, kappa):
    return np.linspace(1.0, float(kappa), dim)

# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------

def compute_tau_int(x, max_lag=5000):
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    if n < 10: return float('inf')
    var = np.var(x)
    if var < 1e-15: return float('inf')
    fft_x = np.fft.fft(x, n=2*n)
    acf = np.real(np.fft.ifft(fft_x * np.conj(fft_x)))[:n] / (var * n)
    tau = 1.0
    for lag in range(1, min(max_lag, n // 2)):
        if acf[lag] < 0.05:
            break
        tau += 2.0 * acf[lag]
    return float(tau)

# ---------------------------------------------------------------------------
# BAOAB NH integrator (fast version)
# ---------------------------------------------------------------------------

def run_trajectory(omega2, dim, g_func, Q, kT, dt, n_steps, seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(0, 0.5, size=dim)
    p = rng.normal(0, np.sqrt(kT), size=dim)
    xi = 0.0
    half_dt = 0.5 * dt
    burnin = n_steps // 5
    n_collect = n_steps - burnin
    obs = np.empty(n_collect)
    obs_idx = 0

    for step in range(n_steps):
        # B: xi half-step
        kinetic = np.sum(p**2)
        xi += half_dt * (kinetic - dim * kT) / Q
        # A: friction half-step
        g_val = float(g_func(xi))
        scale = np.clip(np.exp(-g_val * half_dt), 1e-10, 1e10)
        p *= scale
        # A: force half-step
        p -= half_dt * (omega2 * q)
        # O: drift full step
        q += dt * p
        # A: force half-step
        p -= half_dt * (omega2 * q)
        # A: friction half-step (re-use xi from before drift)
        g_val2 = float(g_func(xi))
        scale2 = np.clip(np.exp(-g_val2 * half_dt), 1e-10, 1e10)
        p *= scale2
        # B: xi half-step
        kinetic = np.sum(p**2)
        xi += half_dt * (kinetic - dim * kT) / Q

        if step >= burnin:
            obs[obs_idx] = q[-1]**2  # stiffest mode
            obs_idx += 1

        if np.isnan(xi) or np.any(np.isnan(q)):
            return float('inf')

    return compute_tau_int(obs[:obs_idx])


def _worker(args):
    return run_trajectory(*args)

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep():
    dim = 10
    kappa = 100
    kT = 1.0
    dt = 0.005
    n_steps = 300_000
    Q_values = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
    seeds = [42, 123, 7]
    omega2 = make_omega2(dim, kappa)

    results = {}
    t0 = time.time()

    for method_name, g_func in FRICTION_FUNCS.items():
        results[method_name] = {}
        for Q in Q_values:
            taus = [run_trajectory(omega2, dim, g_func, Q, kT, dt, n_steps, s)
                    for s in seeds]
            valid = [t for t in taus if t < 1e6 and not np.isinf(t)]
            median_tau = float(np.median(valid)) if valid else float('inf')
            n_inf = sum(1 for t in taus if np.isinf(t) or t >= 1e6)
            results[method_name][Q] = {'median_tau': median_tau, 'n_inf': n_inf}
            print(f"  {method_name:12s} Q={Q:6.1f}: τ={median_tau:8.1f}  n_inf={n_inf}")

    print(f"\nTotal wall time: {time.time()-t0:.1f}s")
    return results


def plot_results(results):
    Q_values = sorted(list(results['alpha=1.0'].keys()))
    methods_alpha = [f'alpha={a}' for a in ALPHA_VALUES]

    # Figure 1: τ_int vs Q for each α
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for method_name in methods_alpha:
        alpha = float(method_name.split('=')[1])
        taus = [results[method_name][Q]['median_tau'] for Q in Q_values]
        ax.semilogy(Q_values, taus, 'o-', color=COLOR_MAP[method_name],
                    label=f'α={alpha:.2f} [g\'(0)={alpha:.2f}]', lw=1.5)
    # log-osc reference
    taus_losc = [results['log-osc'][Q]['median_tau'] for Q in Q_values]
    ax.semilogy(Q_values, taus_losc, 's--', color='#d62728',
                label="log-osc [g'(0)=2.0]", lw=2, alpha=0.7)
    ax.set_xlabel('Q')
    ax.set_ylabel('τ_int (stiffest mode)')
    ax.set_xscale('log')
    ax.set_title(f'd=10, κ=100: τ_int vs Q for g_α = tanh(α·ξ)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Figure 2: best τ_int vs α
    ax2 = axes[1]
    alphas = []
    best_taus = []
    best_Qs = []
    for method_name in methods_alpha:
        alpha = float(method_name.split('=')[1])
        best_Q = min(Q_values, key=lambda Q: results[method_name][Q]['median_tau'])
        best_tau = results[method_name][best_Q]['median_tau']
        alphas.append(alpha)
        best_taus.append(best_tau)
        best_Qs.append(best_Q)

    # tanh reference (α=1)
    tanh_best = best_taus[ALPHA_VALUES.index(1.0)]
    ratios = [tanh_best / t for t in best_taus]

    ax2.plot(alphas, ratios, 'ko-', lw=2, markersize=8)
    # log-osc reference
    losc_best = min(results['log-osc'][Q]['median_tau'] for Q in Q_values)
    losc_ratio = tanh_best / losc_best
    ax2.axhline(losc_ratio, color='#d62728', ls='--', lw=1.5,
                label=f"log-osc ratio={losc_ratio:.2f}")
    ax2.axhline(1.0, color='gray', ls=':', lw=1)
    ax2.set_xlabel('α  [g\'(0) = α]')
    ax2.set_ylabel('τ_int(tanh α=1) / τ_int(α)  [ratio > 1 = faster than tanh]')
    ax2.set_title("Best τ_int vs g'(0), normalized to tanh (α=1)")
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    for i, (a, r, q) in enumerate(zip(alphas, ratios, best_Qs)):
        ax2.annotate(f'Q*={q}', (a, r), textcoords='offset points', xytext=(5, 5), fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'gprime_optimization.png')
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_path}")


def main():
    print("=" * 60)
    print("Orbit 073: g'(0) optimization — family tanh(α·ξ)")
    print("=" * 60)

    results = run_sweep()

    # Save results
    results_path = os.path.join(ORBIT_DIR, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results.json")

    plot_results(results)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Best τ_int per method (min over Q)")
    print("=" * 60)

    tanh_best_tau = min(results['alpha=1.0'][Q]['median_tau']
                        for Q in results['alpha=1.0'])
    tanh_best_Q = min(results['alpha=1.0'], key=lambda Q: results['alpha=1.0'][Q]['median_tau'])

    print(f"\nBaseline tanh (α=1): best τ={tanh_best_tau:.1f} at Q={tanh_best_Q}")
    print()

    for method_name in list(FRICTION_FUNCS.keys()):
        best_Q = min(results[method_name], key=lambda Q: results[method_name][Q]['median_tau'])
        best_tau = results[method_name][best_Q]['median_tau']
        ratio = tanh_best_tau / best_tau
        print(f"  {method_name:12s}: best τ={best_tau:8.1f} at Q={best_Q:5.1f}, "
              f"ratio vs tanh={ratio:.3f}")

    # Find optimal α
    alpha_names = [f'alpha={a}' for a in ALPHA_VALUES]
    best_method = min(alpha_names,
                      key=lambda m: min(results[m][Q]['median_tau'] for Q in results[m]))
    best_alpha = float(best_method.split('=')[1])
    best_tau_overall = min(results[best_method][Q]['median_tau'] for Q in results[best_method])
    best_ratio = tanh_best_tau / best_tau_overall

    print(f"\nOptimal α: {best_alpha:.2f} (g'(0)={best_alpha:.2f}), "
          f"τ={best_tau_overall:.1f}, {best_ratio:.2f}× faster than tanh")
    print(f"log-osc: τ={min(results['log-osc'][Q]['median_tau'] for Q in results['log-osc']):.1f}, "
          f"{tanh_best_tau / min(results['log-osc'][Q]['median_tau'] for Q in results['log-osc']):.2f}× faster than tanh")


if __name__ == '__main__':
    main()
