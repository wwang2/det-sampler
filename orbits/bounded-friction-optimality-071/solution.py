"""
Bounded-friction-optimality-071: Is tanh special among bounded odd friction functions?

Orbit 052 showed g'>=0 is NOT the cause of the 536x gap.
Orbit 069 showed unbounded g causes catastrophic instability.

This experiment asks: among normalized bounded odd functions (g(inf)=1, g'(0)=1),
does the specific shape matter, or is bounded = sufficient?

Friction functions tested (all normalized: g(inf)=1, g'(0)=1):
  1. tanh: g(xi) = tanh(xi)
  2. arctan: g(xi) = (2/pi)*arctan(xi)
  3. erf: g(xi) = erf(sqrt(pi)/2 * xi)  [rescaled so g'(0)=1]
  4. rational: g(xi) = xi/(1+|xi|)
  5. clipped-linear: g(xi) = clip(xi, -1, 1)  [hard-clipped, approx log-osc near origin]

Key question: is tanh special, or does any bounded function work equally well?
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from multiprocessing import Pool
import time
import os
import json
from scipy.special import erf

# ---------------------------------------------------------------------------
# Plotting defaults
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.pad_inches': 0.2,
})

ORBIT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ORBIT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Friction functions — all normalized: g(inf)=1, g'(0)=1
# ---------------------------------------------------------------------------

def g_tanh(xi):
    """tanh: the standard bounded NH friction."""
    return np.tanh(xi)

def g_arctan(xi):
    """arctan, rescaled: (2/pi)*arctan(xi). g'(0) = 2/pi * 1 = 2/pi ≠ 1.
    Normalize: g(xi) = (2/pi)*arctan(pi/2 * xi). g'(0) = (2/pi)*(pi/2)=1. g(inf)=1."""
    return (2.0 / np.pi) * np.arctan((np.pi / 2.0) * xi)

def g_erf(xi):
    """erf rescaled: g(xi) = erf(sqrt(pi)/2 * xi).
    g'(0) = erf' at 0 * sqrt(pi)/2 = (2/sqrt(pi)) * sqrt(pi)/2 = 1. g(inf)=1."""
    return erf((np.sqrt(np.pi) / 2.0) * xi)

def g_rational(xi):
    """Rational: g(xi) = xi/(1+|xi|). g'(0)=1, g(inf)=1."""
    return xi / (1.0 + np.abs(xi))

def g_clipped(xi):
    """Hard-clipped linear: g(xi) = clip(xi, -1, 1).
    This directly tests: does log-osc catch up to tanh if we just clip it?
    g'(0)=1, g(inf)=1."""
    return np.clip(xi, -1.0, 1.0)

FRICTION_FUNCS = {
    'tanh': g_tanh,
    'arctan': g_arctan,
    'erf': g_erf,
    'rational': g_rational,
    'clipped': g_clipped,
}

COLORS = {
    'tanh': '#1f77b4',
    'arctan': '#ff7f0e',
    'erf': '#2ca02c',
    'rational': '#d62728',
    'clipped': '#9467bd',
}

LABELS = {
    'tanh': 'tanh [baseline]',
    'arctan': 'arctan (2/π·arctan(π/2·ξ))',
    'erf': 'erf (erf(√π/2·ξ))',
    'rational': 'rational (ξ/(1+|ξ|))',
    'clipped': 'clipped-linear (clip(ξ,−1,1))',
}

# ---------------------------------------------------------------------------
# Anisotropic Gaussian potential
# ---------------------------------------------------------------------------

class AnisotropicGaussian:
    def __init__(self, dim, kappa, kT=1.0):
        self.dim = dim
        self.kT = kT
        self.kappa = kappa
        # 2D: omega1^2=1, omega2^2=kappa (stiffness ratio = kappa)
        self.omega2 = np.array([1.0, float(kappa)])
        self.name = f"aniso_gauss_k{kappa}"


# ---------------------------------------------------------------------------
# Integrated autocorrelation time
# ---------------------------------------------------------------------------

def compute_tau_int(x, max_lag=5000):
    """Integrated autocorrelation time via FFT."""
    x = x - np.mean(x)
    n = len(x)
    if n < 10:
        return float('inf')
    var = np.var(x)
    if var < 1e-15:
        return float('inf')

    fft_x = np.fft.fft(x, n=2 * n)
    acf = np.real(np.fft.ifft(fft_x * np.conj(fft_x)))[:n] / (var * n)

    tau = 1.0
    for lag in range(1, min(max_lag, n // 2)):
        if acf[lag] < 0.05:
            break
        tau += 2.0 * acf[lag]
    return float(tau)


# ---------------------------------------------------------------------------
# BAOAB NH integrator with generic g(xi)
# ---------------------------------------------------------------------------

def run_trajectory(omega2, dim, g_func, Q, kT, dt, n_steps, seed, mass=1.0):
    """Run BAOAB NH trajectory on anisotropic Gaussian. Returns tau_int for stiffest mode."""
    rng = np.random.default_rng(seed)
    q = rng.normal(0, 0.5, size=dim)
    p = rng.normal(0, np.sqrt(mass * kT), size=dim)
    xi = 0.0

    half_dt = 0.5 * dt
    burnin = n_steps // 5

    n_collect = n_steps - burnin
    obs = np.empty(n_collect)
    obs_idx = 0

    for step in range(n_steps):
        # B: half-step thermostat (xi update)
        kinetic = np.sum(p ** 2) / mass
        xi += half_dt * (kinetic - dim * kT) / Q

        # A: half-step momentum — friction
        g_val = float(g_func(xi))
        scale = np.exp(-g_val * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p *= scale
        # A: half-step momentum — force
        p -= half_dt * (omega2 * q)

        # O: full-step position
        q += dt * p / mass

        # A: half-step momentum — force
        p -= half_dt * (omega2 * q)
        # A: half-step momentum — friction
        g_val = float(g_func(xi))
        scale = np.exp(-g_val * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p *= scale

        # B: half-step thermostat
        kinetic = np.sum(p ** 2) / mass
        xi += half_dt * (kinetic - dim * kT) / Q

        if step >= burnin:
            # observable: q_stiff^2 (stiffest mode = index 1)
            obs[obs_idx] = q[1] ** 2
            obs_idx += 1

        if np.any(np.isnan(q)) or np.any(np.isnan(p)) or np.isnan(xi):
            return {'tau_int': float('inf'), 'nan': True}

    tau = compute_tau_int(obs[:obs_idx])
    return {'tau_int': tau, 'nan': False}


def _worker(args):
    return run_trajectory(**args)


# ---------------------------------------------------------------------------
# Double-well trajectory (for robustness check)
# ---------------------------------------------------------------------------

def run_dwell_trajectory(g_func, Q, kT, dt, n_steps, seed, mass=1.0):
    """2D double-well: V = (q1^2-1)^2 + 0.5*q2^2."""
    rng = np.random.default_rng(seed)
    q = rng.normal(0, 0.5, size=2)
    p = rng.normal(0, np.sqrt(mass * kT), size=2)
    xi = 0.0
    dim = 2

    half_dt = 0.5 * dt
    burnin = n_steps // 5
    n_collect = n_steps - burnin
    obs = np.empty(n_collect)
    obs_idx = 0

    def grad(q):
        return np.array([4.0 * q[0] * (q[0] ** 2 - 1.0), q[1]])

    for step in range(n_steps):
        kinetic = np.sum(p ** 2) / mass
        xi += half_dt * (kinetic - dim * kT) / Q

        g_val = float(g_func(xi))
        scale = np.clip(np.exp(-g_val * half_dt), 1e-10, 1e10)
        p *= scale
        p -= half_dt * grad(q)

        q += dt * p / mass

        p -= half_dt * grad(q)
        g_val = float(g_func(xi))
        scale = np.clip(np.exp(-g_val * half_dt), 1e-10, 1e10)
        p *= scale

        kinetic = np.sum(p ** 2) / mass
        xi += half_dt * (kinetic - dim * kT) / Q

        if step >= burnin:
            obs[obs_idx] = q[0] ** 2
            obs_idx += 1

        if np.any(np.isnan(q)) or np.any(np.isnan(p)):
            return {'tau_int': float('inf'), 'nan': True}

    tau = compute_tau_int(obs[:obs_idx])
    return {'tau_int': tau, 'nan': False}


def _worker_dwell(args):
    return run_dwell_trajectory(**args)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark():
    kT = 1.0
    dt = 0.01
    n_steps = 100_000
    seeds = list(range(16))
    dim = 2

    kappa_values = [10, 100, 1000]
    Q_values = [0.1, 0.3, 1.0, 3.0, 10.0]
    methods = list(FRICTION_FUNCS.keys())

    results = {}

    for kappa in kappa_values:
        pot = AnisotropicGaussian(dim=dim, kappa=kappa, kT=kT)
        results[kappa] = {}

        for method_name in methods:
            g_func = FRICTION_FUNCS[method_name]
            results[kappa][method_name] = {}

            for Q in Q_values:
                job_args = [
                    dict(omega2=pot.omega2, dim=dim, g_func=g_func, Q=Q,
                         kT=kT, dt=dt, n_steps=n_steps, seed=s)
                    for s in seeds
                ]

                with Pool(min(len(seeds), os.cpu_count() or 4)) as pool:
                    seed_results = pool.map(_worker, job_args)

                taus = [r['tau_int'] for r in seed_results if not r['nan'] and r['tau_int'] < 1e6]
                n_nan = sum(1 for r in seed_results if r['nan'])

                if taus:
                    median_tau = float(np.median(taus))
                    mean_tau = float(np.mean(taus))
                    std_tau = float(np.std(taus))
                else:
                    median_tau = float('inf')
                    mean_tau = float('inf')
                    std_tau = float('inf')

                results[kappa][method_name][Q] = {
                    'median_tau': median_tau,
                    'mean_tau': mean_tau,
                    'std_tau': std_tau,
                    'n_nan': n_nan,
                    'n_valid': len(taus),
                }

                print(f"  kappa={kappa}, {method_name}, Q={Q}: "
                      f"median_tau={median_tau:.1f}, nan={n_nan}/{len(seeds)}")

    return results


def run_double_well():
    kT = 1.0
    dt = 0.01
    n_steps = 100_000
    seeds = list(range(16))
    methods = list(FRICTION_FUNCS.keys())
    Q_values = [0.1, 0.3, 1.0, 3.0, 10.0]

    results = {}
    for method_name in methods:
        g_func = FRICTION_FUNCS[method_name]
        results[method_name] = {}
        for Q in Q_values:
            job_args = [
                dict(g_func=g_func, Q=Q, kT=kT, dt=dt, n_steps=n_steps, seed=s)
                for s in seeds
            ]
            with Pool(min(len(seeds), os.cpu_count() or 4)) as pool:
                seed_results = pool.map(_worker_dwell, job_args)

            taus = [r['tau_int'] for r in seed_results if not r['nan'] and r['tau_int'] < 1e6]
            n_nan = sum(1 for r in seed_results if r['nan'])
            median_tau = float(np.median(taus)) if taus else float('inf')
            results[method_name][Q] = {
                'median_tau': median_tau,
                'n_nan': n_nan,
                'n_valid': len(taus),
            }
            print(f"  double-well, {method_name}, Q={Q}: median_tau={median_tau:.1f}, nan={n_nan}")

    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_friction_functions():
    xi = np.linspace(-5, 5, 500)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for name, g in FRICTION_FUNCS.items():
        ax.plot(xi, g(xi), label=LABELS[name], color=COLORS[name], linewidth=2)
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.axhline(1, color='gray', lw=0.5, ls=':', alpha=0.5, label='g=±1 (asymptote)')
    ax.axhline(-1, color='gray', lw=0.5, ls=':', alpha=0.5)
    ax.set_xlabel('ξ')
    ax.set_ylabel('g(ξ)')
    ax.set_title('Normalized bounded friction functions (g(∞)=1, g\'(0)=1)', fontweight='bold')
    ax.legend(frameon=False, fontsize=10)
    ax.set_ylim(-1.5, 1.5)
    fig.savefig(os.path.join(FIG_DIR, 'friction_functions.png'))
    plt.close(fig)
    print("Saved friction_functions.png")


def plot_tau_comparison(results, kappa_values=(10, 100, 1000)):
    n = len(kappa_values)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for idx, kappa in enumerate(kappa_values):
        ax = axes[idx]
        for method_name in FRICTION_FUNCS:
            Q_vals = sorted(results[kappa][method_name].keys())
            medians = [results[kappa][method_name][Q]['median_tau'] for Q in Q_vals]
            valid = [(Q, t) for Q, t in zip(Q_vals, medians) if t < 1e6]
            if valid:
                Qs, taus = zip(*valid)
                ax.plot(Qs, taus, 'o-', label=LABELS[method_name],
                        color=COLORS[method_name], linewidth=2, markersize=5)

        ax.set_xscale('log')
        ax.set_yscale('log')
        panel = chr(ord('a') + idx)
        ax.set_title(f'({panel}) κ={kappa}', fontweight='bold')
        ax.set_xlabel('Thermostat mass Q')
        ax.set_ylabel('Median τ_int (q²_stiff)')
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(FIG_DIR, 'tau_vs_Q.png'))
    plt.close(fig)
    print("Saved tau_vs_Q.png")


def plot_best_tau_bar(results, kappa_values=(10, 100, 1000)):
    """Bar chart: best tau_int (over Q) per method per kappa."""
    methods = list(FRICTION_FUNCS.keys())
    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for i, kappa in enumerate(kappa_values):
        best_taus = []
        for method_name in methods:
            taus_by_Q = [results[kappa][method_name][Q]['median_tau']
                         for Q in results[kappa][method_name]]
            valid = [t for t in taus_by_Q if t < 1e6]
            best_taus.append(min(valid) if valid else float('inf'))

        # normalize to tanh
        tanh_best = best_taus[methods.index('tanh')]
        normalized = [t / tanh_best if t < 1e6 else float('nan') for t in best_taus]

        bars = ax.bar(x + i * width, normalized, width,
                      label=f'κ={kappa}', alpha=0.8)

    ax.axhline(1.0, color='black', lw=1, ls='--', alpha=0.5, label='tanh baseline')
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods, rotation=20, ha='right')
    ax.set_ylabel('τ_int / τ_int(tanh)  [lower=better]')
    ax.set_title('Best τ_int (over Q) relative to tanh', fontweight='bold')
    ax.legend(frameon=False)
    ax.set_yscale('log')
    fig.savefig(os.path.join(FIG_DIR, 'best_tau_bar.png'))
    plt.close(fig)
    print("Saved best_tau_bar.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    print("=" * 60)
    print("Orbit 071: Bounded Friction Optimality")
    print("=" * 60)

    # Plot friction functions
    print("\n--- Friction function shapes ---")
    plot_friction_functions()

    # Main benchmark: anisotropic Gaussian
    print("\n--- Anisotropic Gaussian benchmark ---")
    results = run_benchmark()

    # Double-well benchmark
    print("\n--- Double-well benchmark ---")
    dw_results = run_double_well()

    # Figures
    print("\n--- Generating figures ---")
    plot_tau_comparison(results)
    plot_best_tau_bar(results)

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY: Best τ_int per method (minimized over Q)")
    print("=" * 60)
    methods = list(FRICTION_FUNCS.keys())

    summary = {}
    for kappa in [10, 100, 1000]:
        summary[kappa] = {}
        print(f"\nκ = {kappa}:")
        tanh_best = None
        for method_name in methods:
            taus_by_Q = {Q: results[kappa][method_name][Q]['median_tau']
                         for Q in results[kappa][method_name]}
            valid = {Q: t for Q, t in taus_by_Q.items() if t < 1e6}
            if valid:
                best_Q = min(valid, key=valid.get)
                best_tau = valid[best_Q]
            else:
                best_Q = None
                best_tau = float('inf')
            summary[kappa][method_name] = {'best_tau': best_tau, 'best_Q': best_Q}
            if method_name == 'tanh':
                tanh_best = best_tau

        for method_name in methods:
            bt = summary[kappa][method_name]['best_tau']
            bq = summary[kappa][method_name]['best_Q']
            ratio = bt / tanh_best if tanh_best and tanh_best > 0 else float('nan')
            print(f"  {method_name:12s}: best_tau={bt:8.1f} at Q={bq}, "
                  f"ratio_to_tanh={ratio:.3f}")

    # Double-well summary
    print("\nDouble-well (best over Q):")
    tanh_dw_best = None
    for method_name in methods:
        valid = {Q: dw_results[method_name][Q]['median_tau']
                 for Q in dw_results[method_name]
                 if dw_results[method_name][Q]['median_tau'] < 1e6}
        best_tau = min(valid.values()) if valid else float('inf')
        if method_name == 'tanh':
            tanh_dw_best = best_tau
        ratio = best_tau / tanh_dw_best if tanh_dw_best and tanh_dw_best > 0 else float('nan')
        print(f"  {method_name:12s}: best_tau={best_tau:.1f}, ratio_to_tanh={ratio:.3f}")

    # Headline: is any method better than tanh? at kappa=1000
    best_ratio_1000 = 1.0
    best_method_1000 = 'tanh'
    tanh_best_1000 = summary[1000]['tanh']['best_tau']
    for method_name in methods:
        if method_name == 'tanh':
            continue
        bt = summary[1000][method_name]['best_tau']
        if bt < tanh_best_1000:
            ratio = tanh_best_1000 / bt
            if ratio > best_ratio_1000:
                best_ratio_1000 = ratio
                best_method_1000 = method_name

    print(f"\nHeadline (kappa=1000): best competitor to tanh is '{best_method_1000}' "
          f"with improvement ratio {best_ratio_1000:.3f}x")

    # Save results
    all_results = {
        'aniso_gaussian': {
            str(k): {
                m: {str(Q): v for Q, v in qd.items()}
                for m, qd in kd.items()
            }
            for k, kd in results.items()
        },
        'double_well': {
            m: {str(Q): v for Q, v in qd.items()}
            for m, qd in dw_results.items()
        },
        'summary': {
            str(k): {
                m: {'best_tau': v['best_tau'], 'best_Q': str(v['best_Q'])}
                for m, v in kd.items()
            }
            for k, kd in summary.items()
        },
    }

    with open(os.path.join(ORBIT_DIR, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved results.json")

    elapsed = time.time() - t0
    print(f"Total wall time: {elapsed:.1f}s")

    return results, dw_results, summary


if __name__ == '__main__':
    results, dw_results, summary = main()
