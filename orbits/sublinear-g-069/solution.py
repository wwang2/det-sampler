"""
Sublinear-g-069: Compare friction functions for Nose-Hoover thermostats.

Tests g(xi) = xi * log(1+xi^2) / sqrt(1+xi^2) — unbounded, odd, monotone,
sublinear growth — against tanh, log-osc, and linear on stiff anisotropic
Gaussian targets.

The key question: does unbounded-but-sublinear g avoid both tanh's frequency
ceiling (bounded |g| <= 1) and log-osc's sign reversal (g'(xi) < 0 for |xi|>1)?
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from multiprocessing import Pool
from functools import partial
import time
import os
import json

# ---------------------------------------------------------------------------
# Plotting defaults (from style guide)
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

# ---------------------------------------------------------------------------
# Friction functions g(xi) and their primitives G(xi) = int_0^xi g(s) ds
# ---------------------------------------------------------------------------

def g_tanh(xi):
    return np.tanh(xi)

def G_tanh(xi):
    return np.log(np.cosh(xi))

def g_losc(xi):
    """Log-oscillator: g(xi) = 2*xi / (1 + xi^2)"""
    return 2.0 * xi / (1.0 + xi**2)

def G_losc(xi):
    return np.log(1.0 + xi**2)

def g_linear(xi):
    return xi

def G_linear(xi):
    return 0.5 * xi**2

def g_new(xi):
    """Sublinear unbounded: g(xi) = xi * log(1+xi^2) / sqrt(1+xi^2)"""
    return xi * np.log(1.0 + xi**2) / np.sqrt(1.0 + xi**2)

def G_new(xi):
    """Numerical primitive of g_new via quadrature (vectorized)."""
    # G(xi) = sqrt(1+xi^2) * log(1+xi^2) - 2*sqrt(1+xi^2) + 2
    # Verify: d/dxi [sqrt(1+xi^2)*log(1+xi^2)] = xi/sqrt(1+xi^2)*log(1+xi^2) + xi*2xi/(1+xi^2)*1/sqrt(... wait
    # Let's just compute it numerically for the plot, but use analytical for integration.
    # Actually: let u = 1+xi^2. Then G = integral of xi*log(u)/sqrt(u) dxi
    # With substitution: du = 2xi dxi => xi dxi = du/2
    # G = (1/2) integral log(u)/sqrt(u) du = (1/2)[2*sqrt(u)*log(u) - 4*sqrt(u)] + C
    # G = sqrt(u)*log(u) - 2*sqrt(u) + C, with G(0) = 0: G(0) = 0 - 2 + C = 0 => C = 2
    # G(xi) = sqrt(1+xi^2)*log(1+xi^2) - 2*sqrt(1+xi^2) + 2
    u = 1.0 + xi**2
    return np.sqrt(u) * np.log(u) - 2.0 * np.sqrt(u) + 2.0

FRICTION_FUNCS = {
    'tanh': (g_tanh, G_tanh),
    'log-osc': (g_losc, G_losc),
    'linear': (g_linear, G_linear),
    'sublinear': (g_new, G_new),
}

COLORS = {
    'tanh': '#1f77b4',
    'log-osc': '#d62728',
    'linear': '#2ca02c',
    'sublinear': '#ff7f0e',
}

# ---------------------------------------------------------------------------
# Anisotropic Gaussian potential (d-dimensional)
# ---------------------------------------------------------------------------

class AnisotropicGaussian:
    """U(q) = 0.5 * sum_i omega_i^2 * q_i^2, with log-spaced frequencies."""

    def __init__(self, dim, kappa_max, kT=1.0):
        self.dim = dim
        self.kT = kT
        self.kappa_max = kappa_max
        # Log-spaced frequencies: omega_i^2 from 1 to kappa_max
        self.omega2 = np.logspace(0, np.log10(kappa_max), dim)
        self.name = f"aniso_gauss_d{dim}_k{kappa_max}"

    def energy(self, q):
        return 0.5 * np.sum(self.omega2 * q**2)

    def gradient(self, q):
        return self.omega2 * q


# ---------------------------------------------------------------------------
# Generalized NH integrator with arbitrary friction g(xi)
# BAOAB-style velocity-Verlet splitting
# ---------------------------------------------------------------------------

class GeneralizedNH:
    """Nose-Hoover thermostat with general friction function g(xi).

    dq/dt = p/m
    dp/dt = -dU/dq - g(xi) * p
    dxi/dt = (1/Q) * (p^2/m - d*kT)

    The standard NH uses g(xi) = xi. Tanh-NH uses g = tanh, etc.
    The invariant density is rho ~ exp(-U/kT - p^2/(2m*kT) - Q*G(xi)/kT)
    where G(xi) = int_0^xi g(s) ds, provided G is convex (g' > 0).
    """

    def __init__(self, dim, g_func, kT=1.0, Q=1.0, mass=1.0):
        self.dim = dim
        self.g = g_func
        self.kT = kT
        self.Q = Q
        self.mass = mass
        self.name = "generalized_nh"

    def initial_state(self, q0, p0=None, xi0=0.0, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
        if p0 is None:
            p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        return (q0.copy(), p0.copy(), xi0, 0)


def run_trajectory(potential, g_func, dim, Q, kT, dt, n_steps, seed, mass=1.0):
    """Run a single NH trajectory with given friction function.

    Returns dict with tau_int for the stiffest mode (q_d^2).
    Uses BAOAB-style splitting for stability.
    """
    rng = np.random.default_rng(seed)
    q = rng.normal(0, 0.5, size=dim)
    p = rng.normal(0, np.sqrt(mass * kT), size=dim)
    xi = 0.0

    omega2 = potential.omega2
    half_dt = 0.5 * dt
    burnin = n_steps // 5

    # Pre-allocate for the stiffest-mode observable: q_d^2
    n_collect = n_steps - burnin
    obs = np.empty(n_collect)
    obs_idx = 0

    for step in range(n_steps):
        # --- B: half-step thermostat ---
        kinetic = np.sum(p**2) / mass
        dxidt = (kinetic - dim * kT) / Q
        xi = xi + half_dt * dxidt

        # --- A: half-step momentum (friction + force) ---
        g_val = g_func(xi)
        # Analytical friction scaling: p *= exp(-g(xi)*dt/2)
        scale = np.exp(-g_val * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale
        # Force kick
        grad_U = omega2 * q
        p = p - half_dt * grad_U

        # --- O: full-step position ---
        q = q + dt * p / mass

        # --- A: half-step momentum (force + friction) ---
        grad_U = omega2 * q
        p = p - half_dt * grad_U
        g_val = g_func(xi)
        scale = np.exp(-g_val * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale

        # --- B: half-step thermostat ---
        kinetic = np.sum(p**2) / mass
        dxidt = (kinetic - dim * kT) / Q
        xi = xi + half_dt * dxidt

        # Collect observable after burn-in: q_d^2 (stiffest mode)
        if step >= burnin:
            obs[obs_idx] = q[-1]**2
            obs_idx += 1

        # NaN check
        if np.any(np.isnan(q)) or np.any(np.isnan(p)):
            return {'tau_int': float('inf'), 'nan': True, 'seed': seed}

    # Compute integrated autocorrelation time
    obs = obs[:obs_idx]
    tau = compute_tau_int(obs)

    return {'tau_int': tau, 'nan': False, 'seed': seed, 'mean_obs': np.mean(obs)}


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
# Worker function for parallel seed execution
# ---------------------------------------------------------------------------

def _worker(args):
    """Unpack args and call run_trajectory."""
    return run_trajectory(**args)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark():
    """Run the full benchmark grid: 4 friction functions x Q grid x kappa grid x seeds."""

    dim = 10
    kT = 1.0
    dt = 0.005
    n_steps = 200_000  # 200k steps at dt=0.005
    seeds = list(range(20))  # 20 seeds

    kappa_values = [10, 100, 1000]
    Q_values = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    methods = ['tanh', 'log-osc', 'linear', 'sublinear']

    results = {}

    for kappa in kappa_values:
        potential = AnisotropicGaussian(dim=dim, kappa_max=kappa, kT=kT)
        results[kappa] = {}

        for method_name in methods:
            g_func = FRICTION_FUNCS[method_name][0]
            results[kappa][method_name] = {}

            for Q in Q_values:
                # Build argument list for parallel execution
                job_args = [
                    dict(potential=potential, g_func=g_func, dim=dim, Q=Q,
                         kT=kT, dt=dt, n_steps=n_steps, seed=s)
                    for s in seeds
                ]

                # Run seeds in parallel
                with Pool(min(len(seeds), os.cpu_count() or 4)) as pool:
                    seed_results = pool.map(_worker, job_args)

                taus = [r['tau_int'] for r in seed_results if not r['nan']]
                n_nan = sum(1 for r in seed_results if r['nan'])

                if len(taus) > 0:
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
                    'n_seeds': len(seeds),
                }

                print(f"  kappa={kappa}, {method_name}, Q={Q}: "
                      f"median_tau={median_tau:.1f}, nan={n_nan}/{len(seeds)}")

    return results


def run_additional_targets():
    """Run 1D double-well and 2D 4-Gaussian mixture tests."""
    import sys
    sys.path.insert(0, os.path.join(ORBIT_DIR, '..', '..'))

    from research.eval.potentials import DoubleWell2D, GaussianMixture2D
    from research.eval.evaluator import run_sampler
    from research.eval.baselines import NoseHoover

    # We can't easily plug custom g into the evaluator's run_sampler
    # because it uses the NoseHoover class. Instead, run our own trajectories.

    results = {}

    # 1D double-well: V(x) = (x^2-1)^2
    # We adapt our integrator for 1D
    class DoubleWellPot:
        def __init__(self):
            self.dim = 1
            self.omega2 = None  # marker for non-Gaussian
            self.name = "double_well_1d"
        def energy(self, q):
            return (q[0]**2 - 1)**2
        def gradient(self, q):
            return np.array([4.0 * q[0] * (q[0]**2 - 1)])

    class GMM4Pot:
        def __init__(self, sep=4.0, sigma=0.5):
            self.dim = 2
            self.omega2 = None
            self.name = "gmm4_2d"
            self.centers = np.array([
                [sep/2, sep/2], [-sep/2, sep/2],
                [-sep/2, -sep/2], [sep/2, -sep/2]
            ])
            self.sigma = sigma
        def energy(self, q):
            diffs = self.centers - q
            exponents = -0.5 * np.sum(diffs**2, axis=1) / self.sigma**2
            total = np.sum(np.exp(exponents))
            if total < 1e-300:
                return 700.0
            return -np.log(total)
        def gradient(self, q):
            diffs = self.centers - q
            exponents = -0.5 * np.sum(diffs**2, axis=1) / self.sigma**2
            densities = np.exp(exponents)
            total = np.sum(densities)
            if total < 1e-300:
                return np.zeros(2)
            weighted = densities[:, np.newaxis] * diffs / self.sigma**2
            return -np.sum(weighted, axis=0) / total

    def run_general_trajectory(potential, g_func, dim, Q, kT, dt, n_steps, seed, mass=1.0):
        """General trajectory runner for arbitrary potentials."""
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
            kinetic = np.sum(p**2) / mass
            dxidt = (kinetic - dim * kT) / Q
            xi = xi + half_dt * dxidt

            g_val = g_func(xi)
            scale = np.clip(np.exp(-g_val * half_dt), 1e-10, 1e10)
            p = p * scale
            grad_U = potential.gradient(q)
            p = p - half_dt * grad_U

            q = q + dt * p / mass

            grad_U = potential.gradient(q)
            p = p - half_dt * grad_U
            g_val = g_func(xi)
            scale = np.clip(np.exp(-g_val * half_dt), 1e-10, 1e10)
            p = p * scale

            kinetic = np.sum(p**2) / mass
            dxidt = (kinetic - dim * kT) / Q
            xi = xi + half_dt * dxidt

            if step >= burnin:
                obs[obs_idx] = q[0]**2
                obs_idx += 1

            if np.any(np.isnan(q)) or np.any(np.isnan(p)):
                return {'tau_int': float('inf'), 'nan': True, 'seed': seed}

        obs = obs[:obs_idx]
        tau = compute_tau_int(obs)
        return {'tau_int': tau, 'nan': False, 'seed': seed}

    seeds = list(range(20))
    methods = ['tanh', 'log-osc', 'linear', 'sublinear']
    Q = 1.0
    dt = 0.005
    n_steps = 200_000

    for pot_name, pot in [('double_well_1d', DoubleWellPot()), ('gmm4_2d', GMM4Pot())]:
        results[pot_name] = {}
        for method_name in methods:
            g_func = FRICTION_FUNCS[method_name][0]
            job_results = []
            for s in seeds:
                r = run_general_trajectory(
                    pot, g_func, pot.dim, Q, 1.0, dt, n_steps, s
                )
                job_results.append(r)

            taus = [r['tau_int'] for r in job_results if not r['nan']]
            n_nan = sum(1 for r in job_results if r['nan'])
            median_tau = float(np.median(taus)) if taus else float('inf')
            mean_tau = float(np.mean(taus)) if taus else float('inf')
            std_tau = float(np.std(taus)) if taus else float('inf')

            results[pot_name][method_name] = {
                'median_tau': median_tau,
                'mean_tau': mean_tau,
                'std_tau': std_tau,
                'n_nan': n_nan,
            }
            print(f"  {pot_name}, {method_name}, Q={Q}: median_tau={median_tau:.1f}, nan={n_nan}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_functions():
    """Panel (a): g(xi) and G(xi) for all 4 methods."""
    xi = np.linspace(-5, 5, 500)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    for name, (g, G) in FRICTION_FUNCS.items():
        axes[0].plot(xi, g(xi), label=name, color=COLORS[name], linewidth=2)
        axes[1].plot(xi, G(xi), label=name, color=COLORS[name], linewidth=2)

    axes[0].set_title('(a) Friction function g(xi)', fontweight='bold')
    axes[0].set_xlabel('xi')
    axes[0].set_ylabel('g(xi)')
    axes[0].legend(frameon=False)
    axes[0].axhline(0, color='gray', lw=0.5, ls='--')
    axes[0].axhline(1, color='gray', lw=0.5, ls=':', alpha=0.5)
    axes[0].axhline(-1, color='gray', lw=0.5, ls=':', alpha=0.5)

    axes[1].set_title("(b) Primitive G(xi) = integral of g", fontweight='bold')
    axes[1].set_xlabel('xi')
    axes[1].set_ylabel('G(xi)')
    axes[1].legend(frameon=False)

    return fig


def plot_tau_vs_Q(results, kappa_values_to_plot):
    """Panels (b,c): tau_int vs Q at selected kappa values."""
    n_panels = len(kappa_values_to_plot)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    methods = ['tanh', 'log-osc', 'linear', 'sublinear']
    labels = {'tanh': 'tanh (standard NH)', 'log-osc': 'log-osc (2xi/(1+xi^2))',
              'linear': 'linear (g=xi)', 'sublinear': 'sublinear (new)'}

    for idx, kappa in enumerate(kappa_values_to_plot):
        ax = axes[idx]
        for method_name in methods:
            Q_vals = sorted(results[kappa][method_name].keys())
            medians = [results[kappa][method_name][Q]['median_tau'] for Q in Q_vals]
            # Filter out inf for plotting
            valid = [(Q, t) for Q, t in zip(Q_vals, medians) if t < 1e6]
            if valid:
                Qs, taus = zip(*valid)
                ax.plot(Qs, taus, 'o-', label=labels[method_name],
                        color=COLORS[method_name], linewidth=2, markersize=5)

        ax.set_xscale('log')
        ax.set_yscale('log')
        panel_label = chr(ord('b') + idx)
        ax.set_title(f'({panel_label}) kappa={kappa}', fontweight='bold')
        ax.set_xlabel('Thermostat mass Q')
        ax.set_ylabel('Median tau_int (q_d^2)')
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, alpha=0.3)

    return fig


def make_comparison_figure(results):
    """3-panel figure: (a) functions, (b) kappa=100, (c) kappa=1000."""
    fig = plt.figure(figsize=(20, 5.5), constrained_layout=True)

    # Panel (a): g and G functions
    ax_a1 = fig.add_subplot(1, 3, 1)
    xi = np.linspace(-5, 5, 500)
    for name, (g, G) in FRICTION_FUNCS.items():
        ax_a1.plot(xi, g(xi), label=name, color=COLORS[name], linewidth=2)
    ax_a1.set_title('(a) Friction g(xi)', fontweight='bold')
    ax_a1.set_xlabel('xi')
    ax_a1.set_ylabel('g(xi)')
    ax_a1.legend(frameon=False, fontsize=10)
    ax_a1.axhline(0, color='gray', lw=0.5, ls='--')
    ax_a1.axhline(1, color='gray', lw=0.5, ls=':', alpha=0.4)
    ax_a1.axhline(-1, color='gray', lw=0.5, ls=':', alpha=0.4)
    ax_a1.set_ylim(-4, 4)

    methods = ['tanh', 'log-osc', 'linear', 'sublinear']
    labels = {'tanh': 'tanh', 'log-osc': 'log-osc', 'linear': 'linear', 'sublinear': 'sublinear (new)'}

    for panel_idx, kappa in enumerate([100, 1000]):
        ax = fig.add_subplot(1, 3, panel_idx + 2)
        for method_name in methods:
            Q_vals = sorted(results[kappa][method_name].keys())
            medians = [results[kappa][method_name][Q]['median_tau'] for Q in Q_vals]
            valid = [(Q, t) for Q, t in zip(Q_vals, medians) if t < 1e6]
            if valid:
                Qs, taus = zip(*valid)
                ax.plot(Qs, taus, 'o-', label=labels[method_name],
                        color=COLORS[method_name], linewidth=2, markersize=5)

        ax.set_xscale('log')
        ax.set_yscale('log')
        panel_label = chr(ord('b') + panel_idx)
        ax.set_title(f'({panel_label}) kappa={kappa}', fontweight='bold')
        ax.set_xlabel('Thermostat mass Q')
        ax.set_ylabel('Median tau_int (q_d^2)')
        ax.legend(frameon=False, fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(FIG_DIR, 'comparison.png'))
    plt.close(fig)
    print(f"Saved comparison.png")


# ---------------------------------------------------------------------------
# Convexity check for G_new
# ---------------------------------------------------------------------------

def verify_convexity():
    """Check that G''(xi) = g'(xi) > 0 for all xi (monotonicity of g_new)."""
    xi = np.linspace(-10, 10, 10000)
    g_vals = g_new(xi)
    # Numerical derivative
    dg = np.diff(g_vals) / np.diff(xi)
    min_dg = np.min(dg)
    print(f"Convexity check: min g'(xi) = {min_dg:.6f} (should be > 0)")
    return min_dg > 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    print("=" * 60)
    print("Sublinear-g-069: Friction function comparison")
    print("=" * 60)

    # 1. Verify convexity
    print("\n--- Convexity verification ---")
    convex = verify_convexity()
    print(f"G_new is convex: {convex}")

    # 2. Main benchmark
    print("\n--- Main benchmark (d=10 anisotropic Gaussian) ---")
    results = run_benchmark()

    # 3. Additional targets
    print("\n--- Additional targets ---")
    extra_results = run_additional_targets()

    # 4. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Key comparison: tau_tanh / tau_new at kappa=100 and kappa=1000
    for kappa in [10, 100, 1000]:
        print(f"\nkappa = {kappa}:")
        for Q in sorted(results[kappa]['tanh'].keys()):
            tau_tanh = results[kappa]['tanh'][Q]['median_tau']
            tau_new = results[kappa]['sublinear'][Q]['median_tau']
            if tau_new > 0 and tau_new < 1e6:
                ratio = tau_tanh / tau_new
                print(f"  Q={Q:6.1f}: tau_tanh={tau_tanh:8.1f}, tau_new={tau_new:8.1f}, ratio={ratio:.3f}")

    # Headline metric: best ratio at kappa=100 for Q < 100
    best_ratio = 0
    best_Q = None
    for Q in sorted(results[100]['tanh'].keys()):
        if Q >= 100:
            continue
        tau_tanh = results[100]['tanh'][Q]['median_tau']
        tau_new = results[100]['sublinear'][Q]['median_tau']
        if tau_new > 0 and tau_new < 1e6:
            ratio = tau_tanh / tau_new
            if ratio > best_ratio:
                best_ratio = ratio
                best_Q = Q

    print(f"\nHeadline: tau_tanh/tau_new = {best_ratio:.3f} at kappa=100, Q={best_Q}")

    # Similarly at kappa=1000
    best_ratio_1000 = 0
    best_Q_1000 = None
    for Q in sorted(results[1000]['tanh'].keys()):
        if Q >= 100:
            continue
        tau_tanh = results[1000]['tanh'][Q]['median_tau']
        tau_new = results[1000]['sublinear'][Q]['median_tau']
        if tau_new > 0 and tau_new < 1e6:
            ratio = tau_tanh / tau_new
            if ratio > best_ratio_1000:
                best_ratio_1000 = ratio
                best_Q_1000 = Q

    print(f"kappa=1000: tau_tanh/tau_new = {best_ratio_1000:.3f} at Q={best_Q_1000}")

    # 5. Plot
    print("\n--- Generating figures ---")
    make_comparison_figure(results)

    # Also make a full grid plot
    fig_grid = plot_tau_vs_Q(results, [10, 100, 1000])
    fig_grid.savefig(os.path.join(FIG_DIR, 'tau_vs_Q_all_kappa.png'))
    plt.close(fig_grid)
    print("Saved tau_vs_Q_all_kappa.png")

    # Save raw results as JSON
    # Convert float keys to strings for JSON
    json_results = {}
    for k, v in results.items():
        json_results[str(k)] = {}
        for method, qdict in v.items():
            json_results[str(k)][method] = {str(Q): vals for Q, vals in qdict.items()}

    json_results['additional'] = {}
    for pot_name, methods_dict in extra_results.items():
        json_results['additional'][pot_name] = methods_dict

    with open(os.path.join(ORBIT_DIR, 'results.json'), 'w') as f:
        json.dump(json_results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nTotal wall time: {elapsed:.1f}s")

    return results, extra_results, best_ratio, best_ratio_1000


if __name__ == '__main__':
    results, extra_results, ratio_100, ratio_1000 = main()
