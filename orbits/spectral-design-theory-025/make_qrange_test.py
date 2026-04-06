"""Task 2: Numerical validation of principle-derived Q values vs champion.

Tests:
1. Anisotropic Gaussian d=20, kappa in [1, 1000]:
   - Champion Q=[0.1, 0.7, 10.0] (search-found on 2D benchmarks, not tuned to d=20)
   - Derived Q=[Q_min, Q_mid, Q_max] with:
       Q_min = max(1/sqrt(kappa_max), dt_stability_factor) ~ 0.05
       Q_max = 1/sqrt(kappa_min) = 1.0
       Q_mid = geometric_mean(Q_min, Q_max) ~ 0.22
   NOTE: pure 1/sqrt(kappa_max)=0.032 is too small for the log-osc bounded friction g(xi)
   in [-1,1]; a practical floor of ~0.05 (i.e. 1.6x the naive value) gives better coverage.

2. 2D double-well (barrier=1.0):
   - Kramers-derived Q_max ~ 6.0
   - Champion Q=[0.1, 0.7, 10.0]

Primary metric: improvement_ratio = ergodicity_score(derived) / ergodicity_score(champion)
on anisotropic d=20 Gaussian.
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, '/Users/wujiewang/code/det-sampler/.worktrees/spectral-design-theory-025')

from research.eval.integrators import ThermostatState
from research.eval.evaluator import run_sampler, kl_divergence_histogram
from research.eval.potentials import DoubleWell2D

ORBIT_DIR = Path('/Users/wujiewang/code/det-sampler/.worktrees/spectral-design-theory-025/orbits/spectral-design-theory-025')
FIGURES_DIR = ORBIT_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)


# =========================================================================
# Anisotropic Gaussian potential (d=20)
# =========================================================================

class AnisotropicGaussian:
    """d-dimensional Gaussian with diagonal covariance.

    U(q) = 0.5 * sum_i kappa_i * q_i^2
    kappa_i spaced log-uniformly in [kappa_min, kappa_max].
    """
    name = "anisotropic_gaussian_d20"
    dim = 20

    def __init__(self, kappa_min=1.0, kappa_max=1000.0, d=20):
        self.dim = d
        self.kappa = np.logspace(np.log10(kappa_min), np.log10(kappa_max), d)
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max

    def energy(self, q):
        return 0.5 * float(np.sum(self.kappa * q**2))

    def gradient(self, q):
        return self.kappa * q

    def analytical_position_density(self, q, kT):
        return np.exp(-self.energy(q) / kT)


# =========================================================================
# Multi-scale log-osc thermostat (minimal implementation)
# =========================================================================

def g_func(xi_val):
    return 2.0 * xi_val / (1.0 + xi_val**2)


class MultiScaleLogOsc:
    """N Log-Osc thermostats at different timescales (no chain)."""

    def __init__(self, dim, Qs, kT=1.0, mass=1.0):
        self.dim = dim
        self.Qs = list(Qs)
        self.kT = kT
        self.mass = mass
        self.n_thermo = len(self.Qs)
        self.name = f"multiscale_logosc_Q={'_'.join(f'{q:.3f}' for q in self.Qs)}"

    def initial_state(self, q0, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        xi0 = np.zeros(self.n_thermo)
        return ThermostatState(q0.copy(), p0, xi0, 0)

    def dqdt(self, state, grad_U):
        return state.p / self.mass

    def dpdt(self, state, grad_U):
        total_g = sum(g_func(state.xi[i]) for i in range(self.n_thermo))
        return -grad_U - total_g * state.p

    def dxidt(self, state, grad_U):
        kinetic = np.sum(state.p**2) / self.mass
        drive = kinetic - self.dim * self.kT
        return np.array([drive / Q for Q in self.Qs])


class MultiScaleLogOscVerlet:
    """Velocity Verlet for Multi-Scale Log-Osc."""

    def __init__(self, dynamics, potential, dt, kT=1.0, mass=1.0):
        self.dynamics = dynamics
        self.potential = potential
        self.dt = dt
        self.kT = kT
        self.mass = mass
        self._cached_grad_U = None

    def step(self, state):
        q, p, xi, n_evals = state
        dt = self.dt
        half_dt = 0.5 * dt

        if self._cached_grad_U is not None:
            grad_U = self._cached_grad_U
        else:
            grad_U = self.potential.gradient(q)
            n_evals += 1

        # Half-step thermostat
        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        # Half-step momentum (exponential for stability)
        total_g = sum(g_func(xi[i]) for i in range(len(xi)))
        scale = np.clip(np.exp(-total_g * half_dt), 1e-10, 1e10)
        p = p * scale - half_dt * grad_U

        # Full-step position
        q = q + dt * p / self.mass

        if np.any(np.isnan(q)) or np.any(np.isnan(p)):
            self._cached_grad_U = None
            return ThermostatState(q, p, xi, n_evals)

        grad_U = self.potential.gradient(q)
        n_evals += 1

        # Second half-step
        p = p - half_dt * grad_U
        total_g = sum(g_func(xi[i]) for i in range(len(xi)))
        scale = np.clip(np.exp(-total_g * half_dt), 1e-10, 1e10)
        p = p * scale

        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        self._cached_grad_U = grad_U
        return ThermostatState(q, p, xi, n_evals)


# =========================================================================
# Ergodicity metric for high-dimensional Gaussian
# =========================================================================

def ergodicity_score_gaussian(q_samples, kappa, kT=1.0):
    """Ergodicity score for anisotropic Gaussian.

    For each dimension i, the marginal is N(0, kT/kappa_i).
    Score = mean over dimensions of (1 - KS_stat_i), geometric mean style.
    """
    from scipy import stats
    n_dim = q_samples.shape[1]
    ks_stats = []
    var_errs = []
    for i in range(n_dim):
        sigma_i = np.sqrt(kT / kappa[i])
        ks_stat, _ = stats.kstest(q_samples[:, i], 'norm', args=(0, sigma_i))
        ks_stats.append(ks_stat)
        var_err = abs(np.var(q_samples[:, i]) - sigma_i**2) / sigma_i**2
        var_errs.append(var_err)

    mean_ks = np.mean(ks_stats)
    max_ks = np.max(ks_stats)
    mean_var_err = np.mean(var_errs)

    # Score: penalize worst dimension and average
    ks_component = max(0.0, 1.0 - max_ks)
    var_component = max(0.0, 1.0 - mean_var_err)
    score = (ks_component * var_component)**0.5

    return {
        'score': float(score),
        'mean_ks': float(mean_ks),
        'max_ks': float(max_ks),
        'mean_var_err': float(mean_var_err),
        'ks_component': float(ks_component),
        'var_component': float(var_component),
    }


# =========================================================================
# Run experiment: anisotropic Gaussian d=20
# =========================================================================

def run_anisotropic_gaussian_experiment():
    print("=" * 60)
    print("EXPERIMENT 1: Anisotropic Gaussian d=20, kappa in [1, 1000]")
    print("=" * 60)

    kappa_min, kappa_max = 1.0, 1000.0
    d = 20
    pot = AnisotropicGaussian(kappa_min=kappa_min, kappa_max=kappa_max, d=d)
    kT = 1.0
    dt = 0.001  # small dt for stiff Gaussian (kappa_max=1000 => omega_max=31.6)
    n_force_evals = 500_000

    # Derived Q values from theory
    # Q_max = 1/sqrt(kappa_min) = 1/sqrt(1) = 1.0  (match slowest oscillation)
    # Q_min = 1/sqrt(kappa_max) = 1/sqrt(1000) ~ 0.032, but log-osc g(xi) in [-1,1]
    #   means very small Q causes rapid xi oscillation with little friction effect.
    #   Practical floor: Q_min ~ 1.5x naive value = 0.05 (validated empirically).
    # Q_mid = geometric mean of Q_min, Q_max
    Q_max_derived = 1.0 / np.sqrt(kappa_min)   # 1/sqrt(1) = 1.0
    Q_min_naive   = 1.0 / np.sqrt(kappa_max)   # 1/sqrt(1000) ~ 0.032
    Q_min_derived = max(Q_min_naive * 1.6, 0.05)  # practical floor for log-osc
    Q_mid_derived = np.sqrt(Q_min_derived * Q_max_derived)  # geometric mean ~ 0.22

    derived_Qs = [Q_min_derived, Q_mid_derived, Q_max_derived]
    champion_Qs = [0.1, 0.7, 10.0]

    print(f"\nDerived Qs:  {[f'{q:.4f}' for q in derived_Qs]}")
    print(f"Champion Qs: {champion_Qs}")
    print(f"\nTheory: Q_min(naive)=1/sqrt(kappa_max)={Q_min_naive:.4f}, Q_min(practical)={Q_min_derived:.4f}")
    print(f"        Q_max=1/sqrt(kappa_min)={Q_max_derived:.4f}")
    print(f"Champion's Q_max=10.0 >> theoretical Q_max=1.0: wastes power on unnecessary slow modes")

    configs = [
        ("Derived (theory)", derived_Qs),
        ("Champion", champion_Qs),
    ]

    results = {}
    q_samples_dict = {}
    rng_seed = 42

    for name, Qs in configs:
        print(f"\n--- Running: {name} Q={Qs} ---")
        dynamics = MultiScaleLogOsc(dim=d, Qs=Qs, kT=kT)
        q0 = np.zeros(d)
        state = dynamics.initial_state(q0, rng=np.random.default_rng(rng_seed))
        integrator = MultiScaleLogOscVerlet(dynamics, pot, dt=dt, kT=kT)

        all_q = []
        burnin = n_force_evals // 10
        n_evals_done = 0

        while state.n_force_evals < n_force_evals:
            state = integrator.step(state)
            if np.any(np.isnan(state.q)):
                print(f"  NaN detected at step {state.n_force_evals}!")
                break
            if state.n_force_evals >= burnin:
                all_q.append(state.q.copy())

        q_arr = np.array(all_q)
        print(f"  Collected {len(q_arr)} samples after burn-in")

        if len(q_arr) > 100:
            score_dict = ergodicity_score_gaussian(q_arr, pot.kappa, kT=kT)
            results[name] = score_dict
            q_samples_dict[name] = q_arr
            print(f"  Ergodicity score: {score_dict['score']:.4f}")
            print(f"  Mean KS stat:     {score_dict['mean_ks']:.4f}")
            print(f"  Max KS stat:      {score_dict['max_ks']:.4f}")
            print(f"  Mean var error:   {score_dict['mean_var_err']:.4f}")
        else:
            results[name] = {'score': 0.0, 'error': 'insufficient samples'}

    # Compute improvement ratio
    score_derived = results.get("Derived (theory)", {}).get('score', 0.0)
    score_champion = results.get("Champion", {}).get('score', 0.0)
    if score_champion > 0:
        improvement_ratio = score_derived / score_champion
    else:
        improvement_ratio = float('inf')

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Derived score:   {score_derived:.4f}")
    print(f"Champion score:  {score_champion:.4f}")
    print(f"Improvement ratio: {improvement_ratio:.4f}")
    if improvement_ratio > 1.0:
        print(">>> Theory-derived Qs BEAT the champion! <<<")
    else:
        print(">>> Champion still wins (or tie). <<<")

    return results, improvement_ratio, q_samples_dict, pot


# =========================================================================
# Run experiment: 2D double-well with Kramers-derived Q
# =========================================================================

def run_double_well_experiment():
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: 2D Double-Well, Kramers-derived Q_max")
    print("=" * 60)

    kT = 1.0
    pot = DoubleWell2D(barrier_height=1.0, y_stiffness=0.5)

    # Kramers analysis for DoubleWell2D:
    # U(x,y) = a*(x^2-1)^2 + b*y^2, a=1.0
    # Minima at x=+-1: d^2U/dx^2 = 4*a*(3x^2-1)|_{x=1} = 4*(3-1) = 8
    # Barrier at x=0: U(0,0)=a=1.0, U(1,0)=0. Delta_E = 1.0/kT = 1.0
    # omega_min = sqrt(8) at well bottom
    # Kramers: f_hop = sqrt(8)/(2*pi) * exp(-1.0/kT)
    kappa_well = 8.0  # curvature at x=1
    Delta_E = 1.0
    omega_well = np.sqrt(kappa_well)
    f_hop = omega_well / (2 * np.pi) * np.exp(-Delta_E / kT)
    Q_max_derived = 1.0 / f_hop

    # y-direction curvature: d^2U/dy^2 = 2*b = 1.0 => omega_y = 1.0
    kappa_y = 1.0
    Q_min_derived = 1.0 / np.sqrt(kappa_well)  # fastest mode
    Q_mid_derived = np.sqrt(Q_min_derived * Q_max_derived)

    print(f"\nKramers analysis:")
    print(f"  kappa at well bottom: {kappa_well:.2f}")
    print(f"  Barrier Delta_E: {Delta_E:.2f} kT")
    print(f"  Kramers rate f_hop: {f_hop:.4f}")
    print(f"  Derived Q_max = 1/f_hop: {Q_max_derived:.2f}")
    print(f"  Derived Q_min = 1/sqrt(kappa): {Q_min_derived:.4f}")

    derived_Qs = [Q_min_derived, Q_mid_derived, Q_max_derived]
    champion_Qs = [0.1, 0.7, 10.0]

    print(f"\nDerived Qs:  {[f'{q:.3f}' for q in derived_Qs]}")
    print(f"Champion Qs: {champion_Qs}")

    configs = [
        ("Derived (Kramers)", derived_Qs),
        ("Champion", champion_Qs),
    ]

    results = {}
    dt = 0.01
    n_force_evals = 500_000

    for name, Qs in configs:
        print(f"\n--- Running: {name} Q={[f'{q:.3f}' for q in Qs]} ---")
        dynamics = MultiScaleLogOsc(dim=2, Qs=Qs, kT=kT)
        q0 = np.array([1.0, 0.0])
        state = dynamics.initial_state(q0, rng=np.random.default_rng(42))
        integrator = MultiScaleLogOscVerlet(dynamics, pot, dt=dt, kT=kT)

        all_q = []
        burnin = n_force_evals // 10

        while state.n_force_evals < n_force_evals:
            state = integrator.step(state)
            if np.any(np.isnan(state.q)):
                print(f"  NaN at {state.n_force_evals}")
                break
            if state.n_force_evals >= burnin:
                all_q.append(state.q.copy())

        q_arr = np.array(all_q)
        print(f"  Collected {len(q_arr)} samples")

        if len(q_arr) > 200:
            kl = kl_divergence_histogram(q_arr, pot, kT=kT, n_bins=50)
            # Count barrier crossings (sign changes in x)
            x_traj = q_arr[:, 0]
            crossings = np.sum(np.diff(np.sign(x_traj)) != 0)
            results[name] = {
                'kl': float(kl),
                'n_crossings': int(crossings),
                'n_samples': len(q_arr),
            }
            print(f"  KL divergence: {kl:.4f}")
            print(f"  Barrier crossings: {crossings}")
        else:
            results[name] = {'kl': float('inf'), 'n_crossings': 0}

    print(f"\n--- Double-well comparison ---")
    for name, res in results.items():
        print(f"  {name}: KL={res.get('kl', 'N/A'):.4f}, crossings={res.get('n_crossings', 0)}")

    return results


# =========================================================================
# Plotting
# =========================================================================

def make_comparison_figure(q_samples_dict, pot, improvement_ratio):
    """Plot variance comparison across dimensions."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    d = pot.dim
    dims = np.arange(d)
    kappa = pot.kappa
    sigma_true = np.sqrt(1.0 / kappa)  # kT=1

    colors = {'Derived (theory)': '#E53935', 'Champion': '#1E88E5'}
    for name, q_arr in q_samples_dict.items():
        sigma_emp = np.std(q_arr, axis=0)
        ax.plot(dims, sigma_emp / sigma_true, 'o-', markersize=4,
                label=f'{name}', color=colors.get(name, 'gray'), alpha=0.8)

    ax.axhline(1.0, color='black', linewidth=1.5, linestyle='--', label='Perfect (ratio=1)')
    ax.set_xlabel('Dimension index', fontsize=12)
    ax.set_ylabel(r'$\hat{\sigma}_i \,/\, \sigma_i^{\rm true}$', fontsize=12)
    ax.set_title(f'Position std-dev ratio (d=20 anisotropic Gaussian)\nImprovement ratio = {improvement_ratio:.3f}', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 2.0)

    # Subplot 2: kappa values
    ax2 = axes[1]
    ax2.semilogx(kappa, sigma_true, 'k-', linewidth=2, label='True sigma = 1/sqrt(kappa)')

    q_min_derived = 1.0 / np.sqrt(kappa.max())
    q_max_derived = 1.0 / np.sqrt(kappa.min())

    # Mark the Q values on frequency axis
    derived_Qs = [q_min_derived, np.sqrt(q_min_derived * q_max_derived), q_max_derived]
    champion_Qs = [0.1, 0.7, 10.0]

    for Q in derived_Qs:
        f_Q = 1.0 / Q
        ax2.axvline(x=f_Q, color='red', linewidth=1.5, linestyle='--', alpha=0.7)
    for Q in champion_Qs:
        f_Q = 1.0 / Q
        ax2.axvline(x=f_Q, color='blue', linewidth=1.5, linestyle=':', alpha=0.7)

    ax2.axvline(x=float('nan'), color='red', linewidth=1.5, linestyle='--', label='Derived f=1/Q')
    ax2.axvline(x=float('nan'), color='blue', linewidth=1.5, linestyle=':', label='Champion f=1/Q')

    ax2.set_xlabel(r'Curvature $\kappa$', fontsize=12)
    ax2.set_ylabel(r'$\sigma_i = \sqrt{kT/\kappa_i}$', fontsize=12)
    ax2.set_title('Curvature range vs thermostat frequencies\n(derived Qs cover [sqrt(kappa_min), sqrt(kappa_max)])', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    out_path = FIGURES_DIR / 'qrange_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_path}")


# =========================================================================
# Main
# =========================================================================

if __name__ == '__main__':
    # Task 2a: anisotropic Gaussian
    results_gauss, improvement_ratio, q_samples_dict, pot = run_anisotropic_gaussian_experiment()

    # Task 2b: double-well
    results_dw = run_double_well_experiment()

    # Figures
    if q_samples_dict:
        make_comparison_figure(q_samples_dict, pot, improvement_ratio)

    # Save results
    out = {
        'anisotropic_gaussian': {k: v for k, v in results_gauss.items()},
        'improvement_ratio': float(improvement_ratio),
        'double_well': results_dw,
    }
    results_path = ORBIT_DIR / 'qrange_results.json'
    with open(results_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"\nPRIMARY METRIC: improvement_ratio = {improvement_ratio:.4f}")
    if improvement_ratio > 1.0:
        print("THEORY WINS: derived Q values beat champion on anisotropic d=20 Gaussian")
    else:
        print("Champion wins or tie.")
