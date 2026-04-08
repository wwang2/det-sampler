"""paper-final-049: Three final experiments for the deterministic thermostat paper.

E1 -- HMC Head-to-Head: NH-1 (identity & tanh) vs HMC vs Langevin
E2 -- KL Convergence Race: convergence curves on 4 targets
E3 -- Ergodicity Phase Transition: log-osc resonance ceiling vs omega

Reuses infrastructure from paper-experiments-047.
"""

import json, os, sys, time, warnings
import numpy as np
from multiprocessing import Pool

warnings.filterwarnings("ignore")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================================
# Potentials (from parent orbit + research/eval)
# ============================================================================

class AnisotropicGaussian:
    """N-dimensional anisotropic Gaussian with condition number kappa."""
    def __init__(self, dim, kappa=100.0):
        self.dim = dim
        self.kappas = np.array([kappa ** (i / max(dim - 1, 1)) for i in range(dim)])
    def energy(self, q):
        return 0.5 * float(np.sum(self.kappas * q * q))
    def gradient(self, q):
        return self.kappas * q


class HarmonicOscillator1D:
    dim = 1
    def __init__(self, omega=1.0):
        self.omega = omega
        self.kappas = np.array([omega**2])
    def energy(self, q):
        return 0.5 * self.omega**2 * float(q[0]**2)
    def gradient(self, q):
        return np.array([self.omega**2 * q[0]])


class DoubleWell2D:
    dim = 2
    def __init__(self, barrier_height=1.0, y_stiffness=0.5):
        self.a = barrier_height; self.b = y_stiffness
    def energy(self, q):
        x, y = q[0], q[1]
        return self.a * (x**2 - 1)**2 + self.b * y**2
    def gradient(self, q):
        x, y = q[0], q[1]
        return np.array([4.0*self.a*x*(x**2-1), 2.0*self.b*y])


class GaussianMixture2D:
    dim = 2
    def __init__(self, n_modes=5, radius=3.0, sigma=0.5):
        angles = np.linspace(0, 2*np.pi, n_modes, endpoint=False)
        self.centers = np.column_stack([radius*np.cos(angles), radius*np.sin(angles)])
        self.sigma = sigma; self.n_modes = n_modes
        self.weights = np.ones(n_modes) / n_modes
    def _component_densities(self, q):
        diffs = self.centers - q[np.newaxis, :]
        exponents = -0.5 * np.sum(diffs**2, axis=1) / self.sigma**2
        return self.weights * np.exp(exponents)
    def energy(self, q):
        densities = self._component_densities(q)
        total = np.sum(densities)
        if total < 1e-300: return 700.0
        return -np.log(total)
    def gradient(self, q):
        densities = self._component_densities(q)
        total = np.sum(densities)
        if total < 1e-300: return np.zeros(2)
        diffs = self.centers - q[np.newaxis, :]
        weighted = densities[:, np.newaxis] * diffs / self.sigma**2
        return -np.sum(weighted, axis=0) / total


class GaussianMixtureND:
    """N-dimensional Gaussian mixture with modes on random unit vectors."""
    def __init__(self, dim, n_modes=5, radius=3.0, sigma=1.0, seed=0):
        rng = np.random.default_rng(seed)
        dirs = rng.standard_normal((n_modes, dim))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        self.centers = dirs * radius
        self.sigma = sigma; self.dim = dim; self.n_modes = n_modes
    def energy(self, q):
        diffs = self.centers - q
        exps = -0.5 * np.sum(diffs**2, axis=1) / self.sigma**2
        mx = np.max(exps)
        return -(mx + np.log(np.sum(np.exp(exps - mx))))
    def gradient(self, q):
        diffs = self.centers - q
        exps = -0.5 * np.sum(diffs**2, axis=1) / self.sigma**2
        mx = np.max(exps)
        w = np.exp(exps - mx); w /= np.sum(w)
        return -np.sum(w[:, None] * diffs, axis=0) / self.sigma**2


# ============================================================================
# Friction functions
# ============================================================================

def g_tanh(xi):  return np.tanh(xi)
def g_identity(xi): return xi
def g_logosc(xi): return 2.0 * xi / (1.0 + xi**2)


# ============================================================================
# Integrators
# ============================================================================

def sim_nh(pot, Q, dt, nsteps, kT=1.0, seed=0, rec=1, g_func=None, save_all=False):
    """Single Nose-Hoover integrator with optional nonlinear friction g.
    If g_func is None, uses identity (standard NH).
    If save_all=True, returns positions at every step (for KL checkpointing).
    """
    rng = np.random.default_rng(seed)
    dim = pot.dim
    q = rng.normal(0, 1.0, size=dim)
    if hasattr(pot, 'kappas'):
        q /= np.sqrt(np.maximum(pot.kappas, 1e-6))
    p = rng.normal(0, np.sqrt(kT), size=dim)
    xi = 0.0; h = 0.5 * dt; gU = pot.gradient(q)
    if save_all:
        nr = nsteps; qs = np.empty((nr, dim)); rec = 1
    else:
        nr = nsteps // rec; qs = np.empty((nr, dim))
    ri = 0
    for s in range(nsteps):
        K = float(np.sum(p * p))
        xi += h * (K - dim * kT) / Q
        gval = g_func(xi) if g_func is not None else xi
        p *= np.clip(np.exp(-gval * h), 1e-10, 1e10)
        p -= h * gU; q = q + dt * p; gU = pot.gradient(q)
        p -= h * gU
        gval = g_func(xi) if g_func is not None else xi
        p *= np.clip(np.exp(-gval * h), 1e-10, 1e10)
        K = float(np.sum(p * p))
        xi += h * (K - dim * kT) / Q
        if save_all:
            if ri < nr: qs[ri] = q; ri += 1
        elif (s + 1) % rec == 0 and ri < nr:
            qs[ri] = q; ri += 1
        if not np.isfinite(p).all(): qs[ri:] = np.nan; break
    return qs[:ri]


def sim_nhc(pot, Qs, dt, nsteps, kT=1.0, seed=0, rec=1, save_all=False):
    """Nose-Hoover Chain integrator."""
    rng = np.random.default_rng(seed)
    dim = pot.dim; Qs = np.asarray(Qs, float); M = len(Qs)
    q = rng.normal(0, 1.0, size=dim)
    if hasattr(pot, 'kappas'):
        q /= np.sqrt(np.maximum(pot.kappas, 1e-6))
    p = rng.normal(0, np.sqrt(kT), size=dim)
    xi = np.zeros(M); h = 0.5 * dt; gU = pot.gradient(q)
    if save_all:
        nr = nsteps; qs = np.empty((nr, dim)); rec = 1
    else:
        nr = nsteps // rec; qs = np.empty((nr, dim))
    ri = 0
    def dxi(pv, xv):
        d = np.zeros(M); K = float(np.sum(pv * pv))
        d[0] = (K - dim * kT) / Qs[0]
        if M > 1: d[0] -= xv[1] * xv[0]
        for i in range(1, M):
            d[i] = (Qs[i-1] * xv[i-1]**2 - kT) / Qs[i]
            if i < M-1: d[i] -= xv[i+1] * xv[i]
        return d
    for s in range(nsteps):
        xi += h * dxi(p, xi)
        p *= np.clip(np.exp(-xi[0] * h), 1e-10, 1e10)
        p -= h * gU; q = q + dt * p; gU = pot.gradient(q)
        p -= h * gU
        p *= np.clip(np.exp(-xi[0] * h), 1e-10, 1e10)
        xi += h * dxi(p, xi)
        if save_all:
            if ri < nr: qs[ri] = q; ri += 1
        elif (s + 1) % rec == 0 and ri < nr:
            qs[ri] = q; ri += 1
        if not np.isfinite(p).all(): qs[ri:] = np.nan; break
    return qs[:ri]


def sim_langevin(pot, gamma, dt, nsteps, kT=1.0, seed=0, rec=1, save_all=False):
    """BAOAB Langevin integrator."""
    rng = np.random.default_rng(seed)
    dim = pot.dim
    q = rng.normal(0, 1.0, size=dim)
    if hasattr(pot, 'kappas'):
        q /= np.sqrt(np.maximum(pot.kappas, 1e-6))
    p = rng.normal(0, np.sqrt(kT), size=dim)
    h = 0.5 * dt; gU = pot.gradient(q)
    if save_all:
        nr = nsteps; qs = np.empty((nr, dim)); rec = 1
    else:
        nr = nsteps // rec; qs = np.empty((nr, dim))
    ri = 0
    c1 = np.exp(-gamma * dt); c2 = np.sqrt(kT * (1 - c1**2))
    for s in range(nsteps):
        p -= h * gU; q += h * p
        p = c1 * p + c2 * rng.standard_normal(dim)
        q += h * p; gU = pot.gradient(q)
        p -= h * gU
        if save_all:
            if ri < nr: qs[ri] = q; ri += 1
        elif (s + 1) % rec == 0 and ri < nr:
            qs[ri] = q; ri += 1
        if not np.isfinite(p).all(): qs[ri:] = np.nan; break
    return qs[:ri]


def sim_hmc(pot, mass_diag, dt_lf, L, n_samples, kT=1.0, seed=0, save_all=False):
    """HMC sampler. Returns (samples, n_force_evals, accept_rate).
    mass_diag: diagonal of mass matrix (array of length dim).
    dt_lf: leapfrog step size.
    L: leapfrog steps per proposal.
    """
    rng = np.random.default_rng(seed)
    dim = pot.dim
    M = np.asarray(mass_diag, float)
    q = rng.normal(0, 1.0, size=dim)
    if hasattr(pot, 'kappas'):
        q /= np.sqrt(np.maximum(pot.kappas, 1e-6))
    samples = np.empty((n_samples, dim))
    n_force = 0; n_accept = 0
    for i in range(n_samples):
        p = rng.normal(0, 1, dim) * np.sqrt(M * kT)
        H_old = pot.energy(q) + 0.5 * np.sum(p**2 / M)
        q_new, p_new = q.copy(), p.copy()
        grad = pot.gradient(q_new); n_force += 1
        p_new -= 0.5 * dt_lf * grad
        for step in range(L - 1):
            q_new += dt_lf * p_new / M
            grad = pot.gradient(q_new); n_force += 1
            p_new -= dt_lf * grad
        q_new += dt_lf * p_new / M
        grad = pot.gradient(q_new); n_force += 1
        p_new -= 0.5 * dt_lf * grad
        H_new = pot.energy(q_new) + 0.5 * np.sum(p_new**2 / M)
        if np.isfinite(H_new) and np.log(rng.uniform() + 1e-300) < -(H_new - H_old) / kT:
            q = q_new; n_accept += 1
        samples[i] = q.copy()
    return samples, n_force, n_accept / n_samples


# ============================================================================
# Metrics
# ============================================================================

def acf_tau(x, c=5.0):
    x = np.asarray(x, float) - np.mean(x); n = len(x)
    if n < 16 or np.std(x) < 1e-12: return float(n)
    f = np.fft.fft(x, n=2*n)
    a = np.fft.ifft(f * np.conj(f))[:n].real; a /= a[0]
    tau = 1.0
    for k in range(1, n//4):
        tau += 2*a[k]
        if k >= c*tau: break
    return max(tau, 1.0)


def ess_per_fe(traj, n_force_evals):
    """Compute ESS/force-eval from trajectory."""
    v = traj[~np.isnan(traj[:, 0])]
    if len(v) < 64: return 0.0
    taus = [acf_tau(v[:, d]**2) for d in range(v.shape[1])]
    tau_mean = float(np.mean(taus))
    n_eff = len(v) / tau_mean
    return n_eff / n_force_evals


def kl_divergence_1d(samples_q, potential, kT, bins=80):
    """KL divergence for 1D potential with known Boltzmann distribution."""
    q = samples_q.flatten()
    q = q[np.isfinite(q)]
    if len(q) < 100: return 10.0
    lo, hi = np.percentile(q, [0.5, 99.5])
    lo, hi = min(lo, -4.0), max(hi, 4.0)
    hist, edges = np.histogram(q, bins=bins, range=(lo, hi), density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    dx = edges[1] - edges[0]
    p_true = np.exp(-np.array([potential.energy(np.array([c])) for c in centers]) / kT)
    p_true /= np.sum(p_true) * dx
    mask = (hist > 1e-10) & (p_true > 1e-10)
    if np.sum(mask) < 5: return 10.0
    kl = np.sum(p_true[mask] * np.log(p_true[mask] / hist[mask])) * dx
    return max(float(kl), 0.0)


def kl_divergence_2d(samples, potential, kT, bins=40):
    """KL divergence for 2D potential with known Boltzmann distribution."""
    v = samples[~np.isnan(samples[:, 0])]
    if len(v) < 200: return 10.0
    ranges = [[np.percentile(v[:, d], [1, 99]) for d in range(2)]]
    r = ranges[0]
    r[0] = [min(r[0][0], -4), max(r[0][1], 4)]
    r[1] = [min(r[1][0], -4), max(r[1][1], 4)]
    hist, xe, ye = np.histogram2d(v[:, 0], v[:, 1], bins=bins, range=r, density=True)
    cx = 0.5*(xe[1:]+xe[:-1]); cy = 0.5*(ye[1:]+ye[:-1])
    dx = xe[1]-xe[0]; dy = ye[1]-ye[0]
    xx, yy = np.meshgrid(cx, cy, indexing='ij')
    p_true = np.zeros_like(hist)
    for i in range(bins):
        for j in range(bins):
            p_true[i, j] = np.exp(-potential.energy(np.array([xx[i,j], yy[i,j]])) / kT)
    p_true /= np.sum(p_true) * dx * dy
    mask = (hist > 1e-10) & (p_true > 1e-10)
    if np.sum(mask) < 10: return 10.0
    kl = np.sum(p_true[mask] * np.log(p_true[mask] / hist[mask])) * dx * dy
    return max(float(kl), 0.0)


def kl_variance_proxy(samples, potential, kT):
    """For high-D Gaussians: mean squared relative variance error."""
    v = samples[~np.isnan(samples[:, 0])]
    if len(v) < 100: return 10.0
    if hasattr(potential, 'kappas'):
        true_var = kT / potential.kappas
        emp_var = np.var(v, axis=0)
        return float(np.mean((emp_var - true_var)**2 / true_var**2))
    return float('nan')


# ============================================================================
# E1: HMC Head-to-Head
# ============================================================================

def tune_hmc_dt(pot, mass_diag, L, kT=1.0, seed=42):
    """Quick dt tuning: find dt giving ~65-80% acceptance."""
    dts = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    results = []
    for dt in dts:
        _, _, acc = sim_hmc(pot, mass_diag, dt, L, n_samples=150, kT=kT, seed=seed)
        results.append((dt, acc))
    # Find dt closest to 70% acceptance
    best_dt = 0.01; best_dist = 10.0
    for dt, acc in results:
        if acc < 0.01: continue
        dist = abs(acc - 0.70)
        if dist < best_dist:
            best_dist = dist; best_dt = dt
    best_acc = dict(results).get(best_dt, 0.0)
    return best_dt, best_acc


# Cache for HMC dt tuning (tune once per target, not per seed)
_hmc_dt_cache = {}


def _run_e1_single(args):
    """Worker for a single (method, target, seed) combination."""
    method, target_name, seed, nfe_budget = args
    kT = 1.0; dt = 0.01

    # Create potential
    if target_name == 'aniso5':
        pot = AnisotropicGaussian(dim=5, kappa=100.0)
    elif target_name == 'aniso10':
        pot = AnisotropicGaussian(dim=10, kappa=100.0)
    elif target_name == 'dw2d':
        pot = DoubleWell2D()
    elif target_name == 'gmm2d':
        pot = GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5)
    else:
        raise ValueError(target_name)

    dim = pot.dim
    L = 20  # leapfrog steps for HMC

    if method == 'NH_identity':
        # Standard NH: Q = dim*kT / omega_eff^2
        if hasattr(pot, 'kappas'):
            omega_eff = np.sqrt(np.mean(pot.kappas))
            Q = dim * kT / omega_eff**2
        else:
            Q = float(dim)
        tr = sim_nh(pot, Q, dt, nfe_budget, kT=kT, seed=seed)
        n_fe = nfe_budget  # 1 force eval per step
        return {'ess_per_fe': ess_per_fe(tr, n_fe), 'n_fe': n_fe, 'accept': 1.0}

    elif method == 'NH_tanh':
        Q = 100.0
        tr = sim_nh(pot, Q, dt, nfe_budget, kT=kT, seed=seed, g_func=g_tanh)
        n_fe = nfe_budget
        return {'ess_per_fe': ess_per_fe(tr, n_fe), 'n_fe': n_fe, 'accept': 1.0}

    elif method == 'NHC_M3':
        if hasattr(pot, 'kappas'):
            Q_val = max(10, dim)
        else:
            Q_val = 10.0
        tr = sim_nhc(pot, np.ones(3)*Q_val, dt, nfe_budget, kT=kT, seed=seed)
        n_fe = nfe_budget
        return {'ess_per_fe': ess_per_fe(tr, n_fe), 'n_fe': n_fe, 'accept': 1.0}

    elif method == 'HMC_tuned':
        if hasattr(pot, 'kappas'):
            mass_diag = kT / pot.kappas  # Matched preconditioning
        else:
            mass_diag = np.ones(dim)
        cache_key = (target_name, 'tuned')
        if cache_key not in _hmc_dt_cache:
            _hmc_dt_cache[cache_key] = tune_hmc_dt(pot, mass_diag, L, kT=kT, seed=42)
        dt_hmc, _ = _hmc_dt_cache[cache_key]
        n_samples = nfe_budget // (L + 1)  # L+1 force evals per sample
        tr, n_fe, acc = sim_hmc(pot, mass_diag, dt_hmc, L, n_samples, kT=kT, seed=seed)
        return {'ess_per_fe': ess_per_fe(tr, n_fe), 'n_fe': n_fe, 'accept': acc, 'dt_hmc': dt_hmc}

    elif method == 'HMC_untuned':
        mass_diag = np.ones(dim)  # No preconditioning
        cache_key = (target_name, 'untuned')
        if cache_key not in _hmc_dt_cache:
            _hmc_dt_cache[cache_key] = tune_hmc_dt(pot, mass_diag, L, kT=kT, seed=42)
        dt_hmc, _ = _hmc_dt_cache[cache_key]
        n_samples = nfe_budget // (L + 1)
        tr, n_fe, acc = sim_hmc(pot, mass_diag, dt_hmc, L, n_samples, kT=kT, seed=seed)
        return {'ess_per_fe': ess_per_fe(tr, n_fe), 'n_fe': n_fe, 'accept': acc, 'dt_hmc': dt_hmc}

    elif method == 'Langevin':
        tr = sim_langevin(pot, 1.0, dt, nfe_budget, kT=kT, seed=seed)
        n_fe = nfe_budget
        return {'ess_per_fe': ess_per_fe(tr, n_fe), 'n_fe': n_fe, 'accept': 1.0}


def run_E1():
    print("=" * 60)
    print("E1: HMC Head-to-Head Comparison")
    print("=" * 60)

    targets = ['aniso5', 'aniso10', 'dw2d', 'gmm2d']
    methods = ['NH_identity', 'NH_tanh', 'NHC_M3', 'HMC_tuned', 'HMC_untuned', 'Langevin']
    n_seeds = 5
    nfe_budget = 200_000

    results = {}
    for tgt in targets:
        results[tgt] = {}
        for meth in methods:
            t0 = time.time()
            args_list = [(meth, tgt, 1000 + s, nfe_budget) for s in range(n_seeds)]
            seed_results = [_run_e1_single(a) for a in args_list]

            ess_vals = [r['ess_per_fe'] for r in seed_results]
            acc_vals = [r['accept'] for r in seed_results]
            results[tgt][meth] = {
                'ess_per_fe': ess_vals,
                'median_ess': float(np.median(ess_vals)),
                'iqr_ess': [float(np.percentile(ess_vals, 25)), float(np.percentile(ess_vals, 75))],
                'mean_accept': float(np.mean(acc_vals)),
            }
            if 'dt_hmc' in seed_results[0]:
                results[tgt][meth]['dt_hmc'] = seed_results[0]['dt_hmc']

            elapsed = time.time() - t0
            print(f"  {tgt:8s} {meth:15s}: ESS/FE={np.median(ess_vals):.5f}  "
                  f"acc={np.mean(acc_vals):.2f}  ({elapsed:.1f}s)", flush=True)

    # Save results
    with open(os.path.join(OUT_DIR, "results_hmc.json"), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def plot_E1(results):
    """Bar chart: ESS/force-eval for each method on each target."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    targets = list(results.keys())
    methods = list(results[targets[0]].keys())
    target_labels = {'aniso5': '5D Aniso\n($\\kappa$=100)', 'aniso10': '10D Aniso\n($\\kappa$=100)',
                     'dw2d': '2D Double\nWell', 'gmm2d': '2D GMM\n(5 modes)'}
    method_colors = {
        'NH_identity': '#1f77b4', 'NH_tanh': '#2ca02c', 'NHC_M3': '#ff7f0e',
        'HMC_tuned': '#d62728', 'HMC_untuned': '#9467bd', 'Langevin': '#7f7f7f'
    }
    method_labels = {
        'NH_identity': 'NH (g=id)', 'NH_tanh': 'NH (g=tanh)', 'NHC_M3': 'NHC (M=3)',
        'HMC_tuned': 'HMC (tuned)', 'HMC_untuned': 'HMC (untuned)', 'Langevin': 'Langevin'
    }

    fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=300)
    n_tgt = len(targets); n_meth = len(methods)
    bar_width = 0.12
    x = np.arange(n_tgt)

    for i, meth in enumerate(methods):
        medians = [results[t][meth]['median_ess'] for t in targets]
        lo = [results[t][meth]['iqr_ess'][0] for t in targets]
        hi = [results[t][meth]['iqr_ess'][1] for t in targets]
        yerr_lo = [max(m - l, 0) for m, l in zip(medians, lo)]
        yerr_hi = [max(h - m, 0) for m, h in zip(medians, hi)]
        offset = (i - n_meth/2 + 0.5) * bar_width
        ax.bar(x + offset, medians, bar_width * 0.9,
               yerr=[yerr_lo, yerr_hi], capsize=2,
               color=method_colors[meth], label=method_labels[meth], alpha=0.85,
               edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([target_labels.get(t, t) for t in targets], fontsize=11)
    ax.set_ylabel('ESS / force evaluation', fontsize=14)
    ax.set_title('Sampling Efficiency: Deterministic Thermostats vs HMC', fontsize=15, fontweight='bold')
    ax.legend(fontsize=9, ncol=3, loc='upper right', framealpha=0.9)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig1_hmc_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ============================================================================
# E2: KL Convergence Race
# ============================================================================

def _kl_checkpoints_method(method, target_name, seed, nfe_total, n_checkpoints=20):
    """Run a method and compute KL at checkpoints."""
    kT = 1.0; dt = 0.01

    if target_name == 'ho1d':
        pot = HarmonicOscillator1D(omega=1.0)
    elif target_name == 'dw2d':
        pot = DoubleWell2D()
    elif target_name == 'gmm2d':
        pot = GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5)
    elif target_name == 'aniso5':
        pot = AnisotropicGaussian(dim=5, kappa=100.0)
    else:
        raise ValueError(target_name)

    dim = pot.dim
    L = 20
    checkpoint_fes = np.linspace(nfe_total // n_checkpoints, nfe_total, n_checkpoints).astype(int)

    # Record every 2 steps to save memory (still dense enough for KL)
    rec = 2
    if method == 'NH_tuned':
        if hasattr(pot, 'kappas'):
            Q = dim * kT / np.mean(pot.kappas)
        else:
            Q = float(dim)
        tr = sim_nh(pot, Q, dt, nfe_total, kT=kT, seed=seed, rec=rec)
    elif method == 'NHC_M3':
        Q_val = 10.0
        tr = sim_nhc(pot, np.ones(3)*Q_val, dt, nfe_total, kT=kT, seed=seed, rec=rec)
    elif method == 'NH_tanh':
        tr = sim_nh(pot, 100.0, dt, nfe_total, kT=kT, seed=seed, g_func=g_tanh, rec=rec)
    elif method == 'HMC':
        if hasattr(pot, 'kappas'):
            mass_diag = kT / pot.kappas
        else:
            mass_diag = np.ones(dim)
        cache_key = (target_name, 'hmc_kl')
        if cache_key not in _hmc_dt_cache:
            _hmc_dt_cache[cache_key] = tune_hmc_dt(pot, mass_diag, L, kT=kT, seed=42)
        dt_hmc, _ = _hmc_dt_cache[cache_key]
        n_samples = nfe_total // (L + 1)
        tr, _, _ = sim_hmc(pot, mass_diag, dt_hmc, L, n_samples, kT=kT, seed=seed)
        rec = 1  # HMC already records every sample
    elif method == 'Langevin':
        tr = sim_langevin(pot, 1.0, dt, nfe_total, kT=kT, seed=seed, rec=rec)
    else:
        raise ValueError(method)

    kl_values = []
    for cp_fe in checkpoint_fes:
        # Convert force-eval count to sample index
        if method == 'HMC':
            idx = min(cp_fe // (L + 1), len(tr))
        else:
            idx = min(cp_fe // rec, len(tr))
        sub = tr[:idx]
        if len(sub) < 50:
            kl_values.append(10.0)
            continue

        if target_name == 'ho1d':
            kl = kl_divergence_1d(sub, pot, kT)
        elif target_name in ('dw2d', 'gmm2d'):
            kl = kl_divergence_2d(sub, pot, kT)
        elif target_name == 'aniso5':
            kl = kl_variance_proxy(sub, pot, kT)
        else:
            kl = 10.0
        kl_values.append(kl)

    return checkpoint_fes.tolist(), kl_values


def run_E2():
    print("\n" + "=" * 60)
    print("E2: KL Convergence Race")
    print("=" * 60)

    targets = ['ho1d', 'dw2d', 'gmm2d', 'aniso5']
    methods = ['NH_tuned', 'NHC_M3', 'NH_tanh', 'HMC', 'Langevin']
    n_seeds = 5
    nfe_total = 200_000
    n_checkpoints = 15

    results = {}
    for tgt in targets:
        results[tgt] = {}
        for meth in methods:
            t0 = time.time()
            all_kls = []
            fes = None
            for s in range(n_seeds):
                cp_fes, kls = _kl_checkpoints_method(meth, tgt, 3000 + s, nfe_total, n_checkpoints)
                all_kls.append(kls)
                fes = cp_fes

            all_kls = np.array(all_kls)  # (n_seeds, n_checkpoints)
            results[tgt][meth] = {
                'force_evals': fes,
                'kl_median': np.median(all_kls, axis=0).tolist(),
                'kl_q25': np.percentile(all_kls, 25, axis=0).tolist(),
                'kl_q75': np.percentile(all_kls, 75, axis=0).tolist(),
            }
            elapsed = time.time() - t0
            final_kl = np.median(all_kls[:, -1])
            print(f"  {tgt:8s} {meth:10s}: final KL={final_kl:.4f}  ({elapsed:.1f}s)", flush=True)

    with open(os.path.join(OUT_DIR, "results_kl.json"), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def plot_E2(results):
    """4-panel KL convergence plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    targets = list(results.keys())
    methods = list(results[targets[0]].keys())

    target_titles = {'ho1d': '1D Harmonic Oscillator', 'dw2d': '2D Double Well',
                     'gmm2d': '2D Gaussian Mixture (5 modes)', 'aniso5': '5D Anisotropic Gaussian'}
    method_colors = {
        'NH_tuned': '#1f77b4', 'NHC_M3': '#ff7f0e', 'NH_tanh': '#2ca02c',
        'HMC': '#d62728', 'Langevin': '#7f7f7f'
    }
    method_labels = {
        'NH_tuned': 'NH (Q tuned)', 'NHC_M3': 'NHC (M=3)', 'NH_tanh': 'NH (g=tanh)',
        'HMC': 'HMC (tuned)', 'Langevin': 'Langevin'
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True, dpi=300)

    for ax_i, tgt in enumerate(targets):
        ax = axes[ax_i]
        for meth in methods:
            d = results[tgt][meth]
            fes = np.array(d['force_evals'])
            med = np.array(d['kl_median'])
            q25 = np.array(d['kl_q25'])
            q75 = np.array(d['kl_q75'])
            # Clamp tiny values for log scale
            med = np.maximum(med, 1e-5)
            q25 = np.maximum(q25, 1e-5)
            q75 = np.maximum(q75, 1e-5)
            ax.plot(fes, med, color=method_colors[meth], label=method_labels[meth], linewidth=1.5)
            ax.fill_between(fes, q25, q75, color=method_colors[meth], alpha=0.15)

        ax.axhline(0.01, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Force evaluations', fontsize=11)
        ax.set_title(f'({chr(97+ax_i)}) {target_titles[tgt]}', fontsize=11)
        ax.grid(alpha=0.2, linestyle='--')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        if ax_i == 0:
            ax.set_ylabel('KL divergence (or variance proxy)', fontsize=11)

    axes[-1].legend(fontsize=8, loc='upper right', framealpha=0.9)
    fig.suptitle('Convergence Race: KL Divergence vs Computational Budget', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig2_kl_convergence.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ============================================================================
# E3: Ergodicity Phase Transition at omega
# ============================================================================

def _run_phase_single(args):
    """Run one (method, omega, seed) for the phase transition scan."""
    method, omega, seed = args
    kT = 1.0; dt = 0.005; nsteps = 200_000
    pot = HarmonicOscillator1D(omega=omega)
    dim = 1
    true_var = kT / omega**2

    if method == 'logosc':
        # Log-osc resonance: solve omega^2 = (2Q-1)/(Q(Q+1))
        # When omega > 0.732, no solution -> use Q* = 1.37
        # Numerically find Q from omega^2 * Q * (Q+1) = 2Q - 1
        # omega^2 * Q^2 + (omega^2 + 1)*Q + 1 - 2Q = 0 ... wait let me be precise
        # omega^2 = (2Q-1) / (Q*(Q+1))
        # omega^2 * Q * (Q+1) = 2Q - 1
        # omega^2 * Q^2 + omega^2 * Q - 2Q + 1 = 0
        # omega^2 * Q^2 + (omega^2 - 2)*Q + 1 = 0
        a_coeff = omega**2
        b_coeff = omega**2 - 2
        c_coeff = 1
        disc = b_coeff**2 - 4*a_coeff*c_coeff
        if disc >= 0 and a_coeff > 0:
            Q1 = (-b_coeff + np.sqrt(disc)) / (2*a_coeff)
            Q2 = (-b_coeff - np.sqrt(disc)) / (2*a_coeff)
            # Pick the positive root > 0.5 (physical)
            Q_cands = [q for q in [Q1, Q2] if q > 0.5]
            Q = Q_cands[0] if Q_cands else 1.37
        else:
            Q = 1.37  # ceiling value
        tr = sim_nh(pot, Q, dt, nsteps, kT=kT, seed=seed, g_func=g_logosc)

    elif method == 'tanh':
        Q = 100.0
        tr = sim_nh(pot, Q, dt, nsteps, kT=kT, seed=seed, g_func=g_tanh)

    elif method == 'NHC_M3':
        Q = 1.0
        tr = sim_nhc(pot, np.ones(3)*Q, dt, nsteps, kT=kT, seed=seed)
    else:
        raise ValueError(method)

    v = tr[~np.isnan(tr[:, 0])]
    if len(v) < 100:
        return 10.0
    emp_var = float(np.var(v[:, 0]))
    rel_err = abs(emp_var - true_var) / true_var
    return rel_err


def run_E3():
    print("\n" + "=" * 60)
    print("E3: Ergodicity Phase Transition")
    print("=" * 60)

    omegas = np.concatenate([
        np.linspace(0.1, 0.7, 5),
        np.linspace(0.72, 0.76, 3),
        np.linspace(0.8, 3.0, 8),
    ])
    methods = ['logosc', 'tanh', 'NHC_M3']
    n_seeds = 3

    results = {'omegas': omegas.tolist(), 'methods': {}}

    for meth in methods:
        errs_all = []
        for omega in omegas:
            seed_errs = []
            for s in range(n_seeds):
                err = _run_phase_single((meth, omega, 4000 + s))
                seed_errs.append(err)
            errs_all.append(seed_errs)

        errs_arr = np.array(errs_all)  # (n_omega, n_seeds)
        results['methods'][meth] = {
            'median_err': np.median(errs_arr, axis=1).tolist(),
            'q25_err': np.percentile(errs_arr, 25, axis=1).tolist(),
            'q75_err': np.percentile(errs_arr, 75, axis=1).tolist(),
        }
        print(f"  {meth:8s}: max_err={np.max(np.median(errs_arr, axis=1)):.3f}  "
              f"mean_err={np.mean(np.median(errs_arr, axis=1)):.3f}", flush=True)

    with open(os.path.join(OUT_DIR, "results_phase.json"), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def plot_E3(results):
    """Phase transition: ergodicity error vs omega."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    omegas = np.array(results['omegas'])
    method_colors = {'logosc': '#2ca02c', 'tanh': '#d62728', 'NHC_M3': '#ff7f0e'}
    method_labels = {'logosc': 'Log-osc (N=1)', 'tanh': 'Tanh (N=1)', 'NHC_M3': 'NHC (M=3)'}

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=300)

    for meth in ['logosc', 'tanh', 'NHC_M3']:
        d = results['methods'][meth]
        med = np.array(d['median_err'])
        q25 = np.array(d['q25_err'])
        q75 = np.array(d['q75_err'])
        ax.plot(omegas, med, color=method_colors[meth], label=method_labels[meth], linewidth=2)
        ax.fill_between(omegas, q25, q75, color=method_colors[meth], alpha=0.15)

    ax.axvline(0.732, color='gray', linestyle='--', alpha=0.6, linewidth=1)
    ax.annotate('$\\omega^* = 0.732$\n(resonance ceiling)', xy=(0.732, 0.5),
                xytext=(1.2, 0.7), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='gray', lw=1),
                color='gray', ha='center')

    ax.set_xlabel('Oscillator frequency $\\omega$', fontsize=14)
    ax.set_ylabel('Relative variance error $|\\hat{\\sigma}^2 - \\sigma^2_{true}| / \\sigma^2_{true}$', fontsize=12)
    ax.set_title('Ergodicity Phase Transition: Log-Osc Resonance Ceiling', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 10)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig3_phase_transition.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ============================================================================
# Main
# ============================================================================

def run_all():
    t_start = time.time()

    print("Paper-Final-049: Three Final Experiments")
    print("=" * 60)

    # E1: HMC Head-to-Head
    res_e1 = run_E1()
    plot_E1(res_e1)

    # E2: KL Convergence Race
    res_e2 = run_E2()
    plot_E2(res_e2)

    # E3: Ergodicity Phase Transition
    res_e3 = run_E3()
    plot_E3(res_e3)

    # Summary metric: ESS_NH_tanh / ESS_HMC on 10D aniso
    nh_ess = res_e1['aniso10']['NH_tanh']['median_ess']
    hmc_ess = res_e1['aniso10']['HMC_tuned']['median_ess']
    ratio = nh_ess / hmc_ess if hmc_ess > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"  ESS_NH_tanh / ESS_HMC (10D aniso) = {ratio:.3f}")
    print(f"  Total time: {time.time()-t_start:.0f}s")
    print(f"{'='*60}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'e3':
        # Run only E3
        res_e3 = run_E3()
        plot_E3(res_e3)
    else:
        run_all()
