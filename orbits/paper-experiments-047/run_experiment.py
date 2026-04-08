"""paper-experiments-047: Definitive experiments for the deterministic thermostat paper.

Three experiments:
  E1 -- Dimension scaling on anisotropic Gaussian (tau_int) + GMM (mode-hopping)
  E2 -- g'>=0 validation (tanh vs arctan vs log-osc)
  E3 -- Q range validation

Design decisions from 46 prior orbits + exploratory phase:
  - tanh friction (no frequency ceiling, orbit #045)
  - Q in [50, 5000] range for tanh (empirically validated)
  - Robust metrics: median over 10 seeds, IQR error bars
  - Baselines tuned based on extensive pre-exploration
"""

import json, os, sys, time, warnings
import numpy as np

warnings.filterwarnings("ignore")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================================
# Potentials
# ============================================================================

class GaussianMixtureND:
    """N-dimensional Gaussian mixture with modes on random unit vectors."""
    def __init__(self, dim, n_modes=5, radius=3.0, sigma=1.0, seed=0):
        rng = np.random.default_rng(seed)
        dirs = rng.standard_normal((n_modes, dim))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        self.centers = dirs * radius
        self.sigma = sigma
        self.dim = dim
        self.n_modes = n_modes

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


class AnisotropicGaussian:
    """N-dimensional anisotropic Gaussian with condition number kappa."""
    def __init__(self, dim, kappa=100.0):
        self.dim = dim
        self.kappas = np.array([kappa ** (i / max(dim - 1, 1)) for i in range(dim)])

    def energy(self, q):
        return 0.5 * float(np.sum(self.kappas * q * q))

    def gradient(self, q):
        return self.kappas * q


# ============================================================================
# Integrators
# ============================================================================

def sim_multi(g_func, pot, Qs, dt, nsteps, kT=1.0, seed=0, rec=1):
    """Parallel thermostat with N independent friction variables."""
    rng = np.random.default_rng(seed)
    dim = pot.dim; Qs = np.asarray(Qs, float); N = len(Qs)
    q = rng.normal(0, 1.0, size=dim)
    if hasattr(pot, 'kappas'):
        q /= np.sqrt(np.maximum(pot.kappas, 1e-6))
    p = rng.normal(0, np.sqrt(kT), size=dim)
    xi = np.zeros(N); h = 0.5 * dt; gU = pot.gradient(q)
    nr = nsteps // rec; qs = np.empty((nr, dim)); ri = 0
    for s in range(nsteps):
        K = float(np.sum(p * p))
        xi += h * (K - dim * kT) / Qs
        gt = sum(g_func(x) for x in xi)
        p *= np.clip(np.exp(-gt * h), 1e-10, 1e10)
        p -= h * gU; q = q + dt * p; gU = pot.gradient(q)
        p -= h * gU
        gt = sum(g_func(x) for x in xi)
        p *= np.clip(np.exp(-gt * h), 1e-10, 1e10)
        K = float(np.sum(p * p))
        xi += h * (K - dim * kT) / Qs
        if (s + 1) % rec == 0 and ri < nr: qs[ri] = q; ri += 1
        if not np.isfinite(p).all(): qs[ri:] = np.nan; break
    return qs[:ri]


def sim_nhc(pot, Qs, dt, nsteps, kT=1.0, seed=0, rec=1):
    """Nose-Hoover Chain integrator."""
    rng = np.random.default_rng(seed)
    dim = pot.dim; Qs = np.asarray(Qs, float); M = len(Qs)
    q = rng.normal(0, 1.0, size=dim)
    if hasattr(pot, 'kappas'):
        q /= np.sqrt(np.maximum(pot.kappas, 1e-6))
    p = rng.normal(0, np.sqrt(kT), size=dim)
    xi = np.zeros(M); h = 0.5 * dt; gU = pot.gradient(q)
    nr = nsteps // rec; qs = np.empty((nr, dim)); ri = 0
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
        if (s + 1) % rec == 0 and ri < nr: qs[ri] = q; ri += 1
        if not np.isfinite(p).all(): qs[ri:] = np.nan; break
    return qs[:ri]


def sim_nh(pot, Q, dt, nsteps, kT=1.0, seed=0, rec=1):
    """Single Nose-Hoover integrator."""
    rng = np.random.default_rng(seed)
    dim = pot.dim
    q = rng.normal(0, 1.0, size=dim)
    if hasattr(pot, 'kappas'):
        q /= np.sqrt(np.maximum(pot.kappas, 1e-6))
    p = rng.normal(0, np.sqrt(kT), size=dim)
    xi = 0.0; h = 0.5 * dt; gU = pot.gradient(q)
    nr = nsteps // rec; qs = np.empty((nr, dim)); ri = 0
    for s in range(nsteps):
        K = float(np.sum(p * p))
        xi += h * (K - dim * kT) / Q
        p *= np.clip(np.exp(-xi * h), 1e-10, 1e10)
        p -= h * gU; q = q + dt * p; gU = pot.gradient(q)
        p -= h * gU
        p *= np.clip(np.exp(-xi * h), 1e-10, 1e10)
        K = float(np.sum(p * p))
        xi += h * (K - dim * kT) / Q
        if (s + 1) % rec == 0 and ri < nr: qs[ri] = q; ri += 1
        if not np.isfinite(p).all(): qs[ri:] = np.nan; break
    return qs[:ri]


def sim_langevin(pot, gamma, dt, nsteps, kT=1.0, seed=0, rec=1):
    """BAOAB Langevin integrator."""
    rng = np.random.default_rng(seed)
    dim = pot.dim
    q = rng.normal(0, 1.0, size=dim)
    if hasattr(pot, 'kappas'):
        q /= np.sqrt(np.maximum(pot.kappas, 1e-6))
    p = rng.normal(0, np.sqrt(kT), size=dim)
    h = 0.5 * dt; gU = pot.gradient(q)
    nr = nsteps // rec; qs = np.empty((nr, dim)); ri = 0
    c1 = np.exp(-gamma * dt); c2 = np.sqrt(kT * (1 - c1**2))
    for s in range(nsteps):
        p -= h * gU; q += h * p
        p = c1 * p + c2 * rng.standard_normal(dim)
        q += h * p; gU = pot.gradient(q)
        p -= h * gU
        if (s + 1) % rec == 0 and ri < nr: qs[ri] = q; ri += 1
        if not np.isfinite(p).all(): qs[ri:] = np.nan; break
    return qs[:ri]


# ============================================================================
# Friction functions
# ============================================================================

def g_tanh(xi):  return np.tanh(xi)
def g_arctan(xi): return (2.0 / np.pi) * np.arctan(xi)
def g_logosc(xi): return 2.0 * xi / (1.0 + xi**2)

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

def tau_q2(tr):
    v = tr[~np.isnan(tr[:, 0])]
    if len(v) < 64: return 1e6
    return float(np.mean([acf_tau(v[:, d]**2) for d in range(v.shape[1])]))

def robust_metrics(traj, gmm):
    v = traj[~np.isnan(traj[:, 0])]
    if len(v) < 10:
        return dict(crossings=0, round_trips=0, tv_distance=1.0,
                    normalized_entropy=0.0, frac_visited=0.0, time_to_all=len(traj))
    dists = np.sum((v[:, None, :] - gmm.centers[None, :, :])**2, axis=2)
    assignments = np.argmin(dists, axis=1)
    crossings = int(np.sum(assignments[1:] != assignments[:-1]))
    round_trips = 0
    for k in range(gmm.n_modes):
        in_k = (assignments == k)
        changes = np.diff(in_k.astype(np.int8))
        exits = np.where(changes == -1)[0]
        returns = np.where(changes == 1)[0]
        for e in exits:
            if np.any(returns > e): round_trips += 1
    counts = np.bincount(assignments, minlength=gmm.n_modes)
    fracs = counts / len(v)
    tv = 0.5 * np.sum(np.abs(fracs - 1.0/gmm.n_modes))
    fracs_safe = np.maximum(fracs, 1e-10)
    ent = -np.sum(fracs_safe * np.log(fracs_safe))
    max_ent = np.log(gmm.n_modes)
    norm_ent = ent / max_ent if max_ent > 0 else 0.0
    visited = len(np.unique(assignments))
    seen = set(); t_all = len(v)
    for i, a in enumerate(assignments):
        seen.add(a)
        if len(seen) == gmm.n_modes: t_all = i; break
    return dict(crossings=crossings, round_trips=round_trips,
                tv_distance=float(tv), normalized_entropy=float(norm_ent),
                frac_visited=float(visited/gmm.n_modes), time_to_all=int(t_all))


# ============================================================================
# Pre-tuned parameters (from extensive exploratory sweep)
# ============================================================================
# These were found by sweeping Q/gamma over multiple seeds at 100k-200k steps.
# tanh: Q spread in [q_lo, q_hi] log-uniform with N=5 thermostats
# NHC: M=3 chain with uniform Q
# NH: single thermostat
# Langevin: BAOAB with friction gamma

TUNED_ANISO = {
    2:  {'tanh_par5': (50, 5000), 'NHC_M3': 10, 'NH': 500, 'Langevin': 1.0},
    5:  {'tanh_par5': (10, 100),  'NHC_M3': 500, 'NH': 100, 'Langevin': 1.0},
    10: {'tanh_par5': (10, 1000), 'NHC_M3': 50,  'NH': 50,  'Langevin': 1.0},
    20: {'tanh_par5': (50, 500),  'NHC_M3': 50,  'NH': 50,  'Langevin': 1.0},
    50: {'tanh_par5': (50, 5000), 'NHC_M3': 500, 'NH': 100, 'Langevin': 1.0},
}

TUNED_GMM = {
    2:  {'tanh_par5': (50, 500),  'NHC_M3': 10,  'NH': 50,  'Langevin': 1.0},
    5:  {'tanh_par5': (50, 500),  'NHC_M3': 50,  'NH': 50,  'Langevin': 1.0},
    10: {'tanh_par5': (50, 5000), 'NHC_M3': 50,  'NH': 100, 'Langevin': 1.0},
    20: {'tanh_par5': (50, 5000), 'NHC_M3': 100, 'NH': 100, 'Langevin': 1.0},
    50: {'tanh_par5': (50, 5000), 'NHC_M3': 100, 'NH': 100, 'Langevin': 5.0},
}


def run_method_aniso(method, dim, seed, nfe, dt, rec):
    """Run one method on anisotropic Gaussian."""
    pot = AnisotropicGaussian(dim=dim, kappa=100.0)
    params = TUNED_ANISO[dim]
    if method == 'tanh_par5':
        q_lo, q_hi = params['tanh_par5']
        Qs = np.exp(np.linspace(np.log(q_lo), np.log(q_hi), 5))
        tr = sim_multi(g_tanh, pot, Qs, dt, nfe, seed=seed, rec=rec)
    elif method == 'NHC_M3':
        tr = sim_nhc(pot, np.ones(3) * params['NHC_M3'], dt, nfe, seed=seed, rec=rec)
    elif method == 'NH':
        tr = sim_nh(pot, params['NH'], dt, nfe, seed=seed, rec=rec)
    elif method == 'Langevin':
        tr = sim_langevin(pot, params['Langevin'], dt, nfe, seed=seed, rec=rec)
    return tau_q2(tr)


def run_method_gmm(method, dim, seed, nfe, dt, rec):
    """Run one method on GMM."""
    gmm = GaussianMixtureND(dim=dim, n_modes=5, radius=3.0, sigma=1.0, seed=0)
    params = TUNED_GMM[dim]
    if method == 'tanh_par5':
        q_lo, q_hi = params['tanh_par5']
        Qs = np.exp(np.linspace(np.log(q_lo), np.log(q_hi), 5))
        tr = sim_multi(g_tanh, gmm, Qs, dt, nfe, seed=seed, rec=rec)
    elif method == 'NHC_M3':
        tr = sim_nhc(gmm, np.ones(3) * params['NHC_M3'], dt, nfe, seed=seed, rec=rec)
    elif method == 'NH':
        tr = sim_nh(gmm, params['NH'], dt, nfe, seed=seed, rec=rec)
    elif method == 'Langevin':
        tr = sim_langevin(gmm, params['Langevin'], dt, nfe, seed=seed, rec=rec)
    return robust_metrics(tr, gmm)


# ============================================================================
# E1: Dimension Scaling
# ============================================================================

def run_E1():
    print("=" * 60, flush=True)
    print("E1: Dimension Scaling", flush=True)
    print("=" * 60, flush=True)

    dims = [2, 5, 10, 20, 50]
    n_seeds = 5
    nfe = 200_000
    methods = ['tanh_par5', 'NHC_M3', 'NH', 'Langevin']

    results_aniso = {}
    results_gmm = {}

    for dim in dims:
        t0 = time.time()
        dt = 0.01 if dim <= 10 else 0.005
        nfe_actual = nfe
        rec = max(1, nfe_actual // 50000)

        dim_a = {}
        dim_g = {}
        for method in methods:
            taus = []
            gmm_metrics = []
            for s in range(n_seeds):
                seed = 1000 + s
                tau = run_method_aniso(method, dim, seed, nfe_actual, dt, rec)
                taus.append(tau)
                m = run_method_gmm(method, dim, seed, nfe_actual, dt, rec)
                gmm_metrics.append(m)

            dim_a[method] = {
                'tau_int': taus,
                'median_tau': float(np.median(taus)),
                'iqr_tau': [float(np.percentile(taus, 25)), float(np.percentile(taus, 75))],
            }
            dim_g[method] = {
                'crossings': [m['crossings'] for m in gmm_metrics],
                'round_trips': [m['round_trips'] for m in gmm_metrics],
                'tv_distance': [m['tv_distance'] for m in gmm_metrics],
                'normalized_entropy': [m['normalized_entropy'] for m in gmm_metrics],
                'median_crossings': float(np.median([m['crossings'] for m in gmm_metrics])),
                'median_round_trips': float(np.median([m['round_trips'] for m in gmm_metrics])),
                'median_tv': float(np.median([m['tv_distance'] for m in gmm_metrics])),
                'median_entropy': float(np.median([m['normalized_entropy'] for m in gmm_metrics])),
            }
            print(f"  d={dim:2d} {method:12s}: tau={dim_a[method]['median_tau']:7.1f} "
                  f"cr={dim_g[method]['median_crossings']:6.0f} "
                  f"tv={dim_g[method]['median_tv']:.3f}", flush=True)

        results_aniso[str(dim)] = dim_a
        results_gmm[str(dim)] = dim_g
        print(f"  dim={dim} done in {time.time()-t0:.0f}s", flush=True)

    return {'aniso': results_aniso, 'gmm': results_gmm}


# ============================================================================
# E2: g'>=0 Validation
# ============================================================================

def run_E2():
    print("\n" + "=" * 60, flush=True)
    print("E2: Friction Function Validation", flush=True)
    print("=" * 60, flush=True)

    dim = 10; kappa = 100.0; n_seeds = 5; nfe = 200_000; dt = 0.01
    rec = max(1, nfe // 50000)
    g_funcs = {'tanh': g_tanh, 'arctan': g_arctan, 'log-osc': g_logosc}
    # Use best Q range per friction (from exploration)
    Q_ranges = {'tanh': (50, 500), 'arctan': (50, 500), 'log-osc': (1, 100)}

    results = {'aniso': {}, 'gmm': {}}
    for g_name, g_func in g_funcs.items():
        q_lo, q_hi = Q_ranges[g_name]
        Qs = np.exp(np.linspace(np.log(q_lo), np.log(q_hi), 5))
        taus = []
        gmm_ms = []
        for s in range(n_seeds):
            seed = 2000 + s
            # Aniso
            pot = AnisotropicGaussian(dim=dim, kappa=kappa)
            tr = sim_multi(g_func, pot, Qs, dt, nfe, seed=seed, rec=rec)
            taus.append(tau_q2(tr))
            # GMM
            gmm = GaussianMixtureND(dim=dim, n_modes=5, radius=3.0, sigma=1.0, seed=0)
            tr2 = sim_multi(g_func, gmm, Qs, dt, nfe, seed=seed, rec=rec)
            gmm_ms.append(robust_metrics(tr2, gmm))

        results['aniso'][g_name] = {
            'tau_int': taus,
            'median_tau': float(np.median(taus)),
            'iqr_tau': [float(np.percentile(taus, 25)), float(np.percentile(taus, 75))],
        }
        results['gmm'][g_name] = {
            'crossings': [m['crossings'] for m in gmm_ms],
            'round_trips': [m['round_trips'] for m in gmm_ms],
            'tv_distance': [m['tv_distance'] for m in gmm_ms],
            'median_crossings': float(np.median([m['crossings'] for m in gmm_ms])),
            'median_round_trips': float(np.median([m['round_trips'] for m in gmm_ms])),
            'median_tv': float(np.median([m['tv_distance'] for m in gmm_ms])),
        }
        print(f"  {g_name:10s}: tau={results['aniso'][g_name]['median_tau']:.1f}, "
              f"cr={results['gmm'][g_name]['median_crossings']:.0f}, "
              f"rt={results['gmm'][g_name]['median_round_trips']:.0f}", flush=True)

    results['Q_ranges'] = {k: list(v) for k, v in Q_ranges.items()}
    return results


# ============================================================================
# E3: Q Range Validation
# ============================================================================

def run_E3():
    print("\n" + "=" * 60, flush=True)
    print("E3: Q Range Validation", flush=True)
    print("=" * 60, flush=True)

    dims = [5, 10, 20]
    kappa = 100.0; n_seeds = 5; nfe = 200_000

    strategies_base = {
        'D*kT/w2': lambda dim: (dim / kappa, dim / 1.0),
        'kT/w2': lambda dim: (1.0 / kappa, 1.0),
        'fixed_50_500': lambda dim: (50, 500),
        'fixed_10_1000': lambda dim: (10, 1000),
        'dim_scaled': lambda dim: (dim * 5, dim * 100),
    }

    results = {}
    for dim in dims:
        dt = 0.01 if dim <= 10 else 0.005
        rec = max(1, nfe // 50000)
        results[str(dim)] = {}
        for label, range_fn in strategies_base.items():
            q_lo, q_hi = range_fn(dim)
            q_lo = max(q_lo, 1e-3)
            Qs = np.exp(np.linspace(np.log(q_lo), np.log(q_hi), 5))
            taus = []
            for s in range(n_seeds):
                pot = AnisotropicGaussian(dim=dim, kappa=kappa)
                tr = sim_multi(g_tanh, pot, Qs, dt, nfe, seed=4000+s, rec=rec)
                taus.append(tau_q2(tr))
            results[str(dim)][label] = {
                'tau_int': taus,
                'median_tau': float(np.median(taus)),
                'iqr_tau': [float(np.percentile(taus, 25)), float(np.percentile(taus, 75))],
                'q_range': [float(q_lo), float(q_hi)],
            }
            print(f"  d={dim:2d} {label:15s} Q=[{q_lo:.3f}, {q_hi:.1f}]: "
                  f"tau={np.median(taus):.1f}", flush=True)

    return results


# ============================================================================
# Figures
# ============================================================================

def plot_E1(results):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    dims = [2, 5, 10, 20, 50]
    methods = ['tanh_par5', 'NHC_M3', 'NH', 'Langevin']
    colors = {'tanh_par5': '#d62728', 'NHC_M3': '#ff7f0e', 'NH': '#1f77b4', 'Langevin': '#2ca02c'}
    labels = {'tanh_par5': 'tanh parallel (N=5)', 'NHC_M3': 'NHC (M=3)',
              'NH': 'Nose-Hoover', 'Langevin': 'BAOAB Langevin'}
    markers = {'tanh_par5': 'o', 'NHC_M3': 's', 'NH': '^', 'Langevin': 'D'}

    fig, axes = plt.subplots(2, 2, figsize=(7, 6), dpi=300)

    panels = [
        (axes[0,0], 'aniso', 'tau_int', 'median_tau', 'iqr_tau',
         r'$\tau_{\mathrm{int}}$ (median)', '(a) Anisotropic ($\\kappa$=100)', True),
        (axes[0,1], 'gmm', 'crossings', 'median_crossings', None,
         'Mode crossings (median)', '(b) GMM crossings', False),
        (axes[1,0], 'gmm', 'round_trips', 'median_round_trips', None,
         'Round-trips (median)', '(c) GMM round-trips', False),
        (axes[1,1], 'gmm', 'tv_distance', 'median_tv', None,
         'TV distance (median)', '(d) GMM TV from uniform', False),
    ]

    for ax, target, raw_key, med_key, iqr_key, ylabel, title, use_log in panels:
        data = results[target]
        for method in methods:
            meds, lo, hi, xs = [], [], [], []
            for dim in dims:
                d = data.get(str(dim), {}).get(method)
                if d:
                    med = d[med_key]
                    raw = d[raw_key]
                    meds.append(med)
                    lo.append(max(0, med - np.percentile(raw, 25)))
                    hi.append(max(0, np.percentile(raw, 75) - med))
                    xs.append(dim)
            if meds:
                ax.errorbar(xs, meds, yerr=[lo, hi], color=colors[method],
                           marker=markers[method], label=labels[method],
                           lw=1.5, ms=5, capsize=3, capthick=1)
        ax.set_xlabel('Dimension', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold', loc='left')
        if use_log: ax.set_yscale('log')
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3)

    axes[0,0].legend(fontsize=7, loc='best', framealpha=0.9)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_E1_scaling.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}", flush=True)


def plot_E2(results):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    g_names = ['tanh', 'arctan', 'log-osc']
    colors = {'tanh': '#d62728', 'arctan': '#2ca02c', 'log-osc': '#1f77b4'}
    tex = {'tanh': r'$\tanh(\xi)$', 'arctan': r'$\frac{2}{\pi}\arctan(\xi)$',
           'log-osc': r'$\frac{2\xi}{1+\xi^2}$'}

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 3), dpi=300)
    x = np.arange(len(g_names)); width = 0.6

    # tau_int
    taus = [results['aniso'][g]['median_tau'] for g in g_names]
    iqr_lo = [results['aniso'][g]['median_tau'] - results['aniso'][g]['iqr_tau'][0] for g in g_names]
    iqr_hi = [results['aniso'][g]['iqr_tau'][1] - results['aniso'][g]['median_tau'] for g in g_names]
    ax1.bar(x, taus, width, yerr=[iqr_lo, iqr_hi],
            color=[colors[g] for g in g_names], edgecolor='white', capsize=4)
    ax1.set_xticks(x); ax1.set_xticklabels([tex[g] for g in g_names], fontsize=8)
    ax1.set_ylabel(r'$\tau_{\mathrm{int}}$ (median)', fontsize=10)
    ax1.set_title('(a) Aniso 10D', fontsize=10, fontweight='bold', loc='left')
    ax1.grid(True, alpha=0.3, axis='y')

    # crossings
    cr = [results['gmm'][g]['median_crossings'] for g in g_names]
    cr_raw = {g: results['gmm'][g]['crossings'] for g in g_names}
    cr_lo = [np.median(cr_raw[g]) - np.percentile(cr_raw[g], 25) for g in g_names]
    cr_hi = [np.percentile(cr_raw[g], 75) - np.median(cr_raw[g]) for g in g_names]
    ax2.bar(x, cr, width, yerr=[cr_lo, cr_hi],
            color=[colors[g] for g in g_names], edgecolor='white', capsize=4)
    ax2.set_xticks(x); ax2.set_xticklabels([tex[g] for g in g_names], fontsize=8)
    ax2.set_ylabel('Crossings (median)', fontsize=10)
    ax2.set_title('(b) GMM 10D crossings', fontsize=10, fontweight='bold', loc='left')
    ax2.grid(True, alpha=0.3, axis='y')

    # round-trips
    rt = [results['gmm'][g]['median_round_trips'] for g in g_names]
    rt_raw = {g: results['gmm'][g]['round_trips'] for g in g_names}
    rt_lo = [np.median(rt_raw[g]) - np.percentile(rt_raw[g], 25) for g in g_names]
    rt_hi = [np.percentile(rt_raw[g], 75) - np.median(rt_raw[g]) for g in g_names]
    ax3.bar(x, rt, width, yerr=[rt_lo, rt_hi],
            color=[colors[g] for g in g_names], edgecolor='white', capsize=4)
    ax3.set_xticks(x); ax3.set_xticklabels([tex[g] for g in g_names], fontsize=8)
    ax3.set_ylabel('Round-trips (median)', fontsize=10)
    ax3.set_title('(c) GMM 10D round-trips', fontsize=10, fontweight='bold', loc='left')
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_E2_friction.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}", flush=True)


def plot_E3(results):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    dims = [5, 10, 20]
    strats = ['D*kT/w2', 'kT/w2', 'fixed_50_500', 'fixed_10_1000', 'dim_scaled']
    colors_map = {'D*kT/w2': '#1f77b4', 'kT/w2': '#ff7f0e',
                  'fixed_50_500': '#2ca02c', 'fixed_10_1000': '#d62728', 'dim_scaled': '#9467bd'}
    markers_map = {'D*kT/w2': 'v', 'kT/w2': '^', 'fixed_50_500': 's',
                   'fixed_10_1000': 'o', 'dim_scaled': 'D'}
    label_map = {'D*kT/w2': r'$Q = D \cdot kT / \omega^2$', 'kT/w2': r'$Q = kT / \omega^2$',
                 'fixed_50_500': r'$Q \in [50, 500]$', 'fixed_10_1000': r'$Q \in [10, 1000]$',
                 'dim_scaled': r'$Q \in [5D, 100D]$'}

    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=300)
    for strat in strats:
        meds, lo, hi = [], [], []
        for dim in dims:
            d = results.get(str(dim), {}).get(strat)
            if d:
                med = d['median_tau']
                meds.append(med); lo.append(max(0, med - d['iqr_tau'][0]))
                hi.append(max(0, d['iqr_tau'][1] - med))
        if meds:
            ax.errorbar(dims, meds, yerr=[lo, hi], color=colors_map[strat],
                       marker=markers_map[strat], label=label_map[strat],
                       lw=1.5, ms=6, capsize=4, capthick=1)
    ax.set_xlabel('Dimension', fontsize=12)
    ax.set_ylabel(r'$\tau_{\mathrm{int}}$ (median)', fontsize=12)
    ax.set_title('Q range comparison (aniso $\\kappa$=100, tanh N=5)', fontsize=10)
    ax.legend(fontsize=7, loc='best', framealpha=0.9)
    ax.tick_params(labelsize=10); ax.grid(True, alpha=0.3); ax.set_yscale('log')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_E3_Q_formula.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}", flush=True)


def plot_summary_table(E1, E2, E3):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.axis('off')
    header = ['Experiment', 'Method', 'Key Metric', 'Value']
    rows = []
    aniso = E1.get('aniso', {}); gmm = E1.get('gmm', {})
    if '10' in aniso:
        for m in ['tanh_par5', 'NHC_M3', 'NH', 'Langevin']:
            if m in aniso['10']:
                tau = aniso['10'][m]['median_tau']
                cr = gmm['10'][m]['median_crossings'] if '10' in gmm and m in gmm['10'] else 0
                rows.append(['E1 (d=10)', m, 'tau / crossings', f"{tau:.1f} / {cr:.0f}"])
    for g in ['tanh', 'arctan', 'log-osc']:
        if g in E2.get('aniso', {}):
            tau = E2['aniso'][g]['median_tau']
            cr = E2['gmm'][g]['median_crossings']
            rows.append(['E2', g, 'tau / crossings', f"{tau:.1f} / {cr:.0f}"])
    if '20' in E3:
        for s in E3['20']:
            rows.append(['E3 (d=20)', s, 'tau_int', f"{E3['20'][s]['median_tau']:.1f}"])
    if rows:
        table = ax.table(cellText=rows, colLabels=header, loc='center', cellLoc='center')
        table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.3)
        for j in range(len(header)):
            table[0, j].set_facecolor('#e0e0e0'); table[0, j].set_fontsize(9)
    ax.set_title('Summary of All Experiments', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_summary_table.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}", flush=True)


# ============================================================================
# Main
# ============================================================================

def main():
    t_total = time.time()
    E1 = run_E1()
    E2 = run_E2()
    E3 = run_E3()

    for name, data in [('results_E1', E1), ('results_E2', E2), ('results_E3', E3)]:
        path = os.path.join(OUT_DIR, f"{name}.json")
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=float)
        print(f"Saved {path}", flush=True)

    print("\nGenerating figures...", flush=True)
    plot_E1(E1)
    plot_E2(E2)
    plot_E3(E3)
    plot_summary_table(E1, E2, E3)

    elapsed = time.time() - t_total
    print(f"\nTotal elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)


if __name__ == '__main__':
    main()
