"""
Orbit 076: Log-osc NH Friction + ESH Non-Newtonian Momentum Hybrid

Phase 0: Analytical derivation — naive combination BREAKS canonical measure
Phase 1: Implementation of corrected + heuristic variants
Phase 2: Lean experiments on benchmark systems
Phase 3: Comparison against baselines

Analytical conclusion: v(p) != p/m in position update creates an irreconcilable
mismatch in the Liouville equation. No simple fix preserves exp(-beta*H).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import time, os, json, sys
from scipy import stats

mpl.rcParams.update({
    'font.size': 13, 'axes.titlesize': 14, 'axes.labelsize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.pad_inches': 0.2,
    'axes.spines.top': False, 'axes.spines.right': False,
})

ORBIT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ORBIT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================================
# Friction & ESH velocity
# ============================================================================

def g_losc(xi):
    return 2.0 * xi / (1.0 + xi**2)

def v_esh(p, p0):
    """ESH velocity: tanh(|p|/p0) * p/|p| * p0. Bounded speed."""
    p_norm = np.sqrt(np.sum(p**2))
    if p_norm < 1e-15:
        return p.copy()
    return np.tanh(p_norm / p0) * (p / p_norm) * p0

def v_dot_p_esh(p, p0):
    """v(p).p = tanh(|p|/p0) * |p| * p0."""
    p_norm = np.sqrt(np.sum(p**2))
    if p_norm < 1e-15:
        return np.sum(p**2)
    return np.tanh(p_norm / p0) * p_norm * p0

# ============================================================================
# Autocorrelation
# ============================================================================

def compute_tau_int(x, max_lag=5000):
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    if n < 10:
        return float('inf')
    var = np.var(x)
    if var < 1e-15:
        return float('inf')
    fft_x = np.fft.fft(x, n=2*n)
    acf = np.real(np.fft.ifft(fft_x * np.conj(fft_x)))[:n] / (var * n)
    tau = 1.0
    for lag in range(1, min(max_lag, n // 2)):
        if acf[lag] < 0.05:
            break
        tau += 2.0 * acf[lag]
    return float(tau)

# ============================================================================
# KL divergence
# ============================================================================

def kl_divergence_2d(sx, sy, potential_fn, kT, n_bins=60):
    hist, xedges, yedges = np.histogram2d(sx, sy, bins=n_bins, density=True)
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    log_p = np.zeros((len(xc), len(yc)))
    for i in range(len(xc)):
        for j in range(len(yc)):
            log_p[i, j] = -potential_fn(np.array([xc[i], yc[j]])) / kT
    log_p -= np.max(log_p)
    p_true = np.exp(log_p)
    p_true /= np.sum(p_true) * dx * dy
    mask = (hist > 0) & (p_true > 0)
    if np.sum(mask) == 0:
        return float('inf')
    return float(np.sum(hist[mask] * np.log(hist[mask] / p_true[mask]) * dx * dy))

# ============================================================================
# Ergodicity score for 1D HO
# ============================================================================

def ergodicity_score_ho(q_samples, p_samples, kT=1.0):
    sigma_q = np.sqrt(kT)
    sigma_p = np.sqrt(kT)
    ks_q, _ = stats.kstest(q_samples, 'norm', args=(0, sigma_q))
    ks_p, _ = stats.kstest(p_samples, 'norm', args=(0, sigma_p))
    var_q_err = abs(np.var(q_samples) - sigma_q**2) / sigma_q**2
    var_p_err = abs(np.var(p_samples) - sigma_p**2) / sigma_p**2
    n_grid = 20
    q_bins = np.linspace(-4*sigma_q, 4*sigma_q, n_grid + 1)
    p_bins = np.linspace(-4*sigma_p, 4*sigma_p, n_grid + 1)
    hist, _, _ = np.histogram2d(q_samples, p_samples, bins=[q_bins, p_bins])
    coverage = float(np.sum(hist > 0)) / (n_grid * n_grid)
    ks_component = max(0.0, 1.0 - max(ks_q, ks_p))
    var_component = max(0.0, 1.0 - max(var_q_err, var_p_err))
    score = (ks_component * var_component * coverage) ** (1.0 / 3.0)
    return {'score': score, 'ks_q': ks_q, 'ks_p': ks_p,
            'var_q_err': var_q_err, 'var_p_err': var_p_err,
            'coverage': coverage, 'ergodic': score > 0.85}

# ============================================================================
# Potentials
# ============================================================================

def grad_harmonic_1d(q):
    return np.array([q[0]])

def U_double_well(q):
    return (q[0]**2 - 1)**2 + 0.5 * q[1]**2

def grad_double_well(q):
    return np.array([4.0 * q[0] * (q[0]**2 - 1), q[1]])

def make_aniso_gaussian(dim, kappa):
    omega2 = np.linspace(1.0, float(kappa), dim)
    def grad(q):
        return omega2 * q
    return grad, omega2

# ============================================================================
# Integrators (BAOAB splitting, all methods)
# ============================================================================

def _run_baoab(grad_fn, dim, Q, kT, dt, n_steps, seed,
               use_esh=False, p0=1.0, driving='kinetic',
               use_nhc=False, M=3, collect_full=False):
    """Unified BAOAB integrator for all methods.

    Args:
        use_esh: Use ESH v(p) for position update
        p0: ESH scale (only if use_esh)
        driving: 'kinetic' (standard |p|^2) or 'vdotp' (v(p).p corrected)
        use_nhc: Use NHC chain instead of single NH
        M: chain length for NHC
        collect_full: Return (q_traj, p_traj) instead of tau_int
    """
    rng = np.random.default_rng(seed)
    q = rng.normal(0, 0.5, size=dim)
    p = rng.normal(0, np.sqrt(kT), size=dim)
    half_dt = 0.5 * dt
    burnin = n_steps // 5
    n_collect = n_steps - burnin

    if use_nhc:
        xi = np.zeros(M)
        Q_chain = [float(Q)] * M
    else:
        xi_val = 0.0

    if collect_full:
        q_traj = np.empty((n_collect, dim))
        p_traj = np.empty((n_collect, dim))
    else:
        obs = np.empty(n_collect)
    idx = 0

    for step in range(n_steps):
        if use_nhc:
            # --- NHC thermostat half-step (tail to head) ---
            kinetic = np.sum(p**2)
            G = np.zeros(M)
            G[0] = (kinetic - dim * kT) / Q_chain[0]
            for j in range(1, M):
                G[j] = (Q_chain[j-1] * xi[j-1]**2 - kT) / Q_chain[j]
            xi[M-1] += half_dt * G[M-1]
            for j in range(M-2, -1, -1):
                if j < M-1:
                    xi[j] *= np.exp(-half_dt * xi[j+1])
                xi[j] += half_dt * G[j]
            # friction
            scale = np.clip(np.exp(-xi[0] * half_dt), 1e-10, 1e10)
            p *= scale
        else:
            # --- Single NH half-step ---
            if driving == 'vdotp' and use_esh:
                drv = v_dot_p_esh(p, p0)
            else:
                drv = np.sum(p**2)
            xi_val += half_dt * (drv - dim * kT) / Q
            g_val = g_losc(xi_val)
            scale = np.clip(np.exp(-g_val * half_dt), 1e-10, 1e10)
            p *= scale

        # Force half-step
        p -= half_dt * grad_fn(q)

        # Drift full step
        if use_esh:
            q += dt * v_esh(p, p0)
        else:
            q += dt * p  # Newtonian

        # Force half-step
        p -= half_dt * grad_fn(q)

        if use_nhc:
            # --- NHC thermostat half-step (symmetric) ---
            scale = np.clip(np.exp(-xi[0] * half_dt), 1e-10, 1e10)
            p *= scale
            kinetic = np.sum(p**2)
            G[0] = (kinetic - dim * kT) / Q_chain[0]
            for j in range(1, M):
                G[j] = (Q_chain[j-1] * xi[j-1]**2 - kT) / Q_chain[j]
            for j in range(M):
                xi[j] += half_dt * G[j]
                if j < M-1:
                    xi[j] *= np.exp(-half_dt * xi[j+1])
        else:
            # --- Single NH half-step (symmetric) ---
            g_val2 = g_losc(xi_val)
            scale2 = np.clip(np.exp(-g_val2 * half_dt), 1e-10, 1e10)
            p *= scale2
            if driving == 'vdotp' and use_esh:
                drv = v_dot_p_esh(p, p0)
            else:
                drv = np.sum(p**2)
            xi_val += half_dt * (drv - dim * kT) / Q

        if step >= burnin:
            if collect_full:
                q_traj[idx] = q.copy()
                p_traj[idx] = p.copy()
            else:
                obs[idx] = q[-1]**2
            idx += 1

        # NaN check
        check_val = xi[0] if use_nhc else xi_val
        if np.isnan(check_val) or np.any(np.isnan(q)):
            if collect_full:
                return q_traj[:max(idx,1)], p_traj[:max(idx,1)]
            return float('inf')

    if collect_full:
        return q_traj[:idx], p_traj[:idx]
    return compute_tau_int(obs[:idx])

# ============================================================================
# Convenience wrappers
# ============================================================================

def run_nh_logosc(grad_fn, dim, Q, kT, dt, n_steps, seed, **kw):
    return _run_baoab(grad_fn, dim, Q, kT, dt, n_steps, seed,
                      use_esh=False, driving='kinetic', use_nhc=False, **kw)

def run_hybrid_correct(grad_fn, dim, Q, kT, dt, n_steps, seed, p0=1.0, **kw):
    return _run_baoab(grad_fn, dim, Q, kT, dt, n_steps, seed,
                      use_esh=True, p0=p0, driving='vdotp', use_nhc=False, **kw)

def run_hybrid_heuristic(grad_fn, dim, Q, kT, dt, n_steps, seed, p0=1.0, **kw):
    return _run_baoab(grad_fn, dim, Q, kT, dt, n_steps, seed,
                      use_esh=True, p0=p0, driving='kinetic', use_nhc=False, **kw)

def run_nhc_tanh(grad_fn, dim, Q, kT, dt, n_steps, seed, M=3, **kw):
    return _run_baoab(grad_fn, dim, Q, kT, dt, n_steps, seed,
                      use_esh=False, driving='kinetic', use_nhc=True, M=M, **kw)

def run_esh_pure(grad_fn, dim, kT, dt, n_steps, seed, p0=1.0, **kw):
    """Pure ESH: ESH velocity, no friction. Use large Q to effectively disable thermostat."""
    # Run as hybrid heuristic with very large Q (thermostat effectively off)
    return _run_baoab(grad_fn, dim, 1e6, kT, dt, n_steps, seed,
                      use_esh=True, p0=p0, driving='kinetic', use_nhc=False, **kw)

# ============================================================================
# Experiments (lean version)
# ============================================================================

def experiment_aniso_gaussian():
    """Anisotropic Gaussian d=10 kappa=100: tau_int sweep."""
    print(f"\n{'='*60}")
    print(f"Experiment: Anisotropic Gaussian d=10, kappa=100")
    print(f"{'='*60}")

    grad_fn, omega2 = make_aniso_gaussian(10, 100)
    kT, dt, n_steps = 1.0, 0.005, 200_000
    seeds = [42, 123, 7]
    Q_vals = [0.05, 0.1, 0.3, 1.0]
    p0_vals = [0.5, 1.0, 2.0, 5.0]
    results = {}

    def sweep(label, runner):
        taus = [runner(s) for s in seeds]
        valid = [t for t in taus if np.isfinite(t) and t < 1e6]
        med = float(np.median(valid)) if valid else float('inf')
        results[label] = med
        print(f"  {label:45s}: tau={med:8.1f}")
        return med

    # NH log-osc
    print("\n--- NH log-osc ---")
    for Q in Q_vals:
        sweep(f'nh_logosc_Q={Q}',
              lambda s, Q=Q: run_nh_logosc(grad_fn, 10, Q, kT, dt, n_steps, s))

    # NHC(M=3)
    print("\n--- NHC(M=3) tanh ---")
    for Q in Q_vals:
        sweep(f'nhc_tanh_Q={Q}',
              lambda s, Q=Q: run_nhc_tanh(grad_fn, 10, Q, kT, dt, n_steps, s))

    # Pure ESH
    print("\n--- Pure ESH ---")
    for p0 in p0_vals:
        sweep(f'esh_pure_p0={p0}',
              lambda s, p0=p0: run_esh_pure(grad_fn, 10, kT, dt, n_steps, s, p0=p0))

    # Hybrid corrected
    print("\n--- Hybrid CORRECTED ---")
    for Q in [0.1, 0.3]:
        for p0 in p0_vals:
            sweep(f'hybrid_correct_Q={Q}_p0={p0}',
                  lambda s, Q=Q, p0=p0: run_hybrid_correct(grad_fn, 10, Q, kT, dt, n_steps, s, p0=p0))

    # Hybrid heuristic
    print("\n--- Hybrid HEURISTIC ---")
    for Q in [0.1, 0.3]:
        for p0 in p0_vals:
            sweep(f'hybrid_heuristic_Q={Q}_p0={p0}',
                  lambda s, Q=Q, p0=p0: run_hybrid_heuristic(grad_fn, 10, Q, kT, dt, n_steps, s, p0=p0))

    return results


def experiment_1d_ho():
    """1D HO ergodicity test."""
    print(f"\n{'='*60}")
    print(f"Experiment: 1D Harmonic Oscillator Ergodicity")
    print(f"{'='*60}")

    kT, dt, n_steps = 1.0, 0.005, 500_000
    seed = 42
    results = {}

    def test(label, runner):
        qt, pt = runner()
        erg = ergodicity_score_ho(qt[:, 0], pt[:, 0], kT=kT)
        results[label] = erg
        print(f"  {label:45s}: score={erg['score']:.4f} cov={erg['coverage']:.3f}")
        return erg

    # NH log-osc
    print("\n--- NH log-osc ---")
    for Q in [0.1, 0.3, 1.0]:
        test(f'nh_logosc_Q={Q}',
             lambda Q=Q: run_nh_logosc(grad_harmonic_1d, 1, Q, kT, dt, n_steps, seed, collect_full=True))

    # NHC(M=3)
    print("\n--- NHC(M=3) ---")
    for Q in [0.1, 0.3, 1.0]:
        test(f'nhc_tanh_Q={Q}',
             lambda Q=Q: run_nhc_tanh(grad_harmonic_1d, 1, Q, kT, dt, n_steps, seed, collect_full=True))

    # Hybrid corrected
    print("\n--- Hybrid CORRECTED ---")
    for Q in [0.1, 0.3]:
        for p0 in [0.5, 1.0, 2.0]:
            test(f'hybrid_correct_Q={Q}_p0={p0}',
                 lambda Q=Q, p0=p0: run_hybrid_correct(grad_harmonic_1d, 1, Q, kT, dt, n_steps, seed, p0=p0, collect_full=True))

    # Hybrid heuristic
    print("\n--- Hybrid HEURISTIC ---")
    for Q in [0.1, 0.3]:
        for p0 in [0.5, 1.0, 2.0]:
            test(f'hybrid_heuristic_Q={Q}_p0={p0}',
                 lambda Q=Q, p0=p0: run_hybrid_heuristic(grad_harmonic_1d, 1, Q, kT, dt, n_steps, seed, p0=p0, collect_full=True))

    return results


def experiment_double_well():
    """2D double well KL test."""
    print(f"\n{'='*60}")
    print(f"Experiment: 2D Double Well KL Divergence")
    print(f"{'='*60}")

    kT, dt, n_steps = 1.0, 0.01, 300_000
    seeds = [42, 123, 7]
    results = {}

    def test_kl(label, runner):
        kls = []
        for s in seeds:
            qt, pt = runner(s)
            if len(qt) < 100:
                kls.append(float('inf'))
                continue
            kl = kl_divergence_2d(qt[:, 0], qt[:, 1], U_double_well, kT)
            kls.append(kl)
        valid = [k for k in kls if np.isfinite(k)]
        med = float(np.median(valid)) if valid else float('inf')
        results[label] = {'kl': med, 'kls': [float(k) for k in kls]}
        print(f"  {label:45s}: KL={med:.4f}")
        return med

    # NH log-osc
    print("\n--- NH log-osc ---")
    for Q in [0.1, 0.3, 1.0]:
        test_kl(f'nh_logosc_Q={Q}',
                lambda s, Q=Q: run_nh_logosc(grad_double_well, 2, Q, kT, dt, n_steps, s, collect_full=True))

    # NHC(M=3)
    print("\n--- NHC(M=3) ---")
    for Q in [0.1, 0.3, 1.0]:
        test_kl(f'nhc_tanh_Q={Q}',
                lambda s, Q=Q: run_nhc_tanh(grad_double_well, 2, Q, kT, dt, n_steps, s, collect_full=True))

    # Hybrid corrected
    print("\n--- Hybrid CORRECTED ---")
    for Q in [0.1, 0.3]:
        for p0 in [0.5, 1.0, 2.0]:
            test_kl(f'hybrid_correct_Q={Q}_p0={p0}',
                    lambda s, Q=Q, p0=p0: run_hybrid_correct(grad_double_well, 2, Q, kT, dt, n_steps, s, p0=p0, collect_full=True))

    # Hybrid heuristic
    print("\n--- Hybrid HEURISTIC ---")
    for Q in [0.1, 0.3]:
        for p0 in [0.5, 1.0, 2.0]:
            test_kl(f'hybrid_heuristic_Q={Q}_p0={p0}',
                    lambda s, Q=Q, p0=p0: run_hybrid_heuristic(grad_double_well, 2, Q, kT, dt, n_steps, s, p0=p0, collect_full=True))

    return results


# ============================================================================
# Plotting
# ============================================================================

def plot_aniso_results(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    Q_vals = [0.05, 0.1, 0.3, 1.0]
    p0_vals = [0.5, 1.0, 2.0, 5.0]

    # Panel 1: NH log-osc vs NHC baselines
    ax = axes[0]
    nh_taus = [results.get(f'nh_logosc_Q={Q}', float('inf')) for Q in Q_vals]
    nhc_taus = [results.get(f'nhc_tanh_Q={Q}', float('inf')) for Q in Q_vals]
    ax.semilogy(Q_vals, nh_taus, 'o-', color='#d62728', lw=2, label='NH log-osc')
    ax.semilogy(Q_vals, nhc_taus, 's--', color='#1f77b4', lw=2, label='NHC(M=3)')
    esh_taus = [results.get(f'esh_pure_p0={p0}', float('inf')) for p0 in p0_vals]
    ax.axhline(min(t for t in esh_taus if np.isfinite(t)), color='gray', ls=':', lw=1,
               label=f'Best pure ESH')
    ax.set_xlabel('Q'); ax.set_ylabel(r'$\tau_{int}$'); ax.set_xscale('log')
    ax.set_title('Baselines (d=10, kappa=100)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 2: Hybrids at Q=0.1
    ax = axes[1]
    colors_c = plt.cm.Greens(np.linspace(0.4, 0.9, len(p0_vals)))
    colors_h = plt.cm.Oranges(np.linspace(0.4, 0.9, len(p0_vals)))
    x = np.arange(len(p0_vals))
    width = 0.35
    for Q in [0.1]:
        c_taus = [results.get(f'hybrid_correct_Q={Q}_p0={p0}', float('inf')) for p0 in p0_vals]
        h_taus = [results.get(f'hybrid_heuristic_Q={Q}_p0={p0}', float('inf')) for p0 in p0_vals]
        ax.bar(x - width/2, c_taus, width, label='Corrected', color='#2ca02c', alpha=0.7)
        ax.bar(x + width/2, h_taus, width, label='Heuristic', color='#ff7f0e', alpha=0.7)
    # Reference lines
    nh_best = min(t for t in nh_taus if np.isfinite(t))
    ax.axhline(nh_best, color='#d62728', ls='--', lw=1.5, label=f'NH log-osc best ({nh_best:.0f})')
    ax.set_xticks(x); ax.set_xticklabels([str(p) for p in p0_vals])
    ax.set_xlabel('p0'); ax.set_ylabel(r'$\tau_{int}$')
    ax.set_title(f'Hybrids at Q=0.1')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'aniso_gaussian_comparison.png')
    plt.savefig(fig_path, bbox_inches='tight'); plt.close()
    print(f"Saved {fig_path}")


def plot_ergodicity(results_ho):
    fig, ax = plt.subplots(figsize=(14, 5))
    labels = sorted(results_ho.keys())
    scores = [results_ho[k]['score'] for k in labels]
    colors = []
    for k in labels:
        if 'nhc' in k: colors.append('#1f77b4')
        elif 'nh_logosc' in k: colors.append('#d62728')
        elif 'hybrid_correct' in k: colors.append('#2ca02c')
        else: colors.append('#ff7f0e')
    ax.bar(range(len(labels)), scores, color=colors, alpha=0.8)
    ax.axhline(0.85, color='gray', ls='--', lw=1, label='Ergodic threshold')
    ax.axhline(0.92, color='#1f77b4', ls=':', lw=1, label='NHC baseline (0.92)')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace('_', '\n') for l in labels], rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('Ergodicity Score'); ax.set_title('1D HO Ergodicity')
    ax.legend(); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'ergodicity_comparison.png')
    plt.savefig(fig_path, bbox_inches='tight'); plt.close()
    print(f"Saved {fig_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()
    print("=" * 70)
    print("Orbit 076: Log-osc NH + ESH Hybrid")
    print("=" * 70)
    sys.stdout.flush()

    # Phase 2
    r_aniso = experiment_aniso_gaussian()
    sys.stdout.flush()
    r_ho = experiment_1d_ho()
    sys.stdout.flush()
    r_dw = experiment_double_well()
    sys.stdout.flush()

    # Plotting
    print("\n--- Generating figures ---")
    plot_aniso_results(r_aniso)
    plot_ergodicity(r_ho)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Best tau_int on aniso Gaussian
    nh_best = min(v for k, v in r_aniso.items() if k.startswith('nh_logosc') and np.isfinite(v))
    nhc_best = min(v for k, v in r_aniso.items() if k.startswith('nhc_tanh') and np.isfinite(v))
    hc_vals = [(k, v) for k, v in r_aniso.items() if k.startswith('hybrid_correct') and np.isfinite(v)]
    hh_vals = [(k, v) for k, v in r_aniso.items() if k.startswith('hybrid_heuristic') and np.isfinite(v)]
    esh_vals = [(k, v) for k, v in r_aniso.items() if k.startswith('esh_pure') and np.isfinite(v)]

    hc_best = min(hc_vals, key=lambda x: x[1]) if hc_vals else ('none', float('inf'))
    hh_best = min(hh_vals, key=lambda x: x[1]) if hh_vals else ('none', float('inf'))
    esh_best = min(esh_vals, key=lambda x: x[1]) if esh_vals else ('none', float('inf'))

    print(f"\n  Anisotropic Gaussian d=10 kappa=100 (best tau_int):")
    print(f"    NH log-osc:        {nh_best:8.1f}")
    print(f"    NHC(M=3) tanh:     {nhc_best:8.1f}")
    print(f"    Pure ESH:          {esh_best[1]:8.1f}  ({esh_best[0]})")
    print(f"    Hybrid CORRECTED:  {hc_best[1]:8.1f}  ({hc_best[0]})")
    print(f"    Hybrid HEURISTIC:  {hh_best[1]:8.1f}  ({hh_best[0]})")
    print(f"    Ratio NH_logosc/Hybrid_correct = {nh_best/hc_best[1]:.3f}")
    print(f"    Ratio NH_logosc/Hybrid_heurist = {nh_best/hh_best[1]:.3f}")

    # Best ergodicity
    print(f"\n  1D HO Ergodicity (best score per method):")
    for prefix in ['nh_logosc', 'nhc_tanh', 'hybrid_correct', 'hybrid_heuristic']:
        matching = [(k, v['score']) for k, v in r_ho.items() if k.startswith(prefix)]
        if matching:
            best = max(matching, key=lambda x: x[1])
            print(f"    {prefix:25s}: {best[1]:.4f}  ({best[0]})")

    # Best KL
    print(f"\n  2D Double Well KL (best per method):")
    for prefix in ['nh_logosc', 'nhc_tanh', 'hybrid_correct', 'hybrid_heuristic']:
        matching = [(k, v['kl']) for k, v in r_dw.items() if k.startswith(prefix)]
        if matching:
            best = min(matching, key=lambda x: x[1])
            print(f"    {prefix:25s}: {best[1]:.4f}  ({best[0]})")

    metric = nh_best / hc_best[1] if hc_best[1] > 0 and np.isfinite(hc_best[1]) else 0.0
    total = time.time() - t_start
    print(f"\n  Metric (tau_nh_logosc / tau_hybrid_correct) = {metric:.3f}")
    print(f"  Wall time: {total:.0f}s ({total/60:.1f} min)")

    # Save results
    all_results = {
        'aniso_gaussian': {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                           for k, v in r_aniso.items()},
        'harmonic_1d': {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating, np.integer, np.bool_)) else vv
                            for kk, vv in v.items()}
                        for k, v in r_ho.items()},
        'double_well': {k: {'kl': float(v['kl']), 'kls': [float(x) for x in v['kls']]}
                        for k, v in r_dw.items()},
        'summary': {
            'nh_logosc_best_tau': float(nh_best),
            'nhc_best_tau': float(nhc_best),
            'hybrid_correct_best': {'tau': float(hc_best[1]), 'params': hc_best[0]},
            'hybrid_heuristic_best': {'tau': float(hh_best[1]), 'params': hh_best[0]},
            'esh_pure_best': {'tau': float(esh_best[1]), 'params': esh_best[0]},
            'metric': float(metric),
            'wall_time_s': float(total),
        }
    }
    results_path = os.path.join(ORBIT_DIR, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved {results_path}")

    return all_results


if __name__ == '__main__':
    main()
