"""per-mode-coupling-048: Per-mode thermostat coupling for true xi independence.

The key insight from orbit #046: in standard parallel thermostats, ALL xi_i see
the same kinetic energy K(t) = sum(p^2), so they evolve as xi_i(t) = S(t)/Q_i,
giving perfect correlation rho=1.

Fix: couple each xi_i to only a SUBSET of momentum dimensions. Each thermostat
sees a different partial kinetic energy K_i(t) = sum_{d in G_i} p_d^2.

Experiments:
  1. Verify xi independence (correlation matrix)
  2. PSD of total friction Gamma(t)
  3. Mixing comparison (tau_int, crossings)
  4. Cross-correlation heatmap (xi_i vs q_d^2)
"""

import json, os, sys, time, warnings
import numpy as np
from scipy import signal

warnings.filterwarnings("ignore")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================================
# Potentials
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


# ============================================================================
# Friction function
# ============================================================================

def g_tanh(x):
    return np.tanh(x)


# ============================================================================
# Per-mode integrator (THE KEY INNOVATION)
# ============================================================================

def sim_permode(pot, Qs, groups, dt, nsteps, kT=1.0, seed=0, rec=1,
                record_xi=False, record_q2=False):
    """Per-mode parallel thermostat: each xi_i couples to a subset of dimensions.

    groups[i] = array of dimension indices for thermostat i.
    Each xi_i is driven by K_i = sum_{d in groups[i]} p_d^2, not the full K.
    Friction on p_d comes only from xi_{i(d)} where d in groups[i(d)].

    Vectorized: uses dim_to_therm map + np.bincount to avoid Python loops.
    """
    rng = np.random.default_rng(seed)
    dim = pot.dim
    Qs = np.asarray(Qs, float)
    N = len(Qs)

    # Initialize
    q = rng.normal(0, 1.0, size=dim)
    if hasattr(pot, 'kappas'):
        q /= np.sqrt(np.maximum(pot.kappas, 1e-6))
    p = rng.normal(0, np.sqrt(kT), size=dim)
    xi = np.zeros(N)
    h = 0.5 * dt
    gU = pot.gradient(q)

    # Build reverse map: dim_d -> thermostat index (vectorized)
    dim_to_therm = np.zeros(dim, dtype=int)
    grp_sizes = np.zeros(N, dtype=float)
    for i, grp in enumerate(groups):
        for d in grp:
            dim_to_therm[d] = i
        grp_sizes[i] = len(grp)

    # Recording
    nr = nsteps // rec
    qs = np.empty((nr, dim))
    xi_trace = np.empty((nr, N)) if record_xi else None
    q2_trace = np.empty((nr, dim)) if record_q2 else None
    ri = 0

    for s in range(nsteps):
        # Half-step xi: vectorized via bincount
        p2 = p * p
        K_per_therm = np.bincount(dim_to_therm, weights=p2, minlength=N)
        xi += h * (K_per_therm - grp_sizes * kT) / Qs

        # Half-step p: friction from assigned thermostat (vectorized)
        g_xi = np.tanh(xi)  # (N,)
        friction_per_dim = g_xi[dim_to_therm]  # (dim,) -- each dim gets its thermostat's g
        p *= np.clip(np.exp(-friction_per_dim * h), 1e-10, 1e10)

        # Half-step p: kick from potential
        p -= h * gU

        # Full-step q
        q = q + dt * p

        # Recompute gradient
        gU = pot.gradient(q)

        # Half-step p: kick from potential
        p -= h * gU

        # Half-step p: friction (symmetric, vectorized)
        g_xi = np.tanh(xi)
        friction_per_dim = g_xi[dim_to_therm]
        p *= np.clip(np.exp(-friction_per_dim * h), 1e-10, 1e10)

        # Half-step xi
        p2 = p * p
        K_per_therm = np.bincount(dim_to_therm, weights=p2, minlength=N)
        xi += h * (K_per_therm - grp_sizes * kT) / Qs

        # Record
        if (s + 1) % rec == 0 and ri < nr:
            qs[ri] = q
            if record_xi:
                xi_trace[ri] = xi.copy()
            if record_q2:
                q2_trace[ri] = q ** 2
            ri += 1

        if not np.isfinite(p).all():
            qs[ri:] = np.nan
            break

    result = {'qs': qs[:ri]}
    if record_xi:
        result['xi'] = xi_trace[:ri]
    if record_q2:
        result['q2'] = q2_trace[:ri]
    return result


# ============================================================================
# Shared-K parallel thermostat (baseline, from orbit #047)
# ============================================================================

def sim_shared(pot, Qs, dt, nsteps, kT=1.0, seed=0, rec=1,
               record_xi=False, record_q2=False):
    """Standard parallel thermostat: all xi_i share the full K(t)."""
    rng = np.random.default_rng(seed)
    dim = pot.dim
    Qs = np.asarray(Qs, float)
    N = len(Qs)

    q = rng.normal(0, 1.0, size=dim)
    if hasattr(pot, 'kappas'):
        q /= np.sqrt(np.maximum(pot.kappas, 1e-6))
    p = rng.normal(0, np.sqrt(kT), size=dim)
    xi = np.zeros(N)
    h = 0.5 * dt
    gU = pot.gradient(q)

    nr = nsteps // rec
    qs = np.empty((nr, dim))
    xi_trace = np.empty((nr, N)) if record_xi else None
    q2_trace = np.empty((nr, dim)) if record_q2 else None
    ri = 0

    for s in range(nsteps):
        K = float(np.sum(p * p))
        xi += h * (K - dim * kT) / Qs
        gt = np.sum(np.tanh(xi))
        p *= np.clip(np.exp(-gt * h), 1e-10, 1e10)
        p -= h * gU
        q = q + dt * p
        gU = pot.gradient(q)
        p -= h * gU
        gt = np.sum(np.tanh(xi))
        p *= np.clip(np.exp(-gt * h), 1e-10, 1e10)
        K = float(np.sum(p * p))
        xi += h * (K - dim * kT) / Qs

        if (s + 1) % rec == 0 and ri < nr:
            qs[ri] = q
            if record_xi:
                xi_trace[ri] = xi.copy()
            if record_q2:
                q2_trace[ri] = q ** 2
            ri += 1

        if not np.isfinite(p).all():
            qs[ri:] = np.nan
            break

    result = {'qs': qs[:ri]}
    if record_xi:
        result['xi'] = xi_trace[:ri]
    if record_q2:
        result['q2'] = q2_trace[:ri]
    return result


# ============================================================================
# NHC integrator (baseline)
# ============================================================================

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


# ============================================================================
# Single NH integrator
# ============================================================================

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


# ============================================================================
# Metrics
# ============================================================================

def tau_int_q2(qs, kappas):
    """Integrated autocorrelation time for q^2, averaged across dimensions."""
    n, dim = qs.shape
    taus = []
    for d in range(dim):
        x = qs[:, d] ** 2
        x = x - np.mean(x)
        if np.std(x) < 1e-15:
            taus.append(n)
            continue
        # FFT-based autocorrelation
        nfft = 2 ** int(np.ceil(np.log2(2 * n)))
        xf = np.fft.rfft(x, n=nfft)
        acf = np.fft.irfft(xf * np.conj(xf), n=nfft)[:n]
        acf /= acf[0]
        # Integrate until first negative
        cutoff = np.argmax(acf < 0)
        if cutoff == 0:
            cutoff = n
        tau = 1 + 2 * np.sum(acf[1:cutoff])
        taus.append(max(1.0, tau))
    return float(np.mean(taus))


def count_mode_crossings(qs, centers):
    """Count transitions between nearest modes."""
    dists = np.sum((qs[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(dists, axis=1)
    return int(np.sum(nearest[1:] != nearest[:-1]))


def make_groups(dim, n_therm):
    """Split dimensions into n_therm contiguous groups of roughly equal size."""
    groups = []
    base_size = dim // n_therm
    remainder = dim % n_therm
    idx = 0
    for i in range(n_therm):
        size = base_size + (1 if i < remainder else 0)
        groups.append(list(range(idx, idx + size)))
        idx += size
    return groups


# ============================================================================
# Experiment 1: xi independence verification
# ============================================================================

def run_exp1():
    print("=== Experiment 1: xi independence verification ===")
    dim = 10
    N = 5
    Q_val = 100.0
    Qs = np.full(N, Q_val)
    dt = 0.01
    nsteps = 500_000
    rec = 1  # record every step for correlation analysis
    pot = AnisotropicGaussian(dim, kappa=100.0)

    # Sort dimensions by kappa for grouping
    sorted_dims = np.argsort(pot.kappas)
    groups = make_groups(dim, N)
    # Remap groups to sorted dimensions
    groups_sorted = [[int(sorted_dims[d]) for d in grp] for grp in groups]

    print(f"  Groups (by kappa): {groups_sorted}")
    print(f"  Kappas: {pot.kappas}")

    # Per-mode: each xi_i sees its own group
    print("  Running per-mode...")
    t0 = time.time()
    res_pm = sim_permode(pot, Qs, groups_sorted, dt, nsteps, rec=10,
                         record_xi=True, record_q2=True)
    print(f"  Per-mode done in {time.time()-t0:.1f}s")

    # Shared-K: all xi_i see the same K
    print("  Running shared-K...")
    t0 = time.time()
    res_sk = sim_shared(pot, Qs, dt, nsteps, rec=10,
                        record_xi=True, record_q2=True)
    print(f"  Shared-K done in {time.time()-t0:.1f}s")

    # Compute correlation matrices
    xi_pm = res_pm['xi']  # (T, N)
    xi_sk = res_sk['xi']

    corr_pm = np.corrcoef(xi_pm.T)  # (N, N)
    corr_sk = np.corrcoef(xi_sk.T)

    print(f"\n  Per-mode xi correlations:\n{np.array2string(corr_pm, precision=3)}")
    print(f"\n  Shared-K xi correlations:\n{np.array2string(corr_sk, precision=3)}")

    # Off-diagonal statistics
    mask = ~np.eye(N, dtype=bool)
    pm_offdiag = corr_pm[mask]
    sk_offdiag = corr_sk[mask]
    print(f"\n  Per-mode off-diag: mean={np.mean(np.abs(pm_offdiag)):.4f}, max={np.max(np.abs(pm_offdiag)):.4f}")
    print(f"  Shared-K off-diag: mean={np.mean(np.abs(sk_offdiag)):.4f}, max={np.max(np.abs(sk_offdiag)):.4f}")

    return {
        'corr_permode': corr_pm.tolist(),
        'corr_shared': corr_sk.tolist(),
        'pm_offdiag_mean': float(np.mean(np.abs(pm_offdiag))),
        'pm_offdiag_max': float(np.max(np.abs(pm_offdiag))),
        'sk_offdiag_mean': float(np.mean(np.abs(sk_offdiag))),
        'sk_offdiag_max': float(np.max(np.abs(sk_offdiag))),
        'xi_pm': xi_pm,
        'xi_sk': xi_sk,
        'q2_pm': res_pm['q2'],
        'q2_sk': res_sk['q2'],
        'groups': groups_sorted,
    }


# ============================================================================
# Experiment 2: PSD of total friction
# ============================================================================

def run_exp2(xi_pm, xi_sk):
    print("\n=== Experiment 2: PSD of total friction Gamma(t) ===")

    # Compute Gamma(t) = sum_i tanh(xi_i(t))
    gamma_pm = np.sum(np.tanh(xi_pm), axis=1)
    gamma_sk = np.sum(np.tanh(xi_sk), axis=1)

    # PSD via Welch's method
    fs = 1.0 / (0.01 * 10)  # sampling rate (dt * rec)
    nperseg = min(len(gamma_pm) // 4, 8192)

    f_pm, psd_pm = signal.welch(gamma_pm, fs=fs, nperseg=nperseg)
    f_sk, psd_sk = signal.welch(gamma_sk, fs=fs, nperseg=nperseg)

    # Fit PSD slope in log-log (low frequency range)
    mask_pm = (f_pm > 0.01) & (f_pm < 1.0)
    mask_sk = (f_sk > 0.01) & (f_sk < 1.0)

    if np.sum(mask_pm) > 5:
        lf_pm = np.log10(f_pm[mask_pm])
        lp_pm = np.log10(psd_pm[mask_pm])
        slope_pm = np.polyfit(lf_pm, lp_pm, 1)[0]
    else:
        slope_pm = np.nan

    if np.sum(mask_sk) > 5:
        lf_sk = np.log10(f_sk[mask_sk])
        lp_sk = np.log10(psd_sk[mask_sk])
        slope_sk = np.polyfit(lf_sk, lp_sk, 1)[0]
    else:
        slope_sk = np.nan

    print(f"  Per-mode PSD slope: {slope_pm:.3f}")
    print(f"  Shared-K PSD slope: {slope_sk:.3f}")

    return {
        'f_pm': f_pm, 'psd_pm': psd_pm, 'slope_pm': float(slope_pm),
        'f_sk': f_sk, 'psd_sk': psd_sk, 'slope_sk': float(slope_sk),
    }


# ============================================================================
# Experiment 3: Mixing comparison
# ============================================================================

def run_exp3():
    print("\n=== Experiment 3: Mixing comparison ===")
    dim = 10
    N = 5
    dt = 0.01
    nsteps = 400_000
    kT = 1.0
    n_seeds = 5
    seeds = list(range(1000, 1000 + n_seeds))
    rec = 10

    # Targets
    pot_aniso = AnisotropicGaussian(dim, kappa=100.0)
    pot_gmm = GaussianMixtureND(dim, n_modes=5, radius=3.0, sigma=1.0, seed=0)

    # Groups for per-mode
    sorted_dims = np.argsort(pot_aniso.kappas)
    groups_aniso = [[int(sorted_dims[d]) for d in grp] for grp in make_groups(dim, N)]
    groups_gmm = make_groups(dim, N)  # No special ordering for GMM

    # Method configs
    methods = {
        'permode_tanh': lambda pot, s, grps: sim_permode(
            pot, np.full(N, 100.0), grps, dt, nsteps, kT=kT, seed=s, rec=rec)['qs'],
        'shared_tanh': lambda pot, s, grps: sim_shared(
            pot, np.full(N, 100.0), dt, nsteps, kT=kT, seed=s, rec=rec)['qs'],
        'nhc_m3': lambda pot, s, grps: sim_nhc(
            pot, np.full(3, 50.0), dt, nsteps, kT=kT, seed=s, rec=rec),
        'nh': lambda pot, s, grps: sim_nh(
            pot, 100.0, dt, nsteps, kT=kT, seed=s, rec=rec),
    }

    results = {}

    # Anisotropic Gaussian: tau_int
    print("  Anisotropic Gaussian (tau_int):")
    for mname, mfunc in methods.items():
        taus = []
        for s in seeds:
            grps = groups_aniso
            qs = mfunc(pot_aniso, s, grps)
            tau = tau_int_q2(qs, pot_aniso.kappas)
            taus.append(tau)
        med = float(np.median(taus))
        iqr = float(np.percentile(taus, 75) - np.percentile(taus, 25))
        print(f"    {mname}: tau_int = {med:.1f} (IQR={iqr:.1f})")
        results[f'aniso_{mname}_tau'] = taus
        results[f'aniso_{mname}_median'] = med
        results[f'aniso_{mname}_iqr'] = iqr

    # GMM: mode crossings
    print("  GMM (mode crossings):")
    for mname, mfunc in methods.items():
        crossings_list = []
        for s in seeds:
            grps = groups_gmm
            qs = mfunc(pot_gmm, s, grps)
            cx = count_mode_crossings(qs, pot_gmm.centers)
            crossings_list.append(cx)
        med = float(np.median(crossings_list))
        iqr = float(np.percentile(crossings_list, 75) - np.percentile(crossings_list, 25))
        print(f"    {mname}: crossings = {med:.0f} (IQR={iqr:.0f})")
        results[f'gmm_{mname}_crossings'] = crossings_list
        results[f'gmm_{mname}_median'] = med
        results[f'gmm_{mname}_iqr'] = iqr

    return results


# ============================================================================
# Experiment 4: Cross-correlation heatmap
# ============================================================================

def run_exp4(xi_pm, q2_pm, xi_sk, q2_sk, groups):
    print("\n=== Experiment 4: Cross-correlation heatmap ===")
    N = xi_pm.shape[1]
    dim = q2_pm.shape[1]

    # Cross-correlation: |max_lag corr(xi_i, q_d^2)|
    # We use simple Pearson correlation (lag=0) for clarity
    cc_pm = np.zeros((N, dim))
    cc_sk = np.zeros((N, dim))

    for i in range(N):
        for d in range(dim):
            cc_pm[i, d] = abs(np.corrcoef(xi_pm[:, i], q2_pm[:, d])[0, 1])
            cc_sk[i, d] = abs(np.corrcoef(xi_sk[:, i], q2_sk[:, d])[0, 1])

    print(f"  Per-mode cross-corr range: [{cc_pm.min():.4f}, {cc_pm.max():.4f}]")
    print(f"  Shared-K cross-corr range: [{cc_sk.min():.4f}, {cc_sk.max():.4f}]")

    # Check block-diagonal structure for per-mode
    in_group = np.zeros((N, dim), dtype=bool)
    for i, grp in enumerate(groups):
        for d in grp:
            in_group[i, d] = True

    in_group_mean = float(np.mean(cc_pm[in_group]))
    out_group_mean = float(np.mean(cc_pm[~in_group]))
    print(f"  Per-mode: in-group mean={in_group_mean:.4f}, out-group mean={out_group_mean:.4f}")
    print(f"  Ratio: {in_group_mean / max(out_group_mean, 1e-10):.2f}x")

    return {
        'cc_pm': cc_pm.tolist(),
        'cc_sk': cc_sk.tolist(),
        'in_group_mean': in_group_mean,
        'out_group_mean': out_group_mean,
    }


# ============================================================================
# Plotting
# ============================================================================

def plot_all(exp1, exp2, exp3, exp4):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    # Style
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
    })

    # ---- Figure 1: xi correlation matrices ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    corr_pm = np.array(exp1['corr_permode'])
    corr_sk = np.array(exp1['corr_shared'])

    vmin, vmax = -1, 1
    im0 = axes[0].imshow(corr_pm, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='equal')
    axes[0].set_title('(a) Per-mode coupling')
    axes[0].set_xlabel(r'Thermostat $\xi_j$')
    axes[0].set_ylabel(r'Thermostat $\xi_i$')
    for i in range(corr_pm.shape[0]):
        for j in range(corr_pm.shape[1]):
            axes[0].text(j, i, f'{corr_pm[i,j]:.2f}', ha='center', va='center', fontsize=9,
                        color='white' if abs(corr_pm[i,j]) > 0.5 else 'black')

    im1 = axes[1].imshow(corr_sk, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='equal')
    axes[1].set_title('(b) Shared-K (standard parallel)')
    axes[1].set_xlabel(r'Thermostat $\xi_j$')
    axes[1].set_ylabel(r'Thermostat $\xi_i$')
    for i in range(corr_sk.shape[0]):
        for j in range(corr_sk.shape[1]):
            axes[1].text(j, i, f'{corr_sk[i,j]:.2f}', ha='center', va='center', fontsize=9,
                        color='white' if abs(corr_sk[i,j]) > 0.5 else 'black')

    fig.colorbar(im1, ax=axes, shrink=0.8, label='Pearson correlation')
    fig.suptitle(r'$\xi_i$-$\xi_j$ correlation: per-mode vs shared-K (10D aniso, $\kappa$=100)', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig1_xi_correlation.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig1_xi_correlation.png")

    # ---- Figure 2: PSD comparison ----
    fig, ax = plt.subplots(figsize=(8, 6))
    f_pm, psd_pm = exp2['f_pm'], exp2['psd_pm']
    f_sk, psd_sk = exp2['f_sk'], exp2['psd_sk']

    mask_pm = f_pm > 0
    mask_sk = f_sk > 0
    ax.loglog(f_pm[mask_pm], psd_pm[mask_pm], color='#2ca02c', linewidth=1.5,
              label=f'Per-mode (slope={exp2["slope_pm"]:.2f})', alpha=0.9)
    ax.loglog(f_sk[mask_sk], psd_sk[mask_sk], color='#1f77b4', linewidth=1.5,
              label=f'Shared-K (slope={exp2["slope_sk"]:.2f})', alpha=0.9)

    # Reference slopes
    f_ref = np.logspace(-2, 0, 100)
    ax.loglog(f_ref, 0.1 * f_ref**(-1), 'k--', alpha=0.3, label=r'$1/f$ reference')
    ax.loglog(f_ref, 0.1 * f_ref**(-2), 'k:', alpha=0.3, label=r'$1/f^2$ reference')

    ax.set_xlabel('Frequency')
    ax.set_ylabel(r'PSD of $\Gamma(t) = \sum_i g(\xi_i)$')
    ax.set_title(r'Power spectral density of total friction (10D aniso, $\kappa$=100)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig2_psd_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig2_psd_comparison.png")

    # ---- Figure 3: Mixing benchmark ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    methods = ['permode_tanh', 'shared_tanh', 'nhc_m3', 'nh']
    labels = ['Per-mode\ntanh (N=5)', 'Shared-K\ntanh (N=5)', 'NHC\n(M=3)', 'NH']
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd']

    # Aniso tau_int
    ax = axes[0]
    x = np.arange(len(methods))
    medians = [exp3[f'aniso_{m}_median'] for m in methods]
    iqrs = [exp3[f'aniso_{m}_iqr'] for m in methods]
    bars = ax.bar(x, medians, yerr=iqrs, color=colors, capsize=5, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r'$\tau_{\mathrm{int}}(q^2)$', fontsize=13)
    ax.set_title(r'(a) Aniso Gaussian ($\kappa$=100, d=10)', fontsize=13)
    for i, (m, iq) in enumerate(zip(medians, iqrs)):
        ax.text(i, m + iq + 0.15, f'{m:.1f}', ha='center', va='bottom', fontsize=10)

    # GMM crossings
    ax = axes[1]
    medians = [exp3[f'gmm_{m}_median'] for m in methods]
    iqrs = [exp3[f'gmm_{m}_iqr'] for m in methods]
    bars = ax.bar(x, medians, yerr=iqrs, color=colors, capsize=5, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Mode crossings', fontsize=13)
    ax.set_title('(b) GMM (5 modes, d=10)', fontsize=13)
    for i, (m, iq) in enumerate(zip(medians, iqrs)):
        ax.text(i, m + iq + 2, f'{m:.0f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig3_mixing_benchmark.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig3_mixing_benchmark.png")

    # ---- Figure 4: Cross-correlation heatmap ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cc_pm = np.array(exp4['cc_pm'])
    cc_sk = np.array(exp4['cc_sk'])

    vmax = max(cc_pm.max(), cc_sk.max())
    norm = Normalize(vmin=0, vmax=vmax)

    im0 = axes[0].imshow(cc_pm, cmap='YlOrRd', norm=norm, aspect='auto')
    axes[0].set_title('(a) Per-mode coupling')
    axes[0].set_xlabel('Dimension $d$ (sorted by $\\kappa_d$)')
    axes[0].set_ylabel(r'Thermostat $\xi_i$')

    # Draw group boundaries
    groups = exp1['groups']
    for i, grp in enumerate(groups):
        for d in grp:
            axes[0].add_patch(plt.Rectangle((d-0.5, i-0.5), 1, 1, fill=False,
                                            edgecolor='black', linewidth=1.5))

    im1 = axes[1].imshow(cc_sk, cmap='YlOrRd', norm=norm, aspect='auto')
    axes[1].set_title('(b) Shared-K (standard parallel)')
    axes[1].set_xlabel('Dimension $d$ (sorted by $\\kappa_d$)')
    axes[1].set_ylabel(r'Thermostat $\xi_i$')

    fig.colorbar(im1, ax=axes, shrink=0.8, label=r'$|\rho(\xi_i, q_d^2)|$')
    fig.suptitle(r'Cross-correlation $|\rho(\xi_i, q_d^2)|$: block-diagonal vs uniform', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig4_cross_correlation.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig4_cross_correlation.png")


# ============================================================================
# Experiment 5: Massive thermostatting (N=dim) + spread Q
# ============================================================================

def run_exp5():
    """Test whether N=dim (one thermostat per dimension) gives better independence
    and mixing than N=5 groups. Also test spread Q values."""
    print("\n=== Experiment 5: Massive thermostatting + Q spread ===")
    dim = 10
    dt = 0.01
    nsteps = 500_000
    kT = 1.0
    n_seeds = 5
    seeds = list(range(2000, 2000 + n_seeds))
    rec = 10

    pot = AnisotropicGaussian(dim, kappa=100.0)
    sorted_dims = np.argsort(pot.kappas)

    # Configurations to test
    configs = {}

    # 1. Massive: N=10, each xi gets exactly 1 dimension, Q=100
    groups_massive = [[int(sorted_dims[d])] for d in range(dim)]
    configs['massive_Q100'] = {
        'groups': groups_massive, 'Qs': np.full(dim, 100.0), 'N': dim
    }

    # 2. Massive with spread Q: Q_i = 100 * kappa_i^0.5 (proportional to mode freq)
    kappas_sorted = pot.kappas[sorted_dims]
    Qs_spread = 100.0 * np.sqrt(kappas_sorted / kappas_sorted[0])
    configs['massive_Qspread'] = {
        'groups': groups_massive, 'Qs': Qs_spread, 'N': dim
    }

    # 3. Groups of 2 (N=5), Q=100 (repeat from exp1 for comparison)
    groups_5 = [[int(sorted_dims[d]) for d in grp] for grp in make_groups(dim, 5)]
    configs['groups5_Q100'] = {
        'groups': groups_5, 'Qs': np.full(5, 100.0), 'N': 5
    }

    # 4. Groups of 2 (N=5), spread Q
    Qs_5_spread = np.array([100, 200, 400, 700, 1000], dtype=float)
    configs['groups5_Qspread'] = {
        'groups': groups_5, 'Qs': Qs_5_spread, 'N': 5
    }

    results = {}

    # For each config, measure: xi correlation + tau_int
    for cname, cfg in configs.items():
        print(f"\n  Config: {cname} (N={cfg['N']})")

        # Xi correlation (single seed, long run)
        res = sim_permode(pot, cfg['Qs'], cfg['groups'], dt, nsteps, seed=42,
                          rec=1, record_xi=True)
        xi_trace = res['xi']
        corr = np.corrcoef(xi_trace.T)
        N = cfg['N']
        mask = ~np.eye(N, dtype=bool)
        offdiag = corr[mask]
        mean_abs_rho = float(np.mean(np.abs(offdiag)))
        max_abs_rho = float(np.max(np.abs(offdiag)))
        print(f"    xi corr: mean|rho|={mean_abs_rho:.4f}, max|rho|={max_abs_rho:.4f}")

        # tau_int (multiple seeds)
        taus = []
        for s in seeds:
            res_t = sim_permode(pot, cfg['Qs'], cfg['groups'], dt, 400_000,
                                seed=s, rec=rec)
            tau = tau_int_q2(res_t['qs'], pot.kappas)
            taus.append(tau)
        med_tau = float(np.median(taus))
        iqr_tau = float(np.percentile(taus, 75) - np.percentile(taus, 25))
        print(f"    tau_int: {med_tau:.1f} (IQR={iqr_tau:.1f})")

        results[cname] = {
            'mean_abs_rho': mean_abs_rho,
            'max_abs_rho': max_abs_rho,
            'tau_int_median': med_tau,
            'tau_int_iqr': iqr_tau,
            'taus': taus,
            'corr': corr.tolist(),
        }

    # Also add shared-K baseline
    print(f"\n  Baseline: shared_K (N=5)")
    taus_sk = []
    for s in seeds:
        qs_sk = sim_shared(pot, np.full(5, 100.0), dt, 400_000, seed=s, rec=rec)['qs']
        taus_sk.append(tau_int_q2(qs_sk, pot.kappas))
    med_sk = float(np.median(taus_sk))
    print(f"    tau_int: {med_sk:.1f}")
    results['shared_K'] = {'tau_int_median': med_sk, 'taus': taus_sk,
                           'mean_abs_rho': 1.0, 'max_abs_rho': 1.0}

    return results


def plot_exp5(exp5):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14,
        'xtick.labelsize': 11, 'ytick.labelsize': 11,
    })

    configs = ['massive_Q100', 'massive_Qspread', 'groups5_Q100', 'groups5_Qspread', 'shared_K']
    labels = ['Massive\nN=10, Q=100', 'Massive\nN=10, Q-spread', 'Groups\nN=5, Q=100',
              'Groups\nN=5, Q-spread', 'Shared-K\nN=5, Q=100']
    colors = ['#2ca02c', '#98df8a', '#1f77b4', '#aec7e8', '#d62728']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel (a): mean |rho| off-diagonal
    ax = axes[0]
    rhos = [exp5[c]['mean_abs_rho'] for c in configs]
    x = np.arange(len(configs))
    bars = ax.bar(x, rhos, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r'Mean $|\rho(\xi_i, \xi_j)|$ off-diagonal')
    ax.set_title(r'(a) $\xi$ independence (lower = better)')
    for i, r in enumerate(rhos):
        ax.text(i, r + 0.02, f'{r:.3f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylim(0, 1.15)

    # Panel (b): tau_int
    ax = axes[1]
    taus = [exp5[c]['tau_int_median'] for c in configs]
    iqrs = [exp5[c].get('tau_int_iqr', 0) for c in configs]
    bars = ax.bar(x, taus, yerr=iqrs, color=colors, capsize=4,
                  edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r'$\tau_{\mathrm{int}}(q^2)$')
    ax.set_title(r'(b) Mixing (lower = better)')
    ax.set_ylim(0, None)
    for i, (t, iq) in enumerate(zip(taus, iqrs)):
        ax.text(i, t + iq + 0.5, f'{t:.1f}', ha='center', va='bottom', fontsize=10)

    fig.suptitle('Massive thermostatting vs grouped per-mode coupling (10D aniso, kappa=100)', fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig5_massive_vs_grouped.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig5_massive_vs_grouped.png")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    import sys
    t_total = time.time()

    # Check if running only exp5
    if len(sys.argv) > 1 and sys.argv[1] == 'exp5':
        exp5 = run_exp5()
        with open(os.path.join(OUT_DIR, 'results_exp5.json'), 'w') as f:
            json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'corr'}
                       for k, v in exp5.items()}, f, indent=2,
                      default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        print("\n=== Plotting Exp 5 ===")
        plot_exp5(exp5)
        print(f"\nExp 5 runtime: {time.time()-t_total:.1f}s")
        sys.exit(0)

    # Exp 1
    exp1 = run_exp1()

    # Exp 2 (uses xi traces from exp 1)
    exp2 = run_exp2(exp1.pop('xi_pm'), exp1.pop('xi_sk'))

    # Extract traces for exp 4 before saving
    q2_pm = exp1.pop('q2_pm')
    q2_sk = exp1.pop('q2_sk')

    # Exp 3
    exp3 = run_exp3()

    # Exp 4 (uses traces from exp 1)
    print("\n  Re-running per-mode + shared for cross-correlation...")
    dim = 10; N = 5; dt = 0.01; nsteps = 200_000
    pot = AnisotropicGaussian(dim, kappa=100.0)
    sorted_dims = np.argsort(pot.kappas)
    groups_sorted = [[int(sorted_dims[d]) for d in grp] for grp in make_groups(dim, N)]
    Qs = np.full(N, 100.0)

    res_pm2 = sim_permode(pot, Qs, groups_sorted, dt, nsteps, rec=10,
                          record_xi=True, record_q2=True)
    res_sk2 = sim_shared(pot, Qs, dt, nsteps, rec=10,
                         record_xi=True, record_q2=True)

    exp4 = run_exp4(res_pm2['xi'], res_pm2['q2'], res_sk2['xi'], res_sk2['q2'], groups_sorted)
    exp1['groups'] = groups_sorted

    # Exp 5: massive + spread Q
    exp5 = run_exp5()

    # Save results
    results = {
        'exp1_correlation': {k: v for k, v in exp1.items()},
        'exp2_psd': {k: v for k, v in exp2.items() if not isinstance(v, np.ndarray)},
        'exp3_mixing': {k: v for k, v in exp3.items()},
        'exp4_crosscorr': exp4,
        'exp5_massive': {k: {kk: vv for kk, vv in v.items() if kk != 'corr'}
                         for k, v in exp5.items()},
    }

    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    print(f"\nResults saved to {OUT_DIR}/results.json")

    # Plot
    print("\n=== Plotting ===")
    plot_all(exp1, exp2, exp3, exp4)
    plot_exp5(exp5)

    print(f"\nTotal runtime: {time.time()-t_total:.1f}s")
