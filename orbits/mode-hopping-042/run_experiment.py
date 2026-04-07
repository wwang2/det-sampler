"""mode-hopping-042: Systematic mode-hopping benchmark.

Four experiments comparing multi-scale log-osc thermostat vs NHC vs Langevin
across barrier heights, mode counts, and dimensionalities.

Uses corrected Q range from orbit #040: Q_opt = 2.34 * omega^(-1.55)
"""
import json
import os
import sys
import time
import numpy as np
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Potentials
# ---------------------------------------------------------------------------

class DoubleWell2D:
    name = "double_well_2d"
    dim = 2
    def __init__(self, barrier_height=1.0):
        self.a = barrier_height
    def energy(self, q):
        x, y = q[0], q[1]
        return self.a * (x**2 - 1)**2 + 0.5 * y**2
    def gradient(self, q):
        x, y = q[0], q[1]
        return np.array([4.0 * self.a * x * (x**2 - 1), y])


class RingGMM:
    name = "ring_gmm"
    dim = 2
    def __init__(self, n_modes=5, radius=3.0, sigma=0.5):
        self.n_modes = n_modes
        self.radius = radius
        self.sigma = sigma
        angles = np.linspace(0, 2 * np.pi, n_modes, endpoint=False)
        self.centers = np.column_stack([radius * np.cos(angles),
                                        radius * np.sin(angles)])
        self.weights = np.ones(n_modes) / n_modes
    def _densities(self, q):
        diffs = self.centers - q[np.newaxis, :]
        exponents = -0.5 * np.sum(diffs**2, axis=1) / self.sigma**2
        return self.weights * np.exp(exponents)
    def energy(self, q):
        d = self._densities(q)
        total = np.sum(d)
        return -np.log(total) if total > 1e-300 else 700.0
    def gradient(self, q):
        d = self._densities(q)
        total = np.sum(d)
        if total < 1e-300:
            return np.zeros(2)
        diffs = self.centers - q[np.newaxis, :]
        return -np.sum(d[:, np.newaxis] * diffs / self.sigma**2, axis=0) / total


class HighDimGMM:
    name = "high_dim_gmm"
    def __init__(self, dim, centers, sigma=0.5):
        self.dim = dim
        self.centers = np.asarray(centers, dtype=float)
        self.n_modes = len(centers)
        self.sigma = sigma
        self.weights = np.ones(self.n_modes) / self.n_modes
    def _densities(self, q):
        diffs = self.centers - q[np.newaxis, :]
        exponents = -0.5 * np.sum(diffs**2, axis=1) / self.sigma**2
        return self.weights * np.exp(exponents)
    def energy(self, q):
        d = self._densities(q)
        total = np.sum(d)
        return -np.log(total) if total > 1e-300 else 700.0
    def gradient(self, q):
        d = self._densities(q)
        total = np.sum(d)
        if total < 1e-300:
            return np.zeros(self.dim)
        diffs = self.centers - q[np.newaxis, :]
        return -np.sum(d[:, np.newaxis] * diffs / self.sigma**2, axis=0) / total


# ---------------------------------------------------------------------------
# Integrators
# ---------------------------------------------------------------------------

def g_func(xi):
    return 2.0 * xi / (1.0 + xi * xi)


def simulate_multiscale(potential, Qs, dt, n_steps, kT=1.0, mass=1.0, seed=0,
                        record_every=1):
    rng = np.random.default_rng(seed)
    dim = potential.dim
    Qs = np.asarray(Qs, dtype=float)
    q = rng.normal(0, 1.0, size=dim)
    p = rng.normal(0, np.sqrt(mass * kT), size=dim)
    xi = np.zeros(len(Qs))
    half = 0.5 * dt
    grad_U = potential.gradient(q)
    n_rec = n_steps // record_every
    qs_rec = np.empty((n_rec, dim))
    rec_i = 0
    for step in range(n_steps):
        K = float(np.dot(p, p)) / mass
        xi += half * (K - dim * kT) / Qs
        gtot = float(np.sum(g_func(xi)))
        p *= np.exp(-gtot * half)
        p -= half * grad_U
        q += dt * p / mass
        grad_U = potential.gradient(q)
        p -= half * grad_U
        gtot = float(np.sum(g_func(xi)))
        p *= np.exp(-gtot * half)
        K = float(np.dot(p, p)) / mass
        xi += half * (K - dim * kT) / Qs
        if (step + 1) % record_every == 0 and rec_i < n_rec:
            qs_rec[rec_i] = q
            rec_i += 1
        if not np.isfinite(q[0]):
            break
    return qs_rec[:rec_i]


def simulate_nhc(potential, Qs, dt, n_steps, kT=1.0, mass=1.0, seed=0,
                 record_every=1):
    rng = np.random.default_rng(seed)
    dim = potential.dim
    Qs = np.asarray(Qs, dtype=float)
    M = len(Qs)
    q = rng.normal(0, 1.0, size=dim)
    p = rng.normal(0, np.sqrt(mass * kT), size=dim)
    xi = np.zeros(M)
    half = 0.5 * dt
    grad_U = potential.gradient(q)
    n_rec = n_steps // record_every
    qs_rec = np.empty((n_rec, dim))
    rec_i = 0

    def chain_dxi(p_val, xi_val):
        d = np.zeros(M)
        K = float(np.dot(p_val, p_val)) / mass
        d[0] = (K - dim * kT) / Qs[0]
        if M > 1:
            d[0] -= xi_val[1] * xi_val[0]
        for i in range(1, M):
            G = Qs[i - 1] * xi_val[i - 1]**2 - kT
            d[i] = G / Qs[i]
            if i < M - 1:
                d[i] -= xi_val[i + 1] * xi_val[i]
        return d

    for step in range(n_steps):
        xi += half * chain_dxi(p, xi)
        friction = np.clip(xi[0] * half, -20, 20)  # clip to prevent overflow
        p *= np.exp(-friction)
        p -= half * grad_U
        q += dt * p / mass
        grad_U = potential.gradient(q)
        p -= half * grad_U
        friction = np.clip(xi[0] * half, -20, 20)
        p *= np.exp(-friction)
        xi += half * chain_dxi(p, xi)
        if (step + 1) % record_every == 0 and rec_i < n_rec:
            qs_rec[rec_i] = q
            rec_i += 1
        if not np.isfinite(q[0]):
            break
    return qs_rec[:rec_i]


def simulate_langevin(potential, gamma, dt, n_steps, kT=1.0, mass=1.0, seed=0,
                      record_every=1):
    rng = np.random.default_rng(seed)
    dim = potential.dim
    q = rng.normal(0, 1.0, size=dim)
    p = rng.normal(0, np.sqrt(mass * kT), size=dim)
    half = 0.5 * dt
    grad_U = potential.gradient(q)
    n_rec = n_steps // record_every
    qs_rec = np.empty((n_rec, dim))
    rec_i = 0
    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt((1 - c1**2) * mass * kT)
    for step in range(n_steps):
        p -= half * grad_U
        q += half * p / mass
        p = c1 * p + c2 * rng.normal(size=dim)
        q += half * p / mass
        grad_U = potential.gradient(q)
        p -= half * grad_U
        if (step + 1) % record_every == 0 and rec_i < n_rec:
            qs_rec[rec_i] = q
            rec_i += 1
        if not np.isfinite(q[0]):
            break
    return qs_rec[:rec_i]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def count_barrier_crossings(traj):
    x = traj[:, 0]
    signs = np.sign(x)
    return int(np.sum(np.abs(np.diff(signs)) > 0))

def count_mode_crossings(traj, potential):
    if len(traj) == 0:
        return 0
    centers = potential.centers
    d2 = np.sum((traj[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    assign = np.argmin(d2, axis=1)
    return int(np.sum(assign[1:] != assign[:-1]))

def modes_visited(traj, potential, threshold_factor=2.0):
    centers = potential.centers
    threshold = threshold_factor * potential.sigma
    d2 = np.sum((traj[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    min_d = np.sqrt(np.min(d2, axis=0))
    return int(np.sum(min_d < threshold)) / len(centers)

def autocorr_time(x, c=5.0):
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    if n < 16 or np.std(x) < 1e-12:
        return float(n)
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f))[:n].real
    acf /= acf[0]
    tau = 1.0
    for k in range(1, n // 4):
        tau += 2.0 * acf[k]
        if k >= c * tau:
            break
    return float(max(tau, 1.0))


# ---------------------------------------------------------------------------
# Q range from corrected formula (orbit #040)
# ---------------------------------------------------------------------------

def corrected_Q_range(omega_min, omega_max, N):
    Q_max = 2.34 * omega_min**(-1.55)
    Q_min = 2.34 * omega_max**(-1.55)
    return np.exp(np.linspace(np.log(max(Q_min, 0.01)), np.log(Q_max), N))


# ---------------------------------------------------------------------------
# Worker + parallel
# ---------------------------------------------------------------------------

def _run_one_seed(args):
    sampler_type, sampler_kwargs, potential_dict, seed, metrics_list = args
    pot = _make_potential(potential_dict)
    kw = dict(sampler_kwargs)
    kw["seed"] = seed
    kw["potential"] = pot
    if sampler_type == "multiscale":
        traj = simulate_multiscale(**kw)
    elif sampler_type == "nhc":
        traj = simulate_nhc(**kw)
    elif sampler_type == "langevin":
        traj = simulate_langevin(**kw)
    else:
        raise ValueError(sampler_type)
    result = {"seed": seed}
    for m in metrics_list:
        if m == "barrier_crossings":
            result[m] = count_barrier_crossings(traj) if len(traj) > 1 else 0
        elif m == "mode_crossings":
            result[m] = count_mode_crossings(traj, pot) if len(traj) > 1 else 0
        elif m == "modes_visited":
            result[m] = modes_visited(traj, pot) if len(traj) > 1 else 0.0
        elif m == "tau_int":
            if len(traj) > 64:
                taus = [autocorr_time(traj[:, d]**2) for d in range(traj.shape[1])]
                result[m] = float(np.mean(taus))
            else:
                result[m] = 1e6
    return result

def _make_potential(d):
    cls = d["class"]
    if cls == "DoubleWell2D":
        return DoubleWell2D(barrier_height=d["barrier_height"])
    elif cls == "RingGMM":
        return RingGMM(n_modes=d["n_modes"], radius=d["radius"], sigma=d["sigma"])
    elif cls == "HighDimGMM":
        return HighDimGMM(dim=d["dim"], centers=d["centers"], sigma=d["sigma"])
    raise ValueError(cls)

def run_seeds(sampler_type, sampler_kwargs, potential_dict, seeds, metrics_list):
    n_workers = min(len(seeds), cpu_count())
    tasks = [(sampler_type, sampler_kwargs, potential_dict, s, metrics_list)
             for s in seeds]
    with Pool(n_workers) as pool:
        results = pool.map(_run_one_seed, tasks)
    agg = {}
    for m in metrics_list:
        vals = [r[m] for r in results]
        agg[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                  "values": [float(v) for v in vals]}
    return agg


# ---------------------------------------------------------------------------
# Config: reduced for speed
# ---------------------------------------------------------------------------
N_STEPS = 200_000
RECORD_EVERY = 4
N_SEEDS = 5
N_THERMO = 5
SEEDS_EXP1 = list(range(100, 100 + N_SEEDS))
SEEDS_EXP2 = list(range(200, 200 + N_SEEDS))
SEEDS_EXP3 = list(range(300, 300 + N_SEEDS))


# ---------------------------------------------------------------------------
# Experiment 1: Barrier height sweep
# ---------------------------------------------------------------------------

def exp1_barrier_sweep():
    print("=" * 60)
    print("EXP 1: Barrier height sweep")
    print("=" * 60, flush=True)

    barriers = [0.5, 1.0, 2.0, 4.0, 8.0]
    results = {}

    for a in barriers:
        t0 = time.time()
        print(f"\n  barrier_height = {a}", flush=True)
        pot_dict = {"class": "DoubleWell2D", "barrier_height": a}

        omega_max = max(np.sqrt(8 * a), 1.0)
        omega_min = min(1.0, np.sqrt(2 * a))
        Qs_par = corrected_Q_range(omega_min, omega_max, N_THERMO)
        dt = min(0.02, 0.15 / omega_max)

        par_kwargs = {"Qs": Qs_par.tolist(), "dt": dt,
                      "n_steps": N_STEPS, "record_every": RECORD_EVERY}
        par_res = run_seeds("multiscale", par_kwargs, pot_dict,
                            SEEDS_EXP1, ["barrier_crossings"])

        # NHC: 3 Q_ref values
        Q_refs = [0.1, 1.0, 5.0]
        best_nhc = {"barrier_crossings": {"mean": 0}}
        best_Q_ref = None
        for Q_ref in Q_refs:
            Qs_nhc = np.ones(N_THERMO) * Q_ref
            nhc_kwargs = {"Qs": Qs_nhc.tolist(), "dt": dt,
                          "n_steps": N_STEPS, "record_every": RECORD_EVERY}
            nhc_res = run_seeds("nhc", nhc_kwargs, pot_dict,
                                SEEDS_EXP1, ["barrier_crossings"])
            if nhc_res["barrier_crossings"]["mean"] > best_nhc["barrier_crossings"]["mean"]:
                best_nhc = nhc_res
                best_Q_ref = Q_ref

        # Langevin: 3 gamma values
        gammas = [0.1, 1.0, 10.0]
        best_lang = {"barrier_crossings": {"mean": 0}}
        best_gamma = None
        for gamma in gammas:
            lang_kwargs = {"gamma": gamma, "dt": dt,
                           "n_steps": N_STEPS, "record_every": RECORD_EVERY}
            lang_res = run_seeds("langevin", lang_kwargs, pot_dict,
                                 SEEDS_EXP1, ["barrier_crossings"])
            if lang_res["barrier_crossings"]["mean"] > best_lang["barrier_crossings"]["mean"]:
                best_lang = lang_res
                best_gamma = gamma

        results[str(a)] = {
            "barrier_height": a, "dt": dt,
            "Qs_parallel": Qs_par.tolist(),
            "parallel": par_res,
            "nhc_best_Q_ref": best_Q_ref, "nhc": best_nhc,
            "langevin_best_gamma": best_gamma, "langevin": best_lang,
        }
        elapsed = time.time() - t0
        print(f"    parallel: {par_res['barrier_crossings']['mean']:.1f} +/- {par_res['barrier_crossings']['std']:.1f}", flush=True)
        print(f"    NHC(Q={best_Q_ref}): {best_nhc['barrier_crossings']['mean']:.1f} +/- {best_nhc['barrier_crossings']['std']:.1f}", flush=True)
        print(f"    Langevin(g={best_gamma}): {best_lang['barrier_crossings']['mean']:.1f} +/- {best_lang['barrier_crossings']['std']:.1f}", flush=True)
        print(f"    [{elapsed:.1f}s]", flush=True)

    return results


# ---------------------------------------------------------------------------
# Experiment 2: Mode count sweep
# ---------------------------------------------------------------------------

def exp2_mode_count_sweep():
    print("\n" + "=" * 60)
    print("EXP 2: Mode count sweep")
    print("=" * 60, flush=True)

    mode_counts = [3, 5, 10, 20]
    radius = 3.0
    sigma = 0.5
    dt = 0.02
    omega_min, omega_max = 2.0, 6.0
    Qs_par = corrected_Q_range(omega_min, omega_max, N_THERMO)
    metrics = ["mode_crossings", "modes_visited"]

    results = {}
    for n_modes in mode_counts:
        t0 = time.time()
        print(f"\n  n_modes = {n_modes}", flush=True)
        pot_dict = {"class": "RingGMM", "n_modes": n_modes,
                    "radius": radius, "sigma": sigma}

        par_kwargs = {"Qs": Qs_par.tolist(), "dt": dt,
                      "n_steps": N_STEPS, "record_every": RECORD_EVERY}
        par_res = run_seeds("multiscale", par_kwargs, pot_dict, SEEDS_EXP2, metrics)

        Q_refs = [0.1, 1.0, 5.0]
        best_nhc = {"mode_crossings": {"mean": 0}, "modes_visited": {"mean": 0}}
        best_Q_ref = None
        for Q_ref in Q_refs:
            nhc_kwargs = {"Qs": (np.ones(N_THERMO) * Q_ref).tolist(), "dt": dt,
                          "n_steps": N_STEPS, "record_every": RECORD_EVERY}
            nhc_res = run_seeds("nhc", nhc_kwargs, pot_dict, SEEDS_EXP2, metrics)
            if nhc_res["mode_crossings"]["mean"] > best_nhc["mode_crossings"]["mean"]:
                best_nhc = nhc_res
                best_Q_ref = Q_ref

        gammas = [0.1, 1.0, 10.0]
        best_lang = {"mode_crossings": {"mean": 0}, "modes_visited": {"mean": 0}}
        best_gamma = None
        for gamma in gammas:
            lang_kwargs = {"gamma": gamma, "dt": dt,
                           "n_steps": N_STEPS, "record_every": RECORD_EVERY}
            lang_res = run_seeds("langevin", lang_kwargs, pot_dict, SEEDS_EXP2, metrics)
            if lang_res["mode_crossings"]["mean"] > best_lang["mode_crossings"]["mean"]:
                best_lang = lang_res
                best_gamma = gamma

        results[str(n_modes)] = {
            "n_modes": n_modes, "Qs_parallel": Qs_par.tolist(),
            "parallel": par_res,
            "nhc_best_Q_ref": best_Q_ref, "nhc": best_nhc,
            "langevin_best_gamma": best_gamma, "langevin": best_lang,
        }
        elapsed = time.time() - t0
        print(f"    parallel: cross={par_res['mode_crossings']['mean']:.1f}, vis={par_res['modes_visited']['mean']:.2f}", flush=True)
        print(f"    NHC(Q={best_Q_ref}): cross={best_nhc['mode_crossings']['mean']:.1f}, vis={best_nhc['modes_visited']['mean']:.2f}", flush=True)
        print(f"    Langevin(g={best_gamma}): cross={best_lang['mode_crossings']['mean']:.1f}, vis={best_lang['modes_visited']['mean']:.2f}", flush=True)
        print(f"    [{elapsed:.1f}s]", flush=True)

    return results


# ---------------------------------------------------------------------------
# Experiment 3: Higher-dimensional
# ---------------------------------------------------------------------------

def exp3_high_dim():
    print("\n" + "=" * 60)
    print("EXP 3: Higher-dimensional multi-modal")
    print("=" * 60, flush=True)

    dims = [5, 10]
    n_modes = 5
    radius = 3.0
    sigma = 0.5
    omega_min, omega_max = 2.0, 6.0
    Qs_par = corrected_Q_range(omega_min, omega_max, N_THERMO)
    metrics = ["mode_crossings", "modes_visited", "tau_int"]

    results = {}
    for dim in dims:
        t0 = time.time()
        print(f"\n  dim = {dim}", flush=True)
        rng = np.random.default_rng(777)
        raw = rng.normal(size=(n_modes, dim))
        centers = (raw / np.linalg.norm(raw, axis=1, keepdims=True)) * radius
        pot_dict = {"class": "HighDimGMM", "dim": dim,
                    "centers": centers.tolist(), "sigma": sigma}
        dt = min(0.02, 0.1 / np.sqrt(dim))

        par_kwargs = {"Qs": Qs_par.tolist(), "dt": dt,
                      "n_steps": N_STEPS, "record_every": RECORD_EVERY}
        par_res = run_seeds("multiscale", par_kwargs, pot_dict, SEEDS_EXP3, metrics)

        Q_refs = [0.1, 1.0, 5.0]
        best_nhc = {"mode_crossings": {"mean": 0}, "modes_visited": {"mean": 0}, "tau_int": {"mean": 1e6}}
        best_Q_ref = None
        for Q_ref in Q_refs:
            nhc_kwargs = {"Qs": (np.ones(N_THERMO) * Q_ref).tolist(), "dt": dt,
                          "n_steps": N_STEPS, "record_every": RECORD_EVERY}
            nhc_res = run_seeds("nhc", nhc_kwargs, pot_dict, SEEDS_EXP3, metrics)
            if nhc_res["mode_crossings"]["mean"] > best_nhc["mode_crossings"]["mean"]:
                best_nhc = nhc_res
                best_Q_ref = Q_ref

        results[str(dim)] = {
            "dim": dim, "n_modes": n_modes, "centers": centers.tolist(),
            "Qs_parallel": Qs_par.tolist(),
            "parallel": par_res,
            "nhc_best_Q_ref": best_Q_ref, "nhc": best_nhc,
        }
        elapsed = time.time() - t0
        print(f"    parallel: cross={par_res['mode_crossings']['mean']:.1f}, vis={par_res['modes_visited']['mean']:.2f}", flush=True)
        print(f"    NHC(Q={best_Q_ref}): cross={best_nhc['mode_crossings']['mean']:.1f}, vis={best_nhc['modes_visited']['mean']:.2f}", flush=True)
        print(f"    [{elapsed:.1f}s]", flush=True)

    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(HERE, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    COLOR_NHC = "#ff7f0e"
    COLOR_PAR = "#2ca02c"
    COLOR_LANG = "#9467bd"
    FS_L = 14; FS_T = 12; FS_TT = 16

    # Fig 1: Barrier sweep
    exp1 = results["exp1"]
    barriers = sorted([float(k) for k in exp1.keys()])
    par_m = [exp1[str(a)]["parallel"]["barrier_crossings"]["mean"] for a in barriers]
    par_s = [exp1[str(a)]["parallel"]["barrier_crossings"]["std"] for a in barriers]
    nhc_m = [exp1[str(a)]["nhc"]["barrier_crossings"]["mean"] for a in barriers]
    nhc_s = [exp1[str(a)]["nhc"]["barrier_crossings"]["std"] for a in barriers]
    lang_m = [exp1[str(a)]["langevin"]["barrier_crossings"]["mean"] for a in barriers]
    lang_s = [exp1[str(a)]["langevin"]["barrier_crossings"]["std"] for a in barriers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.errorbar(barriers, par_m, yerr=par_s, marker="o", capsize=4,
                 color=COLOR_PAR, label="Multi-scale log-osc (N=5)", linewidth=2)
    ax1.errorbar(barriers, nhc_m, yerr=nhc_s, marker="s", capsize=4,
                 color=COLOR_NHC, label="NHC (M=5, best Q)", linewidth=2)
    ax1.errorbar(barriers, lang_m, yerr=lang_s, marker="^", capsize=4,
                 color=COLOR_LANG, label="Langevin (best gamma)", linewidth=2)
    ax1.set_xlabel("Barrier height a", fontsize=FS_L)
    ax1.set_ylabel("Barrier crossings (200k steps)", fontsize=FS_L)
    ax1.set_title("(a) Barrier crossings vs height", fontsize=FS_TT)
    ax1.legend(fontsize=11); ax1.tick_params(labelsize=FS_T)
    ax1.set_yscale("log"); ax1.set_xscale("log")

    ratio_nhc = [p / max(n, 1) for p, n in zip(par_m, nhc_m)]
    ratio_lang = [p / max(l, 1) for p, l in zip(par_m, lang_m)]
    ax2.plot(barriers, ratio_nhc, marker="s", color=COLOR_NHC, label="vs NHC", linewidth=2)
    ax2.plot(barriers, ratio_lang, marker="^", color=COLOR_LANG, label="vs Langevin", linewidth=2)
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel("Barrier height a", fontsize=FS_L)
    ax2.set_ylabel("Crossings ratio (ours / baseline)", fontsize=FS_L)
    ax2.set_title("(b) Advantage ratio", fontsize=FS_TT)
    ax2.legend(fontsize=11); ax2.tick_params(labelsize=FS_T); ax2.set_xscale("log")
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig1_barrier_sweep.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig1_barrier_sweep.png", flush=True)

    # Fig 2: Mode count sweep
    exp2 = results["exp2"]
    mcs = sorted([int(k) for k in exp2.keys()])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for label, color, marker, key in [("Multi-scale log-osc", COLOR_PAR, "o", "parallel"),
                                       ("NHC (best Q)", COLOR_NHC, "s", "nhc"),
                                       ("Langevin (best gamma)", COLOR_LANG, "^", "langevin")]:
        cr = [exp2[str(n)][key]["mode_crossings"]["mean"] for n in mcs]
        cr_s = [exp2[str(n)][key]["mode_crossings"]["std"] for n in mcs]
        ax1.errorbar(mcs, cr, yerr=cr_s, marker=marker, capsize=4, color=color, label=label, linewidth=2)
        vis = [exp2[str(n)][key]["modes_visited"]["mean"] for n in mcs]
        vis_s = [exp2[str(n)][key]["modes_visited"]["std"] for n in mcs]
        ax2.errorbar(mcs, vis, yerr=vis_s, marker=marker, capsize=4, color=color, label=label, linewidth=2)
    ax1.set_xlabel("Number of modes", fontsize=FS_L)
    ax1.set_ylabel("Mode crossings (200k steps)", fontsize=FS_L)
    ax1.set_title("(a) Total mode crossings", fontsize=FS_TT)
    ax1.legend(fontsize=11); ax1.tick_params(labelsize=FS_T)
    ax2.set_xlabel("Number of modes", fontsize=FS_L)
    ax2.set_ylabel("Fraction of modes visited", fontsize=FS_L)
    ax2.set_title("(b) Mode coverage", fontsize=FS_TT)
    ax2.legend(fontsize=11); ax2.tick_params(labelsize=FS_T); ax2.set_ylim(0, 1.1)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig2_mode_count_sweep.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig2_mode_count_sweep.png", flush=True)

    # Fig 3: High-dim
    exp3 = results["exp3"]
    dims = sorted([int(k) for k in exp3.keys()])
    fig, axes = plt.subplots(1, len(dims), figsize=(6 * len(dims), 5), sharey=True)
    if len(dims) == 1:
        axes = [axes]
    for i, dim in enumerate(dims):
        ax = axes[i]
        data = exp3[str(dim)]
        means = [data["parallel"]["mode_crossings"]["mean"],
                 data["nhc"]["mode_crossings"]["mean"]]
        stds = [data["parallel"]["mode_crossings"]["std"],
                data["nhc"]["mode_crossings"]["std"]]
        x = np.arange(2)
        ax.bar(x, means, width=0.5, yerr=stds, capsize=5,
               color=[COLOR_PAR, COLOR_NHC], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(["Log-osc\n(N=5)", "NHC\n(M=5)"], fontsize=FS_T)
        ax.set_title(f"({chr(97+i)}) {dim}D, 5 modes", fontsize=FS_TT)
        ax.tick_params(labelsize=FS_T)
        par_mv = data["parallel"]["modes_visited"]["mean"]
        nhc_mv = data["nhc"]["modes_visited"]["mean"]
        ymax = max(means) if max(means) > 0 else 1
        ax.text(0, ymax * 0.15, f"visited:\n{par_mv:.0%}", ha="center",
                fontsize=11, fontweight="bold", color="white")
        ax.text(1, ymax * 0.15, f"visited:\n{nhc_mv:.0%}", ha="center",
                fontsize=11, fontweight="bold", color="white")
    axes[0].set_ylabel("Mode crossings (200k steps)", fontsize=FS_L)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig3_high_dim.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig3_high_dim.png", flush=True)

    # Fig 4: Summary table
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    col_labels = ["Experiment", "Setting", "Log-osc (ours)", "NHC (best)", "Langevin (best)", "Ratio (ours/NHC)"]
    table_data = []
    for a in [1.0, 4.0, 8.0]:
        key = str(float(a))
        if key in exp1:
            d = exp1[key]
            p = d["parallel"]["barrier_crossings"]["mean"]
            n = d["nhc"]["barrier_crossings"]["mean"]
            l = d["langevin"]["barrier_crossings"]["mean"]
            ratio = p / max(n, 1)
            table_data.append(["Barrier sweep", f"a={a}", f"{p:.0f}", f"{n:.0f}", f"{l:.0f}", f"{ratio:.2f}x"])
    for nm in [5, 10, 20]:
        key = str(nm)
        if key in exp2:
            d = exp2[key]
            p = d["parallel"]["mode_crossings"]["mean"]
            n = d["nhc"]["mode_crossings"]["mean"]
            l = d["langevin"]["mode_crossings"]["mean"]
            ratio = p / max(n, 1)
            table_data.append(["Mode sweep", f"n={nm}", f"{p:.0f}", f"{n:.0f}", f"{l:.0f}", f"{ratio:.2f}x"])
    for dim in dims:
        key = str(dim)
        if key in exp3:
            d = exp3[key]
            p = d["parallel"]["mode_crossings"]["mean"]
            n = d["nhc"]["mode_crossings"]["mean"]
            ratio = p / max(n, 1)
            table_data.append(["High-D", f"d={dim}", f"{p:.0f}", f"{n:.0f}", "---", f"{ratio:.2f}x"])
    if table_data:
        table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.0, 1.6)
        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#d4e6f1")
            table[0, j].set_text_props(fontweight="bold")
        for i in range(1, len(table_data) + 1):
            try:
                val = float(table_data[i-1][5].replace("x", ""))
                if val > 1.5:
                    table[i, 5].set_facecolor("#d5f5e3")
                elif val < 0.8:
                    table[i, 5].set_facecolor("#fadbd8")
            except:
                pass
    ax.set_title("Summary: Mode-Hopping Benchmark", fontsize=FS_TT, pad=20)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig4_summary_table.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig4_summary_table.png", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    results = {}
    results["exp1"] = exp1_barrier_sweep()
    results["exp2"] = exp2_mode_count_sweep()
    results["exp3"] = exp3_high_dim()
    elapsed = time.time() - t0
    results["elapsed_sec"] = elapsed

    exp1 = results["exp1"]
    if "4.0" in exp1:
        par = exp1["4.0"]["parallel"]["barrier_crossings"]["mean"]
        nhc = exp1["4.0"]["nhc"]["barrier_crossings"]["mean"]
        ratio = par / max(nhc, 1)
        results["headline_crossings_ratio_barrier4"] = ratio
        print(f"\nHEADLINE: crossings_ratio at barrier=4.0 = {ratio:.3f}", flush=True)

    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved {out_path}  total {elapsed:.1f}s", flush=True)

    print("\nGenerating figures...", flush=True)
    make_figures(results)
    return results

if __name__ == "__main__":
    main()
