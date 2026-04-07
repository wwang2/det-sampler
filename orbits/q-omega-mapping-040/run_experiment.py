"""q-omega-mapping-040: empirical Q-omega scaling for log-osc and NHC.

Pure phenomenology: measure Q_opt(omega) on 1D harmonic + 2D anisotropic Gaussian.
"""
import json
import os
import sys
import time
from multiprocessing import Pool
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))

# Reuse parent integrators via local shim
from parent_shim import (  # type: ignore
    simulate_multiscale, simulate_nhc, tau_q2_mean, autocorr_time,
    AnisotropicGaussian
)


class HarmonicOsc1D:
    name = "harmonic_1d"

    def __init__(self, omega):
        self.omega = float(omega)
        self.dim = 1
        self.kappas = np.array([self.omega ** 2])

    def energy(self, q):
        return 0.5 * self.omega ** 2 * q[0] * q[0]

    def gradient(self, q):
        return self.omega ** 2 * q


# ---------------------------------------------------------------------------
# Single-task workers
# ---------------------------------------------------------------------------
def _worker_1d(args):
    omega, Q, seed, n_steps, sampler = args
    pot = HarmonicOsc1D(omega)
    dt = 0.05 * min(1.0 / omega, np.sqrt(Q))
    Qs = np.array([Q])
    if sampler == "logosc":
        traj = simulate_multiscale(pot, Qs, dt, n_steps, seed=seed, record_every=2)
    else:
        Qs = np.array([Q, Q, Q])  # NHC chain length 3, all Q the same
        traj = simulate_nhc(pot, Qs, dt, n_steps, seed=seed, record_every=2)
    return omega, Q, seed, sampler, tau_q2_mean(traj)


def _worker_2d(args):
    om2, Q1, Q2, seed, n_steps = args
    pot = AnisotropicGaussian([1.0, om2 ** 2])
    dt = 0.05 * min(1.0 / om2, np.sqrt(min(Q1, Q2)))
    Qs = np.array([Q1, Q2])
    traj = simulate_multiscale(pot, Qs, dt, n_steps, seed=seed, record_every=2)
    if len(traj) < 64:
        return om2, Q1, Q2, seed, 1e6
    t1 = autocorr_time(traj[:, 0] ** 2)
    t2 = autocorr_time(traj[:, 1] ** 2)
    return om2, Q1, Q2, seed, float(max(t1, t2))


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------
OMEGAS = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
Q_GRID = np.logspace(-3, 3, 25)
N_STEPS = 200_000
N_SEEDS = 5


def run_part1_part4(pool):
    tasks_lo, tasks_nhc = [], []
    for om in OMEGAS:
        for Q in Q_GRID:
            for s in range(N_SEEDS):
                tasks_lo.append((om, Q, s, N_STEPS, "logosc"))
                tasks_nhc.append((om, Q, s, N_STEPS, "nhc"))
    print(f"Part 1+4: {len(tasks_lo) + len(tasks_nhc)} runs")
    t0 = time.time()
    res_lo = pool.map(_worker_1d, tasks_lo)
    print(f"  log-osc done {time.time()-t0:.1f}s")
    t1 = time.time()
    res_nhc = pool.map(_worker_1d, tasks_nhc)
    print(f"  nhc done {time.time()-t1:.1f}s")

    # Aggregate
    def agg(res):
        out = {float(om): {float(Q): [] for Q in Q_GRID} for om in OMEGAS}
        for om, Q, s, sampler, tau in res:
            out[float(om)][float(Q)].append(tau)
        return {om: {Q: float(np.mean(v)) for Q, v in d.items()} for om, d in out.items()}

    return agg(res_lo), agg(res_nhc)


def run_part3(pool):
    Q1_grid = np.logspace(-2, 2, 12)
    Q2_grid = np.logspace(-2, 2, 12)
    om2_list = [3.0, 10.0, 30.0]
    n_seeds = 3
    tasks = []
    for om2 in om2_list:
        for Q1 in Q1_grid:
            for Q2 in Q2_grid:
                for s in range(n_seeds):
                    tasks.append((om2, Q1, Q2, s, N_STEPS))
    print(f"Part 3: {len(tasks)} runs")
    t0 = time.time()
    res = pool.map(_worker_2d, tasks)
    print(f"  done {time.time()-t0:.1f}s")
    out = {}
    for om2 in om2_list:
        grid = np.full((len(Q1_grid), len(Q2_grid)), np.nan)
        accum = {}
        for o, q1, q2, s, tau in res:
            if o != om2:
                continue
            accum.setdefault((q1, q2), []).append(tau)
        for i, q1 in enumerate(Q1_grid):
            for j, q2 in enumerate(Q2_grid):
                vals = accum.get((q1, q2), [])
                grid[i, j] = float(np.mean(vals)) if vals else np.nan
        out[float(om2)] = {
            "Q1_grid": Q1_grid.tolist(),
            "Q2_grid": Q2_grid.tolist(),
            "tau_grid": grid.tolist(),
        }
    return out


def fit_q_opt(curve_dict):
    """curve_dict: {omega: {Q: tau}} -> Q_opt(omega), fit log-log."""
    omegas = sorted(curve_dict.keys())
    q_opts = []
    for om in omegas:
        d = curve_dict[om]
        Qs = sorted(d.keys())
        taus = np.array([d[Q] for Q in Qs])
        i_opt = int(np.argmin(taus))
        q_opts.append(Qs[i_opt])
    log_om = np.log(omegas)
    log_q = np.log(q_opts)
    slope, intercept, r, _, _ = stats.linregress(log_om, log_q)
    return {
        "omegas": list(omegas),
        "q_opts": q_opts,
        "slope": float(slope),
        "intercept": float(intercept),
        "c": float(np.exp(intercept)),
        "r2": float(r * r),
    }


def main():
    t0 = time.time()
    with Pool(processes=10) as pool:
        lo_curves, nhc_curves = run_part1_part4(pool)
        part3 = run_part3(pool)

    fit_lo = fit_q_opt(lo_curves)
    fit_nhc = fit_q_opt(nhc_curves)

    def serialize(d):
        return {str(om): {str(Q): tau for Q, tau in inner.items()}
                for om, inner in d.items()}

    out = {
        "logosc_curves": serialize(lo_curves),
        "nhc_curves": serialize(nhc_curves),
        "fit_logosc": fit_lo,
        "fit_nhc": fit_nhc,
        "part3_2d": {str(k): v for k, v in part3.items()},
        "elapsed_sec": time.time() - t0,
    }
    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nFit log-osc: Q_opt = {fit_lo['c']:.3f} * omega^{fit_lo['slope']:.3f}  R2={fit_lo['r2']:.3f}")
    print(f"Fit NHC:     Q_opt = {fit_nhc['c']:.3f} * omega^{fit_nhc['slope']:.3f}  R2={fit_nhc['r2']:.3f}")
    print(f"Total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
