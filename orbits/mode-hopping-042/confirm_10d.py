"""Confirm the 10D result with 10 seeds."""
import json, os, sys, time, numpy as np
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))

# Import from run_experiment
from run_experiment import (HighDimGMM, simulate_multiscale, simulate_nhc,
                            count_mode_crossings, modes_visited, corrected_Q_range)

def run_one(args):
    sampler, seed, pot_dict, dt, Qs, n_steps = args
    dim = pot_dict["dim"]
    centers = pot_dict["centers"]
    pot = HighDimGMM(dim=dim, centers=centers, sigma=0.5)
    if sampler == "multiscale":
        traj = simulate_multiscale(pot, Qs, dt, n_steps, seed=seed, record_every=4)
    else:
        traj = simulate_nhc(pot, Qs, dt, n_steps, seed=seed, record_every=4)
    cross = count_mode_crossings(traj, pot) if len(traj) > 1 else 0
    vis = modes_visited(traj, pot) if len(traj) > 1 else 0.0
    return {"seed": seed, "sampler": sampler, "crossings": cross, "visited": vis}

def main():
    dim = 10
    n_modes = 5
    radius = 3.0
    sigma = 0.5
    n_steps = 200_000
    n_seeds = 10

    rng = np.random.default_rng(777)
    raw = rng.normal(size=(n_modes, dim))
    centers = (raw / np.linalg.norm(raw, axis=1, keepdims=True)) * radius

    omega_min, omega_max = 2.0, 6.0
    Qs_par = corrected_Q_range(omega_min, omega_max, 5)
    dt = min(0.02, 0.1 / np.sqrt(dim))

    pot_dict = {"dim": dim, "centers": centers.tolist()}

    # Also test more NHC Q_ref values
    Q_refs_nhc = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

    tasks = []
    seeds = list(range(500, 500 + n_seeds))
    for s in seeds:
        tasks.append(("multiscale", s, pot_dict, dt, Qs_par.tolist(), n_steps))
    for Q_ref in Q_refs_nhc:
        Qs_nhc = (np.ones(5) * Q_ref).tolist()
        for s in seeds:
            tasks.append(("nhc", s, pot_dict, dt, Qs_nhc, n_steps))

    t0 = time.time()
    with Pool(min(len(tasks), cpu_count())) as pool:
        results = pool.map(run_one, tasks)
    elapsed = time.time() - t0

    # Parse results
    par_results = [r for r in results if r["sampler"] == "multiscale"]
    par_cross = [r["crossings"] for r in par_results]
    par_vis = [r["visited"] for r in par_results]
    print(f"Multi-scale (N=5): crossings={np.mean(par_cross):.1f} +/- {np.std(par_cross):.1f}, "
          f"visited={np.mean(par_vis):.2f} +/- {np.std(par_vis):.2f}")

    nhc_results_by_q = {}
    for r in results:
        if r["sampler"] == "nhc":
            # group by task Qs
            pass
    # Simpler: just check NHC results per Q_ref
    idx = n_seeds  # first n_seeds are multiscale
    for Q_ref in Q_refs_nhc:
        nhc_batch = results[idx:idx+n_seeds]
        idx += n_seeds
        cross = [r["crossings"] for r in nhc_batch]
        vis = [r["visited"] for r in nhc_batch]
        print(f"NHC (Q={Q_ref}): crossings={np.mean(cross):.1f} +/- {np.std(cross):.1f}, "
              f"visited={np.mean(vis):.2f} +/- {np.std(vis):.2f}")

    # Best NHC
    idx = n_seeds
    best_nhc_cross = 0
    best_nhc_q = None
    for Q_ref in Q_refs_nhc:
        nhc_batch = results[idx:idx+n_seeds]
        idx += n_seeds
        mean_cross = np.mean([r["crossings"] for r in nhc_batch])
        if mean_cross > best_nhc_cross:
            best_nhc_cross = mean_cross
            best_nhc_q = Q_ref

    ratio = np.mean(par_cross) / max(best_nhc_cross, 1)
    print(f"\nBest NHC Q={best_nhc_q}: {best_nhc_cross:.1f}")
    print(f"Ratio (ours/NHC best): {ratio:.2f}x")
    print(f"Elapsed: {elapsed:.1f}s")

if __name__ == "__main__":
    main()
