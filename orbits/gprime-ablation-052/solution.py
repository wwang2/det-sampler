"""gprime-ablation-052: Isolate the role of sign(g'(xi)) from g'(0) coupling.

Benchmarks five friction-function variants on the same d=10 anisotropic-Gaussian
target as paper-experiments-047 / friction-survey-045 E2:

    d=10, kappa_ratio=100, 20 seeds, 200k force evals, N=5 parallel thermostats,
    Q in [50, 500] (scan, pick best median tau_int per method).

Variants (all g'(0)=2 except the tanh reference):
    log-osc            g = 2 xi / (1 + xi^2)              g' changes sign
    clipped-log-osc    g = max(0, 2 xi/(1+xi^2))           g' >= 0, discontinuous
    tanh-scaled        g = 2 tanh(xi)                      g' >= 0, smooth
    tanh-ref           g =   tanh(xi)                      g'(0)=1 reference
    log-osc-ref        g = 2 xi / (1 + xi^2)               same as log-osc (alias)
"""
from __future__ import annotations
import json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)


# ---------- friction functions -------------------------------------------------
def g_logosc(xi):       return 2.0 * xi / (1.0 + xi * xi)
def g_clipped(xi):
    # Odd-symmetric "g' clipped to >= 0": follow log-osc on |xi|<=1 where g'>=0,
    # then saturate at +/-1 (the peak value of 2xi/(1+xi^2)) for |xi|>1.
    return np.where(np.abs(xi) <= 1.0, 2.0 * xi / (1.0 + xi * xi), np.sign(xi))
def g_tanh_scaled(xi):  return 2.0 * np.tanh(xi)
def g_tanh_ref(xi):     return np.tanh(xi)

FRICTIONS = {
    "log-osc":         g_logosc,
    "clipped-log-osc": g_clipped,
    "tanh-scaled":     g_tanh_scaled,
    "tanh-ref":        g_tanh_ref,
}


# ---------- target: d=10 anisotropic gaussian ---------------------------------
def make_kappas(dim=10, kappa_ratio=100.0):
    # geometric ladder from 1 to kappa_ratio over dim dimensions
    return np.array([kappa_ratio ** (i / (dim - 1)) for i in range(dim)])


# ---------- integrator: parallel thermostats with friction g ------------------
def simulate(g_func, kappas, Qs, dt, nsteps, kT=1.0, seed=0, rec=4):
    rng = np.random.default_rng(seed)
    dim = len(kappas)
    N = len(Qs)
    Qs = np.asarray(Qs, float)
    q = rng.normal(0.0, 1.0, size=dim) / np.sqrt(np.maximum(kappas, 1e-12))
    p = rng.normal(0.0, np.sqrt(kT), size=dim)
    xi = np.zeros(N)
    h = 0.5 * dt
    gU = kappas * q
    nr = nsteps // rec
    qs = np.empty((nr, dim))
    ri = 0
    for s in range(nsteps):
        K = float(np.dot(p, p))
        xi += h * (K - dim * kT) / Qs
        gt = float(np.sum(g_func(xi)))
        p *= np.exp(-np.clip(gt * h, -50, 50))
        p -= h * gU
        q = q + dt * p
        gU = kappas * q
        p -= h * gU
        gt = float(np.sum(g_func(xi)))
        p *= np.exp(-np.clip(gt * h, -50, 50))
        K = float(np.dot(p, p))
        xi += h * (K - dim * kT) / Qs
        if (s + 1) % rec == 0 and ri < nr:
            qs[ri] = q
            ri += 1
        if not np.isfinite(p).all():
            qs[ri:] = np.nan
            break
    return qs[:ri]


# ---------- integrated autocorrelation (on q_d^2, averaged across dims) -------
def acf_tau(x, c=5.0):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    if n < 16 or np.std(x) < 1e-12:
        return float(n)
    f = np.fft.fft(x, n=2 * n)
    a = np.fft.ifft(f * np.conj(f))[:n].real
    a /= a[0]
    tau = 1.0
    for k in range(1, n // 4):
        tau += 2 * a[k]
        if k >= c * tau:
            break
    return max(tau, 1.0)


def tau_int(trajectory):
    v = trajectory[~np.isnan(trajectory[:, 0])]
    if len(v) < 64:
        return 1e6
    return float(np.mean([acf_tau(v[:, d] ** 2) for d in range(v.shape[1])]))


# ---------- single (method, seed, Q-ladder) evaluation -----------------------
def run_one(method, seed, kappas, Qs, dt, nsteps):
    g = FRICTIONS[method]
    try:
        tr = simulate(g, kappas, Qs, dt, nsteps, seed=seed, rec=4)
    except Exception as e:
        return method, seed, 1e6
    return method, seed, tau_int(tr)


# ---------- Q sweep driver ----------------------------------------------------
def sweep(nsteps=200_000, nseeds=20, dim=10, kappa_ratio=100.0,
          Q_centers=(50.0, 100.0, 200.0, 300.0, 500.0),
          max_workers=None):
    kappas = make_kappas(dim, kappa_ratio)
    dt = 0.05 / np.sqrt(kappa_ratio)  # same as friction-survey-045 E2
    Nth = 5
    results = {}
    t0 = time.time()
    # For each method and each Q center, spread Nth thermostats geometrically in [Q/3, 3Q]
    jobs = []
    for method in FRICTIONS:
        results[method] = {}
        for Qc in Q_centers:
            Qs = np.exp(np.linspace(np.log(Qc / 3.0), np.log(3.0 * Qc), Nth))
            results[method][f"Q={Qc}"] = dict(Qs=Qs.tolist(), taus=[])
    tasks = []
    for method in FRICTIONS:
        for Qc in Q_centers:
            Qs = np.exp(np.linspace(np.log(Qc / 3.0), np.log(3.0 * Qc), Nth))
            for s in range(nseeds):
                tasks.append((method, Qc, tuple(Qs.tolist()), 1000 + s))
    print(f"[sweep] {len(tasks)} tasks (methods={len(FRICTIONS)} x Qc={len(Q_centers)} x seeds={nseeds})")
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futs = {}
        for (method, Qc, Qs_tup, seed) in tasks:
            f = pool.submit(run_one, method, seed, kappas, np.array(Qs_tup), dt, nsteps)
            futs[f] = (method, Qc, seed)
        done = 0
        for fut in as_completed(futs):
            method, Qc, seed = futs[fut]
            _, _, tau = fut.result()
            results[method][f"Q={Qc}"]["taus"].append((seed, tau))
            done += 1
            if done % 25 == 0:
                print(f"  [{done}/{len(tasks)}] {method} Qc={Qc} seed={seed} tau={tau:.2f}")
    elapsed = time.time() - t0
    print(f"[sweep] done in {elapsed:.1f}s")

    # summarize: for each method, pick best Qc by median tau, report median+IQR across seeds
    summary = {}
    for method, byQ in results.items():
        best = None
        for qc_key, rec in byQ.items():
            taus = np.array([t for _, t in rec["taus"]])
            med = float(np.median(taus))
            if best is None or med < best["median"]:
                best = dict(Qc=qc_key, median=med,
                            q25=float(np.percentile(taus, 25)),
                            q75=float(np.percentile(taus, 75)),
                            taus=taus.tolist())
        summary[method] = best
        print(f"  [{method}] best {best['Qc']}  median tau_int = {best['median']:.2f} "
              f"(IQR [{best['q25']:.2f}, {best['q75']:.2f}])")
    return dict(config=dict(nsteps=nsteps, nseeds=nseeds, dim=dim,
                            kappa_ratio=kappa_ratio, dt=dt,
                            Q_centers=list(Q_centers), Nth=Nth,
                            elapsed_s=elapsed),
                per_method_per_Q=results, summary=summary)


def main():
    out = sweep()
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"Saved {os.path.join(OUT, 'results.json')}")
    return out


if __name__ == "__main__":
    main()
