"""gprime-ablation-052: Isolate the role of sign(g'(xi)) from g'(0) coupling.

Benchmarks four friction-function variants on the same d=10 anisotropic-Gaussian
target as paper-experiments-047 / friction-survey-045 E2:

    d=10, kappa_ratio=100, 20 seeds, 200k force evals, N=5 parallel thermostats,
    Q_c in {0.3, 1, 3, 10, 30, 100, 300}, pick best median tau_int per method.

Variants (all g'(0)=2 except the tanh reference):
    log-osc            g = 2 xi / (1 + xi^2)              g' changes sign at |xi|=1
    clipped-log-osc    g = 2 xi/(1+xi^2) for |xi|<=1,     g' >= 0, odd-symmetric
                        else sign(xi)                      saturating at +/-1
    tanh-scaled        g = 2 tanh(xi)                      g' >= 0, smooth
    tanh-ref           g =   tanh(xi)                      g'(0)=1, reference
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





# ---------- Item 2: Q=1e8 Hamiltonian-floor control --------------------------
def run_floor_control(nsteps=200_000, nseeds=5, dim=10, kappa_ratio=100.0):
    """Run all 4 friction methods at Q=1e8 (thermostat effectively off).
    
    Expected: all methods give tau_int ~ 4.97-5.02, confirming the Hamiltonian
    floor claim. Results appended to results.json under 'floor_control' key.
    """
    kappas = make_kappas(dim, kappa_ratio)
    dt = 0.05 / np.sqrt(kappa_ratio)
    Qc = 1e8
    Nth = 5
    Qs = np.exp(np.linspace(np.log(Qc / 3.0), np.log(3.0 * Qc), Nth))
    results = {}
    seeds = list(range(1000, 1000 + nseeds))
    for method, g_func in FRICTIONS.items():
        taus = []
        for seed in seeds:
            tr = simulate(g_func, kappas, Qs, dt, nsteps, seed=seed, rec=4)
            taus.append((seed, tau_int(tr)))
        med = float(sorted([t for _, t in taus])[len(taus) // 2])
        results[method] = dict(Qc=Qc, taus=taus, median=med)
        print(f"  [floor_control] {method}: median tau = {med:.2f}")
    return results


# ---------- Item 5: Double-well control experiment ---------------------------
def double_well_force(x, kappas_unused=None):
    """Gradient of V(x) = (x^2 - 1)^2 for 1D, extended to d dims as sum_i V(x_i)."""
    return 4.0 * x * (x * x - 1.0)


def simulate_dw(g_func, dim, Qs, dt, nsteps, kT=1.0, seed=0, rec=4):
    """Simulate parallel thermostats on a 1D symmetric double-well V(x)=sum_i (x_i^2-1)^2."""
    rng = np.random.default_rng(seed)
    N = len(Qs)
    Qs = np.asarray(Qs, float)
    q = rng.normal(0.0, 1.0, size=dim)
    p = rng.normal(0.0, np.sqrt(kT), size=dim)
    xi = np.zeros(N)
    h = 0.5 * dt
    gU = double_well_force(q)
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
        gU = double_well_force(q)
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


def run_double_well_control(Qc=10.0, nsteps=200_000, nseeds=20, dim=1):
    """Run all 4 friction methods on 1D symmetric double-well V(x)=(x^2-1)^2.
    
    This tests that the tau_int estimator works on a non-trivial (non-harmonic)
    target and that the 'no g' sign effect' conclusion is not an estimator artifact.
    Results appended to results.json under 'double_well_control' key.
    """
    dt = 0.005  # conservative for double-well
    Nth = 5
    Qs = np.exp(np.linspace(np.log(max(Qc / 3.0, 0.01)), np.log(3.0 * Qc), Nth))
    results = {"config": dict(Qc=Qc, dt=dt, nsteps=nsteps, nseeds=nseeds, dim=dim,
                              potential="V(x) = (x^2 - 1)^2")}
    seeds = list(range(1000, 1000 + nseeds))
    for method, g_func in FRICTIONS.items():
        taus = []
        for seed in seeds:
            tr = simulate_dw(g_func, dim, Qs, dt, nsteps, seed=seed, rec=4)
            taus.append((seed, tau_int(tr)))
        tau_vals = [t for _, t in taus]
        tau_s = sorted(tau_vals)
        n = len(tau_s)
        med = float((tau_s[n//2 - 1] + tau_s[n//2]) / 2.0) if n % 2 == 0 else float(tau_s[n//2])
        q25_idx = (n - 1) * 0.25; lo = int(q25_idx); frac = q25_idx - lo
        q25 = float(tau_s[lo] * (1 - frac) + tau_s[min(lo+1, n-1)] * frac)
        q75_idx = (n - 1) * 0.75; lo75 = int(q75_idx); frac75 = q75_idx - lo75
        q75 = float(tau_s[lo75] * (1 - frac75) + tau_s[min(lo75+1, n-1)] * frac75)
        results[method] = dict(Qc=Qc, taus=taus, median=med, q25=q25, q75=q75)
        print(f"  [double_well] {method}: median tau = {med:.2f} IQR=[{q25:.2f}, {q75:.2f}]")
    return results


# ---------- Item 3: Compute active_summary (best Q with median > 10) --------
def compute_active_summary(res):
    """Pick argmin(median tau) subject to median > 10 for each method.
    
    The threshold 10 filters out the Hamiltonian floor (~5), ensuring the
    reported 'best' is in the thermostat-active regime. Returns dict keyed
    by method with Qc, median, q25, q75.
    """
    summary = {}
    per = res["per_method_per_Q"]
    def _median(vals):
        s = sorted(vals)
        n = len(s)
        if n % 2 == 1:
            return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2.0

    def _percentile(vals, pct):
        s = sorted(vals)
        k = (len(s) - 1) * pct / 100.0
        lo = int(k)
        hi = min(lo + 1, len(s) - 1)
        frac = k - lo
        return s[lo] * (1 - frac) + s[hi] * frac

    for method, byQ in per.items():
        best = None
        for qk, rec in byQ.items():
            tau_vals = [t for _, t in rec["taus"]]
            med = _median(tau_vals)
            if med <= 10.0:
                continue  # skip Hamiltonian floor
            q25 = _percentile(tau_vals, 25)
            q75 = _percentile(tau_vals, 75)
            if best is None or med < best["median"]:
                best = dict(Qc=qk, median=med, q25=q25, q75=q75)
        if best is None:
            # All Qs are in the floor — take the one closest to threshold
            for qk, rec in byQ.items():
                tau_vals = [t for _, t in rec["taus"]]
                med = _median(tau_vals)
                if best is None or med < best["median"]:
                    best = dict(Qc=qk, median=med, q25=_percentile(tau_vals, 25),
                                q75=_percentile(tau_vals, 75))
        summary[method] = best
    return summary


def run_controls_and_summarize():
    """Run floor control, double-well control, compute active_summary, update results.json."""
    path = os.path.join(OUT, "results.json")
    with open(path) as f:
        res = json.load(f)

    print("\n=== Floor control (Q=1e8) ===")
    res["floor_control"] = run_floor_control()

    print("\n=== Double-well control (Qc=10, 1D) ===")
    res["double_well_control"] = run_double_well_control()

    print("\n=== Active summary (median > 10 filter) ===")
    active = compute_active_summary(res)
    res["active_summary"] = active
    for method, info in active.items():
        print(f"  {method}: {info['Qc']} -> median tau = {info['median']:.2f}")

    with open(path, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\nUpdated {path}")
    return res


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--controls":
        run_controls_and_summarize()
    elif len(sys.argv) > 1 and sys.argv[1] == "--sweep":
        main()
    else:
        main()
