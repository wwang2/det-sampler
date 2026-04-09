"""forensic-qeff-054: Is g'(0) just a Q-rescaling?

Tests the hypothesis that Q_eff = Q / g'(0) is the operative thermostat
parameter, and that the entire performance difference between tanh-ref
(g'(0)=1) and tanh-scaled (g'(0)=2) is due to Q_eff being different.

Experiments:
  E0: Short new trajectories (50k steps, 3 seeds) recording full xi time series
  E1: Decisive Q_eff matching test (tanh-scaled@Q=20 vs tanh-ref@Q=10)
  E2: Kappa sweep (if E1 confirms)
  E3: Residual tail-shape isolation (if E1 confirms)
"""
from __future__ import annotations
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

PARENT = os.path.join(os.path.dirname(OUT), "gprime-ablation-052")

# ---------- friction functions (from parent) ----------------------------------
def g_logosc(xi):       return 2.0 * xi / (1.0 + xi * xi)
def g_tanh_scaled(xi):  return 2.0 * np.tanh(xi)
def g_tanh_ref(xi):     return np.tanh(xi)

FRICTIONS = {
    "log-osc":     g_logosc,
    "tanh-scaled": g_tanh_scaled,
    "tanh-ref":    g_tanh_ref,
}

# g'(0) values for each friction function
GPRIME0 = {
    "log-osc":     2.0,
    "tanh-scaled": 2.0,
    "tanh-ref":    1.0,
}


# ---------- target: d=10 anisotropic gaussian ---------------------------------
def make_kappas(dim=10, kappa_ratio=100.0):
    return np.array([kappa_ratio ** (i / (dim - 1)) for i in range(dim)])


# ---------- integrator (from parent, with optional xi recording) --------------
def simulate(g_func, kappas, Qs, dt, nsteps, kT=1.0, seed=0, rec=4,
             record_xi=False):
    """Parallel-thermostat integrator. Optionally records xi trajectory."""
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
    xis = np.empty((nr, N)) if record_xi else None
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
            if record_xi:
                xis[ri] = xi
            ri += 1
        if not np.isfinite(p).all():
            qs[ri:] = np.nan
            if record_xi:
                xis[ri:] = np.nan
            break
    if record_xi:
        return qs[:ri], xis[:ri]
    return qs[:ri]


# ---------- tau_int estimator (from parent) -----------------------------------
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


# ---------- single run for ProcessPoolExecutor --------------------------------
def run_one(method, seed, kappas, Qs, dt, nsteps):
    g = FRICTIONS[method]
    try:
        tr = simulate(g, kappas, Qs, dt, nsteps, seed=seed, rec=4)
    except Exception:
        return method, seed, 1e6
    return method, seed, tau_int(tr)


def _run_one_alpha(alpha, seed, kappas, Qs, dt, nsteps):
    """Run simulate with g(xi) = alpha * tanh(xi)."""
    def _g(xi):
        return alpha * np.tanh(xi)
    try:
        tr = simulate(_g, kappas, Qs, dt, nsteps, seed=seed, rec=4)
    except Exception:
        return 1e6
    return tau_int(tr)


# ============================================================================
# E0: Re-analysis + xi trajectory recording
# ============================================================================
def run_e0(nsteps=50_000, nseeds=3, dim=10, kappa_ratio=100.0):
    """Short trajectories recording full xi for all methods at Q=10."""
    print("\n=== E0: xi trajectory analysis ===")
    kappas = make_kappas(dim, kappa_ratio)
    dt = 0.05 / np.sqrt(kappa_ratio)
    Qc = 10.0
    Nth = 5
    Qs = np.exp(np.linspace(np.log(Qc / 3.0), np.log(3.0 * Qc), Nth))
    seeds = list(range(1000, 1000 + nseeds))

    results = {}
    for method, g_func in FRICTIONS.items():
        all_xi = []
        all_tau = []
        for seed in seeds:
            qs, xis = simulate(g_func, kappas, Qs, dt, nsteps, seed=seed,
                               rec=4, record_xi=True)
            all_xi.append(xis)
            all_tau.append(tau_int(qs))
        # Concatenate xi across seeds (use the middle thermostat, index N//2)
        xi_cat = np.concatenate([x[:, Nth // 2] for x in all_xi])
        std_xi = float(np.std(xi_cat))
        # Predicted std(xi) = sqrt(kT / (g'(0) * Q_middle))
        Q_mid = Qs[Nth // 2]
        gp0 = GPRIME0[method]
        pred_std = 1.0 / np.sqrt(gp0 * Q_mid)
        # Entropy from histogram
        counts, edges = np.histogram(xi_cat, bins=100, density=True)
        dx = edges[1] - edges[0]
        p_xi = counts * dx
        p_xi = p_xi[p_xi > 0]
        entropy = float(-np.sum(p_xi * np.log(p_xi + 1e-30)))
        med_tau = float(np.median(all_tau))
        results[method] = dict(
            std_xi=std_xi, pred_std=pred_std,
            ratio_std=std_xi / pred_std if pred_std > 0 else float('nan'),
            entropy=entropy, median_tau=med_tau,
            taus=all_tau,
            xi_histogram=dict(counts=counts.tolist(),
                              edges=edges.tolist())
        )
        print(f"  {method}: std(xi)={std_xi:.4f} (pred={pred_std:.4f}, ratio={std_xi/pred_std:.3f}), "
              f"S={entropy:.3f}, tau={med_tau:.1f}")
    return results


def plot_e0(results):
    """2-panel: (a) xi histograms, (b) entropy vs tau_int."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11,
        'figure.dpi': 150, 'savefig.dpi': 300,
        'savefig.pad_inches': 0.2,
    })
    colors = {'log-osc': '#2ca02c', 'tanh-scaled': '#d62728', 'tanh-ref': '#1f77b4'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    # (a) xi histograms
    ax = axes[0]
    for method, r in results.items():
        edges = np.array(r['xi_histogram']['edges'])
        counts = np.array(r['xi_histogram']['counts'])
        centers = 0.5 * (edges[:-1] + edges[1:])
        gp0 = GPRIME0[method]
        label = f"{method} (g'(0)={gp0:.0f}, std={r['std_xi']:.3f})"
        ax.plot(centers, counts, color=colors[method], label=label, lw=1.5)
    ax.set_title("(a) Thermostat variable distribution", fontweight='bold')
    ax.set_xlabel(r"$\xi$ (middle thermostat)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9, frameon=False)

    # (b) entropy vs tau_int
    ax = axes[1]
    for method, r in results.items():
        ax.scatter(r['entropy'], r['median_tau'], color=colors[method],
                   s=100, zorder=5, label=method)
        ax.annotate(f"g'(0)={GPRIME0[method]:.0f}",
                    (r['entropy'], r['median_tau']),
                    textcoords="offset points", xytext=(8, 5), fontsize=10)
    ax.set_title("(b) Thermostat entropy vs mixing", fontweight='bold')
    ax.set_xlabel(r"Entropy $S(\xi)$")
    ax.set_ylabel(r"$\tau_{\rm int}$ (median)")
    ax.legend(fontsize=10, frameon=False)

    path = os.path.join(FIG, "e0_xi_analysis.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ============================================================================
# E1: Decisive Q_eff matching test
# ============================================================================
def run_e1(nsteps=200_000, nseeds=20, dim=10, kappa_ratio=100.0):
    """The razor: does tanh-scaled@Q=20 match tanh-ref@Q=10 (both Q_eff=10)?"""
    print("\n=== E1: Q_eff matching test ===")
    kappas = make_kappas(dim, kappa_ratio)
    dt = 0.05 / np.sqrt(kappa_ratio)
    Nth = 5

    # Conditions to test
    conditions = {
        "tanh-ref@Q=10 (Qeff=10)": ("tanh-ref", 10.0),
        "tanh-scaled@Q=20 (Qeff=10)": ("tanh-scaled", 20.0),
        "tanh-scaled@Q=10 (Qeff=5)": ("tanh-scaled", 10.0),
        "log-osc@Q=20 (Qeff=10)": ("log-osc", 20.0),
        "tanh-ref@Q=20 (Qeff=20)": ("tanh-ref", 20.0),
        "log-osc@Q=10 (Qeff=5)": ("log-osc", 10.0),
    }

    results = {}
    tasks = []
    for label, (method, Qc) in conditions.items():
        Qs = np.exp(np.linspace(np.log(Qc / 3.0), np.log(3.0 * Qc), Nth))
        for s in range(nseeds):
            tasks.append((label, method, Qc, tuple(Qs.tolist()), 1000 + s))

    print(f"  {len(tasks)} tasks across {len(conditions)} conditions")
    t0 = time.time()

    with ProcessPoolExecutor() as pool:
        futs = {}
        for (label, method, Qc, Qs_tup, seed) in tasks:
            f = pool.submit(run_one, method, seed, kappas,
                            np.array(Qs_tup), dt, nsteps)
            futs[f] = (label, seed)
        raw = {label: [] for label in conditions}
        done = 0
        for fut in as_completed(futs):
            label, seed = futs[fut]
            _, _, tau = fut.result()
            raw[label].append(tau)
            done += 1
            if done % 20 == 0:
                print(f"    [{done}/{len(tasks)}]")

    elapsed = time.time() - t0
    print(f"  E1 done in {elapsed:.1f}s")

    for label, taus in raw.items():
        method, Qc = conditions[label]
        arr = np.array(taus)
        med = float(np.median(arr))
        q25 = float(np.percentile(arr, 25))
        q75 = float(np.percentile(arr, 75))
        qeff = Qc / GPRIME0[method]
        results[label] = dict(
            method=method, Qc=Qc, Qeff=qeff,
            median=med, q25=q25, q75=q75,
            taus=arr.tolist()
        )
        print(f"  {label}: tau={med:.1f} [{q25:.1f}, {q75:.1f}]")

    # The decisive comparison
    ref = results["tanh-ref@Q=10 (Qeff=10)"]["median"]
    scaled = results["tanh-scaled@Q=20 (Qeff=10)"]["median"]
    ratio = scaled / ref if ref > 0 else float('nan')
    results["_comparison"] = dict(
        ref_tau=ref, scaled_tau=scaled, ratio=ratio,
        match_within_30pct=abs(ratio - 1.0) < 0.30,
        elapsed_s=elapsed
    )
    print(f"\n  DECISIVE: tanh-scaled@Q=20 / tanh-ref@Q=10 = {ratio:.3f}")
    if abs(ratio - 1.0) < 0.30:
        print("  --> Q_eff hypothesis CONFIRMED (within 30%)")
    else:
        print("  --> Q_eff hypothesis REJECTED (>30% gap)")

    return results


def plot_e1(results):
    """Bar chart comparing tau_int at matched Q_eff."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11,
        'figure.dpi': 150, 'savefig.dpi': 300,
        'savefig.pad_inches': 0.2,
    })

    labels_qeff10 = [
        "tanh-ref@Q=10 (Qeff=10)",
        "tanh-scaled@Q=20 (Qeff=10)",
        "log-osc@Q=20 (Qeff=10)",
    ]

    colors_map = {
        'tanh-ref': '#1f77b4',
        'tanh-scaled': '#d62728',
        'log-osc': '#2ca02c',
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # (a) Box plot at matched Q_eff=10
    ax = axes[0]
    data_boxes = []
    box_labels = []
    box_colors = []
    for label in labels_qeff10:
        r = results[label]
        data_boxes.append(r['taus'])
        short = f"{r['method']}\nQ={r['Qc']:.0f}"
        box_labels.append(short)
        box_colors.append(colors_map[r['method']])

    bp = ax.boxplot(data_boxes, patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    for median_line in bp['medians']:
        median_line.set_color('black')
        median_line.set_linewidth(2)
    ax.set_xticklabels(box_labels, fontsize=11)
    ax.set_ylabel(r"$\tau_{\rm int}$")
    ax.set_title("(a) Matched Q_eff = 10", fontweight='bold')

    ref_med = results["tanh-ref@Q=10 (Qeff=10)"]["median"]
    scaled_med = results["tanh-scaled@Q=20 (Qeff=10)"]["median"]
    ratio = scaled_med / ref_med
    ax.text(0.05, 0.95, f"ratio = {ratio:.2f}",
            transform=ax.transAxes, fontsize=12, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # (b) All conditions: tau_int vs Q_eff (linear x-axis, lines connecting per method)
    ax = axes[1]
    all_labels = list(results.keys())
    all_labels = [l for l in all_labels if not l.startswith("_")]
    # Group by method for connecting lines
    from collections import defaultdict
    by_method = defaultdict(list)
    for label in all_labels:
        r = results[label]
        by_method[r['method']].append(r)
    for method, pts in by_method.items():
        pts_sorted = sorted(pts, key=lambda r: r['Qeff'])
        color = colors_map[method]
        marker = {'tanh-ref': 'o', 'tanh-scaled': 's', 'log-osc': '^'}[method]
        qeffs = [r['Qeff'] for r in pts_sorted]
        meds = [r['median'] for r in pts_sorted]
        yerr_lo = [r['median'] - r['q25'] for r in pts_sorted]
        yerr_hi = [r['q75'] - r['median'] for r in pts_sorted]
        ax.errorbar(qeffs, meds, yerr=[yerr_lo, yerr_hi],
                    fmt=marker + '-', color=color, markersize=10, capsize=5,
                    label=method, zorder=5, lw=1.5)
    ax.set_xlabel(r"$Q_{\rm eff} = Q / g'(0)$")
    ax.set_ylabel(r"$\tau_{\rm int}$ (median)")
    ax.set_title("(b) All conditions vs Q_eff", fontweight='bold')
    ax.legend(fontsize=9, frameon=False, loc='upper right')
    ax.set_yscale('log')

    path = os.path.join(FIG, "e1_qeff_matching.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ============================================================================
# E2: Kappa sweep (alpha scaling)
# ============================================================================
def run_e2(nsteps=200_000, nseeds=10, dim=10):
    """Sweep alpha and kappa to find optimal coupling strength vs anisotropy."""
    print("\n=== E2: Alpha-kappa sweep ===")
    alphas = [0.5, 1.0, 1.5, 2.0, 3.0]
    kappas_list = [1.0, 10.0, 50.0, 100.0, 500.0]
    Qc = 10.0
    Nth = 5
    Qs = np.exp(np.linspace(np.log(Qc / 3.0), np.log(3.0 * Qc), Nth))

    tasks = []
    for kappa_ratio in kappas_list:
        kappas = make_kappas(dim, kappa_ratio)
        dt = 0.05 / np.sqrt(max(kappa_ratio, 1.0))
        for alpha in alphas:
            for s in range(nseeds):
                tasks.append((kappa_ratio, alpha, kappas, dt, tuple(Qs.tolist()), 1000 + s))

    print(f"  {len(tasks)} tasks ({len(kappas_list)} kappas x {len(alphas)} alphas x {nseeds} seeds)")
    t0 = time.time()

    raw = {}
    with ProcessPoolExecutor() as pool:
        futs = {}
        for (kr, alpha, kappas, dt, Qs_tup, seed) in tasks:
            f = pool.submit(_run_one_alpha, alpha, seed, kappas, np.array(Qs_tup), dt, nsteps)
            futs[f] = (kr, alpha, seed)
        done = 0
        for fut in as_completed(futs):
            kr, alpha, seed = futs[fut]
            tau = fut.result()
            key = (kr, alpha)
            if key not in raw:
                raw[key] = []
            raw[key].append(tau)
            done += 1
            if done % 50 == 0:
                print(f"    [{done}/{len(tasks)}]")

    elapsed = time.time() - t0
    print(f"  E2 done in {elapsed:.1f}s")

    results = {"alphas": alphas, "kappas": kappas_list, "data": {}}
    for kr in kappas_list:
        results["data"][str(kr)] = {}
        best_alpha = None
        best_med = 1e9
        for alpha in alphas:
            arr = np.array(raw[(kr, alpha)])
            med = float(np.median(arr))
            q25 = float(np.percentile(arr, 25))
            q75 = float(np.percentile(arr, 75))
            results["data"][str(kr)][str(alpha)] = dict(
                median=med, q25=q25, q75=q75, taus=arr.tolist()
            )
            if med < best_med:
                best_med = med
                best_alpha = alpha
        results["data"][str(kr)]["best_alpha"] = best_alpha
        results["data"][str(kr)]["best_median"] = best_med
        print(f"  kappa={kr}: best alpha={best_alpha}, tau={best_med:.1f}")

    results["elapsed_s"] = elapsed
    return results


def plot_e2(results):
    """Heatmap: alpha vs kappa, color = median tau_int."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11,
        'figure.dpi': 150, 'savefig.dpi': 300,
        'savefig.pad_inches': 0.2,
    })

    alphas = results["alphas"]
    kappas = results["kappas"]
    data = results["data"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # (a) tau_int vs alpha for each kappa
    ax = axes[0]
    cmap = plt.cm.viridis
    for i, kr in enumerate(kappas):
        color = cmap(i / (len(kappas) - 1))
        meds = [data[str(kr)][str(a)]["median"] for a in alphas]
        q25s = [data[str(kr)][str(a)]["q25"] for a in alphas]
        q75s = [data[str(kr)][str(a)]["q75"] for a in alphas]
        yerr_lo = [m - q for m, q in zip(meds, q25s)]
        yerr_hi = [q - m for m, q in zip(meds, q75s)]
        ax.errorbar(alphas, meds, yerr=[yerr_lo, yerr_hi],
                    fmt='o-', color=color, capsize=4, label=f"kappa={kr:.0f}")
    ax.set_xlabel(r"$\alpha$ (coupling strength)")
    ax.set_ylabel(r"$\tau_{\rm int}$ (median)")
    ax.set_title(r"(a) $\tau_{\rm int}$ vs coupling $\alpha$", fontweight='bold')
    ax.legend(fontsize=9, frameon=False)
    ax.set_yscale('log')

    # (b) optimal alpha vs kappa
    ax = axes[1]
    best_alphas = [data[str(kr)]["best_alpha"] for kr in kappas]
    ax.plot(kappas, best_alphas, 'ko-', markersize=10, lw=2, label=r'$\alpha_{\rm opt}$')
    # Overlay 1/sqrt(kappa) prediction (normalized)
    kappas_fine = np.linspace(1, 500, 100)
    scale = best_alphas[0]
    ax.plot(kappas_fine, scale / np.sqrt(kappas_fine),
            'r--', lw=1.5, label=r'$\propto 1/\sqrt{\kappa}$')
    ax.set_xlabel(r"$\kappa$ (anisotropy ratio)")
    ax.set_ylabel(r"$\alpha_{\rm opt}$")
    ax.set_title(r"(b) Optimal coupling vs anisotropy (floor-contaminated)", fontweight='bold', fontsize=13)
    ax.legend(fontsize=11, frameon=False)
    ax.set_xscale('log')

    path = os.path.join(FIG, "e2_kappa_sweep.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ============================================================================
# E3: Residual tail-shape isolation at matched Q_eff
# ============================================================================
def plot_e3(e1_results):
    """Violin plot comparing the three methods at Q_eff=10."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11,
        'figure.dpi': 150, 'savefig.dpi': 300,
        'savefig.pad_inches': 0.2,
    })

    labels = [
        "tanh-ref@Q=10 (Qeff=10)",
        "tanh-scaled@Q=20 (Qeff=10)",
        "log-osc@Q=20 (Qeff=10)",
    ]
    colors = ['#1f77b4', '#d62728', '#2ca02c']
    short_labels = ["tanh-ref\nQ=10, g'(0)=1", "tanh-scaled\nQ=20, g'(0)=2", "log-osc\nQ=20, g'(0)=2"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)

    data = [e1_results[l]['taus'] for l in labels]
    vp = ax.violinplot(data, positions=[1, 2, 3], showmedians=True, widths=0.7)
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(colors[i])
        body.set_alpha(0.5)
    vp['cmedians'].set_color('black')
    vp['cmedians'].set_linewidth(2)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(short_labels, fontsize=11)
    ax.set_ylabel(r"$\tau_{\rm int}$")
    ax.set_title("Tail shape at matched Q_eff = 10", fontweight='bold')

    for i, label in enumerate(labels):
        med = e1_results[label]['median']
        ax.text(i + 1, med * 1.1, f"med={med:.0f}", ha='center', fontsize=10)

    path = os.path.join(FIG, "e3_residual.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ============================================================================
# Main
# ============================================================================
def main():
    all_results = {}

    # E0
    e0 = run_e0()
    all_results["e0"] = e0
    plot_e0(e0)

    # E1
    e1 = run_e1()
    all_results["e1"] = e1
    plot_e1(e1)

    # Check E1 result
    match = e1["_comparison"]["match_within_30pct"]
    ratio = e1["_comparison"]["ratio"]

    # E3: always run (just re-presentation of E1 data)
    plot_e3(e1)

    if match:
        print("\n  E1 confirmed Q_eff hypothesis. Running E2 kappa sweep...")
        e2 = run_e2()
        all_results["e2"] = e2
        plot_e2(e2)
    else:
        print(f"\n  E1 rejected Q_eff hypothesis (ratio={ratio:.3f}). Skipping E2.")

    # Save
    path = os.path.join(OUT, "results.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSaved {path}")

    return all_results


if __name__ == "__main__":
    main()
