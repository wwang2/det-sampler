"""
symmetry-protection-066: Post-processing diagnostics on orbit 064 ensemble.

Two tests on the existing 200-trajectory NH-tanh ensemble:

  H1. Crooks Detailed Fluctuation Theorem (DFT):
      For entropy production sigma obeying time-reversal symmetry,
          log P(+s) / P(-s) = s
      If the slope of log-ratio-vs-s is 1.0, then sigma_bath's bounded
      variance is symmetry-protected (fluctuation-dissipation), not a
      happy accident of parameter choice.

  H4. Kraskov k-NN Mutual Information:
      The 0.044 linear correlation between (sigma_bath-sigma_exact) and
      (sigma_hutch-sigma_exact) could mask nonlinear dependence. We
      estimate I(X;Y) with Kraskov-Stögbauer-Grassberger estimator 1
      (k=5) and compare to a shuffle null.

No new simulation. Reads ensemble_sigmas.npz from orbit 064.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.special import digamma

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/Users/wujiewang/code/det-sampler/.worktrees/symmetry-protection-066")
PARENT_DATA = Path(
    "/Users/wujiewang/code/det-sampler/.worktrees/triple-identity-064"
    "/orbits/triple-identity-064/results/ensemble_sigmas.npz"
)
PARENT_SUMMARY = Path(
    "/Users/wujiewang/code/det-sampler/.worktrees/triple-identity-064"
    "/orbits/triple-identity-064/results/ensemble_summary.json"
)
ORBIT = ROOT / "orbits" / "symmetry-protection-066"
RESULTS = ORBIT / "results"
FIGURES = ORBIT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260411)

# ---------------------------------------------------------------------------
# Phase 0 — load + verify
# ---------------------------------------------------------------------------
print("=" * 72)
print("Phase 0 — load + verify")
print("=" * 72)
with np.load(PARENT_DATA) as d:
    times = d["times"].copy()  # (5001,)
    sigma_exact = d["sigma_exact"].copy()  # (200, 5001)
    sigma_bath = d["sigma_bath"].copy()
    sigma_hutch = d["sigma_hutch"].copy()

with open(PARENT_SUMMARY) as f:
    summary = json.load(f)

N_TRAJ = sigma_exact.shape[0]
print(f"Loaded {N_TRAJ} trajectories over {len(times)} timesteps, "
      f"t in [{times[0]}, {times[-1]}]")

# Verify numbers match summary
dev_bath_T = sigma_bath[:, -1] - sigma_exact[:, -1]
dev_hutch_T = sigma_hutch[:, -1] - sigma_exact[:, -1]

std_bath = dev_bath_T.std(ddof=1)
std_hutch = dev_hutch_T.std(ddof=1)
ratio = std_bath / std_hutch
corr = np.corrcoef(dev_bath_T, dev_hutch_T)[0, 1]

print(f"std_bath     = {std_bath:.4f}  (expected {summary['std_bath_minus_exact_final']:.4f})")
print(f"std_hutch    = {std_hutch:.4f}  (expected {summary['std_hutch_minus_exact_final']:.4f})")
print(f"ratio        = {ratio:.4f}  (expected {summary['ratio_bath_over_hutch_std_final']:.4f})")
print(f"corr         = {corr:.4f}  (expected {summary['corr_bath_hutch_dev_final']:.4f})")

assert np.isclose(std_bath, summary["std_bath_minus_exact_final"], atol=1e-6)
assert np.isclose(std_hutch, summary["std_hutch_minus_exact_final"], atol=1e-6)
print("  verification PASSED — sanity check numbers match summary.")

# time indices for t = 5, 10, 15, 20, 25
t_targets = [5.0, 10.0, 15.0, 20.0, 25.0]
t_idx = {t: int(np.argmin(np.abs(times - t))) for t in t_targets}
print(f"\nTime indices for DFT evaluation: {t_idx}")

# ---------------------------------------------------------------------------
# Phase 1 — Crooks DFT on windowed increments of (σ_bath - σ_exact)
# ---------------------------------------------------------------------------
# A proper fluctuation theorem test uses samples of a (quasi-)stationary
# stochastic process, not point-in-time values from an ensemble with a
# residual startup bias. We extract Δσ_bath - Δσ_exact over NON-OVERLAPPING
# windows of duration Δt from t ≥ 5 (post pre-thermalization), pooled across
# all 200 trajectories. This gives O(1000) samples of a near-zero-mean,
# near-symmetric quantity — exactly what Crooks expects.
print("\n" + "=" * 72)
print("Phase 1 — Crooks DFT on windowed increments Δ(σ_bath - σ_exact)")
print("=" * 72)


def crooks_slope(delta: np.ndarray, n_bins: int = 15) -> dict:
    """Compute DFT slope: log[P(+s)/P(-s)] vs s.

    Returns slope (weighted LS), intercept, and bin data for plotting.
    """
    # Symmetric binning centered at 0
    s_max = np.percentile(np.abs(delta), 95)
    edges = np.linspace(-s_max, s_max, n_bins + 1)
    # Enforce symmetry: use edges symmetric around 0
    counts, _ = np.histogram(delta, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Pair bin i (center +s_i) with bin (n_bins-1-i) (center -s_i)
    # The histogram is already symmetric-around-zero if n_bins is odd; here n_bins=15 odd.
    mid = n_bins // 2  # 7 — the zero bin
    s_vals = []
    log_ratios = []
    weights = []
    for i in range(mid + 1, n_bins):
        j = n_bins - 1 - i  # mirror bin
        s = centers[i]
        cp = counts[i]
        cm = counts[j]
        if cp > 0 and cm > 0:
            s_vals.append(s)
            log_ratios.append(np.log(cp / cm))
            # weight ~ sqrt(harmonic mean of counts), proxy for 1/SE of log-ratio
            weights.append(np.sqrt(cp * cm / (cp + cm)))
    s_vals = np.array(s_vals)
    log_ratios = np.array(log_ratios)
    weights = np.array(weights)

    if len(s_vals) < 2:
        return dict(slope=np.nan, intercept=np.nan,
                    s_vals=s_vals, log_ratios=log_ratios,
                    weights=weights, edges=edges, counts=counts, centers=centers)

    # Weighted least squares: y = a*s + b
    W = np.diag(weights ** 2)
    X = np.column_stack([s_vals, np.ones_like(s_vals)])
    beta, *_ = np.linalg.lstsq(weights[:, None] * X, weights * log_ratios, rcond=None)
    slope, intercept = beta

    return dict(
        slope=float(slope),
        intercept=float(intercept),
        s_vals=s_vals,
        log_ratios=log_ratios,
        weights=weights,
        edges=edges,
        counts=counts,
        centers=centers,
    )


def bootstrap_slope(delta: np.ndarray, n_boot: int = 1000, n_bins: int = 15) -> tuple:
    """Bootstrap 95% CI for DFT slope."""
    slopes = np.empty(n_boot)
    N = len(delta)
    idx_all = RNG.integers(0, N, size=(n_boot, N))
    for b in range(n_boot):
        try:
            r = crooks_slope(delta[idx_all[b]], n_bins=n_bins)
            slopes[b] = r["slope"]
        except Exception:
            slopes[b] = np.nan
    slopes = slopes[~np.isnan(slopes)]
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return float(lo), float(hi), slopes


def windowed_deltas(window_dt: float, t_min: float = 5.0) -> np.ndarray:
    """Non-overlapping windowed increments of (σ_bath - σ_exact) over all
    trajectories, from t >= t_min. Returns a flat array."""
    dt = times[1] - times[0]
    stride = int(round(window_dt / dt))
    start = int(round(t_min / dt))
    idxs = np.arange(start, len(times), stride)
    if len(idxs) < 2:
        return np.array([])
    inc_bath = sigma_bath[:, idxs[1:]] - sigma_bath[:, idxs[:-1]]
    inc_exact = sigma_exact[:, idxs[1:]] - sigma_exact[:, idxs[:-1]]
    dev = (inc_bath - inc_exact).ravel()
    return dev


# Evaluate DFT across several window sizes. DFT theory expects the slope
# to approach 1.0 for sufficiently large windows (large-deviation regime).
dft_by_window = {}
for dt_win in [2.0, 5.0, 10.0]:
    delta = windowed_deltas(dt_win, t_min=5.0)
    delta_c = delta - delta.mean()
    res = crooks_slope(delta_c, n_bins=15)
    lo, hi, _boot = bootstrap_slope(delta_c, n_boot=1000, n_bins=15)
    dft_by_window[dt_win] = dict(
        window_dt=float(dt_win),
        n_samples=int(len(delta)),
        mean=float(delta.mean()),
        std=float(delta_c.std(ddof=1)),
        slope=res["slope"],
        slope_ci=(lo, hi),
        intercept=res["intercept"],
        s_vals=res["s_vals"].tolist(),
        log_ratios=res["log_ratios"].tolist(),
        bin_edges=res["edges"].tolist(),
        bin_counts=res["counts"].tolist(),
    )
    print(f"  Δt={dt_win:4.1f}  N={len(delta):5d}  slope={res['slope']:+.3f}  "
          f"95% CI=[{lo:+.3f}, {hi:+.3f}]  mean={delta.mean():+.3f}  std={delta_c.std(ddof=1):.3f}")

# Also retain pointwise-at-t values for backward compat / cross-check
dft_by_time = {}
for t, idx in t_idx.items():
    delta = sigma_bath[:, idx] - sigma_exact[:, idx]
    delta_c = delta - delta.mean()
    res = crooks_slope(delta_c, n_bins=15)
    lo, hi, _ = bootstrap_slope(delta_c, n_boot=1000, n_bins=15)
    dft_by_time[t] = dict(
        t=t,
        n_traj=len(delta),
        mean=float(delta.mean()),
        std=float(delta.std(ddof=1)),
        slope=res["slope"],
        slope_ci=(lo, hi),
    )

# Headline: we take Δt=5 windows starting from t=5 as the primary test —
# enough sample size (800) to resolve the tails and enough window length
# to be in the large-deviation regime.
PRIMARY_WIN = 5.0
slope_T = dft_by_window[PRIMARY_WIN]["slope"]
slope_ci_T = dft_by_window[PRIMARY_WIN]["slope_ci"]
print(f"\nHeadline (Δt={PRIMARY_WIN} window): DFT slope = {slope_T:+.3f}  95% CI = {slope_ci_T}")

# Check alternative normalizations at primary window
delta_W = windowed_deltas(PRIMARY_WIN, t_min=5.0)
delta_W_c = delta_W - delta_W.mean()
print(f"\nAlternative Crooks normalizations (Δt={PRIMARY_WIN} window, N={len(delta_W)}):")
for name, scale in [
    ("Δσ (raw)", 1.0),
    ("Δσ/σ²", 1.0 / delta_W_c.var(ddof=1)),
    ("Δσ * σ²", delta_W_c.var(ddof=1)),
    ("Δσ * 2", 2.0),
    ("Δσ / 2", 0.5),
]:
    r = crooks_slope(delta_W_c * scale, n_bins=15)
    print(f"  {name:28s}  slope = {r['slope']:+.4f}")

# Gaussian check on windowed samples
from scipy.stats import skew, kurtosis, shapiro
print(f"\nGaussianity check on windowed Δ(σ_bath - σ_exact) (Δt={PRIMARY_WIN}, N={len(delta_W)}):")
print(f"  skew            = {skew(delta_W):+.3f}")
print(f"  excess kurtosis = {kurtosis(delta_W):+.3f}")
sw_stat, sw_p = shapiro(delta_W[:min(5000, len(delta_W))])
print(f"  Shapiro-Wilk stat={sw_stat:.4f}, p={sw_p:.4f}")

# Save DFT results
dft_out = dict(
    method="Crooks DFT on windowed Δ(σ_bath - σ_exact)",
    n_traj=int(N_TRAJ),
    n_bins=15,
    n_bootstrap=1000,
    primary_window_dt=float(PRIMARY_WIN),
    primary_n_samples=int(len(delta_W)),
    headline_slope=float(slope_T),
    headline_slope_CI=list(slope_ci_T),
    gaussian_skew=float(skew(delta_W)),
    gaussian_excess_kurtosis=float(kurtosis(delta_W)),
    shapiro_W=float(sw_stat),
    shapiro_p=float(sw_p),
    by_window=dft_by_window,
    by_time_pointwise=dft_by_time,
)
# Custom serializer for numpy/tuples
def _to_json(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not serializable: {type(o)}")

with open(RESULTS / "dft_bootstrap.json", "w") as f:
    json.dump(dft_out, f, indent=2, default=_to_json)
print(f"\n  DFT results written to {RESULTS / 'dft_bootstrap.json'}")


# ---------------------------------------------------------------------------
# Phase 2 — Kraskov k-NN Mutual Information
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("Phase 2 — Kraskov-Stögbauer-Grassberger MI between σ_bath and σ_hutch")
print("=" * 72)


def ksg_mi(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """Kraskov estimator (algorithm 1, KSG 2004, arXiv:cond-mat/0305641).

    I(X;Y) = ψ(k) + ψ(N) - <ψ(n_x+1) + ψ(n_y+1)>

    with Chebyshev (L∞) distances in the joint (x,y) space.

    x, y: 1-D arrays of length N (scalar variables).
    """
    x = np.atleast_2d(x).T if x.ndim == 1 else x
    y = np.atleast_2d(y).T if y.ndim == 1 else y
    N = x.shape[0]
    assert y.shape[0] == N
    # Add microscopic jitter to break ties (KSG recommends this)
    x = x + 1e-10 * RNG.standard_normal(x.shape)
    y = y + 1e-10 * RNG.standard_normal(y.shape)

    xy = np.hstack([x, y])
    # k-th neighbor in joint space under Chebyshev metric (p=inf)
    tree_xy = cKDTree(xy)
    # k+1 because the point is its own 0-th neighbor
    dists, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
    eps = dists[:, -1]  # distance to k-th neighbor (exclusive)

    # Marginals: count points in x-space with |x_i - x_j| < eps_i (strictly)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)
    # query_ball_point returns neighbors within <= r; we want < eps_i strictly
    # Per KSG: use eps_i - tiny so we get strict inequality, matching the
    # k-th neighbor definition that uses max of marginals.
    nx = np.array([len(tree_x.query_ball_point(x[i], eps[i] - 1e-15, p=np.inf)) - 1
                    for i in range(N)])
    ny = np.array([len(tree_y.query_ball_point(y[i], eps[i] - 1e-15, p=np.inf)) - 1
                    for i in range(N)])
    mi = digamma(k) + digamma(N) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return float(mi)


# At t=25
x_T = sigma_bath[:, t_idx[25.0]] - sigma_exact[:, t_idx[25.0]]
y_T = sigma_hutch[:, t_idx[25.0]] - sigma_exact[:, t_idx[25.0]]
k = 5

print(f"N = {N_TRAJ}, k = {k}")
mi_empirical = ksg_mi(x_T, y_T, k=k)
print(f"Empirical MI(σ_bath - σ_exact ; σ_hutch - σ_exact) = {mi_empirical:+.4f} nats")

# Shuffle null
n_surrogates = 500
null_mi = np.empty(n_surrogates)
for s in range(n_surrogates):
    y_shuf = RNG.permutation(y_T)
    null_mi[s] = ksg_mi(x_T, y_shuf, k=k)
null_mean = float(null_mi.mean())
null_std = float(null_mi.std(ddof=1))
excess_sigma = (mi_empirical - null_mean) / null_std
print(f"Null: mean = {null_mean:+.4f}, std = {null_std:.4f}")
print(f"Excess = {excess_sigma:+.2f} σ over shuffle null")

# Upper bound: for Gaussian with std s, H = 0.5 log(2πe s²)
h_bath = 0.5 * np.log(2 * np.pi * np.e * x_T.var(ddof=1))
h_hutch = 0.5 * np.log(2 * np.pi * np.e * y_T.var(ddof=1))
mi_upper_bound = min(h_bath, h_hutch)
print(f"Gaussian entropy upper bound min(H_bath, H_hutch) = {mi_upper_bound:.4f} nats")
print(f"  (H_bath = {h_bath:.4f}, H_hutch = {h_hutch:.4f})")

# Also at intermediate times for reference
mi_by_time = {}
for t, idx in t_idx.items():
    xT = sigma_bath[:, idx] - sigma_exact[:, idx]
    yT = sigma_hutch[:, idx] - sigma_exact[:, idx]
    mi_t = ksg_mi(xT, yT, k=k)
    # Mini null (100 shuffles) for each t
    nulls = np.empty(100)
    for s in range(100):
        nulls[s] = ksg_mi(xT, RNG.permutation(yT), k=k)
    mi_by_time[t] = dict(
        t=float(t),
        mi=float(mi_t),
        null_mean=float(nulls.mean()),
        null_std=float(nulls.std(ddof=1)),
        excess_sigma=float((mi_t - nulls.mean()) / nulls.std(ddof=1)),
        linear_corr=float(np.corrcoef(xT, yT)[0, 1]),
    )
    print(f"  t={t:5.1f}  MI={mi_t:+.4f}  null={nulls.mean():+.4f}±{nulls.std(ddof=1):.4f}  "
          f"excess={mi_by_time[t]['excess_sigma']:+.2f}σ  corr={mi_by_time[t]['linear_corr']:+.3f}")

mi_out = dict(
    method="Kraskov-Stögbauer-Grassberger MI estimator 1 (k=5)",
    n_traj=int(N_TRAJ),
    k=k,
    n_surrogates=n_surrogates,
    mi_empirical=float(mi_empirical),
    null_mean=null_mean,
    null_std=null_std,
    excess_sigma=float(excess_sigma),
    mi_upper_bound_gaussian=float(mi_upper_bound),
    by_time=mi_by_time,
)
with open(RESULTS / "mi_surrogate.json", "w") as f:
    json.dump(mi_out, f, indent=2)
print(f"\n  MI results written to {RESULTS / 'mi_surrogate.json'}")


# ---------------------------------------------------------------------------
# Phase 3 — Figure
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("Phase 3 — figure")
print("=" * 72)

mpl.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR_BATH = "#1f77b4"
COLOR_HUTCH = "#ff7f0e"
COLOR_NULL = "#888888"
COLOR_UNIT = "#d62728"

fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)

# --- Panel (a): Crooks DFT on windowed increments ---
ax = axes[0]
r = crooks_slope(delta_W_c, n_bins=15)
# log-ratio points
# Proper error on log-ratio: SE = sqrt(1/n+ + 1/n-). Rebuild from raw counts.
edges = r["edges"]
counts = r["counts"]
n_bins = len(counts)
mid = n_bins // 2
s_pts, lr_pts, lr_err = [], [], []
for i in range(mid + 1, n_bins):
    j = n_bins - 1 - i
    cp, cm = counts[i], counts[j]
    if cp > 0 and cm > 0:
        s_pts.append(0.5 * (edges[i] + edges[i + 1]))
        lr_pts.append(np.log(cp / cm))
        lr_err.append(np.sqrt(1.0 / cp + 1.0 / cm))
s_pts = np.array(s_pts)
lr_pts = np.array(lr_pts)
lr_err = np.array(lr_err)

ax.errorbar(
    s_pts,
    lr_pts,
    yerr=lr_err,
    fmt="o",
    color=COLOR_BATH,
    ms=8,
    capsize=3,
    label=r"data: $\log[P(+s)/P(-s)]$",
)
# Unit-slope reference line
s_max_plot = max(r["s_vals"]) * 1.1 if len(r["s_vals"]) else 1.0
s_ref = np.linspace(0, s_max_plot, 100)
ax.plot(s_ref, s_ref, "--", color=COLOR_UNIT, lw=2, label=r"Crooks DFT: $y = s$")
# Fitted slope line
y_fit = r["slope"] * s_ref + r["intercept"]
ax.plot(s_ref, y_fit, "-", color=COLOR_BATH, lw=1.8, alpha=0.7,
        label=f"WLS fit: slope = {r['slope']:+.3f}")
ax.axhline(0, color="k", lw=0.5, alpha=0.3)
ax.set_xlabel(r"$s = \Delta\sigma_{\rm bath} - \Delta\sigma_{\rm exact}$ (centered)")
ax.set_ylabel(r"$\log[P(+s)/P(-s)]$")
ax.set_title(f"(a) Crooks DFT on windowed increments ($\\Delta t={PRIMARY_WIN}$)")
ax.legend(loc="upper left", frameon=False, fontsize=10)
# y limits to include unit-slope reference AND data
ymax = max(s_max_plot, max(abs(lr_pts).max() if len(lr_pts) else 1, 2.0))
ax.set_ylim(-1.5, ymax * 1.05)

# Inset: histogram with Gaussian overlay
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
ax_in = inset_axes(ax, width="38%", height="32%", loc="lower right", borderpad=1.5)
ax_in.hist(delta_W_c, bins=30, density=True, color=COLOR_BATH, alpha=0.5, edgecolor="k", lw=0.5)
xg = np.linspace(delta_W_c.min(), delta_W_c.max(), 200)
sigma_g = delta_W_c.std(ddof=1)
pdfg = np.exp(-0.5 * (xg / sigma_g) ** 2) / (sigma_g * np.sqrt(2 * np.pi))
ax_in.plot(xg, pdfg, "k-", lw=1.5)
ax_in.set_xlabel("s", fontsize=9)
ax_in.set_ylabel("P", fontsize=9)
ax_in.tick_params(labelsize=8)
ax_in.set_title(r"$\Delta\sigma$ pdf + Gaussian", fontsize=9)

# annotation
ax.text(
    0.04, 0.72,
    f"slope = {slope_T:+.3f}\n"
    f"95% CI [{slope_ci_T[0]:+.2f}, {slope_ci_T[1]:+.2f}]\n"
    f"target = +1.000\n"
    f"N = {len(delta_W)}",
    transform=ax.transAxes,
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.7"),
    verticalalignment="top",
)

# --- Panel (b): MI and joint scatter ---
ax = axes[1]
# Scatter of joint distribution
ax.scatter(x_T, y_T, s=30, alpha=0.5, color=COLOR_BATH, edgecolor="k", lw=0.3)
ax.axhline(0, color="k", lw=0.5, alpha=0.3)
ax.axvline(0, color="k", lw=0.5, alpha=0.3)
ax.set_xlabel(r"$\sigma_{\rm bath} - \sigma_{\rm exact}$ at $t=25$")
ax.set_ylabel(r"$\sigma_{\rm hutch} - \sigma_{\rm exact}$ at $t=25$")
ax.set_title("(b) Joint $(\\sigma_{\\rm bath},\\sigma_{\\rm hutch})$: MI vs null")

ax.text(
    0.97, 0.97,
    f"linear corr = {corr:+.3f}\n"
    f"KSG MI = {mi_empirical:+.3f} nats\n"
    f"null = {null_mean:+.3f} ± {null_std:.3f}\n"
    f"excess = {excess_sigma:+.2f} σ\n"
    f"Gaussian bound = {mi_upper_bound:.2f}",
    transform=ax.transAxes,
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.7"),
    verticalalignment="top",
    horizontalalignment="right",
)

# Inset: null distribution vs empirical (lower-left)
ax_in2 = inset_axes(ax, width="38%", height="32%", loc="lower left", borderpad=1.5)
ax_in2.hist(null_mi, bins=30, color=COLOR_NULL, alpha=0.7, edgecolor="k", lw=0.3)
ax_in2.axvline(mi_empirical, color=COLOR_UNIT, lw=2, label="empirical")
ax_in2.axvline(null_mean, color="k", lw=1, linestyle="--", alpha=0.5, label="null mean")
ax_in2.set_xlabel("MI (nats)", fontsize=9)
ax_in2.set_ylabel("count", fontsize=9)
ax_in2.tick_params(labelsize=8)
ax_in2.set_title("null vs empirical", fontsize=9)
ax_in2.legend(fontsize=7, frameon=False, loc="upper right")

plt.suptitle(
    "Symmetry protection probes on orbit 064 ensemble (N=200, NH-tanh double well, t=25)",
    fontsize=13, y=1.04, weight="bold",
)

out_pdf = FIGURES / "fig_symmetry_protection.pdf"
out_png = FIGURES / "fig_symmetry_protection.png"
fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.2)
fig.savefig(out_png, bbox_inches="tight", pad_inches=0.2)
plt.close(fig)
print(f"  figure saved: {out_pdf}")
print(f"  figure saved: {out_png}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)

# Verdict
# A more honest test: both "is slope in [0.5, 1.5]?" (strong pass) and "does
# CI contain 1.0?" (weak pass).
dft_strong_pass = 0.5 <= slope_T <= 1.5
dft_ci_contains_unity = slope_ci_T[0] <= 1.0 <= slope_ci_T[1]
dft_ci_contains_zero = slope_ci_T[0] <= 0.0 <= slope_ci_T[1]
mi_pass = abs(excess_sigma) < 2.0
print(f"DFT (Δt={PRIMARY_WIN}) slope = {slope_T:+.3f}  95% CI = [{slope_ci_T[0]:+.2f}, {slope_ci_T[1]:+.2f}]")
print(f"  Strong pass (slope in [0.5, 1.5]): {dft_strong_pass}")
print(f"  Weak pass (CI contains 1.0): {dft_ci_contains_unity}")
print(f"  CI also contains 0: {dft_ci_contains_zero} (if True, test is inconclusive)")
print(f"KSG MI at t=25   : {mi_empirical:+.4f}  excess = {excess_sigma:+.2f} σ over shuffle null")
print(f"  Independence passes (|excess| < 2σ): {mi_pass}")

if dft_strong_pass and mi_pass:
    verdict = "symmetry-protected AND independent (H1 + H4 both pass)"
elif dft_ci_contains_unity and not dft_strong_pass:
    verdict = ("DFT INCONCLUSIVE (point estimate far from 1 but CI covers 1 — "
               "too few tail samples); independence " + ("confirmed" if mi_pass else "FAILED"))
elif (not dft_ci_contains_unity) and mi_pass:
    verdict = "independent but NOT symmetry-protected (H4 pass, H1 REJECTED)"
else:
    verdict = "NEITHER symmetry-protected nor independent (H1 + H4 both fail)"
print(f"Verdict: {verdict}")
print()
print("done.")
