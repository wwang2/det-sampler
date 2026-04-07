"""nhc-qualitative-038: density + trajectory figures, NHC vs parallel multi-scale log-osc."""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import cm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)

from research.eval.potentials import GaussianMixture2D


class AnisotropicGaussian:
    name = "anisotropic_gaussian"
    def __init__(self, kappas):
        self.kappas = np.asarray(kappas, dtype=float)
        self.dim = len(self.kappas)
    def energy(self, q):
        return 0.5 * float(np.sum(self.kappas * q * q))
    def gradient(self, q):
        return self.kappas * q


def g_func(xi):
    return 2.0 * xi / (1.0 + xi * xi)


def simulate_multiscale(potential, Qs, dt, n_steps, kT=1.0, mass=1.0, seed=0,
                        record_every=1, record_thermo=False):
    rng = np.random.default_rng(seed)
    dim = potential.dim
    Qs = np.asarray(Qs, dtype=float)
    N = len(Qs)
    if hasattr(potential, "kappas"):
        q = rng.normal(0, 1.0, size=dim) / np.sqrt(np.maximum(potential.kappas, 1e-6))
    else:
        q = rng.normal(0, 1.0, size=dim)
    p = rng.normal(0, np.sqrt(mass * kT), size=dim)
    xi = np.zeros(N)
    half = 0.5 * dt
    grad_U = potential.gradient(q)
    n_rec = n_steps // record_every
    qs_rec = np.empty((n_rec, dim))
    xi_rec = np.empty((n_rec, N)) if record_thermo else None
    fr_rec = np.empty(n_rec) if record_thermo else None
    rec_i = 0
    for step in range(n_steps):
        K = float(np.sum(p * p)) / mass
        xi = xi + half * (K - dim * kT) / Qs
        gtot = float(np.sum(g_func(xi)))
        p = p * np.exp(-gtot * half)
        p = p - half * grad_U
        q = q + dt * p / mass
        grad_U = potential.gradient(q)
        p = p - half * grad_U
        gtot = float(np.sum(g_func(xi)))
        p = p * np.exp(-gtot * half)
        K = float(np.sum(p * p)) / mass
        xi = xi + half * (K - dim * kT) / Qs
        if (step + 1) % record_every == 0 and rec_i < n_rec:
            qs_rec[rec_i] = q
            if record_thermo:
                xi_rec[rec_i] = xi
                fr_rec[rec_i] = float(np.sum(g_func(xi)))
            rec_i += 1
        if not np.isfinite(p).all() or not np.isfinite(q).all():
            break
    if record_thermo:
        return qs_rec[:rec_i], xi_rec[:rec_i], fr_rec[:rec_i]
    return qs_rec[:rec_i]


def simulate_nhc(potential, Qs, dt, n_steps, kT=1.0, mass=1.0, seed=0,
                 record_every=1, record_thermo=False):
    rng = np.random.default_rng(seed)
    dim = potential.dim
    Qs = np.asarray(Qs, dtype=float)
    M = len(Qs)
    if hasattr(potential, "kappas"):
        q = rng.normal(0, 1.0, size=dim) / np.sqrt(np.maximum(potential.kappas, 1e-6))
    else:
        q = rng.normal(0, 1.0, size=dim)
    p = rng.normal(0, np.sqrt(mass * kT), size=dim)
    xi = np.zeros(M)
    half = 0.5 * dt
    grad_U = potential.gradient(q)
    n_rec = n_steps // record_every
    qs_rec = np.empty((n_rec, dim))
    xi_rec = np.empty((n_rec, M)) if record_thermo else None
    fr_rec = np.empty(n_rec) if record_thermo else None
    rec_i = 0
    def chain_dxi(p_val, xi_val):
        d = np.zeros(M)
        K = float(np.sum(p_val * p_val)) / mass
        d[0] = (K - dim * kT) / Qs[0]
        if M > 1:
            d[0] -= xi_val[1] * xi_val[0]
        for i in range(1, M):
            G = Qs[i - 1] * xi_val[i - 1] ** 2 - kT
            d[i] = G / Qs[i]
            if i < M - 1:
                d[i] -= xi_val[i + 1] * xi_val[i]
        return d
    for step in range(n_steps):
        xi = xi + half * chain_dxi(p, xi)
        p = p * np.exp(-xi[0] * half)
        p = p - half * grad_U
        q = q + dt * p / mass
        grad_U = potential.gradient(q)
        p = p - half * grad_U
        p = p * np.exp(-xi[0] * half)
        xi = xi + half * chain_dxi(p, xi)
        if (step + 1) % record_every == 0 and rec_i < n_rec:
            qs_rec[rec_i] = q
            if record_thermo:
                xi_rec[rec_i] = xi
                fr_rec[rec_i] = xi[0]
            rec_i += 1
        if not np.isfinite(p).all() or not np.isfinite(q).all():
            break
    if record_thermo:
        return qs_rec[:rec_i], xi_rec[:rec_i], fr_rec[:rec_i]
    return qs_rec[:rec_i]


SEED = 42
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

KAPPAS = np.array([1.0, 100.0])
gauss = AnisotropicGaussian(KAPPAS)
DT_G = 0.05 / np.sqrt(KAPPAS.max())
N_STEPS_G = 200_000

gmm = GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5)
DT_M = 0.02
N_STEPS_M = 200_000


def logu(N, lo, hi):
    return np.exp(np.linspace(np.log(lo), np.log(hi), N))


Qs_par_g = logu(3, 1.0 / np.sqrt(KAPPAS.max()), 1.0)
Qs_nhc_g = np.ones(3) * 1.0
Qs_par_m = logu(5, 0.1, 10.0)
Qs_nhc_m = np.ones(3) * 1.0

print("NHC 2D Gaussian...")
nhc_g_q, nhc_g_xi, nhc_g_fr = simulate_nhc(gauss, Qs_nhc_g, DT_G, N_STEPS_G, seed=SEED, record_every=1, record_thermo=True)
print("parallel 2D Gaussian...")
par_g_q, par_g_xi, par_g_fr = simulate_multiscale(gauss, Qs_par_g, DT_G, N_STEPS_G, seed=SEED, record_every=1, record_thermo=True)
print("NHC GMM...")
nhc_m_q = simulate_nhc(gmm, Qs_nhc_m, DT_M, N_STEPS_M, seed=SEED, record_every=1)
print("parallel GMM...")
par_m_q = simulate_multiscale(gmm, Qs_par_m, DT_M, N_STEPS_M, seed=SEED, record_every=1)

BURN = 20_000
nhc_g_qs = nhc_g_q[BURN:]
par_g_qs = par_g_q[BURN:]
nhc_m_qs = nhc_m_q[BURN:]
par_m_qs = par_m_q[BURN:]


def thin_to(arr, n):
    if len(arr) <= n:
        return arr
    idx = np.linspace(0, len(arr) - 1, n).astype(int)
    return arr[idx]


# Figure 1
print("Fig 1")
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
sigmas = 1.0 / np.sqrt(KAPPAS)
nhc_s = thin_to(nhc_g_qs, 5000)
par_s = thin_to(par_g_qs, 5000)
xlim = (-3.5, 3.5); ylim = (-0.5, 0.5)
for ax, s, title, color in [
    (axes[0], nhc_s, "NHC (M=3, Q=1)", "#ff7f0e"),
    (axes[1], par_s, "parallel multi-scale (N=3)", "#2ca02c"),
]:
    ax.scatter(s[:, 0], s[:, 1], s=3, alpha=0.35, c=color, rasterized=True)
    for k in (1, 2):
        e = Ellipse((0, 0), width=2*k*sigmas[0], height=2*k*sigmas[1],
                    fill=False, edgecolor="black", lw=1.0, ls="--", alpha=0.7)
        ax.add_patch(e)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel(r"$q_{\mathrm{slow}}$ ($\kappa=1$)")
    ax.set_ylabel(r"$q_{\mathrm{fast}}$ ($\kappa=100$)")
    ax.set_title(title)
fig.suptitle(r"2D anisotropic Gaussian: 5000 samples (seed=42). Dashed: 1$\sigma$, 2$\sigma$ contours.")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "density_2d_gauss.png"), dpi=150)
plt.close(fig)

# Figure 2
print("Fig 2")
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
nhc_s = thin_to(nhc_m_qs, 5000)
par_s = thin_to(par_m_qs, 5000)
centers = gmm.centers
colors_modes = cm.tab10(np.arange(5))
gx = np.linspace(-5, 5, 200); gy = np.linspace(-5, 5, 200)
GX, GY = np.meshgrid(gx, gy)
dens = np.zeros_like(GX)
for k in range(5):
    d2 = (GX - centers[k, 0])**2 + (GY - centers[k, 1])**2
    dens += np.exp(-0.5 * d2 / gmm.sigma**2)
for ax, s, title in [
    (axes[0], nhc_s, "NHC (M=3, Q=1)"),
    (axes[1], par_s, "parallel multi-scale (N=5)"),
]:
    ax.contour(GX, GY, dens, levels=6, colors="gray", alpha=0.4, linewidths=0.6)
    d2 = np.sum((s[:, None, :] - centers[None, :, :])**2, axis=2)
    assign = np.argmin(d2, axis=1)
    for k in range(5):
        m = assign == k
        ax.scatter(s[m, 0], s[m, 1], s=4, alpha=0.45, c=[colors_modes[k]], rasterized=True)
    ax.scatter(centers[:, 0], centers[:, 1], marker="x", s=140, c="black", lw=2.5, zorder=10)
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_aspect("equal")
    ax.set_xlabel(r"$q_x$"); ax.set_ylabel(r"$q_y$"); ax.set_title(title)
fig.suptitle("2D Gaussian mixture (5 modes, ring): 5000 samples colored by nearest mode.")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "density_gmm.png"), dpi=150)
plt.close(fig)

# Figure 3
print("Fig 3")
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
N_TRAJ = 2000
nhc_t = nhc_g_qs[:N_TRAJ]; par_t = par_g_qs[:N_TRAJ]
t = np.arange(N_TRAJ)
for ax, traj, title in [
    (axes[0], nhc_t, "NHC (M=3)"),
    (axes[1], par_t, "parallel multi-scale (N=3)"),
]:
    ax.plot(traj[:, 0], traj[:, 1], color="gray", lw=0.3, alpha=0.4)
    ax.scatter(traj[:, 0], traj[:, 1], c=t, cmap="viridis", s=4, alpha=0.85, rasterized=True)
    ax.set_xlabel(r"$q_{\mathrm{slow}}$"); ax.set_ylabel(r"$q_{\mathrm{fast}}$")
    ax.set_title(title); ax.set_xlim(-3.5, 3.5); ax.set_ylim(-0.5, 0.5)
fig.suptitle("Trajectory in (q_slow, q_fast), 2000 consecutive steps; color = time")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "traj_2d_gauss.png"), dpi=150)
plt.close(fig)

# Figure 4
print("Fig 4")
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
N_TRAJ = 5000
nhc_t = nhc_m_qs[:N_TRAJ]; par_t = par_m_qs[:N_TRAJ]
t = np.arange(N_TRAJ)
for ax, traj, title in [
    (axes[0], nhc_t, "NHC (M=3)"),
    (axes[1], par_t, "parallel multi-scale (N=5)"),
]:
    ax.plot(traj[:, 0], traj[:, 1], color="gray", lw=0.3, alpha=0.5)
    ax.scatter(traj[:, 0], traj[:, 1], c=t, cmap="viridis", s=4, alpha=0.75, rasterized=True)
    ax.scatter(centers[:, 0], centers[:, 1], marker="x", s=140, c="red", lw=2.5, zorder=10)
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_aspect("equal")
    ax.set_xlabel(r"$q_x$"); ax.set_ylabel(r"$q_y$"); ax.set_title(title)
fig.suptitle("Trajectory on 2D GMM, 5000 consecutive steps; color = time")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "traj_gmm.png"), dpi=150)
plt.close(fig)

# Figure 5
print("Fig 5")
Qs_p5 = logu(5, 1.0 / np.sqrt(KAPPAS.max()), 1.0)
par5_q, par5_xi, par5_fr = simulate_multiscale(gauss, Qs_p5, DT_G, 50_000, seed=SEED, record_every=1, record_thermo=True)
N_T = 2000
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
t_axis = np.arange(N_T) * DT_G
nhc_xi_p = nhc_g_xi[BURN:BURN + N_T]
for i in range(3):
    axes[0].plot(t_axis, nhc_xi_p[:, i], lw=0.9, label=fr"$\xi_{i+1}$")
axes[0].set_ylabel(r"NHC $\xi_i(t)$")
axes[0].legend(loc="upper right", fontsize=9, ncol=3)
axes[0].set_title("NHC(M=3): single dominant timescale")

if len(par5_xi) >= BURN + N_T:
    par_xi_p = par5_xi[BURN:BURN + N_T]
    par_fr_p = par5_fr[BURN:BURN + N_T]
else:
    par_xi_p = par5_xi[:N_T]
    par_fr_p = par5_fr[:N_T]
cmap = cm.viridis(np.linspace(0, 0.9, 5))
for i in range(5):
    axes[1].plot(t_axis[:len(par_xi_p)], par_xi_p[:, i], lw=0.9, color=cmap[i],
                 label=fr"$Q_{i+1}={Qs_p5[i]:.2f}$")
axes[1].set_ylabel(r"parallel $\xi_i(t)$")
axes[1].legend(loc="upper right", fontsize=8, ncol=5)
axes[1].set_title("parallel multi-scale (N=5): coexisting timescales")

nhc_fr_p = nhc_g_fr[BURN:BURN + N_T]
axes[2].plot(t_axis, nhc_fr_p, lw=0.9, color="#ff7f0e", label=r"NHC: $\xi_1$", alpha=0.85)
axes[2].plot(t_axis[:len(par_fr_p)], par_fr_p, lw=0.9, color="#2ca02c",
             label=r"parallel: $\sum g(\xi_i)$", alpha=0.85)
axes[2].set_ylabel(r"friction $\Gamma(t)$")
axes[2].set_xlabel("time")
axes[2].legend(loc="upper right", fontsize=9)
axes[2].axhline(0, color="black", lw=0.5, alpha=0.4)
fig.suptitle("Thermostat internal variables: NHC vs parallel multi-scale on 2D Gaussian")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "traj_thermostat_vars.png"), dpi=150)
plt.close(fig)

# Figure 6
print("Fig 6")
def cumulative_occupation(traj, centers):
    d2 = np.sum((traj[:, None, :] - centers[None, :, :])**2, axis=2)
    assign = np.argmin(d2, axis=1)
    n_modes = centers.shape[0]
    counts = np.zeros((len(traj), n_modes))
    for k in range(n_modes):
        counts[:, k] = np.cumsum(assign == k)
    totals = np.arange(1, len(traj) + 1)[:, None]
    return counts / totals

occ_nhc = cumulative_occupation(nhc_m_qs, centers)
occ_par = cumulative_occupation(par_m_qs, centers)
fevals = np.arange(1, len(nhc_m_qs) + 1) + BURN
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, occ, title in [
    (axes[0], occ_nhc, "NHC (M=3)"),
    (axes[1], occ_par, "parallel multi-scale (N=5)"),
]:
    for k in range(5):
        ax.plot(fevals, occ[:, k], color=colors_modes[k], lw=1.2, label=f"mode {k}")
    ax.axhline(0.2, ls="--", color="black", lw=1.0, alpha=0.6, label="target 1/5")
    ax.set_xlabel("force evaluations"); ax.set_title(title); ax.set_ylim(0, 1)
axes[0].set_ylabel("cumulative fraction in mode")
axes[1].legend(loc="upper right", fontsize=8, ncol=2)
fig.suptitle("Mode occupation convergence on 2D GMM")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "mode_occupation_gmm.png"), dpi=150)
plt.close(fig)

print("All 6 figures written to", FIGDIR)
