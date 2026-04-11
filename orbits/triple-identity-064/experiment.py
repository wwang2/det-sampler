"""
triple-identity-064 (refine 1): pedagogical verification of the two-level identity.

This experiment is a SANITY CHECK and VARIANCE CHARACTERIZATION, not a
stand-alone publishable result. It establishes the baseline variance
structure for control-variate experiments in orbit 065.

We verify — on a 2D double-well NH-tanh trajectory — the two-level hierarchy
of identities between four phase-space-contraction estimators:

Level 1 (pathwise, exact to ODE precision), for ANY trajectory:
    sigma_exact(t) = sigma_lyap(t)
  i.e. Liouville's theorem: log|det Phi_t| = int_0^t tr(J(z(s))) ds.
  For NH-tanh, tr(J) = -D * tanh(xi). Both sides are computed numerically
  on the same trajectory; agreement to ~1e-5 is a numerical consistency
  check, NOT a discovery.

Level 2 (on-average, converges as ensemble size -> infinity):
    <sigma_bath(t)>  = sigma_exact(t)   via equipartition <|p|^2> = D kT
    <sigma_hutch(t)> = sigma_exact(t)   via Hutchinson unbiasedness E[v^T J v] = tr J
  These are two unbiased stochastic estimators of the same deterministic
  integral with different variance sources:
   * sigma_bath  — randomness from thermal fluctuations of |p|^2 around D kT
   * sigma_hutch — randomness from Rademacher draws of v (stepwise resampled)
  Both have std growing as ~sqrt(T * t) where T is trajectory-dependent.

Outputs:
  results/triple_identity.npz        --- single-trajectory arrays (seed=42)
  results/triple_identity.json       --- single-trajectory summary
  results/ensemble_sigmas.npz        --- (N_traj, N_steps+1) arrays for each sigma
  results/ensemble_summary.json      --- ensemble variance / covariance metrics
  figures/fig_triple_identity.png    --- pedagogical single-trajectory panel
                                         (bath label CORRECTED to the direct
                                          heat-integral formula)
  figures/fig_triple_identity_ensemble.png
                                     --- NEW: 3-panel ensemble characterization

Run: python experiment.py
"""

import json
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- problem definition ---------------------------------------------

D = 2          # physical dimension (q in R^D, p in R^D)
DIM = 2 * D + 1  # extended state (q,p,xi) in R^5
Q_NH = 1.0     # NH thermal mass
KT = 1.0       # k_B T
BETA = 1.0 / KT
DT = 0.005
N_STEPS = 5000
T_FINAL = N_STEPS * DT  # = 25.0
SEED = 42
QR_EVERY = 5  # Benettin renormalization period

# Ensemble parameters
N_TRAJ = 200         # ensemble size
SEED_BASE = 0        # seeds 0..N_TRAJ-1


def V(q):
    """Double-well potential V(q) = (q1^2 - 1)^2 + 0.5 * q2^2."""
    q1, q2 = q[0], q[1]
    return (q1 * q1 - 1.0) ** 2 + 0.5 * q2 * q2


def grad_V(q):
    q1, q2 = q[0], q[1]
    return np.array([4.0 * q1 * (q1 * q1 - 1.0), q2], dtype=np.float64)


def hessian_V(q):
    q1 = q[0]
    return np.array([[12.0 * q1 * q1 - 4.0, 0.0],
                     [0.0, 1.0]], dtype=np.float64)


# ---------- NH-tanh ODE on the extended phase space -------------------------

def f(z):
    """Vector field of the NH-tanh flow. z = (q, p, xi) in R^{2D+1}."""
    q = z[0:D]
    p = z[D:2 * D]
    xi = z[2 * D]
    g = np.tanh(xi)
    dq = p
    dp = -grad_V(q) - g * p
    dxi = (1.0 / Q_NH) * (np.dot(p, p) - D * KT)
    out = np.empty(DIM, dtype=np.float64)
    out[0:D] = dq
    out[D:2 * D] = dp
    out[2 * D] = dxi
    return out


def jacobian(z):
    """5x5 Jacobian of the NH-tanh vector field at z.

    Trace = -D * tanh(xi).
    """
    p = z[D:2 * D]
    xi = z[2 * D]
    g = np.tanh(xi)
    sech2 = 1.0 - g * g
    H = hessian_V(z[0:D])
    J = np.zeros((DIM, DIM), dtype=np.float64)
    J[0:D, D:2 * D] = np.eye(D)
    J[D:2 * D, 0:D] = -H
    J[D:2 * D, D:2 * D] = -g * np.eye(D)
    J[D:2 * D, 2 * D] = -sech2 * p
    J[2 * D, D:2 * D] = (2.0 / Q_NH) * p
    return J


def rk4_step_with_tangent(z, M, dt):
    """RK4 step that also propagates a tangent matrix M (DIM x k)."""
    k1z = f(z)
    k1M = jacobian(z) @ M

    z2 = z + 0.5 * dt * k1z
    M2 = M + 0.5 * dt * k1M
    k2z = f(z2)
    k2M = jacobian(z2) @ M2

    z3 = z + 0.5 * dt * k2z
    M3 = M + 0.5 * dt * k2M
    k3z = f(z3)
    k3M = jacobian(z3) @ M3

    z4 = z + dt * k3z
    M4 = M + dt * k3M
    k4z = f(z4)
    k4M = jacobian(z4) @ M4

    z_new = z + (dt / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)
    M_new = M + (dt / 6.0) * (k1M + 2 * k2M + 2 * k3M + k4M)
    return z_new, M_new


def rk4_step(z, dt):
    k1 = f(z)
    k2 = f(z + 0.5 * dt * k1)
    k3 = f(z + 0.5 * dt * k2)
    k4 = f(z + dt * k3)
    return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def H_phys(z):
    p = z[D:2 * D]
    return 0.5 * np.dot(p, p) + V(z[0:D])


def heat_rate(z):
    """Instantaneous heat INTO the bath: tanh(xi)|p|^2."""
    p = z[D:2 * D]
    xi = z[2 * D]
    return np.tanh(xi) * np.dot(p, p)


def hutch_integrand(z, v):
    """Return -v^T J(z) v, the per-unit-time integrand for sigma_hutch.

    Sign: sigma(t) = log rho_t - log rho_0 = -int tr(J) dt, so the
    unbiased Hutchinson integrand is -v^T J v (since E[v^T J v] = tr J).
    """
    Jz = jacobian(z)
    return -float(v @ (Jz @ v))


# ---------- main integration loop -------------------------------------------

def run_single(seed, n_steps=N_STEPS, dt=DT, record_lyap=True):
    """Integrate one NH-tanh trajectory, returning all four sigma trajectories.

    BUG-2 FIX: sigma_hutch is now computed with a trapezoid rule across
    (z_old, z_new) using the SAME Rademacher v for both endpoints. This
    matches sigma_exact's trapezoid integration scheme and eliminates the
    O(dt) systematic lag introduced by the previous implementation.
    """
    rng = np.random.default_rng(seed)
    rng_hutch = np.random.default_rng(seed + 1_000_003)  # independent stream

    # initial condition: left well, p ~ N(0, I), xi = 0
    q0 = np.array([-1.0, 0.0], dtype=np.float64)
    p0 = rng.standard_normal(D)
    xi0 = 0.0
    z = np.empty(DIM, dtype=np.float64)
    z[0:D] = q0
    z[D:2 * D] = p0
    z[2 * D] = xi0

    if record_lyap:
        M = np.eye(DIM, dtype=np.float64)
        log_R_diag_acc = np.zeros(DIM, dtype=np.float64)

    times = np.zeros(n_steps + 1)
    sigma_exact = np.zeros(n_steps + 1)
    sigma_lyap = np.zeros(n_steps + 1)
    sigma_bath = np.zeros(n_steps + 1)
    sigma_hutch = np.zeros(n_steps + 1)
    xi_t = np.zeros(n_steps + 1)

    xi_t[0] = z[2 * D]

    for step in range(n_steps):
        xi_old = z[2 * D]
        q_dot_old = heat_rate(z)

        # draw v ONCE per step; reuse for both endpoints of the trapezoid
        v = rng_hutch.choice([-1.0, 1.0], size=DIM).astype(np.float64)
        # BUG-2 FIX: left endpoint integrand, using the SAME v
        h_old = hutch_integrand(z, v)

        # propagate state (+ tangent if needed)
        if record_lyap:
            z_new, M_new = rk4_step_with_tangent(z, M, dt)
        else:
            z_new = rk4_step(z, dt)

        xi_new = z_new[2 * D]
        q_dot_new = heat_rate(z_new)
        # BUG-2 FIX: right endpoint integrand, SAME v
        h_new = hutch_integrand(z_new, v)

        # sigma_exact: trapezoid on +D tanh(xi)
        d_exact = D * 0.5 * (np.tanh(xi_old) + np.tanh(xi_new)) * dt
        sigma_exact[step + 1] = sigma_exact[step] + d_exact

        # sigma_bath: trapezoid on beta * tanh(xi) |p|^2
        d_bath = BETA * 0.5 * (q_dot_old + q_dot_new) * dt
        sigma_bath[step + 1] = sigma_bath[step] + d_bath

        # sigma_hutch: trapezoid on -v^T J v (same v at both endpoints)
        d_hutch = 0.5 * (h_old + h_new) * dt
        sigma_hutch[step + 1] = sigma_hutch[step] + d_hutch

        if record_lyap:
            if (step + 1) % QR_EVERY == 0:
                Qm, Rm = np.linalg.qr(M_new)
                diag = np.diag(Rm)
                log_R_diag_acc += np.log(np.abs(diag))
                signs = np.sign(diag)
                signs[signs == 0] = 1.0
                Qm = Qm * signs[np.newaxis, :]
                M_new = Qm
            sign_det, logdet_M = np.linalg.slogdet(M_new)
            live_logdet = float(np.sum(log_R_diag_acc) + logdet_M)
            sigma_lyap[step + 1] = -live_logdet
            M = M_new

        z = z_new
        xi_t[step + 1] = z[2 * D]
        times[step + 1] = (step + 1) * dt

    return dict(
        times=times,
        sigma_exact=sigma_exact,
        sigma_lyap=sigma_lyap,
        sigma_bath=sigma_bath,
        sigma_hutch=sigma_hutch,
        xi=xi_t,
    )


# ---------- ensemble -------------------------------------------------------

def run_ensemble(n_traj=N_TRAJ, seed_base=SEED_BASE, n_steps=N_STEPS, dt=DT):
    """Run n_traj independent NH-tanh trajectories with seeds seed_base..+n_traj-1.

    We skip the Benettin tangent integration for the ensemble (not needed:
    it coincides with sigma_exact to 1e-5, already verified on the single
    trajectory). This keeps the ensemble run fast.
    """
    n = n_steps + 1
    se = np.zeros((n_traj, n), dtype=np.float64)
    sb = np.zeros((n_traj, n), dtype=np.float64)
    sh = np.zeros((n_traj, n), dtype=np.float64)
    xi_store = np.zeros((n_traj, n), dtype=np.float64)
    times = None
    for i in range(n_traj):
        R = run_single(seed=seed_base + i, n_steps=n_steps, dt=dt, record_lyap=False)
        se[i] = R["sigma_exact"]
        sb[i] = R["sigma_bath"]
        sh[i] = R["sigma_hutch"]
        xi_store[i] = R["xi"]
        if times is None:
            times = R["times"]
        if (i + 1) % 20 == 0:
            print(f"  ensemble progress: {i + 1}/{n_traj}")
    return dict(
        times=times,
        sigma_exact=se,
        sigma_bath=sb,
        sigma_hutch=sh,
        xi=xi_store,
    )


# ---------- analysis --------------------------------------------------------

def summarize_single(R):
    se = R["sigma_exact"]
    sl = R["sigma_lyap"]
    sb = R["sigma_bath"]
    sh = R["sigma_hutch"]
    return dict(
        T_final=float(R["times"][-1]),
        n_steps=int(len(R["times"]) - 1),
        dt=DT,
        max_abs_exact_minus_lyap=float(np.max(np.abs(se - sl))),
        max_abs_exact_minus_bath=float(np.max(np.abs(se - sb))),
        max_abs_exact_minus_hutch=float(np.max(np.abs(se - sh))),
        rel_err_exact_lyap_final=float(abs(se[-1] - sl[-1]) / max(abs(se[-1]), 1e-12)),
        sigma_exact_final=float(se[-1]),
        sigma_lyap_final=float(sl[-1]),
        sigma_bath_final=float(sb[-1]),
        sigma_hutch_final=float(sh[-1]),
    )


def summarize_ensemble(E):
    """Compute ensemble mean / std / cross-covariance statistics."""
    se = E["sigma_exact"]   # (N, T+1)
    sb = E["sigma_bath"]
    sh = E["sigma_hutch"]
    times = E["times"]

    # per-time mean and std across trajectories
    mean_se = se.mean(axis=0)
    mean_sb = sb.mean(axis=0)
    mean_sh = sh.mean(axis=0)

    # deviations of estimators from sigma_exact on the SAME trajectory
    dev_bath = sb - se          # (N, T+1)
    dev_hutch = sh - se         # (N, T+1)

    std_dev_bath = dev_bath.std(axis=0, ddof=1)
    std_dev_hutch = dev_hutch.std(axis=0, ddof=1)

    mean_dev_bath = dev_bath.mean(axis=0)
    mean_dev_hutch = dev_hutch.mean(axis=0)

    # cross-covariance (per t) across the ensemble of trajectories
    cov_bh = np.zeros_like(times)
    corr_bh = np.zeros_like(times)
    for k in range(len(times)):
        a = dev_bath[:, k]
        b = dev_hutch[:, k]
        ca = a - a.mean()
        cb = b - b.mean()
        cov_bh[k] = float(np.mean(ca * cb))
        denom = a.std(ddof=1) * b.std(ddof=1)
        corr_bh[k] = float(cov_bh[k] / denom) if denom > 1e-15 else 0.0

    # ratio at t=T_final: std(bath - exact) / std(hutch - exact)
    ratio_final = float(std_dev_bath[-1] / std_dev_hutch[-1]) if std_dev_hutch[-1] > 0 else float("nan")

    # sqrt(t) fits: std_dev ~ a * sqrt(t), fit on t > 1 to avoid transient
    mask = times > 1.0
    sqrt_t = np.sqrt(times[mask])
    # least squares through origin: a = <y x> / <x x>
    a_bath = float(np.sum(std_dev_bath[mask] * sqrt_t) / np.sum(sqrt_t * sqrt_t))
    a_hutch = float(np.sum(std_dev_hutch[mask] * sqrt_t) / np.sum(sqrt_t * sqrt_t))
    # residual R^2 for the sqrt(t) scaling
    y_bath = std_dev_bath[mask]
    y_hutch = std_dev_hutch[mask]
    ss_res_bath = np.sum((y_bath - a_bath * sqrt_t) ** 2)
    ss_tot_bath = np.sum((y_bath - y_bath.mean()) ** 2)
    r2_bath = float(1.0 - ss_res_bath / ss_tot_bath) if ss_tot_bath > 0 else float("nan")
    ss_res_hutch = np.sum((y_hutch - a_hutch * sqrt_t) ** 2)
    ss_tot_hutch = np.sum((y_hutch - y_hutch.mean()) ** 2)
    r2_hutch = float(1.0 - ss_res_hutch / ss_tot_hutch) if ss_tot_hutch > 0 else float("nan")

    summary = dict(
        n_traj=int(se.shape[0]),
        T_final=float(times[-1]),
        # ensemble-mean convergence: at t=T_final, how close are the means?
        mean_sigma_exact_final=float(mean_se[-1]),
        mean_sigma_bath_final=float(mean_sb[-1]),
        mean_sigma_hutch_final=float(mean_sh[-1]),
        mean_bias_bath_final=float(mean_dev_bath[-1]),
        mean_bias_hutch_final=float(mean_dev_hutch[-1]),
        # ensemble std of deviations at t=T_final
        std_bath_minus_exact_final=float(std_dev_bath[-1]),
        std_hutch_minus_exact_final=float(std_dev_hutch[-1]),
        # headline metric: ratio bath / hutch
        ratio_bath_over_hutch_std_final=ratio_final,
        # cross-correlation at t=T_final
        corr_bath_hutch_dev_final=float(corr_bh[-1]),
        # sqrt(t) scaling fits
        sqrt_t_coeff_bath=a_bath,
        sqrt_t_coeff_hutch=a_hutch,
        sqrt_t_r2_bath=r2_bath,
        sqrt_t_r2_hutch=r2_hutch,
    )

    diagnostics = dict(
        mean_se=mean_se,
        mean_sb=mean_sb,
        mean_sh=mean_sh,
        std_dev_bath=std_dev_bath,
        std_dev_hutch=std_dev_hutch,
        mean_dev_bath=mean_dev_bath,
        mean_dev_hutch=mean_dev_hutch,
        cov_bh=cov_bh,
        corr_bh=corr_bh,
    )
    return summary, diagnostics


# ---------- plotting --------------------------------------------------------

def make_figure_single(R, out_path):
    """Pedagogical single-trajectory figure.

    BUG-1 FIX: the sigma_bath label is now the direct heat-integral formula,
    NOT beta * Delta H_phys (which only holds for Hamiltonian-NH).
    """
    times = R["times"]
    se = R["sigma_exact"]
    sl = R["sigma_lyap"]
    sb = R["sigma_bath"]
    sh = R["sigma_hutch"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(times, se, color="black", lw=2.5, label=r"$\sigma_{\rm exact}=D\!\int\!\tanh\xi\,dt$")
    ax.plot(times, sl, color="crimson", lw=1.6, ls="--",
            label=r"$\sigma_{\rm lyap}=-\log|\det\Phi_t|$ (Benettin)")
    # BUG-1 FIX: correct label (direct heat integral for NH-tanh)
    ax.plot(times, sb, color="royalblue", lw=1.4, ls=":",
            label=r"$\sigma_{\rm bath}=\beta\!\int\!\tanh\xi\,|p|^2\,dt$")
    ax.plot(times, sh, color="forestgreen", lw=0.8, alpha=0.85,
            label=r"$\sigma_{\rm hutch}=-\!\int\!v^\top\!J\,v\,dt$ (FFJORD)")
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$\sigma(t)\;=\;\log\rho_t-\log\rho_0$")
    ax.set_title("Single trajectory (seed=42): four estimators of $\\sigma(t)$")
    ax.legend(loc="best", frameon=False, fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    eps = 1e-18
    ax.semilogy(times, np.abs(se - sl) + eps, color="crimson",
                label=r"$|\sigma_{\rm exact}-\sigma_{\rm lyap}|$ (pathwise)")
    ax.semilogy(times, np.abs(se - sb) + eps, color="royalblue",
                label=r"$|\sigma_{\rm exact}-\sigma_{\rm bath}|$ (stochastic)")
    ax.semilogy(times, np.abs(se - sh) + eps, color="forestgreen", alpha=0.85,
                label=r"$|\sigma_{\rm exact}-\sigma_{\rm hutch}|$ (stochastic)")
    ax.set_xlabel("time $t$")
    ax.set_ylabel("absolute deviation")
    ax.set_title("Deviation from $\\sigma_{\\rm exact}$")
    ax.legend(loc="best", frameon=False, fontsize=8)
    ax.grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_figure_ensemble(E, diag, summary, out_path):
    """Three-panel ensemble figure:

    (a) Mean +/- std band of each sigma(t).
    (b) Ensemble std of (sigma_X - sigma_exact) vs time, with sqrt(t) fits.
    (c) Ensemble correlation Corr(sigma_hutch - sigma_exact,
                                   sigma_bath  - sigma_exact).
    """
    times = E["times"]
    se = E["sigma_exact"]
    sb = E["sigma_bath"]
    sh = E["sigma_hutch"]

    mean_se = diag["mean_se"]
    mean_sb = diag["mean_sb"]
    mean_sh = diag["mean_sh"]
    std_se = se.std(axis=0, ddof=1)
    std_sb = sb.std(axis=0, ddof=1)
    std_sh = sh.std(axis=0, ddof=1)

    std_dev_bath = diag["std_dev_bath"]
    std_dev_hutch = diag["std_dev_hutch"]
    corr_bh = diag["corr_bh"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    # ---- Panel (a): mean +/- std band ----
    ax = axes[0]
    ax.plot(times, mean_se, color="black", lw=2.4, label=r"$\langle\sigma_{\rm exact}\rangle$")
    ax.fill_between(times, mean_se - std_se, mean_se + std_se,
                    color="black", alpha=0.12)
    ax.plot(times, mean_sb, color="royalblue", lw=1.4, ls="--",
            label=r"$\langle\sigma_{\rm bath}\rangle$")
    ax.fill_between(times, mean_sb - std_sb, mean_sb + std_sb,
                    color="royalblue", alpha=0.18)
    ax.plot(times, mean_sh, color="forestgreen", lw=1.4, ls=":",
            label=r"$\langle\sigma_{\rm hutch}\rangle$")
    ax.fill_between(times, mean_sh - std_sh, mean_sh + std_sh,
                    color="forestgreen", alpha=0.18)
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$\sigma(t)$")
    ax.set_title(f"(a) Ensemble mean $\\pm$ std   ($N={se.shape[0]}$ trajectories)")
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    # ---- Panel (b): ensemble std of deviations, with sqrt(t) fits ----
    ax = axes[1]
    ax.plot(times, std_dev_bath, color="royalblue", lw=1.8,
            label=r"$\mathrm{std}(\sigma_{\rm bath}-\sigma_{\rm exact})$")
    ax.plot(times, std_dev_hutch, color="forestgreen", lw=1.8,
            label=r"$\mathrm{std}(\sigma_{\rm hutch}-\sigma_{\rm exact})$")
    # sqrt(t) fits
    a_b = summary["sqrt_t_coeff_bath"]
    a_h = summary["sqrt_t_coeff_hutch"]
    ts_fit = np.linspace(0.0, times[-1], 400)
    ax.plot(ts_fit, a_b * np.sqrt(ts_fit), color="royalblue", lw=1.0, ls="--",
            label=fr"${a_b:.3f}\sqrt{{t}}$ (fit, $R^2$={summary['sqrt_t_r2_bath']:.3f})")
    ax.plot(ts_fit, a_h * np.sqrt(ts_fit), color="forestgreen", lw=1.0, ls="--",
            label=fr"${a_h:.3f}\sqrt{{t}}$ (fit, $R^2$={summary['sqrt_t_r2_hutch']:.3f})")
    ax.set_xlabel("time $t$")
    ax.set_ylabel("ensemble std of deviation")
    ratio = summary["ratio_bath_over_hutch_std_final"]
    ax.set_title(fr"(b) $\sqrt{{t}}$ variance growth   ratio $b/h$ @ $t={times[-1]:.0f}$: {ratio:.3f}")
    ax.legend(loc="best", frameon=False, fontsize=8)
    ax.grid(alpha=0.25)

    # ---- Panel (c): cross-correlation ----
    ax = axes[2]
    ax.plot(times, corr_bh, color="purple", lw=1.8)
    ax.axhline(0.0, color="black", lw=0.6, alpha=0.5)
    ax.axhline(1.0, color="black", lw=0.4, ls=":", alpha=0.5)
    ax.axhline(-1.0, color="black", lw=0.4, ls=":", alpha=0.5)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"Corr across ensemble")
    rho_final = summary["corr_bath_hutch_dev_final"]
    ax.set_title(fr"(c) $\mathrm{{Corr}}(\sigma_{{\rm hutch}}-\sigma_{{\rm exact}},\;\sigma_{{\rm bath}}-\sigma_{{\rm exact}})$   @ $t={times[-1]:.0f}$: {rho_final:.3f}")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------- main ------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(here, "results")
    fig_dir = os.path.join(here, "figures")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # ---- 1. single trajectory (seed=42), pedagogical figure ----
    print("[triple-identity-064] single-trajectory run (seed=42) with Benettin Lyapunov ...")
    R_single = run_single(seed=SEED, n_steps=N_STEPS, dt=DT, record_lyap=True)
    summary_single = summarize_single(R_single)
    print("  single-trajectory summary:")
    for k, v in summary_single.items():
        print(f"    {k}: {v}")

    np.savez(os.path.join(res_dir, "triple_identity.npz"), **R_single)
    with open(os.path.join(res_dir, "triple_identity.json"), "w") as fh:
        json.dump(summary_single, fh, indent=2)

    fig_path = os.path.join(fig_dir, "fig_triple_identity.png")
    make_figure_single(R_single, fig_path)
    print(f"  wrote {fig_path}")

    # ---- 2. ensemble run, ensemble figure ----
    print(f"[triple-identity-064] ensemble run: N={N_TRAJ} trajectories, seeds {SEED_BASE}..{SEED_BASE + N_TRAJ - 1} ...")
    E = run_ensemble(n_traj=N_TRAJ, seed_base=SEED_BASE, n_steps=N_STEPS, dt=DT)

    summary_ens, diag = summarize_ensemble(E)
    print("  ensemble summary:")
    for k, v in summary_ens.items():
        print(f"    {k}: {v}")

    np.savez(os.path.join(res_dir, "ensemble_sigmas.npz"),
             times=E["times"],
             sigma_exact=E["sigma_exact"],
             sigma_bath=E["sigma_bath"],
             sigma_hutch=E["sigma_hutch"],
             xi=E["xi"])
    with open(os.path.join(res_dir, "ensemble_summary.json"), "w") as fh:
        json.dump(summary_ens, fh, indent=2)

    fig_path2 = os.path.join(fig_dir, "fig_triple_identity_ensemble.png")
    make_figure_ensemble(E, diag, summary_ens, fig_path2)
    print(f"  wrote {fig_path2}")

    # ---- headline metric ----
    ratio = summary_ens["ratio_bath_over_hutch_std_final"]
    print()
    print(f"[triple-identity-064] HEADLINE METRIC")
    print(f"  std(sigma_bath - sigma_exact) / std(sigma_hutch - sigma_exact) @ t={T_FINAL}")
    print(f"  = {ratio:.6f}")
    if ratio < 1.0:
        print(f"  => bath noise SMALLER than Hutchinson noise by factor {1.0/ratio:.2f}x")
        print(f"     supports control-variate hypothesis (orbit 065 should proceed)")
    else:
        print(f"  => Hutchinson noise SMALLER than bath noise by factor {ratio:.2f}x")
        print(f"     rethink control-variate direction for orbit 065")


if __name__ == "__main__":
    main()
