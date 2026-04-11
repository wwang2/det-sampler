"""
triple-identity-064: hero figure for Paper 1.

Demonstrates that on a single Nose-Hoover (NH-tanh) trajectory in a 2D
double-well potential, four quantities — all derived completely independently
of each other — coincide on the same trajectory.

  1. sigma_exact   = +D * int tanh(xi) dt
                     ( = log rho_t - log rho_0 along the flow ).
                     Trivial calculus, computed by trapezoid rule on xi(t).
  2. sigma_lyap    = - sum_i log|R_ii_total|, the negative log of |det Phi_t|
                     of the tangent flow operator, accumulated by the Benettin
                     algorithm (RK4 tangent + periodic QR).
                     Pesin / Liouville: log|det Phi_t| = int tr(J(z(s))) ds
                     = -D int tanh(xi) ds; so -log|det Phi_t| = sigma_exact.
  3. sigma_bath    = beta * Q_bath = beta * int tanh(xi) |p|^2 ds.
                     Gallavotti / Evans-Searles: phase-space contraction
                     equals beta times the heat dumped to the bath. Equal to
                     sigma_exact only on AVERAGE (equipartition <|p|^2> = D kT);
                     it fluctuates around sigma_exact with amplitude tied to
                     instantaneous |p|^2 - D kT.
  4. sigma_hutch   = - int v^T J(z) v dt (Rademacher v resampled per step).
                     FFJORD-style stochastic trace estimator. Unbiased; equal
                     to sigma_exact in expectation, with O(sqrt(t)) random walk
                     fluctuations around it.

(1) and (2) are PATHWISE identical up to ODE truncation error (~ dt^4 RK4 +
QR roundoff). (3) and (4) are stochastic estimators of the same mean and only
agree on average. Together the four overlap visually — that overlap *is* the
hero figure.

Run: python experiment.py
Outputs:
  results/triple_identity.npz   --- raw arrays
  results/triple_identity.json  --- summary metrics
  figures/fig_triple_identity.png --- 2-panel hero figure
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
QR_EVERY = 5  # Benettin renormalization period (more frequent => better orthogonality)


def V(q):
    """Double-well potential V(q) = (q1^2 - 1)^2 + 0.5 * q2^2."""
    q1, q2 = q[0], q[1]
    return (q1 * q1 - 1.0) ** 2 + 0.5 * q2 * q2


def grad_V(q):
    q1, q2 = q[0], q[1]
    return np.array([4.0 * q1 * (q1 * q1 - 1.0), q2], dtype=np.float64)


def hessian_V(q):
    """Hessian of V (D x D)."""
    q1 = q[0]
    return np.array([[12.0 * q1 * q1 - 4.0, 0.0],
                     [0.0, 1.0]], dtype=np.float64)


# ---------- NH-tanh ODE on the extended phase space -------------------------

def f(z):
    """Vector field of the NH-tanh flow.

    z = (q[0..D-1], p[0..D-1], xi)  in R^{2D+1}
    """
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

    block structure (rows = derivatives of (dq, dp, dxi); cols = (q, p, xi)):
        d(dq)/dq = 0          d(dq)/dp = I_D       d(dq)/dxi = 0
        d(dp)/dq = -Hess V    d(dp)/dp = -tanh(xi) I_D   d(dp)/dxi = -sech^2(xi) p
        d(dxi)/dq = 0         d(dxi)/dp = (2/Q) p^T      d(dxi)/dxi = 0

    Trace = -D * tanh(xi).  (Confirms sigma_exact below.)
    """
    p = z[D:2 * D]
    xi = z[2 * D]
    g = np.tanh(xi)
    sech2 = 1.0 - g * g  # sech^2(xi)
    H = hessian_V(z[0:D])
    J = np.zeros((DIM, DIM), dtype=np.float64)
    # d(dq)/dp = I
    J[0:D, D:2 * D] = np.eye(D)
    # d(dp)/dq = -H
    J[D:2 * D, 0:D] = -H
    # d(dp)/dp = -g I
    J[D:2 * D, D:2 * D] = -g * np.eye(D)
    # d(dp)/dxi = -sech^2(xi) p
    J[D:2 * D, 2 * D] = -sech2 * p
    # d(dxi)/dp = (2/Q) p
    J[2 * D, D:2 * D] = (2.0 / Q_NH) * p
    # d(dxi)/dxi = 0; d(dxi)/dq = 0; d(dq)/dq = 0; d(dq)/dxi = 0
    return J


def rk4_step(z, dt):
    k1 = f(z)
    k2 = f(z + 0.5 * dt * k1)
    k3 = f(z + 0.5 * dt * k2)
    k4 = f(z + dt * k3)
    return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_step_with_tangent(z, M, dt):
    """RK4 step that also propagates a tangent matrix M (DIM x k).

    Tangent flow:  dM/dt = J(z(t)) * M
    Both states are advanced with the SAME RK4 stages so that the linearization
    is consistent with the actual trajectory points.
    """
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


# ---------- Nose extended Hamiltonian ---------------------------------------

def H_phys(z):
    """Physical Hamiltonian K + V (NOT conserved -- changes by bath heat)."""
    p = z[D:2 * D]
    return 0.5 * np.dot(p, p) + V(z[0:D])


def heat_rate(z):
    """Instantaneous heat dissipated INTO the bath:  q_dot = tanh(xi) * |p|^2.

    Derivation:  d/dt H_phys = p . (dp/dt) + grad V . (dq/dt)
                              = p . (-grad V - tanh(xi) p) + grad V . p
                              = -tanh(xi) |p|^2.
    The bath gains exactly what the system loses, so Q_dot_bath = +tanh(xi)|p|^2.
    """
    p = z[D:2 * D]
    xi = z[2 * D]
    return np.tanh(xi) * np.dot(p, p)


# ---------- main integration loop -------------------------------------------

def run():
    rng = np.random.default_rng(SEED)
    # initial condition: left well, p ~ N(0,I), xi=0
    q0 = np.array([-1.0, 0.0], dtype=np.float64)
    p0 = rng.standard_normal(D)
    xi0 = 0.0
    z = np.empty(DIM, dtype=np.float64)
    z[0:D] = q0
    z[D:2 * D] = p0
    z[2 * D] = xi0

    # Benettin tangent basis: 5 orthonormal vectors
    M = np.eye(DIM, dtype=np.float64)
    log_R_diag_acc = np.zeros(DIM, dtype=np.float64)

    # Storage
    times = np.zeros(N_STEPS + 1)
    sigma_exact = np.zeros(N_STEPS + 1)
    sigma_lyap = np.zeros(N_STEPS + 1)
    sigma_bath = np.zeros(N_STEPS + 1)
    sigma_hutch = np.zeros(N_STEPS + 1)
    H_phys_t = np.zeros(N_STEPS + 1)
    xi_t = np.zeros(N_STEPS + 1)
    q_t = np.zeros((N_STEPS + 1, D))

    H_phys_t[0] = H_phys(z)
    xi_t[0] = z[2 * D]
    q_t[0] = z[0:D]

    # bath entropy production at t=0 is 0
    sigma_bath[0] = 0.0
    sigma_exact[0] = 0.0
    sigma_lyap[0] = 0.0
    sigma_hutch[0] = 0.0

    # Hutchinson v: resampled at every step. Use Rademacher (+/- 1).
    rng_hutch = np.random.default_rng(SEED + 1)

    for step in range(N_STEPS):
        # ----- Hutchinson estimate at CURRENT state ---------------------
        v = rng_hutch.choice([-1.0, 1.0], size=DIM).astype(np.float64)
        Jz = jacobian(z)
        # Hutchinson trace estimate: v^T J v.
        # sign: log rho_t - log rho_0 = -int tr(J) dt, so accumulate -v^T J v dt.
        trace_est = float(v @ (Jz @ v))
        sigma_hutch[step + 1] = sigma_hutch[step] - trace_est * DT

        # ----- bath heat at CURRENT state (left endpoint of step) -------
        q_dot_old = heat_rate(z)
        xi_old = z[2 * D]

        # ----- propagate state + tangent --------------------------------
        z_new, M_new = rk4_step_with_tangent(z, M, DT)

        # ----- exact divergence (trapezoid on xi) -----------------------
        xi_new = z_new[2 * D]
        # log rho_t - log rho_0 = -int tr(J) dt = -int(-D tanh xi) dt = +D int tanh xi dt
        d_sigma_exact = D * 0.5 * (np.tanh(xi_old) + np.tanh(xi_new)) * DT
        sigma_exact[step + 1] = sigma_exact[step] + d_sigma_exact

        # ----- bath entropy production (Gallavotti) ---------------------
        # sigma_bath = beta * int Q_dot_bath dt, with Q_dot_bath = tanh(xi)|p|^2.
        # By equipartition <|p|^2>_eq = D kT, so the integrand averages to
        # D kT tanh(xi), matching sigma_exact's integrand after multiplying by beta=1/kT.
        # Use trapezoid rule for second-order accuracy.
        q_dot_new = heat_rate(z_new)
        sigma_bath[step + 1] = sigma_bath[step] + BETA * 0.5 * (q_dot_old + q_dot_new) * DT

        # ----- Benettin: periodic QR ------------------------------------
        if (step + 1) % QR_EVERY == 0:
            Qm, Rm = np.linalg.qr(M_new)
            # accumulate log of diagonal magnitudes (sign-aware)
            diag = np.diag(Rm)
            log_R_diag_acc += np.log(np.abs(diag))
            # absorb signs back into Q to keep an honest basis
            signs = np.sign(diag)
            signs[signs == 0] = 1.0
            Qm = Qm * signs[np.newaxis, :]
            M_new = Qm

        # Cumulative log|det Phi_t|. Between QR resets, M_new is no longer
        # orthogonal, so the live tangent volume is captured by log|det M_new|;
        # add it to the history accumulated at prior QR resets to get a smooth
        # (non-sawtooth) estimate of log|det Phi_t|.
        sign_det, logdet_M = np.linalg.slogdet(M_new)
        live_logdet = float(np.sum(log_R_diag_acc) + logdet_M)
        # Pesin / Liouville: log|det Phi_t| = int tr J ds = -D int tanh xi ds.
        # log rho_t - log rho_0 = +D int tanh xi ds = -log|det Phi_t|.
        sigma_lyap[step + 1] = -live_logdet

        # commit
        z = z_new
        M = M_new
        H_phys_t[step + 1] = H_phys(z)
        xi_t[step + 1] = z[2 * D]
        q_t[step + 1] = z[0:D]
        times[step + 1] = (step + 1) * DT

    return dict(
        times=times,
        sigma_exact=sigma_exact,
        sigma_lyap=sigma_lyap,
        sigma_bath=sigma_bath,
        sigma_hutch=sigma_hutch,
        H_phys=H_phys_t,
        xi=xi_t,
        q=q_t,
    )


# ---------- analysis & plotting --------------------------------------------

def summarize(R):
    se = R["sigma_exact"]
    sl = R["sigma_lyap"]
    sb = R["sigma_bath"]
    sh = R["sigma_hutch"]

    summary = dict(
        T_final=float(R["times"][-1]),
        n_steps=int(len(R["times"]) - 1),
        dt=DT,
        # ----- pathwise identities (should be tight) -----
        max_abs_exact_minus_lyap=float(np.max(np.abs(se - sl))),
        # ----- on-average (stochastic) identities -----
        max_abs_exact_minus_bath=float(np.max(np.abs(se - sb))),
        max_abs_exact_minus_hutch=float(np.max(np.abs(se - sh))),
        std_exact_minus_bath=float(np.std(se - sb)),
        std_exact_minus_hutch=float(np.std(se - sh)),
        # final-time relative agreement
        rel_err_exact_lyap_final=float(abs(se[-1] - sl[-1]) / max(abs(se[-1]), 1e-12)),
        rel_err_exact_bath_final=float(abs(se[-1] - sb[-1]) / max(abs(se[-1]), 1e-12)),
        rel_err_exact_hutch_final=float(abs(se[-1] - sh[-1]) / max(abs(se[-1]), 1e-12)),
        sigma_exact_final=float(se[-1]),
        sigma_lyap_final=float(sl[-1]),
        sigma_bath_final=float(sb[-1]),
        sigma_hutch_final=float(sh[-1]),
    )
    return summary


def make_figure(R, out_path):
    times = R["times"]
    se = R["sigma_exact"]
    sl = R["sigma_lyap"]
    sb = R["sigma_bath"]
    sh = R["sigma_hutch"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel (a) — overlay
    ax = axes[0]
    ax.plot(times, se, color="black", lw=2.5, label=r"$\sigma_{\rm exact}$")
    ax.plot(times, sl, color="crimson", lw=1.6, ls="--", label=r"$\sigma_{\rm lyap}$ (Benettin)")
    ax.plot(times, sb, color="royalblue", lw=1.4, ls=":", label=r"$\sigma_{\rm bath}=\beta\,\Delta H_{\rm phys}$")
    ax.plot(times, sh, color="forestgreen", lw=0.8, alpha=0.85, label=r"$\sigma_{\rm hutch}$ (FFJORD)")
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$\sigma(t)$  =  $\log\rho_t - \log\rho_0$")
    ax.set_title("Four equivalent interpretations of NH phase-space contraction")
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    # Panel (b) — deviations from sigma_exact (log scale)
    ax = axes[1]
    eps = 1e-18
    ax.semilogy(times, np.abs(se - sl) + eps, color="crimson",
                label=r"$|\sigma_{\rm exact}-\sigma_{\rm lyap}|$")
    ax.semilogy(times, np.abs(se - sb) + eps, color="royalblue",
                label=r"$|\sigma_{\rm exact}-\sigma_{\rm bath}|$")
    ax.semilogy(times, np.abs(se - sh) + eps, color="forestgreen", alpha=0.85,
                label=r"$|\sigma_{\rm exact}-\sigma_{\rm hutch}|$")
    ax.set_xlabel("time $t$")
    ax.set_ylabel("absolute deviation")
    ax.set_title("Pointwise deviation from $\\sigma_{\\rm exact}$")
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(here, "results")
    fig_dir = os.path.join(here, "figures")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("[triple-identity-064] integrating NH-tanh flow ...")
    R = run()

    print("[triple-identity-064] summarizing ...")
    summary = summarize(R)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    np.savez(os.path.join(res_dir, "triple_identity.npz"), **R)
    with open(os.path.join(res_dir, "triple_identity.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    fig_path = os.path.join(fig_dir, "fig_triple_identity.png")
    make_figure(R, fig_path)
    print(f"[triple-identity-064] wrote {fig_path}")


if __name__ == "__main__":
    main()
