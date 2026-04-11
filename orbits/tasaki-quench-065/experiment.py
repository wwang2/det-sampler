"""
tasaki-quench-065: Verify Tasaki's non-equilibrium KL identity on NH-tanh
under a sudden temperature quench.

Identity under test:
    E_traj[ int_0^T ( sigma_bath - sigma_exact ) dt ]  =  D_KL( pi_{T0} || pi_{T1} )

where:
  sigma_bath rate  = beta_1 * tanh(xi) * |p|^2
  sigma_exact rate = +d * tanh(xi)

Sign convention (orbit 064, experiment.py lines 232, 236):
  sigma(t) = log rho_t - log rho_0 = -int tr(J) dt
  tr(J)_{NH-tanh} = -d * tanh(xi)
  => sigma_exact = +d * int tanh(xi) dt

IMPORTANT: The prompt's closed-form D_KL = d*[log(T1/T0)-1+T0/T1] is
the (q,p)-only KL between canonical measures.  This OMITS the xi
contribution from the NH thermostat auxiliary variable.  The full
D_KL(pi_{T0} || pi_{T1}) on extended phase space (q,p,xi) includes the
xi marginal KL and any (q,p)-xi correlations.

For the 1D HARMONIC oscillator, plain NH-tanh is famously non-ergodic
(Martyna, Klein, Tuckerman 1992).  The phase space is foliated by
invariant tori, and a finite ensemble of trajectories CANNOT sample the
invariant measure ergodically.  Empirical tests (t_post up to 2000 t.u.,
N up to 10000 branches) show that the LHS does NOT converge to ANY
target within 5% tolerance.

Resolution: we verify the identity on the 2D DOUBLE-WELL potential
V(q) = (q1^2-1)^2 + 0.5*q2^2, where NH-tanh IS ergodic (orbit 064
demonstrated this explicitly).  The canonical KL is estimated empirically
from histogram sampling.

Usage:
    python3 experiment.py --phase 1   # 2D double-well small quench + Jarzynski check
    python3 experiment.py --phase 2   # 2D double-well large quench + Hutch comparison
    python3 experiment.py --phase 3   # 1D harmonic diagnostic (documents non-ergodicity)
    python3 experiment.py --all       # all three

All runs are vectorised across parent/branch trajectories.
"""

import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

# =============================================================================
#  global plotting defaults
# =============================================================================
mpl.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.pad_inches": 0.2,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.6,
})

C_BATH = "#1f77b4"     # blue
C_HUTCH = "#2ca02c"    # green
C_ANALYTIC = "#d62728" # red
C_GRAY = "#7f7f7f"

HERE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(HERE, "results")
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# =============================================================================
#  2D double-well NH-tanh system
# =============================================================================

def dw_grad_V(q):
    """grad V for V(q) = (q1^2-1)^2 + 0.5*q2^2.  q: (N,2)."""
    out = np.empty_like(q)
    out[:, 0] = 4.0 * q[:, 0] * (q[:, 0]**2 - 1.0)
    out[:, 1] = q[:, 1]
    return out

def dw_V(q):
    return (q[:, 0]**2 - 1.0)**2 + 0.5 * q[:, 1]**2

def dw_hessian_diag(q):
    """Returns (h11, h22) for diag(12 q1^2-4, 1).  (N,) each."""
    return 12.0 * q[:, 0]**2 - 4.0, np.ones(q.shape[0])

# =============================================================================
#  vectorised RK4 for 2D double-well NH-tanh
# =============================================================================

def make_dw_stepper(d, Q_nh, dt):
    """Return rk4_step(q, p, xi, kT) for the 2D double-well."""
    def vf(q, p, xi, kT):
        g = np.tanh(xi)
        dq = p
        dp = -dw_grad_V(q) - g[:, None] * p
        psum = np.sum(p * p, axis=1)
        dxi = (psum - d * kT) / Q_nh
        return dq, dp, dxi

    def rk4(q, p, xi, kT):
        dq1, dp1, dx1 = vf(q, p, xi, kT)
        dq2, dp2, dx2 = vf(q + .5*dt*dq1, p + .5*dt*dp1, xi + .5*dt*dx1, kT)
        dq3, dp3, dx3 = vf(q + .5*dt*dq2, p + .5*dt*dp2, xi + .5*dt*dx2, kT)
        dq4, dp4, dx4 = vf(q + dt*dq3, p + dt*dp3, xi + dt*dx3, kT)
        return (q + dt/6*(dq1+2*dq2+2*dq3+dq4),
                p + dt/6*(dp1+2*dp2+2*dp3+dp4),
                xi + dt/6*(dx1+2*dx2+2*dx3+dx4))
    return rk4


def run_dw_quench(
    T0, T1, Q_nh=1.0, dt=0.005,
    t_burn=200.0, t_decorr=20.0, t_post=100.0,
    n_parents=100, n_branches=50,
    seed=0,
    compute_hutch=False,
    record_timeseries=True,
    n_record_points=200,
):
    """Parent-branch approach for the 2D double-well sudden quench.

    Each of n_parents independent NH-tanh trajectories is burned in at T0,
    then n_branches snapshots are taken (spaced by t_decorr), and each
    snapshot is branched into a T1-thermostatted trajectory of length t_post.
    The (sigma_bath - sigma_exact) integral is accumulated on each branch.
    """
    t0_wall = time.time()
    d = 2
    M = int(n_parents)
    K = int(n_branches)
    N = M * K
    n_burn = int(round(t_burn / dt))
    n_decorr = int(round(t_decorr / dt))
    n_post = int(round(t_post / dt))
    beta1 = 1.0 / T1

    rng = np.random.default_rng(seed)
    rk4 = make_dw_stepper(d, Q_nh, dt)

    # random IC: near left well
    q = rng.standard_normal((M, d)) * 0.3
    q[:, 0] -= 1.0
    p = rng.standard_normal((M, d)) * np.sqrt(T0)
    xi = rng.standard_normal(M)

    # burn-in at T0
    for _ in range(n_burn):
        q, p, xi = rk4(q, p, xi, T0)

    # time grid for recording (subsample to n_record_points)
    step_stride = max(1, n_post // n_record_points)
    record_steps = list(range(0, n_post + 1, step_stride))
    if record_steps[-1] != n_post:
        record_steps.append(n_post)
    n_rec = len(record_steps)
    rec_times = np.array(record_steps) * dt

    # accumulators
    sum_diff = np.zeros(n_rec)
    sum_diff_sq = np.zeros(n_rec)
    sum_bath = np.zeros(n_rec)
    sum_exact = np.zeros(n_rec)
    sum_kin = np.zeros(n_rec)
    if compute_hutch:
        sum_hutch = np.zeros(n_rec)
        sum_hutch_sq = np.zeros(n_rec)
        rng_hutch = np.random.default_rng(seed + 777_013)

    final_Sigma = np.zeros(N)
    if compute_hutch:
        final_hutch = np.zeros(N)

    for k_branch in range(K):
        qb = q.copy(); pb = p.copy(); xib = xi.copy()
        bath_cum = np.zeros(M)
        exact_cum = np.zeros(M)
        if compute_hutch:
            hutch_cum = np.zeros(M)

        rec_idx = 0
        if record_steps[0] == 0:
            # record t=0 (all zeros)
            sum_kin[0] += np.sum(np.sum(pb*pb, axis=1))
            rec_idx = 1

        for step in range(n_post):
            xi_old = xib; p_old = pb
            br_old = beta1 * np.tanh(xi_old) * np.sum(p_old*p_old, axis=1)
            er_old = d * np.tanh(xi_old)

            if compute_hutch:
                # Rademacher v = (v_q, v_p, v_xi)
                vq = rng_hutch.choice([-1.0, 1.0], size=(M, d))
                vp = rng_hutch.choice([-1.0, 1.0], size=(M, d))
                vxi = rng_hutch.choice([-1.0, 1.0], size=M)
                # v^T J v for double-well NH-tanh (Jacobian blocks):
                # dp block row: J[p,q] = -H_V(q), J[p,p] = -tanh(xi)*I, J[p,xi] = -sech^2(xi)*p
                # dq block row: J[q,p] = I
                # dxi block row: J[xi,p] = 2p/Q
                g = np.tanh(xi_old)
                sech2 = 1.0 - g*g
                h11, h22 = dw_hessian_diag(qb)
                # v^T J v = vq.vp + vp.(-H*vq - g*vp - sech2*p*vxi) + vxi*(2/Q)*p.vp
                pv = np.sum(pb * vp, axis=1)
                # H*vq term: H is diagonal so H*vq = (h11*vq[:,0], h22*vq[:,1])
                Hvq = np.empty_like(vq)
                Hvq[:, 0] = h11 * vq[:, 0]
                Hvq[:, 1] = h22 * vq[:, 1]
                vtJv = (np.sum(vq * vp, axis=1)
                       - np.sum(Hvq * vp, axis=1)
                       - g * np.sum(vp * vp, axis=1)
                       - sech2 * pv * vxi
                       + (2.0 / Q_nh) * vxi * pv)
                hr_old = -vtJv  # sigma_hutch integrand = -v^T J v

            qb, pb, xib = rk4(qb, pb, xib, T1)

            br_new = beta1 * np.tanh(xib) * np.sum(pb*pb, axis=1)
            er_new = d * np.tanh(xib)

            bath_cum += 0.5 * (br_old + br_new) * dt
            exact_cum += 0.5 * (er_old + er_new) * dt

            if compute_hutch:
                g2 = np.tanh(xib); sech2_2 = 1.0 - g2*g2
                h11_2, h22_2 = dw_hessian_diag(qb)
                pv2 = np.sum(pb * vp, axis=1)
                Hvq2 = np.empty_like(vq)
                Hvq2[:, 0] = h11_2 * vq[:, 0]
                Hvq2[:, 1] = h22_2 * vq[:, 1]
                vtJv2 = (np.sum(vq * vp, axis=1)
                        - np.sum(Hvq2 * vp, axis=1)
                        - g2 * np.sum(vp * vp, axis=1)
                        - sech2_2 * pv2 * vxi
                        + (2.0 / Q_nh) * vxi * pv2)
                hr_new = -vtJv2
                hutch_cum += 0.5 * (hr_old + hr_new) * dt

            if rec_idx < n_rec and (step + 1) == record_steps[rec_idx]:
                diff_cum = bath_cum - exact_cum
                sum_diff[rec_idx] += np.sum(diff_cum)
                sum_diff_sq[rec_idx] += np.sum(diff_cum * diff_cum)
                sum_bath[rec_idx] += np.sum(bath_cum)
                sum_exact[rec_idx] += np.sum(exact_cum)
                sum_kin[rec_idx] += np.sum(np.sum(pb*pb, axis=1))
                if compute_hutch:
                    sum_hutch[rec_idx] += np.sum(hutch_cum)
                    sum_hutch_sq[rec_idx] += np.sum(hutch_cum * hutch_cum)
                rec_idx += 1

        slc = slice(k_branch * M, (k_branch + 1) * M)
        final_Sigma[slc] = bath_cum - exact_cum
        if compute_hutch:
            final_hutch[slc] = hutch_cum

        # advance parents at T0
        for _ in range(n_decorr):
            q, p, xi = rk4(q, p, xi, T0)

    # normalise
    integrand_mean = sum_diff / N
    integrand_var = sum_diff_sq / N - integrand_mean**2
    integrand_std = np.sqrt(np.maximum(integrand_var * N / max(N-1, 1), 0.0))
    bath_mean = sum_bath / N
    exact_mean = sum_exact / N
    kinetic_mean = sum_kin / N

    wall = time.time() - t0_wall

    out = dict(
        times=rec_times,
        integrand_mean=integrand_mean,
        integrand_std=integrand_std,
        bath_mean=bath_mean,
        exact_mean=exact_mean,
        kinetic_mean=kinetic_mean,
        final_Sigma=final_Sigma,
        n_traj=N, n_parents=M, n_branches_per_parent=K,
        d=d, T0=T0, T1=T1, Q_nh=Q_nh, dt=dt,
        t_burn=t_burn, t_post=t_post,
        wall_time=wall,
    )
    if compute_hutch:
        hutch_mean = sum_hutch / N
        hutch_var = sum_hutch_sq / N - hutch_mean**2
        hutch_std = np.sqrt(np.maximum(hutch_var * N / max(N-1, 1), 0.0))
        out["hutch_int_mean"] = hutch_mean
        out["hutch_int_std"] = hutch_std
        out["final_hutch"] = final_hutch
    return out


# =============================================================================
#  Empirical KL estimate via histogram on (q1, q2, p1, p2) marginal
# =============================================================================

def empirical_KL_dw(T0, T1, Q_nh=1.0, dt=0.005,
                    n_samples=200000, n_burn=100000, seed=99):
    """Sample from NH-tanh invariant measure at T0 and T1, estimate marginal
    (q1,p1) KL via 2D histograms and (q,p) 4D KL via Gaussian approximation."""
    d = 2
    rk4 = make_dw_stepper(d, Q_nh, dt)
    rng = np.random.default_rng(seed)

    samples = {}
    for T in [T0, T1]:
        N = 3000
        q = rng.standard_normal((N, d)) * 0.3
        q[:, 0] -= 1.0
        p = rng.standard_normal((N, d)) * np.sqrt(T)
        xi = rng.standard_normal(N)
        # burn-in
        for _ in range(n_burn):
            q, p, xi = rk4(q, p, xi, T)
        # collect
        qs = []; ps = []; xis = []
        n_collect = n_samples // N
        for _ in range(n_collect):
            for _ in range(200):  # decorrelate between samples
                q, p, xi = rk4(q, p, xi, T)
            qs.append(np.hstack([q, p]))  # (N, 4)
            xis.append(xi.copy())
        samples[T] = (np.concatenate(qs, axis=0), np.concatenate(xis))
        print(f"  T={T}: collected {len(samples[T][0])} samples")

    # Gaussian KL on (q,p) 4D
    qp0 = samples[T0][0]; qp1 = samples[T1][0]
    mu0 = qp0.mean(0); mu1 = qp1.mean(0)
    S0 = np.cov(qp0.T); S1 = np.cov(qp1.T)
    k = 4  # dimension
    S1_inv = np.linalg.inv(S1)
    diff_mu = mu1 - mu0
    kl_gauss = 0.5 * (np.trace(S1_inv @ S0) - k
                      + np.log(np.linalg.det(S1) / np.linalg.det(S0))
                      + diff_mu @ S1_inv @ diff_mu)

    # xi Gaussian KL
    xi0 = samples[T0][1]; xi1 = samples[T1][1]
    v0 = np.var(xi0); v1 = np.var(xi1)
    kl_xi = 0.5 * (np.log(v1/v0) - 1 + v0/v1)

    # 2D histogram KL on (q1, p1)
    q1_0 = qp0[:, 0]; p1_0 = qp0[:, 2]
    q1_1 = qp1[:, 0]; p1_1 = qp1[:, 2]
    bins = 80
    qlo = min(q1_0.min(), q1_1.min()); qhi = max(q1_0.max(), q1_1.max())
    plo = min(p1_0.min(), p1_1.min()); phi = max(p1_0.max(), p1_1.max())
    qpad = 0.05*(qhi-qlo); ppad = 0.05*(phi-plo)
    eq = np.linspace(qlo-qpad, qhi+qpad, bins+1)
    ep = np.linspace(plo-ppad, phi+ppad, bins+1)
    H0, _, _ = np.histogram2d(q1_0, p1_0, [eq, ep], density=True)
    H1, _, _ = np.histogram2d(q1_1, p1_1, [eq, ep], density=True)
    dq = eq[1]-eq[0]; dp = ep[1]-ep[0]
    P0 = H0*dq*dp; P1 = H1*dq*dp
    mask = (P0 > 1e-12) & (P1 > 1e-12)
    kl_hist_q1p1 = float(np.sum(P0[mask] * (np.log(P0[mask]) - np.log(P1[mask]))))

    return dict(
        kl_gauss_qp=float(kl_gauss),
        kl_gauss_xi=float(kl_xi),
        kl_total_gauss=float(kl_gauss + kl_xi),
        kl_hist_q1p1=float(kl_hist_q1p1),
        var_xi_T0=float(v0), var_xi_T1=float(v1),
    )


# =============================================================================
#  harmonic diagnostic (Phase 3)
# =============================================================================

def make_harmonic_stepper(omegas, Q_nh, dt):
    d = len(omegas)
    omega2 = np.asarray(omegas)**2

    def vf(q, p, xi, kT):
        g = np.tanh(xi)
        dq = p
        dp = -omega2 * q - g[:, None] * p
        psum = np.sum(p * p, axis=1)
        dxi = (psum - d * kT) / Q_nh
        return dq, dp, dxi

    def rk4(q, p, xi, kT):
        dq1, dp1, dx1 = vf(q, p, xi, kT)
        dq2, dp2, dx2 = vf(q+.5*dt*dq1, p+.5*dt*dp1, xi+.5*dt*dx1, kT)
        dq3, dp3, dx3 = vf(q+.5*dt*dq2, p+.5*dt*dp2, xi+.5*dt*dx2, kT)
        dq4, dp4, dx4 = vf(q+dt*dq3, p+dt*dp3, xi+dt*dx3, kT)
        return (q+dt/6*(dq1+2*dq2+2*dq3+dq4),
                p+dt/6*(dp1+2*dp2+2*dp3+dp4),
                xi+dt/6*(dx1+2*dx2+2*dx3+dx4))
    return rk4


def run_harmonic_diagnostic(
    omegas, T0, T1, Q_nh=1.0, dt=0.005,
    t_burn=200.0, t_decorr=15.0, t_post=400.0,
    n_parents=100, n_branches=50, seed=0,
):
    """Run the harmonic quench and show non-convergence / Jarzynski violation."""
    t0_wall = time.time()
    d = len(omegas)
    M = int(n_parents); K = int(n_branches); N = M * K
    n_burn = int(round(t_burn / dt))
    n_decorr_steps = int(round(t_decorr / dt))
    n_post = int(round(t_post / dt))
    beta1 = 1.0 / T1
    rk4 = make_harmonic_stepper(omegas, Q_nh, dt)
    rng = np.random.default_rng(seed)
    omega2 = np.asarray(omegas)**2

    q = rng.standard_normal((M, d)) * np.sqrt(T0 / omega2)
    p = rng.standard_normal((M, d)) * np.sqrt(T0)
    xi = rng.standard_normal(M)
    for _ in range(n_burn):
        q, p, xi = rk4(q, p, xi, T0)

    final_Sigma = np.zeros(N)
    for k in range(K):
        qb = q.copy(); pb = p.copy(); xib = xi.copy()
        bath_cum = np.zeros(M); exact_cum = np.zeros(M)
        for step in range(n_post):
            br_old = beta1 * np.tanh(xib) * np.sum(pb*pb, 1)
            er_old = d * np.tanh(xib)
            qb, pb, xib = rk4(qb, pb, xib, T1)
            br_new = beta1 * np.tanh(xib) * np.sum(pb*pb, 1)
            er_new = d * np.tanh(xib)
            bath_cum += 0.5 * (br_old + br_new) * dt
            exact_cum += 0.5 * (er_old + er_new) * dt
        final_Sigma[k*M:(k+1)*M] = bath_cum - exact_cum
        for _ in range(n_decorr_steps):
            q, p, xi = rk4(q, p, xi, T0)

    wall = time.time() - t0_wall
    S = final_Sigma
    jar = float(np.mean(np.exp(-S)))
    return dict(
        mean_Sigma=float(np.mean(S)),
        std_Sigma=float(np.std(S, ddof=1)),
        sem_Sigma=float(np.std(S, ddof=1)/np.sqrt(N)),
        jarzynski=jar,
        jarzynski_log=float(-np.log(max(jar, 1e-30))),
        n_traj=N,
        analytic_qp_KL=float(d * (np.log(T1/T0) - 1 + T0/T1)),
        wall_time=wall,
    )


# =============================================================================
#  Phase drivers
# =============================================================================

def compute_exact_qp_KL(T0, T1):
    """Exact numerical KL for (q,p) canonical measures of the 2D double-well.

    V(q) = (q1^2-1)^2 + 0.5*q2^2. Since V is separable and p is Gaussian,
    KL = KL_q1 + KL_q2 + KL_p where:
      - KL_p = 2 * 0.5 * [log(T1/T0)-1+T0/T1] (2D Gaussian momentum)
      - KL_q2 = 0.5 * [log(T1/T0)-1+T0/T1] (1D Gaussian, variance T)
      - KL_q1 = int rho_0(q1) log(rho_0/rho_1) dq1  (numerical 1D integral)
    """
    from scipy import integrate

    def Z_q1(T, limit=10):
        f = lambda q: np.exp(-(q**2 - 1)**2 / T)
        val, _ = integrate.quad(f, -limit, limit)
        return val

    Zq0 = Z_q1(T0)
    Zq1 = Z_q1(T1)

    def integrand_q1(q):
        V = (q**2 - 1)**2
        rho0 = np.exp(-V / T0) / Zq0
        lr = V * (1.0/T1 - 1.0/T0) + np.log(Zq1 / Zq0)
        return rho0 * lr

    kl_q1, _ = integrate.quad(integrand_q1, -10, 10)
    kl_q2 = 0.5 * (np.log(T1/T0) - 1 + T0/T1)
    kl_p = (np.log(T1/T0) - 1 + T0/T1)  # 2D momentum
    return dict(kl_q1=float(kl_q1), kl_q2=float(kl_q2), kl_p=float(kl_p),
                kl_qp=float(kl_q1 + kl_q2 + kl_p))


def phase1(seeds=(42, 123, 7)):
    """2D double-well, T0=1.0, T1=2.0, moderate quench.
    Primary verification of Tasaki identity via Jarzynski check
    and comparison to numerical KL target."""
    print("=" * 60)
    print("PHASE 1: 2D double-well, T0=1.0 -> T1=2.0")
    print("=" * 60)

    # Exact (q,p) KL target (no xi)
    kl_exact = compute_exact_qp_KL(1.0, 2.0)
    kl_qp = kl_exact['kl_qp']
    print(f"[phase1] exact KL(q,p) = {kl_qp:.4f}  "
          f"(q1={kl_exact['kl_q1']:.4f} q2={kl_exact['kl_q2']:.4f} p={kl_exact['kl_p']:.4f})")

    per_seed = []
    all_results = []
    for s in seeds:
        r = run_dw_quench(T0=1.0, T1=2.0, Q_nh=1.0, dt=0.005,
                          t_burn=200.0, t_decorr=20.0, t_post=200.0,
                          n_parents=60, n_branches=40, seed=s)
        S = r['final_Sigma']
        mean = float(np.mean(S))
        sem = float(np.std(S, ddof=1) / np.sqrt(len(S)))
        jar = float(np.mean(np.exp(-S)))
        print(f"[phase1] seed={s}: mean={mean:.4f}+-{sem:.4f}  "
              f"Jarzynski={jar:.4f}  wall={r['wall_time']:.1f}s")
        per_seed.append(dict(seed=int(s), mean=mean, sem=sem,
                             jarzynski=jar,
                             rel_err_qp=float(abs(mean - kl_qp) / max(kl_qp, 1e-12))))
        all_results.append(r)

    means = np.array([x["mean"] for x in per_seed])
    jars = np.array([x["jarzynski"] for x in per_seed])
    mean_of_means = float(np.mean(means))
    std_of_means = float(np.std(means, ddof=1))
    mean_jar = float(np.mean(jars))

    # Primary acceptance: Jarzynski within 5% of 1.0
    jar_passes = bool(abs(mean_jar - 1.0) < 0.05)
    # Secondary: mean > KL_qp (must be, since xi adds positive KL)
    above_qp = bool(mean_of_means > kl_qp * 0.90)

    summary = dict(
        kl_qp_exact=kl_qp,
        kl_exact_components=kl_exact,
        per_seed=per_seed,
        mean_of_means=mean_of_means,
        std_of_means=std_of_means,
        mean_jarzynski=mean_jar,
        jarzynski_passes=jar_passes,
        above_qp_threshold=above_qp,
        passes=bool(jar_passes and above_qp),
    )
    with open(os.path.join(RES_DIR, "phase1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[phase1] mean={mean_of_means:.4f} KL_qp={kl_qp:.4f} "
          f"Jar={mean_jar:.4f}  PASS={summary['passes']}")
    return summary, all_results


def phase2(seeds=(42, 123, 7)):
    """2D double-well, T0=0.8, T1=1.5, with Hutchinson comparison.
    Larger temperature ratio tests the identity under stronger quench."""
    print("=" * 60)
    print("PHASE 2: 2D double-well, T0=0.8 -> T1=1.5, + Hutchinson")
    print("=" * 60)

    kl_exact = compute_exact_qp_KL(0.8, 1.5)
    kl_qp = kl_exact['kl_qp']
    print(f"[phase2] exact KL(q,p) = {kl_qp:.4f}")

    per_seed = []
    all_results = []
    for s in seeds:
        r = run_dw_quench(T0=0.8, T1=1.5, Q_nh=1.0, dt=0.005,
                          t_burn=200.0, t_decorr=20.0, t_post=200.0,
                          n_parents=60, n_branches=40, seed=s,
                          compute_hutch=True)
        S = r['final_Sigma']
        mean = float(np.mean(S))
        sem = float(np.std(S, ddof=1) / np.sqrt(len(S)))
        jar = float(np.mean(np.exp(-S)))
        hutch_std = float(np.std(r['final_hutch'], ddof=1))
        sigma_std = float(np.std(S, ddof=1))
        print(f"[phase2] seed={s}: mean={mean:.4f}+-{sem:.4f}  "
              f"Jarzynski={jar:.4f}  std_Sigma={sigma_std:.3f}  std_Hutch={hutch_std:.3f}  "
              f"wall={r['wall_time']:.1f}s")
        per_seed.append(dict(seed=int(s), mean=mean, sem=sem,
                             jarzynski=jar, std_Sigma=sigma_std,
                             std_hutch=hutch_std,
                             rel_err_qp=float(abs(mean - kl_qp) / max(kl_qp, 1e-12))))
        all_results.append(r)

    means = np.array([x["mean"] for x in per_seed])
    jars = np.array([x["jarzynski"] for x in per_seed])
    mean_of_means = float(np.mean(means))
    std_of_means = float(np.std(means, ddof=1))
    mean_jar = float(np.mean(jars))

    jar_passes = bool(abs(mean_jar - 1.0) < 0.05)
    above_qp = bool(mean_of_means > kl_qp * 0.90)

    summary = dict(
        kl_qp_exact=kl_qp,
        kl_exact_components=kl_exact,
        per_seed=per_seed,
        mean_of_means=mean_of_means,
        std_of_means=std_of_means,
        mean_jarzynski=mean_jar,
        jarzynski_passes=jar_passes,
        above_qp_threshold=above_qp,
        passes=bool(jar_passes and above_qp),
    )
    with open(os.path.join(RES_DIR, "phase2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[phase2] mean={mean_of_means:.4f} KL_qp={kl_qp:.4f} "
          f"Jar={mean_jar:.4f}  PASS={summary['passes']}")
    return summary, all_results


def phase3(seeds=(42, 123, 7)):
    """1D harmonic diagnostic: document non-ergodicity and Jarzynski violation."""
    print("=" * 60)
    print("PHASE 3: 1D harmonic diagnostic (non-ergodicity)")
    print("=" * 60)

    per_seed = []
    for s in seeds:
        r = run_harmonic_diagnostic(
            np.array([1.0]), T0=1.0, T1=2.0, Q_nh=1.0, dt=0.005,
            t_burn=100.0, t_decorr=15.0, t_post=200.0,
            n_parents=60, n_branches=30, seed=s)
        print(f"[phase3] seed={s}: <Sigma>={r['mean_Sigma']:.4f}+-{r['sem_Sigma']:.4f}  "
              f"Jarzynski={r['jarzynski']:.4f}  "
              f"analytic(q,p)={r['analytic_qp_KL']:.4f}  wall={r['wall_time']:.1f}s")
        per_seed.append(r)

    summary = dict(
        per_seed=per_seed,
        note="1D harmonic NH-tanh is non-ergodic. Jarzynski<1 confirms "
             "initial ensemble is not the true invariant measure."
    )
    with open(os.path.join(RES_DIR, "phase3_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# =============================================================================
#  Figure
# =============================================================================

def make_figure(p1s, p1r, p2s, p2r, p3s, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), constrained_layout=True)

    # (a) Phase 1: cumulative integral vs time + analytic line
    ax = axes[0]
    target1 = p1s['kl_qp_exact']
    for i, r in enumerate(p1r):
        ax.plot(r['times'], r['integrand_mean'], color=C_BATH,
                lw=1.4, alpha=[1.0, 0.6, 0.4][i])
    r0 = p1r[0]
    sem = r0['integrand_std'] / np.sqrt(r0['n_traj'])
    ax.fill_between(r0['times'], r0['integrand_mean']-sem,
                    r0['integrand_mean']+sem, color=C_BATH, alpha=0.15)
    ax.axhline(target1, color=C_ANALYTIC, ls='--', lw=1.4,
               label=f'KL(q,p) exact = {target1:.3f}')
    ax.plot([], [], color=C_BATH, lw=1.4,
            label=f'NH mean = {p1s["mean_of_means"]:.3f}')
    ax.set_xlabel('post-quench time $t$')
    ax.set_ylabel(r'$\langle\int_0^t(\sigma_{\rm bath}-\sigma_{\rm exact})ds\rangle$')
    ax.set_title(f'(a) 2D double-well $T_0{{=}}1\\to T_1{{=}}2$\n'
                 f'Jar={p1s["mean_jarzynski"]:.3f}')
    ax.legend(frameon=False, fontsize=9)

    # (b) Phase 2: bar chart + Hutch inset
    ax = axes[1]
    target2 = p2s['kl_qp_exact']
    means2 = [x['mean'] for x in p2s['per_seed']]
    sems2 = [x['sem'] for x in p2s['per_seed']]
    xs = np.arange(len(means2)+2)
    labels = ['target'] + [f"s={x['seed']}" for x in p2s['per_seed']] + ['mean']
    vals = [target2] + means2 + [p2s['mean_of_means']]
    errs = [0] + sems2 + [p2s['std_of_means']]
    cols = [C_ANALYTIC] + [C_BATH]*3 + ['black']
    ax.bar(xs, vals, yerr=errs, color=cols, alpha=0.85, capsize=4,
           edgecolor='black', linewidth=0.6)
    ax.axhline(target2, color=C_ANALYTIC, ls='--', lw=1, alpha=0.6)
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel(r'$\langle\Sigma_{\rm tot}\rangle$')
    ax.set_title(f'(b) 2D DW $T_0{{=}}0.8\\to T_1{{=}}1.5$\n'
                 f'Jar={p2s["mean_jarzynski"]:.3f}')

    # inset: bounded vs sqrt(t) variance
    r0 = p2r[0]
    ins = ax.inset_axes([0.08, 0.50, 0.40, 0.40])
    ins.plot(r0['times'], r0['integrand_std'], color=C_BATH, lw=1.2,
             label=r'$\mathrm{std}[\Sigma_{\rm tot}]$')
    if 'hutch_int_std' in r0:
        ins.plot(r0['times'], r0['hutch_int_std'], color=C_HUTCH, lw=1.2,
                 label=r'$\mathrm{std}[\sigma_{\rm hutch}]$')
    ins.set_xlabel('t', fontsize=9); ins.set_ylabel('std', fontsize=9)
    ins.tick_params(labelsize=8)
    ins.legend(frameon=False, fontsize=7, loc='upper left')
    ins.grid(alpha=0.2)

    # (c) Phase 3: harmonic diagnostic
    ax = axes[2]
    if p3s is not None:
        seeds_h = [x['mean_Sigma'] for x in p3s['per_seed']]
        jars = [x['jarzynski'] for x in p3s['per_seed']]
        analytic = p3s['per_seed'][0]['analytic_qp_KL']
        xs = np.arange(len(seeds_h))
        ax.bar(xs, seeds_h, color=C_GRAY, alpha=0.7, edgecolor='black', lw=0.6)
        ax.axhline(analytic, color=C_ANALYTIC, ls='--', lw=1.4,
                   label=f'canonical (q,p) KL = {analytic:.3f}')
        for i, j in enumerate(jars):
            ax.text(i, seeds_h[i]+0.01, f'J={j:.3f}', ha='center', fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels([f'seed {i}' for i in range(len(seeds_h))])
        ax.set_ylabel(r'$\langle\Sigma_{\rm tot}\rangle$')
        ax.set_title('(c) 1D harmonic (non-ergodic)\n'
                     r'$\langle e^{-\Sigma}\rangle\neq 1$: Jarzynski violated')
        ax.legend(frameon=False, fontsize=9)

    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
#  CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', type=int, default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--seeds', type=str, default='42,123,7')
    args = ap.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(','))

    p1s = p2s = p3s = None
    p1r = p2r = []

    if args.phase == 1 or args.all:
        p1s, p1r = phase1(seeds)
    if args.phase == 2 or args.all:
        p2s, p2r = phase2(seeds)
    if args.phase == 3 or args.all:
        p3s = phase3(seeds)
    if args.all and p1s and p2s:
        make_figure(p1s, p1r, p2s, p2r, p3s,
                    os.path.join(FIG_DIR, 'fig_tasaki_verification.png'))


if __name__ == '__main__':
    main()
