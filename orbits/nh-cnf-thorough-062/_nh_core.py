"""Core NH-tanh CNF primitives shared across e3_variance / e3_training / e3_walltime.

Provides:
    nh_tanh_rhs(q, p, xi, grad_V_fn, Q, kT)      -> dq, dp, dxi
    div_exact(xi_start, xi_end, d, dt)           -> exact div integral (trap rule)
    div_hutch(v_fn, q, p, xi, Q, kT, k, grad_V_fn) -> Hutchinson(k) estimate of div
    rk4_step_nh(q, p, xi, grad_V_fn, dt, Q, kT)  -> new (q, p, xi)
"""

import torch
import numpy as np


def nh_tanh_f(q, p, xi, grad_V_fn, Q, kT):
    """NH-tanh ODE RHS. q,p shape [B,d]; xi shape [B,1]."""
    d = q.shape[-1]
    gv = grad_V_fn(q)
    g = torch.tanh(xi)
    dq = p
    dp = -gv - g * p
    dxi = (1.0 / Q) * ((p * p).sum(-1, keepdim=True) - d * kT)
    return dq, dp, dxi


def rk4_step_nh(q, p, xi, grad_V_fn, dt, Q=1.0, kT=1.0):
    """Return (q_new, p_new, xi_new)."""
    k1q, k1p, k1x = nh_tanh_f(q,                 p,                 xi,                 grad_V_fn, Q, kT)
    k2q, k2p, k2x = nh_tanh_f(q + 0.5*dt*k1q,    p + 0.5*dt*k1p,    xi + 0.5*dt*k1x,    grad_V_fn, Q, kT)
    k3q, k3p, k3x = nh_tanh_f(q + 0.5*dt*k2q,    p + 0.5*dt*k2p,    xi + 0.5*dt*k2x,    grad_V_fn, Q, kT)
    k4q, k4p, k4x = nh_tanh_f(q + dt*k3q,        p + dt*k3p,        xi + dt*k3x,        grad_V_fn, Q, kT)
    q_new  = q  + (dt/6.0) * (k1q + 2*k2q + 2*k3q + k4q)
    p_new  = p  + (dt/6.0) * (k1p + 2*k2p + 2*k3p + k4p)
    xi_new = xi + (dt/6.0) * (k1x + 2*k2x + 2*k3x + k4x)
    return q_new, p_new, xi_new


def div_exact_step(xi_start, xi_end, d, dt):
    """Exact divergence integral for one NH-tanh step: int -d*tanh(xi) dt (trapezoidal)."""
    g0 = torch.tanh(xi_start).squeeze(-1)
    g1 = torch.tanh(xi_end).squeeze(-1)
    return -d * 0.5 * (g0 + g1) * dt


def hutch_div_step(q, p, xi, grad_V_fn, k=1, Q=1.0, kT=1.0,
                   generator=None, create_graph=False):
    """Hutchinson(k) trace estimate of d/d(q,p,xi) . f for the NH-tanh RHS.

    Shared single source of truth used by both the variance experiments
    (detached forward use) and the training experiments (needs
    `create_graph=True` to backprop through theta).

    Returns a [B]-shaped tensor; caller may detach if they don't need grads.
    """
    q_ = q.requires_grad_(True) if not q.requires_grad else q
    p_ = p.requires_grad_(True) if not p.requires_grad else p
    xi_ = xi.requires_grad_(True) if not xi.requires_grad else xi
    dq, dp, dxi = nh_tanh_f(q_, p_, xi_, grad_V_fn, Q, kT)
    f_flat = torch.cat([dq, dp, dxi], dim=-1)

    acc = torch.zeros(q.shape[0], device=q.device)
    for _ in range(k):
        eps = (torch.randint(0, 2, f_flat.shape, generator=generator,
                             device=q.device).float() * 2 - 1)
        s = (f_flat * eps).sum()
        gq, gp, gxi = torch.autograd.grad(
            s, [q_, p_, xi_], retain_graph=True, create_graph=create_graph,
        )
        grad_flat = torch.cat([gq, gp, gxi], dim=-1)
        acc = acc + (grad_flat * eps).sum(-1)
    return acc / k


def trace_jac_hutch_step(q, p, xi, grad_V_fn, Q, kT, k, generator=None):
    """Hutchinson(k) estimate of trace(Jac(f)) for the NH-tanh ODE RHS.

    For the full augmented state (q, p, xi) of dimension 2d+1, the Jacobian
    trace equals
         d/dq (p) + d/dp (-gv - tanh(xi) p) + d/dxi ((||p||^2 - d kT)/Q)
       = 0        + -d * tanh(xi)           + 0
       = -d * tanh(xi)            (analytic)

    A Hutchinson estimate over the full augmented state should recover this
    in expectation. To emulate what a FFJORD-style estimator running on this
    RHS would see (it does NOT know the structure), we sample eps_q, eps_p,
    eps_xi Rademacher and compute eps^T (Jac f) eps via autograd.

    Returns estimate shape [B] (detached, no grad graph).
    """
    # Thin wrapper over hutch_div_step. Detaches inputs first (variance-mode
    # use case) and returns a detached result.
    q_ = q.detach().clone()
    p_ = p.detach().clone()
    xi_ = xi.detach().clone()
    return hutch_div_step(q_, p_, xi_, grad_V_fn, k=k, Q=Q, kT=kT,
                          generator=generator, create_graph=False).detach()


def run_nh_cnf_batch(grad_V_fn, n_samples, d, n_steps, dt=0.01,
                     Q=1.0, kT=1.0, seed=42, mode='exact', hutch_k=1):
    """Integrate NH-CNF ODE, tracking log-density.

    mode: 'exact' -> analytic div integral; 'hutch' -> Hutchinson(k) per step.

    Returns (q_numpy [n_samples, d], log_probs_numpy [n_samples]).
    """
    g = torch.Generator()
    g.manual_seed(seed)
    q = torch.randn(n_samples, d, generator=g)
    p = torch.randn(n_samples, d, generator=g)
    xi = torch.zeros(n_samples, 1)

    # log prob under N(0,I)_{2d+1} ignoring xi (xi pin at 0 deterministic, treat as
    # augmented Dirac -> prob density on x is marginal; we track relative log p of x
    # via the standard CNF identity)
    log_p0 = -0.5 * d * np.log(2 * np.pi) - 0.5 * (q * q).sum(-1)
    cum_div = torch.zeros(n_samples)

    for step in range(n_steps):
        if mode == 'hutch':
            # hutchinson: compute div estimate at current state, scale by dt
            div_est = trace_jac_hutch_step(q, p, xi, grad_V_fn, Q, kT, hutch_k, generator=g)
            cum_div = cum_div + div_est * dt
            q, p, xi = rk4_step_nh(q, p, xi, grad_V_fn, dt, Q, kT)
        else:
            q_new, p_new, xi_new = rk4_step_nh(q, p, xi, grad_V_fn, dt, Q, kT)
            cum_div = cum_div + div_exact_step(xi, xi_new, d, dt)
            q, p, xi = q_new, p_new, xi_new

    log_probs = log_p0 + cum_div
    return q.detach().numpy(), log_probs.detach().numpy()


# ----- target families -----

def make_iso_gaussian(d):
    def V(x):
        return 0.5 * (x * x).sum(-1)
    def grad_V(x):
        return x
    return V, grad_V


def make_aniso_gaussian(d, kappa_min=1.0, kappa_max=100.0, seed=0):
    rng = np.random.default_rng(seed)
    kappas = np.logspace(np.log10(kappa_min), np.log10(kappa_max), d)
    rng.shuffle(kappas)
    kappas_t = torch.tensor(kappas, dtype=torch.float32)
    def V(x):
        return 0.5 * (kappas_t * x * x).sum(-1)
    def grad_V(x):
        return kappas_t * x
    return V, grad_V


def make_bimodal(d):
    """-log(0.5 N(-e1) + 0.5 N(+e1)) (isotropic cov around each mode).

    Analytical differentiable gradient so Hutchinson autograd flows through q.
    For unit-variance components at +/- e1, grad_V(x)[0] = x[0] - tanh(x[0]);
    grad_V(x)[k>=1] = x[k].
    """
    e1 = torch.zeros(d); e1[0] = 1.0

    def V(x):
        d_pos = x - e1
        d_neg = x + e1
        quad_pos = 0.5 * (d_pos * d_pos).sum(-1)
        quad_neg = 0.5 * (d_neg * d_neg).sum(-1)
        m = torch.minimum(quad_pos, quad_neg)
        return -torch.log(0.5 * torch.exp(-(quad_pos - m)) + 0.5 * torch.exp(-(quad_neg - m))) + m

    def grad_V(x):
        x1 = x[..., 0:1]
        tanh_x1 = torch.tanh(x1)
        # shift only the first coordinate; keep graph-connected to x
        shift = torch.zeros_like(x)
        shift[..., 0:1] = -tanh_x1
        return x + shift

    return V, grad_V
