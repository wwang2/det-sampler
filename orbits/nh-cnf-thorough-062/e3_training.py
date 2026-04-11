"""E3.3 Training stability: train an NH-CNF with parametric potential V_theta(x).

For a 2D target (pinwheel / 8-Gaussians), fit the density by maximum
log-likelihood. The potential V_theta is a small MLP (2 hidden, 32 units).
Compare three density-tracking schemes during training:

  (a) NH-CNF exact divergence  (analytical, zero variance)
  (b) FFJORD-style Hutchinson(1)
  (c) FFJORD-style Hutchinson(5)

Setup:
  - forward: x <- N(0, I), momenta p <- N(0, I), xi=0
  - integrate NH-tanh RK4 for T steps
  - log p(x_T) = log p_0(x_0) + int div dt
    (where div = -d tanh(xi) analytically, or Hutchinson trace for (b)/(c))
  - loss = -mean log p_target, where p_target is evaluated on training samples
    from the target by time-reversing. We use the "reverse KL" setup: push
    base samples forward through the flow and match to target via
    -log p_target(x_T) + log p_base(x_0) - cum_div  (reverse KL).

Actually we use a simpler and more standard training: forward KL via
importance-weighted ELBO is heavy. Instead we implement **reverse KL**:

  loss = E_{x~base}[ -log p_target(T(x)) - sum_step div ]

This is minimised when the flow pushes the base to the target. It uses
V_theta as part of the flow. Both schemes track div; the Hutchinson schemes
inject noise into the loss gradient.

We measure:
  - training loss curve (mean +/- std over 5 inits)
  - final test reverse-KL against ground-truth 8-Gaussians (since it's known)
  - gradient variance (norm of per-minibatch grad wrt theta, std/mean)
  - wall-clock per iteration
"""

import os, time, json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 130, 'savefig.dpi': 220, 'savefig.pad_inches': 0.2,
})

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGDIR, exist_ok=True)
RESDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESDIR, exist_ok=True)


# ---------------- target: 8-Gaussians ----------------

def make_8gauss_logprob(sigma=0.3, radius=3.0):
    angles = torch.tensor([i * np.pi / 4 for i in range(8)])
    centers = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=-1)  # [8, 2]

    def log_p_target(x):  # [B, 2]
        diff = x.unsqueeze(1) - centers.unsqueeze(0)  # [B, 8, 2]
        sq = (diff * diff).sum(-1)                    # [B, 8]
        log_comp = -0.5 * sq / (sigma * sigma) - np.log(2 * np.pi * sigma * sigma)  # [B, 8]
        return torch.logsumexp(log_comp, dim=-1) - np.log(8.0)
    return log_p_target, centers


# ---------------- parametric potential ----------------

class MLPPotential(nn.Module):
    def __init__(self, d=2, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1, bias=False),  # bias has no effect on grad_V; avoid None .grad
        )

    def forward(self, x):  # [B, d] -> [B]
        return self.net(x).squeeze(-1)

    def grad_V(self, x):
        x_ = x.requires_grad_(True)
        V = self.forward(x_).sum()
        gv, = torch.autograd.grad(V, x_, create_graph=True)
        return gv


# ---------------- NH-tanh step, differentiable through theta ----------------

def nh_tanh_rhs(q, p, xi, grad_V_fn, Q=1.0, kT=1.0):
    d = q.shape[-1]
    gv = grad_V_fn(q)
    g = torch.tanh(xi)
    dq = p
    dp = -gv - g * p
    dxi = (1.0 / Q) * ((p * p).sum(-1, keepdim=True) - d * kT)
    return dq, dp, dxi


def rk4_nh(q, p, xi, grad_V_fn, dt, Q=1.0, kT=1.0):
    k1q, k1p, k1x = nh_tanh_rhs(q, p, xi, grad_V_fn, Q, kT)
    k2q, k2p, k2x = nh_tanh_rhs(q + 0.5*dt*k1q, p + 0.5*dt*k1p, xi + 0.5*dt*k1x, grad_V_fn, Q, kT)
    k3q, k3p, k3x = nh_tanh_rhs(q + 0.5*dt*k2q, p + 0.5*dt*k2p, xi + 0.5*dt*k2x, grad_V_fn, Q, kT)
    k4q, k4p, k4x = nh_tanh_rhs(q + dt*k3q, p + dt*k3p, xi + dt*k3x, grad_V_fn, Q, kT)
    q_new  = q  + (dt/6.0) * (k1q + 2*k2q + 2*k3q + k4q)
    p_new  = p  + (dt/6.0) * (k1p + 2*k2p + 2*k3p + k4p)
    xi_new = xi + (dt/6.0) * (k1x + 2*k2x + 2*k3x + k4x)
    return q_new, p_new, xi_new


def hutch_div_step(q, p, xi, grad_V_fn, k=1, generator=None):
    """Hutchinson-k estimate of Tr(Jac) for the NH-tanh RHS on augmented state.

    Returns shape [B]. Uses Rademacher noise.
    """
    B, d = q.shape
    q_ = q.detach().clone().requires_grad_(True)
    p_ = p.detach().clone().requires_grad_(True)
    xi_ = xi.detach().clone().requires_grad_(True)
    dq, dp, dxi = nh_tanh_rhs(q_, p_, xi_, grad_V_fn)
    f_flat = torch.cat([dq, dp, dxi], dim=-1)  # [B, 2d+1]
    acc = torch.zeros(B)
    for _ in range(k):
        eps = (torch.randint(0, 2, f_flat.shape, generator=generator).float() * 2 - 1)
        s = (f_flat * eps).sum()
        gq, gp, gxi = torch.autograd.grad(s, [q_, p_, xi_], retain_graph=True)
        grad_flat = torch.cat([gq, gp, gxi], dim=-1)
        acc = acc + (grad_flat * eps).sum(-1)
    return acc / k


# ---------------- flow integration with loss ----------------

def flow_reverse_kl(V_theta, log_p_target, batch_size, n_steps, dt, method, hutch_k=1, gen=None):
    """Push base samples through NH-CNF. Return -log p_target(x_T) + log p_0(x_0) - cum_div, mean.

    Reverse KL: E_q[log q(x) - log p(x)] where q is the pushforward of base.
    log q(x_T) = log p_0(x_0) - cum_div. We minimise E[log q - log p].
    """
    d = 2
    q = torch.randn(batch_size, d)
    p = torch.randn(batch_size, d)
    xi = torch.zeros(batch_size, 1)
    log_p0 = -0.5 * d * np.log(2 * np.pi) - 0.5 * (q * q).sum(-1)
    cum_div = torch.zeros(batch_size)

    def gv(x):
        return V_theta.grad_V(x)

    for step in range(n_steps):
        if method == 'exact':
            q_new, p_new, xi_new = rk4_nh(q, p, xi, gv, dt)
            # exact: int -d tanh(xi) dt (trapezoidal)
            d_dim = q.shape[-1]
            div_incr = -d_dim * 0.5 * (torch.tanh(xi).squeeze(-1) + torch.tanh(xi_new).squeeze(-1)) * dt
            cum_div = cum_div + div_incr
            q, p, xi = q_new, p_new, xi_new
        else:
            div_est = hutch_div_step(q, p, xi, gv, k=hutch_k, generator=gen)
            cum_div = cum_div + div_est * dt
            q, p, xi = rk4_nh(q, p, xi, gv, dt)

    log_q = log_p0 - cum_div  # note: cum_div is int div f dt, so log q decreases by cum_div (change of variables)
    log_p = log_p_target(q)
    loss = (log_q - log_p).mean()
    return loss, q.detach()


# ---------------- training ----------------

def train_one(method, seed, log_p_target, n_iters=150, lr=3e-3, batch=128,
              n_flow_steps=20, dt=0.05, hutch_k=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator(); gen.manual_seed(seed + 1000)

    V = MLPPotential(d=2, hidden=32)
    opt = torch.optim.Adam(V.parameters(), lr=lr)

    losses = []
    grad_norms = []
    times = []

    for it in range(n_iters):
        t0 = time.perf_counter()
        opt.zero_grad()
        loss, _ = flow_reverse_kl(
            V, log_p_target, batch_size=batch, n_steps=n_flow_steps, dt=dt,
            method=method, hutch_k=hutch_k, gen=gen,
        )
        loss.backward()
        # grad norm
        gn = 0.0
        for p_ in V.parameters():
            if p_.grad is not None:
                gn += float((p_.grad * p_.grad).sum())
        gn = gn ** 0.5
        opt.step()
        t1 = time.perf_counter()

        losses.append(float(loss.item()))
        grad_norms.append(gn)
        times.append(t1 - t0)

    return {
        'loss': np.array(losses),
        'grad_norm': np.array(grad_norms),
        'time_per_iter': np.array(times),
        'model_state': V.state_dict(),
    }


def eval_test_nll(V, log_p_target, n_samples=512, n_flow_steps=20, dt=0.05):
    with torch.no_grad():
        pass  # we still need grads for V.grad_V (first-order through potential), but no backprop
    q = torch.randn(n_samples, 2)
    p = torch.randn(n_samples, 2)
    xi = torch.zeros(n_samples, 1)
    log_p0 = -0.5 * 2 * np.log(2 * np.pi) - 0.5 * (q * q).sum(-1)
    cum_div = torch.zeros(n_samples)

    def gv(x):
        return V.grad_V(x)

    for step in range(n_flow_steps):
        q_new, p_new, xi_new = rk4_nh(q, p, xi, gv, dt)
        div_incr = -2 * 0.5 * (torch.tanh(xi).squeeze(-1) + torch.tanh(xi_new).squeeze(-1)) * dt
        cum_div = cum_div + div_incr
        q, p, xi = q_new.detach(), p_new.detach(), xi_new.detach()
    log_q = (log_p0 - cum_div).detach()
    log_p = log_p_target(q).detach()
    rev_kl = float((log_q - log_p).mean().item())
    return rev_kl


def main():
    log_p_target, centers = make_8gauss_logprob()
    methods = [
        ('NH exact',  'exact', 0, '#1f77b4'),
        ('Hutch(1)',  'hutch', 1, '#ff7f0e'),
        ('Hutch(5)',  'hutch', 5, '#9467bd'),
    ]
    N_SEEDS = 4
    N_ITERS = 120
    BATCH = 64
    N_FLOW = 16
    DT = 0.05

    results = {}
    for label, method, k, _c in methods:
        print(f"\n=== {label} ===")
        runs = []
        for s in range(N_SEEDS):
            t0 = time.time()
            r = train_one(method, s, log_p_target,
                          n_iters=N_ITERS, batch=BATCH,
                          n_flow_steps=N_FLOW, dt=DT, hutch_k=k)
            # eval final
            V = MLPPotential(d=2, hidden=32)
            V.load_state_dict(r['model_state'])
            test_nll = eval_test_nll(V, log_p_target,
                                     n_samples=512, n_flow_steps=N_FLOW, dt=DT)
            r['test_nll'] = test_nll
            runs.append(r)
            print(f"  seed {s}: final train loss={r['loss'][-1]:.3f} test rev-KL={test_nll:.3f} "
                  f"t_iter={r['time_per_iter'].mean()*1000:.1f}ms total={time.time()-t0:.1f}s")
        results[label] = runs

    # save summary
    summary = {}
    for label, runs in results.items():
        losses = np.stack([r['loss'] for r in runs])           # [S, T]
        gnorms = np.stack([r['grad_norm'] for r in runs])      # [S, T]
        times_ = np.stack([r['time_per_iter'] for r in runs])  # [S, T]
        test_nlls = np.array([r['test_nll'] for r in runs])
        summary[label] = {
            'loss_mean': losses.mean(0).tolist(),
            'loss_std':  losses.std(0, ddof=1).tolist(),
            'grad_norm_mean': gnorms.mean(0).tolist(),
            'grad_norm_std':  gnorms.std(0, ddof=1).tolist(),
            'time_per_iter_mean': float(times_.mean()),
            'time_per_iter_std':  float(times_.std(ddof=1)),
            'test_rev_kl_mean': float(test_nlls.mean()),
            'test_rev_kl_std':  float(test_nlls.std(ddof=1)),
        }
    with open(os.path.join(RESDIR, 'e3_training.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # --- figure: 2 x 2 panel ---
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 2, 1)  # training loss
    ax2 = fig.add_subplot(2, 2, 2)  # gradient norm (proxy for gradient variance)
    ax3 = fig.add_subplot(2, 2, 3)  # final test rev-KL bar
    ax4 = fig.add_subplot(2, 2, 4)  # samples from trained flow

    it = np.arange(N_ITERS)
    for label, _method, _k, c in methods:
        m = np.array(summary[label]['loss_mean'])
        s = np.array(summary[label]['loss_std'])
        ax1.plot(it, m, '-', color=c, lw=2, label=label)
        ax1.fill_between(it, m - s, m + s, alpha=0.22, color=c)

        gm = np.array(summary[label]['grad_norm_mean'])
        gs = np.array(summary[label]['grad_norm_std'])
        ax2.plot(it, gm, '-', color=c, lw=2, label=label)
        ax2.fill_between(it, np.maximum(gm - gs, 1e-6), gm + gs, alpha=0.22, color=c)

    ax1.set_xlabel('iteration'); ax1.set_ylabel('training loss (reverse KL)')
    ax1.set_title('Training loss vs iteration (mean +/- std over %d inits)' % N_SEEDS)
    ax1.grid(True, alpha=0.3); ax1.legend(loc='upper right', framealpha=0.95)

    ax2.set_xlabel('iteration'); ax2.set_ylabel('||grad(theta) loss||')
    ax2.set_yscale('log')
    ax2.set_title('Gradient magnitude (Hutchinson inflates variance)')
    ax2.grid(True, alpha=0.3, which='both'); ax2.legend(loc='upper right', framealpha=0.95)

    # final rev-KL bar
    labels = [m[0] for m in methods]
    means = [summary[l]['test_rev_kl_mean'] for l in labels]
    stds  = [summary[l]['test_rev_kl_std'] for l in labels]
    colors = [m[3] for m in methods]
    ax3.bar(labels, means, yerr=stds, color=colors, capsize=6, alpha=0.85, edgecolor='black')
    ax3.set_ylabel('test reverse-KL (lower is better)')
    ax3.set_title('Final fit quality (mean +/- std over seeds)')
    ax3.grid(True, axis='y', alpha=0.3)

    # samples from the exact-trained flow
    V_ex = MLPPotential(d=2, hidden=32)
    V_ex.load_state_dict(results['NH exact'][0]['model_state'])
    with torch.no_grad():
        pass
    q = torch.randn(2000, 2); p = torch.randn(2000, 2); xi = torch.zeros(2000, 1)
    def gv(x): return V_ex.grad_V(x)
    for _ in range(N_FLOW):
        q, p, xi = rk4_nh(q, p, xi, gv, DT)
    samples = q.detach().numpy()
    ax4.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5, color='#1f77b4')
    c_np = centers.numpy()
    ax4.scatter(c_np[:, 0], c_np[:, 1], s=80, marker='x', color='red',
                linewidths=2, label='target modes')
    ax4.set_xlabel('x_1'); ax4.set_ylabel('x_2')
    ax4.set_title('NH exact — pushforward samples after training')
    ax4.set_aspect('equal'); ax4.grid(True, alpha=0.3); ax4.legend(loc='upper right', framealpha=0.95)
    ax4.set_xlim(-5, 5); ax4.set_ylim(-5, 5)

    fig.suptitle('E3.3  NH-CNF training stability: exact divergence vs Hutchinson', y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig_training_dynamics.png'), bbox_inches='tight')
    fig.savefig(os.path.join(FIGDIR, 'fig_training_dynamics.pdf'), bbox_inches='tight')
    print('saved fig_training_dynamics')


if __name__ == '__main__':
    main()
