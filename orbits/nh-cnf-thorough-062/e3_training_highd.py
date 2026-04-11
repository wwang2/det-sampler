"""E3.3 Training stability — the headline figure.

Two panels:

  (a) Loss variance across 100 independent Monte Carlo draws, for
      NH exact, Hutch(1), Hutch(5), Hutch(20). NH exact should be
      exactly zero (deterministic divergence integral).

  (b) Gradient noise-to-signal ratio std(dL/dtheta)/||mean|| as a
      function of dimension d in {2, 5, 10, 20, 50}. NH exact sits at
      machine precision; Hutchinson grows with d.

Protocol (simpler than full end-to-end training):

  * Fix a parametric potential V_theta(x) = MLP(x) with two hidden Tanh
    layers of 32 units and a bias-free output layer (bias=False removes
    the NoneType .grad issue since the output bias never influences
    grad_V).
  * Freeze V_theta at its random init.
  * The training loss is the reverse-KL
        L(theta) = E_{x0}[ log q_T(x_T) - log p_target(x_T) ],
    where q_T is the NH-CNF pushforward of N(0, I) after T steps and
    log q_T is tracked with either the exact NH formula (int -d tanh xi dt)
    or a Hutchinson(k) trace estimate of d/dq . f on the augmented state.
  * The gradient noise-to-signal panel and the loss-noise panel both use
    this same frozen V_theta.
"""

import os, json, time
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


# ------------------------ model ------------------------

class MLPPotential(nn.Module):
    """V_theta(x) : R^d -> R. Final Linear has bias=False so every theta
    parameter has a nonzero gradient through grad_V (the output bias would
    otherwise give None/zero grads and crash Adam)."""

    def __init__(self, d, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def grad_V(self, x):
        x_ = x.requires_grad_(True)
        V = self.forward(x_).sum()
        gv, = torch.autograd.grad(V, x_, create_graph=True)
        return gv


# ------------------------ NH-tanh primitives ------------------------

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


def hutch_div_step(q, p, xi, grad_V_fn, k=1, generator=None, create_graph=False):
    """Hutchinson(k) trace estimate of d/d(q,p,xi) . f for the NH-tanh RHS."""
    q_ = q.requires_grad_(True) if not q.requires_grad else q
    p_ = p.requires_grad_(True) if not p.requires_grad else p
    xi_ = xi.requires_grad_(True) if not xi.requires_grad else xi
    dq, dp, dxi = nh_tanh_rhs(q_, p_, xi_, grad_V_fn)
    f_flat = torch.cat([dq, dp, dxi], dim=-1)

    acc = torch.zeros(q.shape[0])
    for _ in range(k):
        eps = (torch.randint(0, 2, f_flat.shape, generator=generator).float() * 2 - 1)
        s = (f_flat * eps).sum()
        gq, gp, gxi = torch.autograd.grad(
            s, [q_, p_, xi_], retain_graph=True, create_graph=create_graph,
        )
        grad_flat = torch.cat([gq, gp, gxi], dim=-1)
        acc = acc + (grad_flat * eps).sum(-1)
    return acc / k


def target_logp(x):
    """Isotropic standard normal target in d dimensions."""
    return -0.5 * (x * x).sum(-1) - 0.5 * x.shape[-1] * np.log(2 * np.pi)


def flow_loss(V, x0, n_steps, dt, method, hutch_k=1, gen=None, create_graph=False, p0=None):
    """Reverse-KL loss on base samples x0 pushed through the NH-CNF.

    Returns a scalar tensor.  `create_graph` must be True when we want to
    take grad through theta (for gradient-noise measurement).

    x0 and p0 are fixed base/momentum samples. Only the Hutchinson noise
    (driven by `gen`) varies across calls; NH exact is then deterministic.
    """
    d = x0.shape[-1]
    q = x0
    if p0 is None:
        p = torch.randn(q.shape[0], d, generator=gen) if gen is not None else torch.randn_like(q)
    else:
        p = p0
    xi = torch.zeros(q.shape[0], 1)
    log_p0 = -0.5 * d * np.log(2 * np.pi) - 0.5 * (q * q).sum(-1)
    cum_div = torch.zeros(q.shape[0])

    def gv(x):
        return V.grad_V(x)

    for _step in range(n_steps):
        if method == 'exact':
            q_new, p_new, xi_new = rk4_nh(q, p, xi, gv, dt)
            incr = -d * 0.5 * (torch.tanh(xi).squeeze(-1) + torch.tanh(xi_new).squeeze(-1)) * dt
            cum_div = cum_div + incr
            q, p, xi = q_new, p_new, xi_new
        else:
            div_est = hutch_div_step(
                q, p, xi, gv, k=hutch_k, generator=gen, create_graph=create_graph,
            )
            cum_div = cum_div + div_est * dt
            q, p, xi = rk4_nh(q, p, xi, gv, dt)

    log_q = log_p0 - cum_div
    log_p = target_logp(q)
    return (log_q - log_p).mean()


# ------------------------ panel (a): loss variance ------------------------

def panel_a_loss_variance(d=10, n_draws=100, n_flow=16, dt=0.05, batch=64):
    """Return dict: method -> list of n_draws loss values."""
    torch.manual_seed(0)
    V = MLPPotential(d=d, hidden=32)
    for p_ in V.parameters(): p_.requires_grad_(False)  # freeze theta
    x0 = torch.randn(batch, d)                          # fixed base sample
    p0 = torch.randn(batch, d)                          # fixed base momentum

    methods = [
        ('NH exact',  'exact', 0),
        ('Hutch(1)',  'hutch', 1),
        ('Hutch(5)',  'hutch', 5),
        ('Hutch(20)', 'hutch', 20),
    ]
    out = {}
    for name, mode, k in methods:
        vals = []
        for t in range(n_draws):
            gen = torch.Generator(); gen.manual_seed(9000 + t)
            # freshly sample p inside flow_loss via the generator
            with torch.no_grad():
                pass
            loss = flow_loss(V, x0, n_steps=n_flow, dt=dt, method=mode,
                             hutch_k=k, gen=gen, create_graph=False, p0=p0)
            vals.append(float(loss.detach().item()))
        out[name] = vals
        print(f"  (a) {name:10s} loss std = {np.std(vals, ddof=1):.3e}  mean = {np.mean(vals):.3f}")
    return out


# ------------------------ panel (b): gradient noise vs d ------------------------

def measure_grad_noise(d, method, hutch_k=0, n_trials=12, n_flow=10, dt=0.05, batch=32):
    torch.manual_seed(0)
    V = MLPPotential(d=d, hidden=32)
    x0 = torch.randn(batch, d)
    p0 = torch.randn(batch, d)

    grads = []
    for t in range(n_trials):
        gen = torch.Generator(); gen.manual_seed(100 + t)
        for p_ in V.parameters():
            p_.grad = None
        loss = flow_loss(V, x0, n_steps=n_flow, dt=dt, method=method,
                         hutch_k=hutch_k, gen=gen, create_graph=True, p0=p0)
        loss.backward()
        g_parts = []
        for p_ in V.parameters():
            assert p_.grad is not None, 'None grad — check bias=False fix'
            g_parts.append(p_.grad.detach().flatten())
        grads.append(torch.cat(g_parts))
    G = torch.stack(grads)   # [T, P]
    g_mean = G.mean(0)
    g_std = G.std(0, unbiased=True)
    rel = float(g_std.norm() / (g_mean.norm() + 1e-30))
    return rel, float(g_mean.norm()), float(g_std.norm())


def panel_b_grad_noise(dims=(2, 5, 10, 20, 50), n_trials=10):
    methods = [
        ('NH exact',  'exact', 0),
        ('Hutch(1)',  'hutch', 1),
        ('Hutch(5)',  'hutch', 5),
        ('Hutch(20)', 'hutch', 20),
    ]
    out = {name: {} for name, _, _ in methods}
    for d in dims:
        for name, mode, k in methods:
            rel, mn, sn = measure_grad_noise(d, mode, hutch_k=k, n_trials=n_trials)
            out[name][d] = {'rel_noise': rel, 'mean_norm': mn, 'std_norm': sn}
            print(f"  (b) d={d:3d} {name:10s} rel-noise={rel:.3e}  ||mean||={mn:.3e}")
    return out


# ------------------------ main ------------------------

def main():
    t_start = time.time()
    print('panel (a) — loss variance at d=10')
    loss_data = panel_a_loss_variance(d=10, n_draws=100)
    print('panel (b) — grad noise vs dimension')
    DIMS = [2, 5, 10, 20, 50]
    grad_data = panel_b_grad_noise(dims=DIMS, n_trials=10)

    # headline metric: grad noise ratio at d=10 for NH exact
    metric_nh_d10 = grad_data['NH exact'][10]['rel_noise']
    print(f"\nHEADLINE metric: grad-noise ratio at d=10 for NH exact = {metric_nh_d10:.3e}")

    with open(os.path.join(RESDIR, 'e3_training_highd.json'), 'w') as f:
        json.dump({
            'loss_variance_d10': loss_data,
            'grad_noise_vs_d':   {m: {str(d): v for d, v in md.items()}
                                  for m, md in grad_data.items()},
            'metric_nh_exact_d10': metric_nh_d10,
            'DIMS': DIMS,
        }, f, indent=2)

    # -------- figure --------
    fig = plt.figure(figsize=(12, 4.5))
    ax_a = fig.add_subplot(1, 2, 1)
    ax_b = fig.add_subplot(1, 2, 2)

    colors = {
        'NH exact':  '#1f77b4',
        'Hutch(1)':  '#ff7f0e',
        'Hutch(5)':  '#9467bd',
        'Hutch(20)': '#8c564b',
    }
    method_order = ['NH exact', 'Hutch(1)', 'Hutch(5)', 'Hutch(20)']

    # (a) loss variance — boxplot + scatter, log-y on std
    data_a = [loss_data[m] for m in method_order]
    bp = ax_a.boxplot(
        data_a, tick_labels=method_order, widths=0.55, patch_artist=True,
        medianprops=dict(color='black', lw=1.5),
        whiskerprops=dict(color='#555', lw=1.2),
        capprops=dict(color='#555', lw=1.2),
        flierprops=dict(marker='.', markerfacecolor='#888', markersize=4),
    )
    for patch, m in zip(bp['boxes'], method_order):
        patch.set_facecolor(colors[m]); patch.set_alpha(0.55)
        patch.set_edgecolor('black')
    ax_a.set_ylabel('reverse-KL loss (100 fresh MC draws)')
    ax_a.set_title('(a) Loss variance, $d=10$')
    ax_a.grid(True, axis='y', alpha=0.3)

    # annotate stds
    for i, m in enumerate(method_order):
        s = np.std(loss_data[m], ddof=1)
        ax_a.annotate(f'std={s:.2e}', xy=(i + 1, max(loss_data[m])),
                      xytext=(0, 10), textcoords='offset points',
                      ha='center', fontsize=9, color=colors[m])
    ax_a.set_xticklabels(method_order, rotation=0)

    # (b) gradient noise vs d
    DIMS_a = np.array(DIMS)
    for m in method_order:
        rel = np.array([grad_data[m][d]['rel_noise'] for d in DIMS])
        ax_b.plot(DIMS_a, np.maximum(rel, 1e-16), '-o', color=colors[m],
                  lw=2, ms=6, label=m)
    ax_b.set_xscale('log')
    ax_b.set_yscale('log')
    ax_b.set_xlabel('dimension  $d$')
    ax_b.set_ylabel(r'$\|\mathrm{std}(\nabla_\theta L)\| \,/\, \|\mathrm{mean}(\nabla_\theta L)\|$')
    ax_b.set_title('(b) Gradient noise-to-signal vs dimension')
    ax_b.grid(True, alpha=0.3, which='both')
    ax_b.legend(loc='lower right', framealpha=0.95)

    fig.suptitle(
        'E3.3  NH-CNF training stability: exact divergence is deterministic at every $d$',
        y=1.02,
    )
    fig.tight_layout()
    out_png = os.path.join(FIGDIR, 'fig_training_stability.png')
    out_pdf = os.path.join(FIGDIR, 'fig_training_stability.pdf')
    fig.savefig(out_png, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    print('saved', out_png, f'({time.time() - t_start:.1f}s total)')


if __name__ == '__main__':
    main()
