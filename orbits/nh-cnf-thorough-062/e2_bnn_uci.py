"""E2 Bayesian NN posterior sampling via NH-CNF vs SGLD.

Three small regression problems. Methods: NH-CNF (NH-tanh flow sampling
weights from the posterior) and SGLD. Metrics: test NLL, 95% coverage.

Stability notes:
  - We use a tempered posterior (potential scaled by 1/N_data + prior) so
    gradients stay O(1). This keeps both NH and SGLD numerically stable.
  - NH-CNF uses n_chains independent trajectories and returns the final state.
"""

import os, json, time
import numpy as np
import torch
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


# ---------------- dataset ----------------

def make_dataset(name, seed=0):
    rng = np.random.default_rng(seed)
    if name == 'sine':
        n = 200
        X = rng.uniform(-3, 3, (n, 1)).astype(np.float32)
        y = (np.sin(2 * X[:, 0]) + 0.1 * X[:, 0]**2
             + rng.normal(0, 0.1, n)).astype(np.float32)
    elif name == 'boston-like':
        n = 240
        X = rng.standard_normal((n, 4)).astype(np.float32)
        w = np.array([1.0, -0.5, 0.3, 0.7], dtype=np.float32)
        y = (X @ w + 0.3 * X[:, 0] * X[:, 1]
             + rng.normal(0, 0.15, n)).astype(np.float32)
    elif name == 'concrete-like':
        n = 260
        X = rng.uniform(0, 1, (n, 6)).astype(np.float32)
        y = (2 * np.tanh(X[:, 0] - X[:, 1])
             + X[:, 2] * X[:, 3]
             + 0.5 * X[:, 4]
             + rng.normal(0, 0.1, n)).astype(np.float32)
    else:
        raise ValueError(name)
    n_train = int(0.8 * len(X))
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    Xtr, ytr = X[:n_train], y[:n_train]
    Xte, yte = X[n_train:], y[n_train:]
    y_mean, y_std = ytr.mean(), ytr.std() + 1e-6
    ytr_n = (ytr - y_mean) / y_std
    yte_n = (yte - y_mean) / y_std
    # Also normalise X
    x_mean = Xtr.mean(0); x_std = Xtr.std(0) + 1e-6
    Xtr_n = (Xtr - x_mean) / x_std
    Xte_n = (Xte - x_mean) / x_std
    return (torch.tensor(Xtr_n), torch.tensor(ytr_n),
            torch.tensor(Xte_n), torch.tensor(yte_n),
            float(y_mean), float(y_std))


# ---------------- BNN ----------------

class FlatBNN:
    def __init__(self, d_in, d_hid=8):
        self.d_in = d_in
        self.d_hid = d_hid
        self.shapes = [(d_hid, d_in), (d_hid,), (1, d_hid), (1,)]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.dim = sum(self.sizes)

    def unflatten(self, theta):
        out, i = [], 0
        for s, sz in zip(self.shapes, self.sizes):
            out.append(theta[i:i + sz].reshape(s)); i += sz
        return out

    def forward(self, theta, X):
        W1, b1, W2, b2 = self.unflatten(theta)
        h = torch.tanh(X @ W1.T + b1)
        return (h @ W2.T + b2).squeeze(-1)


def make_tempered_potential(bnn, X, y, sigma_prior=1.0, sigma_lik=0.3, temperature=None):
    """Tempered posterior: V(theta) = -log p(theta) - (1/T) log p(D|theta).
    T = len(D) keeps gradients O(1)."""
    N = X.shape[0]
    if temperature is None:
        temperature = float(N)

    def V(theta):  # theta [P]
        log_prior = -0.5 * (theta * theta).sum(-1) / (sigma_prior * sigma_prior)
        yhat = bnn.forward(theta, X)
        log_lik = -0.5 * ((yhat - y) ** 2).sum(-1) / (sigma_lik * sigma_lik)
        return -(log_prior + log_lik / temperature)

    def V_batch(Theta):
        return torch.stack([V(Theta[b]) for b in range(Theta.shape[0])])

    def grad_V(Theta):
        Theta_ = Theta.detach().clone().requires_grad_(True)
        Vs = V_batch(Theta_)
        gv, = torch.autograd.grad(Vs.sum(), Theta_)
        return gv.detach()

    return V_batch, grad_V


# ---------------- samplers ----------------

def nh_posterior_samples(grad_V, d, n_samples, n_steps=200, dt=0.01, Q=1.0, kT=1.0, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(n_samples, d) * 0.3
    p = torch.randn(n_samples, d) * 0.3
    xi = torch.zeros(n_samples, 1)

    def f(q_, p_, xi_):
        gv = grad_V(q_)
        g = torch.tanh(xi_)
        dq = p_
        dp = -gv - g * p_
        dxi = (1.0 / Q) * ((p_ * p_).sum(-1, keepdim=True) - d * kT)
        return dq, dp, dxi

    for step in range(n_steps):
        k1q, k1p, k1x = f(q, p, xi)
        k2q, k2p, k2x = f(q + 0.5*dt*k1q, p + 0.5*dt*k1p, xi + 0.5*dt*k1x)
        k3q, k3p, k3x = f(q + 0.5*dt*k2q, p + 0.5*dt*k2p, xi + 0.5*dt*k2x)
        k4q, k4p, k4x = f(q + dt*k3q, p + dt*k3p, xi + dt*k3x)
        q = q + (dt/6.0) * (k1q + 2*k2q + 2*k3q + k4q)
        p = p + (dt/6.0) * (k1p + 2*k2p + 2*k3p + k4p)
        xi = xi + (dt/6.0) * (k1x + 2*k2x + 2*k3x + k4x)
    return q


def sgld_samples(grad_V, d, n_samples, n_steps=500, eps=0.005, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(n_samples, d) * 0.3
    for step in range(n_steps):
        gv = grad_V(q)
        q = q - eps * gv + torch.randn_like(q) * np.sqrt(2 * eps)
    return q


# ---------------- evaluation ----------------

def evaluate(samples, bnn, Xte, yte, sigma_lik=0.3):
    preds = torch.stack([bnn.forward(samples[i], Xte) for i in range(samples.shape[0])])
    mu = preds.mean(0)
    var = preds.var(0) + sigma_lik ** 2
    nll = 0.5 * np.log(2 * np.pi) + 0.5 * torch.log(var) + 0.5 * (yte - mu) ** 2 / var
    std_pred = torch.sqrt(var)
    z = (yte - mu).abs() / std_pred
    coverage = (z < 1.96).float().mean()
    return float(nll.mean().item()), float(coverage.item())


# ---------------- main ----------------

def main():
    datasets = ['sine', 'boston-like', 'concrete-like']
    methods = ['NH-CNF', 'SGLD']
    N_SEEDS = 3
    N_POST = 40

    results = {}
    for name in datasets:
        results[name] = {}
        for method in methods:
            nll_seeds, cov_seeds, t_seeds = [], [], []
            for s in range(N_SEEDS):
                Xtr, ytr, Xte, yte, ym, ysd = make_dataset(name, seed=s)
                bnn = FlatBNN(d_in=Xtr.shape[1], d_hid=8)
                _, grad_V = make_tempered_potential(bnn, Xtr, ytr,
                                                    sigma_prior=1.0, sigma_lik=0.3)

                t0 = time.time()
                if method == 'NH-CNF':
                    samples = nh_posterior_samples(grad_V, bnn.dim, n_samples=N_POST,
                                                   n_steps=300, dt=0.01, seed=s * 10 + 1)
                else:
                    samples = sgld_samples(grad_V, bnn.dim, n_samples=N_POST,
                                           n_steps=600, eps=0.003, seed=s * 10 + 1)
                t_elapsed = time.time() - t0

                # Check for NaN
                if torch.isnan(samples).any() or torch.isinf(samples).any():
                    print(f"  {name:14s} {method:8s} seed={s} DIVERGED")
                    continue
                nll, cov = evaluate(samples, bnn, Xte, yte)
                if not np.isfinite(nll):
                    print(f"  {name:14s} {method:8s} seed={s} non-finite NLL={nll}")
                    continue
                nll_seeds.append(nll)
                cov_seeds.append(cov)
                t_seeds.append(t_elapsed)
            if len(nll_seeds) == 0:
                results[name][method] = {'nll_mean': float('nan'), 'nll_std': 0.0,
                                         'cov_mean': 0.0, 'cov_std': 0.0, 'wall_mean': 0.0}
            else:
                results[name][method] = {
                    'nll_mean': float(np.mean(nll_seeds)),
                    'nll_std':  float(np.std(nll_seeds, ddof=1) if len(nll_seeds) > 1 else 0.0),
                    'cov_mean': float(np.mean(cov_seeds)),
                    'cov_std':  float(np.std(cov_seeds, ddof=1) if len(cov_seeds) > 1 else 0.0),
                    'wall_mean': float(np.mean(t_seeds)),
                }
            print(f"  {name:14s} {method:8s} NLL={results[name][method]['nll_mean']:.3f}+/-"
                  f"{results[name][method]['nll_std']:.3f}  "
                  f"cov={results[name][method]['cov_mean']:.2f}  "
                  f"wall={results[name][method]['wall_mean']:.1f}s")

    with open(os.path.join(RESDIR, 'e2_bnn_uci.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(datasets))
    w = 0.35
    colors = {'NH-CNF': '#1f77b4', 'SGLD': '#2ca02c'}
    for ax, key, ylabel, title in [
        (axes[0], 'nll', 'test NLL (lower is better)', 'Posterior predictive NLL'),
        (axes[1], 'cov', '95% CI coverage', 'Posterior predictive coverage'),
    ]:
        for i, method in enumerate(methods):
            means = [results[d][method][f'{key}_mean'] for d in datasets]
            stds  = [results[d][method][f'{key}_std']  for d in datasets]
            ax.bar(x + (i - 0.5) * w, means, w, yerr=stds, label=method,
                   color=colors[method], capsize=4, alpha=0.9, edgecolor='black')
        ax.set_xticks(x); ax.set_xticklabels(datasets, rotation=15)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(loc='upper right', framealpha=0.95)
    axes[1].axhline(0.95, color='red', ls='--', lw=1, alpha=0.6)

    fig.suptitle('E2  BNN posterior sampling: NH-CNF vs SGLD (tempered posterior, 3 seeds)', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig_bnn_uci.png'), bbox_inches='tight')
    fig.savefig(os.path.join(FIGDIR, 'fig_bnn_uci.pdf'), bbox_inches='tight')
    print('saved fig_bnn_uci')


if __name__ == '__main__':
    main()
