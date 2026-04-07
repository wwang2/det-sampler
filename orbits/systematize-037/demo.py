"""Demo: three targets exercised by det_sampler.MultiScaleThermostat.

Targets
-------
1. 1D harmonic oscillator (basic correctness)
2. 10D anisotropic Gaussian, kappa_ratio = 100 (auto kappa estimation)
3. 2D Gaussian Mixture (multi-modal mode hopping)

Saves a triple-panel figure to figures/demo_output.png.
"""
from __future__ import annotations

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from det_sampler import MultiScaleThermostat  # noqa: E402


def demo_1d_harmonic():
    grad_log_p = lambda q: -q  # N(0, 1)
    sampler = MultiScaleThermostat(grad_log_p, dim=1, kappa_range=(1.0, 1.0), seed=1)
    samples = sampler.sample(np.zeros(1), n_samples=20000, burn_in=2000)
    diag = sampler.diagnostics()
    return samples[:, 0], diag


def demo_10d_anisotropic():
    dim = 10
    kappas = np.geomspace(1.0, 100.0, dim)  # diagonal precisions
    grad_log_p = lambda q: -kappas * q
    sampler = MultiScaleThermostat(grad_log_p, dim=dim, seed=2)
    samples = sampler.sample(np.zeros(dim), n_samples=60000, burn_in=5000)
    diag = sampler.diagnostics()
    return samples, kappas, diag


def demo_2d_mixture():
    centers = np.array([[-2.0, 0.0], [2.0, 0.0]])
    sigma2 = 0.5

    def grad_log_p(q):
        # log p = log( sum_k exp(-||q-c_k||^2 / (2 sigma2)) )
        diffs = q[None, :] - centers
        log_w = -0.5 * np.sum(diffs * diffs, axis=1) / sigma2
        log_w -= log_w.max()
        w = np.exp(log_w)
        w /= w.sum()
        grad = -np.sum(w[:, None] * diffs, axis=0) / sigma2
        return grad

    sampler = MultiScaleThermostat(
        grad_log_p, dim=2, kappa_range=(1.0 / sigma2, 1.0 / sigma2), seed=3
    )
    samples = sampler.sample(np.array([-2.0, 0.0]), n_samples=20000, burn_in=2000)
    diag = sampler.diagnostics()
    return samples, centers, sigma2, diag


def main():
    print("=== Demo 1: 1D harmonic ===")
    s1, d1 = demo_1d_harmonic()
    print(f"  mean={s1.mean():+.3f}  var={s1.var():.3f} (target 0, 1)")
    print(f"  ESS/eval = {d1['ess_per_force_eval']:.4e}")

    print("=== Demo 2: 10D anisotropic ===")
    s2, kappas, d2 = demo_10d_anisotropic()
    target_var = 1.0 / kappas
    emp_var = s2.var(0)
    rel_err = np.abs(emp_var - target_var) / target_var
    print(f"  max rel var err = {rel_err.max():.3f}  (per dim: {np.round(rel_err, 3)})")
    print(f"  tau_int: {np.round(d2['tau_int'], 1)}")
    print(f"  detected kappa range = {d2['kappa_range']}")
    print(f"  N = {d2['N']}, dt = {d2['dt']:.4g}")
    print(f"  ESS/eval (min dim) = {d2['ess_per_force_eval']:.4e}")
    metric_10d = d2["ess_per_force_eval"]

    print("=== Demo 3: 2D mixture ===")
    s3, centers, sigma2, d3 = demo_2d_mixture()
    n_left = int((s3[:, 0] < 0).sum())
    n_right = len(s3) - n_left
    print(f"  left mode: {n_left}, right mode: {n_right}")
    print(f"  ESS/eval = {d3['ess_per_force_eval']:.4e}")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))

    ax = axes[0]
    xs = np.linspace(-4, 4, 200)
    ax.hist(s1, bins=60, density=True, alpha=0.55, color="#4477AA",
            edgecolor="none", label="sampler")
    ax.plot(xs, np.exp(-0.5 * xs ** 2) / math.sqrt(2 * math.pi),
            "k-", lw=1.4, label="N(0,1)")
    ax.set_title("(a) 1D harmonic")
    ax.set_xlabel("q")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    dims = np.arange(len(kappas))
    ax.plot(dims, target_var, "k-o", lw=1.2, ms=4, label="target 1/$\\kappa$")
    ax.plot(dims, emp_var, "s", color="#CC6677", ms=5, label="sampler")
    ax.set_yscale("log")
    ax.set_title("(b) 10D anisotropic")
    ax.set_xlabel("dimension")
    ax.set_ylabel("variance")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.scatter(s3[::5, 0], s3[::5, 1], s=2, alpha=0.35, color="#117733")
    ax.scatter(centers[:, 0], centers[:, 1], marker="x",
               color="k", s=60, lw=1.5, label="modes")
    ax.set_title("(c) 2D mixture")
    ax.set_xlabel("$q_1$")
    ax.set_ylabel("$q_2$")
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig_dir = os.path.join(HERE, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.join(fig_dir, "demo_output.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure -> {out_path}")

    print(f"\n*** Headline metric (ESS/eval on 10D anisotropic): {metric_10d:.4e} ***")
    return metric_10d


if __name__ == "__main__":
    main()
