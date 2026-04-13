"""
Adaptive Simulated Annealing via NH Bath Entropy Signal

Core idea: NH dynamics produce dsigma_bath/dt = tanh(xi) * |p|^2 / (d*T).
When EMA of this signal decays near zero, the system has equilibrated and T
can be lowered. We compare adaptive vs fixed geometric cooling.

Key insight from iteration 1: a single trajectory gets trapped in one mode.
For mode coverage, we need N_traj=30 independent trajectories per method,
as specified. The metric is about KL quality, not single-chain mode hopping.
"""

import numpy as np
from multiprocessing import Pool
import json
import os
import time

# ============================================================
# Potentials
# ============================================================

def gaussian_mixture_2d(q, sep=4.0, sigma=0.5):
    """4-mode Gaussian mixture. Returns (U, grad_U)."""
    means = np.array([[sep, sep], [-sep, sep], [-sep, -sep], [sep, -sep]])
    inv_var = 1.0 / (sigma ** 2)
    diffs = q[None, :] - means  # (4, 2)
    exponents = -0.5 * inv_var * np.sum(diffs ** 2, axis=1)
    max_exp = np.max(exponents)
    weights = np.exp(exponents - max_exp)
    Z = np.sum(weights)
    U = -(max_exp + np.log(Z)) + np.log(4.0)
    grad_U = inv_var * np.sum(weights[:, None] * diffs, axis=0) / Z
    return U, grad_U


def double_well_1d(q):
    """V(x) = (x^2 - 1)^2."""
    x = q[0]
    U = (x ** 2 - 1.0) ** 2
    grad_U = np.array([4.0 * x * (x ** 2 - 1.0)])
    return U, grad_U


# ============================================================
# Integrators
# ============================================================

def nh_step(q, p, xi, dt, T, Q, potential_fn):
    """One BAOAB step with g(xi)=tanh(xi) NH thermostat. Returns updated (q, p, xi, kinetic, tanh_xi, force_evals)."""
    U, grad_U = potential_fn(q)
    p = p - 0.5 * dt * grad_U
    q = q + 0.5 * dt * p

    tanh_xi = np.tanh(xi)
    kinetic = np.dot(p, p)
    dim = len(q)

    p = p * np.exp(-0.5 * dt * tanh_xi)
    xi = xi + dt * (kinetic / (dim * T) - 1.0) / Q
    p = p * np.exp(-0.5 * dt * np.tanh(xi))

    q = q + 0.5 * dt * p
    U, grad_U = potential_fn(q)
    p = p - 0.5 * dt * grad_U

    return q, p, xi, kinetic, tanh_xi, 2


def run_nh_adaptive_single(potential_fn, dim, T_start, T_final, dt,
                           max_anneal_steps, alpha, ema_tau, eps_threshold,
                           post_anneal_steps, Q, rng):
    """Single trajectory: NH adaptive annealing."""
    q = rng.randn(dim) * 1.0
    p = rng.randn(dim) * np.sqrt(T_start)
    xi = 0.0
    T = T_start

    ema_signal = 1.0
    ema_alpha_val = 1.0 / ema_tau

    T_history = [T]
    force_evals = 0

    step = 0
    while T > T_final + 1e-8 and step < max_anneal_steps:
        q, p, xi, kinetic, tanh_xi, fe = nh_step(q, p, xi, dt, T, Q, potential_fn)
        force_evals += fe

        dsigma = tanh_xi * kinetic / (dim * T)
        ema_signal = (1 - ema_alpha_val) * ema_signal + ema_alpha_val * abs(dsigma)

        if ema_signal < eps_threshold:
            T_new = max(T * alpha, T_final)
            if T_new < T:
                T = T_new
                kinetic_now = np.dot(p, p) / dim
                if kinetic_now > 1e-10:
                    p *= np.sqrt(T / max(kinetic_now, 1e-10))
                ema_signal = 1.0

        T_history.append(T)
        step += 1

    anneal_steps = step
    anneal_fe = force_evals

    # Production at T_final
    T = T_final
    samples = []
    for _ in range(post_anneal_steps):
        q, p, xi, _, _, fe = nh_step(q, p, xi, dt, T, Q, potential_fn)
        force_evals += fe
        samples.append(q.copy())

    return np.array(samples), T_history, force_evals, anneal_steps, anneal_fe


def run_nh_fixed_single(potential_fn, dim, T_start, T_final, dt,
                        total_anneal_steps, alpha, post_anneal_steps, Q, rng):
    """Single trajectory: NH fixed geometric cooling."""
    q = rng.randn(dim) * 1.0
    p = rng.randn(dim) * np.sqrt(T_start)
    xi = 0.0
    T = T_start

    n_stages = int(np.ceil(np.log(T_final / T_start) / np.log(alpha)))
    N_stage = max(1, total_anneal_steps // n_stages)

    T_history = [T]
    force_evals = 0

    for step in range(total_anneal_steps):
        q, p, xi, _, _, fe = nh_step(q, p, xi, dt, T, Q, potential_fn)
        force_evals += fe

        if (step + 1) % N_stage == 0 and T > T_final + 1e-8:
            T_new = max(T * alpha, T_final)
            if T_new < T:
                T = T_new
                kinetic_now = np.dot(p, p) / dim
                if kinetic_now > 1e-10:
                    p *= np.sqrt(T / max(kinetic_now, 1e-10))

        T_history.append(T)

    T = T_final
    samples = []
    for _ in range(post_anneal_steps):
        q, p, xi, _, _, fe = nh_step(q, p, xi, dt, T, Q, potential_fn)
        force_evals += fe
        samples.append(q.copy())

    return np.array(samples), T_history, force_evals, total_anneal_steps, force_evals


def run_langevin_fixed_single(potential_fn, dim, T_start, T_final, dt,
                              total_anneal_steps, alpha, post_anneal_steps, rng):
    """Single trajectory: overdamped Langevin with fixed geometric cooling."""
    q = rng.randn(dim) * 1.0
    T = T_start

    n_stages = int(np.ceil(np.log(T_final / T_start) / np.log(alpha)))
    N_stage = max(1, total_anneal_steps // n_stages)

    T_history = [T]
    force_evals = 0

    for step in range(total_anneal_steps):
        _, grad_U = potential_fn(q)
        force_evals += 1
        noise = rng.randn(dim)
        q = q - grad_U / T * dt + np.sqrt(2.0 / T * dt) * noise

        if (step + 1) % N_stage == 0 and T > T_final + 1e-8:
            T = max(T * alpha, T_final)
        T_history.append(T)

    T = T_final
    samples = []
    for _ in range(post_anneal_steps):
        _, grad_U = potential_fn(q)
        force_evals += 1
        noise = rng.randn(dim)
        q = q - grad_U / T * dt + np.sqrt(2.0 / T * dt) * noise
        samples.append(q.copy())

    return np.array(samples), T_history, force_evals


# ============================================================
# Metrics
# ============================================================

def kl_2d_histogram(samples, target_logpdf_fn, n_bins=50, bounds=(-8, 8)):
    edges = np.linspace(bounds[0], bounds[1], n_bins + 1)
    H, _, _ = np.histogram2d(samples[:, 0], samples[:, 1], bins=[edges, edges])
    H = H / (H.sum() + 1e-30)

    cx = 0.5 * (edges[:-1] + edges[1:])
    X, Y = np.meshgrid(cx, cx, indexing='ij')
    grid = np.stack([X, Y], axis=-1)

    log_target = target_logpdf_fn(grid)
    target = np.exp(log_target)
    target = target / (target.sum() + 1e-30)

    mask = (H > 0) & (target > 0)
    if mask.sum() == 0:
        return 10.0
    kl = np.sum(H[mask] * np.log(H[mask] / target[mask]))
    return max(kl, 0.0)


def kl_1d_histogram(samples_1d, target_logpdf_fn, n_bins=100, bounds=(-3, 3)):
    edges = np.linspace(bounds[0], bounds[1], n_bins + 1)
    H, _ = np.histogram(samples_1d, bins=edges)
    H = H.astype(float) / (H.sum() + 1e-30)

    cx = 0.5 * (edges[:-1] + edges[1:])
    log_target = target_logpdf_fn(cx)
    target = np.exp(log_target)
    target = target / (target.sum() + 1e-30)

    mask = (H > 0) & (target > 0)
    if mask.sum() == 0:
        return 10.0
    kl = np.sum(H[mask] * np.log(H[mask] / target[mask]))
    return max(kl, 0.0)


def mode_coverage_2d(samples, sep=4.0, radius=2.0):
    """Fraction of 4 modes with >5% of nearby samples."""
    means = np.array([[sep, sep], [-sep, sep], [-sep, -sep], [sep, -sep]])
    counts = np.zeros(4)
    for i, m in enumerate(means):
        dists = np.sqrt(np.sum((samples - m[None, :]) ** 2, axis=1))
        counts[i] = np.sum(dists < radius)
    total_near = counts.sum()
    if total_near == 0:
        return 0.0
    fracs = counts / total_near
    return np.mean(fracs > 0.05)


# ============================================================
# Target log-pdfs
# ============================================================

def gmm_logpdf(grid, sep=4.0, sigma=0.5):
    means = np.array([[sep, sep], [-sep, sep], [-sep, -sep], [sep, -sep]])
    shape = grid.shape[:-1]
    flat = grid.reshape(-1, 2)
    inv_var = 1.0 / (sigma ** 2)
    diffs = flat[:, None, :] - means[None, :, :]
    exponents = -0.5 * inv_var * np.sum(diffs ** 2, axis=2)
    max_exp = np.max(exponents, axis=1, keepdims=True)
    log_sum = max_exp[:, 0] + np.log(np.sum(np.exp(exponents - max_exp), axis=1))
    return log_sum.reshape(shape)


def dw_logpdf(x):
    return -(x ** 2 - 1.0) ** 2


# ============================================================
# Per-seed runner (parallelized)
# ============================================================

def run_single_seed(args):
    """Run all methods x all targets for one seed, using N_traj=30 trajectories."""
    seed, sep_values = args
    t0 = time.time()
    rng = np.random.RandomState(seed)

    dt = 0.01
    T_start = 5.0
    T_final = 1.0
    alpha = 0.95       # gentler cooling
    max_anneal_steps = 15000
    post_anneal_steps = 8000   # more production samples per traj
    Q = 1.0
    ema_tau = 50
    eps_threshold = 0.05
    N_traj = 30

    results = {}

    for sep in sep_values:
        pot_fn = lambda q, s=sep: gaussian_mixture_2d(q, sep=s)
        target_fn = lambda g, s=sep: gmm_logpdf(g, sep=s)
        bounds = (-sep - 3, sep + 3)

        # Collect samples from N_traj independent trajectories
        for method_name in ['nh_adaptive', 'nh_fixed', 'langevin_fixed']:
            all_samples = []
            total_fe = 0
            all_anneal_steps = []
            T_hist_example = None

            for traj_i in range(N_traj):
                traj_rng = np.random.RandomState(rng.randint(0, 2**31))

                if method_name == 'nh_adaptive':
                    samps, T_hist, fe, a_steps, _ = run_nh_adaptive_single(
                        pot_fn, dim=2, T_start=T_start, T_final=T_final, dt=dt,
                        max_anneal_steps=max_anneal_steps, alpha=alpha,
                        ema_tau=ema_tau, eps_threshold=eps_threshold,
                        post_anneal_steps=post_anneal_steps, Q=Q, rng=traj_rng
                    )
                    all_anneal_steps.append(a_steps)
                    if traj_i == 0:
                        T_hist_example = T_hist

                elif method_name == 'nh_fixed':
                    # Use median adaptive anneal steps as budget
                    if f'nh_adaptive_sep{sep}' in results:
                        budget = int(results[f'nh_adaptive_sep{sep}']['median_anneal_steps'])
                    else:
                        budget = max_anneal_steps // 2
                    samps, T_hist, fe, _, _ = run_nh_fixed_single(
                        pot_fn, dim=2, T_start=T_start, T_final=T_final, dt=dt,
                        total_anneal_steps=budget, alpha=alpha,
                        post_anneal_steps=post_anneal_steps, Q=Q, rng=traj_rng
                    )
                    if traj_i == 0:
                        T_hist_example = T_hist

                elif method_name == 'langevin_fixed':
                    if f'nh_adaptive_sep{sep}' in results:
                        budget = int(results[f'nh_adaptive_sep{sep}']['median_anneal_steps'])
                    else:
                        budget = max_anneal_steps // 2
                    samps, T_hist, fe = run_langevin_fixed_single(
                        pot_fn, dim=2, T_start=T_start, T_final=T_final, dt=dt,
                        total_anneal_steps=budget, alpha=alpha,
                        post_anneal_steps=post_anneal_steps, rng=traj_rng
                    )
                    if traj_i == 0:
                        T_hist_example = T_hist

                all_samples.append(samps)
                total_fe += fe

            all_samples = np.concatenate(all_samples, axis=0)
            kl = kl_2d_histogram(all_samples, target_fn, bounds=(bounds[0], bounds[1]))
            mc = mode_coverage_2d(all_samples, sep=sep)

            entry = {
                'kl': float(kl),
                'mode_coverage': float(mc),
                'force_evals': total_fe,
                'n_samples': len(all_samples),
            }
            if all_anneal_steps:
                entry['median_anneal_steps'] = int(np.median(all_anneal_steps))
                entry['mean_anneal_steps'] = float(np.mean(all_anneal_steps))
            if T_hist_example is not None and seed == 42:
                entry['T_history'] = [float(t) for t in T_hist_example[:3000]]

            results[f'{method_name}_sep{sep}'] = entry

    # 1D double well
    pot_fn_1d = lambda q: double_well_1d(q)
    target_fn_1d = lambda x: dw_logpdf(x)

    for method_name in ['nh_adaptive', 'nh_fixed', 'langevin_fixed']:
        all_samples_1d = []
        total_fe = 0

        for traj_i in range(N_traj):
            traj_rng = np.random.RandomState(rng.randint(0, 2**31))

            if method_name == 'nh_adaptive':
                samps, _, fe, a_steps, _ = run_nh_adaptive_single(
                    pot_fn_1d, dim=1, T_start=T_start, T_final=T_final, dt=dt,
                    max_anneal_steps=max_anneal_steps, alpha=alpha,
                    ema_tau=ema_tau, eps_threshold=eps_threshold,
                    post_anneal_steps=post_anneal_steps, Q=Q, rng=traj_rng
                )
            elif method_name == 'nh_fixed':
                budget = max_anneal_steps // 2
                if f'nh_adaptive_dw1d' in results and 'median_anneal_steps' in results[f'nh_adaptive_dw1d']:
                    budget = int(results[f'nh_adaptive_dw1d']['median_anneal_steps'])
                samps, _, fe, _, _ = run_nh_fixed_single(
                    pot_fn_1d, dim=1, T_start=T_start, T_final=T_final, dt=dt,
                    total_anneal_steps=budget, alpha=alpha,
                    post_anneal_steps=post_anneal_steps, Q=Q, rng=traj_rng
                )
            elif method_name == 'langevin_fixed':
                budget = max_anneal_steps // 2
                if f'nh_adaptive_dw1d' in results and 'median_anneal_steps' in results[f'nh_adaptive_dw1d']:
                    budget = int(results[f'nh_adaptive_dw1d']['median_anneal_steps'])
                samps, _, fe = run_langevin_fixed_single(
                    pot_fn_1d, dim=1, T_start=T_start, T_final=T_final, dt=dt,
                    total_anneal_steps=budget, alpha=alpha,
                    post_anneal_steps=post_anneal_steps, rng=traj_rng
                )

            all_samples_1d.append(samps)
            total_fe += fe

        all_samples_1d = np.concatenate(all_samples_1d, axis=0)
        kl = kl_1d_histogram(all_samples_1d[:, 0] if all_samples_1d.ndim > 1 else all_samples_1d, target_fn_1d)

        results[f'{method_name}_dw1d'] = {
            'kl': float(kl),
            'force_evals': total_fe,
            'n_samples': len(all_samples_1d),
        }
        if method_name == 'nh_adaptive' and 'median_anneal_steps' not in results.get(f'nh_adaptive_dw1d', {}):
            pass  # already set

    elapsed = time.time() - t0
    return {'seed': seed, 'results': results, 'wall_time': elapsed}


# ============================================================
# Aggregation
# ============================================================

def aggregate_results(all_seed_results, sep_values):
    methods = ['nh_adaptive', 'nh_fixed', 'langevin_fixed']
    summary = {}

    for method in methods:
        summary[method] = {}
        for sep in sep_values:
            key = f'{method}_sep{sep}'
            kls = [r['results'][key]['kl'] for r in all_seed_results]
            mcs = [r['results'][key]['mode_coverage'] for r in all_seed_results]
            fes = [r['results'][key]['force_evals'] for r in all_seed_results]
            summary[method][f'sep{sep}'] = {
                'kl_mean': float(np.mean(kls)),
                'kl_std': float(np.std(kls)),
                'mc_mean': float(np.mean(mcs)),
                'mc_std': float(np.std(mcs)),
                'fe_mean': float(np.mean(fes)),
            }

        key = f'{method}_dw1d'
        kls = [r['results'][key]['kl'] for r in all_seed_results]
        summary[method]['dw1d'] = {
            'kl_mean': float(np.mean(kls)),
            'kl_std': float(np.std(kls)),
        }

    return summary


# ============================================================
# Plotting
# ============================================================

def make_figures(all_seed_results, summary, sep_values, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.pad_inches': 0.2,
    })

    colors = {
        'nh_adaptive': '#2ca02c',
        'nh_fixed': '#1f77b4',
        'langevin_fixed': '#ff7f0e',
    }
    labels = {
        'nh_adaptive': 'NH-Adaptive',
        'nh_fixed': 'NH-Fixed',
        'langevin_fixed': 'Langevin-Fixed',
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # (a) KL bar chart at sep=4
    ax = axes[0]
    methods = ['nh_adaptive', 'nh_fixed', 'langevin_fixed']
    x_pos = np.arange(len(methods))
    for i, method in enumerate(methods):
        kl_mean = summary[method]['sep4']['kl_mean']
        kl_std = summary[method]['sep4']['kl_std']
        ax.bar(x_pos[i], kl_mean, yerr=kl_std, color=colors[method],
               capsize=5, edgecolor='black', linewidth=0.5, alpha=0.85, width=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([labels[m] for m in methods], rotation=15, ha='right')
    ax.set_ylabel('KL Divergence')
    ax.set_title('(a) KL at sep=4 (matched budget)', fontweight='bold')
    ax.axhline(y=0.01, color='gray', linestyle='--', linewidth=1, label='KL=0.01 target')
    ax.legend(frameon=False, fontsize=10)

    # (b) T(t) schedule examples
    ax = axes[1]
    seed42 = [r for r in all_seed_results if r['seed'] == 42]
    if seed42:
        seed42 = seed42[0]
        for method in ['nh_adaptive', 'nh_fixed']:
            key = f'{method}_sep4'
            T_hist = seed42['results'][key].get('T_history', [])
            if T_hist:
                steps = np.arange(len(T_hist))
                ax.plot(steps, T_hist, color=colors[method], label=labels[method], linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Temperature T')
    ax.set_title('(b) Cooling Schedule (sep=4, seed=42)', fontweight='bold')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend(frameon=False)

    # (c) Mode coverage vs sep
    ax = axes[2]
    for method in methods:
        mc_means = [summary[method][f'sep{s}']['mc_mean'] for s in sep_values]
        mc_stds = [summary[method][f'sep{s}']['mc_std'] for s in sep_values]
        ax.errorbar(sep_values, mc_means, yerr=mc_stds, color=colors[method],
                     label=labels[method], marker='o', linewidth=2, capsize=4)
    ax.set_xlabel('Separation (sep)')
    ax.set_ylabel('Mode Coverage')
    ax.set_title('(c) Mode Coverage vs Separation', fontweight='bold')
    ax.set_ylim(-0.05, 1.15)
    ax.legend(frameon=False)

    fig.savefig(os.path.join(output_dir, 'annealing_comparison.png'), bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(base_dir, 'figures')
    res_dir = os.path.join(base_dir, 'results')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    sep_values = [2, 4, 6]
    seeds = [42, 123, 7]  # 3 seeds minimum

    # Parallel across seeds
    args = [(s, sep_values) for s in seeds]
    with Pool(len(seeds)) as pool:
        all_seed_results = pool.map(run_single_seed, args)

    summary = aggregate_results(all_seed_results, sep_values)

    # Key metric
    adaptive_kl = summary['nh_adaptive']['sep4']['kl_mean']
    fixed_kl = summary['nh_fixed']['sep4']['kl_mean']
    metric = fixed_kl / adaptive_kl if adaptive_kl > 1e-8 else 1.0

    print(f"\n=== KEY METRIC (fixed_KL / adaptive_KL at sep=4) = {metric:.4f} ===")
    print(f"  NH-Adaptive KL: {adaptive_kl:.4f} +/- {summary['nh_adaptive']['sep4']['kl_std']:.4f}")
    print(f"  NH-Fixed KL:    {fixed_kl:.4f} +/- {summary['nh_fixed']['sep4']['kl_std']:.4f}")
    print(f"  Langevin KL:    {summary['langevin_fixed']['sep4']['kl_mean']:.4f} +/- {summary['langevin_fixed']['sep4']['kl_std']:.4f}")

    print("\nPer-seed results (sep=4):")
    print("| Seed | NH-Adapt KL | NH-Fixed KL | Langevin KL | Ratio | Time |")
    print("|------|-------------|-------------|-------------|-------|------|")
    for r in all_seed_results:
        s = r['seed']
        akl = r['results']['nh_adaptive_sep4']['kl']
        fkl = r['results']['nh_fixed_sep4']['kl']
        lkl = r['results']['langevin_fixed_sep4']['kl']
        ratio = fkl / akl if akl > 1e-8 else 1.0
        wt = r.get('wall_time', 0)
        print(f"| {s:4d} | {akl:.4f}      | {fkl:.4f}      | {lkl:.4f}      | {ratio:.3f} | {wt:.1f}s |")

    print("\nMode coverage (mean +/- std across seeds):")
    for sep in sep_values:
        print(f"  sep={sep}:")
        for method in ['nh_adaptive', 'nh_fixed', 'langevin_fixed']:
            mc_m = summary[method][f'sep{sep}']['mc_mean']
            mc_s = summary[method][f'sep{sep}']['mc_std']
            print(f"    {method}: {mc_m:.3f} +/- {mc_s:.3f}")

    print("\n1D Double Well KL:")
    for method in ['nh_adaptive', 'nh_fixed', 'langevin_fixed']:
        kl = summary[method]['dw1d']['kl_mean']
        std = summary[method]['dw1d']['kl_std']
        print(f"  {method}: {kl:.4f} +/- {std:.4f}")

    # Save
    with open(os.path.join(res_dir, 'summary.json'), 'w') as f:
        json.dump({'summary': summary, 'metric': metric, 'seeds': seeds}, f, indent=2)

    seed42 = [r for r in all_seed_results if r['seed'] == 42]
    if seed42:
        seed42 = seed42[0]
        schedules = {}
        for method in ['nh_adaptive', 'nh_fixed']:
            key = f'{method}_sep4'
            schedules[method] = seed42['results'][key].get('T_history', [])
        with open(os.path.join(res_dir, 'schedule_examples.json'), 'w') as f:
            json.dump(schedules, f)

    make_figures(all_seed_results, summary, sep_values, fig_dir)
    print(f"\nFigure saved to {fig_dir}/annealing_comparison.png")

    return metric, summary


if __name__ == '__main__':
    metric, summary = main()
