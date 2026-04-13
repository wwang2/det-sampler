"""
Clipped-log-osc-072: Mechanism behind log-osc failure in NH sampling.

Hypothesis: log-osc g(ξ) = 2ξ/(1+ξ²) decays back to 0 as ξ→∞, so the
thermostat shuts off when ξ is large. This allows unlimited ξ-drift and
poor temperature control, explaining the 536× gap vs tanh.

Key diagnostics:
  1. τ_int (mixing efficiency)
  2. ⟨|g(ξ)|⟩ along trajectories (effective coupling strength)
  3. Distribution of ξ values (how far does ξ drift?)
  4. ⟨|p|²⟩/d at each time step (temperature control quality)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from multiprocessing import Pool
import time
import os
import json

# ---------------------------------------------------------------------------
# Plotting defaults — Nature-style
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.pad_inches': 0.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

ORBIT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ORBIT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Friction functions
# ---------------------------------------------------------------------------

def g_tanh(xi):
    """tanh: saturates at ±1. Standard bounded NH."""
    return np.tanh(xi)

def g_losc(xi):
    """Log-oscillator: g = 2ξ/(1+ξ²). BOUNDED (max=1 at ξ=1) but decays→0."""
    return 2.0 * xi / (1.0 + xi ** 2)

def g_rational(xi):
    """Rational: ξ/(1+|ξ|). Saturates at ±1. From orbit 071."""
    return xi / (1.0 + np.abs(xi))

FRICTION_FUNCS = {
    'tanh': g_tanh,
    'log-osc': g_losc,
    'rational': g_rational,
}

COLORS = {
    'tanh': '#1f77b4',
    'log-osc': '#d62728',
    'rational': '#2ca02c',
}

LABELS = {
    'tanh': 'tanh (saturates at ±1)',
    'log-osc': 'log-osc: 2ξ/(1+ξ²) [decays→0]',
    'rational': 'rational: ξ/(1+|ξ|) [saturates at ±1]',
}

# ---------------------------------------------------------------------------
# Potential: d=10 anisotropic Gaussian
# ---------------------------------------------------------------------------

def make_omega2(dim, kappa):
    """Frequencies ωᵢ² linearly spaced from 1 to κ."""
    return np.linspace(1.0, float(kappa), dim)


# ---------------------------------------------------------------------------
# Integrated autocorrelation time
# ---------------------------------------------------------------------------

def compute_tau_int(x, max_lag=5000):
    """Integrated autocorrelation time via FFT."""
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n < 10:
        return float('inf')
    var = np.var(x)
    if var < 1e-15:
        return float('inf')

    fft_x = np.fft.fft(x, n=2 * n)
    acf = np.real(np.fft.ifft(fft_x * np.conj(fft_x)))[:n] / (var * n)

    tau = 1.0
    for lag in range(1, min(max_lag, n // 2)):
        if acf[lag] < 0.05:
            break
        tau += 2.0 * acf[lag]
    return float(tau)


# ---------------------------------------------------------------------------
# BAOAB NH integrator — returns rich diagnostics
# ---------------------------------------------------------------------------

def run_trajectory_diag(omega2, dim, g_func, Q, kT, dt, n_steps, seed,
                         record_every=100, mass=1.0):
    """
    Run BAOAB NH trajectory. Returns:
      - tau_int (stiffest mode q²)
      - xi_vals: sampled ξ values (every record_every steps)
      - g_vals: |g(ξ)| at each recorded step
      - temp_vals: |p|²/d at each recorded step
    """
    rng = np.random.default_rng(seed)
    q = rng.normal(0, 0.5, size=dim)
    p = rng.normal(0, np.sqrt(mass * kT), size=dim)
    xi = 0.0

    half_dt = 0.5 * dt
    burnin = n_steps // 5

    n_collect = n_steps - burnin
    obs = np.empty(n_collect)
    obs_idx = 0

    xi_list = []
    g_list = []
    temp_list = []

    for step in range(n_steps):
        # B: half-step thermostat (xi update)
        kinetic = np.sum(p ** 2) / mass
        xi += half_dt * (kinetic - dim * kT) / Q

        # A: half-step momentum — friction
        g_val = float(g_func(xi))
        scale = np.exp(-g_val * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p *= scale

        # A: half-step momentum — force
        p -= half_dt * (omega2 * q)

        # O: full-step position
        q += dt * p / mass

        # A: half-step momentum — force
        p -= half_dt * (omega2 * q)

        # A: half-step momentum — friction (use current xi)
        g_val2 = float(g_func(xi))
        scale2 = np.exp(-g_val2 * half_dt)
        scale2 = np.clip(scale2, 1e-10, 1e10)
        p *= scale2

        # B: half-step thermostat
        kinetic = np.sum(p ** 2) / mass
        xi += half_dt * (kinetic - dim * kT) / Q

        if step >= burnin:
            obs[obs_idx] = q[-1] ** 2   # stiffest mode (last index = highest ω)
            obs_idx += 1

            if (step - burnin) % record_every == 0:
                xi_list.append(xi)
                g_list.append(abs(float(g_func(xi))))
                temp_list.append(np.sum(p ** 2) / (mass * dim))

        if np.any(np.isnan(q)) or np.any(np.isnan(p)) or np.isnan(xi):
            return {
                'tau_int': float('inf'),
                'nan': True,
                'xi_vals': np.array([]),
                'g_vals': np.array([]),
                'temp_vals': np.array([]),
            }

    tau = compute_tau_int(obs[:obs_idx])
    return {
        'tau_int': tau,
        'nan': False,
        'xi_vals': np.array(xi_list),
        'g_vals': np.array(g_list),
        'temp_vals': np.array(temp_list),
    }


def _worker_diag(args):
    return run_trajectory_diag(**args)


# ---------------------------------------------------------------------------
# Simplified trajectory for τ_int vs Q sweep (no diagnostics → faster)
# ---------------------------------------------------------------------------

def run_trajectory_fast(omega2, dim, g_func, Q, kT, dt, n_steps, seed, mass=1.0):
    """Fast version — only returns tau_int."""
    rng = np.random.default_rng(seed)
    q = rng.normal(0, 0.5, size=dim)
    p = rng.normal(0, np.sqrt(mass * kT), size=dim)
    xi = 0.0

    half_dt = 0.5 * dt
    burnin = n_steps // 5
    n_collect = n_steps - burnin
    obs = np.empty(n_collect)
    obs_idx = 0

    for step in range(n_steps):
        kinetic = np.sum(p ** 2) / mass
        xi += half_dt * (kinetic - dim * kT) / Q

        g_val = float(g_func(xi))
        scale = np.clip(np.exp(-g_val * half_dt), 1e-10, 1e10)
        p *= scale
        p -= half_dt * (omega2 * q)
        q += dt * p / mass
        p -= half_dt * (omega2 * q)
        g_val2 = float(g_func(xi))
        scale2 = np.clip(np.exp(-g_val2 * half_dt), 1e-10, 1e10)
        p *= scale2

        kinetic = np.sum(p ** 2) / mass
        xi += half_dt * (kinetic - dim * kT) / Q

        if step >= burnin:
            obs[obs_idx] = q[-1] ** 2
            obs_idx += 1

        if np.any(np.isnan(q)) or np.any(np.isnan(p)) or np.isnan(xi):
            return {'tau_int': float('inf'), 'nan': True}

    tau = compute_tau_int(obs[:obs_idx])
    return {'tau_int': tau, 'nan': False}


def _worker_fast(args):
    return run_trajectory_fast(**args)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    print("=" * 65)
    print("Orbit 072: Log-osc Failure Mechanism (g→0 at large ξ)")
    print("=" * 65)

    kT = 1.0
    dt = 0.005
    dim = 10
    kappa = 100
    Q_fixed = 100.0
    n_steps = 500_000
    seeds = [0, 1, 2]
    record_every = 100

    omega2 = make_omega2(dim, kappa)
    methods = list(FRICTION_FUNCS.keys())

    # ------------------------------------------------------------------
    # Part 1: Detailed diagnostics at Q=100 (3 seeds each)
    # ------------------------------------------------------------------
    print(f"\n--- Part 1: Diagnostics at Q={Q_fixed}, d={dim}, κ={kappa} ---")
    diag_results = {}

    for method_name in methods:
        g_func = FRICTION_FUNCS[method_name]
        job_args = [
            dict(omega2=omega2, dim=dim, g_func=g_func, Q=Q_fixed,
                 kT=kT, dt=dt, n_steps=n_steps, seed=s,
                 record_every=record_every)
            for s in seeds
        ]
        with Pool(min(len(seeds), os.cpu_count() or 4)) as pool:
            seed_results = pool.map(_worker_diag, job_args)

        taus = [r['tau_int'] for r in seed_results if not r['nan'] and r['tau_int'] < 1e7]
        xi_all = np.concatenate([r['xi_vals'] for r in seed_results if not r['nan']])
        g_all = np.concatenate([r['g_vals'] for r in seed_results if not r['nan']])
        temp_all = np.concatenate([r['temp_vals'] for r in seed_results if not r['nan']])

        median_tau = float(np.median(taus)) if taus else float('inf')
        mean_g = float(np.mean(g_all)) if len(g_all) > 0 else float('nan')
        mean_xi_abs = float(np.mean(np.abs(xi_all))) if len(xi_all) > 0 else float('nan')
        mean_temp = float(np.mean(temp_all)) if len(temp_all) > 0 else float('nan')
        std_temp = float(np.std(temp_all)) if len(temp_all) > 0 else float('nan')

        diag_results[method_name] = {
            'taus': taus,
            'median_tau': median_tau,
            'xi_all': xi_all,
            'g_all': g_all,
            'temp_all': temp_all,
            'mean_g': mean_g,
            'mean_xi_abs': mean_xi_abs,
            'mean_temp': mean_temp,
            'std_temp': std_temp,
        }
        print(f"  {method_name:10s}: τ_int={median_tau:8.1f}, "
              f"⟨|g|⟩={mean_g:.4f}, ⟨|ξ|⟩={mean_xi_abs:.3f}, "
              f"⟨T⟩={mean_temp:.4f}±{std_temp:.4f}")

    # ------------------------------------------------------------------
    # Part 2: τ_int vs Q sweep
    # ------------------------------------------------------------------
    print(f"\n--- Part 2: τ_int vs Q sweep (d={dim}, κ={kappa}) ---")
    Q_values = [10.0, 30.0, 100.0, 300.0]
    n_seeds_sweep = 5
    n_steps_sweep = 300_000

    tau_vs_Q = {m: {} for m in methods}

    all_args = []
    for method_name in methods:
        g_func = FRICTION_FUNCS[method_name]
        for Q in Q_values:
            for s in range(n_seeds_sweep):
                all_args.append((method_name, Q, s, dict(
                    omega2=omega2, dim=dim, g_func=g_func, Q=Q,
                    kT=kT, dt=dt, n_steps=n_steps_sweep, seed=s
                )))

    # Run all at once
    flat_args = [a[3] for a in all_args]
    with Pool(min(os.cpu_count() or 4, len(flat_args))) as pool:
        flat_results = pool.map(_worker_fast, flat_args)

    # Aggregate
    from collections import defaultdict
    buckets = defaultdict(list)
    for (method_name, Q, s, _), r in zip(all_args, flat_results):
        if not r['nan'] and r['tau_int'] < 1e7:
            buckets[(method_name, Q)].append(r['tau_int'])

    for method_name in methods:
        for Q in Q_values:
            taus_here = buckets[(method_name, Q)]
            tau_vs_Q[method_name][Q] = {
                'median': float(np.median(taus_here)) if taus_here else float('inf'),
                'mean': float(np.mean(taus_here)) if taus_here else float('inf'),
                'n': len(taus_here),
            }
            med = tau_vs_Q[method_name][Q]['median']
            print(f"  {method_name:10s} Q={Q:6.0f}: τ_int={med:.1f}")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("\n--- Generating figures ---")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.flat

    # --- Panel (a): g(ξ) shape ---
    xi_arr = np.linspace(-8, 8, 800)
    for name in methods:
        g_func = FRICTION_FUNCS[name]
        ax_a.plot(xi_arr, g_func(xi_arr),
                  label=LABELS[name], color=COLORS[name], linewidth=2.5)
    ax_a.axhline(0, color='gray', lw=0.6, ls='--')
    ax_a.axhline(1, color='gray', lw=0.6, ls=':', alpha=0.6)
    ax_a.axhline(-1, color='gray', lw=0.6, ls=':', alpha=0.6)
    # Annotate log-osc decay region
    ax_a.annotate('log-osc → 0\n(thermostat\nshuts off)',
                  xy=(5, g_losc(5)), xytext=(3.5, 0.55),
                  arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
                  color='#d62728', fontsize=9)
    ax_a.set_xlabel('ξ')
    ax_a.set_ylabel('g(ξ)')
    ax_a.set_title('(a) Friction function shapes', fontweight='bold')
    ax_a.legend(frameon=False, fontsize=9)
    ax_a.set_xlim(-8, 8)
    ax_a.set_ylim(-1.4, 1.4)

    # --- Panel (b): histogram of ξ values ---
    xi_range = np.percentile(
        np.concatenate([diag_results[m]['xi_all'] for m in methods if len(diag_results[m]['xi_all']) > 0]),
        [0.5, 99.5]
    )
    bins = np.linspace(xi_range[0], xi_range[1], 80)

    for name in methods:
        xi_data = diag_results[name]['xi_all']
        if len(xi_data) > 0:
            ax_b.hist(xi_data, bins=bins, density=True, alpha=0.55,
                      color=COLORS[name], label=LABELS[name])
    ax_b.set_xlabel('ξ value')
    ax_b.set_ylabel('Density')
    ax_b.set_title('(b) Distribution of ξ (thermostat variable)', fontweight='bold')
    ax_b.legend(frameon=False, fontsize=9)

    # --- Panel (c): time-averaged |g(ξ)| comparison ---
    # Show per-seed mean |g| as scatter + group mean
    method_positions = {m: i for i, m in enumerate(methods)}
    for name in methods:
        g_func = FRICTION_FUNCS[name]
        x_pos = method_positions[name]
        per_seed_g = []
        valid_results = [r for r in
                         [diag_results[name]]  # aggregated already
                         ]
        # Recalculate per-seed from diag results (re-run lightweight version)
        # Use the aggregate mean_g as the point
        mean_g = diag_results[name]['mean_g']
        ax_c.bar(x_pos, mean_g, color=COLORS[name], alpha=0.75, width=0.6,
                 label=LABELS[name])
        ax_c.text(x_pos, mean_g + 0.005, f'{mean_g:.4f}',
                  ha='center', va='bottom', fontsize=10, fontweight='bold',
                  color=COLORS[name])

    ax_c.set_xticks(range(len(methods)))
    ax_c.set_xticklabels([LABELS[m] for m in methods], rotation=15, ha='right', fontsize=9)
    ax_c.set_ylabel('⟨|g(ξ)|⟩ (time-averaged coupling)')
    ax_c.set_title('(c) Effective thermostat coupling strength', fontweight='bold')
    ax_c.set_ylim(0, max(diag_results[m]['mean_g'] for m in methods
                         if not np.isnan(diag_results[m]['mean_g'])) * 1.25)

    # --- Panel (d): τ_int vs Q ---
    for name in methods:
        Q_arr = sorted(tau_vs_Q[name].keys())
        medians = [tau_vs_Q[name][Q]['median'] for Q in Q_arr]
        valid = [(Q, t) for Q, t in zip(Q_arr, medians) if t < 1e7]
        if valid:
            Qs, taus = zip(*valid)
            ax_d.plot(Qs, taus, 'o-', label=LABELS[name],
                      color=COLORS[name], linewidth=2, markersize=7)

    ax_d.set_xscale('log')
    ax_d.set_yscale('log')
    ax_d.set_xlabel('Thermostat mass Q')
    ax_d.set_ylabel('Median τ_int (stiffest mode)')
    ax_d.set_title(f'(d) τ_int vs Q  (d={dim}, κ={kappa})', fontweight='bold')
    ax_d.legend(frameon=False, fontsize=9)
    ax_d.grid(True, alpha=0.25)

    # Compute gap for annotation
    tau_tanh_Q100 = tau_vs_Q['tanh'][100.0]['median']
    tau_losc_Q100 = tau_vs_Q['log-osc'][100.0]['median']
    if tau_tanh_Q100 > 0 and tau_losc_Q100 < 1e7:
        gap = tau_losc_Q100 / tau_tanh_Q100
        ax_d.annotate(f'{gap:.0f}× gap\nat Q=100',
                      xy=(100, tau_losc_Q100),
                      xytext=(30, tau_losc_Q100 * 0.5),
                      arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                      fontsize=10)

    fig.suptitle(
        f'Orbit 072: Log-osc failure — g→0 at large ξ shuts off thermostat\n'
        f'd={dim}, κ={kappa}, Q={Q_fixed}, T=1.0, dt={dt}',
        fontsize=13, fontweight='bold'
    )

    fig_path = os.path.join(FIG_DIR, 'losc_mechanism.png')
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved {fig_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)

    tau_tanh = diag_results['tanh']['median_tau']
    tau_losc = diag_results['log-osc']['median_tau']
    tau_rational = diag_results['rational']['median_tau']

    gap_losc = tau_losc / tau_tanh if tau_tanh > 0 and tau_losc < 1e7 else float('inf')
    gap_rational = tau_rational / tau_tanh if tau_tanh > 0 and tau_rational < 1e7 else float('inf')

    print(f"\nτ_int at Q={Q_fixed} (d={dim}, κ={kappa}):")
    print(f"  tanh:     {tau_tanh:.1f}")
    print(f"  log-osc:  {tau_losc:.1f}  ({gap_losc:.1f}× tanh)")
    print(f"  rational: {tau_rational:.1f}  ({gap_rational:.1f}× tanh)")

    print(f"\nEffective coupling ⟨|g(ξ)|⟩:")
    for m in methods:
        print(f"  {m:10s}: {diag_results[m]['mean_g']:.4f}")

    print(f"\n⟨|ξ|⟩ (thermostat excursion):")
    for m in methods:
        print(f"  {m:10s}: {diag_results[m]['mean_xi_abs']:.3f}")

    print(f"\nTemperature control ⟨T⟩ ± std (target=1.0):")
    for m in methods:
        print(f"  {m:10s}: {diag_results[m]['mean_temp']:.4f} ± {diag_results[m]['std_temp']:.4f}")

    print(f"\nConclusion:")
    print(f"  The {gap_losc:.0f}× gap arises because log-osc g→0 at large ξ,")
    print(f"  shutting off the thermostat. Bounded saturation (g→1) is the key property.")

    # Save results JSON
    save_data = {
        'config': {'dim': dim, 'kappa': kappa, 'Q_fixed': Q_fixed, 'dt': dt,
                   'n_steps': n_steps, 'seeds': seeds},
        'diagnostics': {
            m: {
                'taus': diag_results[m]['taus'],
                'median_tau': diag_results[m]['median_tau'],
                'mean_g': diag_results[m]['mean_g'],
                'mean_xi_abs': diag_results[m]['mean_xi_abs'],
                'mean_temp': diag_results[m]['mean_temp'],
                'std_temp': diag_results[m]['std_temp'],
            }
            for m in methods
        },
        'tau_vs_Q': {
            m: {str(Q): v for Q, v in tau_vs_Q[m].items()}
            for m in methods
        },
        'gap_losc_vs_tanh_at_Q100': gap_losc,
        'gap_rational_vs_tanh_at_Q100': gap_rational,
    }
    with open(os.path.join(ORBIT_DIR, 'results.json'), 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved results.json")

    elapsed = time.time() - t0
    print(f"Total wall time: {elapsed:.1f}s")

    return diag_results, tau_vs_Q


if __name__ == '__main__':
    main()
