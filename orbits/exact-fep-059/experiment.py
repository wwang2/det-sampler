"""
Exact Free Energy Perturbation with NH-CNF
==========================================

Experiments testing whether NH-CNF's exact log-density tracking
provides any advantage for free energy perturbation calculations.

E1: Double-well DeltaF with exact weights vs TI vs Langevin FEP
E2: Multi-well free energy landscape (2D)
E3: Variance scaling with perturbation size (harmonic)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from scipy import integrate, special

# --- Global plot defaults ---
mpl.rcParams.update({
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

SEED = 42
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# Colors
C_NHCNF = '#1f77b4'   # blue - NH-CNF exact FEP
C_TI = '#ff7f0e'       # orange - thermodynamic integration
C_LANG = '#2ca02c'     # green - Langevin FEP
C_TRUE = '#d62728'     # red - ground truth


# =============================================================================
# NH-tanh RK4 integrator (from parent orbit nh-cnf-deep-057)
# =============================================================================

def nh_tanh_rk4_step(q, p, xi, grad_V_fn, dt, Q=1.0, kT=1.0, d=1):
    """One RK4 step of the NH-tanh ODE. Pure numpy."""
    def f(q_, p_, xi_):
        gv = grad_V_fn(q_)
        g = np.tanh(xi_)
        dq = p_.copy()
        dp = -gv - g * p_
        dxi = (1.0 / Q) * (np.sum(p_**2) - d * kT)
        return dq, dp, dxi

    k1q, k1p, k1x = f(q, p, xi)
    k2q, k2p, k2x = f(q + 0.5*dt*k1q, p + 0.5*dt*k1p, xi + 0.5*dt*k1x)
    k3q, k3p, k3x = f(q + 0.5*dt*k2q, p + 0.5*dt*k2p, xi + 0.5*dt*k2x)
    k4q, k4p, k4x = f(q + dt*k3q, p + dt*k3p, xi + dt*k3x)

    q_new = q + (dt/6.0) * (k1q + 2*k2q + 2*k3q + k4q)
    p_new = p + (dt/6.0) * (k1p + 2*k2p + 2*k3p + k4p)
    xi_new = xi + (dt/6.0) * (k1x + 2*k2x + 2*k3x + k4x)

    # Divergence integral (trapezoidal) for density tracking
    g_start = np.tanh(xi)
    g_end = np.tanh(xi_new)
    div_integral = -d * 0.5 * (g_start + g_end) * dt

    return q_new, p_new, xi_new, div_integral


# =============================================================================
# Potentials for E1: Asymmetric double-well with restraints
# =============================================================================

def V_doublewell(x):
    """Asymmetric double-well: V(x) = (x^2 - 1)^2 + 0.3*x"""
    return (x**2 - 1)**2 + 0.3 * x

def grad_V_doublewell(x):
    return 4*x*(x**2 - 1) + 0.3

def V_A(x, k_restraint=5.0):
    """State A: double-well + harmonic restraint near x=-1"""
    return V_doublewell(x) + 0.5 * k_restraint * (x + 1.0)**2

def V_B(x, k_restraint=5.0):
    """State B: double-well + harmonic restraint near x=+1"""
    return V_doublewell(x) + 0.5 * k_restraint * (x - 1.0)**2

def grad_V_A(x, k_restraint=5.0):
    return grad_V_doublewell(x) + k_restraint * (x + 1.0)

def grad_V_B(x, k_restraint=5.0):
    return grad_V_doublewell(x) + k_restraint * (x - 1.0)

def grad_V_lambda(x, lam, k_restraint=5.0):
    """Gradient of V_lambda = (1-lam)*V_A + lam*V_B"""
    return (1 - lam) * grad_V_A(x, k_restraint) + lam * grad_V_B(x, k_restraint)


# =============================================================================
# Ground truth DeltaF by numerical integration
# =============================================================================

def compute_true_deltaF(kT=1.0, k_restraint=5.0):
    """Compute exact DeltaF = F_B - F_A by numerical integration."""
    x_grid = np.linspace(-5, 5, 10000)
    Z_A = np.trapezoid(np.exp(-V_A(x_grid, k_restraint) / kT), x_grid)
    Z_B = np.trapezoid(np.exp(-V_B(x_grid, k_restraint) / kT), x_grid)
    F_A = -kT * np.log(Z_A)
    F_B = -kT * np.log(Z_B)
    return F_B - F_A


# =============================================================================
# Samplers
# =============================================================================

def run_nh_cnf_sampler(grad_V_fn, n_steps, dt=0.005, Q=1.0, kT=1.0,
                       q_init=0.0, seed=42, burn_frac=0.1, thin=10):
    """Run NH-tanh sampler, return (samples_q, log_density_corrections).

    The log_density correction tracks cumulative divergence integral,
    which gives log p(q_t, p_t, xi_t) up to a constant.
    """
    np.random.seed(seed)
    d = 1
    q = np.array([q_init])
    p = np.random.randn(d) * np.sqrt(kT)
    xi = 0.0

    samples = []
    log_corrections = []
    cum_div = 0.0
    burn_in = int(n_steps * burn_frac)

    for step in range(n_steps):
        q, p, xi, div = nh_tanh_rk4_step(q, p, xi, grad_V_fn, dt, Q, kT, d)
        cum_div += div
        if step >= burn_in and (step - burn_in) % thin == 0:
            samples.append(q[0])
            log_corrections.append(cum_div)

    return np.array(samples), np.array(log_corrections)


def run_langevin_sampler(grad_V_fn, n_steps, eps=0.005, kT=1.0,
                         q_init=0.0, seed=42, burn_frac=0.1, thin=10):
    """Unadjusted Langevin dynamics, return samples."""
    np.random.seed(seed)
    d = 1
    x = np.array([q_init])
    samples = []
    burn_in = int(n_steps * burn_frac)

    for step in range(n_steps):
        gv = grad_V_fn(x)
        noise = np.random.randn(d) * np.sqrt(2 * eps * kT)
        x = x - eps * gv + noise
        if step >= burn_in and (step - burn_in) % thin == 0:
            samples.append(x[0])

    return np.array(samples)


# =============================================================================
# FEP estimators
# =============================================================================

def fep_estimate(samples_A, V_A_fn, V_B_fn, kT=1.0):
    """Standard FEP: DeltaF = -kT * ln <exp(-(V_B - V_A)/kT)>_A

    Uses log-sum-exp for numerical stability.
    """
    delta_V = np.array([V_B_fn(np.array([s])) - V_A_fn(np.array([s])) for s in samples_A])
    # log-sum-exp trick
    exponents = -delta_V / kT
    max_exp = np.max(exponents)
    log_avg = max_exp + np.log(np.mean(np.exp(exponents - max_exp)))
    return -kT * log_avg


def ti_estimate(n_windows, n_steps_per_window, dt=0.005, Q=1.0, kT=1.0,
                k_restraint=5.0, seed=42):
    """Thermodynamic Integration: DeltaF = integral_0^1 <dV/dlambda>_lambda dlambda.

    dV/dlambda = V_B - V_A (since V_lambda = (1-lam)*V_A + lam*V_B).
    """
    lambdas = np.linspace(0, 1, n_windows)
    mean_dVdl = np.zeros(n_windows)

    for i, lam in enumerate(lambdas):
        grad_fn = lambda x, l=lam: grad_V_lambda(x, l, k_restraint)
        # Initialize near the expected minimum for this lambda
        q_init = -1.0 + 2.0 * lam
        samples, _ = run_nh_cnf_sampler(
            grad_fn, n_steps_per_window, dt=dt, Q=Q, kT=kT,
            q_init=q_init, seed=seed + i, burn_frac=0.2, thin=10
        )
        # dV/dlambda = V_B(x) - V_A(x)
        dVdl = np.array([V_B(np.array([s]), k_restraint) - V_A(np.array([s]), k_restraint)
                         for s in samples])
        mean_dVdl[i] = np.mean(dVdl)

    # Trapezoidal integration
    deltaF = np.trapezoid(mean_dVdl, lambdas)
    return deltaF


# =============================================================================
# E1: Double-well DeltaF comparison
# =============================================================================

def experiment_e1():
    """Compare FEP accuracy: NH-CNF vs TI vs Langevin, varying NFE budget."""
    print("=" * 60)
    print("E1: Double-well DeltaF with exact weights")
    print("=" * 60)

    kT = 1.0
    k_restraint = 5.0
    dt = 0.005
    Q = 1.0

    deltaF_true = compute_true_deltaF(kT, k_restraint)
    print(f"True DeltaF = {deltaF_true:.6f}")

    # NFE budgets to test
    nfe_list = [1000, 2000, 5000, 10000, 20000, 50000]
    n_runs = 20  # independent runs for variance estimation

    # Storage
    nhcnf_errors = np.zeros((len(nfe_list), n_runs))
    nhcnf_estimates = np.zeros((len(nfe_list), n_runs))
    lang_errors = np.zeros((len(nfe_list), n_runs))
    lang_estimates = np.zeros((len(nfe_list), n_runs))
    ti_errors = np.zeros((len(nfe_list), n_runs))
    ti_estimates = np.zeros((len(nfe_list), n_runs))

    for ni, nfe in enumerate(nfe_list):
        print(f"\n  NFE = {nfe}")
        # For TI: distribute budget across windows
        n_windows = 11
        nfe_per_window = nfe // n_windows

        for run in range(n_runs):
            seed = 42 + run * 137

            # --- NH-CNF FEP ---
            # NH-tanh: each RK4 step = 4 force evals (4 stages)
            n_steps_nh = nfe // 4
            samples_nh, _ = run_nh_cnf_sampler(
                lambda x: grad_V_A(x, k_restraint),
                n_steps_nh, dt=dt, Q=Q, kT=kT,
                q_init=-1.0, seed=seed, burn_frac=0.1, thin=1
            )
            df_nh = fep_estimate(samples_nh,
                                 lambda x: V_A(x, k_restraint),
                                 lambda x: V_B(x, k_restraint), kT)
            nhcnf_errors[ni, run] = abs(df_nh - deltaF_true)
            nhcnf_estimates[ni, run] = df_nh

            # --- Langevin FEP ---
            # Langevin: 1 force eval per step
            samples_lang = run_langevin_sampler(
                lambda x: grad_V_A(x, k_restraint),
                nfe, eps=dt, kT=kT,
                q_init=-1.0, seed=seed, burn_frac=0.1, thin=1
            )
            df_lang = fep_estimate(samples_lang,
                                   lambda x: V_A(x, k_restraint),
                                   lambda x: V_B(x, k_restraint), kT)
            lang_errors[ni, run] = abs(df_lang - deltaF_true)
            lang_estimates[ni, run] = df_lang

            # --- TI (only for larger budgets, skip smallest to save time) ---
            if nfe >= 2000:
                n_steps_ti = nfe_per_window // 4  # RK4 steps per window
                df_ti = ti_estimate(n_windows, max(n_steps_ti, 50), dt=dt, Q=Q, kT=kT,
                                    k_restraint=k_restraint, seed=seed)
                ti_errors[ni, run] = abs(df_ti - deltaF_true)
                ti_estimates[ni, run] = df_ti
            else:
                ti_errors[ni, run] = np.nan
                ti_estimates[ni, run] = np.nan

        print(f"    NH-CNF: |err| = {np.mean(nhcnf_errors[ni]):.4f} +/- {np.std(nhcnf_errors[ni]):.4f}")
        print(f"    Langevin: |err| = {np.mean(lang_errors[ni]):.4f} +/- {np.std(lang_errors[ni]):.4f}")
        if not np.isnan(ti_errors[ni, 0]):
            print(f"    TI (11 win): |err| = {np.nanmean(ti_errors[ni]):.4f} +/- {np.nanstd(ti_errors[ni]):.4f}")

    # --- Plot E1 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # (a) DeltaF error vs NFE
    ax = axes[0]
    mean_nh = np.mean(nhcnf_errors, axis=1)
    std_nh = np.std(nhcnf_errors, axis=1)
    mean_lang = np.mean(lang_errors, axis=1)
    std_lang = np.std(lang_errors, axis=1)
    mean_ti = np.nanmean(ti_errors, axis=1)
    std_ti = np.nanstd(ti_errors, axis=1)

    ax.plot(nfe_list, mean_nh, 'o-', color=C_NHCNF, label='NH-CNF FEP', lw=2)
    ax.fill_between(nfe_list, mean_nh - std_nh, mean_nh + std_nh, color=C_NHCNF, alpha=0.2)
    ax.plot(nfe_list, mean_lang, 's-', color=C_LANG, label='Langevin FEP', lw=2)
    ax.fill_between(nfe_list, mean_lang - std_lang, mean_lang + std_lang, color=C_LANG, alpha=0.2)

    # TI: mask NaN entries
    ti_mask = ~np.isnan(mean_ti)
    nfe_ti = np.array(nfe_list)[ti_mask]
    ax.plot(nfe_ti, mean_ti[ti_mask], '^-', color=C_TI, label='TI (11 windows)', lw=2)
    ax.fill_between(nfe_ti, (mean_ti - std_ti)[ti_mask], (mean_ti + std_ti)[ti_mask],
                    color=C_TI, alpha=0.2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Force Evaluations (NFE)')
    ax.set_ylabel(r'$|\Delta F_{est} - \Delta F_{true}|$')
    ax.set_title(r'(a) $\Delta F$ Error vs NFE', fontweight='bold')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    # (b) Variance of DeltaF estimate vs NFE
    ax = axes[1]
    var_nh = np.var(nhcnf_estimates, axis=1)
    var_lang = np.var(lang_estimates, axis=1)
    var_ti = np.nanvar(ti_estimates, axis=1)

    ax.plot(nfe_list, var_nh, 'o-', color=C_NHCNF, label='NH-CNF FEP', lw=2)
    ax.plot(nfe_list, var_lang, 's-', color=C_LANG, label='Langevin FEP', lw=2)
    ax.plot(nfe_ti, var_ti[ti_mask], '^-', color=C_TI, label='TI (11 windows)', lw=2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Force Evaluations (NFE)')
    ax.set_ylabel(r'Var[$\Delta F_{est}$]')
    ax.set_title(r'(b) Variance of $\Delta F$ Estimate', fontweight='bold')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    # (c) Double-well potential with states marked
    ax = axes[2]
    x = np.linspace(-2.5, 2.5, 500)
    ax.plot(x, V_doublewell(x), 'k-', lw=2, label=r'$V(x) = (x^2-1)^2 + 0.3x$')
    ax.plot(x, V_A(x, k_restraint), '--', color=C_NHCNF, lw=1.5, alpha=0.7, label=r'$V_A$ (restrained)')
    ax.plot(x, V_B(x, k_restraint), '--', color=C_TI, lw=1.5, alpha=0.7, label=r'$V_B$ (restrained)')

    # Mark minima
    ax.axvline(-1, color=C_NHCNF, ls=':', alpha=0.5)
    ax.axvline(1, color=C_TI, ls=':', alpha=0.5)
    ax.text(-1.0, -0.3, 'State A', ha='center', fontsize=12, color=C_NHCNF, fontweight='bold')
    ax.text(1.0, -0.3, 'State B', ha='center', fontsize=12, color=C_TI, fontweight='bold')

    ax.set_ylim(-1, 15)
    ax.set_xlabel('x')
    ax.set_ylabel('V(x)')
    ax.set_title('(c) Asymmetric Double-Well', fontweight='bold')
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add true DeltaF annotation
    ax.text(0.5, 0.95, rf'$\Delta F_{{true}}$ = {deltaF_true:.4f}',
            transform=ax.transAxes, ha='center', va='top', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    fig.savefig(os.path.join(FIGDIR, 'e1_fep.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved e1_fep.png")

    # Return key metric: error at 10k NFE
    idx_10k = nfe_list.index(10000)
    return {
        'deltaF_true': deltaF_true,
        'nhcnf_error_10k': np.mean(nhcnf_errors[idx_10k]),
        'lang_error_10k': np.mean(lang_errors[idx_10k]),
        'ti_error_10k': np.nanmean(ti_errors[idx_10k]),
        'nhcnf_var_10k': np.var(nhcnf_estimates[idx_10k]),
        'lang_var_10k': np.var(lang_estimates[idx_10k]),
    }


# =============================================================================
# E2: Multi-well free energy landscape (2D)
# =============================================================================

def V_threewell(x, y, asym=0.15):
    """3-fold symmetric potential with slight asymmetry.
    V = (r^2 - 1)^2 - 0.5*cos(3*theta) + asym*x
    """
    r2 = x**2 + y**2
    theta = np.arctan2(y, x)
    return (r2 - 1)**2 - 0.5 * np.cos(3 * theta) + asym * x

def grad_V_threewell(xy, asym=0.15):
    """Gradient of 3-well potential. xy is array [x, y]."""
    x, y = xy[0], xy[1]
    r2 = x**2 + y**2
    theta = np.arctan2(y, x)

    # d/dx of (r^2 - 1)^2 = 4x(r^2 - 1)
    dr2_dx = 4 * x * (r2 - 1)
    dr2_dy = 4 * y * (r2 - 1)

    # d/dx of -0.5*cos(3*theta) = 0.5*sin(3*theta) * 3 * dtheta/dx
    # dtheta/dx = -y/r^2, dtheta/dy = x/r^2
    r2_safe = max(r2, 1e-10)
    dcos_dx = 1.5 * np.sin(3 * theta) * (-y / r2_safe)
    dcos_dy = 1.5 * np.sin(3 * theta) * (x / r2_safe)

    gx = dr2_dx + dcos_dx + asym
    gy = dr2_dy + dcos_dy
    return np.array([gx, gy])


def experiment_e2():
    """2D three-well free energy landscape."""
    print("\n" + "=" * 60)
    print("E2: Multi-well free energy landscape (2D)")
    print("=" * 60)

    kT = 1.0
    asym = 0.15

    # Ground truth: numerical integration on a grid
    x_grid = np.linspace(-2.5, 2.5, 500)
    y_grid = np.linspace(-2.5, 2.5, 500)
    X, Y = np.meshgrid(x_grid, y_grid)
    V_grid = V_threewell(X, Y, asym)
    boltz = np.exp(-V_grid / kT)
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    Z_total = np.sum(boltz) * dx * dy

    # Define wells by angle sectors (3-fold symmetry)
    theta_grid = np.arctan2(Y, X)
    R_grid = np.sqrt(X**2 + Y**2)
    # Well centers at theta ~ 0, 2pi/3, -2pi/3 on unit circle
    well_masks = [
        (theta_grid > -np.pi/3) & (theta_grid <= np.pi/3) & (R_grid > 0.3),        # Well 1 (right)
        (theta_grid > np.pi/3) & (theta_grid <= np.pi) & (R_grid > 0.3),            # Well 2 (upper-left)
        (theta_grid > -np.pi) & (theta_grid <= -np.pi/3) & (R_grid > 0.3),          # Well 3 (lower-left)
    ]

    well_names = ['Right', 'Upper-Left', 'Lower-Left']
    F_wells = []
    for i, mask in enumerate(well_masks):
        Z_i = np.sum(boltz[mask]) * dx * dy
        F_i = -kT * np.log(Z_i)
        F_wells.append(F_i)
        print(f"  Well {i+1} ({well_names[i]}): F = {F_i:.4f}, Z = {Z_i:.4f}")

    # Relative to well 1
    F_rel = [f - F_wells[0] for f in F_wells]
    print(f"\n  Relative free energies: {[f'{f:.4f}' for f in F_rel]}")

    # NH-CNF sampling from full potential (no restraint)
    n_steps = 50000
    dt = 0.003
    Q = 1.0
    np.random.seed(SEED)

    d = 2
    q = np.array([1.0, 0.0])  # start near well 1
    p = np.random.randn(d) * np.sqrt(kT)
    xi = 0.0

    samples_2d = []
    burn_in = 5000
    thin = 5
    for step in range(n_steps):
        q, p, xi, div = nh_tanh_rk4_step(
            q, p, xi, lambda x: grad_V_threewell(x, asym), dt, Q, kT, d
        )
        if step >= burn_in and (step - burn_in) % thin == 0:
            samples_2d.append(q.copy())

    samples_2d = np.array(samples_2d)
    print(f"  Collected {len(samples_2d)} samples from NH-CNF")

    # Assign samples to wells and compute free energies
    sample_theta = np.arctan2(samples_2d[:, 1], samples_2d[:, 0])
    sample_r = np.sqrt(samples_2d[:, 0]**2 + samples_2d[:, 1]**2)

    well_counts = []
    for i, name in enumerate(well_names):
        if i == 0:
            mask = (sample_theta > -np.pi/3) & (sample_theta <= np.pi/3) & (sample_r > 0.3)
        elif i == 1:
            mask = (sample_theta > np.pi/3) & (sample_theta <= np.pi) & (sample_r > 0.3)
        else:
            mask = (sample_theta > -np.pi) & (sample_theta <= -np.pi/3) & (sample_r > 0.3)
        well_counts.append(np.sum(mask))

    total_assigned = sum(well_counts)
    F_sampled = [-kT * np.log(max(c, 1) / total_assigned) for c in well_counts]
    F_sampled_rel = [f - F_sampled[0] for f in F_sampled]

    print(f"  Well counts: {well_counts}")
    print(f"  Sampled relative F: {[f'{f:.4f}' for f in F_sampled_rel]}")
    print(f"  True relative F:    {[f'{f:.4f}' for f in F_rel]}")

    # --- Plot E2 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    # (a) Potential contour
    ax = axes[0]
    levels = np.linspace(-0.5, 4, 20)
    cs = ax.contourf(X, Y, V_grid, levels=levels, cmap='viridis')
    ax.contour(X, Y, V_grid, levels=levels, colors='k', linewidths=0.3, alpha=0.5)
    plt.colorbar(cs, ax=ax, label='V(x,y)')

    # Mark well centers
    well_centers = [(1.0, 0.0), (-0.5, 0.866), (-0.5, -0.866)]
    for i, (cx, cy) in enumerate(well_centers):
        ax.plot(cx, cy, 'w*', markersize=15, markeredgecolor='k', markeredgewidth=0.5)
        ax.annotate(f'Well {i+1}', (cx, cy), textcoords='offset points',
                   xytext=(10, 10), fontsize=10, color='white', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('(a) Three-Well Potential', fontweight='bold')
    ax.set_aspect('equal')

    # (b) NH-CNF samples colored by well assignment
    ax = axes[1]
    ax.contour(X, Y, V_grid, levels=levels, colors='gray', linewidths=0.3, alpha=0.3)
    ax.scatter(samples_2d[:, 0], samples_2d[:, 1], s=1, alpha=0.3, c='steelblue')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('(b) NH-CNF Samples', fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)

    # (c) Free energy comparison (bar chart)
    ax = axes[2]
    x_pos = np.arange(3)
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, F_rel, width, label='True (numerical)',
                   color=C_TRUE, alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, F_sampled_rel, width, label='NH-CNF (sampled)',
                   color=C_NHCNF, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(well_names)
    ax.set_ylabel(r'$\Delta F$ (relative to Well 1)')
    ax.set_title(r'(c) Free Energy Comparison', fontweight='bold')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3, axis='y')

    fig.savefig(os.path.join(FIGDIR, 'e2_landscape.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved e2_landscape.png")

    return {
        'F_rel_true': F_rel,
        'F_rel_sampled': F_sampled_rel,
        'max_error': max(abs(t - s) for t, s in zip(F_rel, F_sampled_rel)),
    }


# =============================================================================
# E3: Variance scaling with perturbation size (harmonic)
# =============================================================================

def experiment_e3():
    """Harmonic FEP: variance scaling with k_B/k_A ratio."""
    print("\n" + "=" * 60)
    print("E3: Variance scaling with perturbation size")
    print("=" * 60)

    kT = 1.0
    k_A = 1.0
    # k_B/k_A ratios to test
    ratios = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]

    nfe = 10000
    n_runs = 20
    dt = 0.005
    Q = 1.0

    # Analytical DeltaF for harmonic: DeltaF = 0.5*kT*ln(k_B/k_A)
    def true_deltaF_harmonic(k_A, k_B, kT):
        return 0.5 * kT * np.log(k_B / k_A)

    nhcnf_var = np.zeros(len(ratios))
    lang_var = np.zeros(len(ratios))
    nhcnf_bias = np.zeros(len(ratios))
    lang_bias = np.zeros(len(ratios))

    for ri, ratio in enumerate(ratios):
        k_B = k_A * ratio
        df_true = true_deltaF_harmonic(k_A, k_B, kT)

        grad_VA = lambda x, ka=k_A: ka * x
        VA_fn = lambda x, ka=k_A: 0.5 * ka * x**2
        VB_fn = lambda x, kb=k_B: 0.5 * kb * x**2

        nh_ests = np.zeros(n_runs)
        lang_ests = np.zeros(n_runs)

        for run in range(n_runs):
            seed = 42 + run * 137

            # NH-CNF
            n_steps_nh = nfe // 4
            samples_nh, _ = run_nh_cnf_sampler(
                grad_VA, n_steps_nh, dt=dt, Q=Q, kT=kT,
                q_init=0.0, seed=seed, burn_frac=0.1, thin=1
            )
            nh_ests[run] = fep_estimate(samples_nh, VA_fn, VB_fn, kT)

            # Langevin
            samples_lang = run_langevin_sampler(
                grad_VA, nfe, eps=dt, kT=kT,
                q_init=0.0, seed=seed, burn_frac=0.1, thin=1
            )
            lang_ests[run] = fep_estimate(samples_lang, VA_fn, VB_fn, kT)

        nhcnf_var[ri] = np.var(nh_ests)
        lang_var[ri] = np.var(lang_ests)
        nhcnf_bias[ri] = abs(np.mean(nh_ests) - df_true)
        lang_bias[ri] = abs(np.mean(lang_ests) - df_true)

        print(f"  k_B/k_A = {ratio:.1f}: DeltaF_true = {df_true:.4f}")
        print(f"    NH-CNF: mean={np.mean(nh_ests):.4f}, var={nhcnf_var[ri]:.6f}, bias={nhcnf_bias[ri]:.4f}")
        print(f"    Langevin: mean={np.mean(lang_ests):.4f}, var={lang_var[ri]:.6f}, bias={lang_bias[ri]:.4f}")

    # --- Plot E3 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # (a) Variance vs perturbation ratio
    ax = axes[0]
    ax.plot(ratios, nhcnf_var, 'o-', color=C_NHCNF, label='NH-CNF FEP', lw=2)
    ax.plot(ratios, lang_var, 's-', color=C_LANG, label='Langevin FEP', lw=2)
    ax.set_xlabel(r'$k_B / k_A$')
    ax.set_ylabel(r'Var[$\Delta F_{est}$]')
    ax.set_title(r'(a) Variance vs Perturbation Size', fontweight='bold')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # (b) Bias vs perturbation ratio
    ax = axes[1]
    ax.plot(ratios, nhcnf_bias, 'o-', color=C_NHCNF, label='NH-CNF FEP', lw=2)
    ax.plot(ratios, lang_bias, 's-', color=C_LANG, label='Langevin FEP', lw=2)
    ax.set_xlabel(r'$k_B / k_A$')
    ax.set_ylabel(r'$|\langle\Delta F_{est}\rangle - \Delta F_{true}|$')
    ax.set_title('(b) Bias vs Perturbation Size', fontweight='bold')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # (c) Variance ratio (Langevin / NH-CNF)
    ax = axes[2]
    # Avoid division by zero
    var_ratio = np.where(nhcnf_var > 1e-15, lang_var / nhcnf_var, 1.0)
    ax.plot(ratios, var_ratio, 'D-', color='#9467bd', lw=2)
    ax.axhline(1.0, ls='--', color='gray', alpha=0.5)
    ax.set_xlabel(r'$k_B / k_A$')
    ax.set_ylabel(r'Var[Langevin] / Var[NH-CNF]')
    ax.set_title('(c) Relative Efficiency', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Annotate
    if np.any(var_ratio > 1):
        ax.text(0.5, 0.95, 'Above 1 = NH-CNF better', transform=ax.transAxes,
                ha='center', va='top', fontsize=11, style='italic', color='gray')

    fig.savefig(os.path.join(FIGDIR, 'e3_variance.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved e3_variance.png")

    return {
        'ratios': ratios,
        'nhcnf_var': nhcnf_var.tolist(),
        'lang_var': lang_var.tolist(),
        'var_ratio': var_ratio.tolist(),
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("Exact FEP with NH-CNF — Experiment Suite")
    print("=" * 60)

    results_e1 = experiment_e1()
    results_e2 = experiment_e2()
    results_e3 = experiment_e3()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nE1 (Double-well DeltaF at 10k NFE):")
    print(f"  True DeltaF = {results_e1['deltaF_true']:.6f}")
    print(f"  NH-CNF error = {results_e1['nhcnf_error_10k']:.6f}")
    print(f"  Langevin error = {results_e1['lang_error_10k']:.6f}")
    print(f"  TI error = {results_e1['ti_error_10k']:.6f}")
    print(f"\nE2 (Three-well landscape):")
    print(f"  Max relative F error = {results_e2['max_error']:.4f}")
    print(f"\nE3 (Harmonic variance scaling):")
    print(f"  Variance ratio (Lang/NHCNF) at k_B/k_A=10: {results_e3['var_ratio'][-1]:.2f}")

    # Key metric for orbit
    metric = results_e1['nhcnf_error_10k']
    print(f"\n*** ORBIT METRIC (DeltaF error at 10k NFE): {metric:.6f} ***")
