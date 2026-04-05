#!/usr/bin/env python3
"""Figure 2: 'The Solution' -- Bounded friction breaks KAM tori.

6-panel (2x3) layout:
  (a) Friction functions g(xi): NH (linear) vs Log-Osc (bounded) vs Tanh
  (b) Thermostat potentials V(xi): NH (parabola) vs Log-Osc (logarithmic)
  (c) Phase space NH on 1D HO (torus visible)
  (d) Phase space Log-Osc on 1D HO (space-filling)
  (e) Lyapunov exponent vs Q
  (f) Ergodicity score vs Q
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from research.eval.potentials import HarmonicOscillator1D
from research.eval.baselines import NoseHoover
from research.eval.integrators import VelocityVerletThermostat
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "log_osc_001",
    os.path.join(os.path.dirname(__file__), '..', 'log-osc-001', 'solution.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LogOscThermostat = _mod.LogOscThermostat
LogOscVelocityVerlet = _mod.LogOscVelocityVerlet

# ── Style constants ──
COLOR_NH = '#1f77b4'
COLOR_NHC = '#ff7f0e'
tab10 = plt.cm.tab10
COLOR_LOGOSC = tab10(2)
COLOR_TANH = tab10(3)
COLOR_ARCTAN = tab10(4)
FONTSIZE_LABEL = 14
FONTSIZE_TICK = 12
FONTSIZE_TITLE = 16
DPI = 300

SEED = 42
N_STEPS = 500_000
DT = 0.01
KT = 1.0


def g_nh(xi):
    return xi

def g_logosc(xi):
    return 2.0 * xi / (1.0 + xi**2)

def g_tanh(xi):
    return np.tanh(xi)

def g_arctan(xi):
    return (2.0 / np.pi) * np.arctan(xi)

def V_nh(xi, Q=1.0):
    return 0.5 * Q * xi**2

def V_logosc(xi, Q=1.0):
    return Q * np.log(1.0 + xi**2)


def run_sampler(SamplerClass, IntegratorClass, Q=1.0, dt=0.01, n_steps=N_STEPS):
    rng = np.random.default_rng(SEED)
    pot = HarmonicOscillator1D(omega=1.0)
    dyn = SamplerClass(dim=1, kT=KT, mass=1.0, Q=Q)
    q0 = np.array([1.0])
    state = dyn.initial_state(q0, rng=rng)
    integrator = IntegratorClass(dyn, pot, dt=dt, kT=KT, mass=1.0)

    qs = np.empty(n_steps)
    ps = np.empty(n_steps)
    xis = np.empty(n_steps)

    for i in range(n_steps):
        qs[i] = state.q[0]
        ps[i] = state.p[0]
        xis[i] = state.xi[0]
        state = integrator.step(state)

    return qs, ps, xis


def compute_ergodicity_score(qs, ps, kT=1.0, grid_n=20):
    """Compute ergodicity score for 1D HO samples."""
    from scipy.stats import kstest
    burn = len(qs) // 10
    q_post = qs[burn:]
    p_post = ps[burn:]

    # KS statistic
    ks_q = kstest(q_post, 'norm', args=(0, np.sqrt(kT))).statistic
    ks_p = kstest(p_post, 'norm', args=(0, np.sqrt(kT))).statistic
    ks_comp = 1.0 - max(ks_q, ks_p)

    # Variance match
    var_q_err = abs(np.var(q_post) / kT - 1.0)
    var_p_err = abs(np.var(p_post) / kT - 1.0)
    var_comp = 1.0 - max(var_q_err, var_p_err)

    # Phase space coverage
    q_edges = np.linspace(-4, 4, grid_n + 1)
    p_edges = np.linspace(-4, 4, grid_n + 1)
    H, _, _ = np.histogram2d(q_post, p_post, bins=[q_edges, p_edges])
    coverage = np.sum(H > 0) / (grid_n * grid_n)

    # Geometric mean
    comps = np.array([max(ks_comp, 0.01), max(var_comp, 0.01), max(coverage, 0.01)])
    score = np.prod(comps) ** (1.0/3.0)
    return score


def compute_lyapunov(SamplerClass, IntegratorClass, Q, dt=0.01, n_steps=200000, eps=1e-7):
    """Estimate maximal Lyapunov exponent via tangent-space method."""
    rng = np.random.default_rng(SEED)
    pot = HarmonicOscillator1D(omega=1.0)
    dyn = SamplerClass(dim=1, kT=KT, mass=1.0, Q=Q)
    q0 = np.array([1.0])

    # Reference trajectory
    state = dyn.initial_state(q0, rng=rng)
    integ = IntegratorClass(dyn, pot, dt=dt, kT=KT, mass=1.0)

    # Perturbed trajectory
    rng2 = np.random.default_rng(SEED)
    state_p = dyn.initial_state(q0, rng=rng2)
    # Add small perturbation to q
    state_p = state_p._replace(q=state_p.q + eps)
    integ_p = IntegratorClass(dyn, pot, dt=dt, kT=KT, mass=1.0)

    lyap_sum = 0.0
    n_renorm = 0
    renorm_interval = 100

    for i in range(n_steps):
        state = integ.step(state)
        state_p = integ_p.step(state_p)

        if (i + 1) % renorm_interval == 0:
            dq = state_p.q[0] - state.q[0]
            dp = state_p.p[0] - state.p[0]
            dxi = state_p.xi[0] - state.xi[0]
            dist = np.sqrt(dq**2 + dp**2 + dxi**2)
            if dist > 0 and np.isfinite(dist):
                lyap_sum += np.log(dist / eps)
                n_renorm += 1
                # Renormalize
                scale = eps / dist
                state_p = state_p._replace(
                    q=state.q + (state_p.q - state.q) * scale,
                    p=state.p + (state_p.p - state.p) * scale,
                    xi=state.xi + (state_p.xi - state.xi) * scale,
                )
                # Reset FSAL caches
                integ_p._cached_grad_U = None

    if n_renorm == 0:
        return 0.0
    return lyap_sum / (n_renorm * renorm_interval * dt)


def make_figure():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # ── Panel (a): Friction functions g(xi) ──
    ax = axes[0, 0]
    xi = np.linspace(-5, 5, 500)
    ax.plot(xi, g_nh(xi), color=COLOR_NH, lw=2.5, label=r'NH: $g(\xi) = \xi$')
    ax.plot(xi, g_logosc(xi), color=COLOR_LOGOSC, lw=2.5, label=r'Log-Osc: $\frac{2\xi}{1+\xi^2}$')
    ax.plot(xi, g_tanh(xi), color=COLOR_TANH, lw=2.5, ls='--', label=r'Tanh: $\tanh(\xi)$')
    ax.axhline(1, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax.axhline(-1, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax.annotate('bounded\n$|g| \\leq 1$', xy=(3.5, 1.0), fontsize=10, ha='center', va='bottom',
                color='gray')
    ax.set_xlabel(r'$\xi$', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'$g(\xi)$', fontsize=FONTSIZE_LABEL)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-3, 5)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.legend(fontsize=10, loc='upper left')
    ax.text(0.03, 0.93, '(a)', transform=ax.transAxes, fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    # ── Panel (b): Thermostat potentials V(xi) ──
    ax = axes[0, 1]
    xi = np.linspace(-6, 6, 500)
    Q = 1.0
    ax.plot(xi, V_nh(xi, Q), color=COLOR_NH, lw=2.5, label=r'NH: $\frac{Q}{2}\xi^2$')
    ax.plot(xi, V_logosc(xi, Q), color=COLOR_LOGOSC, lw=2.5, label=r'Log-Osc: $Q\ln(1+\xi^2)$')

    # Show corresponding xi distributions (inset-like shading)
    ax2 = ax.twinx()
    rho_nh = np.exp(-V_nh(xi, Q) / KT)
    rho_lo = np.exp(-V_logosc(xi, Q) / KT)
    rho_nh /= np.trapezoid(rho_nh, xi)
    rho_lo /= np.trapezoid(rho_lo, xi)
    ax2.fill_between(xi, rho_nh, alpha=0.12, color=COLOR_NH)
    ax2.fill_between(xi, rho_lo, alpha=0.12, color=COLOR_LOGOSC)
    ax2.plot(xi, rho_nh, color=COLOR_NH, lw=1, ls=':', alpha=0.6)
    ax2.plot(xi, rho_lo, color=COLOR_LOGOSC, lw=1, ls=':', alpha=0.6)
    ax2.set_ylabel(r'$\rho(\xi) \propto e^{-V(\xi)/kT}$', fontsize=11, color='gray')
    ax2.tick_params(labelsize=FONTSIZE_TICK, colors='gray')
    ax2.set_ylim(0, 0.55)

    ax.set_xlabel(r'$\xi$', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'$V(\xi)$', fontsize=FONTSIZE_LABEL)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-0.5, 15)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.legend(fontsize=10, loc='upper left')
    ax.annotate('heavy-tailed', xy=(4, 0.5), fontsize=9, color=COLOR_LOGOSC, ha='center',
                fontstyle='italic')
    ax.text(0.03, 0.93, '(b)', transform=ax.transAxes, fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    # ── Panel (c): Phase space NH ──
    ax = axes[1, 0]
    qs_nh, ps_nh, _ = run_sampler(NoseHoover, VelocityVerletThermostat, Q=1.0)
    thin = 5
    idx = np.arange(0, len(qs_nh), thin)
    ax.scatter(qs_nh[idx], ps_nh[idx], c=np.arange(len(idx)), cmap='coolwarm',
               s=0.2, alpha=0.35, rasterized=True)
    theta = np.linspace(0, 2*np.pi, 200)
    for s in [1, 2, 3]:
        ax.plot(s*np.cos(theta), s*np.sin(theta), '--', color='gray', alpha=0.3, lw=0.8)
    ax.set_xlabel(r'$q$', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'$p$', fontsize=FONTSIZE_LABEL)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.set_title('Nose-Hoover', fontsize=13, color=COLOR_NH)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.text(0.03, 0.93, '(c)', transform=ax.transAxes, fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    # ── Panel (d): Phase space Log-Osc ──
    ax = axes[1, 1]
    qs_lo, ps_lo, _ = run_sampler(LogOscThermostat, LogOscVelocityVerlet, Q=0.5)
    idx = np.arange(0, len(qs_lo), thin)
    ax.scatter(qs_lo[idx], ps_lo[idx], c=np.arange(len(idx)), cmap='coolwarm',
               s=0.2, alpha=0.35, rasterized=True)
    for s in [1, 2, 3]:
        ax.plot(s*np.cos(theta), s*np.sin(theta), '--', color='gray', alpha=0.3, lw=0.8)
    ax.set_xlabel(r'$q$', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'$p$', fontsize=FONTSIZE_LABEL)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.set_title('Log-Osc', fontsize=13, color=COLOR_LOGOSC)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.text(0.03, 0.93, '(d)', transform=ax.transAxes, fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    # ── Panel (e): Lyapunov exponent vs Q ──
    ax = axes[0, 2]
    Q_values = [0.1, 0.3, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0]
    print("Computing Lyapunov exponents...")
    lyap_nh = [compute_lyapunov(NoseHoover, VelocityVerletThermostat, Q=q, n_steps=100000) for q in Q_values]
    print("  NH done")
    lyap_lo = [compute_lyapunov(LogOscThermostat, LogOscVelocityVerlet, Q=q, n_steps=100000) for q in Q_values]
    print("  Log-Osc done")

    ax.plot(Q_values, lyap_nh, 'o-', color=COLOR_NH, lw=2, markersize=5, label='NH')
    ax.plot(Q_values, lyap_lo, 's-', color=COLOR_LOGOSC, lw=2, markersize=5, label='Log-Osc')
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel(r'$Q$ (thermostat mass)', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'$\lambda_{\max}$ (Lyapunov exponent)', fontsize=FONTSIZE_LABEL)
    ax.set_xscale('log')
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.legend(fontsize=10, loc='best')
    ax.text(0.03, 0.93, '(e)', transform=ax.transAxes, fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    # ── Panel (f): Ergodicity score vs Q ──
    ax = axes[1, 2]
    print("Computing ergodicity scores...")
    Q_ergo = [0.1, 0.3, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0]
    ergo_nh = []
    ergo_lo = []
    for q in Q_ergo:
        qs_, ps_, _ = run_sampler(NoseHoover, VelocityVerletThermostat, Q=q, n_steps=200000)
        ergo_nh.append(compute_ergodicity_score(qs_, ps_))
        qs_, ps_, _ = run_sampler(LogOscThermostat, LogOscVelocityVerlet, Q=q, n_steps=200000)
        ergo_lo.append(compute_ergodicity_score(qs_, ps_))
    print("  Done")

    ax.plot(Q_ergo, ergo_nh, 'o-', color=COLOR_NH, lw=2, markersize=5, label='NH')
    ax.plot(Q_ergo, ergo_lo, 's-', color=COLOR_LOGOSC, lw=2, markersize=5, label='Log-Osc')
    ax.axhline(0.85, color='gray', ls='--', lw=1, alpha=0.7)
    ax.annotate('ergodic threshold', xy=(8, 0.86), fontsize=9, color='gray', ha='right')
    ax.set_xlabel(r'$Q$ (thermostat mass)', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Ergodicity score', fontsize=FONTSIZE_LABEL)
    ax.set_xscale('log')
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.legend(fontsize=10, loc='best')
    ax.text(0.03, 0.93, '(f)', transform=ax.transAxes, fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    fig.suptitle('Bounded Friction Breaks KAM Tori', fontsize=FONTSIZE_TITLE, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(os.path.dirname(__file__), 'figures', 'fig2_solution.png')
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    make_figure()
