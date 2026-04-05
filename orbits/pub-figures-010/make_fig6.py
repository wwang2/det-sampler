#!/usr/bin/env python3
"""Figure 6: 'Scaling to High Dimensions' -- Bounded friction helps MORE in high-D.

3-panel (1x3) layout:
  (a) ESS/force_eval vs dimensionality (DOF)
  (b) Energy distribution P(E) for LJ-7 (2D): samplers vs reference
  (c) Marginal variance errors on 10D Gaussian: bar chart by dimension index
"""

import sys, os, importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import kstest

from research.eval.potentials import LennardJonesCluster
from research.eval.baselines import NoseHoover, NoseHooverChain
from research.eval.integrators import VelocityVerletThermostat

# Import Log-Osc
_base = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "log_osc_001", os.path.join(_base, '..', 'log-osc-001', 'solution.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LogOscThermostat = _mod.LogOscThermostat
LogOscVelocityVerlet = _mod.LogOscVelocityVerlet
LogOscChain = _mod.LogOscChain
LogOscChainVerlet = _mod.LogOscChainVerlet

# Import Multi-Scale Log-Osc
_spec2 = importlib.util.spec_from_file_location(
    "log_osc_multiT_005", os.path.join(_base, '..', 'log-osc-multiT-005', 'solution.py'))
_mod2 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_mod2)
MultiScaleLogOsc = _mod2.MultiScaleLogOsc
MultiScaleLogOscVerlet = _mod2.MultiScaleLogOscVerlet

# ── Style constants ──
COLOR_NH = '#1f77b4'
COLOR_NHC = '#ff7f0e'
tab10 = plt.cm.tab10
COLOR_LOGOSC = tab10(2)
COLOR_LOCR = tab10(3)
COLOR_MSLO = tab10(4)
FONTSIZE_LABEL = 14
FONTSIZE_TICK = 12
FONTSIZE_TITLE = 16
DPI = 300

SEED = 42
KT = 1.0


class NDGaussian:
    """N-dimensional isotropic Gaussian potential: U(q) = 0.5 * |q|^2."""
    def __init__(self, dim):
        self.dim = dim
        self.name = f"gaussian_{dim}d"
    def energy(self, q):
        return 0.5 * np.sum(q**2)
    def gradient(self, q):
        return q.copy()


def compute_ess(samples, thin=10):
    """Estimate ESS via autocorrelation of energy-like observable."""
    x = samples[::thin]
    n = len(x)
    if n < 100:
        return 1.0
    x = x - np.mean(x)
    var = np.var(x)
    if var < 1e-15:
        return 1.0

    # Autocorrelation via FFT
    fft_x = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(fft_x * np.conj(fft_x))[:n].real
    acf = acf / acf[0]

    # Integrated autocorrelation time
    tau = 1.0
    for k in range(1, n//2):
        if acf[k] < 0.05:
            break
        tau += 2.0 * acf[k]

    ess = n * thin / tau
    return ess


def run_sampler_nd(sampler_cls, integ_cls, potential, dim, n_evals, dt, seed=SEED, **kwargs):
    """Run sampler on N-D potential, return position trajectory."""
    rng = np.random.default_rng(seed)
    dyn = sampler_cls(dim=dim, kT=KT, mass=1.0, **kwargs)
    q0 = rng.normal(0, 0.3, size=dim)
    state = dyn.initial_state(q0, rng=rng)
    integrator = integ_cls(dyn, potential, dt=dt, kT=KT, mass=1.0)

    n_steps = min(n_evals, 500000)
    qs = np.empty((n_steps, dim))
    energies = np.empty(n_steps)

    for i in range(n_steps):
        qs[i] = state.q[:dim]
        energies[i] = potential.energy(state.q[:dim])
        state = integrator.step(state)

    return qs, energies


def make_figure():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # ── Sampler configs ──
    samplers = {
        'NH': (NoseHoover, VelocityVerletThermostat, {'Q': 1.0}, 0.01),
        'NHC': (NoseHooverChain, VelocityVerletThermostat, {'Q': 1.0, 'chain_length': 3}, 0.01),
        'Log-Osc': (LogOscThermostat, LogOscVelocityVerlet, {'Q': 0.5}, 0.01),
        'LOCR': (LogOscChain, LogOscChainVerlet, {'Q': 0.5, 'chain_length': 3}, 0.01),
        'MSLO': (MultiScaleLogOsc, MultiScaleLogOscVerlet, {'Qs': [0.1, 1.0, 10.0]}, 0.02),
    }
    colors = {
        'NH': COLOR_NH, 'NHC': COLOR_NHC, 'Log-Osc': COLOR_LOGOSC,
        'LOCR': COLOR_LOCR, 'MSLO': COLOR_MSLO,
    }
    markers = {'NH': 'o', 'NHC': 's', 'Log-Osc': '^', 'LOCR': 'D', 'MSLO': 'v'}

    # ── Panel (a): ESS/force_eval vs dimensionality ──
    ax = axes[0]
    dims = [2, 4, 8, 14, 20]
    n_evals = 300000
    print("Computing ESS vs dimensionality...")
    for name, (cls, integ, params, dt) in samplers.items():
        ess_vals = []
        for d in dims:
            pot = NDGaussian(d)
            # Use smaller dt for higher dims
            dt_use = min(dt, 0.5 / np.sqrt(d))
            qs, energies = run_sampler_nd(cls, integ, pot, d, n_evals, dt_use, **params)
            # ESS based on energy trace
            ess = compute_ess(energies) / n_evals
            ess_vals.append(ess)
            print(f"  {name} dim={d}: ESS/eval={ess:.6f}")
        ax.plot(dims, ess_vals, f'{markers[name]}-', color=colors[name], lw=2,
                markersize=7, label=name)

    ax.set_xlabel('Dimensionality (DOF)', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('ESS / force eval', fontsize=FONTSIZE_LABEL)
    ax.set_yscale('log')
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.legend(fontsize=10, loc='best')
    ax.text(0.03, 0.93, '(a)', transform=ax.transAxes, fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    # ── Panel (b): Energy distribution for LJ-7 ──
    ax = axes[1]
    print("Running LJ-7 energy distributions...")
    pot_lj = LennardJonesCluster(n_atoms=7, spatial_dim=2)
    dim_lj = pot_lj.dim
    n_evals_lj = 200000

    for name in ['NH', 'NHC', 'Log-Osc', 'LOCR', 'MSLO']:
        cls, integ, params, dt = samplers[name]
        dt_use = min(dt, 0.002)  # LJ needs small dt
        try:
            qs, energies = run_sampler_nd(cls, integ, pot_lj, dim_lj, n_evals_lj, dt_use, **params)
            burn = len(energies) // 5
            e_post = energies[burn:]
            # Filter out extreme values
            e_post = e_post[np.isfinite(e_post)]
            e_post = e_post[e_post < np.percentile(e_post, 99)]
            e_post = e_post[e_post > np.percentile(e_post, 1)]
            ax.hist(e_post, bins=60, density=True, alpha=0.5, color=colors[name],
                    label=name, histtype='stepfilled', edgecolor=colors[name], linewidth=0.8)
            print(f"  {name}: <E>={np.mean(e_post):.2f}, std={np.std(e_post):.2f}")
        except Exception as e:
            print(f"  {name} failed: {e}")

    ax.set_xlabel('Energy $E$', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('$P(E)$', fontsize=FONTSIZE_LABEL)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.legend(fontsize=9, loc='best')
    ax.set_title('LJ-7 (2D) Energy Distribution', fontsize=13)
    ax.text(0.03, 0.93, '(b)', transform=ax.transAxes, fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    # ── Panel (c): Marginal variance errors on 10D Gaussian ──
    ax = axes[2]
    print("Computing marginal variance errors on 10D Gaussian...")
    dim_gauss = 10
    pot_g = NDGaussian(dim_gauss)
    n_evals_g = 500000

    width = 0.15
    x_dims = np.arange(dim_gauss)
    offset = 0

    for name in ['NH', 'NHC', 'Log-Osc', 'LOCR', 'MSLO']:
        cls, integ, params, dt = samplers[name]
        qs, _ = run_sampler_nd(cls, integ, pot_g, dim_gauss, n_evals_g, dt, **params)
        burn = len(qs) // 10
        q_post = qs[burn:]
        # Variance per dimension -- should be kT = 1.0
        var_per_dim = np.var(q_post, axis=0)
        var_err = np.abs(var_per_dim / KT - 1.0)
        ax.bar(x_dims + offset * width, var_err, width, color=colors[name],
               alpha=0.8, label=name, edgecolor='black', linewidth=0.3)
        offset += 1
        print(f"  {name}: max var error = {np.max(var_err):.4f}, mean = {np.mean(var_err):.4f}")

    ax.set_xlabel('Dimension index', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Relative variance error $|\\sigma^2/kT - 1|$', fontsize=FONTSIZE_LABEL)
    ax.set_xticks(x_dims + 2*width)
    ax.set_xticklabels([str(i) for i in range(dim_gauss)], fontsize=FONTSIZE_TICK)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.set_title('10D Gaussian Marginal Accuracy', fontsize=13)
    ax.text(0.03, 0.93, '(c)', transform=ax.transAxes, fontsize=FONTSIZE_LABEL, fontweight='bold', va='top')

    fig.suptitle('Scaling to Higher Dimensions', fontsize=FONTSIZE_TITLE, fontweight='bold', y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(os.path.dirname(__file__), 'figures', 'fig6_scaling.png')
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    make_figure()
