"""Generalized Chain Thermostat: Universal chain coupling via K_eff = xi * V'(xi).

This implements the Generalized Chain Theorem, extending NHC-style chains to
arbitrary confining thermostat potentials V(xi). The key formula:

    K_eff(xi) = xi * V'(xi)

replaces the standard NHC coupling G_j = Q_{j-1}*xi_{j-1}^2 and enables
chains with bounded friction (Tanh, Arctan, Log-Osc).

References:
    - Martyna et al. (1992) https://doi.org/10.1063/1.463940 -- NHC original
    - Watanabe & Kobayashi (2007) https://doi.org/10.1103/PhysRevE.75.040102 -- generalized thermostats
    - Parent orbit: unified-theory-007 (Master Theorem)
"""

from __future__ import annotations

import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import NamedTuple, Optional
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =====================================================================
# Inlined framework types (avoids Python 3.9 compat issues with upstream)
# =====================================================================

class ThermostatState(NamedTuple):
    q: np.ndarray
    p: np.ndarray
    xi: np.ndarray
    n_force_evals: int


# Import potentials directly (no 3.10 syntax in that file)
sys.path.insert(0, str(PROJECT_ROOT))
from research.eval.potentials import (
    Potential, HarmonicOscillator1D, DoubleWell2D,
)


# =====================================================================
# Inlined metric functions from evaluator.py
# =====================================================================

def kl_divergence_histogram(samples, potential, kT, n_bins=100):
    dim = samples.shape[1]
    if dim == 1:
        hist, edges = np.histogram(samples[:, 0], bins=n_bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        log_p = np.array([-potential.energy(np.array([c])) / kT for c in centers])
        log_p -= np.max(log_p)
        p_true = np.exp(log_p)
        p_true /= np.sum(p_true) * (centers[1] - centers[0])
        mask = (hist > 0) & (p_true > 0)
        if np.sum(mask) == 0:
            return float('inf')
        return float(np.sum(hist[mask] * np.log(hist[mask] / p_true[mask]) * (centers[1] - centers[0])))
    elif dim == 2:
        hist, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=n_bins, density=True)
        dx = xedges[1] - xedges[0]
        dy = yedges[1] - yedges[0]
        xc = 0.5 * (xedges[:-1] + xedges[1:])
        yc = 0.5 * (yedges[:-1] + yedges[1:])
        XX, YY = np.meshgrid(xc, yc, indexing='ij')
        log_p = np.zeros_like(XX)
        for i in range(len(xc)):
            for j in range(len(yc)):
                log_p[i, j] = -potential.energy(np.array([XX[i, j], YY[i, j]])) / kT
        log_p -= np.max(log_p)
        p_true = np.exp(log_p)
        p_true /= np.sum(p_true) * dx * dy
        mask = (hist > 0) & (p_true > 0)
        if np.sum(mask) == 0:
            return float('inf')
        return float(np.sum(hist[mask] * np.log(hist[mask] / p_true[mask]) * dx * dy))
    else:
        raise ValueError(f"Histogram KL only supports dim 1 or 2, got {dim}")


def autocorrelation_time(samples, max_lag=5000):
    x = samples[:, 0]
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


def effective_sample_size(samples, n_force_evals):
    n = len(samples)
    tau = autocorrelation_time(samples)
    ess = n / tau
    return {"ess": float(ess), "tau": float(tau), "ess_per_force_eval": float(ess / max(n_force_evals, 1))}


def ergodicity_score_harmonic(q_samples, p_samples, kT=1.0, omega=1.0, mass=1.0):
    sigma_q = np.sqrt(kT / omega**2)
    sigma_p = np.sqrt(mass * kT)
    ks_q_stat, _ = stats.kstest(q_samples, 'norm', args=(0, sigma_q))
    ks_p_stat, _ = stats.kstest(p_samples, 'norm', args=(0, sigma_p))
    var_q_err = abs(np.var(q_samples) - sigma_q**2) / sigma_q**2
    var_p_err = abs(np.var(p_samples) - sigma_p**2) / sigma_p**2
    q_range = 4 * sigma_q
    p_range = 4 * sigma_p
    n_grid = 20
    q_bins = np.linspace(-q_range, q_range, n_grid + 1)
    p_bins = np.linspace(-p_range, p_range, n_grid + 1)
    hist, _, _ = np.histogram2d(q_samples, p_samples, bins=[q_bins, p_bins])
    coverage = float(np.sum(hist > 0)) / (n_grid * n_grid)
    ks_component = max(0.0, 1.0 - max(ks_q_stat, ks_p_stat))
    var_component = max(0.0, 1.0 - max(var_q_err, var_p_err))
    score = (ks_component * var_component * coverage) ** (1.0 / 3.0)
    return {
        "ks_q_stat": float(ks_q_stat), "ks_p_stat": float(ks_p_stat),
        "var_q_rel_err": float(var_q_err), "var_p_rel_err": float(var_p_err),
        "coverage": coverage, "ks_component": float(ks_component),
        "var_component": float(var_component), "score": float(score),
        "ergodic": score > 0.85,
    }


def run_sampler(dynamics, potential, dt, n_force_evals, kT=1.0, mass=1.0,
                q0=None, burnin_frac=0.1, kl_checkpoints=20, rng=None,
                integrator_cls=None):
    """Run a thermostat sampler and compute all metrics."""
    if rng is None:
        rng = np.random.default_rng(42)
    if q0 is None:
        q0 = rng.normal(0, 0.5, size=potential.dim)

    state = dynamics.initial_state(q0, rng=rng)
    integrator = integrator_cls(dynamics, potential, dt, kT=kT, mass=mass)

    all_q, all_p = [], []
    kl_trace = []
    checkpoint_interval = max(n_force_evals // kl_checkpoints, 1)
    burnin_evals = int(n_force_evals * burnin_frac)
    nan_detected = False

    t_start = time.time()
    step_count = 0

    while state.n_force_evals < n_force_evals:
        state = integrator.step(state)
        step_count += 1
        if np.any(np.isnan(state.q)) or np.any(np.isnan(state.p)) or \
           np.any(np.isinf(state.q)) or np.any(np.isinf(state.p)):
            nan_detected = True
            break
        if state.n_force_evals >= burnin_evals:
            all_q.append(state.q.copy())
            all_p.append(state.p.copy())
        if state.n_force_evals > 0 and state.n_force_evals % checkpoint_interval < 3:
            if len(all_q) > 100 and potential.dim <= 2:
                q_arr = np.array(all_q)
                kl = kl_divergence_histogram(q_arr, potential, kT, n_bins=50)
                kl_trace.append((state.n_force_evals, kl))

    wall_time = time.time() - t_start

    if nan_detected or len(all_q) == 0:
        return {
            "sampler": dynamics.name, "potential": potential.name,
            "kl_divergence": float('inf'), "kl_trace": kl_trace,
            "ess_metrics": None, "ergodicity": None,
            "time_to_threshold_force_evals": None,
            "wall_seconds": wall_time, "n_samples": 0, "nan_detected": True,
        }

    q_samples = np.array(all_q)
    p_samples = np.array(all_p)
    actual_force_evals = state.n_force_evals

    kl_final = None
    if potential.dim <= 2 and len(q_samples) > 0:
        kl_final = max(0.0, kl_divergence_histogram(q_samples, potential, kT))

    ess_metrics = effective_sample_size(q_samples, actual_force_evals) if len(q_samples) > 10 else None

    ergodicity = None
    if isinstance(potential, HarmonicOscillator1D) and len(q_samples) > 100:
        ergodicity = ergodicity_score_harmonic(
            q_samples[:, 0], p_samples[:, 0], kT=kT, omega=potential.omega, mass=mass)

    ttt = None
    for n_ev, kl in kl_trace:
        if kl < 0.01:
            ttt = n_ev
            break

    return {
        "sampler": dynamics.name, "potential": potential.name,
        "kl_divergence": kl_final, "kl_trace": kl_trace,
        "ess_metrics": ess_metrics, "ergodicity": ergodicity,
        "time_to_threshold_force_evals": ttt,
        "wall_seconds": wall_time, "n_samples": len(q_samples),
    }


# =====================================================================
# Thermostat Potential Definitions
# =====================================================================

class ThermostatPotential:
    """Base class for thermostat potentials V(xi)."""
    name: str

    def V(self, xi: float) -> float:
        raise NotImplementedError

    def Vp(self, xi: float) -> float:
        """V'(xi)"""
        raise NotImplementedError

    def Vpp(self, xi: float) -> float:
        """V''(xi)"""
        raise NotImplementedError

    def g(self, xi: float, Q: float) -> float:
        """Friction function g(xi) = V'(xi)/Q"""
        return self.Vp(xi) / Q

    def K_eff(self, xi: float) -> float:
        """Effective kinetic energy: xi * V'(xi)"""
        return xi * self.Vp(xi)


class NHPotential(ThermostatPotential):
    """Standard Nose-Hoover: V(xi) = Q*xi^2/2"""
    name = "NH"

    def __init__(self, Q: float = 1.0):
        self.Q = Q

    def V(self, xi): return self.Q * xi**2 / 2
    def Vp(self, xi): return self.Q * xi
    def Vpp(self, xi): return self.Q


class LogOscPotential(ThermostatPotential):
    """Log-Osc: V(xi) = Q*log(1 + xi^2). Bounded friction |g| <= 1."""
    name = "LogOsc"

    def __init__(self, Q: float = 1.0):
        self.Q = Q

    def V(self, xi): return self.Q * np.log(1 + xi**2)
    def Vp(self, xi): return 2 * self.Q * xi / (1 + xi**2)
    def Vpp(self, xi): return 2 * self.Q * (1 - xi**2) / (1 + xi**2)**2


class TanhPotential(ThermostatPotential):
    """Tanh: V(xi) = Q*log(cosh(xi)). Bounded friction |g| < 1."""
    name = "Tanh"

    def __init__(self, Q: float = 1.0):
        self.Q = Q

    def V(self, xi): return self.Q * np.log(np.cosh(xi))
    def Vp(self, xi): return self.Q * np.tanh(xi)
    def Vpp(self, xi): return self.Q / np.cosh(xi)**2


class ArctanPotential(ThermostatPotential):
    """Arctan: V(xi) = Q*(xi*arctan(xi) - log(1+xi^2)/2). Bounded friction |g| < pi/2."""
    name = "Arctan"

    def __init__(self, Q: float = 1.0):
        self.Q = Q

    def V(self, xi): return self.Q * (xi * np.arctan(xi) - 0.5 * np.log(1 + xi**2))
    def Vp(self, xi): return self.Q * np.arctan(xi)
    def Vpp(self, xi): return self.Q / (1 + xi**2)


# =====================================================================
# Generalized Chain Thermostat
# =====================================================================

class GeneralizedChainThermostat:
    """Generalized Nose-Hoover Chain with arbitrary thermostat potentials.

    Equations of motion:
        dq/dt   = p / m
        dp/dt   = -grad_U(q) - g_1(xi_1) * p
        dxi_1/dt = (1/Q_1)(K_phys - d*kT) - g_2(xi_2)*xi_1
        dxi_j/dt = (1/Q_j)(K_eff(xi_{j-1}) - kT) - g_{j+1}(xi_{j+1})*xi_j
        dxi_M/dt = (1/Q_M)(K_eff(xi_{M-1}) - kT)

    where g_j = V_j'(xi_j)/Q_j and K_eff(xi) = xi*V'(xi).
    """

    def __init__(self, dim: int, potentials: list[ThermostatPotential],
                 kT: float = 1.0, mass: float = 1.0,
                 Q: float | list[float] = 1.0, name_override: str | None = None):
        self.dim = dim
        self.M = len(potentials)
        self.potentials = potentials
        self.kT = kT
        self.mass = mass

        if isinstance(Q, (int, float)):
            self.Q = [float(Q)] * self.M
        else:
            self.Q = list(Q)

        if name_override:
            self._name = name_override
        else:
            pot_names = "-".join(p.name for p in potentials)
            self._name = f"GenChain({pot_names},M={self.M})"

    @property
    def name(self) -> str:
        return self._name

    def initial_state(self, q0: np.ndarray, rng: np.random.Generator | None = None) -> ThermostatState:
        if rng is None:
            rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        xi0 = np.zeros(self.M)
        return ThermostatState(q0.copy(), p0, xi0, 0)

    def dqdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        return state.p / self.mass

    def dpdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        xi = state.xi
        g1 = self.potentials[0].Vp(xi[0]) / self.Q[0]
        return -grad_U - g1 * state.p

    def dxidt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        xi = state.xi
        M = self.M
        dxi = np.zeros(M)

        # Physical kinetic energy
        K_phys = np.sum(state.p**2) / self.mass

        # Level 1: driven by physical kinetic energy
        G1 = K_phys - self.dim * self.kT
        dxi[0] = G1 / self.Q[0]
        if M > 1:
            g2 = self.potentials[1].Vp(xi[1]) / self.Q[1]
            dxi[0] -= g2 * xi[0]

        # Middle levels: driven by K_eff of previous level
        for j in range(1, M - 1):
            K_eff_prev = xi[j-1] * self.potentials[j-1].Vp(xi[j-1])
            Gj = K_eff_prev - self.kT
            g_next = self.potentials[j+1].Vp(xi[j+1]) / self.Q[j+1]
            dxi[j] = Gj / self.Q[j] - g_next * xi[j]

        # Last level
        if M > 1:
            K_eff_prev = xi[M-2] * self.potentials[M-2].Vp(xi[M-2])
            GM = K_eff_prev - self.kT
            dxi[M-1] = GM / self.Q[M-1]

        return dxi


# =====================================================================
# Custom Velocity Verlet for Generalized Chains
# =====================================================================

class GeneralizedChainVV:
    """Velocity Verlet integrator adapted for generalized chain thermostats.

    Uses the analytical momentum rescaling exp(-g_1*dt/2) from Martyna et al.
    (1996), with the generalized friction g_1 = V_1'(xi_1)/Q_1.
    """

    def __init__(self, dynamics: GeneralizedChainThermostat, potential, dt: float,
                 kT: float = 1.0, mass: float = 1.0):
        self.dynamics = dynamics
        self.potential = potential
        self.dt = dt
        self.kT = kT
        self.mass = mass
        self._cached_grad_U = None

    def step(self, state: ThermostatState) -> ThermostatState:
        q, p, xi, n_evals = state
        dt = self.dt
        half_dt = 0.5 * dt
        dyn = self.dynamics

        # Get gradient
        if self._cached_grad_U is not None:
            grad_U = self._cached_grad_U
        else:
            grad_U = self.potential.gradient(q)
            n_evals += 1

        # Half-step thermostat variables
        xi_dot = dyn.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        # Half-step momenta: analytical friction rescaling + kick
        g1 = dyn.potentials[0].Vp(xi[0]) / dyn.Q[0]
        scale = np.exp(-g1 * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale
        p = p - half_dt * grad_U

        # Full-step positions
        q = q + dt * p / self.mass

        # NaN check
        if np.any(np.isnan(q)) or np.any(np.isnan(p)):
            self._cached_grad_U = None
            return ThermostatState(q, p, xi, n_evals)

        # Recompute forces
        grad_U = self.potential.gradient(q)
        n_evals += 1

        # Half-step momenta: kick + friction rescaling
        p = p - half_dt * grad_U
        g1 = dyn.potentials[0].Vp(xi[0]) / dyn.Q[0]
        scale = np.exp(-g1 * half_dt)
        scale = np.clip(scale, 1e-10, 1e10)
        p = p * scale

        # Half-step thermostat
        xi_dot = dyn.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        self._cached_grad_U = grad_U
        return ThermostatState(q, p, xi, n_evals)


# =====================================================================
# Convenience constructors
# =====================================================================

def make_nhc_chain(dim, M=3, kT=1.0, mass=1.0, Q=1.0):
    """Standard NHC with M levels."""
    pots = [NHPotential(Q=Q) for _ in range(M)]
    return GeneralizedChainThermostat(dim, pots, kT=kT, mass=mass, Q=Q,
                                      name_override=f"NHC(M={M})")

def make_logosc_chain(dim, M=3, kT=1.0, mass=1.0, Q=1.0):
    """Log-Osc chain with M levels."""
    pots = [LogOscPotential(Q=Q) for _ in range(M)]
    return GeneralizedChainThermostat(dim, pots, kT=kT, mass=mass, Q=Q,
                                      name_override=f"LogOscChain(M={M})")

def make_tanh_chain(dim, M=3, kT=1.0, mass=1.0, Q=1.0):
    """Tanh chain with M levels."""
    pots = [TanhPotential(Q=Q) for _ in range(M)]
    return GeneralizedChainThermostat(dim, pots, kT=kT, mass=mass, Q=Q,
                                      name_override=f"TanhChain(M={M})")

def make_arctan_chain(dim, M=3, kT=1.0, mass=1.0, Q=1.0):
    """Arctan chain with M levels."""
    pots = [ArctanPotential(Q=Q) for _ in range(M)]
    return GeneralizedChainThermostat(dim, pots, kT=kT, mass=mass, Q=Q,
                                      name_override=f"ArctanChain(M={M})")


# =====================================================================
# Benchmark runner
# =====================================================================

def run_benchmark(seed=42):
    """Run all 4 chain types on HO and DW benchmarks."""
    rng = np.random.default_rng(seed)
    kT = 1.0
    mass = 1.0
    M = 3
    Q = 1.0
    n_force_evals = 1_000_000
    dt_ho = 0.005
    dt_dw = 0.01

    potentials_config = [
        ("harmonic_1d", HarmonicOscillator1D(omega=1.0), dt_ho),
        ("double_well_2d", DoubleWell2D(barrier_height=1.0, y_stiffness=0.5), dt_dw),
    ]

    chain_constructors = [
        ("NHC", make_nhc_chain),
        ("LogOscChain", make_logosc_chain),
        ("TanhChain", make_tanh_chain),
        ("ArctanChain", make_arctan_chain),
    ]

    results = {}

    for pot_name, potential, dt in potentials_config:
        print(f"\n{'='*60}")
        print(f"Potential: {pot_name}")
        print(f"{'='*60}")

        for chain_name, constructor in chain_constructors:
            print(f"\n--- {chain_name}(M={M}) on {pot_name} ---")
            dynamics = constructor(dim=potential.dim, M=M, kT=kT, mass=mass, Q=Q)

            sub_rng = np.random.default_rng(rng.integers(0, 2**31))
            q0 = sub_rng.normal(0, 0.5, size=potential.dim)

            result = run_sampler(
                dynamics, potential, dt=dt,
                n_force_evals=n_force_evals, kT=kT, mass=mass,
                q0=q0, rng=sub_rng,
                integrator_cls=GeneralizedChainVV,
            )

            key = f"{chain_name}_{pot_name}"
            results[key] = result

            print(f"  KL divergence: {result['kl_divergence']}")
            if result['ess_metrics']:
                print(f"  ESS/force_eval: {result['ess_metrics']['ess_per_force_eval']:.6f}")
            if result['ergodicity']:
                erg = result['ergodicity']
                print(f"  Ergodicity score: {erg['score']:.4f} ({'ergodic' if erg['ergodic'] else 'NOT ergodic'})")
            print(f"  Wall time: {result['wall_seconds']:.2f}s")
            if result['time_to_threshold_force_evals']:
                print(f"  Time to KL<0.01: {result['time_to_threshold_force_evals']}")

    return results


# =====================================================================
# Plotting
# =====================================================================

def plot_keff_functions(save_path=None):
    """Fig 1: K_eff(xi) for all 4 thermostat types."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    xi_range = np.linspace(-4, 4, 500)
    Q = 1.0

    pots = [
        ("NH: $Q\\xi^2$", NHPotential(Q), '#1f77b4'),
        ("Log-Osc: $2Q\\xi^2/(1+\\xi^2)$", LogOscPotential(Q), '#2ca02c'),
        ("Tanh: $Q\\xi\\tanh(\\xi)$", TanhPotential(Q), '#d62728'),
        ("Arctan: $Q\\xi\\arctan(\\xi)$", ArctanPotential(Q), '#9467bd'),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300)

    for label, pot, color in pots:
        K_eff_vals = np.array([pot.K_eff(x) for x in xi_range])
        ax.plot(xi_range, K_eff_vals, label=label, color=color, linewidth=2)

    # Reference line at kT=1
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='$k_BT = 1$')

    ax.set_xlabel(r'$\xi$', fontsize=14)
    ax.set_ylabel(r'$K_{\mathrm{eff}}(\xi) = \xi \cdot V^\prime(\xi)$', fontsize=14)
    ax.set_title(r'Effective Kinetic Energy $K_{\mathrm{eff}}(\xi)$ for Generalized Chains', fontsize=16)
    ax.legend(fontsize=12, loc='upper center')
    ax.tick_params(labelsize=12)
    ax.set_ylim(-1, 12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_benchmark_comparison(results, save_path=None):
    """Fig 2: 2x2 benchmark comparison."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    chain_names = ["NHC", "LogOscChain", "TanhChain", "ArctanChain"]
    display_names = ["NHC", "Log-Osc", "Tanh", "Arctan"]
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)

    # Panel (0,0): HO KL bars
    ax = axes[0, 0]
    ho_kls = []
    for cn in chain_names:
        key = f"{cn}_harmonic_1d"
        kl = results.get(key, {}).get('kl_divergence', None)
        ho_kls.append(kl if kl is not None else 0)
    bars = ax.bar(display_names, ho_kls, color=colors)
    ax.axhline(y=0.01, color='gray', linestyle='--', linewidth=1, label='threshold')
    ax.set_ylabel('KL Divergence', fontsize=14)
    ax.set_title('1D Harmonic Oscillator: KL', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=10)

    # Panel (0,1): HO Ergodicity bars
    ax = axes[0, 1]
    ho_ergs = []
    for cn in chain_names:
        key = f"{cn}_harmonic_1d"
        erg = results.get(key, {}).get('ergodicity', None)
        ho_ergs.append(erg['score'] if erg else 0)
    bars = ax.bar(display_names, ho_ergs, color=colors)
    ax.axhline(y=0.85, color='gray', linestyle='--', linewidth=1, label='ergodic threshold')
    ax.set_ylabel('Ergodicity Score', fontsize=14)
    ax.set_title('1D Harmonic Oscillator: Ergodicity', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=10)

    # Panel (1,0): DW KL bars
    ax = axes[1, 0]
    dw_kls = []
    for cn in chain_names:
        key = f"{cn}_double_well_2d"
        kl = results.get(key, {}).get('kl_divergence', None)
        dw_kls.append(kl if kl is not None else 0)
    bars = ax.bar(display_names, dw_kls, color=colors)
    ax.axhline(y=0.01, color='gray', linestyle='--', linewidth=1, label='threshold')
    ax.set_ylabel('KL Divergence', fontsize=14)
    ax.set_title('2D Double Well: KL', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=10)

    # Panel (1,1): DW KL convergence traces
    ax = axes[1, 1]
    for cn, dn, color in zip(chain_names, display_names, colors):
        key = f"{cn}_double_well_2d"
        trace = results.get(key, {}).get('kl_trace', [])
        if trace:
            evals = [t[0] for t in trace]
            kls = [t[1] for t in trace]
            ax.plot(evals, kls, label=dn, color=color, linewidth=2)
    ax.axhline(y=0.01, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Force Evaluations', fontsize=14)
    ax.set_ylabel('KL Divergence', fontsize=14)
    ax.set_title('2D Double Well: KL Convergence', fontsize=14)
    ax.set_yscale('log')
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    ORBIT_DIR = Path(__file__).parent
    FIG_DIR = ORBIT_DIR / "figures"
    FIG_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("GENERALIZED CHAIN THERMOSTAT BENCHMARK")
    print("=" * 60)

    # Fig 1: K_eff functions
    print("\n--- Generating K_eff figure ---")
    plot_keff_functions(save_path=str(FIG_DIR / "keff_functions.png"))

    # Run benchmarks
    results = run_benchmark(seed=42)

    # Fig 2: Benchmark comparison
    print("\n--- Generating benchmark comparison figure ---")
    plot_benchmark_comparison(results, save_path=str(FIG_DIR / "benchmark_comparison.png"))

    # Save results as JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    results_path = ORBIT_DIR / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {results_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Chain':<18} {'HO KL':<10} {'HO Erg':<10} {'DW KL':<10}")
    print("-" * 48)
    for cn, dn in zip(["NHC", "LogOscChain", "TanhChain", "ArctanChain"],
                       ["NHC", "Log-Osc", "Tanh", "Arctan"]):
        ho_key = f"{cn}_harmonic_1d"
        dw_key = f"{cn}_double_well_2d"
        ho_kl = results.get(ho_key, {}).get('kl_divergence', float('inf'))
        ho_erg = results.get(ho_key, {}).get('ergodicity', {})
        ho_erg_score = ho_erg.get('score', 0) if ho_erg else 0
        dw_kl = results.get(dw_key, {}).get('kl_divergence', float('inf'))
        ho_kl_s = f"{ho_kl:.4f}" if ho_kl != float('inf') else "inf"
        dw_kl_s = f"{dw_kl:.4f}" if dw_kl != float('inf') else "inf"
        print(f"{dn:<18} {ho_kl_s:<10} {ho_erg_score:<10.4f} {dw_kl_s:<10}")
