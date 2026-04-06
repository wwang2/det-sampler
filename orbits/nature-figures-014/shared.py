"""Shared utilities for Nature-quality figures.

Contains all sampler classes (LogOsc, MultiScale, NHCTail),
simulation runners, KL computation, and style constants.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from research.eval.potentials import (
    HarmonicOscillator1D, DoubleWell2D, GaussianMixture2D, Rosenbrock2D
)
from research.eval.baselines import NoseHooverChain, NoseHoover
from research.eval.integrators import ThermostatState, VelocityVerletThermostat

# ── Style constants (from research/style.md + task spec) ──
COLOR_NH   = '#1f77b4'   # blue - secondary reference
COLOR_NHC  = '#ff7f0e'   # orange - PRIMARY baseline
COLOR_LO   = '#2ca02c'   # green - Log-Osc
COLOR_MS   = '#d62728'   # red - MultiScale
COLOR_NHCT = '#9467bd'   # purple - NHCTail

FONTSIZE_LABEL = 14
FONTSIZE_TICK  = 12
FONTSIZE_TITLE = 16
FONTSIZE_ANNOT = 10
DPI = 300

SEEDS = [42, 123, 7, 999, 314]

# ── Friction function ──
def g_func(xi_val):
    """Bounded friction: g(xi) = 2*xi/(1+xi^2), in [-1, 1]."""
    xi2 = xi_val * xi_val
    if np.isscalar(xi_val):
        if xi2 > 1e30:
            return 2.0 / xi_val if abs(xi_val) > 0 else 0.0
        return 2.0 * xi_val / (1.0 + xi2)
    # array case
    result = np.zeros_like(np.asarray(xi_val, dtype=float))
    big = xi2 > 1e30
    small = ~big
    if np.any(small):
        result[small] = 2.0 * xi_val[small] / (1.0 + xi2[small])
    if np.any(big):
        result[big] = np.where(xi_val[big] != 0, 2.0 / xi_val[big], 0.0)
    return result


# ── LogOsc Thermostat ──
class LogOscThermostat:
    name = "log_osc"

    def __init__(self, dim, kT=1.0, mass=1.0, Q=0.5):
        self.dim, self.kT, self.mass, self.Q = dim, kT, mass, Q

    def initial_state(self, q0, rng=None):
        if rng is None: rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        return ThermostatState(q0.copy(), p0, np.array([0.0]), 0)

    def dqdt(self, state, grad_U): return state.p / self.mass
    def dpdt(self, state, grad_U):
        xi = state.xi[0]
        return -grad_U - g_func(xi) * state.p
    def dxidt(self, state, grad_U):
        K = np.sum(state.p**2) / self.mass
        return np.array([(K - self.dim * self.kT) / self.Q])


class LogOscVelocityVerlet:
    """Custom VV for Log-Osc using exp(-g(xi)*dt/2) rescaling."""
    def __init__(self, dynamics, potential, dt, kT=1.0, mass=1.0):
        self.dynamics = dynamics
        self.potential = potential
        self.dt = dt
        self.kT = kT
        self.mass = mass
        self._cached_grad_U = None

    def step(self, state):
        q, p, xi, n_evals = state
        dt = self.dt
        half_dt = 0.5 * dt

        if self._cached_grad_U is not None:
            grad_U = self._cached_grad_U
        else:
            grad_U = self.potential.gradient(q)
            n_evals += 1

        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        gxi = g_func(xi[0])
        scale = np.clip(np.exp(-gxi * half_dt), 1e-10, 1e10)
        p = p * scale
        p = p - half_dt * grad_U

        q = q + dt * p / self.mass

        if np.any(np.isnan(q)) or np.any(np.isnan(p)):
            self._cached_grad_U = None
            return ThermostatState(q, p, xi, n_evals)

        grad_U = self.potential.gradient(q)
        n_evals += 1

        p = p - half_dt * grad_U
        gxi = g_func(xi[0])
        scale = np.clip(np.exp(-gxi * half_dt), 1e-10, 1e10)
        p = p * scale

        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        self._cached_grad_U = grad_U
        return ThermostatState(q, p, xi, n_evals)


# ── MultiScale Log-Osc ──
class MultiScaleLogOsc:
    name = "multi_scale_log_osc"

    def __init__(self, dim, kT=1.0, mass=1.0, Qs=None):
        self.dim, self.kT, self.mass = dim, kT, mass
        self.Qs = list(Qs) if Qs else [0.1, 0.7, 10.0]
        self.n_thermo = len(self.Qs)

    def initial_state(self, q0, rng=None):
        if rng is None: rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        xi0 = np.zeros(self.n_thermo)
        return ThermostatState(q0.copy(), p0, xi0, 0)

    def dqdt(self, state, grad_U): return state.p / self.mass
    def dpdt(self, state, grad_U):
        total_g = sum(g_func(state.xi[i]) for i in range(self.n_thermo))
        return -grad_U - total_g * state.p
    def dxidt(self, state, grad_U):
        K = np.sum(state.p**2) / self.mass
        drive = K - self.dim * self.kT
        return np.array([drive / Q for Q in self.Qs])


class MultiScaleLogOscVerlet:
    def __init__(self, dynamics, potential, dt, kT=1.0, mass=1.0):
        self.dynamics = dynamics
        self.potential = potential
        self.dt = dt
        self.kT = kT
        self.mass = mass
        self._cached_grad_U = None

    def step(self, state):
        q, p, xi, n_evals = state
        dt = self.dt
        half_dt = 0.5 * dt

        if self._cached_grad_U is not None:
            grad_U = self._cached_grad_U
        else:
            grad_U = self.potential.gradient(q)
            n_evals += 1

        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        total_g = sum(g_func(xi[i]) for i in range(len(xi)))
        scale = np.clip(np.exp(-total_g * half_dt), 1e-10, 1e10)
        p = p * scale
        p = p - half_dt * grad_U

        q = q + dt * p / self.mass

        if np.any(np.isnan(q)) or np.any(np.isnan(p)):
            self._cached_grad_U = None
            return ThermostatState(q, p, xi, n_evals)

        grad_U = self.potential.gradient(q)
        n_evals += 1

        p = p - half_dt * grad_U
        total_g = sum(g_func(xi[i]) for i in range(len(xi)))
        scale = np.clip(np.exp(-total_g * half_dt), 1e-10, 1e10)
        p = p * scale

        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        self._cached_grad_U = grad_U
        return ThermostatState(q, p, xi, n_evals)


# ── NHCTail: MultiScale Log-Osc + NHC chain tails on each thermostat ──
class NHCTailThermostat:
    """MultiScale Log-Osc with NHC chain tails.

    Each of the N log-osc thermostats gets an M-length NHC tail.
    The first variable in each group uses log-osc friction g(xi),
    and subsequent chain members use standard NHC coupling.

    xi layout: [xi_1_0, xi_1_1, ..., xi_1_M, xi_2_0, ..., xi_N_M]
    Total xi dimension: N * (M+1) where N = len(Qs), M = chain tail length
    """
    name = "nhc_tail"

    def __init__(self, dim, kT=1.0, mass=1.0, Qs=None, chain_length=2, Q_chain=1.0):
        self.dim, self.kT, self.mass = dim, kT, mass
        self.Qs = list(Qs) if Qs else [0.1, 0.7, 10.0]
        self.N = len(self.Qs)
        self.M = chain_length  # tail length per thermostat
        self.Q_chain = Q_chain
        self.n_xi = self.N * (1 + self.M)

    def initial_state(self, q0, rng=None):
        if rng is None: rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        xi0 = np.zeros(self.n_xi)
        return ThermostatState(q0.copy(), p0, xi0, 0)

    def dqdt(self, state, grad_U): return state.p / self.mass

    def dpdt(self, state, grad_U):
        # Total friction from all N primary thermostats
        total_g = 0.0
        for k in range(self.N):
            idx = k * (1 + self.M)
            total_g += g_func(state.xi[idx])
        return -grad_U - total_g * state.p

    def dxidt(self, state, grad_U):
        xi = state.xi
        K = np.sum(state.p**2) / self.mass
        drive = K - self.dim * self.kT
        dxi = np.zeros(self.n_xi)
        stride = 1 + self.M

        for k in range(self.N):
            base = k * stride
            # Primary thermostat: driven by kinetic energy mismatch
            dxi[base] = drive / self.Qs[k]
            if self.M > 0:
                dxi[base] -= xi[base + 1] * xi[base]

            # Chain tail members
            for j in range(1, self.M + 1):
                idx = base + j
                if j == 1:
                    # Driven by effective KE of primary log-osc variable
                    xi_b2 = xi[base]**2
                    if xi_b2 > 1e30:
                        eff_ke = 2.0 * self.Qs[k]
                    else:
                        eff_ke = 2.0 * self.Qs[k] * xi_b2 / (1.0 + xi_b2)
                    Gj = eff_ke - self.kT
                else:
                    Gj = self.Q_chain * xi[idx - 1]**2 - self.kT
                dxi[idx] = Gj / self.Q_chain
                if j < self.M:
                    dxi[idx] -= xi[idx + 1] * xi[idx]

        return dxi


class NHCTailVerlet:
    def __init__(self, dynamics, potential, dt, kT=1.0, mass=1.0):
        self.dynamics = dynamics
        self.potential = potential
        self.dt = dt
        self.kT = kT
        self.mass = mass
        self._cached_grad_U = None

    def step(self, state):
        q, p, xi, n_evals = state
        dt = self.dt
        half_dt = 0.5 * dt

        if self._cached_grad_U is not None:
            grad_U = self._cached_grad_U
        else:
            grad_U = self.potential.gradient(q)
            n_evals += 1

        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        # Total friction from primary thermostats
        N = self.dynamics.N
        stride = 1 + self.dynamics.M
        total_g = sum(g_func(xi[k * stride]) for k in range(N))
        scale = np.clip(np.exp(-total_g * half_dt), 1e-10, 1e10)
        p = p * scale
        p = p - half_dt * grad_U

        q = q + dt * p / self.mass

        if np.any(np.isnan(q)) or np.any(np.isnan(p)):
            self._cached_grad_U = None
            return ThermostatState(q, p, xi, n_evals)

        grad_U = self.potential.gradient(q)
        n_evals += 1

        p = p - half_dt * grad_U
        total_g = sum(g_func(xi[k * stride]) for k in range(N))
        scale = np.clip(np.exp(-total_g * half_dt), 1e-10, 1e10)
        p = p * scale

        xi_dot = self.dynamics.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        self._cached_grad_U = grad_U
        return ThermostatState(q, p, xi, n_evals)


# ── Generic simulation runner ──
def run_trajectory(sampler_type, integrator_type, potential, dt, n_force_evals,
                   seed=42, q0=None, collect_xi=False, **sampler_kwargs):
    """Run a sampler and collect (q, p) trajectory. Returns dict with arrays."""
    rng = np.random.default_rng(seed)
    dim = potential.dim
    dyn = sampler_type(dim=dim, **sampler_kwargs)
    if q0 is None:
        q0 = rng.normal(0, 0.5, size=dim)
    state = dyn.initial_state(q0, rng=rng)
    integrator = integrator_type(dyn, potential, dt=dt)

    qs = []
    ps = []
    xis = [] if collect_xi else None

    while state.n_force_evals < n_force_evals:
        qs.append(state.q.copy())
        ps.append(state.p.copy())
        if collect_xi:
            xis.append(state.xi.copy())
        state = integrator.step(state)
        if np.any(np.isnan(state.q)) or np.any(np.isnan(state.p)):
            break

    result = {
        'q': np.array(qs),
        'p': np.array(ps),
        'n_force_evals': state.n_force_evals,
    }
    if collect_xi:
        result['xi'] = np.array(xis)
    return result


# ── KL divergence computation ──
def compute_kl_trace(qs, potential, kT=1.0, n_checkpoints=50, burnin_frac=0.1):
    """Compute KL divergence at multiple points along the trajectory.

    Uses histogram-based KL for 1D/2D potentials.
    Returns (frac_array, kl_array) where frac is fraction of total samples used.
    """
    n_total = len(qs)
    burnin = int(n_total * burnin_frac)
    qs_post = qs[burnin:]
    n_post = len(qs_post)

    checkpoints = np.logspace(np.log10(max(100, n_post // 100)),
                              np.log10(n_post), n_checkpoints).astype(int)
    checkpoints = np.unique(np.clip(checkpoints, 100, n_post))

    evals_at_check = burnin + checkpoints  # approximate force evals
    kl_vals = []

    dim = qs.shape[1] if qs.ndim > 1 else 1

    for cp in checkpoints:
        samples = qs_post[:cp]
        if dim == 1:
            kl = _kl_1d(samples.ravel(), potential, kT)
        else:
            kl = _kl_2d(samples, potential, kT)
        kl_vals.append(kl)

    return evals_at_check, np.array(kl_vals)


def _kl_1d(samples, potential, kT, n_bins=200):
    """1D KL divergence via histogram."""
    lo, hi = np.percentile(samples, [0.5, 99.5])
    margin = 0.2 * (hi - lo)
    lo -= margin
    hi += margin
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dx = edges[1] - edges[0]

    counts, _ = np.histogram(samples, bins=edges)
    p_sample = counts / (len(samples) * dx)
    p_sample = np.maximum(p_sample, 1e-30)

    # True density
    log_p_true = np.array([-potential.energy(np.array([c])) / kT for c in centers])
    log_p_true -= np.max(log_p_true)  # shift for stability
    p_true = np.exp(log_p_true)
    p_true /= np.sum(p_true) * dx
    p_true = np.maximum(p_true, 1e-30)

    # KL(true || sample)
    kl = np.sum(p_true * np.log(p_true / p_sample) * dx)
    return max(kl, 0.0)


def _kl_2d(samples, potential, kT, n_bins=60):
    """2D KL divergence via histogram."""
    lox, hix = np.percentile(samples[:, 0], [0.5, 99.5])
    loy, hiy = np.percentile(samples[:, 1], [0.5, 99.5])
    mx = 0.2 * (hix - lox)
    my = 0.2 * (hiy - loy)

    edges_x = np.linspace(lox - mx, hix + mx, n_bins + 1)
    edges_y = np.linspace(loy - my, hiy + my, n_bins + 1)
    dx = edges_x[1] - edges_x[0]
    dy = edges_y[1] - edges_y[0]

    counts, _, _ = np.histogram2d(samples[:, 0], samples[:, 1],
                                   bins=[edges_x, edges_y])
    p_sample = counts / (len(samples) * dx * dy)
    p_sample = np.maximum(p_sample, 1e-30)

    cx = 0.5 * (edges_x[:-1] + edges_x[1:])
    cy = 0.5 * (edges_y[:-1] + edges_y[1:])
    CX, CY = np.meshgrid(cx, cy, indexing='ij')

    log_p_true = np.zeros_like(CX)
    for i in range(len(cx)):
        for j in range(len(cy)):
            log_p_true[i, j] = -potential.energy(np.array([CX[i, j], CY[i, j]])) / kT
    log_p_true -= np.max(log_p_true)
    p_true = np.exp(log_p_true)
    p_true /= np.sum(p_true) * dx * dy
    p_true = np.maximum(p_true, 1e-30)

    kl = np.sum(p_true * np.log(p_true / p_sample) * dx * dy)
    return max(kl, 0.0)


def get_potential(name):
    """Get potential by short name."""
    pots = {
        'HO': HarmonicOscillator1D(omega=1.0),
        'DW': DoubleWell2D(),
        'GMM': GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5),
        'RB': Rosenbrock2D(a=0.0, b=5.0),
    }
    return pots[name]


def get_sampler_config(sampler_name, pot_name):
    """Get (sampler_cls, integrator_cls, dt, sampler_kwargs) for a given combination."""
    dt_map = {
        ('NHC', 'HO'):  0.005,
        ('NHC', 'DW'):  0.01,
        ('NHC', 'GMM'): 0.01,
        ('NHC', 'RB'):  0.01,
        ('LogOsc', 'HO'):  0.005,
        ('LogOsc', 'DW'):  0.035,
        ('LogOsc', 'GMM'): 0.03,
        ('LogOsc', 'RB'):  0.02,
        ('MultiScale', 'HO'):  0.005,
        ('MultiScale', 'DW'):  0.035,
        ('MultiScale', 'GMM'): 0.03,
        ('MultiScale', 'RB'):  0.02,
        ('NHCTail', 'HO'):  0.005,
        ('NHCTail', 'DW'):  0.055,
        ('NHCTail', 'GMM'): 0.03,
        ('NHCTail', 'RB'):  0.03,
    }

    dt = dt_map.get((sampler_name, pot_name), 0.01)

    if sampler_name == 'NHC':
        return NoseHooverChain, VelocityVerletThermostat, dt, {'chain_length': 3, 'Q': 1.0}
    elif sampler_name == 'LogOsc':
        return LogOscThermostat, LogOscVelocityVerlet, dt, {'Q': 0.5}
    elif sampler_name == 'MultiScale':
        return MultiScaleLogOsc, MultiScaleLogOscVerlet, dt, {'Qs': [0.1, 0.7, 10.0]}
    elif sampler_name == 'NHCTail':
        return NHCTailThermostat, NHCTailVerlet, dt, {'Qs': [0.1, 0.7, 10.0], 'chain_length': 2, 'Q_chain': 1.0}
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")


SAMPLER_LABELS = {
    'NHC': 'NHC (M=3)',
    'LogOsc': 'Log-Osc',
    'MultiScale': 'MultiScale',
    'NHCTail': 'NHCTail',
}
SAMPLER_COLORS = {
    'NHC': COLOR_NHC,
    'LogOsc': COLOR_LO,
    'MultiScale': COLOR_MS,
    'NHCTail': COLOR_NHCT,
}
