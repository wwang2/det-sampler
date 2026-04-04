"""High-dimensional scaling study for deterministic thermostat samplers.

Tests NH, NHC(M=3), Multi-Scale Log-Osc, and LOCR on:
1. LJ-7  (2D, 14 DOF)
2. LJ-13 (3D, 39 DOF)
3. 20D Correlated Gaussian (condition number 10000)
4. 10D Gaussian Mixture (3 modes, separation 5, sigma 0.5)

Metrics (high-D adapted):
- Energy distribution KS test
- Marginal variance error (for Gaussian systems)
- Radial distribution function g(r) (for LJ systems)
- Autocorrelation time of energy
- ESS per force eval
- Mode visitation count (for mixture models)

References:
- Nose (1984), Hoover (1985) -- NH thermostat
- Martyna et al. (1992) -- NHC
- Fukuda & Nakamura (2002) -- multiple thermostats
- Wales & Doye (1997) -- LJ cluster energy landscapes
  https://doi.org/10.1021/jp970984n
"""

import sys
import os
import time
import numpy as np
from scipy import stats
from typing import Optional

# Add project root to path
try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _this_dir = os.path.abspath('.')
project_root = os.path.abspath(os.path.join(_this_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from research.eval.integrators import ThermostatState, VelocityVerletThermostat
from research.eval.potentials import LennardJonesCluster, Potential
from research.eval.baselines import NoseHoover, NoseHooverChain

# =========================================================================
# Self-contained sampler implementations (copied from parent orbits to
# avoid import issues with hyphenated directory names)
# =========================================================================

def g_func(xi_val):
    """Bounded friction: g(xi) = 2*xi/(1+xi^2). Range: [-1, 1]."""
    return 2.0 * xi_val / (1.0 + xi_val**2)

def eff_ke(Q_j, xi_j):
    """Effective kinetic energy of xi_j in the log measure."""
    xi2 = xi_j * xi_j
    if xi2 > 1e30:
        return 2.0 * Q_j
    return 2.0 * Q_j * xi2 / (1.0 + xi2)


class MultiScaleLogOsc:
    """N Log-Osc thermostats at different timescales.
    Copied from orbits/log-osc-multiT-005/solution.py.
    """
    name = "multi_scale_log_osc"

    def __init__(self, dim, kT=1.0, mass=1.0,
                 Q_fast=0.1, Q_med=1.0, Q_slow=10.0, Qs=None):
        self.dim = dim
        self.kT = kT
        self.mass = mass
        if Qs is not None:
            self.Qs = list(Qs)
        else:
            self.Qs = [Q_fast, Q_med, Q_slow]
        self.n_thermo = len(self.Qs)

    def initial_state(self, q0, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        xi0 = np.zeros(self.n_thermo)
        return ThermostatState(q0.copy(), p0, xi0, 0)

    def dqdt(self, state, grad_U):
        return state.p / self.mass

    def dpdt(self, state, grad_U):
        total_friction = sum(g_func(state.xi[i]) for i in range(self.n_thermo))
        return -grad_U - total_friction * state.p

    def dxidt(self, state, grad_U):
        kinetic = np.sum(state.p**2) / self.mass
        drive = kinetic - self.dim * self.kT
        return np.array([drive / Q for Q in self.Qs])


class MultiScaleLogOscVerlet:
    """Velocity Verlet for Multi-Scale Log-Osc.
    Copied from orbits/log-osc-multiT-005/solution.py.
    """
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


class LogOscChainRotation:
    """LOCR thermostat -- copied from log-osc-chain-002 for portability."""
    name = "log_osc_chain_rotation"

    def __init__(self, dim, chain_length=2, kT=1.0, mass=1.0,
                 Q=1.0, alpha=0.0):
        self.dim = dim
        self.M = chain_length
        self.kT = kT
        self.mass = mass
        self.alpha = alpha
        if isinstance(Q, (int, float)):
            self.Q = [float(Q)] * chain_length
        else:
            self.Q = list(Q)

    def initial_state(self, q0, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        xi0 = np.zeros(self.M)
        return ThermostatState(q0.copy(), p0, xi0, 0)

    def dqdt(self, state, grad_U):
        return state.p / self.mass

    def dpdt(self, state, grad_U):
        xi1 = state.xi[0]
        return -grad_U - g_func(xi1) * state.p

    def dxidt(self, state, grad_U):
        xi = state.xi
        M = self.M
        Q = self.Q
        kT = self.kT
        alpha = self.alpha
        dxi = np.zeros(M)
        kinetic = np.sum(state.p ** 2) / self.mass

        G0 = kinetic - self.dim * kT
        dxi[0] = G0 / Q[0]
        if M > 1:
            dxi[0] -= xi[1] * xi[0]
        if M > 1 and alpha != 0:
            dxi[0] += alpha * g_func(xi[1])

        for j in range(1, M):
            if j == 1:
                Gj = eff_ke(Q[0], xi[0])
            else:
                Gj = Q[j - 1] * xi[j - 1] ** 2
            dxi[j] = (Gj - kT) / Q[j]
            if j < M - 1:
                dxi[j] -= xi[j + 1] * xi[j]
            if alpha != 0:
                dxi[j] -= alpha * g_func(xi[j - 1])
                if j < M - 1:
                    dxi[j] += alpha * g_func(xi[j + 1])
        return dxi


class LOCRIntegrator:
    """Velocity Verlet for LOCR -- copied from log-osc-chain-002."""

    def __init__(self, dynamics, potential, dt, kT=1.0, mass=1.0):
        self.dynamics = dynamics
        self.potential = potential
        self.dt = dt
        self.kT = kT
        self.mass = mass
        self._cached_grad_U = None

    def _half_step_chain(self, state, grad_U, half_dt, reverse=False):
        q, p, xi, n_evals = state
        xi = xi.copy()
        M = self.dynamics.M
        Q = self.dynamics.Q
        kT = self.dynamics.kT
        alpha = self.dynamics.alpha
        dim = self.dynamics.dim
        mass = self.dynamics.mass

        indices = range(0, M) if reverse else range(M - 1, -1, -1)
        for j in indices:
            if j == 0:
                kinetic = np.sum(p ** 2) / mass
                force = (kinetic - dim * kT) / Q[0]
            else:
                if j == 1:
                    Gj = eff_ke(Q[0], xi[0])
                else:
                    Gj = Q[j - 1] * xi[j - 1] ** 2
                force = (Gj - kT) / Q[j]
            if j < M - 1:
                force -= xi[j + 1] * xi[j]
            if alpha != 0:
                if j == 0 and M > 1:
                    force += alpha * g_func(xi[1])
                elif j > 0:
                    force -= alpha * g_func(xi[j - 1])
                    if j < M - 1:
                        force += alpha * g_func(xi[j + 1])
            xi[j] = xi[j] + half_dt * force
        return ThermostatState(q, p, xi, n_evals)

    def step(self, state):
        q, p, xi, n_evals = state
        dt = self.dt
        half_dt = 0.5 * dt

        if self._cached_grad_U is not None:
            grad_U = self._cached_grad_U
        else:
            grad_U = self.potential.gradient(q)
            n_evals += 1

        state_tmp = self._half_step_chain(
            ThermostatState(q, p, xi, n_evals), grad_U, half_dt, reverse=False)
        xi = state_tmp.xi

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

        state_tmp = self._half_step_chain(
            ThermostatState(q, p, xi, n_evals), grad_U, half_dt, reverse=True)
        xi = state_tmp.xi

        self._cached_grad_U = grad_U
        return ThermostatState(q, p, xi, n_evals)


# =========================================================================
# Additional High-D Potentials
# =========================================================================

class CorrelatedGaussianND(Potential):
    """N-D correlated Gaussian: U(q) = 0.5 * q^T Sigma^{-1} q.

    Sigma has eigenvalues spanning [lam_min, lam_max] (log-spaced).
    Tests whether thermostats handle stiff systems (large condition number).

    Analytical canonical distribution at kT:
        P(q) ~ N(0, kT * Sigma)
        var(q_i) = kT * eigenvalue_i (in eigenbasis)
    """
    name = "correlated_gaussian"

    def __init__(self, dim=20, lam_min=0.01, lam_max=100.0, seed=42):
        self.dim = dim
        self.lam_min = lam_min
        self.lam_max = lam_max
        # Log-spaced eigenvalues
        eigenvalues = np.logspace(np.log10(lam_min), np.log10(lam_max), dim)
        self.eigenvalues = eigenvalues
        # Random orthogonal rotation
        rng = np.random.default_rng(seed)
        H = rng.standard_normal((dim, dim))
        Q_orth, _ = np.linalg.qr(H)
        self.rotation = Q_orth
        # Sigma = Q diag(lam) Q^T
        self.Sigma = Q_orth @ np.diag(eigenvalues) @ Q_orth.T
        self.Sigma_inv = Q_orth @ np.diag(1.0 / eigenvalues) @ Q_orth.T
        # Precompute for marginal variance checks
        self.expected_variances = np.diag(self.Sigma)  # var(q_i) = kT * Sigma_ii at kT=1

    def energy(self, q):
        return 0.5 * float(q @ self.Sigma_inv @ q)

    def gradient(self, q):
        return self.Sigma_inv @ q


class GaussianMixtureND(Potential):
    """N-D Gaussian mixture: U(q) = -log(sum_k w_k * N(q; mu_k, sigma^2 I)).

    Multi-modal distribution in high dimensions.
    """
    name = "gaussian_mixture_nd"

    def __init__(self, dim=10, n_modes=3, separation=5.0, sigma=0.5, seed=42):
        self.dim = dim
        self.n_modes = n_modes
        self.sigma = sigma
        self.separation = separation
        # Place modes along first few principal axes for reproducibility
        rng = np.random.default_rng(seed)
        self.centers = np.zeros((n_modes, dim))
        for k in range(n_modes):
            direction = np.zeros(dim)
            direction[k % dim] = 1.0
            self.centers[k] = separation * direction * (1 if k % 2 == 0 else -1)
            # Add small random offset in other dims
            self.centers[k] += rng.normal(0, 0.1, dim)
        self.weights = np.ones(n_modes) / n_modes
        self.log_norm = -0.5 * dim * np.log(2 * np.pi * sigma**2)

    def _component_log_densities(self, q):
        """Log densities for each component."""
        diffs = self.centers - q[np.newaxis, :]
        exponents = -0.5 * np.sum(diffs**2, axis=1) / self.sigma**2
        return np.log(self.weights) + self.log_norm + exponents

    def energy(self, q):
        log_densities = self._component_log_densities(q)
        max_log = np.max(log_densities)
        total = max_log + np.log(np.sum(np.exp(log_densities - max_log)))
        return -total

    def gradient(self, q):
        log_densities = self._component_log_densities(q)
        max_log = np.max(log_densities)
        weights = np.exp(log_densities - max_log)
        total = np.sum(weights)
        weights /= total
        # grad U = -sum_k w_k * (mu_k - q) / sigma^2
        diffs = self.centers - q[np.newaxis, :]
        return -np.sum(weights[:, np.newaxis] * diffs / self.sigma**2, axis=0)


# =========================================================================
# LJ cluster initialization -- find a reasonable starting config
# =========================================================================

def init_lj_config(n_atoms, spatial_dim, rng=None):
    """Initialize LJ cluster near a local minimum."""
    if rng is None:
        rng = np.random.default_rng(42)

    if spatial_dim == 2 and n_atoms == 7:
        # Hexagonal arrangement for LJ-7 2D
        pos = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.866],
            [-0.5, 0.866],
            [-1.0, 0.0],
            [-0.5, -0.866],
            [0.5, -0.866],
        ]) * 1.12  # near LJ minimum at r=2^(1/6)~1.122
    elif spatial_dim == 3 and n_atoms == 13:
        # Icosahedral arrangement for LJ-13 3D
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        # 12 vertices of icosahedron + center
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
        ])
        verts = verts / np.linalg.norm(verts[0])  # normalize to unit distance
        verts *= 1.12  # scale to near LJ minimum
        pos = np.vstack([np.zeros(3), verts])
    else:
        # Generic: place on a grid with spacing near LJ minimum
        pos = rng.normal(0, 1.0, (n_atoms, spatial_dim))
        # scale so mean nearest-neighbor distance ~ 1.12
        from scipy.spatial.distance import pdist
        dists = pdist(pos)
        mean_d = np.mean(np.sort(dists)[:n_atoms])
        pos *= 1.12 / mean_d

    # Center of mass at origin
    pos -= pos.mean(axis=0)
    return pos.flatten()


# =========================================================================
# Metrics for high-D systems
# =========================================================================

def compute_energy_trace(q_traj, p_traj, potential, mass=1.0):
    """Compute total energy H = U(q) + K(p) for each frame."""
    n_frames = len(q_traj)
    energies = np.zeros(n_frames)
    for i in range(n_frames):
        U = potential.energy(q_traj[i])
        K = 0.5 * np.sum(p_traj[i]**2) / mass
        energies[i] = U + K
    return energies


def energy_ks_test(energies, dim, kT=1.0):
    """KS test of energy distribution against Boltzmann.

    For a system with dim DOF, E ~ Gamma(dim/2, scale=kT) shifted by U_min.
    In practice, we compare the CDF of (E - E_min) against Gamma(dim/2, kT).
    Since U_min is unknown, we use the empirical distribution shape test.

    Returns KS statistic (lower = better match to Boltzmann).
    """
    E = energies - np.min(energies)
    E = E[E > 0]
    if len(E) < 100:
        return 1.0
    # Fit gamma distribution and compute KS
    shape_fit, _, scale_fit = stats.gamma.fit(E, floc=0)
    ks_stat, _ = stats.kstest(E, 'gamma', args=(shape_fit, 0, scale_fit))
    return ks_stat


def marginal_variance_error(q_traj, expected_vars):
    """Relative error of marginal variances vs theory.

    Returns max relative error across all dimensions.
    """
    q_arr = np.array(q_traj)
    empirical_vars = np.var(q_arr, axis=0)
    rel_errors = np.abs(empirical_vars - expected_vars) / (expected_vars + 1e-10)
    return np.max(rel_errors), np.mean(rel_errors), rel_errors


def autocorrelation_time(x, max_lag=None):
    """Integrated autocorrelation time using initial positive sequence estimator.

    Follows Geyer (1992) initial positive sequence method.
    """
    n = len(x)
    if max_lag is None:
        max_lag = min(n // 2, 10000)
    x_centered = x - np.mean(x)
    var = np.var(x)
    if var < 1e-15:
        return float('inf')

    # FFT-based autocorrelation
    fft_x = np.fft.rfft(x_centered, n=2*n)
    acf = np.fft.irfft(fft_x * np.conj(fft_x))[:max_lag] / (n * var)

    # Integrated autocorrelation time: tau = 1 + 2 * sum(acf[1:])
    # Cut when acf first goes negative (initial positive sequence)
    tau = 1.0
    for k in range(1, max_lag):
        if acf[k] < 0:
            break
        tau += 2.0 * acf[k]
    return tau


def ess_per_force_eval(energies, n_force_evals):
    """Effective sample size per force evaluation."""
    tau = autocorrelation_time(energies)
    n_eff = len(energies) / tau
    return n_eff / n_force_evals


def radial_distribution_function(q_traj, n_atoms, spatial_dim, r_max=3.0, n_bins=100):
    """Compute g(r) from trajectory frames.

    Returns r_centers, g_r arrays.
    """
    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = r_edges[1] - r_edges[0]
    hist = np.zeros(n_bins)

    n_frames = len(q_traj)
    sample_every = max(1, n_frames // 5000)  # subsample for speed

    count = 0
    for frame_idx in range(0, n_frames, sample_every):
        q = q_traj[frame_idx]
        pos = q.reshape(n_atoms, spatial_dim)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r = np.linalg.norm(pos[i] - pos[j])
                if r < r_max:
                    bin_idx = int(r / dr)
                    if bin_idx < n_bins:
                        hist[bin_idx] += 1
        count += 1

    # Normalize: ideal gas density
    n_pairs = n_atoms * (n_atoms - 1) / 2
    if spatial_dim == 2:
        shell_vol = np.pi * (r_edges[1:]**2 - r_edges[:-1]**2)
    else:
        shell_vol = (4.0/3.0) * np.pi * (r_edges[1:]**3 - r_edges[:-1]**3)

    # Approximate volume from typical cluster size
    if count > 0:
        # Just normalize to get shape; absolute normalization would need box volume
        norm = hist.sum() * dr if hist.sum() > 0 else 1.0
        g_r = hist / (norm + 1e-10) * n_bins
    else:
        g_r = np.zeros(n_bins)

    return r_centers, g_r


def mode_visitation(q_traj, centers, threshold=2.0):
    """Count transitions between modes.

    A transition occurs when the nearest mode changes.
    Returns dict with n_transitions, visits_per_mode.
    """
    n_frames = len(q_traj)
    assignments = np.zeros(n_frames, dtype=int)
    for i in range(n_frames):
        dists = np.linalg.norm(centers - q_traj[i], axis=1)
        assignments[i] = np.argmin(dists)

    transitions = np.sum(assignments[1:] != assignments[:-1])
    visits = np.bincount(assignments, minlength=len(centers))

    return {
        'n_transitions': int(transitions),
        'visits_per_mode': visits.tolist(),
        'transition_rate': transitions / n_frames if n_frames > 0 else 0,
    }


# =========================================================================
# Sampler factory
# =========================================================================

def create_sampler_and_integrator(name, potential, dim, dt, kT=1.0, mass=1.0):
    """Create a (dynamics, integrator) pair by name."""
    from research.eval.integrators import VelocityVerletThermostat

    if name == 'NH':
        dyn = NoseHoover(dim=dim, kT=kT, mass=mass, Q=1.0)
        integ = VelocityVerletThermostat(dyn, potential, dt=dt, kT=kT, mass=mass)
    elif name == 'NHC':
        dyn = NoseHooverChain(dim=dim, chain_length=3, kT=kT, mass=mass, Q=1.0)
        integ = VelocityVerletThermostat(dyn, potential, dt=dt, kT=kT, mass=mass)
    elif name == 'MSLO':
        # Multi-Scale Log-Osc with 3 timescales
        dyn = MultiScaleLogOsc(dim=dim, kT=kT, mass=mass,
                               Qs=[0.1, 1.0, 10.0])
        integ = MultiScaleLogOscVerlet(dyn, potential, dt=dt, kT=kT, mass=mass)
    elif name == 'LOCR':
        dyn = LogOscChainRotation(dim=dim, chain_length=3, kT=kT, mass=mass,
                                  Q=[1.0, 1.0, 1.0], alpha=0.5)
        integ = LOCRIntegrator(dyn, potential, dt=dt, kT=kT, mass=mass)
    else:
        raise ValueError(f"Unknown sampler: {name}")

    return dyn, integ


# =========================================================================
# Main simulation runner
# =========================================================================

def run_simulation(sampler_name, potential, dim, dt, n_force_evals,
                   kT=1.0, mass=1.0, q0=None, seed=42,
                   save_every=100):
    """Run a thermostat simulation and collect trajectory data.

    Returns dict with q_traj, p_traj, energies, xi_traj, timing info.
    """
    rng = np.random.default_rng(seed)

    if q0 is None:
        q0 = rng.normal(0, 0.5, size=dim)

    dyn, integ = create_sampler_and_integrator(
        sampler_name, potential, dim, dt, kT, mass)

    state = dyn.initial_state(q0, rng=rng)

    # Pre-allocate storage
    est_steps = n_force_evals  # ~1 force eval per step
    n_saved = est_steps // save_every + 1
    q_traj = []
    p_traj = []
    xi_traj = []

    t_start = time.time()
    step_count = 0
    nan_count = 0

    while state.n_force_evals < n_force_evals:
        state = integ.step(state)
        step_count += 1

        if np.any(np.isnan(state.q)) or np.any(np.isnan(state.p)):
            nan_count += 1
            # Reset to random state
            q_new = rng.normal(0, 0.5, size=dim)
            p_new = rng.normal(0, np.sqrt(mass * kT), size=dim)
            xi_new = np.zeros_like(state.xi)
            state = ThermostatState(q_new, p_new, xi_new, state.n_force_evals)
            integ._cached_grad_U = None
            if nan_count > 100:
                print(f"  WARNING: {nan_count} NaN resets, aborting")
                break
            continue

        if step_count % save_every == 0:
            q_traj.append(state.q.copy())
            p_traj.append(state.p.copy())
            xi_traj.append(state.xi.copy())

    t_elapsed = time.time() - t_start

    return {
        'q_traj': q_traj,
        'p_traj': p_traj,
        'xi_traj': xi_traj,
        'n_force_evals': state.n_force_evals,
        'n_steps': step_count,
        'wall_time': t_elapsed,
        'nan_count': nan_count,
    }


# =========================================================================
# Experiment definitions
# =========================================================================

EXPERIMENTS = {
    'LJ7_2D': {
        'potential_cls': lambda: LennardJonesCluster(n_atoms=7, spatial_dim=2),
        'dim': 14,
        'n_force_evals': 10_000_000,
        'dt': 0.002,
        'init_fn': lambda rng: init_lj_config(7, 2, rng),
        'save_every': 50,
        'system_type': 'lj',
        'n_atoms': 7,
        'spatial_dim': 2,
    },
    'LJ13_3D': {
        'potential_cls': lambda: LennardJonesCluster(n_atoms=13, spatial_dim=3),
        'dim': 39,
        'n_force_evals': 10_000_000,
        'dt': 0.001,
        'init_fn': lambda rng: init_lj_config(13, 3, rng),
        'save_every': 50,
        'system_type': 'lj',
        'n_atoms': 13,
        'spatial_dim': 3,
    },
    'Gauss_20D': {
        'potential_cls': lambda: CorrelatedGaussianND(dim=20, lam_min=0.01, lam_max=100.0),
        'dim': 20,
        'n_force_evals': 5_000_000,
        'dt': 0.005,
        'init_fn': lambda rng: rng.normal(0, 0.5, size=20),
        'save_every': 20,
        'system_type': 'gaussian',
    },
    'GMM_10D': {
        # separation=3.0, sigma=1.0 gives barrier ~ 0.5*(3/1)^2 = 4.5 kT per dim
        # feasible but challenging for deterministic thermostats
        'potential_cls': lambda: GaussianMixtureND(dim=10, n_modes=3, separation=3.0, sigma=1.0),
        'dim': 10,
        'n_force_evals': 5_000_000,
        'dt': 0.01,
        'init_fn': lambda rng: rng.normal(0, 1.0, size=10),
        'save_every': 50,
        'system_type': 'gmm',
    },
}

SAMPLER_NAMES = ['NH', 'NHC', 'MSLO', 'LOCR']


def run_experiment(exp_name, sampler_name, seed=42):
    """Run a single experiment and compute metrics."""
    exp = EXPERIMENTS[exp_name]
    potential = exp['potential_cls']()
    dim = exp['dim']
    rng = np.random.default_rng(seed)
    q0 = exp['init_fn'](rng)

    print(f"  Running {sampler_name} on {exp_name} (dim={dim}, "
          f"n_evals={exp['n_force_evals']}, dt={exp['dt']})...")

    result = run_simulation(
        sampler_name, potential, dim,
        dt=exp['dt'],
        n_force_evals=exp['n_force_evals'],
        q0=q0, seed=seed,
        save_every=exp['save_every'],
    )

    q_traj = result['q_traj']
    p_traj = result['p_traj']

    if len(q_traj) < 100:
        print(f"    WARNING: only {len(q_traj)} frames collected")
        return {'error': 'insufficient_frames', **result}

    # Burn-in: discard first 10%
    burnin = len(q_traj) // 10
    q_traj = q_traj[burnin:]
    p_traj = p_traj[burnin:]

    # Common metrics
    energies = compute_energy_trace(q_traj, p_traj, potential)
    ks = energy_ks_test(energies, dim)
    tau_e = autocorrelation_time(energies)
    ess = ess_per_force_eval(energies, result['n_force_evals'])

    metrics = {
        'energy_ks': float(ks),
        'autocorr_time_energy': float(tau_e),
        'ess_per_force_eval': float(ess),
        'wall_time': result['wall_time'],
        'nan_count': result['nan_count'],
        'n_frames': len(q_traj),
        'mean_energy': float(np.mean(energies)),
        'std_energy': float(np.std(energies)),
    }

    # System-specific metrics
    if exp['system_type'] == 'gaussian':
        expected_vars = potential.expected_variances  # at kT=1
        max_err, mean_err, all_errs = marginal_variance_error(q_traj, expected_vars)
        metrics['var_max_rel_error'] = float(max_err)
        metrics['var_mean_rel_error'] = float(mean_err)
        metrics['var_errors'] = all_errs.tolist()

    elif exp['system_type'] == 'lj':
        r_centers, g_r = radial_distribution_function(
            q_traj, exp['n_atoms'], exp['spatial_dim'])
        metrics['g_r'] = (r_centers.tolist(), g_r.tolist())

    elif exp['system_type'] == 'gmm':
        mv = mode_visitation(q_traj, potential.centers)
        metrics['mode_transitions'] = mv['n_transitions']
        metrics['visits_per_mode'] = mv['visits_per_mode']
        metrics['transition_rate'] = mv['transition_rate']

    # Autocorrelation of first coordinate
    q_first = np.array([q[0] for q in q_traj])
    tau_q0 = autocorrelation_time(q_first)
    metrics['autocorr_time_q0'] = float(tau_q0)

    print(f"    Done in {result['wall_time']:.1f}s. "
          f"KS={ks:.4f}, tau_E={tau_e:.1f}, ESS/eval={ess:.6f}")

    return {**metrics, 'energies': energies, 'q_traj': q_traj, 'p_traj': p_traj}


def run_all_experiments(seed=42):
    """Run all experiments for all samplers."""
    results = {}
    for exp_name in EXPERIMENTS:
        results[exp_name] = {}
        print(f"\n{'='*60}")
        print(f"System: {exp_name}")
        print(f"{'='*60}")
        for sname in SAMPLER_NAMES:
            try:
                results[exp_name][sname] = run_experiment(exp_name, sname, seed=seed)
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
                results[exp_name][sname] = {'error': str(e)}
    return results


# =========================================================================
# Plotting
# =========================================================================

def make_all_figures(results, fig_dir):
    """Generate all comparison figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)

    # Style guide compliance
    COLORS = {
        'NH': '#1f77b4',
        'NHC': '#ff7f0e',
        'MSLO': '#2ca02c',
        'LOCR': '#d62728',
    }

    # 1. Energy distribution comparison per system
    for exp_name in results:
        fig, axes = plt.subplots(1, len(SAMPLER_NAMES), figsize=(16, 4),
                                 sharey=True)
        fig.suptitle(f'Energy Distribution: {exp_name}', fontsize=16)
        for idx, sname in enumerate(SAMPLER_NAMES):
            ax = axes[idx]
            r = results[exp_name].get(sname, {})
            if 'energies' in r:
                E = r['energies']
                ax.hist(E, bins=80, density=True, alpha=0.7,
                        color=COLORS[sname], label=sname)
                ax.set_title(f"{sname}\nKS={r.get('energy_ks', 'N/A'):.4f}",
                             fontsize=12)
            else:
                ax.set_title(f"{sname}\n(failed)", fontsize=12)
            ax.set_xlabel('Energy', fontsize=12)
            if idx == 0:
                ax.set_ylabel('Density', fontsize=14)
            ax.tick_params(labelsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'energy_dist_{exp_name}.png'), dpi=150)
        plt.close()

    # 2. Bar chart comparison of metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 2a. Energy KS statistic
    ax = axes[0]
    exp_names = list(results.keys())
    x = np.arange(len(exp_names))
    width = 0.2
    for i, sname in enumerate(SAMPLER_NAMES):
        vals = []
        for exp_name in exp_names:
            r = results[exp_name].get(sname, {})
            vals.append(r.get('energy_ks', np.nan))
        ax.bar(x + i * width, vals, width, label=sname, color=COLORS[sname])
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(exp_names, fontsize=10)
    ax.set_ylabel('Energy KS statistic', fontsize=14)
    ax.set_title('Energy Distribution Quality', fontsize=16)
    ax.legend(fontsize=12)
    ax.set_yscale('log')
    ax.tick_params(labelsize=10)

    # 2b. ESS per force eval
    ax = axes[1]
    for i, sname in enumerate(SAMPLER_NAMES):
        vals = []
        for exp_name in exp_names:
            r = results[exp_name].get(sname, {})
            vals.append(r.get('ess_per_force_eval', 0))
        ax.bar(x + i * width, vals, width, label=sname, color=COLORS[sname])
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(exp_names, fontsize=10)
    ax.set_ylabel('ESS / force eval', fontsize=14)
    ax.set_title('Sampling Efficiency', fontsize=16)
    ax.legend(fontsize=12)
    ax.set_yscale('log')
    ax.tick_params(labelsize=10)

    # 2c. Autocorrelation time
    ax = axes[2]
    for i, sname in enumerate(SAMPLER_NAMES):
        vals = []
        for exp_name in exp_names:
            r = results[exp_name].get(sname, {})
            vals.append(r.get('autocorr_time_energy', np.nan))
        ax.bar(x + i * width, vals, width, label=sname, color=COLORS[sname])
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(exp_names, fontsize=10)
    ax.set_ylabel('Autocorrelation time', fontsize=14)
    ax.set_title('Mixing Speed (lower = better)', fontsize=16)
    ax.legend(fontsize=12)
    ax.set_yscale('log')
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'metrics_comparison.png'), dpi=150)
    plt.close()

    # 3. g(r) for LJ systems
    for exp_name in ['LJ7_2D', 'LJ13_3D']:
        if exp_name not in results:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        for sname in SAMPLER_NAMES:
            r = results[exp_name].get(sname, {})
            if 'g_r' in r:
                r_centers, g_r = r['g_r']
                ax.plot(r_centers, g_r, label=sname, color=COLORS[sname],
                        linewidth=2)
        ax.set_xlabel('r', fontsize=14)
        ax.set_ylabel('g(r) (relative)', fontsize=14)
        ax.set_title(f'Radial Distribution Function: {exp_name}', fontsize=16)
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'g_r_{exp_name}.png'), dpi=150)
        plt.close()

    # 4. Marginal variance errors for Gaussian
    if 'Gauss_20D' in results:
        fig, ax = plt.subplots(figsize=(10, 6))
        for sname in SAMPLER_NAMES:
            r = results['Gauss_20D'].get(sname, {})
            if 'var_errors' in r:
                ax.plot(range(20), r['var_errors'], 'o-', label=sname,
                        color=COLORS[sname], linewidth=2, markersize=5)
        ax.axhline(0.1, color='gray', linestyle='--', label='10% error')
        ax.set_xlabel('Dimension index', fontsize=14)
        ax.set_ylabel('Relative variance error', fontsize=14)
        ax.set_title('Marginal Variance Accuracy: 20D Correlated Gaussian',
                      fontsize=16)
        ax.legend(fontsize=12)
        ax.set_yscale('log')
        ax.tick_params(labelsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'var_errors_gauss20d.png'), dpi=150)
        plt.close()

    # 5. Mode visitation for GMM
    if 'GMM_10D' in results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Transition rates
        ax = axes[0]
        snames = []
        rates = []
        for sname in SAMPLER_NAMES:
            r = results['GMM_10D'].get(sname, {})
            if 'transition_rate' in r:
                snames.append(sname)
                rates.append(r['transition_rate'])
        colors_list = [COLORS[s] for s in snames]
        ax.bar(snames, rates, color=colors_list)
        ax.set_ylabel('Mode transition rate', fontsize=14)
        ax.set_title('Mode Hopping: 10D GMM (3 modes)', fontsize=16)
        ax.tick_params(labelsize=12)

        # Visits per mode
        ax = axes[1]
        x = np.arange(3)
        width = 0.2
        for i, sname in enumerate(SAMPLER_NAMES):
            r = results['GMM_10D'].get(sname, {})
            if 'visits_per_mode' in r:
                v = r['visits_per_mode']
                ax.bar(x + i * width, v, width, label=sname,
                       color=COLORS[sname])
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(['Mode 0', 'Mode 1', 'Mode 2'], fontsize=12)
        ax.set_ylabel('Visits (frames)', fontsize=14)
        ax.set_title('Mode Visitation Balance', fontsize=16)
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'mode_visitation_gmm10d.png'), dpi=150)
        plt.close()

    # 6. Dimension scaling summary
    fig, ax = plt.subplots(figsize=(10, 6))
    dims = {'LJ7_2D': 14, 'Gauss_20D': 20, 'LJ13_3D': 39}
    for sname in SAMPLER_NAMES:
        d_vals = []
        ess_vals = []
        for exp_name, d in sorted(dims.items(), key=lambda x: x[1]):
            r = results.get(exp_name, {}).get(sname, {})
            if 'ess_per_force_eval' in r:
                d_vals.append(d)
                ess_vals.append(r['ess_per_force_eval'])
        if d_vals:
            ax.plot(d_vals, ess_vals, 'o-', label=sname, color=COLORS[sname],
                    linewidth=2, markersize=8)
    ax.set_xlabel('Dimensionality (DOF)', fontsize=14)
    ax.set_ylabel('ESS / force eval', fontsize=14)
    ax.set_title('Sampling Efficiency vs Dimension', fontsize=16)
    ax.set_yscale('log')
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'ess_vs_dim.png'), dpi=150)
    plt.close()

    print(f"\nAll figures saved to {fig_dir}")


# =========================================================================
# Main
# =========================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', type=str, default=None,
                        help='Run specific system (LJ7_2D, LJ13_3D, Gauss_20D, GMM_10D)')
    parser.add_argument('--sampler', type=str, default=None,
                        help='Run specific sampler (NH, NHC, MSLO, LOCR)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quick', action='store_true',
                        help='Reduced budget for quick testing')
    args = parser.parse_args()

    if args.quick:
        for exp in EXPERIMENTS.values():
            exp['n_force_evals'] = min(exp['n_force_evals'], 500_000)

    fig_dir = os.path.join(os.path.dirname(__file__), 'figures')

    if args.system and args.sampler:
        result = run_experiment(args.system, args.sampler, seed=args.seed)
        for k, v in result.items():
            if k not in ('energies', 'q_traj', 'p_traj', 'g_r', 'var_errors'):
                print(f"  {k}: {v}")
    else:
        results = run_all_experiments(seed=args.seed)

        # Print summary table
        print(f"\n{'='*80}")
        print("SUMMARY TABLE")
        print(f"{'='*80}")
        header = f"{'System':<12} {'Sampler':<8} {'KS':>8} {'tau_E':>8} {'ESS/eval':>10} {'Wall(s)':>8} {'NaN':>5}"
        print(header)
        print('-' * 80)
        for exp_name in EXPERIMENTS:
            for sname in SAMPLER_NAMES:
                r = results[exp_name].get(sname, {})
                if 'error' in r and isinstance(r.get('error'), str):
                    print(f"{exp_name:<12} {sname:<8} {'ERROR':>8}")
                    continue
                ks = r.get('energy_ks', float('nan'))
                tau = r.get('autocorr_time_energy', float('nan'))
                ess = r.get('ess_per_force_eval', 0)
                wt = r.get('wall_time', 0)
                nan_c = r.get('nan_count', 0)
                print(f"{exp_name:<12} {sname:<8} {ks:>8.4f} {tau:>8.1f} {ess:>10.6f} {wt:>8.1f} {nan_c:>5d}")

        # System-specific metrics
        print(f"\n{'='*60}")
        print("GAUSSIAN 20D: Marginal Variance Errors")
        print(f"{'='*60}")
        for sname in SAMPLER_NAMES:
            r = results.get('Gauss_20D', {}).get(sname, {})
            if 'var_max_rel_error' in r:
                print(f"  {sname}: max_err={r['var_max_rel_error']:.4f}, "
                      f"mean_err={r['var_mean_rel_error']:.4f}")

        print(f"\n{'='*60}")
        print("GMM 10D: Mode Visitation")
        print(f"{'='*60}")
        for sname in SAMPLER_NAMES:
            r = results.get('GMM_10D', {}).get(sname, {})
            if 'mode_transitions' in r:
                print(f"  {sname}: transitions={r['mode_transitions']}, "
                      f"visits={r['visits_per_mode']}, "
                      f"rate={r['transition_rate']:.6f}")

        make_all_figures(results, fig_dir)
