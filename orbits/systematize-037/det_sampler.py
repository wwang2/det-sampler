"""det_sampler — Multi-Scale Log-Oscillator Thermostat Sampler.

A deterministic, gradient-only sampler for unnormalized smooth densities.

The dynamics are
    dq/dt   = p / m
    dp/dt   = -grad U(q) - Gamma(xi) p,    Gamma(xi) = sum_i g(xi_i)
    dxi_i/dt = (K - dim*kT) / Q_i,         g(xi) = 2 xi / (1 + xi^2)
where K = 0.5 * p.p / m is the kinetic energy and {Q_i} are N thermostat
inertias spaced log-uniformly across the curvature spectrum of -log p.

The "log-osc" friction g and the multi-scale Q ladder come from the
analysis in orbit #034 (F1 prescription: Q_min = 1/sqrt(kappa_max),
Q_max = 1/sqrt(kappa_min)).

Single-file, numpy + scipy only. Drop-in alternative to Langevin for
smooth targets where the gradient is cheap and the curvature spectrum
spans a moderate dynamic range.

Example
-------
>>> import numpy as np
>>> from det_sampler import MultiScaleThermostat
>>> grad_log_p = lambda q: -q                       # standard normal
>>> sampler = MultiScaleThermostat(grad_log_p, dim=2)
>>> samples = sampler.sample(np.zeros(2), n_samples=5000, burn_in=500)
>>> samples.mean(0), samples.std(0)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Core dynamics
# ---------------------------------------------------------------------------

def _g(xi: np.ndarray) -> np.ndarray:
    """Log-oscillator friction g(xi) = 2 xi / (1 + xi^2)."""
    return 2.0 * xi / (1.0 + xi * xi)


# ---------------------------------------------------------------------------
# Auto parameter selection
# ---------------------------------------------------------------------------

def estimate_curvature_range(
    grad_U: Callable[[np.ndarray], np.ndarray],
    q: np.ndarray,
    eps: float = 1e-3,
    n_probes: int = 16,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Estimate (kappa_min, kappa_max) via finite-difference diagonal Hessian.

    Probes ``n_probes`` random points around ``q`` (Gaussian jitter, scale eps)
    and computes the diagonal of the Hessian by central differences:
        H_ii ~ (grad_i(q + eps e_i) - grad_i(q - eps e_i)) / (2 eps).
    Returns the (clipped) min and max absolute diagonal entry.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    dim = q.size
    diag_estimates = []
    for _ in range(n_probes):
        q0 = q + 0.1 * rng.standard_normal(dim)
        diag = np.empty(dim)
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = eps
            gp = grad_U(q0 + e)[i]
            gm = grad_U(q0 - e)[i]
            diag[i] = (gp - gm) / (2.0 * eps)
        diag_estimates.append(diag)
    H = np.abs(np.stack(diag_estimates)).mean(0)
    H = np.clip(H, 1e-6, 1e12)
    return float(H.min()), float(H.max())


def auto_params(kappa_min: float, kappa_max: float) -> dict:
    """F1 prescription from orbit #034 + heuristics for N and dt.

    Returns dict with Q_min, Q_max, N, dt.
    """
    Q_min = 1.0 / math.sqrt(kappa_max)
    Q_max = 1.0 / math.sqrt(kappa_min)
    ratio = kappa_max / kappa_min
    N = max(3, int(math.ceil(math.log10(ratio) + 2)))
    dt = 0.05 * min(Q_min, 1.0 / math.sqrt(kappa_max))
    return dict(Q_min=Q_min, Q_max=Q_max, N=N, dt=dt)


# ---------------------------------------------------------------------------
# Diagnostics helpers
# ---------------------------------------------------------------------------

def _autocorr_1d(x: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """Normalised autocorrelation via FFT (one trace)."""
    n = x.size
    if max_lag is None:
        max_lag = min(n // 4, 500)
    x = x - x.mean()
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f))[:n].real
    acf /= acf[0] + 1e-30
    return acf[:max_lag]


def integrated_autocorr_time(samples: np.ndarray) -> np.ndarray:
    """tau_int per dimension via Sokal's automated windowing.

    samples: (n_samples, dim)
    """
    n, dim = samples.shape
    taus = np.empty(dim)
    for d in range(dim):
        acf = _autocorr_1d(samples[:, d])
        # Sokal: window M smallest with M >= c*tau, c=5
        tau = 1.0 + 2.0 * np.cumsum(acf[1:])
        c = 5.0
        ms = np.arange(1, tau.size + 1)
        ok = ms >= c * tau
        if ok.any():
            M = int(np.argmax(ok))
            taus[d] = max(tau[M], 1.0)
        else:
            taus[d] = max(tau[-1], 1.0)
    return taus


# ---------------------------------------------------------------------------
# Main sampler
# ---------------------------------------------------------------------------

@dataclass
class _State:
    q: np.ndarray
    p: np.ndarray
    xi: np.ndarray            # shape (N,)
    grad_U: np.ndarray        # FSAL cache (gradient of U = -log p at q)
    n_force_evals: int = 0


class MultiScaleThermostat:
    """Multi-scale log-oscillator thermostat sampler.

    Parameters
    ----------
    grad_log_prob : callable
        Function ``q -> grad log p(q)`` (returns array of shape ``(dim,)``).
        We use ``grad U = -grad log p``.
    dim : int
        Dimension of ``q``.
    kappa_range : (float, float), optional
        Curvature range ``(kappa_min, kappa_max)`` of ``-log p``. If ``None``,
        estimated automatically by a finite-difference diagonal Hessian on
        an exploratory burn-in.
    kT : float, default 1.0
        Target temperature. Equilibrium has ``<p^2/m> = kT`` per dof.
    mass : float, default 1.0
    N : int, optional
        Number of thermostat scales. Auto-selected if ``None``.
    dt : float, optional
        Integrator step. Auto-selected if ``None``.
    seed : int, default 0
    """

    def __init__(
        self,
        grad_log_prob: Callable[[np.ndarray], np.ndarray],
        dim: int,
        kappa_range: Optional[Tuple[float, float]] = None,
        kT: float = 1.0,
        mass: float = 1.0,
        N: Optional[int] = None,
        dt: Optional[float] = None,
        seed: int = 0,
    ):
        self.grad_log_prob = grad_log_prob
        self.grad_U = lambda q: -np.asarray(grad_log_prob(q), dtype=float)
        self.dim = int(dim)
        self.kT = float(kT)
        self.mass = float(mass)
        self.rng = np.random.default_rng(seed)

        self._kappa_range = kappa_range
        self._user_N = N
        self._user_dt = dt
        self._configured = False
        self._diagnostics: dict = {}

    # ---- configuration ---------------------------------------------------

    def _configure(self, q0: np.ndarray) -> None:
        if self._kappa_range is None:
            kappa_min, kappa_max = estimate_curvature_range(
                self.grad_U, q0, rng=self.rng
            )
        else:
            kappa_min, kappa_max = self._kappa_range
        kappa_min = max(kappa_min, 1e-6)
        kappa_max = max(kappa_max, kappa_min * 1.0001)
        params = auto_params(kappa_min, kappa_max)

        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.N = int(self._user_N) if self._user_N is not None else params["N"]
        self.dt = float(self._user_dt) if self._user_dt is not None else params["dt"]
        # Log-uniform Q ladder
        if self.N == 1:
            self.Q = np.array([math.sqrt(params["Q_min"] * params["Q_max"])])
        else:
            self.Q = np.exp(
                np.linspace(math.log(params["Q_min"]),
                            math.log(params["Q_max"]),
                            self.N)
            )
        self._configured = True

    # ---- one BAOAB-style step -------------------------------------------

    def _step(self, st: _State) -> _State:
        dt = self.dt
        h = 0.5 * dt
        kT_dim = self.dim * self.kT  # target: <p.p/m> = dim*kT

        # half-step xi: dxi/dt = (K - dim*kT) / Q
        K = np.dot(st.p, st.p) / self.mass  # 2*kinetic
        xi = st.xi + h * (K - kT_dim) / self.Q

        # half-step momenta: friction (analytical) then kick
        Gamma = _g(xi).sum()
        scale = math.exp(-Gamma * h)
        # clip for safety
        if scale > 1e10:
            scale = 1e10
        elif scale < 1e-10:
            scale = 1e-10
        p = st.p * scale
        p = p - h * st.grad_U

        # full position step
        q = st.q + dt * p / self.mass

        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(p))):
            raise FloatingPointError("Sampler diverged — try smaller dt.")

        grad_U_new = self.grad_U(q)
        n_evals = st.n_force_evals + 1

        # half-step momenta: kick then friction
        p = p - h * grad_U_new
        K = np.dot(p, p) / self.mass
        # update xi using current K (palindromic)
        # Note: Gamma still uses xi from before final xi half-step; we
        # apply friction first then xi update for symmetry.
        Gamma = _g(xi).sum()
        scale = math.exp(-Gamma * h)
        if scale > 1e10:
            scale = 1e10
        elif scale < 1e-10:
            scale = 1e-10
        p = p * scale

        # half-step xi
        K = np.dot(p, p) / self.mass
        xi = xi + h * (K - kT_dim) / self.Q

        return _State(q=q, p=p, xi=xi, grad_U=grad_U_new, n_force_evals=n_evals)

    # ---- public API ------------------------------------------------------

    def sample(
        self,
        q_init: np.ndarray,
        n_samples: int,
        burn_in: int = 1000,
        thin: int = 1,
    ) -> np.ndarray:
        """Run the sampler. Returns array of shape (n_samples, dim)."""
        q_init = np.asarray(q_init, dtype=float).reshape(-1)
        if q_init.size != self.dim:
            raise ValueError(f"q_init has size {q_init.size}, expected {self.dim}")

        if not self._configured:
            self._configure(q_init)

        # Init momenta from Maxwell–Boltzmann at target T
        p0 = self.rng.standard_normal(self.dim) * math.sqrt(self.mass * self.kT)
        xi0 = np.zeros(self.N)
        g0 = self.grad_U(q_init)
        st = _State(q=q_init.copy(), p=p0, xi=xi0, grad_U=g0, n_force_evals=1)

        # Burn-in
        for _ in range(burn_in):
            st = self._step(st)

        # Sampling
        out = np.empty((n_samples, self.dim))
        for i in range(n_samples):
            for _ in range(thin):
                st = self._step(st)
            out[i] = st.q

        # Diagnostics
        n_evals_used = st.n_force_evals
        taus = integrated_autocorr_time(out)
        ess = n_samples / (2.0 * taus)            # per dimension
        ess_min = float(ess.min())
        ess_per_eval = ess_min / max(n_evals_used, 1)

        emp_var = out.var(0)
        # Heuristic: warn only if any dim's tau exceeds 20% of the chain length
        warnings_list = []
        if (taus > 0.2 * n_samples).any():
            warnings_list.append(
                "Some dims have tau_int > 20% of chain length — possible non-ergodicity."
            )

        self._diagnostics = dict(
            n_force_evals=n_evals_used,
            tau_int=taus,
            ess=ess,
            ess_min=ess_min,
            ess_per_force_eval=ess_per_eval,
            empirical_variance=emp_var,
            warnings=warnings_list,
            kappa_range=(self.kappa_min, self.kappa_max),
            Q=self.Q,
            N=self.N,
            dt=self.dt,
        )
        for w in warnings_list:
            warnings.warn(w)
        return out

    def diagnostics(self) -> dict:
        """Return diagnostics from the most recent ``sample()`` call."""
        return dict(self._diagnostics)
