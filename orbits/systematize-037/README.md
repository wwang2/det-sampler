# det_sampler — Multi-Scale Log-Oscillator Thermostat

A deterministic, gradient-only sampler for smooth unnormalised densities.
Single file, numpy + scipy only. Drop-in alternative to Langevin where the
target's curvature spectrum spans a moderate dynamic range.

## 3 lines to replace Langevin

```python
from det_sampler import MultiScaleThermostat
sampler = MultiScaleThermostat(grad_log_prob, dim=10)        # auto kappa, N, dt
samples = sampler.sample(q_init, n_samples=10_000, burn_in=1000)
```

## What it does

Dynamics:

```
dq/dt    = p / m
dp/dt    = -grad U(q) - Gamma(xi) * p,   Gamma(xi) = sum_i 2 xi_i / (1 + xi_i^2)
dxi_i/dt = (p.p/m - dim*kT) / Q_i,       i = 1..N
```

N thermostats with Q_i log-uniformly spaced across the curvature spectrum
of -log p. F1 prescription from orbit #034:

```
Q_min = 1 / sqrt(kappa_max)
Q_max = 1 / sqrt(kappa_min)
N     = max(3, ceil(log10(kappa_max/kappa_min) + 2))
dt    = 0.05 * min(Q_min, 1/sqrt(kappa_max))
```

When kappa_range is not provided the sampler estimates it via a
finite-difference diagonal Hessian on the initial point.

Integration is BAOAB-style with FSAL caching of the gradient — exactly one
force evaluation per step after burn-in.

## API

```python
MultiScaleThermostat(
    grad_log_prob,                 # callable q -> grad log p(q)
    dim,
    kappa_range=None,              # (kappa_min, kappa_max) or None for auto
    kT=1.0,
    mass=1.0,
    N=None, dt=None,               # override auto-tuning
    seed=0,
)

sampler.sample(q_init, n_samples, burn_in=1000, thin=1) -> ndarray
sampler.diagnostics() -> dict     # tau_int, ESS, ESS/force_eval, warnings, ...
```

## Run the demo

```bash
./run.sh
```

Produces figures/demo_output.png with three panels:
1. 1D harmonic oscillator
2. 10D anisotropic Gaussian (kappa_ratio=100)
3. 2D Gaussian mixture (mode hopping)
