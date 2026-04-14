---
strategy: logosc-esh-hybrid
type: experiment
status: complete
eval_version: eval-v1
metric: 0.781
issue: 76
parents:
  - orbit/gprime-optimization-073
---

# Orbit 076: Log-osc NH Friction + ESH Non-Newtonian Momentum Hybrid

## Glossary

- **g(xi) = 2*xi/(1+xi^2)**: log-osc NH friction; g'(0)=2, g->0 at large xi
- **v(p) = tanh(|p|/p0) * p/|p| * p0**: ESH non-Newtonian velocity (Versteeg 2021)
- **p0**: ESH scale parameter; p0->inf recovers Newtonian v=p/m
- **tau_int**: integrated autocorrelation time (stiffest mode q_d^2)
- **Hybrid corrected**: ESH v(p) + log-osc g(xi) + modified xi-driving by v(p).p
- **Hybrid heuristic**: ESH v(p) + log-osc g(xi) + standard xi-driving by K(p)

## Approach

Test whether combining ESH non-Newtonian momentum v(p) with NH log-osc friction
can improve sampling beyond either method alone. ESH bounds position velocity,
potentially preventing momentum blow-up and aiding ergodicity; log-osc friction
provides adaptive damping (orbit 073: 47% faster than tanh).

## Phase 0: Analytical Derivation

### Combined system equations

```
dq/dt = v(p)           where v(p) = tanh(|p|/p0) * p/|p| * p0
dp/dt = -grad_U(q) - g(xi)*p     where g(xi) = 2*xi/(1+xi^2)
dxi/dt = (K(p) - D*kT) / Q       where K(p) = |p|^2/(2m)
```

### Liouville condition: FAILS

Target invariant measure: mu(q,p,xi) = exp(-beta*H(q,p) - Q*xi^2/(2*kT))

**Divergence of the flow:** div(f) = -D*g(xi)

**Condition:** div(f) + f . grad(log mu) = 0

The critical mismatch: the position term contributes v(p).(-beta*grad_U), while
the momentum term contributes (p/m).(-beta*grad_U). For standard NH these cancel.
For the hybrid, the residual v(p).grad_U - (p/m).grad_U != 0 for general grad_U.

**Conclusion: Naive combination BREAKS the canonical invariant measure.**

No simple modification of the thermostat driving restores exp(-beta*H) as the
invariant measure. We tested two empirical variants:

1. **Corrected**: dxi/dt = (v(p).p - D*kT)/Q — preserves some measure, not canonical
2. **Heuristic**: dxi/dt = (|p|^2 - D*kT)/Q — standard driving, no measure preservation

## Phase 2: Results

### Anisotropic Gaussian d=10, kappa=100

| Method | Best tau_int | Best params | vs NH log-osc |
|--------|-------------|-------------|---------------|
| **NH log-osc** | **20.0** | Q=0.05-0.3 | **1.000** |
| NHC(M=3) tanh | 22.7 | Q=1.0 | 0.881 |
| Hybrid CORRECTED | 25.6 | Q=0.3, p0=5.0 | 0.781 |
| Hybrid HEURISTIC | 26.4 | Q=0.3, p0=5.0 | 0.757 |
| Pure ESH | 3785.9 | p0=0.5 | 0.005 |

**Key finding: ESH v(p) monotonically degrades tau_int as p0 decreases.**

At p0=5.0 (weak ESH): tau ~26 (28% worse than NH log-osc)
At p0=0.5 (strong ESH): tau ~200 (10x worse)
At p0=inf (no ESH): tau=20.0 (NH log-osc)

Corrected and heuristic give nearly identical tau_int (within 3%).

### 1D Harmonic Oscillator Ergodicity

| Method | Best score | Best params | Ergodic? |
|--------|-----------|-------------|----------|
| **NHC(M=3)** | **0.895** | Q=0.1 | **Yes** |
| NH log-osc | 0.715 | Q=0.3 | No |
| Hybrid CORRECTED | 0.643 | Q=0.1, p0=2.0 | No |
| Hybrid HEURISTIC | 0.639 | Q=0.1, p0=2.0 | No |

**Key finding: ESH v(p) does NOT improve ergodicity. It makes it worse.**

At p0=0.5, the corrected hybrid has score=0.000 despite coverage=0.73 — the system
explores phase space but samples the WRONG distribution. The momentum marginal
variance error is 5.2x (var_p_err=5.18) because ESH distorts the momentum distribution.

### 2D Double Well KL Divergence

| Method | Best KL | Best params |
|--------|---------|-------------|
| **NHC(M=3)** | **0.055** | Q=0.1 |
| NH log-osc | 0.069 | Q=1.0 |
| Hybrid CORRECTED | 0.148 | Q=0.3, p0=0.5 |
| Hybrid HEURISTIC | 0.256 | Q=0.1, p0=1.0 |

All hybrids have 2-4x worse KL than baselines.

## Key Findings

### 1. The naive ESH+NH combination is provably incorrect (Phase 0)

The Liouville condition requires the position velocity and the measure gradient
to be consistent. Replacing v(p) = p/m with ESH velocity creates an irreconcilable
mismatch. No modification of the thermostat driving can fix this while preserving
the standard canonical distribution.

### 2. ESH bounded velocity monotonically degrades mixing (Phase 2)

On the anisotropic Gaussian, tau_int increases monotonically as p0 decreases:
p0=5.0 (weak ESH): 26 | p0=2.0: 55 | p0=1.0: 100 | p0=0.5: 200

The bounded velocity SLOWS DOWN phase-space exploration because large momenta
cannot translate into fast position changes. In a multi-scale system (kappa=100),
the stiffest mode needs large momentum excursions to mix — bounding the velocity
prevents this.

### 3. ESH v(p) distorts the momentum distribution catastrophically

On the 1D HO, the hybrid corrected with p0=0.5 achieves coverage=0.73 but
score=0.000. The system explores phase space but with a WRONG distribution:
the momentum marginal has variance error 5.18x (vs 0.004 for NHC).

### 4. Corrected vs heuristic driving makes no practical difference

Both variants give nearly identical tau_int and ergodicity scores. The driving
mechanism is not the bottleneck — the bounded velocity is.

## What I Learned

1. **ESH non-Newtonian momentum is fundamentally incompatible with NH-type thermostats**
   because the Liouville condition requires v(p) = nabla_p H for the measure to work.
   ESH achieves its invariant measure through a different mechanism (energy conservation
   in an expanded space) that cannot be spliced into NH dynamics.

2. **Bounded velocity is anti-helpful for mixing** on multi-scale systems. The stiffest
   mode needs unbounded momentum to traverse the energy landscape efficiently. This is
   the opposite of what ESH was designed for (HMC-like sampling on single-scale systems).

3. **Log-osc NH friction (orbit 073) remains the best single-thermostat approach.**
   Its adaptive damping mechanism (high g'(0) + g->0 decay) cannot be improved by
   adding ESH velocity.

4. **Coverage != correctness.** The hybrid achieves good phase-space coverage (0.73 on
   1D HO) but samples the wrong distribution. This is a cautionary tale about using
   coverage alone as a metric.

## Compute

Wall time: 910s (~15 min). 3 experiments, ~90 parameter configurations total.

## References

- Orbit 073: gprime-optimization — log-osc 47% faster than tanh via adaptive damping
- Versteeg 2021 (NeurIPS): ESH dynamics — non-Newtonian momentum for sampling
