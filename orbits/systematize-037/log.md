---
strategy: systematize-037
type: experiment
status: complete
eval_version: eval-v1
metric: 1.48e-03
issue: 37
parents:
  - orbit/optimal-spectrum-theory-034
---

# Systematized Reference Implementation

## Result

Headline metric: **ESS / force-eval = 1.48e-3** on the 10D anisotropic Gaussian
benchmark (kappa_ratio = 100, all parameters auto-selected).

| Demo | Result |
|---|---|
| 1D harmonic, N(0,1)        | mean ~ 0, var = 1.06 (target 1) |
| 10D anisotropic, kr=100    | max relative variance error = 1.04 (slow mode); per-dim tau_int 10..311 |
| 2D Gaussian mixture        | ~25/75 mode split with mode-hopping, both modes occupied |

Auto-tuning on the 10D problem detected kappa_range = (1.0, 100.0) via
diag-Hessian probing, then selected N=4, dt=0.005.

## Approach

Single-file det_sampler.py (~330 lines) providing MultiScaleThermostat with
the F1 prescription from orbit #034:

- Q_min = 1/sqrt(kappa_max), Q_max = 1/sqrt(kappa_min), log-uniform ladder
- N = max(3, ceil(log10(kappa_max/kappa_min) + 2))
- dt = 0.05 * min(Q_min, 1/sqrt(kappa_max))

Auto kappa estimation: averaged finite-difference diagonal-Hessian probes
around the initial point.

Integrator: BAOAB-style palindromic splitting with analytic friction
rescaling exp(-Gamma * dt/2) and FSAL gradient caching — exactly one force
evaluation per step in steady state.

Diagnostics: tau_int per dim (Sokal windowing), ESS, ESS/force-eval, and a
warning when any dim's tau exceeds 20% of chain length.

## Files

- det_sampler.py  — single-file library (numpy + scipy only)
- demo.py         — three target demos + figure
- figures/demo_output.png — 3-panel demo output
- run.sh          — reproducer
- README.md       — usage guide

## Notes

The dynamics convention is dxi/dt = (p.p/m - dim*kT)/Q (matches
log-osc-001/solution.py); equivalent to (2K - dim*kT)/Q with K = 0.5 p.p/m.
1D harmonic recovery (var ~ 1.06 from 20k samples) confirms units are
correct end-to-end.
