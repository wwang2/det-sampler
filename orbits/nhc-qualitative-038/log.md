---
strategy: nhc-qualitative
status: complete
eval_version: v1
metric: 6
issue: 38
parents:
  - q-optimization-035
---

# nhc-qualitative-038: qualitative figures NHC vs parallel multi-scale

## Result
6 figures generated (seed=42), all in `figures/`. The visual story confirms #035's
quantitative numbers (2.48x faster tau_int, 4x more mode crossings).

## Approach
Reused the BAOAB integrators from `orbits/q-optimization-035/run_experiment.py` and
extended them with optional thermostat-variable recording (xi_i and friction Gamma).
Single-seed (42) representative runs:
- 2D anisotropic Gaussian, kappas = (1, 100), dt = 0.005, 200k steps, burn-in 20k.
- 2D Gaussian mixture (5 modes ring, r=3, sigma=0.5), dt = 0.02, 200k steps, burn-in 20k.

Parallel multi-scale uses log-uniform Q schedule (Q in [1/sqrt(kappa_max), 1] for the
Gaussian; [0.1, 10] for the GMM). NHC uses M=3 with Q_ref=1 (the standard recommendation).

## Figures
- `density_2d_gauss.png`: side-by-side scatter; NHC concentrates near the origin in
  the slow direction, parallel fills the 1-2 sigma ellipse.
- `density_gmm.png`: NHC misses the top-right mode entirely; parallel covers all 5.
- `traj_2d_gauss.png`: NHC mostly oscillates in q_fast; parallel sweeps q_slow too.
- `traj_gmm.png`: NHC mostly stays in 1-2 modes over 5000 steps; parallel hops between modes.
- `traj_thermostat_vars.png`: 3-panel time series of NHC chain xi_i (1 dominant scale)
  vs parallel xi_i for N=5 (multiple coexisting timescales) and the friction Gamma(t).
- `mode_occupation_gmm.png`: cumulative fraction in each mode -> parallel converges
  toward 0.2 for all modes; NHC stays unbalanced.

## What I learned
The visual difference is dramatic on the GMM: NHC simply does not visit one of the 5
modes within 200k force evals, while parallel multi-scale balances them. On the
anisotropic Gaussian the qualitative gap is the slow-direction coverage, exactly the
mechanism the parallel-spectrum prescription targets.

## Prior Art & Novelty
This orbit produces no new science; it visualizes results from #035. Methods used
(NHC and parallel multi-scale log-osc) have established prior art documented in the
parent orbits.

## References
- Parent orbit: orbits/q-optimization-035 (head-to-head NHC vs parallel)
- [Martyna, Klein, Tuckerman (1992)](https://doi.org/10.1063/1.463940) - NHC original
