---
strategy: visual-validation
status: in-progress
eval_version: eval-v1
metric: null
issue: 15
parent: multiscale-chain-009
---

# Extended Visual Validation Across All Systems

## Goal
Create extensive visual validation showing the samplers working on every benchmark system.
Trajectory overlays on landscapes, density comparisons, convergence plots, efficiency bars.

## Seeds & Parameters
- Seeds: [42, 123, 7, 999, 314]
- Force evaluations: 2M per run
- NHC baseline: M=3, Q=1.0
- NHCTail champion: Qs=[0.1, 0.7, 10.0], chain_length=2

## Iteration 1: Figures 1, 2, 5

### Figure 1: Trajectory Overlays (fig1_trajectory_overlays.png)
- 4x3 panel: DW, GMM, Rosenbrock, HO x {Landscape, NHC, NHCTail}
- Landscape contours faded in trajectory panels so trajectory lines are visible
- NHC trajectories in blue, NHCTail in green

### Figure 2: Density Comparison (fig2_density_comparison.png)
- 4x3 panel with KL annotations
- KL Results (seed=42, 2M force evals):
  - Double Well: NHC=0.0185, NHCTail=0.0123
  - Gaussian Mixture: NHC=0.2939, NHCTail=0.0176 (16x improvement!)
  - Rosenbrock: NHC=0.0061, NHCTail=0.0035
  - HO: NHC=0.0019, NHCTail=0.0016

### Figure 5: HO Phase Space Coverage (fig5_ho_coverage.png)
- 2x4 panel: NHC vs NHCTail at 10k, 50k, 200k, 1M force evals
- NHC clearly shows KAM tori (concentric ring patterns)
- NHCTail fills phase space progressively

### Key Findings (so far)
- GMM is the standout case: NHCTail achieves 16x lower KL than NHC
- NHC fails to visit all GMM modes uniformly (sparse dots in density plot)
- HO ergodicity difference visible in phase space coverage
- All systems show NHCTail matching or exceeding NHC

## Iteration 2: Figures 3, 4 (in progress)
- Figure 3: Convergence curves with error bands (4 samplers x 4 systems x 5 seeds)
- Figure 4: ESS/force-eval bars + time-to-threshold + ergodicity scores

## References
- [Martyna et al. (1992)](https://doi.org/10.1063/1.463940) -- Nose-Hoover chains, J. Chem. Phys. 97, 2635
- [Nose (1984)](https://doi.org/10.1063/1.447334) -- Original Nose thermostat
- Parent orbit: #12 (multiscale-chain-009) -- MultiScaleNHCTail champion sampler
- Grandparent: #8 (log-osc-multiT-005) -- Multi-scale approach
- [KAM theorem](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold%E2%80%93Moser_theorem) -- explains NH non-ergodicity on HO
