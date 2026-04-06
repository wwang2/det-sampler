---
strategy: visual-validation
status: complete
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

## Iteration 2: Figures 3, 4 (complete)

### Figure 3: Convergence Studies (fig3_convergence_studies.png)
4 samplers × 4 systems × 5 seeds, 300K force evals, log-log axes, KL=0.01 threshold.

| System | NHC | LogOsc | MultiScale | NHCTail |
|--------|-----|--------|------------|---------|
| HO | — | — | — | ~0.013 |
| DW | — | — | — | ~0.013 |
| GMM | — | — | ~0.56 | ~0.19 |
| Rosenbrock | ~0.036 | ~0.039 | **0.014** | **0.013** |

### Figure 4: Efficiency Comparison (fig4_efficiency_comparison.png)
ESS/force-eval across systems (5 seeds, 300K evals):
- NHC: ~0.003 ESS/eval (HO)
- LogOsc: ~0.003 ESS/eval
- MultiScale: **~0.010 ESS/eval** (3x over NHC)
- NHCTail: **~0.010 ESS/eval** (3x over NHC)

## References
- [Martyna et al. (1992)](https://doi.org/10.1063/1.463940) -- Nose-Hoover chains, J. Chem. Phys. 97, 2635
- [Nose (1984)](https://doi.org/10.1063/1.447334) -- Original Nose thermostat
- Parent orbit: #12 (multiscale-chain-009) -- MultiScaleNHCTail champion sampler
- Grandparent: #8 (log-osc-multiT-005) -- Multi-scale approach
- [KAM theorem](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold%E2%80%93Moser_theorem) -- explains NH non-ergodicity on HO
