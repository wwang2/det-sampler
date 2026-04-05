---
strategy: publication-figures
status: complete
eval_version: eval-v1
metric: N/A (figure quality)
issue: 10
parent: log-osc-001, log-osc-multiT-005, log-osc-chain-002
---

## Goal

Create 6 publication-quality (Nature/Science-level) figures for the deterministic thermostat sampler paper.

## Figures

| Figure | Title | Status | Description |
|--------|-------|--------|-------------|
| Fig 1 | The Problem | Done | NH fails on HO: torus trapping, bimodal P(q), quasi-periodic xi |
| Fig 2 | The Solution | Done | Bounded friction g(xi), phase space comparison, Lyapunov + ergodicity vs Q |
| Fig 3 | Trajectory on Landscape | Done | NH vs Log-Osc on double-well, MSLO on GMM, Log-Osc on Rosenbrock |
| Fig 4 | Quantitative Comparison | Done | Bar charts: ergodicity, KL on double-well, KL on GMM (log scale) |
| Fig 5 | Multi-Scale Mechanism | Done | Friction signals, xi timeseries, PSD, GMM mode coverage |
| Fig 6 | Scaling to High Dimensions | Done | ESS vs DOF, LJ-7 energy, 10D Gaussian variance errors |

## Approach

Each figure script is self-contained with:
- Fixed random seeds (42, 123, 7 for multi-seed runs)
- Style guide compliance: 14pt labels, 12pt ticks, 16pt titles, 300 DPI
- Panel labels (a)-(f) in upper-left
- Consistent colors: NH=#1f77b4, NHC=#ff7f0e, novel samplers from tab10(2+)

## Key Results

### Figure 4 (300k force evals, 3 seeds)
- Ergodicity (1D HO): NH=0.68, NHC=0.88, Log-Osc=0.84, LOCR=0.88, MSLO=0.85
- KL double-well: MSLO=0.014 (best), NHC=0.035, LOCR=0.052, Log-Osc=0.061, NH=0.59
- KL GMM: NH=0.54, NHC=0.52, MSLO=1.05, LOCR=1.09, Log-Osc=3.72

### Figure 6 (100k force evals)
- ESS/eval: MSLO consistently highest (0.11-0.14), LOCR stable at 0.10
- LJ-7: LOCR achieves lowest energy (<E>=-6.24), Log-Osc <E>=-3.97
- 10D Gaussian: LOCR has lowest marginal variance error (max=0.96, mean=0.46)

### Key findings
- MSLO dominates on KL divergence for double-well (barrier crossing)
- LOCR shows best dimensional scaling and accuracy on isotropic targets
- GMM remains challenging for all novel samplers at this budget
- LOCR (chain variant) stability improved with Q=1.0, chain_length=2 (was unstable at Q=0.5, chain=3)

## References

- [Nose (1984)](https://doi.org/10.1063/1.447334) -- Original Nose thermostat
- [Hoover (1985)](https://doi.org/10.1103/PhysRevA.31.1695) -- Nose-Hoover formulation
- [Martyna et al. (1992)](https://doi.org/10.1063/1.463940) -- Nose-Hoover chains
- [KAM theorem](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold%E2%80%93Moser_theorem) -- Why NH fails on HO
- Parent orbits: #3 (log-osc-001), #7 (log-osc-multiT-005), #4 (log-osc-chain-002)
