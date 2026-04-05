---
strategy: publication-figures
status: in-progress
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
| Fig 4 | Quantitative Comparison | Running | Bar charts: ergodicity, KL on double-well, KL on GMM |
| Fig 5 | Multi-Scale Mechanism | Done | Friction signals, xi timeseries, PSD, GMM mode coverage |
| Fig 6 | Scaling to High Dimensions | Running | ESS vs DOF, LJ-7 energy, 10D Gaussian variance errors |

## Approach

Each figure script is self-contained with:
- Fixed random seeds (42, 123, 7 for multi-seed runs)
- Style guide compliance: 14pt labels, 12pt ticks, 16pt titles, 300 DPI
- Panel labels (a)-(f) in upper-left
- Consistent colors: NH=#1f77b4, NHC=#ff7f0e, novel samplers from tab10(2+)

## Key Results from Figure 4 (first run)

- Ergodicity scores: NH=0.68, NHC=0.93, Log-Osc=0.90, MSLO=0.93
- KL double-well: NHC=0.011, MSLO=0.005 (best), NH=0.55
- KL GMM: NHC=0.10, NH=0.37, MSLO=0.40

MSLO achieves best KL on double-well. NHC remains strong baseline.
LOCR (chain variant) had stability issues with Q=0.5, chain=3; revised to Q=1.0, chain=2.

## References

- [Nose (1984)](https://doi.org/10.1063/1.447334) -- Original Nose thermostat
- [Hoover (1985)](https://doi.org/10.1103/PhysRevA.31.1695) -- Nose-Hoover formulation
- [Martyna et al. (1992)](https://doi.org/10.1063/1.463940) -- Nose-Hoover chains
- [KAM theorem](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold%E2%80%93Moser_theorem) -- Why NH fails on HO
- Parent orbits: #3 (log-osc-001), #7 (log-osc-multiT-005), #4 (log-osc-chain-002)
