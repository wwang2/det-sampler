---
strategy: pub-narrative-011
status: complete
eval_version: eval-v1
metric: N/A (narrative/figures orbit)
issue: 11
parent: unified-theory-007
---

# Publication Narrative and Schematic Figures

## Summary

Created the full paper narrative outline and 7 publication-quality figures
(5 conceptual schematics + 2 toy system illustrations) for the generalized
friction thermostat paper.

## Deliverables

### Paper Outline (paper_outline.md)

Six-section narrative with engaging physicist-targeted paragraphs:

1. **Introduction** -- The KAM problem, why NH fails, what we do differently
2. **Generalized Friction Framework** -- Master theorem in accessible language
3. **Why Bounded Friction Works** -- ABS braking analogy, Lyapunov evidence, two-regime theory
4. **Multi-Scale Dynamics** -- 1/f spectrum analogy, log-spaced Q values
5. **Results** -- Progressive difficulty (HO -> DW -> GMM -> LJ-13)
6. **Discussion** -- Proven vs conjectured, open questions, Langevin connection

Each section includes key insight boxes and figure placement suggestions.

### Schematic Figures

| Figure | Description | Key Feature |
|--------|-------------|-------------|
| fig_schematic_thermostat.png | Thermostat concept diagram | NH vs Log-Osc coupling knob analogy |
| fig_schematic_friction.png | Friction function gallery | g(xi), V(xi), p(xi) side-by-side for 4 thermostats |
| fig_schematic_kam.png | KAM tori vs ergodic orbits | Phase space structure comparison |
| fig_schematic_multiscale.png | Multi-scale thermostat | 3 oscillators + combined broadband signal |
| fig_schematic_progression.png | Discovery progression timeline | NH -> NHC -> Log-Osc -> Multi-Scale -> LOCR |

### Toy System Illustrations

| Figure | Description | Key Feature |
|--------|-------------|-------------|
| fig_toy_doublewell.png | 1D double-well dynamics | Potential + trajectory overlay + time series with 89 crossings |
| fig_toy_harmonic.png | HO phase portrait comparison | NH torus vs Log-Osc space-filling + marginal histograms |

## Seeds

All simulations use seed=42. Deterministic and reproducible via run.sh.

## References

- [Nose (1984)](https://doi.org/10.1063/1.447334) -- original thermostat
- [Hoover (1985)](https://doi.org/10.1103/PhysRevA.31.1695) -- canonical dynamics
- [Martyna et al. (1992)](https://doi.org/10.1063/1.463940) -- Nose-Hoover chains
- [KAM theorem](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold%E2%80%93Moser_theorem) -- non-ergodicity mechanism
- [1/f noise](https://en.wikipedia.org/wiki/Pink_noise) -- spectral density connection
- Builds on #3 (log-osc-001) which discovered the log-osc thermostat
- Builds on #10 (unified-theory-007) which proved the master theorem
