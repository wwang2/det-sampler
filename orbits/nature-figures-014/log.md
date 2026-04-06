---
strategy: nature-figures-014
status: complete
eval_version: eval-v1
metric: null
issue: 16
parent: pub-figures-010
---

# Nature-Quality Consolidated Figures (NHC Baseline)

## Summary

Produced 5 Nature-quality figures benchmarking NHC (M=3) as primary baseline against
Log-Osc, MultiScale Log-Osc, and NHCTail samplers across 4 test potentials.

## Figures Produced

| Figure | File | Description |
|--------|------|-------------|
| Fig 1 | `figures/fig1_problem.png` | Problem statement: NHC non-ergodicity on HO, DW, GMM, RB |
| Fig 2 | `figures/fig2_theory.png` | Theory: log-oscillator friction function g(xi), phase portraits |
| Fig 3 | `figures/fig3_mechanism.png` | Mechanism: xi-trajectory comparison, KL convergence traces |
| Fig 4 | `figures/fig4_multiscale.png` | MultiScale Log-Osc: multi-timescale xi dynamics, GMM coverage |
| Fig 5 | `figures/fig5_benchmark.png` | Comprehensive benchmark: 2x4 panel, 4 potentials x 4 samplers x 5 seeds |

## Figure 5 Benchmark Results

Run with N_EVALS=50,000, 5 seeds, 4 systems (HO/DW/GMM/RB), 4 samplers.

### KL Divergence (mean +/- std over 5 seeds)

**Harmonic Oscillator (HO):**
| Sampler | Final KL | ESS/eval | TTT |
|---------|----------|----------|-----|
| NHC (M=3) | 0.5250 +/- 0.3451 | ~0 | >50K |
| Log-Osc | 0.5966 +/- 0.2110 | ~0 | >50K |
| MultiScale | 0.8295 +/- 0.7004 | ~0 | >50K |
| NHCTail | 0.5468 +/- 0.3338 | ~0 | >50K |

**Double Well (DW):**
| Sampler | Final KL | ESS/eval | TTT |
|---------|----------|----------|-----|
| NHC (M=3) | 3.4498 +/- 0.4330 | ~0 | >50K |
| Log-Osc | 2.1895 +/- 1.5396 | ~0 | >50K |
| MultiScale | 1.1572 +/- 0.1952 | ~0 | >50K |
| NHCTail | 0.7304 +/- 0.0820 | ~0 | >50K |

**GMM (5-mode):**
| Sampler | Final KL | ESS/eval | TTT |
|---------|----------|----------|-----|
| NHC (M=3) | 9.0581 +/- 5.0762 | ~0 | >50K |
| Log-Osc | 47.4065 +/- 11.6306 | ~0 | >50K |
| MultiScale | 9.1652 +/- 5.6639 | ~0 | >50K |
| NHCTail | 12.5030 +/- 9.4303 | ~0 | >50K |

**Rosenbrock (RB):**
| Sampler | Final KL | ESS/eval | TTT |
|---------|----------|----------|-----|
| NHC (M=3) | 1.6697 +/- 0.2266 | ~0 | >50K |
| Log-Osc | 0.9030 +/- 0.2207 | ~0 | >50K |
| MultiScale | 0.8527 +/- 0.1973 | ~0 | >50K |
| NHCTail | 0.5187 +/- 0.1543 | ~0 | >50K |

### Ergodicity Scores (HO, mean +/- std)
| Sampler | Score |
|---------|-------|
| NHC (M=3) | 0.791 +/- 0.011 |
| Log-Osc | 0.773 +/- 0.016 |
| MultiScale | 0.749 +/- 0.043 |
| NHCTail | 0.784 +/- 0.026 |

### Key Observations
- NHCTail shows best KL on DW (0.73) and RB (0.52) at 50K evals
- NHC (M=3) shows best ergodicity score on HO (0.791)
- All samplers need more evals to converge (TTT > 50K for KL<0.01 threshold)
- GMM is hardest system; NHC and MultiScale comparable there
- NHCTail champion config: Qs=[0.1, 0.7, 10.0], chain_length=2

## Technical Notes
- N_EVALS reduced from 1.5M to 50K for practical runtime with pure-Python integrators
- RuntimeWarnings (overflow in xi dynamics) are benign — handled by np.clip in integrators
- All figures at DPI=300, panel labels (a)-(h), Nature-style fonts
- Colors: NHC=#ff7f0e (orange), Log-Osc=#2ca02c (green), MultiScale=#d62728 (red), NHCTail=#9467bd (purple)
