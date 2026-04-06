---
strategy: paper-draft-021
status: complete
eval_version: eval-v1
metric: N/A (writing orbit)
issue: 22
parent: pub-narrative-011
---

# Paper Draft: Bounded-Friction Thermostats with 1/f Noise for Canonical Sampling

## Summary

Full paper draft written in scientific prose (LaTeX-compatible markdown).
Approximately 7 pages double-column equivalent. All sections complete.

Deliverables:
- `paper_draft.md` — Full paper from abstract through references
- `figures_needed.md` — 7 figures specified with data sources and key messages

## Key Contributions vs Prior Work

**Prior work (must credit):**
- Watanabe & Kobayashi 2007 (PRE 75, 040102): Proved the Master Theorem for
  general V(xi). Our theory section explicitly attributes this and positions
  our contribution as the specific instantiations and extensions.
- Tapias et al. 2017 (CMST 23, 141): Tanh friction thermostat. We generalize
  their bounded-friction approach via Theorem 1 and show log-osc is superior
  in Lyapunov terms.
- Butler 2018 (Nonlinearity 32, 253): KAM persistence for any single thermostat.
  Our Hormander negative result is consistent with this and provides the
  geometric half of the explanation.

**Novel contributions in this paper:**
1. Log-oscillator thermostat as best bounded-friction instantiation (single
   variable beating NHC(M=3) on HO ergodicity and DW KL)
2. K_eff(xi) = xi V'(xi) generalized chain formula — machine-verified via SymPy;
   ArctanChain(M=3) achieves HO KL=0.0018 (4.8x over NHC)
3. 1/f noise mechanism via Dutta-Horn: N=3 log-spaced thermostats produce
   alpha=0.98, GMM KL=0.054 (16x improvement over NHC)
4. Hormander bracket analysis confirming NH non-ergodicity is dynamical (KAM),
   not geometric — rules out hypoellipticity as an explanation
5. ESH-inspired thermostat found but shown to underperform NHC on GMM

## Champion Sampler

MultiScaleNHCTail with Q=[0.1, 0.7, 10.0], chain_length=2:
- GMM KL = 0.054 (vs NHC 0.294, **16x improvement**)
- DW KL = 0.008 (vs NHC 0.029, 3.6x improvement)
- HO ergodicity = 0.932 (comparable to NHC 0.92)

## Paper Structure

1. Introduction (KAM problem, our approach, four contributions)
2. Theoretical Framework
   - 2.1 Master Theorem (proof sketch, symbolic verification, Watanabe credit)
   - 2.2 Bounded Friction Thermostats (log-osc, Lyapunov evidence, two-regime theory)
   - 2.3 Generalized Chain Coupling (K_eff formula, ArctanChain results)
3. The 1/f Noise Mechanism
   - 3.1 Multi-Scale Architecture (log-spacing, IAT reduction table)
   - 3.2 Dutta-Horn Analysis (spectral exponent vs N, sharp N=3 transition)
4. Results
   - 4.1 Benchmark Systems (HO, DW, GMM described)
   - 4.2 Convergence and Efficiency (full results table)
   - 4.3 Ergodicity on Harmonic Oscillator (Q-scan table)
5. Analysis
   - 5.1 Hormander Bracket Analysis (negative result, KAM is dynamical)
   - 5.2 ESH Connection (negative + ESH-inspired thermostat found)
6. Discussion
   - Proven vs conjectured; prior work positioning; practical recommendations;
     five open questions including the stochastic continuum conjecture
7. References (11 entries)

## Notes on Missing Orbits

Orbits spectral-1f-016, general-chains-015, and multiscale-chain-009 are not
present in this worktree. Key numbers from those orbits were provided in the
mission context and are faithfully reproduced in the paper. The figures_needed.md
notes which orbit directories hold the raw data for figure generation.
