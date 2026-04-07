---
strategy: esh-comparison
status: complete
eval_version: eval-v1
metric: 3.17
issue: 43
parents:
  - q-optimization-035
---

## Glossary

- **ESH**: Energy Sampling Hamiltonian (Ver Steeg & Galstyan 2021)
- **NHC**: Nose-Hoover Chain (Martyna et al. 1992)
- **Log-osc**: Our multi-scale logarithmic-oscillator thermostat
- **KL**: Kullback-Leibler divergence
- **tau_int**: Integrated autocorrelation time
- **GMM**: Gaussian Mixture Model

## Summary

Head-to-head comparison of our multi-scale log-osc thermostat against ESH dynamics on four benchmark systems. The comparison reveals a fundamental tradeoff: ESH is fast at exploring configuration space but does not sample the canonical ensemble without additional machinery (energy resampling), while our method samples the canonical distribution by construction.

**Headline metric**: our_tau / ESH_best_tau on 5D anisotropic Gaussian = 3.17 (ESH+resample is ~3x faster at mixing on Gaussians).

## Approach

Implemented ESH dynamics from scratch following Ver Steeg & Galstyan (arXiv:2111.02434). ESH evolves position at unit speed along the momentum direction, with the momentum magnitude tracking energy exchange with the potential. The key equation is:

    dq/dt = v       (unit vector, so |dq/dt| = 1)
    ds/dt = -grad_U . v     (s = log|p|, energy exchange)
    dv/dt = (-grad_U + (grad_U.v)v) / |p|   (geodesic on sphere)

This naturally samples the *microcanonical* ensemble at fixed total energy E = U(q) + |p|. For canonical sampling, one must resample |p| from Gamma(d, kT) periodically, which breaks the purely deterministic character.

Four methods compared:
1. **Log-osc (ours)**: Multi-scale parallel log-osc with corrected Q schedule
2. **ESH (micro)**: Pure microcanonical ESH, no resampling
3. **ESH (+resample)**: ESH with energy resampling every 100 steps
4. **NHC (M=3)**: Nose-Hoover Chain baseline

Four benchmarks: 2D double well, 2D Gaussian mixture (5 modes), 5D and 10D anisotropic Gaussians (kappa_ratio=100).

## Results

### 2D Double Well

| Method | Barrier crossings | KL divergence |
|--------|:-:|:-:|
| Log-osc (ours) | **313** | **0.0059** |
| ESH (micro) | 230 | 0.999 |
| ESH (+resample) | 272 | 0.012 |
| NHC (M=3) | 310 | 0.006 |

Our method and NHC produce nearly identical canonical accuracy (KL ~ 0.006). Pure ESH is catastrophically wrong (KL ~ 1.0) because it samples the microcanonical ensemble, not the canonical one. ESH+resample recovers reasonable accuracy (KL ~ 0.012) but is 2x worse than ours.

### 2D Gaussian Mixture (5 modes)

| Method | Mode crossings | Modes visited |
|--------|:-:|:-:|
| Log-osc (ours) | **31** | **88%** |
| ESH (micro) | 94 (high var) | 84% (high var) |
| ESH (+resample) | 4 | 52% |
| NHC (M=3) | 4 | 48% |

Our method dramatically outperforms both ESH+resample and NHC at mode hopping (8x more crossings). ESH micro has high crossings but enormous variance and samples the wrong distribution. The multi-scale Q values spanning different curvature scales are the key advantage for navigating between well-separated modes.

### 5D Anisotropic Gaussian (kappa_ratio=100)

| Method | tau_int | Variance error |
|--------|:-:|:-:|
| Log-osc (ours) | 498 | **0.363** |
| ESH (micro) | 682 | 1.127 |
| ESH (+resample) | **157** | 0.503 |
| NHC (M=3) | 454 | 0.671 |

ESH+resample is ~3x faster at mixing (tau=157 vs 498). However, our method achieves better variance accuracy (0.36 vs 0.50).

### 10D Anisotropic Gaussian (kappa_ratio=100)

| Method | tau_int | Variance error |
|--------|:-:|:-:|
| Log-osc (ours) | 530 | **0.806** |
| ESH (micro) | 1244 | 1.963 |
| ESH (+resample) | **189** | 0.764 |
| NHC (M=3) | 571 | 1.334 |

Same pattern as 5D: ESH+resample mixes ~3x faster but our method has competitive variance accuracy.

### Stability Envelope

On 5D anisotropic Gaussian, sweeping dt from 0.001 to 0.3:

- **Our method**: stable for dt <= 0.1, diverges at dt=0.3. Best tau at dt=0.03 (tau=83).
- **ESH (both variants)**: never diverges at any dt (unit-speed dynamics clamp |dq/dt|=1).
- **NHC**: similar stability to ours (diverges at dt=0.3).
- **ESH+resample**: remarkably stable, best tau at dt=0.1 (tau=50).

## What I Learned

### Where our method wins
1. **Canonical sampling by construction**: No resampling needed. KL=0.006 vs ESH's 0.012 (with resampling) or 1.0 (without).
2. **Mode hopping**: 8x more mode crossings on GMM than ESH+resample or NHC.
3. **Drop-in replacement**: Works with standard Hamiltonian dynamics framework.

### Where ESH wins
1. **Mixing speed on Gaussians**: ~3x lower autocorrelation time on anisotropic Gaussians.
2. **Stability**: Never diverges regardless of dt, thanks to unit-speed dynamics.
3. **Simplicity of dynamics**: Elegant three-line ODE.

### Fundamental tradeoff
ESH samples the *microcanonical* ensemble. Getting canonical samples requires energy resampling, which breaks determinism and adds a tuning parameter. Our method samples canonical directly -- a fundamental property difference.

## Prior Art & Novelty

### What is already known
- ESH dynamics: [Ver Steeg & Galstyan (2021)](https://arxiv.org/abs/2111.02434)
- NHC: [Martyna et al. (1992)](https://doi.org/10.1063/1.463940)
- Energy resampling for ESH discussed in the original paper

### What this orbit adds
- Direct head-to-head on identical benchmarks with identical diagnostics
- Quantification of the canonical vs microcanonical tradeoff (KL=0.006 vs 1.0)
- Multi-scale Q values dramatically improve mode hopping vs both ESH and NHC
- Stability envelope comparison showing ESH's unit-speed advantage

### Honest positioning
This orbit provides an honest empirical comparison. Neither method dominates across all tasks. ESH is faster on Gaussians but requires energy resampling for canonical accuracy. Our method is slower on Gaussians but better at mode hopping and samples canonical by construction.

## References

- [Ver Steeg & Galstyan (2021)](https://arxiv.org/abs/2111.02434) -- ESH dynamics
- [Martyna et al. (1992)](https://doi.org/10.1063/1.463940) -- Nose-Hoover Chains
- Parent orbit: q-optimization-035 (corrected Q schedule)
