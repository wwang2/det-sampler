---
strategy: thermostat-dynamics-046
type: experiment
status: complete
eval_version: eval-v1
metric: 1.029
issue: 46
parents:
  - orbit/q-omega-mapping-040
---

# Thermostat Internal Dynamics Study

## Glossary

- **PSD**: Power Spectral Density
- **DH band**: Dutta-Horn band -- the frequency range where a superposition of Lorentzians with log-uniform time constants produces 1/f noise
- **NHC**: Nose-Hoover Chain
- **NH**: Nose-Hoover (single thermostat variable)
- **KE**: Kinetic Energy
- **alpha**: PSD exponent -- PSD ~ f^{-alpha}, so alpha=1 means 1/f noise

## Summary

Deep diagnostic study of the multi-scale log-oscillator thermostat's internal dynamics on a 5D anisotropic Gaussian (kappa = [1, 3.16, 10, 31.6, 100]).

### Key Finding: Perfect Thermostat Correlation

All N parallel thermostat variables xi_i evolve in perfect lockstep (Pearson rho = 1.000). This occurs because they all share the same drive signal:

    dxi_i/dt = (K - dim*kT) / Q_i

where K = sum(p_d^2/m) is the total kinetic energy. Since all xi_i start at zero and are driven by the same K(t), they differ only by a Q_i-dependent scaling: xi_i(t) = integral((K-dim*kT)/Q_i) dt. Thus xi_i / xi_j = Q_j / Q_i (constant), making all xi perfectly correlated.

This means the Dutta-Horn mechanism for 1/f noise does NOT apply in its standard form. Independent Lorentzians would require independent noise sources, but our oscillators share a single drive. Despite this, for N=5 the composite PSD of Gamma(t) = sum g(xi_i) shows alpha_DH = 1.03, remarkably close to 1/f. This happens because g(xi) = 2*xi/(1+xi^2) is nonlinear, so g(xi_i) at different Q values produces different frequency content even from the same drive.

### NHC Contrast

The NHC(M=5) thermostat chain has decorrelated xi variables (rho(xi_0, xi_1) = 0.02). The chain coupling dxi_j/dt includes terms like -xi_{j+1}*xi_j that break the lockstep. This is a fundamental structural advantage of the chain architecture.

## Results

### Experiment 1: Gamma PSD vs N

| N  | alpha_DH | alpha_broad | rho(xi0,xi1) |
|----|----------|-------------|--------------|
| 3  | 1.644    | 1.563       | 1.000        |
| 5  | 1.029    | 1.201       | 1.000        |
| 10 | -0.610   | 0.323       | 1.000        |
| 20 | -1.615   | -0.879      | 1.000        |
| NHC M=5 | -- | 12.3 | 0.020 |

N=5 achieves near-perfect 1/f (alpha_DH = 1.03). N=3 is too steep, N=10,20 develop an excess of low-frequency power leading to negative slopes.

### Experiment 2: Individual xi PSDs (N=5)

Each g(xi_i) shows a quasi-Lorentzian PSD with peak near the theoretical frequency f ~ sqrt(dim*kT/Q_i)/(2*pi). Despite perfect xi correlation, the nonlinear g() function maps the common drive into frequency-separated contributions. The composite Gamma PSD approximates 1/f in the DH band.

### Experiment 3: Cross-Correlation

The Pearson cross-correlation between xi_i and q_d^2 is uniform across thermostat index i (all rows identical), confirming that no thermostat preferentially couples to any mode. The correlation is very weak (~0.01) and mode-dependent only through q_d^2 statistics.

### Experiment 4: Energy Equilibration

Starting from 5x-hot initial conditions, all three thermostats (multi-scale, NHC, NH) equilibrate total KE quickly (~0.1-0.3 time units). Per-mode equipartition is noisy but roughly satisfied for all methods.

## What I Learned

1. The parallel multi-scale thermostat has a fundamental correlation problem: all xi share the same drive. The NHC chain structure avoids this via inter-variable coupling.

2. Despite perfect correlation, the nonlinear g(xi) function provides enough frequency separation that N=5 gives alpha~1.0. This is a surprisingly robust result.

3. N is not "more is better" -- too many oscillators (N=10, 20) produce WORSE PSD slopes. The sweet spot is N~5 for this system.

4. The cross-correlation experiment confirms thermostats do NOT preferentially couple to specific modes. This is expected since they all couple through total KE.

## Prior Art & Novelty

### What is already known
- Dutta-Horn 1/f mechanism via superposition of Lorentzians with log-uniform relaxation times (Dutta & Horn 1981)
- NHC thermostat design and chain coupling (Martyna et al. 1992)
- Multi-scale Q approach for anisotropic systems (explored in parent orbits)

### What this orbit adds
- Demonstrates that parallel log-osc thermostats are perfectly correlated (rho=1) due to shared drive signal
- Shows the nonlinear g(xi) provides enough frequency separation for approximate 1/f despite correlation
- Identifies N=5 as the sweet spot; larger N degrades the PSD slope
- Contrasts with NHC where chain coupling decorrelates the thermostat variables

### Honest positioning
This orbit is a diagnostic study, not a novel method. It reveals an important limitation of the parallel multi-scale architecture and explains why N~5 works well despite a theoretical concern (perfect correlation).

## References

- Dutta & Horn (1981), "1/f noise in metals" -- original Dutta-Horn mechanism
- Martyna et al. (1992), "Nose-Hoover chains" -- NHC thermostat
