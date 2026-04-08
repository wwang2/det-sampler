---
strategy: per-mode-coupling-048
status: in-progress
eval_version: eval-v1
metric: tau_int=5.0 (per-mode) vs 4.1 (shared-K)
issue: 48
parents:
  - orbit/paper-experiments-047
  - orbit/friction-survey-045
---

# Per-Mode Thermostat Coupling: Breaking xi Correlation

## Glossary

- **Per-mode coupling**: Each thermostat xi_i is driven by a subset of momentum dimensions, not the full kinetic energy
- **Shared-K**: Standard parallel thermostat where all xi_i share the same drive signal K(t) = sum(p^2)
- **NHC**: Nose-Hoover Chain (Martyna et al. 1992)
- **NH**: Single Nose-Hoover thermostat
- **GMM**: Gaussian Mixture Model
- **tau_int**: Integrated autocorrelation time of q^2
- **PSD**: Power Spectral Density

## Result (Iteration 1)

Per-mode coupling successfully breaks the perfect xi correlation (rho=1.000 -> mean |rho|=0.216), but does NOT improve mixing: tau_int=5.0 vs shared-K=4.1, and PSD becomes steeper (slope=-4.05 vs +0.27), not flatter.

### Exp 1: xi independence

| Metric | Per-mode | Shared-K |
|--------|----------|----------|
| Mean |rho| off-diag | 0.216 | 1.000 |
| Max |rho| off-diag | 0.941 | 1.000 |

Per-mode breaks perfect correlation but residual coupling persists through Hamiltonian dynamics. Adjacent-stiffness groups (xi_1, xi_2) show -0.94 anti-correlation. The low-stiffness group (xi_0) is essentially independent (|rho| < 0.01).

### Exp 2: PSD of Gamma(t)

| Metric | Per-mode | Shared-K |
|--------|----------|----------|
| PSD slope | -4.05 | +0.27 |

Per-mode produces a narrowband, steep-spectrum friction signal. Shared-K produces a flatter spectrum. Neither achieves 1/f noise.

### Exp 3: Mixing comparison (5 seeds, 400k steps, d=10)

| Method | tau_int (aniso) | GMM crossings |
|--------|----------------|---------------|
| Per-mode tanh (N=5) | 5.0 (IQR=4.3) | 546 (IQR=15) |
| Shared-K tanh (N=5) | 4.1 (IQR=0.0) | 413 (IQR=282) |
| NHC (M=3) | 4.6 (IQR=0.2) | 588 (IQR=165) |
| NH | 4.1 (IQR=0.0) | 508 (IQR=304) |

Per-mode is slightly worse on tau_int but has lower variance on GMM crossings.

### Exp 4: Cross-correlation xi_i vs q_d^2

Expected block-diagonal structure was NOT observed. Instead, high-stiffness thermostats (xi_3, xi_4) correlate with low-stiffness position modes. In-group mean correlation (0.0005) is LOWER than out-group (0.0875).

## Approach

Standard parallel thermostats couple ALL xi_i to the same K(t) = sum(p_d^2), giving xi_i(t) proportional to S(t)/Q_i and perfect correlation rho=1. Per-mode coupling assigns each xi_i to a subset G_i of dimensions:

  dxi_i/dt = (sum_{d in G_i} p_d^2 - |G_i| kT) / Q_i

Each p_d gets friction only from its assigned thermostat:

  dp_d/dt = -dU/dq_d - g(xi_{i(d)}) p_d

The invariant measure is preserved because each (p_d, xi_{i(d)}) pair satisfies the NH-type cancellation independently.

### Grouping strategy

Sort dimensions by kappa, split into N=5 contiguous groups of 2 dimensions each. This ensures each thermostat sees a distinct frequency band of the dynamics.

## What Happened

1. Per-mode coupling successfully breaks perfect xi correlation (mean |rho| drops from 1.000 to 0.216)
2. However, residual correlations remain strong between adjacent-stiffness groups (-0.94 for xi_1-xi_2)
3. The PSD becomes STEEPER, not flatter -- the opposite of the Dutta-Horn 1/f prediction
4. Mixing (tau_int) is not improved; per-mode is slightly worse
5. Cross-correlation shows INVERTED structure: thermostats correlate most with modes OUTSIDE their group

## What I Learned

1. Breaking the shared-K drive is necessary but not sufficient for independent xi. The Hamiltonian dynamics create cross-group coupling through the potential energy surface.

2. For an anisotropic Gaussian, U(q) = sum kappa_d q_d^2 / 2, the dimensions are completely decoupled in the potential. The residual correlation comes from the fact that different groups partial kinetic energies fluctuate on similar timescales when kappas are similar.

3. The inverted cross-correlation is physically intuitive: xi_4 (controlling stiff modes d=8,9 with kappa=60-100) fluctuates rapidly, creating time-varying friction that affects its assigned momenta. Through energy conservation, this couples to position fluctuations of ALL modes, but the soft modes (low kappa) have the largest variance, so the correlation appears strongest there.

4. Per-mode coupling may be more useful when there are more dimensions per group (so partial K averages are more distinct) or when the potential has stronger cross-dimension coupling.

## Seeds

- Exp 1: seed=42 (default)
- Exp 3: seeds 1000-1004
- Exp 4: seed=42

## Prior Art & Novelty

### What is already known
- "Massive thermostatting" -- assigning separate NH thermostats to each degree of freedom -- is a well-known technique: [Tobias et al. (1993)](https://doi.org/10.1021/j100120a038)
- Per-atom thermostats in MD: each atom or group of atoms has its own NH thermostat. Standard in MD packages (LAMMPS, NAMD).
- Invariant measure preservation for independent NH thermostats: covered in [Martyna et al. (1992)](https://doi.org/10.1063/1.463940)

### What this orbit adds (if anything)
- Empirical verification that per-mode coupling breaks the perfect xi correlation found in orbit #046
- Demonstration that breaking correlation does NOT automatically improve mixing
- Discovery of inverted cross-correlation structure
- Quantitative comparison against shared-K parallel and NHC baselines

### Honest positioning
Per-mode thermostat coupling is equivalent to "massive thermostatting" applied to groups of normal modes rather than individual atoms. This is a known technique in molecular dynamics. The orbit's contribution is testing whether it fixes the correlation problem identified in #046 and measuring whether independence translates to better sampling. The answer is nuanced: independence is partially achieved but does not improve mixing metrics.

## References

- Tobias, Martyna, Klein (1993) "Molecular dynamics simulations of a protein in mixed solvent" J. Phys. Chem. 97, 12959
- Martyna, Klein, Tuckerman (1992) "Nose-Hoover chains" J. Chem. Phys. 97, 2635
- Orbit #046: discovered rho=1.000 perfect correlation in standard parallel thermostats
- Orbit #047: paper-experiments establishing tanh with Q=100 as competitive with NHC
- Orbit #045: friction survey confirming tanh as best bounded friction function
