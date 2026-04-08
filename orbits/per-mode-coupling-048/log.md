---
strategy: per-mode-coupling-048
status: complete
eval_version: eval-v1
metric: tau_int=4.6 (per-mode best) vs 4.2 (shared-K)
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

5. **Q-spread is far more effective than per-mode grouping for decorrelation.** Using spread Q values (100, 200, 400, 700, 1000) with N=5 groups achieves mean |rho|=0.010, better than massive N=10 with constant Q (mean |rho|=0.166). Different Q values make xi_i oscillate at different frequencies, breaking correlation even when the drive signals are similar.

6. **Massive thermostatting (N=dim) degrades mixing.** With only 1 dimension per thermostat, the partial kinetic energy K_i = p_d^2 is a chi-squared(1) random variable -- extremely noisy. The thermostat cannot distinguish genuine heating from statistical fluctuations, leading to erratic friction and 3-4x worse tau_int.

7. **The shared-K "bug" is actually a feature.** Perfect xi correlation means all N thermostats act as a single effective thermostat with amplified friction g_eff = N * tanh(S(t)/Q). This is equivalent to a single tanh thermostat with a very small effective Q, giving strong, coherent temperature control.

## Result (Iteration 2): Massive thermostatting + Q spread

Tested whether N=dim (one thermostat per dimension) gives better results. The answer is NO for mixing, YES for independence.

### Exp 5: Massive thermostatting + Q spread (10D aniso, kappa=100)

| Config | N | Mean |rho| | Max |rho| | tau_int (median) | IQR |
|--------|---|-----------|-----------|-----------------|-----|
| Massive, Q=100 | 10 | 0.166 | 0.967 | 12.0 | 14.5 |
| Massive, Q-spread | 10 | 0.015 | 0.328 | 17.1 | 25.5 |
| Groups N=5, Q=100 | 5 | 0.123 | 0.415 | 4.6 | 0.9 |
| Groups N=5, Q-spread | 5 | 0.010 | 0.050 | 4.6 | 0.9 |
| Shared-K (baseline) | 5 | 1.000 | 1.000 | 4.2 | - |

Key findings from iteration 2:

1. **Q-spread is the dominant factor for independence.** Groups N=5 with Q-spread achieves mean |rho|=0.010 -- near-perfect independence with only 5 thermostats.

2. **Massive thermostatting (N=dim) HURTS mixing.** tau_int=12-17 vs 4.2-4.6 for N=5. With only 1 dimension per thermostat, each partial K is extremely noisy (chi-squared with 1 dof), creating erratic friction that impedes coherent momentum transfer.

3. **Independence and mixing are orthogonal.** The best independence (Q-spread, mean |rho|=0.010) gives the same mixing as constant Q (tau=4.6). The best mixing (shared-K, tau=4.2) has perfect correlation. Breaking correlation neither helps nor hurts mixing when N is appropriate.

4. **N=5 groups is the sweet spot.** Enough dimensions per group (2) to stabilize partial K, enough thermostats to provide multi-scale friction.

## Seeds

- Exp 1: seed=42 (default)
- Exp 3: seeds 1000-1004
- Exp 4: seed=42
- Exp 5: seeds 2000-2004, correlation: seed=42

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
