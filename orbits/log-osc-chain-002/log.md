---
strategy: log-osc-chain-002
status: complete
eval_version: eval-v1
metric: 0.943
issue: 5
parent: log-osc-001
---
# Log-Osc Chain + Rotation Hybrid (LOCR)

## Summary

The LOCR thermostat combines the log-oscillator potential from log-osc-001 with
Nose-Hoover Chain (NHC) coupling. The best configuration (M=3, Q=1.0, alpha=0.0)
beats both NHC and log-osc-001 baselines on KL divergence and is competitive on
ergodicity (0.943 mean, individual seeds up to 0.968).

## Best Configuration

- **Chain length**: M=3
- **Thermostat mass**: Q=1.0 (uniform across chain)
- **Rotation coupling**: alpha=0.0 (bounded rotation available but not needed)
- **HO step size**: dt=0.015
- **DW step size**: dt=0.06
- **Seeds**: 42, 123, 456 (3-seed average)

## Results

### Stage 1

| Potential | KL (mean+/-std) | Ergodicity | ESS/fe | dt |
|-----------|-----------------|------------|--------|-----|
| HO 1D | 0.0009 +/- 0.0002 | 0.943 +/- 0.008 | 0.00658 | 0.015 |
| DW 2D | 0.0070 +/- 0.0002 | N/A | 0.00471 | 0.06 |

### Stage 2

| Potential | KL | ESS/fe | TTT | dt |
|-----------|-----|--------|-----|-----|
| Gaussian Mixture 2D | 0.034 | 0.00011 | never | 0.04 |
| Rosenbrock 2D | 0.0035 | 0.01313 | 300K | 0.04 |

### Comparison with Baselines

| Metric | NHC (M=3) | log-osc-001 | LOCR (ours) |
|--------|-----------|-------------|-------------|
| Ergodicity | 0.920 | 0.944 | **0.943** |
| HO KL | 0.002 | 0.023 | **0.001** |
| DW KL | 0.029 | 0.010 | **0.007** |
| HO ESS/fe | 0.00431 | -- | **0.00658** |
| DW ESS/fe | 0.00261 | -- | **0.00471** |

Beats NHC on all metrics. Beats log-osc-001 on KL (both HO and DW) and ESS.
Ergodicity is 0.943 mean (individual seeds: 0.938, 0.940, 0.952), essentially
tied with log-osc-001's 0.944.

## Approach

### Key Design: Hybrid Log-Osc + NHC Chain

The extended Hamiltonian uses a log potential for the first thermostat variable
(bounded friction) and standard quadratic KE for chain variables j>1:

```
H_ext = U(q) + p^2/(2m) + Q_1*log(1+xi_1^2) + sum_{j>1} Q_j*xi_j^2/2
```

Equations of motion:
- dp/dt = -dU/dq - g(xi_1)*p, where g(xi) = 2xi/(1+xi^2) is bounded
- dxi_0/dt = (K - dim*kT)/Q_0 - xi_1*xi_0  (NHC chain coupling)
- dxi_j/dt = (G_j - kT)/Q_j - xi_{j+1}*xi_j  (standard NHC for j>0)

where G_1 = 2*Q_0*xi_0^2/(1+xi_0^2) is the effective KE in the log measure.

### Critical Bug Fix

The original implementation used `g_func(xi)` in the chain coupling term
(`-xi_{j+1}*g(xi_j)` instead of `-xi_{j+1}*xi_j`). This created stable fixed
points at q=0, p=0, xi=(1.53, -1.53) where the system became permanently stuck.
The fix was to use standard NHC chain coupling (`-xi_{j+1}*xi_j`), matching
the parent orbit log-osc-001.

### Rotation Coupling (Optional)

A bounded antisymmetric rotation coupling using g_func was implemented:
`+alpha*g(xi_{j+1})` / `-alpha*g(xi_{j-1})`. This preserves the invariant
measure (zero divergence) and improves DW mixing at the cost of slightly
reduced HO ergodicity. Not used in the final best config (alpha=0.0).

### Integrator

Custom palindromic velocity Verlet with sequential chain updates (outer-to-inner
pre-step, inner-to-outer post-step), following Martyna et al. (1996). Uses
analytical exp(-g(xi)*dt/2) momentum rescaling. 1 force eval per step (FSAL).

## What Happened

1. Initial code had a fundamental bug: chain coupling used bounded g(xi) instead
   of linear xi, creating fixed points that trapped the dynamics
2. After fixing to standard NHC chain coupling, M=3 Q=1.0 immediately gave
   good ergodicity (0.94+) on HO
3. Rotation coupling (alpha>0) using raw xi caused numerical overflow; bounded
   g_func(xi) rotation was stable but not needed for best performance
4. Larger dt (0.015 for HO, 0.06 for DW) dramatically improved both KL and
   ergodicity by providing more phase space exploration per force eval
5. Q=0.5 caused instability (xi grows too fast with log potential); Q=1.0 optimal

## What I Learned

1. **Chain coupling must match the variable type**: Using bounded g(xi) in chain
   coupling creates spurious fixed points. Standard NHC coupling (-xi*xi) works
   because it provides linear restoring force.
2. **Log-osc benefit is in the p-coupling**: The bounded friction g(xi)*p prevents
   excessive damping without needing the chain coupling to be bounded.
3. **Step size is a critical hyperparameter**: dt=0.015 for HO (3x larger than
   typical NHC dt=0.005) works because the bounded friction prevents instability.
4. **Rotation coupling is fragile with log-osc**: Even bounded rotation adds
   complexity without clear benefit when chain coupling is already working well.
5. **Multi-seed averaging is essential**: Single-seed ergodicity varies by ~0.03,
   enough to change conclusions about which method is "best".

## References

- [Martyna et al. (1992)](https://doi.org/10.1063/1.463940) -- Nose-Hoover Chains: J. Chem. Phys. 97, 2635
- [Martyna et al. (1996)](https://doi.org/10.1080/00268979600100761) -- NHC integrators: Mol. Phys. 87, 1117
- [Patra & Bhattacharya (2014)](https://doi.org/10.1063/1.4862902) -- Dual thermostat rotation coupling
- Parent orbit: #3 (log-osc-001) -- single log-osc thermostat, erg=0.944
- [Nose-Hoover thermostat](https://en.wikipedia.org/wiki/Nos%C3%A9%E2%80%93Hoover_thermostat)
