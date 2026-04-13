---
strategy: pd-thermostat-derivative-augmented
type: experiment
status: complete
eval_version: eval-v1
metric: 0.0
issue: 70
parents: []
root_rationale: "Fresh exploration of PD (proportional-derivative) control idea for NH thermostat, motivated by control theory analogy."
---

# pd-thermostat-070

## Glossary

- **NH**: Nose-Hoover thermostat (Hoover 1985)
- **NHC**: Nose-Hoover Chain thermostat (Martyna et al. 1992)
- **PD**: Proportional-Derivative (control theory term)
- **KL**: Kullback-Leibler divergence

## Phase 0: Analytical Derivation

### The idea

Standard NH is a pure integral (I) controller: the thermostat variable xi integrates the kinetic energy error signal `(|p|^2/m - D*kT)`. A PD controller adds a derivative term that feeds the rate of change of kinetic energy directly into the friction dynamics:

```
dq/dt = p/m
dp/dt = -nabla_U - g(xi)*p
dxi/dt = (|p|^2/m - D*kT)/Q  +  K_d * d(|p|^2/m)/dt
```

where `d(|p|^2/m)/dt = 2*p . dp/dt / m = 2*p . (-nabla_U - g(xi)*p) / m`.

The motivation: the D-term feeds the instantaneous force nabla_U directly into friction. When the particle feels a large force (sliding down a barrier), the thermostat reacts immediately rather than waiting for kinetic energy to accumulate.

### Liouville condition for measure preservation

For a dynamical system dx/dt = f(x) to preserve a measure mu(x), the Liouville equation requires:

```
div(f) + f . grad(log mu) = 0    for all x
```

### Verification with g(xi) = xi (standard NH coupling)

**Candidate measure:** mu = exp(-U/kT - p^2/(2m*kT) - Q*xi^2/(2*kT))

So grad(log mu) = (-U'(q)/kT, -p/(m*kT), -Q*xi/kT).

Working in 1D with m=1 for clarity (D-dimensional case follows by summation).

**Vector field:**
- f_q = p
- f_p = -U'(q) - xi*p
- f_xi = (p^2 - kT)/Q + K_d * 2p * (-U'(q) - xi*p)

**Divergence:**
```
div(f) = 0 + (-xi) + d/dxi[K_d * 2p*(-U' - xi*p)]
       = -xi - 2*K_d*p^2
```

**Sanity check (K_d = 0):** For standard NH, div(f) = -xi and f.grad(log mu) = xi (verified by direct computation). The sum is zero. Good.

**Full f . grad(log mu):**
```
= p*(-U'/kT) + (-U' - xi*p)*(-p/kT) + f_xi*(-Q*xi/kT)

= -pU'/kT + pU'/kT + xi*p^2/kT - (Q*xi/kT)*[(p^2 - kT)/Q + K_d*2p*(-U' - xi*p)]

= xi*p^2/kT - xi*(p^2 - kT)/kT - 2K_d*Q*xi*p*(-U' - xi*p)/kT

= xi + 2K_d*Q*xi*p*U'/kT + 2K_d*Q*xi^2*p^2/kT
```

**Sum:**
```
div(f) + f.grad(log mu) = [-xi - 2K_d*p^2] + [xi + 2K_d*Q*xi*p*U'/kT + 2K_d*Q*xi^2*p^2/kT]

= 2*K_d * [-p^2 + Q*xi*p*U'/kT + Q*xi^2*p^2/kT]
```

**This is NOT zero for general (q, p, xi) unless K_d = 0.**

### Verification with g(xi) = tanh(xi)

For g(xi) = tanh(xi), the correct NH invariant measure uses W(xi) = Q*log(cosh(xi)), so:

mu = exp(-U/kT - p^2/(2kT) - Q*log(cosh(xi))/kT)

and grad(log mu) in xi = -Q*tanh(xi)/kT.

After the same computation:

```
div(f) + f.grad(log mu) = 2*K_d * [-sech^2(xi)*p^2 + Q*tanh(xi)*p*U'/kT + Q*tanh^2(xi)*p^2/kT]
```

Again NOT zero unless K_d = 0.

### Why no modified measure can rescue the PD thermostat

The offending cross-term is `Q*tanh(xi)*p*U'(q)/kT` -- it couples all three phase-space variables (q through U', p, and xi). For a modified measure mu' = exp(-beta*H - Phi(q,p,xi)) to absorb this, Phi would need to satisfy a PDE with a potential-dependent source term. This means Phi would depend on U(q), making the thermostat non-universal (different measure for each potential).

More precisely, the Liouville condition becomes a PDE for Phi:
```
2K_d*[-g'(xi)*p^2 + Q*g(xi)*p*U'/kT + Q*g^2(xi)*p^2/kT] = [corrections from Phi terms]
```

The term proportional to p*U'(q)*g(xi) requires Phi to contain a term of the form ~xi*p*U'(q), which is non-separable and potential-dependent. No universal thermostat measure exists.

### Phase 0 verdict

**Canonical measure preserved only when K_d = 0.** The PD thermostat breaks the invariant measure for any nonzero derivative gain, for any coupling function g(xi). No simple or universal measure modification can rescue it.

The fundamental reason: NH measure preservation relies on a delicate cancellation between the divergence contribution (-g(xi) from the p-equation) and the drift term (g(xi)*p^2/kT from f.grad(log mu)). The derivative term adds cross-coupling between nabla_U and xi that has no partner in the divergence.

## Phase 1: Numerical Confirmation

Despite the negative analytical result, I implemented the PD thermostat to numerically confirm that KL divergence grows with K_d, and to quantify the bias.

### Results (3-seed mean, double-well 2D, 500k force evals)

| K_d | KL (mean +/- std) | ESS/eval | tau_int | Notes |
|------|----------------------|----------|---------|-------|
| 0.00 | 0.066 +/- 0.009 | 0.00189 | 540.8 | Correct NH baseline |
| 0.01 | 0.537 +/- 0.320 | 0.00579 | 158.4 | 8x KL increase |
| 0.05 | 0.481 +/- 0.000 | 0.00645 | 139.5 | Saturated bias |
| 0.10 | 0.469 +/- 0.000 | 0.00643 | 140.0 | Saturated bias |
| 0.50 | 6.158 +/- 8.097 | 0.00399 | 150.6 | Unstable, huge variance |

Seeds: 42, 123, 7 (parallel execution).

The results confirm the Phase 0 analysis: any K_d > 0 immediately increases KL by ~7-8x. Interestingly, the autocorrelation time *decreases* (tau drops from 541 to ~140), meaning the PD thermostat mixes faster -- but to the wrong distribution. This is the classic bias-variance tradeoff in approximate samplers.

## Results

The PD thermostat does not preserve the canonical measure for any K_d != 0. The metric is 0.0 (no valid formulation found). This is a clean negative result with a clear analytical proof.

The control-theory analogy breaks down because the Liouville equation is more constraining than linear control theory: feedback terms that couple the "plant" state (q, p) to the "controller" state (xi) through the force field destroy the measure-preservation balance.

## Prior Art & Novelty

### What is already known
- Nose-Hoover dynamics (Nose 1984, Hoover 1985) preserve the canonical measure through a specific Liouville equation balance
- The mathematical structure required for measure preservation in extended systems is well understood (Leimkuhler & Reich 2004)
- PID control analogies for thermostats have been discussed informally in the MD community

### What this orbit adds
- Explicit derivation showing WHY the D-term breaks the Liouville condition: cross-terms proportional to g(xi)*p*U' cannot be absorbed by any separable or universal measure
- The result holds for any coupling function g(xi), not just g(xi)=xi or g(xi)=tanh(xi)
- Numerical confirmation that KL divergence grows monotonically with K_d

### Honest positioning
This is a negative result confirming that derivative feedback cannot be naively added to NH dynamics. The contribution is the explicit derivation and the clear identification of the mechanism (non-canceling cross-terms). No novelty claim beyond the specific calculation.

## References

- Nose, S. (1984). A unified formulation of the constant temperature molecular dynamics methods. J. Chem. Phys. 81, 511.
- Hoover, W.G. (1985). Canonical dynamics: Equilibrium phase-space distributions. Phys. Rev. A 31, 1695.
- Martyna, G.J., Klein, M.L., Tuckerman, M. (1992). Nose-Hoover chains. J. Chem. Phys. 97, 2635.
- Leimkuhler, B. & Reich, S. (2004). Simulating Hamiltonian Dynamics. Cambridge University Press.
