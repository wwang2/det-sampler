# ESH-Thermostat Duality: Mathematical Derivation

**Orbit:** esh-duality-018  
**Date:** 2026-04-05  
**Question:** Is ESH (Versteeg 2021, arXiv:2111.02434) a special case of our generalized thermostat Master Theorem?

---

## 1. The Two Frameworks

### 1.1 Our Generalized Thermostat (Master Theorem)

For any confining potential V(ξ), the extended dynamics:

```
dq/dt  = p
dp/dt  = -∇U(q) - g(ξ)·p,   where g(ξ) = V'(ξ)/Q
dξ/dt  = (p²/kT - 1) / Q
```

preserves the extended canonical measure:

```
ρ(q, p, ξ) ∝ exp(-U(q)/kT) · exp(-p²/(2kT)) · exp(-V(ξ)/kT)
```

**Key property:** This is a **dissipative** system. The friction term -g(ξ)p dissipates/injects energy, and the ξ-equation drives the system toward equipartition.

Known potentials:

| Name      | V(ξ)                         | g(ξ)                   | Bounded? |
|-----------|------------------------------|------------------------|----------|
| NH        | ξ²/2                         | ξ/Q                    | No       |
| Log-Osc   | Q·log(1+ξ²)                  | 2ξ/(Q(1+ξ²))           | Yes, [-2/Q, 2/Q] |
| Tanh      | Q·log(cosh(ξ))               | tanh(ξ)/Q              | Yes, [-1/Q, 1/Q] |
| Arctan    | Q·(ξ·arctan(ξ) - log(1+ξ²)/2) | arctan(ξ)/Q          | Yes, [-π/(2Q), π/(2Q)] |

### 1.2 ESH Dynamics (Versteeg 2021)

Logarithmic kinetic energy: K(v) = (d/2)·log(‖v‖²/d)

**In d dimensions:**
```
dx/dt = v / ‖v‖                    (unit-speed motion)
dv/dt = -∇U(x) · ‖v‖ / d          (force scaled by speed)
```

**In 1D (d=1), with v = p:**
```
dx/dt = sign(p)                    (always unit speed!)
dp/dt = -dU/dx · |p|              (force scaled by |p|)
```

**ESH Hamiltonian:**
```
H_ESH = U(x) + (d/2)·log(‖p‖²/d)
```

This is conserved exactly: dH_ESH/dt = 0.

---

## 2. Why ESH Is NOT a Special Case of the Master Theorem

### 2.1 The p-Equation Structure Differs Fundamentally

In our framework:
```
dp/dt = -∇U(q) - g(ξ)·p    [potential force + FRICTION term]
```

In ESH (1D):
```
dp/dt = -dU/dx · |p|         [force SCALED by speed — no separate friction]
```

These have incompatible structure. For them to be equal for generic U:
```
-∇U(q) · |p| = -∇U(q) - g(ξ)·p
```
This would require g(ξ) = (1 - |p|)·∇U/p, which depends on both ∇U and p simultaneously — impossible to satisfy with a function g(ξ) alone.

### 2.2 The ξ-Equation Structure Differs

Define ξ = log(|p|/√(kT)). Under this change of variables, ESH gives:

```
dξ/dt = d(log|p|)/dt = (1/|p|)·(dp/dt)·sign(p) = -dU/dx · sign(p)
```

This is **force-driven** — depends on ∇U(q).

In our thermostat framework (substituting p = √(kT)·exp(ξ)):
```
dξ/dt = (p²/kT - 1)/Q = (exp(2ξ) - 1)/Q
```

This is **kinetic-energy-driven** — autonomous in (ξ, p), independent of ∇U.

**These are structurally different differential equations.**

### 2.3 The Stationary Measures Differ

Our framework produces:
- **p-marginal:** Gaussian, P(p) ∝ exp(-p²/(2kT)) — proper, normalizable

ESH (1D):
- **p-marginal:** Power-law, P(p) ∝ |p|^(-1/kT)
- For kT=1: P(p) ∝ 1/|p| — **NOT normalizable** on ℝ!

In d dimensions, ESH gives P(v) ∝ ‖v‖^(-d), which has a logarithmic divergence at both ‖v‖→0 and ‖v‖→∞.

ESH does not have a proper canonical (Boltzmann) distribution in the momentum. The x-marginal is approximately correct only because the dynamics sample x-space ergodically (when they do), but the joint (x, v) measure is not canonical.

### 2.4 Conservative vs. Dissipative

| Property              | Our Thermostat            | ESH                         |
|-----------------------|---------------------------|-----------------------------|
| Type                  | Dissipative               | Conservative (Hamiltonian)  |
| Conserved quantity    | Extended Hamiltonian (ergodically distributed) | H_ESH (exactly conserved) |
| Phase-space flow      | Liouville div ≠ 0 (compressible) | Liouville div = 0 (incompressible) |
| Friction term         | Yes, -g(ξ)·p              | No                          |
| Ergodicity source     | Friction + force injection | Irrational frequency ratios |
| p-distribution        | Gaussian                  | Power-law (improper in 1D)  |

---

## 3. The Coordinate Transformation Attempt

The proposed transformation T: (q, p_ESH) → (q, p_thermo, ξ) via ξ = log(|p|/√(kT)) fails because:

1. **The transformation is not bijective** in the required sense — sign(p) information is lost in |p|
2. **The vector fields don't match** after transformation (different dependence on ∇U)
3. **No time reparameterization exists** that converts ESH into a thermostat

Specifically, under the arc-length time rescaling τ with dτ/dt = |p|/√(kT):

```
dq/dτ = sign(p)
dp/dτ = (-dU/dq - g(ξ)·p) · √(kT)/|p|
```

For this to match ESH's dp/dτ = -dU/dq·sign(p), we need g(ξ) = 0 — i.e., no thermostat. So the only thermostat that reduces to ESH under time rescaling is the trivial one with no friction.

---

## 4. ESH as Inspiration: A New Thermostat Potential

Despite the negative result (ESH ≠ our Master Theorem), ESH inspires a **new potential** in our framework.

The question: what V(ξ) would give a thermostat that, in some sense, captures ESH's adaptive speed?

The ξ-equation for our framework (with ξ = log(|p|/√(kT))) is:
```
dξ/dt = (exp(2ξ) - 1)/Q
```

This is the kinetic-energy driven version that has the same fixed point as ESH (ξ=0, i.e., |p|=√(kT)).

The V(ξ) that produces this via g(ξ) = V'(ξ)/Q = exp(2ξ) - 1:
```
V_ESH(ξ) = Q·(exp(2ξ)/2 - ξ) + const
```

**Properties of V_ESH:**
- V_ESH(0) = Q/2 (minimum)
- V_ESH → ∞ as ξ → +∞ (confining: suppresses fast particles)
- V_ESH → ∞ as ξ → -∞ (confining: suppresses slow particles)
- P(ξ) ∝ exp(-V_ESH/kT) = exp(-Q·(exp(2ξ)/2 - ξ)/kT) — normalizable!

**Friction function:**
```
g_ESH_thermo(ξ) = exp(2ξ) - 1
```
- g(0) = 0: no friction at equilibrium temperature ✓
- g(ξ) > 0 for ξ > 0: slows fast particles ✓
- g(ξ) < 0 for ξ < 0: heats slow particles ✓
- g is UNBOUNDED (unlike Log-Osc/Tanh/Arctan)

**This V(ξ) = Q·(exp(2ξ)/2 - ξ) is a valid new member of the Master Theorem family, but it is NOT one of the known examples (NH/Log-Osc/Tanh/Arctan).**

---

## 5. Comparison of Friction Functions

```
NH:            g(ξ) = ξ/Q                     [linear, unbounded]
Log-Osc:       g(ξ) = 2ξ/(Q(1+ξ²))           [bounded, max = 1/Q at ξ=1]
Tanh:          g(ξ) = tanh(ξ)/Q               [bounded, max = 1/Q]
Arctan:        g(ξ) = arctan(ξ)/Q             [bounded, max = π/(2Q)]
ESH-Thermo:    g(ξ) = (exp(2ξ) - 1)           [unbounded, exp-growing]
ESH (actual):  no friction — conservative!    [N/A]
```

ESH-Thermo has exponentially growing friction, which is very aggressive for large ξ (fast particles). This may give rapid thermalization but potentially poor ergodicity for slow particles.

---

## 6. Conclusions

### Mathematical Statement

**ESH is NOT a special case of the generalized thermostat Master Theorem.** They are parallel frameworks with fundamentally different mechanisms:

- ESH is a **conservative Hamiltonian system** with logarithmic kinetic energy
- Our thermostats are **dissipative systems** with friction-mediated thermalization

### Structural Differences (3 key facts)

1. **dp/dt differs:** ESH scales the force by |p|; thermostats add a friction term -g(ξ)·p
2. **dξ/dt differs:** ESH's log-momentum evolves via ∇U; thermostat's ξ evolves via p²
3. **Stationary measure differs:** ESH has power-law p-distribution (improper in 1D); thermostats have Gaussian p

### New Contribution

A new thermostat potential V(ξ) = Q·(exp(2ξ)/2 - ξ) is identified, inspired by ESH's fixed point (|p|=√(kT)). This is a valid Master Theorem member not previously catalogued.

### ESH + Multi-scale

ESH could be extended with a multi-scale approach (like our NHCTail concept) by running multiple ESH trajectories at different |p| scales and combining samples. However, since ESH is conservative, multi-scale mixing would require either:
- Stochastic velocity refreshments (making it hybrid/MCMC)
- Or a sequence of ESH runs with different initial |p| values

This is less principled than our thermostat multi-scale approach, which achieves multi-scale exploration through the chain structure.

---

## References

1. Versteeg, V. et al. (2021). "Energy-Sampler-Hamiltonian dynamics." arXiv:2111.02434
2. Nosé, S. (1984). J. Chem. Phys. 81, 511.
3. Hoover, W. G. (1985). Phys. Rev. A, 31, 1695.
4. Martyna, G. J. et al. (1992). J. Chem. Phys. 97, 2635.
