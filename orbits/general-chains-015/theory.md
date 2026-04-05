# Generalized Chain Theorem: Universal Chain Coupling via K_eff

## Abstract

We derive the universal chain coupling formula for extending Nose-Hoover-style
thermostat chains to arbitrary confining potentials V(xi). The key insight is
that the effective kinetic energy coupling between chain levels is:

    K_eff(xi) = xi * V'(xi)

This generalizes the standard NHC coupling G_j = Q_{j-1} * xi_{j-1}^2 (which
arises when V is quadratic) to any admissible thermostat potential. We prove
that <K_eff> = kT for any confining V, derive the generalized chain dynamics,
and verify the Liouville equation for the full chain.

---

## 1. The Effective Kinetic Energy

### 1.1 Motivation

In standard Nose-Hoover Chains (Martyna et al. 1992), each thermostat variable
xi_j has a "kinetic energy" Q_j * xi_j^2 that drives the next thermostat in
the chain. This works because the marginal distribution of xi_j is:

    rho(xi_j) ~ exp(-Q_j * xi_j^2 / (2kT))

so <Q_j * xi_j^2> = kT (equipartition).

But what if the thermostat potential is NOT quadratic? For a general confining
V(xi), the marginal is:

    rho(xi) ~ exp(-V(xi) / kT)

and the "kinetic energy" Q * xi^2 no longer has the right expectation value.

### 1.2 Definition

**Definition.** For a thermostat potential V(xi), the *effective kinetic energy*
is:

    K_eff(xi) = xi * V'(xi)

### 1.3 Theorem (K_eff Equipartition)

**Theorem.** Let V: R -> R be admissible (C^2, confining, bounded below). Then:

    <K_eff> = <xi * V'(xi)>_V = kT

where the expectation is taken with respect to rho(xi) ~ exp(-V(xi)/kT).

**Proof.** By integration by parts:

    <xi * V'(xi)> = (1/Z) integral_{-inf}^{inf} xi * V'(xi) * exp(-V(xi)/kT) dxi

Let u = xi, dv = V'(xi) * exp(-V(xi)/kT) dxi.

Note that V'(xi) * exp(-V(xi)/kT) = -kT * d/dxi[exp(-V(xi)/kT)], so:

    dv = -kT * d[exp(-V(xi)/kT)]

Therefore:

    <xi * V'(xi)> = (1/Z) * { [-kT * xi * exp(-V(xi)/kT)]_{-inf}^{inf}
                              + kT * integral_{-inf}^{inf} exp(-V(xi)/kT) dxi }

The boundary term vanishes because V is confining (V -> inf implies
xi * exp(-V/kT) -> 0 as |xi| -> inf). The remaining integral is Z:

    <xi * V'(xi)> = (1/Z) * kT * Z = kT.     QED.

### 1.4 Verification for Specific Potentials

| Thermostat | V(xi) | K_eff(xi) = xi*V'(xi) | Standard NHC K_eff |
|------------|--------|------------------------|---------------------|
| Nose-Hoover | Q*xi^2/2 | Q*xi^2 | Q*xi^2 (matches!) |
| Log-Osc | Q*log(1+xi^2) | 2Q*xi^2/(1+xi^2) | -- |
| Tanh | Q*log(cosh(xi)) | Q*xi*tanh(xi) | -- |
| Arctan | Q*(xi*atan(xi) - log(1+xi^2)/2) | Q*xi*atan(xi) | -- |

For NH, the effective kinetic energy is exactly the standard NHC coupling,
confirming that our formula is the correct generalization.

---

## 2. Generalized Chain Dynamics

### 2.1 Equations of Motion

For a chain of M thermostat variables xi_1, ..., xi_M with thermostat
potentials V_1, ..., V_M and thermostat masses Q_1, ..., Q_M:

    dq/dt   = p / m                                                    (GC1)
    dp/dt   = -grad_U(q) - g_1(xi_1) * p                              (GC2)
    dxi_1/dt = (1/Q_1)(K_phys - d*kT) - g_2(xi_2)*xi_1               (GC3)
    dxi_j/dt = (1/Q_j)(K_eff(xi_{j-1}) - kT) - g_{j+1}(xi_{j+1})*xi_j  (GC4, j=2..M-1)
    dxi_M/dt = (1/Q_M)(K_eff(xi_{M-1}) - kT)                          (GC5)

where:
- g_j(xi_j) = V_j'(xi_j) / Q_j  is the friction function for level j
- K_phys = |p|^2 / m  is the physical kinetic energy
- K_eff(xi_{j-1}) = xi_{j-1} * V_{j-1}'(xi_{j-1})  is the effective kinetic energy

### 2.2 Extended Hamiltonian

The invariant measure is:

    rho ~ exp(-H_ext / kT)

where:

    H_ext = U(q) + |p|^2/(2m) + sum_{j=1}^{M} V_j(xi_j)

This is the natural extension: each thermostat level contributes its own
potential V_j to the extended Hamiltonian.

### 2.3 Theorem (Generalized Chain Invariance)

**Theorem 2.** The generalized chain dynamics (GC1)-(GC5) preserve the measure
rho ~ exp(-H_ext/kT) for any collection of admissible potentials {V_j}.

**Proof.** We verify div(rho * F) = 0, i.e., div(F) + F . grad(log rho) = 0.

**Divergence computation:**

    div_q(dq/dt) = 0
    div_p(dp/dt) = -d * g_1(xi_1)
    div_{xi_1}(dxi_1/dt) = -g_2(xi_2)   [from the -g_2(xi_2)*xi_1 term]
    div_{xi_j}(dxi_j/dt) = -g_{j+1}(xi_{j+1})   [j = 2, ..., M-1]
    div_{xi_M}(dxi_M/dt) = 0

So:

    div(F) = -d*g_1(xi_1) - sum_{j=1}^{M-1} g_{j+1}(xi_{j+1})
           = -d*g_1 - g_2 - g_3 - ... - g_M

**Gradient of log rho:**

    grad_q(log rho) = -grad_U / kT
    grad_p(log rho) = -p / (m*kT)
    d/dxi_j(log rho) = -V_j'(xi_j) / kT

**Dot product F . grad(log rho):**

q-sector + p-sector (same cancellation as single thermostat):
    = g_1 * K_phys / kT                                              (i)

xi_1 sector:
    [(K_phys - d*kT)/Q_1 - g_2*xi_1] * (-V_1'/kT)
    = -V_1'*(K_phys - d*kT)/(Q_1*kT) + g_2*xi_1*V_1'/kT            (ii)

xi_j sector (j = 2, ..., M-1):
    [(K_eff(xi_{j-1}) - kT)/Q_j - g_{j+1}*xi_j] * (-V_j'/kT)
    = -V_j'*(K_eff(xi_{j-1}) - kT)/(Q_j*kT) + g_{j+1}*xi_j*V_j'/kT  (iii)

xi_M sector:
    [(K_eff(xi_{M-1}) - kT)/Q_M] * (-V_M'/kT)
    = -V_M'*(K_eff(xi_{M-1}) - kT)/(Q_M*kT)                          (iv)

**Substituting g_j = V_j'/Q_j and K_eff(xi_{j-1}) = xi_{j-1}*V_{j-1}'(xi_{j-1}):**

From (i): V_1'*K_phys/(Q_1*kT)

From (ii): -V_1'*K_phys/(Q_1*kT) + d*V_1'/Q_1 + g_2*xi_1*V_1'/kT

The K_phys terms from (i) and (ii) cancel, leaving from levels 1:
    d*V_1'/Q_1 + g_2*xi_1*V_1'/kT
    = d*g_1 + V_2'*xi_1*V_1'/(Q_2*kT)

Now from (iii) for level j:
    -V_j'*xi_{j-1}*V_{j-1}'/(Q_j*kT) + V_j'/Q_j + g_{j+1}*xi_j*V_j'/kT

The cross-term -V_j'*xi_{j-1}*V_{j-1}'/(Q_j*kT) cancels with the
+g_j*xi_{j-1}*V_{j-1}'/kT = V_j'*xi_{j-1}*V_{j-1}'/(Q_j*kT) from the
previous level.

After telescoping cancellation across all levels:

    div(F) + F.grad(log rho) = -d*g_1 - g_2 - ... - g_M
                                + d*g_1 + g_2 + ... + g_M
                              = 0.     QED.

### 2.4 Normalization and Stability Constraints

For the generalized chain to be well-defined:

1. **Normalizability:** Each V_j must be confining (V_j -> inf as |xi_j| -> inf)
   so that integral exp(-V_j/kT) dxi_j < inf.

2. **Thermostat mass selection:** The thermostat masses Q_j determine the
   response timescale of each chain level. Following Martyna et al. (1996):
   - Q_1 ~ d * kT * tau^2 where tau is the target oscillation period
   - Q_j ~ kT * tau^2 for j >= 2

3. **Bounded friction advantage:** When g_j is bounded (e.g., tanh, arctan),
   the momentum rescaling exp(-g_1*dt/2) in the integrator is bounded away
   from 0 and infinity, improving numerical stability. This is a key advantage
   of non-quadratic thermostat potentials.

4. **K_eff boundedness:** For bounded-friction thermostats, K_eff(xi) is also
   bounded (e.g., for tanh: K_eff <= Q*|xi|*1 which grows linearly, slower
   than the quadratic growth of NHC). This means the chain coupling is
   "softer" and less prone to instabilities.

---

## 3. Physical Interpretation

The generalized chain has a clear physical interpretation:

- **Level 1** (xi_1): Controls the physical temperature. Absorbs/releases
  kinetic energy to maintain K_phys ~ d*kT.

- **Level j >= 2** (xi_j): Controls the "temperature" of xi_{j-1}. The
  effective kinetic energy K_eff(xi_{j-1}) plays the role of kinetic energy
  for the (j-1)-th thermostat variable.

- **Chain termination** (xi_M): The last thermostat has no controller above it,
  but for M >= 3, the chain provides sufficient ergodicity.

The K_eff formula connects thermostats via their natural energy scale: the
product xi * V'(xi) measures how much "kinetic-like" energy the thermostat
variable carries, in a way that respects the non-quadratic geometry of V.

---

## 4. Connection to Prior Work

- **Standard NHC (Martyna et al. 1992):** Recovered exactly when all V_j are
  quadratic: V_j = Q_j*xi_j^2/2, giving K_eff = Q_j*xi_j^2 and g_j = xi_j.

- **Master Theorem (Watanabe 2007, unified-theory-007):** Our Theorem 2
  extends the single-thermostat result of Theorem 1 to chains of arbitrary
  length. The K_eff formula is the novel contribution.

- **Log-Osc thermostat (log-osc-001):** The log-osc chain uses V = Q*log(1+xi^2),
  giving K_eff = 2Q*xi^2/(1+xi^2), which is bounded. This explains the
  improved stability observed in that orbit.

## References

- [Martyna et al. (1992)](https://doi.org/10.1063/1.463940) -- Nose-Hoover chains: the canonical ensemble via continuous dynamics
- [Martyna et al. (1996)](https://doi.org/10.1080/00268979600100761) -- Explicit reversible integrators for extended systems dynamics
- [Watanabe & Kobayashi (2007)](https://doi.org/10.1103/PhysRevE.75.040102) -- Generalized Nose-Hoover thermostat
- Builds on unified-theory-007 (parent orbit) which established the Master Theorem
- Builds on log-osc-001 which introduced bounded-friction thermostats
