# Dual-Bath Thermostat with Hamiltonian Cross-Coupling: Derivation

## Equations of Motion

We propose a thermostat with two auxiliary variables (xi, eta) that both couple to the physical momentum through additive friction, and are cross-coupled through a measure-preserving Hamiltonian rotation:

```
dq/dt  = p / m
dp/dt  = -dU/dq - (xi + eta) * p
dxi/dt = (1/Q_xi) * (sum_i p_i^2/m - d*kT) + alpha * sqrt(Q_eta/Q_xi) * eta
deta/dt = (1/Q_eta) * (sum_i p_i^2/m - d*kT) - alpha * sqrt(Q_xi/Q_eta) * xi
```

where:
- q in R^d: positions
- p in R^d: momenta  
- xi, eta in R: thermostat variables
- Q_xi, Q_eta > 0: thermostat masses (coupling strengths)
- alpha >= 0: rotation coupling parameter
- kT: target temperature
- m: particle mass
- d: spatial dimension

## Invariant Measure (Tier 1 Proof)

**Claim:** The canonical distribution

```
rho(q, p, xi, eta) = Z^{-1} exp(-S(q, p, xi, eta))
```

where

```
S = U(q)/kT + |p|^2/(2*m*kT) + Q_xi*xi^2/(2*kT) + Q_eta*eta^2/(2*kT)
```

is an invariant measure of the dynamics.

**Proof:** By the Liouville equation, rho is invariant if and only if

```
dS/dt = div(v)
```

where v = (dq/dt, dp/dt, dxi/dt, deta/dt) is the velocity field.

### Computing dS/dt

```
dS/dt = (dS/dq)(dq/dt) + (dS/dp)(dp/dt) + (dS/dxi)(dxi/dt) + (dS/deta)(deta/dt)
```

The partial derivatives of S are:

```
dS/dq = (1/kT) * dU/dq
dS/dp = p / (m*kT)
dS/dxi = Q_xi * xi / kT
dS/deta = Q_eta * eta / kT
```

Computing each term:

**Term 1:** `(dS/dq)(dq/dt) = (dU/dq / kT) * (p/m)`

**Term 2:** `(dS/dp)(dp/dt) = (p/(m*kT)) * (-dU/dq - (xi+eta)*p)`
`= -p*dU/dq/(m*kT) - (xi+eta)*|p|^2/(m*kT)`

Note: Terms 1 and 2 combine: the `p*dU/dq/(m*kT)` terms cancel, leaving
`-(xi+eta) * |p|^2 / (m*kT)`

**Term 3:** `(dS/dxi)(dxi/dt) = (Q_xi*xi/kT) * [(|p|^2/m - d*kT)/Q_xi + alpha*sqrt(Q_eta/Q_xi)*eta]`
`= xi*(|p|^2/m - d*kT)/kT + alpha*sqrt(Q_xi*Q_eta)*xi*eta/kT`

**Term 4:** `(dS/deta)(deta/dt) = (Q_eta*eta/kT) * [(|p|^2/m - d*kT)/Q_eta - alpha*sqrt(Q_xi/Q_eta)*xi]`
`= eta*(|p|^2/m - d*kT)/kT - alpha*sqrt(Q_xi*Q_eta)*xi*eta/kT`

**Terms 3+4:** The rotation terms (`alpha*sqrt(Q_xi*Q_eta)*xi*eta/kT`) cancel exactly! Remaining:
`(xi+eta) * (|p|^2/m - d*kT) / kT`
`= (xi+eta)*|p|^2/(m*kT) - (xi+eta)*d`

**Total dS/dt:**
```
dS/dt = -(xi+eta)*|p|^2/(m*kT) + (xi+eta)*|p|^2/(m*kT) - (xi+eta)*d
     = -(xi+eta)*d
```

### Computing div(v)

```
div(v) = d(dq/dt)/dq + d(dp/dt)/dp + d(dxi/dt)/dxi + d(deta/dt)/deta
```

- `d(p/m)/dq = 0` (p is independent of q in the phase space coordinates)
- `d(-dU/dq - (xi+eta)*p)/dp = -(xi+eta) * I_d` -> trace = `-d*(xi+eta)`
- `d(dxi/dt)/dxi = 0` (dxi/dt has no xi dependence from the kinetic term, and the rotation term alpha*sqrt(Q_eta/Q_xi)*eta has no xi)
- `d(deta/dt)/deta = 0` (similar reasoning)

```
div(v) = 0 - d*(xi+eta) + 0 + 0 = -d*(xi+eta)
```

### Verification

```
dS/dt - div(v) = -(xi+eta)*d - (-(xi+eta)*d) = 0  ✓
```

The Liouville equation is satisfied identically for all q, p, xi, eta, and for arbitrary potential U(q). The canonical distribution is therefore an invariant measure.

This was also verified symbolically using SymPy (see `verify_invariant.py`).

## Design Rationale

### Why Two Parallel Baths?

The standard Nose-Hoover thermostat with a single xi variable is known to be non-ergodic for the 1D harmonic oscillator due to KAM tori in the (q, p, xi) phase space. The NHC approach chains thermostats serially: xi_2 thermostatizes xi_1, etc.

Our approach is different: two thermostats operate in **parallel** on the same physical momentum. This creates a higher-dimensional thermostat space (R^2 vs R^1) with fundamentally different dynamics.

### The Hamiltonian Rotation

The cross-coupling terms `alpha*sqrt(Q_eta/Q_xi)*eta` and `-alpha*sqrt(Q_xi/Q_eta)*xi` generate a Hamiltonian rotation in the (xi, eta) plane with respect to the thermostat energy `H_th = Q_xi*xi^2/2 + Q_eta*eta^2/2`.

This rotation:
1. **Preserves the measure** (divergence-free, cancels in dS/dt)
2. **Breaks integrability**: Without the rotation (alpha=0), the two thermostats are independent and the system has extra conserved quantities. The rotation couples them, reducing the number of integrals of motion.
3. **Creates circular flow**: In the (xi, eta) subspace, the thermostat variables trace out elliptical orbits (when the kinetic driving term is constant). This prevents them from settling on fixed points.

### Connection to Existing Methods

- **alpha=0, Q_eta -> infinity**: Reduces to standard Nose-Hoover
- **alpha=0**: Two independent NH thermostats (Candidate 1, still novel)
- **NHC(M=2)**: Serial chain where xi_2 thermostatizes xi_1. Different topology: serial vs parallel.
- **Patra-Bhattacharya**: Uses configurational temperature for one thermostat. Ours uses kinetic temperature for both but with rotation coupling.
- **Fukuda-Nakamura (2002)**: Coupled Nose-Hoover equations with different coupling structure.

### Parameter Selection

The effective friction coefficient is `xi + eta`. For the combined variable `zeta = xi + eta`:
- `d(zeta)/dt = (1/Q_xi + 1/Q_eta)(|p|^2/m - d*kT) + rotation terms`
- Effective thermostat mass: `Q_eff = Q_xi*Q_eta / (Q_xi + Q_eta)` (harmonic mean)

To match the coupling strength of a single NH with mass Q, set:
`Q_xi * Q_eta / (Q_xi + Q_eta) = Q`

For Q_xi = Q_eta = Q_0: `Q_0/2 = Q`, so `Q_0 = 2Q`.

The alpha parameter controls the rotation speed. Larger alpha gives faster thermostat mixing but can cause integration instability. Optimal alpha depends on the system timescales.

## References

- [Nose (1984)](https://doi.org/10.1063/1.447334) — Extended Hamiltonian for canonical ensemble
- [Hoover (1985)](https://doi.org/10.1103/PhysRevA.31.1695) — Canonical dynamics: equilibrium phase-space distributions  
- [Martyna et al. (1992)](https://doi.org/10.1063/1.463940) — Nose-Hoover chains
- [Patra & Bhattacharya (2015)](https://doi.org/10.1103/PhysRevE.93.023308) — Deterministic dynamics with configurational temperature
- [Fukuda & Nakamura (2002)](https://doi.org/10.1103/PhysRevE.65.026105) — Coupled Nose-Hoover equations
- [Rugh (1997)](https://doi.org/10.1103/PhysRevLett.78.772) — Dynamical approach to temperature
