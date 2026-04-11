---
strategy: tasaki-temperature-quench
type: experiment
status: in-progress
eval_version: eval-v1
metric: null
issue: 65
parents:
  - orbit/triple-identity-064
  - orbit/nh-cnf-thorough-062
---

# Tasaki Temperature Quench -- NH-tanh verifies non-equilibrium KL identity

We verify that the Tasaki/Evans-Searles non-equilibrium identity holds for the NH-tanh thermostat under a sudden temperature quench: the ensemble-averaged total entropy production equals the KL divergence between the pre- and post-quench invariant measures.

## Glossary

- **NH**: Nose-Hoover thermostat (plain, M=1 chain length)
- **NH-tanh**: Nose-Hoover with `g(xi) = tanh(xi)` replacing linear `g(xi) = xi`
- **NHC**: Nose-Hoover Chain (M >= 2); not used in this orbit
- **KL**: Kullback-Leibler divergence D_KL(P || Q) = E_P[log(P/Q)]
- **RK4**: Classical 4th-order Runge-Kutta integrator
- **DW**: Double-well potential V(q) = (q1^2-1)^2 + 0.5 q2^2

## Phase 0 -- Theory

### Definitions copied from orbit 064

From `orbits/triple-identity-064/experiment.py`:

- **sigma_exact** (line 232): `d_exact = D * 0.5 * (np.tanh(xi_old) + np.tanh(xi_new)) * dt`
  Cumulative: `sigma_exact(t) = +d * integral_0^t tanh(xi(s)) ds` (trapezoid rule)

- **sigma_bath** (line 236): `d_bath = BETA * 0.5 * (q_dot_old + q_dot_new) * dt`
  where `q_dot = heat_rate(z) = tanh(xi) * |p|^2` (line 163).
  Cumulative: `sigma_bath(t) = beta * integral_0^t tanh(xi(s)) |p(s)|^2 ds`

- **Sign convention** (line 43): `sigma(t) = log rho_t - log rho_0 = -integral tr(J) dt`
  with `tr(J)_{NH-tanh} = -d * tanh(xi)`, hence `sigma_exact = +d * integral tanh(xi) dt`.

### The Tasaki identity

For a deterministic flow with phase-space contraction rate `div(f) = -d tanh(xi)`, starting from the invariant measure `pi_{T0}` on the full extended space (q, p, xi), and evolving under the T1-thermostatted flow:

    E_{pi_{T0}}[ sigma_bath(t) - sigma_exact(t) ]  -->  D_KL( pi_{T0} || pi_{T1} )   as t -> infinity

where `pi_T` is the NH-tanh invariant measure at temperature T. This follows from:

1. **Liouville**: `log rho_t(z_t) - log rho_0(z_0) = -integral_0^t div(f)(z_s) ds = sigma_exact(t)`
2. **Total entropy production**: `Sigma_tot = sigma_bath - sigma_exact = beta_1 integral tanh(xi)|p|^2 dt - d integral tanh(xi) dt`
3. **Second law**: `<Sigma_tot>_{t=infinity} = D_KL(pi_{T0} || pi_{T1}) - D_KL(rho_t || pi_{T1})`

As the system relaxes (`rho_t -> pi_{T1}`), the residual KL vanishes and `<Sigma_tot> -> D_KL(pi_{T0} || pi_{T1})`.

A strictly equivalent check is the **Jarzynski identity**: `<exp(-Sigma_tot)> = 1` at all times t >= 0. This holds exactly without needing the system to fully relax.

### Closed-form KL for the (q,p) canonical marginal

For the 2D double-well V(q) = (q1^2-1)^2 + 0.5 q2^2, the canonical (q,p) measure factorizes as:

    rho_T(q,p) = Z_T^{-1} exp(-[|p|^2/2 + V(q)] / T)

Since q1, q2, p1, p2 are all independent:
- KL_p = 2 * [0.5 * (log(T1/T0) - 1 + T0/T1)] = log(T1/T0) - 1 + T0/T1 (2D Gaussian momentum)
- KL_q2 = 0.5 * [log(T1/T0) - 1 + T0/T1] (harmonic mode, variance T)
- KL_q1 = integral rho_0(q1) log(rho_0/rho_1) dq1 (numerical 1D quadrature for the quartic potential)

The numerical KL_q1 for the double-well is SMALLER than the harmonic KL because the bimodal distribution "spreads" differently with temperature than a Gaussian.

### The xi contribution

**The prompt's formula `d * [log(T1/T0) - 1 + T0/T1]` omits the xi auxiliary variable.** The NH-tanh invariant measure lives on extended phase space (q, p, xi). The total D_KL includes a contribution from the xi marginal:

    D_KL(pi_{T0} || pi_{T1}) = D_KL_qp + D_KL_xi + cross-terms

For NH-tanh, the xi marginal is NOT the canonical Gaussian `N(0, T/Q)` (unlike plain Hoover-NH which has a conserved extended Hamiltonian). The tanh thermostat has no conserved extended energy, and empirically `<xi^2>` at T=1 on the 2D double-well is ~2.58 (not 1.0). The xi distribution scales approximately as `T` but with a non-canonical prefactor.

The total D_KL is estimated empirically via k-nearest-neighbor methods on samples from the full (q, p, xi) invariant measure.

### NHC housekeeping heat

For NHC with M >= 2, the Jacobian trace becomes `-d tanh(xi_1) - sum_{j=2}^M tanh(xi_j)`, and the bath entropy production must include heat flows between successive chain stages. This requires tracking `xi_j * xi_{j+1} * Q_j` coupling terms. The derivation is messy and not attempted here. We use plain NH (M=1) throughout.

### Non-ergodicity of NH on harmonic oscillators

Plain NH (and NH-tanh) on UNCOUPLED harmonic oscillators is famously non-ergodic (Martyna, Klein, Tuckerman 1992). The phase space is foliated into invariant tori, and finite ensembles cannot sample the invariant measure. We verified this empirically:

- 1D harmonic, T0=1->T1=2: `<Sigma>` oscillates between 0.22 and 0.29 depending on t_post, never converging. Jarzynski `<exp(-Sigma)> = 0.92 != 1`.
- 10D uncoupled harmonic: `<Sigma> ~ 0.30` at t_post=800, versus analytic KL_qp = 0.72. Mode-mode relaxation through a single xi is extremely slow.

**Resolution**: We verify the identity on the 2D double-well potential, where NH-tanh IS ergodic (orbit 064 demonstrated this with 200-trajectory ensembles showing proper canonical sampling of both wells).

## Phase 1 -- 2D double-well, T0=1.0 -> T1=2.0

(Results filled after experiment completes)

## Phase 2 -- 2D double-well, T0=0.5 -> T1=1.5, with Hutchinson comparison

(Results filled after experiment completes)

## Phase 3 -- 1D harmonic diagnostic (non-ergodicity)

(Results filled after experiment completes)

## References

- Martyna, Klein, Tuckerman (1992) "Nose-Hoover chains" J. Chem. Phys. 97, 2635 -- NHC for ergodicity
- Evans, Searles (2002) "The fluctuation theorem" Adv. Phys. 51, 1529 -- entropy production identity
- Jarzynski (1997) "Nonequilibrium equality for free energy differences" Phys. Rev. Lett. 78, 2690
- Crooks (1999) "Entropy production fluctuation theorem" Phys. Rev. E 60, 2721
- Orbit 064: triple-identity-064 -- sigma definitions, sign conventions, variance characterization
