---
strategy: tensor-nh-onsager
type: experiment
status: complete
eval_version: eval-v1
metric: 1.01
issue: 78
parents: []
---

# Onsager Tensor NH -- Symmetric Xi_{ij} Monitoring Full Kinetic Energy Tensor

## Hypothesis

Replace scalar friction xi in Nose-Hoover with a symmetric tensor Xi_{ij} that
monitors the full kinetic energy tensor p_i*p_j / m. This should improve sampling
of anisotropic distributions because each momentum component gets separate feedback.

## Phase 0: Analytical Derivation

### Equations of motion (D-dimensional)

    dq_i/dt = p_i / m                              (unchanged)
    dp_i/dt = -dU/dq_i - sum_j Xi_{ij} * p_j       (tensor friction)
    dXi_{ij}/dt = (p_i*p_j/m - kT*delta_{ij}) / Q   (monitor kinetic energy tensor)

For 2D: Xi is symmetric => 3 independent variables: Xi_xx, Xi_xy, Xi_yy.

### Invariant measure proof

Target: mu ~ exp(-beta*H(q,p)) * exp(-Q/(2kT) * sum_{ij} Xi_{ij}^2)

Divergence of the flow:
- d/dq_i(dq_i/dt) = 0
- d/dp_i(dp_i/dt) = -Xi_{ii}  (diagonal friction coefficient)
- d/dXi_{ij}(dXi_{ij}/dt) = 0

Total div = -sum_i Xi_{ii} = -tr(Xi)

For Liouville condition div(f) + f . grad(log mu) = 0:

- dq . grad_q(log mu) = (p/m) . (-beta*grad_U) = -beta*p.grad_U/m
- dp . grad_p(log mu) = (-grad_U - Xi.p) . (-beta*p/m) = beta(p.grad_U/m + p.Xi.p/m)
- dXi . grad_Xi(log mu) = sum_{ij}[(p_ip_j/m - kT*delta_{ij})/Q] * (-Q*Xi_{ij}/kT)
                         = -sum_{ij} Xi_{ij}(p_ip_j/m - kT*delta_{ij}) / kT

Combining:
- -beta*p.grad_U/m + beta*p.grad_U/m + beta*p.Xi.p/m - sum_{ij}Xi_{ij}*p_ip_j/(m*kT) + tr(Xi)
- = (beta - 1/kT)*p.Xi.p/m + tr(Xi)
- = 0 (since beta=1/kT) + tr(Xi) - tr(Xi) = 0   QED

**The tensor NH preserves the canonical measure** for the symmetric case with uniform Q.

### Friction variants

1. **Linear:** g(Xi) = Xi (standard)
2. **Log-osc:** g(Xi) = 2*Xi/(1+Xi^2) element-wise (bounded, |g|<=1)

## Phase 1: Implementation

- `tensor_nh.py`: TensorNH dynamics class + TensorNHIntegrator (BAOAB splitting)
- Integrator uses analytical 2x2 matrix exponential for the friction step
- Cost: ~3x slower per step than scalar NH (matrix exp overhead)
- AnisotropicGaussian2D potential for testing

## Phase 2: Experimental Results

### Experiment 1: Anisotropic Gaussian 2D (kappa=10, 100)

200k force evals, dt=0.005, 3 seeds, mean values reported.

**kappa=10:**

| Sampler            | Q=0.3 KL | Q=1.0 KL | Q=3.0 KL |
|--------------------|----------|----------|----------|
| NH_scalar          | 0.285    | 0.427    | 0.478    |
| NHC_M3             | 0.153    | 0.153    | 0.429    |
| TensorNH_linear    | 0.153    | 0.186    | 0.311    |
| TensorNH_logosc    | 0.215    | 0.154    | 0.246    |

At kappa=10: TensorNH matches NHC at Q=0.3, and BEATS NHC at Q=3.0 (0.246 vs 0.429).
Tensor NH is more Q-robust than NHC.

**kappa=100:**

| Sampler            | Q=0.3 KL | Q=1.0 KL | Q=3.0 KL |
|--------------------|----------|----------|----------|
| NH_scalar          | 0.693    | 0.807    | 0.908    |
| NHC_M3             | 0.083    | 0.103    | 0.282    |
| TensorNH_linear    | 0.611    | 0.886    | 1.084    |
| TensorNH_logosc    | 1.400    | 0.223    | 0.592    |

At kappa=100: NHC dominates across all Q values. TensorNH_logosc is 2nd best at
Q=1.0 (0.223 vs NHC 0.103), significantly better than NH_scalar (0.807).
TensorNH_linear is poor -- similar to NH_scalar.

### Experiment 2: Deep run kappa=100, Q=1.0, 500k evals

| Sampler            | KL (mean +/- std) | tau   |
|--------------------|--------------------|-------|
| NHC_M3             | 0.056 +/- 0.007   | 40.0  |
| TensorNH_logosc    | 0.231 +/- 0.033   | 39.7  |
| TensorNH_linear    | 0.724 +/- 0.492   | 40.1  |
| NH_scalar          | 0.746 +/- 0.250   | 40.1  |

NHC: 4x better KL than TensorNH_logosc. Tau is identical across all methods (~40).

### Experiment 3: Q_offdiag sweep (kappa=100, Q_diag=1)

| Q_offdiag | TensorNH_linear KL | TensorNH_logosc KL |
|-----------|--------------------|--------------------|
| 0.1       | 0.646              | 0.232              |
| 1.0       | 0.886              | 0.223              |
| 5.0       | 1.130              | 0.281              |

Off-diagonal Q has minimal effect on logosc (0.22-0.28 range). Linear degrades
with higher Q_offdiag. No benefit over uniform Q.

### Experiment 4: Double-well 2D (Q=1, 500k evals)

| Sampler            | KL     | tau   | ESS/fe  |
|--------------------|--------|-------|---------|
| TensorNH_logosc    | 0.051  | 374.0 | 0.00242 |
| TensorNH_linear    | 0.052  | 323.2 | 0.00278 |
| NHC_M3             | 0.053  | 326.7 | 0.00275 |
| NH_scalar          | 0.135  | 246.9 | 0.00417 |

**Tensor NH wins on double-well!** Both tensor variants slightly beat NHC_M3
in KL, while significantly beating NH_scalar. The double-well has inherent
anisotropy (barrier in x, harmonic in y), which tensor NH handles well.

### Experiment 5: 1D Harmonic Oscillator (ergodicity, 1M evals)

| Sampler            | Ergodicity | KL     |
|--------------------|------------|--------|
| NHC_M3             | 0.904      | 0.003  |
| NH_scalar          | 0.573      | 0.060  |
| TensorNH_linear_1D | 0.573      | 0.060  |

As expected: in 1D, tensor NH reduces to scalar NH (same dynamics, same ergodicity).
NHC remains the only ergodic option for 1D HO.

## Key Answers

**Q1: Does monitoring p_x*p_y help for anisotropic distributions?**
Partially. At moderate anisotropy (kappa=10), tensor NH matches NHC and beats it
at suboptimal Q. At high anisotropy (kappa=100), NHC's chain structure is more
effective -- the bottleneck is not the observable but the feedback architecture.

**Q2: Does tensor NH outperform scalar NH log-osc on kappa=100?**
Yes, TensorNH_logosc (KL=0.231) vastly outperforms NH_scalar (KL=0.746) at kappa=100.
But it does not beat NHC_M3 (KL=0.056).

**Q3: Does tensor NH outperform NHC(M=3)?**
- Double-well: YES (0.051 vs 0.053, marginal)
- Anisotropic kappa=10, Q=3: YES (0.246 vs 0.429, significant -- more Q-robust)
- Anisotropic kappa=100: NO (0.231 vs 0.056, NHC wins 4x)
- 1D HO ergodicity: NO (identical to NH scalar in 1D)

**Q4: Does off-diagonal coupling Xi_xy help?**
Minimal effect. The off-diagonal Q_offdiag sweep shows no improvement over uniform Q.

## Metric

metric = tau_NH / tau_TensorNH_logosc = 40.1 / 39.7 = **1.01**

The tensor NH does NOT improve mixing time (autocorrelation). All methods have
the same tau (~40) at kappa=100 because tau is dominated by the physical timescale
of the stiff mode, not the thermostat. The improvement is in KL only.

## Verdict

**Mixed results. Not a clear win over NHC(M=3).**

Positives:
- Proven invariant measure (rigorous)
- Wins on double-well (slight KL improvement over NHC)
- More Q-robust than NHC at moderate anisotropy (kappa=10)
- TensorNH_logosc beats NH_scalar by 3x at kappa=100

Negatives:
- NHC_M3 dominates at high anisotropy (kappa=100), 4x better KL
- No mixing time improvement (tau ratio = 1.01)
- 3x computational overhead per step (matrix exponential)
- 1D: reduces to scalar NH, no ergodicity improvement
- Off-diagonal coupling provides no benefit

**Recommendation:** Tensor NH is a valid deterministic thermostat with a proven
invariant measure, but it does not justify replacing NHC(M=3) as the default.
The chain architecture of NHC is more effective at high anisotropy than
monitoring the full kinetic energy tensor.

A promising direction would be combining tensor friction WITH chains
(Tensor-NHC), which could get both the anisotropy handling and the
self-thermostating of the chain. This is left for a future orbit.
