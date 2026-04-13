---
strategy: adaptive-annealing-nh-sigma-bath
type: experiment
status: in-progress
eval_version: eval-v1
metric: null
issue: 68
parents:
  - orbit/tasaki-quench-065
---

# adaptive-annealing-068

Use σ_bath's instantaneous rate dσ_bath/dt = β·tanh(ξ)·|p|² as a real-time
equilibration detector to drive an adaptive simulated annealing schedule.

## Hypothesis

An NH-driven adaptive schedule (lower T when EMA of |dσ_bath/dt| < threshold ε)
reaches the target distribution faster (fewer force evaluations) than a fixed
geometric cooling schedule, on multimodal targets where the optimal cooling rate
is not known a priori.

## Setup

- Targets: (1) 2D mixture of 4 Gaussians (varying mode separation d ∈ {2,4,6})
           (2) 1D double-well V(x) = (x²-1)²
- NH integrator: BAOAB, dt=0.01, g(ξ)=tanh(ξ), Q=1.0
- Temperature range: T_start=5.0 → T_final=1.0
- Adaptive rule: EMA(|dσ_bath/dt|, τ=50 steps) < ε → multiply T by α=0.9
- Baselines:
    (1) Geometric fixed schedule: T(t) = T_start · α^(t/N_per_stage)
    (2) Plain overdamped Langevin anneal (no ξ signal)
- Metric: KL divergence to target at final T=1.0 vs cumulative force evaluations
- N_traj=50 independent runs per method per target

(Agent will fill in results)
