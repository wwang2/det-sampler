---
strategy: sublinear-unbounded-g-arnold
type: experiment
status: in-progress
eval_version: eval-v1
metric: null
issue: 69
parents:
  - orbit/gprime-ablation-052
---

# sublinear-g-069

Test g(ξ) = ξ·log(1+ξ²)/√(1+ξ²): Arnold's prediction that unbounded-but-sublinear g
avoids both tanh's frequency ceiling (bounded range) and log-osc's sign reversal.

## Hypothesis
g_new outperforms tanh on stiff targets (κ=1000) where ω > ω_max=1.0, while
matching or exceeding tanh on moderate targets. No frequency ceiling because g is unbounded.

## Setup
- Benchmark: orbit 052 grid (d=10, κ in {10,100,1000}, same Q grid)
- Methods: g_new, tanh, log-osc, linear (g=ξ)
- Also: 1D double-well, 2D 4-mode GMM at Q=1.0
- Metric: ratio τ_tanh/τ_new at κ=1000 (higher = better than tanh)

(Agent fills results)
