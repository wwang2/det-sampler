---
strategy: pd-thermostat-derivative-augmented
type: experiment
status: in-progress
eval_version: eval-v1
metric: null
issue: 70
parents: []
---

# pd-thermostat-070

Add derivative term to NH thermostat: dξ/dt = (|p|²-DkT)/Q + K_d·2p·(-∇U - g(ξ)p)
This feeds potential landscape information directly into the friction (Wiener: PD controller).

## Phase 0 (analytical)
Derive divergence condition: div(f) = -Dg(ξ) - 2K_d·g'(ξ)|p|²
Check whether ∇·(f·μ) = 0 holds for canonical μ = exp(-βH)·exp(-ξ²/2QkT).
If not, find modified invariant measure or condition on K_d.

## Phase 1 (numerical, if measure preserved)
- Targets: 1D double-well, 2D 4-mode GMM
- Compare: NH-tanh vs PD-tanh (K_d ∈ {0.01,0.1,0.5}) vs NHC(M=3)
- Metric: τ_int, mode coverage, barrier crossing rate

(Agent fills results)
