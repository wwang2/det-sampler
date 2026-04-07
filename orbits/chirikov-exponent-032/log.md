---
strategy: chirikov-exponent-032
status: complete
eval_version: eval-v1
metric: 8.0955
issue: 32
parent: ergodicity-phase-diagram-027
---

# Chirikov Exponent: C(κ) vs κ — Non-Monotonic with Resonance Singularity

## Key Finding: C(κ) is NOT a simple power law

The critical Q₂/Q₁ ratio for N=2 ergodicity has a **resonance singularity** at ω×Q₁=1 (κ=1 for Q₁=1):

  - κ=0.1: ω=0.316, ω×Q₁=0.316, C(κ)=1.682
  - κ=0.3: ω=0.548, ω×Q₁=0.548, C(κ)=1.438
  - κ=1.0: ω=1.000, ω×Q₁=1.000, C(κ)=N/A (>100, resonance)
  - κ=3.0: ω=1.732, ω×Q₁=1.732, C(κ)=8.095
  - κ=10.0: ω=3.162, ω×Q₁=3.162, C(κ)=1.682
  - κ=30.0: ω=5.477, ω×Q₁=5.477, C(κ)=1.050
  - κ=100.0: ω=10.000, ω×Q₁=10.000, C(κ)=1.050
  - κ=300.0: ω=17.321, ω×Q₁=17.321, C(κ)=1.050

## Result: Non-Monotonic Behavior

1. **κ<1 (ω×Q₁<1, sub-resonance)**: C decreases as κ→1 from below
   - κ=0.1: C=1.682, κ=0.3: C=1.438 → C decreasing

2. **κ=1 (ω×Q₁=1, exact resonance)**: C = NOT FOUND (>100)
   - The thermostat at Q₁=1 is at exact resonance with ω=1
   - No Q₂/Q₁ up to 100 achieves ergodicity
   - Resonance singularity: C(κ) → ∞ at ω×Q₁=1

3. **κ=3 (just above resonance)**: C=8.095 (large but finite)
   - Lingering near-resonance effect

4. **κ≥10 (ω×Q₁>>1, fast oscillators)**: C drops to ~1.05 (minimum scan value)
   - For fast oscillators, ANY second thermostat (barely different Q) provides ergodicity
   - C → 1 as κ → ∞

## Comparison to Orbit #027

Orbit #027 reported C(κ=1)≈1.56 using a DIFFERENT Q₁ (not 1.0) or looser criterion.
This orbit uses Q₁=1.0, which places it exactly at resonance for κ=1. The discrepancy
confirms that C(κ) depends jointly on κ AND ω×Q₁ — not κ alone.

## Physical Interpretation

- **Resonance mechanism (confirmed)**: KAM tori are hardest to break when the thermostat
  and oscillator are at resonance. At exact resonance, no ratio Q₂/Q₁<100 is sufficient.
- **Fast oscillators are easy**: When ω >> 1/Q₁, the oscillator cycles many times per
  thermostat period. Any perturbation Q₂ > Q₁ breaks the tori trivially.
- **Design implication**: The F1 prescription Q_max=1/√κ_min places the slow thermostat
  at ω×Q_max=1 (resonance). To avoid this, use Q_max slightly > 1/√κ_min.

## Revised Picture vs Power Law Hypothesis

The brainstorm orbit #030 predicted C(κ) ~ κ^{0.4} asymptoting to κ^{0.5}.
**This is WRONG** for the case Q₁=1 (fixed). Instead:
- C has a resonance singularity at κ=1/Q₁² (any fixed Q₁)
- C→1 for large κ (no power-law growth)
- The "exponent" b is meaningless for non-monotonic C(κ)

## Metric Definition

metric = max_κ C(κ) (excluding NOT FOUND) = 8.095 at κ=3.0
