---
strategy: deep-geometry-012
status: complete
eval_version: eval-v1
metric: null
issue: 14
parent: unified-theory-007
---

# Deep Geometry and Intuition of Thermostat Dynamics

## Objective

Create publication-quality visualizations revealing the GEOMETRIC meaning of
thermostat dynamics -- why bounded friction (Log-Osc) works better than
unbounded friction (NH/NHC) for canonical sampling.

## Approach

Five complementary visualizations, each targeting a different aspect:

1. **3D Phase Space Flow** (`make_phase_space_3d.py`) -- Trajectory structure in (q,p,xi) space
2. **Friction Function Geometry** (`make_friction_geometry.py`) -- Master Theorem catalog of V(xi), g(xi), rho(xi)
3. **Torus Comparison** (`make_torus_comparison.py`) -- Phase coverage vs thermostat mass Q
4. **Poincare Sections** (`make_poincare_sections.py`) -- Direct evidence of KAM tori vs chaotic sea
5. **Mechanism Schematic** (`make_mechanism_schematic.py`) -- Feedback loop diagrams + time series

## Results

### Figure 1: 3D Phase Space Flow
- NHC(M=3) trajectory in (q,p,xi) space forms a flattened torus with xi ranging [-4.3, 4.4]
- Log-Osc trajectory is more compact in xi [-1.8, 1.8] due to bounded friction
- (q,p) projections show NHC samples a broad ring while Log-Osc fills space more uniformly

### Figure 2: Friction Function Geometry
- Clear comparison of 4 thermostat potentials from the Master Theorem: quadratic (NH), logarithmic (Log-Osc), cosh, double-well
- Key insight: NH friction g(xi) = xi is unbounded -- large xi causes excessive damping/driving
- Log-Osc and cosh friction are bounded |g| <= 1, preventing resonant feedback
- Marginal distributions: NH gives Gaussian tails, Log-Osc gives heavy Cauchy-like tails
- Heavy tails in xi mean the thermostat variable can explore large values without proportional friction effect

### Figure 3: Torus Comparison (Q = 0.1, 0.5, 1.0)
- At all Q values, NHC shows regular ring-like structure in (q,p)
- Log-Osc at small Q (0.1) is initially confined but shows time-varied exploration
- Time coloring (viridis) reveals NHC revisits the same regions (same colors overlap)

### Figure 4: Poincare Sections (most diagnostic)
- **NH**: Single closed curve = KAM torus. Trajectory forever trapped on one invariant curve.
- **NHC(M=3)**: Scattered points filling area. Chain breaks tori partially.
- **Log-Osc**: Dense filled region with sharp boundary = chaotic sea. Ergodic.
- This is the most direct geometric evidence of ergodicity breaking vs restoration.

### Figure 5: Mechanism Schematic
- NH: resonant feedback loop -- hot -> xi grows -> friction grows unboundedly -> cold -> repeat (KAM)
- Log-Osc: broken feedback loop -- hot -> xi grows -> friction SATURATES at 1 -> partial cooling -> xi escapes -> torus destruction
- Time series confirm: NH friction oscillates with growing amplitude; Log-Osc stays bounded
- Kinetic energy: NH shows periodic patterns (non-ergodic); Log-Osc is irregular (chaotic)

## Key Insights

1. **Bounded friction is the mechanism**: The Master Theorem says g(xi) = V'(xi)/Q.
   For NH, g = xi (unbounded), creating resonant energy exchange that preserves KAM tori.
   For Log-Osc, g = 2xi/(1+xi^2) (bounded by 1), which breaks the resonance.

2. **Heavy-tailed xi distribution**: Log-Osc's Cauchy-like tails mean xi frequently visits
   large values where friction ~ 2/xi -> 0. This creates intermittent friction -- alternating
   between strong damping (xi ~ 1) and free dynamics (xi >> 1), which is the chaos generator.

3. **Poincare sections are definitive**: The NH section shows a single curve (integrable),
   NHC shows partial filling (weakly chaotic), Log-Osc shows dense filling (strongly chaotic).

## Parameters

- All simulations: 1D Harmonic Oscillator, omega=1, kT=1, mass=1
- Phase space / torus: dt=0.01, 500k steps, seed=42
- Poincare: dt=0.008, 2M steps, seed=42 (~2500 crossings per method)
- Mechanism: dt=0.01, 50k steps, seed=42

## References

- [Nose (1984)](https://doi.org/10.1063/1.447334) -- Original Nose thermostat
- [Hoover (1985)](https://doi.org/10.1103/PhysRevA.31.1695) -- Nose-Hoover reformulation
- [Martyna et al. (1992)](https://doi.org/10.1063/1.463940) -- Nose-Hoover Chains
- [KAM theorem](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold%E2%80%93Moser_theorem) -- Why NH is non-ergodic for HO
- [Poincare map](https://en.wikipedia.org/wiki/Poincar%C3%A9_map) -- Section method for detecting chaos
- Builds on #7 (unified-theory-007) which established the Master Theorem framework
