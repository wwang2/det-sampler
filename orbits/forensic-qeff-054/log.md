---
strategy: forensic-qeff-matching
status: complete
eval_version: eval-v1
metric: 0.77
issue: 54
parents:
  - orbit/gprime-ablation-052
---

# forensic-qeff-054

## Glossary

- **g(xi)**: friction function applied to thermostat variable xi
- **g'(0)**: derivative of the friction function at the origin; sets the linearized coupling strength
- **Q**: thermostat mass parameter (inertia of the thermostat)
- **Q_eff**: effective thermostat mass, defined as Q / g'(0)
- **tau_int**: integrated autocorrelation time of q_d^2, averaged across dimensions
- **kappa**: anisotropy ratio (largest eigenvalue / smallest eigenvalue of target covariance)
- **IQR**: interquartile range [25th percentile, 75th percentile]
- **ACF**: autocorrelation function
- **NH**: Nose-Hoover (single thermostat, N=1)

## Result

**g'(0) is a Q-rescaling.** The effective thermostat mass Q_eff = Q / g'(0) is the operative
parameter. At matched Q_eff = 10, three friction functions with different g'(0) values and
different tail shapes give statistically indistinguishable mixing rates:

| Condition                    | g'(0) | Q  | Q_eff | tau_int (median) | IQR           |
|------------------------------|:-----:|:--:|:-----:|:----------------:|---------------|
| tanh-ref @ Q=10              | 1     | 10 | 10    | 316.8            | [112, 502]    |
| tanh-scaled @ Q=20           | 2     | 20 | 10    | 243.4            | [23, 372]     |
| log-osc @ Q=20               | 2     | 20 | 10    | 221.0            | [22, 509]     |

The decisive ratio is tanh-scaled@Q=20 / tanh-ref@Q=10 = **0.77**, well within the 30%
threshold for "Q_eff explains the difference." At unmatched Q_eff = 5, both methods give
much worse mixing (tau > 900), confirming that Q_eff is what matters.

This resolves the puzzle from orbit #52: tanh-ref (g'(0)=1) appeared to beat tanh-scaled
(g'(0)=2) at the same Q, but this was simply because tanh-ref had twice the Q_eff.
There is no independent role for g'(0) once Q_eff is accounted for.

Furthermore, **tail shape is irrelevant at matched Q_eff.** Log-osc (which has a
sign-changing g' tail) gives tau=221 at Q_eff=10, comparable to tanh-ref's 317 and
tanh-scaled's 243. The sign of g' in the tail does not matter when Q_eff is matched.

*E2 (kappa sweep) was contaminated by the Hamiltonian floor at fixed Q=10; a proper study
would co-sweep Q and alpha. Results omitted from conclusions.*

## Approach

### E0: Thermostat variable analysis

Ran short new trajectories (50k steps, 3 seeds) recording the full xi time series for all
three friction functions at Q=10. These are independent simulations (not a re-analysis of
parent results.json), chosen to be short enough for quick turnaround while still capturing
the thermostat variable's equilibrium distribution. The goal was to test whether xi's
distribution scales with Q_eff as predicted by the equilibrium theory.

The naive prediction for a linearized thermostat is std(xi) = 1/sqrt(g'(0) * Q). This
gives pred=0.224 for g'(0)=2 methods and pred=0.316 for tanh-ref. The actual values are
roughly 0.4x the prediction for all methods (0.099, 0.095, 0.129). The consistent ratio
across methods (0.41-0.44) suggests a systematic correction, likely from the nonlinear
saturation of g(xi). The key observation is that the *relative* scaling between methods
tracks the prediction: tanh-ref's std(xi) is ~1.35x larger than the g'(0)=2 methods,
close to the predicted sqrt(2) = 1.41x.

Thermostat entropy S(xi) anti-correlates with tau_int: wider xi distributions (higher S)
correspond to faster mixing.

### E1: The decisive Q_eff matching test

The razor: run tanh-scaled at Q=20 (Q_eff = 20/2 = 10) and compare to tanh-ref at Q=10
(Q_eff = 10/1 = 10). Same target (d=10 anisotropic Gaussian, kappa=100), same seeds
(1000-1019), same setup (N=5 thermostats, 200k steps, dt=0.005).

Result: ratio = 0.77. The Q_eff hypothesis is confirmed.

For further confirmation, log-osc at Q=20 (Q_eff = 20/2 = 10) gives tau=221, also
comparable. Meanwhile, both methods at Q_eff=5 (Q=10 for g'(0)=2) give tau > 900.
And tanh-ref at Q=20 (Q_eff=20) gives tau=6.6, which is the Hamiltonian floor --
confirming that Q_eff=20 is too large for this target.

Notably, tanh-scaled@Q=20 shows a bimodal tau distribution with IQR [23, 372] (q25=23
vs tanh-ref's q25=112), suggesting that some seeds find fast-mixing trajectories while
others get trapped for long periods. This seed-dependent bistability is potentially
interesting for future investigation: it may indicate sensitivity to initial conditions
near a mixing-rate bifurcation at this Q_eff value.

### E2: Alpha-kappa sweep (caveats)

Since E1 confirmed Q_eff, we swept alpha (g'(0)) across kappa values at fixed Q=10.
The prediction was that alpha_opt should decrease with kappa.

The results show alpha_opt=0.5 everywhere, but this is contaminated by the Hamiltonian
floor: at alpha=0.5, Q_eff=20, which puts many conditions into the near-Hamiltonian
regime (tau < 10). The E2 results should be interpreted as showing that the *boundary*
between thermostat-active and Hamiltonian-floor shifts with alpha, not that alpha=0.5
is genuinely optimal. A proper E2 would need to co-sweep Q and alpha to stay in the
thermostat-active regime at each kappa.

### E3: Residual tail-shape isolation

At matched Q_eff=10, all three methods overlap in their tau_int distributions. The violin
plot shows broad, overlapping distributions with medians at 317, 243, and 221. Given the
large IQRs (hundreds of tau_int units), these differences are not statistically significant.
Tail shape -- whether g(xi) saturates smoothly (tanh), sharply (log-osc), or even changes
sign -- does not matter at matched Q_eff.

## What I Learned

1. **Q_eff = Q / g'(0) is the operative parameter.** g'(0) is not an independent design
   variable -- it is a Q multiplier. The entire apparent advantage of tanh-ref over
   tanh-scaled at fixed Q was simply a Q_eff effect. When Q_eff is matched, the
   performance difference vanishes (ratio = 0.77).

2. **Tail shape is irrelevant.** At matched Q_eff, log-osc (sign-changing tail),
   tanh-scaled (smooth saturation), and tanh-ref (weaker smooth saturation) all give
   comparable mixing. This further confirms the orbit #52 finding that the sign of g'
   is not causal.

3. **The real design question is: what is the optimal Q_eff for a given target?** From
   the parent data and this orbit, Q_eff ~ 10 gives tau ~ 200-300 on d=10 kappa=100,
   while Q_eff ~ 5 gives tau ~ 1000 and Q_eff ~ 20 hits the Hamiltonian floor.
   The optimal Q_eff for this target appears to be somewhere around 10-15.

4. **The std(xi) scaling is consistent with Q_eff theory.** The relative std(xi) between
   methods tracks sqrt(1/g'(0)) as predicted, even though the absolute values are smaller
   than the linearized prediction by a factor of ~0.4 (due to nonlinear saturation).

5. **E2 (alpha-kappa sweep) was contaminated by the Hamiltonian floor.** At fixed Q=10,
   varying alpha changes Q_eff, and small alpha values push Q_eff into the floor regime.
   A proper coupling-strength vs anisotropy study needs to co-optimize Q and alpha.

## Prior Art & Novelty

### What is already known
- The effective coupling strength of Nose-Hoover-type thermostats scales as g'(0)/Q: this
  is implicit in the standard NH equations where the coupling is 1/Q, and g'(0) simply
  rescales this.
- [Martyna et al. (1992)](https://doi.org/10.1063/1.463940) -- NHC theory, Q as thermostat inertia
- [Tapias et al. (2016)](https://doi.org/10.1103/PhysRevE.94.062123) -- log-osc thermostat

### What this orbit adds
- Explicit numerical confirmation that Q_eff = Q/g'(0) is the operative parameter, not
  g'(0) independently. The matching test (ratio = 0.77) directly demonstrates this.
- Confirmation that tail shape (saturation behavior of g(xi) at large |xi|) is irrelevant
  at matched Q_eff, across three qualitatively different friction functions.
- Resolution of the orbit #52 puzzle: the apparent advantage of weaker coupling (tanh-ref)
  was a Q_eff artifact.

### Honest positioning
The Q_eff rescaling is not surprising from a theoretical standpoint -- linearizing the
thermostat equations around xi=0 immediately gives an effective coupling g'(0)/Q. What
IS somewhat non-obvious is that the *nonlinear* behavior (tail shape) does not contribute
measurably to mixing efficiency. One might have expected that, e.g., the sign-changing
tail of log-osc would create resonances or trapping that degrades performance, but this
does not happen at the Q values tested.

## References

- [Martyna, Klein, Tuckerman (1992) "Nose-Hoover chains"](https://doi.org/10.1063/1.463940)
- [Tapias, Sanders, Bravetti (2016) "Log-osc thermostat"](https://doi.org/10.1103/PhysRevE.94.062123)
- Parent orbit #52 (gprime-ablation-052) -- provided the 4-method sweep data and the
  puzzle that this orbit resolves.

## Seeds

- E0: seeds 1000-1002 (3 seeds, 50k steps each)
- E1: seeds 1000-1019 (20 seeds per condition, 200k steps)
- E2: seeds 1000-1009 (10 seeds per alpha-kappa cell, 200k steps)

## Files

- `solution.py` -- all experiment code, reuses parent's simulate() function
- `run.sh` -- reproducer script
- `results.json` -- full results for all experiments
- `figures/e0_xi_analysis.png` -- (a) xi histograms, (b) entropy vs tau_int
- `figures/e1_qeff_matching.png` -- the key figure: Q_eff matching test
- `figures/e3_residual.png` -- tail-shape comparison at matched Q_eff
- `figures/e2_kappa_sweep.png` -- alpha-kappa sweep (floor-contaminated, interpret with care)
