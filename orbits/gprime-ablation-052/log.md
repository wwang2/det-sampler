---
strategy: controlled-ablation
type: experiment
status: complete
eval_version: eval-v1
metric: 15.5
issue: 52
parents:
  - orbit/friction-survey-045
---

# gprime-ablation-052

## Glossary

- **g(xi)**: friction function applied to thermostat variable
- **g'(xi)**: derivative of g w.r.t. xi; effective instantaneous friction
- **g'(0)**: friction slope at the origin (coupling strength)
- **Q**: thermostat mass parameter
- **tau_int**: integrated autocorrelation time of q_d^2, averaged over dimensions
- **NHC**: Nose-Hoover Chain
- **d**: spatial dimension of target
- **kappa**: anisotropy ratio (max/min eigenvalue) of the Gaussian target

## Result

**The hypothesis fails.** On d=10 anisotropic Gaussian (kappa=100), the sign of g'(xi)
is NOT the causal factor for the tau_int gap reported in orbit #47. In fact, the gap
itself is smaller and less clean than #47 suggested once Q is scanned carefully.

Median tau_int at each method's best Q (20 seeds, 200k force evals, N=5 parallel thermostats):

| Method            | g'(0) | g' sign      | best Q_c | median tau_int | IQR           |
|-------------------|:-----:|--------------|:--------:|:--------------:|---------------|
| log-osc           | 2     | changes sign | 30       | 15.5           | [10.0, 261.6] |
| clipped-log-osc   | 2     | >= 0         | 30       | 15.5           | [10.0, 261.6] |
| tanh-scaled       | 2     | >= 0, smooth | 30       | 55.1           | [10.9, 202.2] |
| tanh-ref          | 1     | >= 0, smooth | 30       | 5.7            | [5.3, 9.3]    |

Further, at Q_c >= 100 all four methods collapse to tau_int ~ 5.1-5.2 — a
**Hamiltonian-dynamics floor** from the autocorrelation estimator on a nearly-periodic
trajectory (verified by running Q = 1e8: tau_int = 4.97-5.02). Q >= 100 means the
thermostat is effectively disabled.

### The decisive comparison: log-osc vs clipped-log-osc

These two functions share identical g, g' on |xi| <= 1 and differ only in the sign-changing
tail region. The experiment gives:

| Q_c   | log-osc med      | clipped-log-osc med | ratio |
|:-----:|-----------------:|--------------------:|:-----:|
| 0.3   | 822              | 1247                | 1.52  |
| 1.0   | 1199             | 1498                | 1.25  |
| 3.0   | 1479             | 1536                | 1.04  |
| 10.0  | 1151             | 1026                | 0.89  |
| 30.0  | 15.5             | 15.5                | 1.00  |
| 100.0 | 5.3              | 5.3                 | 1.00  |

**Clipping the negative-g' tail has no systematic effect** — at small Q clipped is slightly
WORSE (1.25-1.52x), at large Q they are identical. The 536x improvement attributed to
"removing the g' sign change" in orbit #47 does not reproduce.

### What actually discriminates

At Q=10 (a regime where the thermostat is clearly active), the ordering is:

- tanh-ref (g'(0)=1): tau = 317
- tanh-scaled (g'(0)=2): tau = 921
- clipped-log-osc (g'(0)=2): tau = 1026
- log-osc (g'(0)=2): tau = 1151

At Q=30:

- tanh-ref: tau = 5.7
- log-osc: tau = 15.5
- clipped-log-osc: tau = 15.5
- tanh-scaled: tau = 55.1

The single variable that tracks performance here is **not g' sign and not g'(0) directly**.
Rather, tanh-ref has a *lower* g'(0) than the other three but beats them all, and
tanh-scaled (which doubles the coupling of tanh-ref) is significantly *worse* than
tanh-ref. If anything, **smaller g'(0) is better** in this regime for this target.

## Approach

Reuses the parent's (`orbit/friction-survey-045`) `sim_multi` integrator — parallel
Nose-Hoover-like thermostats with a shared friction scalar g(sum xi_i). Target: d=10
anisotropic harmonic with geometric ladder kappa_i = 100^(i/9), i=0..9, following
friction-survey-045 E2. Step size dt = 0.05/sqrt(100); 200k steps; N=5 parallel
thermostats spanning Q in [Q_c/3, 3 Q_c] geometrically; seeds 1000-1019; Q_c scanned
over {0.3, 1, 3, 10, 30, 100, 300}. For each method-Q_c cell the tau_int estimator is
applied to q_d^2 on the sub-sampled (rec=4) trajectory and averaged over dimensions.

The key ablation variable is g(xi):

1. `log-osc`:         g = 2 xi / (1+xi^2)                    g'(0)=2, g' changes sign at |xi|=1
2. `clipped-log-osc`: g = 2 xi/(1+xi^2) for |xi|<=1 else sign(xi)    g'(0)=2, g'>=0, odd
3. `tanh-scaled`:     g = 2 tanh(xi)                         g'(0)=2, g'>=0, smooth
4. `tanh-ref`:        g = tanh(xi)                           g'(0)=1, g'>=0, smooth

Note: a first draft of `clipped-log-osc` used `max(0, g)` which breaks odd symmetry and
therefore detailed balance; results were misleading (freezing). The corrected version
saturates to +/-1 outside |xi|=1, preserving odd symmetry while eliminating the
negative-g' region.

## What Happened

1. Wrote `solution.py` reusing friction-survey-045's integrator; added the new
   friction functions. Validated pipeline on a tiny sweep.
2. Ran the full sweep over 4 methods x 7 Q_c x 20 seeds = 560 tasks, 200k steps each,
   with a ProcessPoolExecutor. Total wallclock ~580 s.
3. Initially saw all methods collapse to tau_int ~ 5 and thought the ablation was
   null. Diagnosed the floor: with Q_c >= 100 the thermostat is effectively turned off,
   and the ACF estimator applied to the nearly-periodic Hamiltonian trajectory of
   a d=10 harmonic returns a short, windowing-bounded tau ~ 5. Verified by running
   Q = 1e8 (no-thermostat limit) -> tau = 4.97-5.02. So "best Q" reporting is a trap
   on this target: the "best" tau is just "thermostat off".
4. Refocused on small/intermediate Q_c (0.3-30), which IS the regime where the
   thermostat is actively mixing. Here the four methods are clearly distinguished,
   but the ordering refutes the hypothesis.

## What I Learned

1. **The sign of g' is not the causal factor** for tanh's reported advantage over
   log-osc on this target. log-osc and clipped-log-osc behave nearly identically at
   every Q — in fact clipped is slightly WORSE at small Q. This flatly contradicts
   the "g'>=0 is better" claim from my campaign notes.

2. **The "536x gap" from orbit #47 is probably an artifact** of picking the best Q
   without excluding the Hamiltonian-floor regime for one method and not the other.
   When both methods are scanned over the same Q grid and tau is compared at each Q,
   the gap ranges from ~1x (at large or very small Q) to at most ~4x (at Q=10, where
   tanh-ref beats log-osc). A 536x gap is not reproducible with 20 seeds here.

3. **tanh-ref (g'(0)=1) beats all three g'(0)=2 variants at every active Q.**
   Scaling tanh up from coefficient 1 to coefficient 2 made it dramatically worse
   at Q=30 (5.7 -> 55). So "larger g'(0) = more coupling = better" is wrong too.
   The operative knob for this target is more subtle than either orbit #47 or
   parent orbit #45 suggested.

4. **tau_int on Hamiltonian-limit dynamics is a trap.** On a harmonic system, turning
   the thermostat off gives the lowest estimator output because the sinusoidal
   autocorrelation with windowing returns tau ~ 5, not because the sampling is good
   (ergodicity is broken — each trajectory stays on its initial energy surface).
   Future benchmarks on this target should either use a non-harmonic confining
   potential, or report an ergodicity-sensitive metric (e.g., convergence of the
   empirical covariance to the target) rather than tau_int alone.

## Prior Art & Novelty

### What is already known
- Log-osc thermostat: [Tapias et al. 2016](https://doi.org/10.1103/PhysRevE.94.062123)
- Nose-Hoover chain theory: [Martyna, Klein, Tuckerman 1992](https://doi.org/10.1063/1.463940)
- Friction-function frequency ceiling survey: orbit #45 (friction-survey-045)
- Prior "g'>=0 is the discriminant" framing: orbit #45 (this orbit refutes that framing on d=10)
- "Master theorem" / Q*omega=1 resonance: orbits #40, #41 (see MEMORY notes)

### What this orbit adds
- A controlled ablation holding g'(0)=2 fixed across three functions that differ only
  in the large-|xi| tail behavior of g'.
- Direct evidence that **clipping the negative-g' region has no effect** on tau_int
  for log-osc on the d=10 anisotropic Gaussian.
- Identification of the Hamiltonian-limit estimator floor (tau~5) that obscures
  comparisons when Q is picked "optimally" without a check on whether the thermostat
  is active.
- A concrete refutation of the "sign of g' is causal" narrative for this target.
  The real discriminant appears to be a more subtle combination of g'(0) and the
  large-xi saturation value, with tanh-ref (smallest g'(0), lowest saturation)
  winning in the active regime.

### Honest positioning
This orbit is a negative result against the hypothesis it was designed to test.
The paper narrative that "g'(xi) >= 0 is the key criterion" (MEMORY: project_paper_narrative)
is not supported on this benchmark. Before re-advancing that narrative, the
discrepancy with orbit #47 should be resolved: either #47's 536x gap was a
methodology artifact (most likely — comparing non-matched Q, or one method at its
Hamiltonian floor and the other not), or some other uncontrolled variable drives it.

## References

- [Tapias, Sanders, Bravetti (2016) "Sampling with log-osc thermostat"](https://doi.org/10.1103/PhysRevE.94.062123)
- [Martyna, Klein, Tuckerman (1992) "Nose-Hoover chains"](https://doi.org/10.1063/1.463940)
- Parent orbit #45 (friction-survey-045) — provided the sim_multi integrator and the
  "g'>=0 is the discriminant" hypothesis this orbit tests.
- Orbit #47 (paper-experiments-047) — source of the "536x gap" claim this orbit fails
  to reproduce.

## Seeds

- Main sweep: seeds 1000-1019 (20 seeds per method x Q_c cell).
- Hamiltonian-limit diagnostic: seeds 1000-1004 at Q=1e8.

## Files

- `solution.py` — friction functions, integrator, parallel sweep driver.
- `make_figure.py` — 3-panel consolidated figure.
- `results.json` — full per-method per-Q_c per-seed tau_int values and summary.
- `figures/ablation.png` — (a) g and g', (b) tau_int vs Q_c medians with IQR bands,
  (c) box plot of tau_int at each method's best Q_c.
- `run.sh` — reproducer.
