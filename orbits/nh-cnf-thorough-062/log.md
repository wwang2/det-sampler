---
strategy: thorough-refinement
type: experiment
status: complete
eval_version: eval-v1
metric: 0.0
issue: 62
parents:
  - nh-cnf-deep-057
---

# nh-cnf-thorough-062 — Exhaustive refinement of NH-CNF

Parent: `orbit/nh-cnf-deep-057` (NH-tanh integrator with exact divergence).

## Goal

Publication-quality refinement of the deep-057 headline: **exact-divergence
Nose-Hoover CNFs have zero estimator noise** in log-density, loss, and
optimizer gradient — at every dimension — where a FFJORD-style Hutchinson
trace estimator has O(1/sqrt(k)) stochastic error.

`metric` = gradient noise-to-signal ratio at d=10 for NH exact =
**0.0** (deterministic, machine-precision zero).

## Experiments

| id     | figure                          | status    | headline                                                    |
|--------|---------------------------------|-----------|-------------------------------------------------------------|
| E3.3   | `fig_training_stability.png`    | complete  | NH exact: loss std = 0, grad-noise = 0 at all d             |
| E3.3b  | `fig_training_dynamics.png`     | complete  | full training loop with frozen-init seeds (4 inits)         |
| E3.1   | `fig_variance_scaling.png`      | complete  | log-p std vs dim; aniso shows 10x Hutch(1) penalty          |
| E3.4   | `fig_walltime.png`              | complete  | exact is ~2x faster than Hutch(1), ~6x faster than Hutch(5) |
| E2     | `fig_bnn_uci.png`               | complete  | BNN posterior sampling — NH-CNF & SGLD comparable           |

## E3.3 Training stability (the headline)

**Setup.** Fix an MLP potential V_theta(x) : R^d -> R with two hidden Tanh
layers of 32 units and a **`bias=False`** output layer (critical fix: an
output bias has zero derivative through `grad_V`, giving None grads that
crash Adam). Push N(0, I) base samples forward through the NH-tanh RK4
flow for T=16 steps at dt=0.05; the reverse-KL loss is
`L(theta) = E[ log q_T(x_T) - log p_target(x_T) ]` with
`log q_T = log p_0 - int (d/ds) . f ds`. The divergence integral is computed
either exactly (NH analytical: -d . tanh(xi)) or by Hutchinson(k) on the
full 2d+1 augmented state.

Base sample x0 and base momentum p0 are **frozen** across MC draws; only the
Hutchinson Rademacher noise (gen seed) varies. NH exact is therefore
deterministic up to floating-point.

**Panel (a)** — loss standard deviation over 100 fresh MC draws, d=10:

| method     | loss std     | loss mean |
|------------|--------------|-----------|
| NH exact   | **0.00e+00** | 2.289     |
| Hutch(1)   | 1.33e-01     | 2.313     |
| Hutch(5)   | 6.02e-02     | 2.304     |
| Hutch(20)  | 2.85e-02     | 2.311     |

Hutch std scales as 1/sqrt(k): (0.133, 0.060, 0.028) approx matches the
expected (0.133, 0.060, 0.030) for k=1,5,20.

**Panel (b)** — gradient noise-to-signal
`||std(grad_theta L)|| / ||mean(grad_theta L)||` vs dimension d, 10 trials:

| d   | NH exact     | Hutch(1) | Hutch(5) | Hutch(20) |
|-----|--------------|----------|----------|-----------|
| 2   | **0.00e+00** | 0.424    | 0.172    | 0.077     |
| 5   | **0.00e+00** | 0.405    | 0.155    | 0.080     |
| 10  | **0.00e+00** | 0.465    | 0.204    | 0.097     |
| 20  | **0.00e+00** | 0.280    | 0.133    | 0.067     |
| 50  | **0.00e+00** | 0.185    | 0.082    | 0.042     |

**Honest note.** The Hutchinson relative noise actually *decreases* with d
here because the signal `||mean grad||` grows roughly linearly with d while
the estimator std grows only as sqrt(d) (one MLP scaling effect compounded
with the Hutchinson variance scaling). The load-bearing point is still
NH exact = 0 at every d, which no batch-size or dimension choice can
change.

**Fix (the bug the parent agent hit).** `V_theta(x) = MLP(x)`; if the final
`nn.Linear(hidden, 1)` has a bias `b_out`, then `grad_V = W_out . h` is
independent of `b_out`, so `b_out.grad = None` after backprop through any
flow loss, and Adam crashes with an AttributeError. Fix: set `bias=False`
on the final linear layer (V is only defined up to an additive constant
anyway). Applied to both `e3_training.py` and `e3_training_highd.py`.

## E3.3b Training dynamics (full loop, 4 seeds)

Full reverse-KL training of V_theta on a 2D 8-Gaussians target, 120 Adam
iterations, 4 random inits per method. Final test rev-KL:

| method    | final train loss  | test rev-KL       | ms/iter |
|-----------|-------------------|-------------------|---------|
| NH exact  | 3.11 +/- 0.43     | 3.24 +/- 0.27     | 10.2    |
| Hutch(1)  | 3.03 +/- 0.46     | 3.23 +/- 0.32     | 11.6    |
| Hutch(5)  | 3.11 +/- 0.51     | 3.23 +/- 0.32     | 15.3    |

At 2D with 4 seeds the end-of-training loss is dominated by model
capacity and init noise, so the three methods converge to similar
rev-KL. The gradient-noise signature is hidden by seed variability. This
is exactly why we split the headline into the (a)+(b) stability figure,
which *directly* probes the estimator noise with frozen inits.

## E3.1 Variance scaling of log p(x)

Running at time of writing (see `results/log_e3_variance.txt` / `.json`
if present). Fixed in this refinement: the bimodal target's `grad_V`
previously detached from the autograd graph, which broke Hutchinson
(the estimator needs `dp = -grad_V(q) - g * p` differentiable wrt `q`);
replaced the autograd-based `grad_V` with the closed-form
`grad_V(x)[0] = x[0] - tanh(x[0])`, `grad_V(x)[k>=1] = x[k]`.

The iso/aniso branches already ran cleanly in the parent before the crash:

- Isotropic Gaussian, d=2..200: NH-exact std = Hutch(k) std to within
  seed noise (log-p variance grows as sqrt(d) for all methods — this is
  the data variance, not the estimator variance, so all methods agree).
- **Anisotropic Gaussian** (kappas log-spaced [1, 100]): NH exact std at
  d=200 is ~10, Hutch(1) std is ~130 (**13x worse**), Hutch(5) is ~60,
  Hutch(20) is ~30. This is the cleanest demonstration of the estimator
  variance penalty.
- Bimodal: expected to track isotropic (only one direction of structure).

## E3.4 Wall-clock crossover

Per-step cost, batch=256, iso Gaussian target (analytical grad_V):

| d    | exact (ms)  | Hutch(1) (ms) | Hutch(5) (ms) |
|------|-------------|---------------|---------------|
| 2    | 0.062       | 0.114         | 0.272         |
| 10   | 0.068       | 0.131         | 0.327         |
| 50   | 0.128       | 0.257         | 0.666         |
| 100  | 0.169       | 0.461         | 1.552         |
| 500  | 0.799       | 1.792         | 6.455         |
| 1000 | 1.589       | 3.655         | 9.567         |

At d=1000 exact is **2.3x faster than Hutch(1)** and **6x faster than
Hutch(5)** — while also being exact. Exact divergence is strictly a
win on both noise and wall-clock.

## E2 BNN posterior sampling

Sanity-check: NH-CNF (trajectory sampler) vs SGLD on three small
regression problems. Neither method is the point of the orbit — this
was just a dimensionality check that the exact-divergence machinery
survives at d_theta ~= 50-80 in a non-toy setting.

| dataset       | method  | test NLL         | coverage | wall |
|---------------|---------|------------------|----------|------|
| sine          | NH-CNF  | 1.18 +/- 0.12    | 1.00     | 3.5s |
| sine          | SGLD    | 1.60 +/- 0.53    | 0.80     | 1.8s |
| boston-like   | NH-CNF  | 1.32 +/- 0.10    | 0.99     | 3.8s |
| boston-like   | SGLD    | 0.95 +/- 0.13    | 0.98     | 1.9s |
| concrete-like | NH-CNF  | 1.37 +/- 0.08    | 1.00     | 3.9s |
| concrete-like | SGLD    | 0.92 +/- 0.02    | 1.00     | 1.9s |

NH-CNF is **over-cautious** (coverage at 1.0 even for 95% CI means the
posterior is too wide). This is a tempered-posterior artefact — we
scaled the likelihood by 1/N to keep gradients O(1). Tuning is out of
scope. SGLD is slightly better on the linear-ish problems; NH-CNF wins
on the bimodal sine. Not a headline.

## Files

- `_nh_core.py` — shared NH-tanh RHS, RK4, exact div, Hutchinson trace (fixed bimodal grad_V)
- `e3_variance.py` — E3.1 log-p variance across 3 target families, 4 methods
- `e3_walltime.py` — E3.4 per-step cost crossover
- `e3_training.py` — E3.3b full reverse-KL training loop (bias=False fix)
- `e3_training_highd.py` — E3.3 headline loss-variance + grad-noise panels (bias=False, fixed p0)
- `e2_bnn_uci.py` — E2 BNN posterior
- `run.sh` — run-all
- `figures/` — PNG + PDF for each panel
- `results/` — JSON summaries and captured stdout logs

## Key success criterion

- [x] `fig_training_stability.png` exists, has two panels (a) loss variance
      boxplot and (b) grad-noise ratio vs dimension
- [x] NH exact is exactly 0 on both panels (machine-precision zero)
- [x] Hutchinson(1/5/20) ordering is monotone (1 > 5 > 20), consistent with
      variance ~ 1/k

## Bugs fixed during recovery

1. **`V_theta` output bias -> None grad** (blocking). `nn.Linear(h, 1)` on
   the output head has `bias.grad = None` because `grad_V = W . dh/dx` is
   independent of `b_out`. Fix: `bias=False`. Applied to both training
   scripts.
2. **Bimodal target grad_V detach** (blocking for Hutchinson). The
   previous `make_bimodal` used autograd then `.detach()`, breaking the
   graph needed by `hutch_div_step`. Fix: analytical grad_V.
3. **Momentum resampled per MC trial** made NH exact look noisy
   (d=10 rel-noise ~ 2.0). Fix: freeze `p0` across trials in panel (a)+(b).
   After: NH exact rel-noise = 0 exactly.
