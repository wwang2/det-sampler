---
strategy: log-osc-stage2-hermite-theory
type: experiment
status: complete
eval_version: eval-v1
metric: 3.181
issue: 77
parents:
  - orbit/spectral-gap-cert-075
---

# Log-osc Stage 2 Benchmarks + Hermite Spectral Theory

## Part A: Stage 2 Benchmarks

### Question

How does log-osc NH friction g(xi) = 2xi/(1+xi^2) compare to tanh NH and NHC(M=3) on Stage 2 systems (2D Gaussian mixture, Rosenbrock banana)?

### Method

- Custom Verlet integrator using g(xi) for friction rescaling (not raw xi)
- dt=0.005, 500K force evals, 3 seeds per condition
- Q sweep: {0.05, 0.1, 0.3, 1.0}
- Metrics: KL divergence (primary), tau_int, ESS/force-eval

### Results

#### Gaussian Mixture 2D (5 modes, radius=3, sigma=0.5)

| Method         | Q    | KL (mean +/- std)  | tau_int  | ESS/fe   |
|----------------|------|---------------------|----------|----------|
| NHC(M=3) tanh  | 1.0  | 0.286 +/- 0.229     | 19.9     | 0.00453  |
| NH tanh        | 0.05 | 1.178 +/- 0.109     | 2794     | 0.000042 |
| NH tanh        | 0.1  | **0.708 +/- 0.074** | 5757     | 0.000016 |
| NH tanh        | 0.3  | 1.500 +/- 0.495     | 3087     | 0.00140  |
| NH tanh        | 1.0  | 1.052 +/- 0.157     | 3145     | 0.000029 |
| NH log-osc     | 0.05 | 4.898 +/- 0.692     | 27.4     | 0.00371  |
| NH log-osc     | 0.1  | 4.327 +/- 0.702     | 28.0     | 0.00368  |
| NH log-osc     | 0.3  | 3.626 +/- 0.387     | 20.0     | 0.00450  |
| NH log-osc     | 1.0  | **0.908 +/- 0.091** | 3220     | 0.00151  |

**NHC(M=3) is clearly best** (KL=0.286). Log-osc is worse than tanh on GMM (best log-osc KL=0.908 vs best tanh KL=0.708). At small Q, log-osc fails catastrophically (KL~4-5) because xi grows large and g(xi)->0, removing friction entirely.

#### Rosenbrock 2D (a=0, b=5)

| Method         | Q    | KL (mean +/- std)  | tau_int  | ESS/fe   |
|----------------|------|---------------------|----------|----------|
| NHC(M=3) tanh  | 1.0  | 0.051 +/- 0.002     | 46.2     | 0.00196  |
| NH tanh        | 0.05 | 0.045 +/- 0.004     | 54.0     | 0.00170  |
| NH tanh        | 0.1  | 0.043 +/- 0.001     | 48.2     | 0.00187  |
| NH tanh        | 0.3  | **0.040 +/- 0.001** | 51.6     | 0.00175  |
| NH tanh        | 1.0  | 0.043 +/- 0.002     | 55.4     | 0.00167  |
| NH log-osc     | 0.05 | 0.180 +/- 0.017     | 44.1     | 0.00205  |
| NH log-osc     | 0.1  | 0.105 +/- 0.009     | 43.0     | 0.00209  |
| NH log-osc     | 0.3  | 0.056 +/- 0.006     | 52.8     | 0.00177  |
| NH log-osc     | 1.0  | **0.049 +/- 0.006** | 53.3     | 0.00170  |

Rosenbrock is an easier target. All methods achieve reasonable KL. Log-osc at Q=1.0 (KL=0.049) nearly matches NHC(M=3) (KL=0.051), but tanh at Q=0.3 (KL=0.040) beats both.

### Primary Metric

- metric = KL_losc_best / KL_nhc_baseline
- GMM: 0.908 / 0.286 = **3.18** (log-osc 3x worse than NHC)
- Rosenbrock: 0.049 / 0.051 = **0.96** (log-osc slightly better than NHC)
- **Overall metric: 3.18** (GMM is the harder, more diagnostic test)

### Interpretation

Log-osc NH's self-limiting friction g(xi)->0 as |xi|->inf is a **liability on multi-modal distributions**. When the sampler needs to cross barriers between distant modes (radius=3, sigma=0.5), large xi excursions are needed, and log-osc's vanishing friction allows xi to escape to infinity, destroying the thermostat coupling. Tanh's saturating friction (g->1) maintains control even at large xi.

On the unimodal Rosenbrock banana, the difference is small because xi rarely makes large excursions — the system stays near equilibrium where g'(0)=2 dominates for both functions.

---

## Part B: Hermite Spectral Gap Computation

### Question

What is the spectral gap of the NH Liouville operator for different g(xi) functions, and how does log-osc compare to tanh?

### Method

- Build Liouville operator L in orthonormal Hermite basis on L^2(mu)
- 1D HO, omega=1, kT=1, N=8 per dimension (dim=512 matrix)
- Spectral gap = min |Re(lambda)| over non-zero eigenvalues
- NH dynamics have time-reversal symmetry: eigenvalues come in +/-Re pairs
- Sweep alpha in {0.5, 1.0, sqrt(2), 2.0, 3.0, 4.0}, Q in {0.1, 1.0, 10.0}

### Results

| Method         | Q=0.1     | Q=1.0     | Q=10.0    |
|----------------|-----------|-----------|-----------|
| tanh(0.5*xi)   | 0.0156    | 0.0206    | 0.0125    |
| tanh(1.0*xi)   | **0.0415** | 0.0042   | 0.0110    |
| tanh(sqrt2*xi) | 0.0406    | 0.0402    | 0.0058    |
| tanh(2.0*xi)   | 0.0404    | 0.0392    | 0.0033    |
| tanh(3.0*xi)   | 0.0404    | 0.0341    | 0.0103    |
| tanh(4.0*xi)   | 0.0400    | **0.0659** | **0.0129** |
| log-osc        | **0.0434** | 0.0292   | 0.0113    |
| Kac optimal    | alpha=4.47 | alpha=1.41 | alpha=0.45 |

### Spectral Gap Ratios

- Q=0.1: log-osc/best_tanh = 1.048 (log-osc 5% better)
- Q=1.0: log-osc/best_tanh = 0.443 (log-osc 56% worse)
- Q=10.0: log-osc/best_tanh = 0.877 (log-osc 12% worse)

### Interpretation

1. **At Q=0.1** (tight coupling): log-osc has the largest spectral gap (0.043), 5% better than the best tanh. This is consistent with orbit 075's finding that log-osc advantages emerge at small Q / high condition number.

2. **At Q=1.0**: tanh with alpha=4.0 dominates (gap=0.066). Log-osc (gap=0.029) is much worse. The Kac prediction alpha_opt=sqrt(2) gives gap=0.040, but alpha=4.0 does better — suggesting the Kac formula underestimates the optimal damping strength at Q=1.

3. **Convergence note**: The gap decreases as basis size N increases (0.16 at N=4, 0.04 at N=8, 0.03 at N=10), consistent with NH being non-ergodic on 1D HO in the infinite-dimensional limit. The *relative* comparison between methods is the meaningful result.

4. **Log-osc g'(0)=2** acts like tanh(2*xi) near the origin but with vanishing tails. The spectral gap data confirms that the tails matter: at Q=1.0, tanh(4*xi) with much stronger saturation beats log-osc significantly.

---

## Figures

- `figures/stage2_gaussian_mixture_2d.png` — KL and tau vs Q for GMM
- `figures/stage2_rosenbrock_2d.png` — KL and tau vs Q for Rosenbrock
- `figures/stage2_summary_bar.png` — Best KL per method comparison
- `figures/spectral_gap_vs_alpha.png` — Spectral gap vs alpha for each Q
- `figures/spectral_gap_heatmap.png` — Heatmap + best-tanh vs log-osc bars

## Key Takeaways

1. **Log-osc NH fails on multi-modal distributions** (GMM KL=0.91 vs NHC KL=0.29, tanh KL=0.71). The vanishing friction at large xi is catastrophic for mode-hopping.

2. **Log-osc NH is competitive on unimodal curved distributions** (Rosenbrock KL=0.049 vs NHC KL=0.051).

3. **NHC(M=3) remains the strongest baseline** for multi-modal sampling. Single-thermostat NH (any g) struggles with separated modes.

4. **Spectral gap confirms log-osc advantage is Q-dependent**: better at small Q (tight coupling), worse at Q=1.0 where tanh with large alpha dominates.

5. **The self-limiting property of log-osc is a double-edged sword**: it prevents over-damping (good for stiff systems per orbit 075) but also prevents necessary friction for barrier crossing.
