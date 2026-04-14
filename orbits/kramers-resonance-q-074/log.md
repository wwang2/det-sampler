---
strategy: kramers-resonance-multiscale-Q
type: experiment
status: falsified
eval_version: eval-v1
metric: 0.98
issue: 74
parents:
  - orbit/gprime-optimization-073
---

# Kramers-Resonance Multi-scale Q — Q_j = DkT/w_j^2 per NHC Chain Member

## Hypothesis

For NHC with M=2 chain members on anisotropic systems, set Q_j = s * D*kT/w_j^2
where w_j is the j-th characteristic frequency of the potential. This matches each
chain member to a different physical mode, potentially improving mixing vs uniform Q.

## Result: FALSIFIED

Kramers Q is consistently **worse** than uniform Q by ~2% across both kappa=10
and kappa=100. The frequency-matching prescription does not help; if anything, the
asymmetric Q values slightly hurt the chain coupling dynamics.

metric = tau_int(NHC2_uniform_best) / tau_int(NHC2_Kramers_best) = 0.98

## Phase 1: Fast Scan (1 seed, 100k force evals)

### Anisotropic Gaussian kappa=10 (w_fast=3.2)
| Q     | NH-losc | NHC2-uni | NHC2-Kr | NHC3-tanh | NHC2-KrR |
|-------|---------|----------|---------|-----------|----------|
| 0.03  | 145.9   | 145.0    | 148.5   | 145.3     | 131.1    |
| 0.1   | 141.8   | 137.4    | 151.2   | 135.9     | 130.4    |
| 0.3   | 126.6   | 129.9    | 130.5   | 130.1     | 127.2    |
| 1.0   | 132.0   | 127.9    | 130.6   | 126.7     | 126.6    |
| 3.0   | 127.2   | 127.3    | 127.2   | 126.7     | 126.6    |

### Anisotropic Gaussian kappa=100 (w_fast=10.0)
| Q     | NH-losc | NHC2-uni | NHC2-Kr | NHC3-tanh | NHC2-KrR |
|-------|---------|----------|---------|-----------|----------|
| 0.03  | 62.7    | 67.5     | 79.5    | 67.8      | 64.9     |
| 0.1   | 65.4    | 66.9     | 71.9    | 66.9      | 66.3     |
| 0.3   | 66.2    | 66.7     | 81.7    | 66.7      | 66.8     |
| 1.0   | 66.5    | 66.5     | 71.4    | 66.6      | 66.6     |
| 3.0   | 66.5    | 66.6     | 67.3    | 66.6      | 66.6     |

Key finding at kappa=100: Kramers Q (NHC2-Kr) is *worse* across all Q values,
especially at Q=0.3 where tau=81.7 vs 66.7 for uniform. The reversed assignment
(NHC2-KrR) performs similarly to uniform.

### Double-Well 2D (KL divergence)
| Q   | NH-losc | NHC2-uni | NHC2-Kr | NHC3-tanh |
|-----|---------|----------|---------|-----------|
| 0.1 | 0.2273  | 0.2250   | 0.2136  | 0.2166    |
| 0.3 | 0.2269  | 0.2309   | 0.2224  | 0.2280    |
| 1.0 | 0.2676  | 0.2356   | 0.2223  | 0.2305    |
| 3.0 | 0.3091  | 0.2129   | 0.2692  | 0.2731    |

Kramers Q shows slight KL improvement at small Q but the differences are small
and inconsistent.

### 1D Harmonic Oscillator (ergodicity score)
| Q   | NH-losc | NHC2-uni | NHC2-Kr | NHC3-tanh |
|-----|---------|----------|---------|-----------|
| 0.1 | 0.4761  | 0.8513   | 0.8513  | 0.8350    |
| 0.3 | 0.5721  | 0.8395   | 0.8395  | 0.8450    |
| 1.0 | 0.7738  | 0.8500   | 0.8500  | 0.8282    |
| 3.0 | 0.6783  | 0.8189   | 0.8189  | 0.8412    |

Kramers Q = uniform Q for 1D (one frequency, expected). All NHC methods
substantially outperform plain NH on ergodicity.

## Phase 2: Confirmation (3 seeds, 300k force evals)

### Anisotropic Gaussian kappa=10
| Method            | mean tau | std  |
|-------------------|----------|------|
| NHC2_Kr_s0.1      | 145.1    | 3.0  |
| NHC2_Kr_s0.3      | 135.1    | 2.5  |
| NHC2_Kr_s1.0      | 130.2    | 1.8  |
| NHC2_uni_Q0.1     | 134.2    | 0.6  |
| NHC2_uni_Q0.3     | 128.7    | 1.0  |
| NHC2_uni_Q1.0     | 127.9    | 1.3  |
| NHC3_tanh_Q0.1    | 132.8    | 1.0  |
| NHC3_tanh_Q0.3    | 128.7    | 0.4  |
| NHC3_tanh_Q1.0    | 127.4    | 0.1  |
| NH_logosc_Q0.1    | 133.6    | 11.7 |
| NH_logosc_Q0.3    | 127.5    | 7.6  |
| NH_logosc_Q1.0    | 128.3    | 3.6  |

Best NHC2 Kramers: 130.2 (s=1.0) vs best NHC2 uniform: 127.9 (Q=1.0).
Kramers is **1.8% worse**. NHC3 tanh Q=1.0 wins at 127.4.

### Anisotropic Gaussian kappa=100
| Method            | mean tau | std  |
|-------------------|----------|------|
| NHC2_Kr_s0.1      | 74.1     | 1.2  |
| NHC2_Kr_s0.3      | 74.5     | 3.6  |
| NHC2_Kr_s1.0      | 68.0     | 0.7  |
| NHC2_uni_Q0.1     | 66.9     | 0.0  |
| NHC2_uni_Q0.3     | 66.7     | 0.0  |
| NHC2_uni_Q1.0     | 66.6     | 0.0  |
| NHC3_tanh_Q0.1    | 67.1     | 0.1  |
| NHC3_tanh_Q0.3    | 66.7     | 0.0  |
| NHC3_tanh_Q1.0    | 66.6     | 0.0  |
| NH_logosc_Q0.1    | 67.0     | 1.1  |
| NH_logosc_Q0.3    | 67.9     | 1.2  |
| NH_logosc_Q1.0    | 67.0     | 0.4  |

Best NHC2 Kramers: 68.0 (s=1.0) vs best NHC2 uniform: 66.6 (Q=1.0).
Kramers is **2.1% worse**. Uniform Q shows remarkably low variance (std=0.0).

## Interpretation

1. **Kramers Q does not help**: The frequency-matching prescription Q_j = s*D*kT/w_j^2
   consistently underperforms uniform Q. The asymmetric mass assignment creates an
   imbalanced chain that couples poorly.

2. **Uniform Q is remarkably robust**: At kappa=100, NHC2 uniform Q shows near-zero
   variance across seeds and Q values in {0.1, 0.3, 1.0}. The chain self-regulates
   effectively with equal masses.

3. **NHC2 log-osc ~ NHC3 tanh**: Both perform similarly, with NHC3 tanh having
   slightly lower variance. Neither shows large advantage over plain NH log-osc.

4. **NH log-osc has higher seed variance**: Plain NH shows std up to 11.7 vs <1.3
   for NHC methods, confirming the chain improves consistency.

Total wall time: 1071s (17.9 min)
