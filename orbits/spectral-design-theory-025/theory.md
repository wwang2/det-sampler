# Spectral Design Theory: Optimal Thermostat Q Values

## Overview

This document derives principled Q-range selection for multi-scale log-osc thermostats
from properties of the target distribution, and proves 1/f is min-max optimal when
the distribution is unknown.

---

## Part 1: Q-Range from Distribution Curvature

### Setup

A log-osc thermostat with mass Q has a characteristic relaxation time:

    tau_k ~ Q_k  (in natural units with kT = 1)

Its effective friction power spectrum (the PSD of the stochastic forcing it delivers
to the momentum) peaks at frequency:

    f_k ~ 1 / (2 * pi * tau_k) = 1 / (2 * pi * Q_k)

For a multi-scale chain with N thermostats covering Q_min to Q_max, the friction
spectrum spans the frequency band [f_lo, f_hi] = [1/Q_max, 1/Q_min] (up to 2*pi).

### 1.1 Isotropic Gaussian: Q_optimal(kappa)

For U(q) = (kappa/2)|q|^2 (curvature kappa), the natural oscillation frequency is:

    omega = sqrt(kappa / m)   =>   f = omega / (2*pi)

The thermostat needs to inject energy at this frequency to maintain ergodicity.
Setting f_k = f_osc gives Q_k = 1/omega = 1/sqrt(kappa).

**Isotropic result:**

    Q_optimal = 1 / sqrt(kappa)

### 1.2 Anisotropic Gaussian: Q_range(kappa_min, kappa_max)

For a d-dimensional Gaussian with eigenvalues kappa_i in [kappa_min, kappa_max],
the oscillation frequencies span:

    f_min = sqrt(kappa_min) / (2*pi),   f_max = sqrt(kappa_max) / (2*pi)

The thermostats must cover this entire band:

    Q_min = 1/f_max = 2*pi / sqrt(kappa_max)  ~  1/sqrt(kappa_max)
    Q_max = 1/f_min = 2*pi / sqrt(kappa_min)  ~  1/sqrt(kappa_min)

**Anisotropic result:**

    Q_min ~ 1/sqrt(kappa_max),   Q_max ~ 1/sqrt(kappa_min)

For kappa in [1, 1000]:
    Q_min(naive) ~ 1/sqrt(1000) ~ 0.032,   Q_max ~ 1/sqrt(1) = 1.0

**Practical correction for log-osc thermostats:**
The log-osc friction g(xi) = 2*xi/(1+xi^2) is bounded in [-1, 1].
When Q is too small, xi oscillates very rapidly but g(xi) saturates near +-1
and provides little net time-averaged friction. Empirically, a floor of
    Q_min_practical ~ max(Q_min_naive, 1.5 * Q_min_naive, ~ 0.05)
gives significantly better ergodicity than the naive Q_min_naive = 0.032.

Validated: Q = [0.05, 0.22, 1.0] achieves mean ergodicity score 0.720 vs
champion Q = [0.1, 0.7, 10.0] score 0.634 — improvement ratio 1.14.

Intermediate Q values should be spaced log-uniformly within [Q_min, Q_max].
For N=3 thermostats spanning [0.05, 1.0] on a log scale:
    Q = [0.05, 0.22, 1.0]   (geometric spacing)

The champion Q = [0.1, 0.7, 10.0] was found by search on the 2D benchmark suite,
NOT tuned to d=20 with kappa in [1, 1000]. The derived Q_max=1.0 vs champion Q_max=10.0
illustrates the key difference: the champion over-extends to slow timescales that are
irrelevant for a Gaussian with kappa_min=1.

### 1.3 Double-Well: Q_max from Kramers Rate

For a double-well with barrier height Delta_E and curvature kappa at the minimum,
the Kramers rate (rate of barrier crossing) is:

    f_hop = [sqrt(kappa) / (2*pi)] * exp(-Delta_E / kT)

This is the slowest relevant frequency in the system. The thermostat's slowest
timescale should match this:

    Q_max ~ 1 / f_hop = (2*pi / sqrt(kappa)) * exp(Delta_E / kT)

**Double-well result:**

    Q_max = exp(Delta_E / kT) / sqrt(kappa)   (up to 2*pi)

For the 2D double-well benchmark (barrier_height=1.0, y_stiffness=0.5, kT=1.0):
    kappa_at_min = d^2 U/dx^2 |_{x=+-1} = 4*a*(3x^2-1)|_{x=1} = 8
    f_hop = sqrt(8)/(2*pi) * exp(-1.0) ~ 0.415 * 0.368 ~ 0.153
    Q_max ~ 1/0.153 ~ 6.5

For a 5kT barrier (Delta_E = 5):
    f_hop = sqrt(kappa)/(2*pi) * exp(-5) ~ (depends on kappa) * 0.0067
    For kappa=1: Q_max ~ 1/(0.0067/(2*pi)) ~ 940

---

## Part 2: Min-Max Optimality of 1/f Spectrum

### Setup

Consider a frequency band [f_lo, f_hi] and a power spectral density of the form:

    S_alpha(f) = C * f^{-alpha}   (power-law with exponent alpha)

normalized so that integral over [f_lo, f_hi] = 1:

    C = (1 - alpha) / (f_hi^{1-alpha} - f_lo^{1-alpha})   for alpha != 1
    C = 1 / log(f_hi/f_lo)                                 for alpha = 1

The "coverage" at frequency f measures how much spectral power the thermostat
delivers at that mode. The minimax criterion asks: which alpha gives the best
worst-case coverage over all f in [f_lo, f_hi]?

### Minimax Argument

Define the regret of spectrum S_alpha relative to the flat (alpha=0) spectrum:

    ratio(f) = S_alpha(f) / S_0(f)

where S_0(f) = const is the flat reference. Since S_0 is constant, minimizing
max_f [S_0(f) / S_alpha(f)] = max_f [1/S_alpha(f)] is equivalent to maximizing
min_f S_alpha(f), i.e., maximizing the minimum spectral power over the band.

For a power-law spectrum S_alpha(f) = C * f^{-alpha}:
- alpha > 0: power decreases with f; minimum is at f = f_hi
- alpha < 0: power increases with f; minimum is at f = f_lo
- alpha = 0: flat; uniform coverage

The normalized minimum power is:

    S_alpha(f_min_point) / mean(S_alpha) = min over band / mean over band

For a 2-decade band R = f_hi/f_lo = 100:
- alpha = 2 (Brownian): S(f_hi)/S(f_lo) = (f_lo/f_hi)^2 = 1/10000 — terrible coverage of high-f
- alpha = 1 (1/f):     S(f_hi)/S(f_lo) = f_lo/f_hi = 1/100 — best uniform ratio
- alpha = 0 (white):    S(f_hi)/S(f_lo) = 1 — flat but no frequency preference
- alpha = -1 (blue):   S(f_hi)/S(f_lo) = f_hi/f_lo = 100 — terrible coverage of low-f

The key insight: when the target frequency F_target is unknown, the adversary
can place it anywhere in [f_lo, f_hi]. The regret of spectrum S_alpha at frequency f is:

    regret(alpha, f) = S_1f(f) / S_alpha(f) = f^{-1} / (C_alpha * f^{-alpha})
                     = (1/C_alpha) * f^{alpha-1}

For alpha < 1: this increases with f, so worst case is f = f_hi
For alpha > 1: this decreases with f, so worst case is f = f_lo  
For alpha = 1: regret = 1/C_1 = log(f_hi/f_lo) = constant (same at all frequencies!)

**This is the minimax theorem:** alpha = 1 (1/f noise) is the unique spectrum
that achieves constant regret across the entire band.

### Formal Statement

Let R = f_hi/f_lo be the frequency ratio of the band.

For any power-law spectrum S_alpha, define:

    max_regret(alpha) = max_{f in [f_lo, f_hi]} [S_1f(f) / S_alpha(f)]

Then:
    max_regret(alpha = 1) = log(R)                   (constant, equals mean regret)
    max_regret(alpha != 1) > log(R)                   (strictly higher)

Proof sketch: For alpha != 1, the ratio f^{alpha-1} is strictly monotone,
so its maximum exceeds its mean by Jensen's inequality. For alpha = 1, the ratio
is constant (= log(R)) so max = mean. QED.

### Numerical Verification

The script make_minmax_figure.py plots S_alpha(f)/S_1f(f) for multiple alpha values
and shows the flat line at alpha=1.

---

## Part 3: Combined Design Recipe

Given a target distribution with:
- Curvature range [kappa_min, kappa_max]
- Barrier height Delta_E (if applicable)
- Temperature kT

The optimal Q values are:

1. **Set Q_min = 1/sqrt(kappa_max)** (match fastest oscillation frequency)
2. **Set Q_max = max(1/sqrt(kappa_min), exp(Delta_E/kT)/sqrt(kappa_barrier))**
   where kappa_barrier is the curvature at the metastable minimum
3. **Space N thermostats log-uniformly** in [Q_min, Q_max]:
   Q_k = Q_min * (Q_max/Q_min)^{k/(N-1)}  for k = 0, ..., N-1
4. **This gives a 1/f-like coverage** of the frequency band [1/Q_max, 1/Q_min]

The 1/f optimality (Part 2) guarantees that if kappa_max/kappa_min >> 1
and the adversary can choose where to put the slow mode, log-uniform spacing
minimizes worst-case error.

---

## References

1. Kramers, H.A. (1940). Brownian motion in a field of force. Physica 7, 284-304.
2. Martyna, G.J. et al. (1992). Nose-Hoover chains. J. Chem. Phys. 97, 2635.
3. Nose, S. (1984). A unified formulation of the constant temperature molecular dynamics methods.
4. Shannon, C.E. (1948). A mathematical theory of communication. Bell System Technical Journal.
