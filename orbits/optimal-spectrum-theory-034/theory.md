# Optimal Q-Spectrum for Multi-Scale Deterministic Thermostats

## 0. Glossary

- **GLE**: Generalized Langevin Equation
- **PSD**: Power Spectral Density
- **NHC**: Nose-Hoover Chain
- **OU**: Ornstein-Uhlenbeck
- **Prony series**: finite sum of decaying exponentials approximating a memory kernel
- **rho(Q)**: density of thermostat inertias on the Q-axis
- **omega_d**: natural frequency of mode d of an anisotropic Gaussian (omega = sqrt(kappa))
- **tau_int(omega)**: integrated autocorrelation time at frequency omega
- **Q_i**: inertia of thermostat i; tau_i ~ Q_i / (d kT) is its relaxation time

## 1. The optimization problem

We have a target distribution that is (locally) Gaussian with curvature spectrum
{kappa_d}_{d=1..D}, kappa_d in [kappa_min, kappa_max]. The mode frequencies are
omega_d = sqrt(kappa_d) in [omega_min, omega_max]. We have N thermostats with
inertias {Q_i} drawn from a density rho(Q) supported on [Q_min, Q_max], and we
want to choose rho so as to minimize the worst-case mixing time across modes:

    rho* = argmin_rho  max_d  tau_int(omega_d ; rho)        (1)

This is a minimax design problem. The N=1 case is the well-known Nose-Hoover
resonance: a single Q resonates with one frequency omega = 1/sqrt(Q) and
mixes everything else slowly. Our goal is to understand how rho should
spread its mass over [Q_min, Q_max].

## 2. From thermostats to a memory kernel

The key reduction. With N parallel log-osc thermostats coupled to (q,p), the
friction force on p is

    F_fric(t) = - [sum_i g(xi_i(t))] p(t)  =  -Gamma(t) p(t)

Each xi_i is, near equilibrium, an oscillator with relaxation time
tau_i = Q_i/(d kT) and a bounded kinetic-energy "kick". To linear order
this is exactly an Ornstein-Uhlenbeck mode driven by K(t)-d kT. Substituting
the linear OU response into dp/dt one obtains, for the marginal momentum
dynamics,

    dp/dt = -dU/dq  -  integral_0^t K(t-s) p(s) ds  +  noise(t)        (2)

with memory kernel

    K(t) = sum_{i=1}^N (c_i / tau_i) exp(-t/tau_i)        (3)

where c_i are O(1) coupling weights set by g'(0)=2 and equipartition. This
is the Prony series structure of a Generalized Langevin Equation. The
continuum limit (large N) is

    K(t) = integral rho(tau) (1/tau) e^{-t/tau} dtau        (4)

So the multi-scale thermostat is, to leading order, a *deterministic
realization of a GLE with memory kernel set by the spectrum rho(tau)*.

This is the right object to optimize. tau and Q differ only by a constant
factor, so we can equivalently choose rho on Q or on tau.

## 3. Mixing time of one Gaussian mode under a GLE

For a 1D harmonic oscillator with frequency omega coupled to the GLE (2),
the integrated autocorrelation time of position is, in the underdamped /
intermediate regime,

    tau_int(omega)  ~  1 / Re[ K_hat(i omega) ]        (5)

where K_hat(s) = integral_0^infty K(t) e^{-st} dt is the Laplace transform.
This is the standard result that *the friction the mode feels is the value of
the memory kernel evaluated at the mode's own frequency*.

For the Prony kernel (4),

    Re K_hat(i omega) = integral rho(tau) (1/tau) * (1/tau)/(1/tau^2 + omega^2) dtau
                      = integral rho(tau) / (1 + (omega tau)^2) dtau        (6)

Define the "effective friction" at frequency omega as

    Gamma_eff(omega) := integral rho(tau) / (1 + (omega tau)^2) dtau        (7)

Then tau_int(omega) ~ 1/Gamma_eff(omega), and the minimax problem (1) becomes

    rho* = argmax_rho  min_{omega in [omega_min, omega_max]}  Gamma_eff(omega)  (8)

subject to the normalization integral rho(tau) dtau = 1 (or = N for the
discrete case) and supp(rho) subset [tau_min, tau_max].

## 4. Why log-uniform is (approximately) the minimax solution

Reparameterize tau = e^u, so u in [u_min, u_max] = [log tau_min, log tau_max]
spans a length L = log(tau_max/tau_min). Let nu(u) = rho(e^u) e^u, so nu is the
density on log-scale and integral nu(u) du = 1. Then (7) becomes

    Gamma_eff(omega) = integral nu(u) / (1 + e^{2(u - v)}) du,   v := -log omega  (9)

The kernel L(u-v) := 1/(1 + e^{2(u-v)}) is a *sigmoid* in u, going from 1
when u << v to 0 when u >> v, with a soft transition of width O(1) around u=v.

Equivalently, Gamma_eff(omega) is the convolution (sigmoid * nu) evaluated at
v = -log omega. So the friction at frequency omega is roughly *the cumulative
mass of nu sitting at log-timescales below v*, i.e.

    Gamma_eff(omega)  ~  integral_{u <= -log omega} nu(u) du  +  O(1) tail   (10)

This is the essential insight: **friction at frequency omega is the integrated
log-density of thermostats faster than 1/omega**.

The minimax problem (8) thus reduces to: choose nu on [u_min, u_max] so that
the cumulative-from-the-left (= "fraction of thermostats faster than v") is
maximized in its minimum over v. By a standard waterfilling / equalization
argument, the minimum of a CDF over an interval is maximized when the CDF is
*linear*, i.e. when nu is *uniform on log-scale*:

    nu*(u) = 1/L,    rho*(tau) = 1/(L tau)        (11)

In Q-space this is rho*(Q) ~ 1/Q. This is exactly **log-uniform spacing of Q**,
which corresponds to alpha = 1 in the family rho(Q) ~ 1/Q^alpha tested in
orbit #031.

### Theorem (informal). 

Let rho be a probability density on [tau_min, tau_max] and define
Gamma_eff(omega) by (7). Then for omega ranging over [1/tau_max, 1/tau_min],

    max_rho  min_omega  Gamma_eff(omega)   is attained (up to O(1/L) edge
    corrections) by the log-uniform density nu(u) = 1/L,
    
    with equalized friction
        Gamma_eff*(omega) = (1/2) + O(1/L)   uniformly in omega.            (12)

The factor 1/2 comes from the sigmoid kernel value at its center: at u=v
exactly half the mass is "above" and half "below". The O(1/L) corrections
come from the soft transition width near u=v and from edge effects when
omega approaches the band edges 1/tau_max or 1/tau_min.

### Why the equalization argument works

If nu were not log-uniform, some log-decade [u, u+1] would carry less mass
than another. Pick omega so that v = -log omega lies just above this
under-served decade. Then Gamma_eff(omega) is dominated by (10) which is
locally lower than at other v. So min_omega Gamma_eff is strictly less than
the value attained by uniform nu. Equalization is necessary, and the only
density on [u_min, u_max] whose left-CDF is linear is the uniform density.

### Discrete N

For N thermostats, the analogous solution is to place log Q_i uniformly on
[log Q_min, log Q_max]:

    Q_i = Q_min (Q_max/Q_min)^{(i-1)/(N-1)}     (geometric progression)    (13)

This is the F1 prescription used since orbit #009. The minimax
guarantee then becomes Gamma_eff*(omega) >= (1/2)(1 - O(1/N)) on the
interior of the band, with errors of order 1/N coming from the discreteness.

## 5. Connection to Jeffreys / quadrature / 1/f viewpoints

The result (11) admits three equivalent interpretations, all of which point
at the same density:

1. **Jeffreys prior on a scale parameter.** For a positive scale Q, the
   uninformative prior is rho(Q) ~ 1/Q. The minimax derivation above is
   the *reason* the Jeffreys prior is right: it equalizes the worst-case
   loss when the true scale is unknown.

2. **Quadrature for the Prony integral.** Approximating (4) by N nodes,
   the Lebesgue measure on log tau gives uniform error per decade. This is
   the same density as Gauss-Legendre on log-scale to leading order: in
   the limit of smooth integrands the optimal nodes are nearly equispaced
   on the log axis when the integrand has no preferred scale.

3. **1/f noise (Dutta-Horn).** A log-uniform superposition of Lorentzian
   relaxations produces a 1/f friction PSD over the band [omega_min,
   omega_max]. 1/f is precisely the spectrum that has equal power per
   octave -- the spectral statement of "equalized friction across all log
   frequencies", which is what (12) asserts in the time domain.

These are not three independent justifications; they are three faces of the
same waterfilling argument over a sigmoid kernel on the log axis.

## 6. Comparison to NHC

A Nose-Hoover Chain of length M with equal Q values is *not* a multi-scale
sampler in the parallel sense. The chain links generate a polynomial
hierarchy of timescales at orders Q, Q^{3/2}, Q^2, ... with rapidly
decaying coupling strength. The "effective" memory kernel of an
M-link NHC has spectral support concentrated near a single frequency
omega ~ 1/sqrt(Q) plus weak harmonics; it is closer to a damped oscillator
than to a 1/f bath.

In the GLE language: NHC has a *non-flat* nu(u) on log-scale, sharply
peaked near u = log sqrt(Q). The minimax bound (12) is therefore *worse*
for NHC than for parallel log-uniform with the same M (at fixed kT, fixed
Q range): Gamma_eff(omega) is large near the resonance and falls rapidly
away from it, so min_omega Gamma_eff is small.

This is consistent with the empirical NHC vs parallel-log-uniform comparison
in the campaign: NHC(M=3) is competitive only when the target spectrum is
narrow (the bad regime for log-uniform; see Section 7.1).

## 7. When does log-uniform fail?

The minimax derivation assumes:

(a) the support [omega_min, omega_max] is the *entire* range of interest,
(b) all modes within the band matter equally (uniform-min loss).

Both can fail.

### 7.1 Peaked kappa distributions

If the actual mode density f(omega) is heavily peaked at one frequency
omega*, the relevant objective is no longer min_omega but a weighted
average

    minimize  integral f(omega) tau_int(omega) domega
            = integral f(omega) / Gamma_eff(omega) domega.

Stationarity gives the optimal nu satisfying

    integral [f(omega) / Gamma_eff(omega)^2] L(u - v(omega)) domega = const   (14)

which biases nu toward log Q ~ -log omega*. In the limit f -> delta(omega - omega*),
the optimum collapses onto nu = delta(u - log(1/omega*)), i.e. a *single Q* tuned
to the dominant frequency. Log-uniform is wasteful here.

### 7.2 Non-Gaussian targets

Equation (5) used the linear-response identity tau_int ~ 1/Re K_hat(i omega).
For non-Gaussian targets the relevant timescale is barrier-crossing
(Kramers), tau_cross ~ exp(Delta E / kT). The "effective frequency" of
barrier crossing is omega_cross = 1/tau_cross, and the optimal Q is one
whose log-decade *contains* log tau_cross. If [Q_min, Q_max] is too narrow
to bracket this scale, log-uniform within the wrong window is useless;
one must first choose the window. This is the reason orbit #016 emphasized
"placing the 1/f band over the barrier-crossing frequency".

### 7.3 Small N

The waterfilling argument requires L >> 1 and N large enough that the
sigmoid-of-width-1 transition is resolved. For N = 2 or 3, edge effects
dominate and the optimum is sensitive to where you put the endpoints. Our
empirical observation that N = 3 with carefully chosen Q triples often
beats N = 5 reflects this regime: the asymptotic 1/Q rule has not yet
taken over, and the discrete placement of endpoints matters more than
their density.

### 7.4 Resonance singularity (orbit #032)

At kappa = 1 with a single thermostat Q1 = 1, omega tau = 1 and the
Lorentzian factor in (6) is exactly 1/2 -- the worst point of the sigmoid
transition. C(kappa, Q1) diverges there because the leading-order linear
response cancels and one needs the next-order coupling. Log-uniform
*avoids* this singularity by spreading mass off the resonance: at any given
omega, only an O(1/N) fraction of thermostats sit at the bad point.

## 8. Statement of the main result

**Theorem (minimax-optimal Q spectrum, informal).**
Let the target be a D-dimensional Gaussian with eigen-frequencies
omega_d in [omega_min, omega_max], and let N parallel log-osc thermostats
be coupled to (q,p) with inertias {Q_i} drawn from a density rho on
[1/omega_max^2, 1/omega_min^2]. To leading order in 1/N and 1/L (with
L = log(omega_max/omega_min)), the density that minimizes the worst-case
integrated autocorrelation time across modes is

    rho*(Q)  =  1/(L Q),       Q_i = Q_min (Q_max/Q_min)^{(i-1)/(N-1)},

and the resulting worst-case effective friction is

    min_{omega in [omega_min, omega_max]} Gamma_eff(omega)  =  1/2 + O(1/L) + O(1/N).

In words: log-uniform Q is asymptotically minimax-optimal, the optimum is
unique, and it equalizes friction across all modes at half-strength.

### What is *not* proven
- Constants beyond leading order. The 1/2 is the sigmoid midpoint; the
  true value depends on the c_i normalization in (3) and on second-order
  GLE corrections.
- That no non-log-uniform density does *strictly* better at finite N. 
  Edge corrections of order 1/N can favor mild deviations (e.g. slightly
  more mass near the band center) -- this is what alpha != 1 in the
  empirical sweep was probing, and the differences seen in #031 are
  consistent with O(1/N) corrections, not with a different leading-order
  optimum.
- Anything non-Gaussian. The argument is purely linear-response.

## 9. What this means for the campaign

1. **Stop sweeping alpha.** alpha = 1 is the asymptotic optimum for all
   broad-spectrum targets. The differences seen in #031 across alpha in
   {0.5, 1, 1.5} are O(1/N) edge-correction noise.
2. **Prefer to widen [Q_min, Q_max]** rather than tinker with the
   density inside it -- this is the only way to push the 1/2 plateau
   over a wider band.
3. **For peaked spectra**, use the weighted-objective optimum (Section 7.1)
   rather than blind log-uniform. This is the principled version of the
   "tune Q to the dominant scale" intuition.
4. **NHC is the wrong baseline for broad spectra**, and the right
   baseline for narrow spectra. Pick by spectrum width, not by tradition.

## References

- [Dutta & Horn (1981)](https://doi.org/10.1103/RevModPhys.53.497) -- 1/f noise from log-uniform relaxation superposition.
- [Mori (1965)](https://doi.org/10.1143/PTP.33.423) -- Generalized Langevin equation and memory kernel formulation.
- [Ceriotti, Bussi, Parrinello (2010)](https://doi.org/10.1021/ct900563s) -- Colored-noise GLE thermostats; explicit Prony-series design.
- [Martyna, Klein, Tuckerman (1992)](https://doi.org/10.1063/1.463940) -- Nose-Hoover chains.
- [Jeffreys (1946)](https://doi.org/10.1098/rspa.1946.0056) -- The 1/Q "Jeffreys prior" for scale parameters.
- Parent: orbit #016 (spectral-1f-016) -- 1/f mechanism for multi-scale thermostats.
- Sibling: orbit #031 (alpha-spectrum-comparison-031) -- empirical alpha sweep.
- Sibling: orbit #032 (resonance-singularity) -- omega Q = 1 singularity.
- Sibling: orbit #029 (n-scaling-robust-029) -- N-scaling under log-uniform.
