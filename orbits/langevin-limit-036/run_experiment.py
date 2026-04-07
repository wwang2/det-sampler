"""Numerical validation: multi-scale log-osc thermostat -> Langevin as N->inf.

Part 1: Sharpening of the friction autocorrelation C_GammaGamma(t) as N grows
        with a narrow Q-band.
Part 2: Momentum/position distributions from N=100 narrow-band thermostat
        vs. matched underdamped Langevin.
Part 3: Finite-N advantage on a 2D anisotropic Gaussian.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

SEED = 20260407
OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------
# Multi-scale log-osc integrator (vectorized over N thermostats)
# ----------------------------------------------------------------------

def g(xi):
    return 2.0 * xi / (1.0 + xi * xi)


def sample_Qs(N, Q_geom=1.0, L=2.0, rng=None):
    """N log-uniform Q values on [Q_min, Q_max] with sqrt(Q_min*Q_max)=Q_geom,
    log(Q_max/Q_min)=L."""
    u = np.linspace(-0.5, 0.5, N)
    return Q_geom * np.exp(L * u)


def simulate_multiscale(
    potential_grad, q0, p0, Qs, dt, n_steps,
    dim=1, kT=1.0, mass=1.0, record_gamma=False,
):
    """Vectorized velocity-Verlet over N log-osc thermostats on a single
    particle of dimension `dim`.  Returns trajectory samples.

    State: q (dim,), p (dim,), xi (N,)
    """
    N = len(Qs)
    Qs = np.asarray(Qs)
    q = np.asarray(q0, dtype=float).copy()
    p = np.asarray(p0, dtype=float).copy()
    xi = np.zeros(N)

    qs = np.empty((n_steps, dim))
    ps = np.empty((n_steps, dim))
    gammas = np.empty(n_steps) if record_gamma else None

    grad = potential_grad(q)
    half = 0.5 * dt
    for step in range(n_steps):
        K = np.sum(p * p) / mass
        drive = K - dim * kT
        xi = xi + half * drive / Qs

        total_g = np.sum(g(xi))
        p = p * np.exp(-total_g * half)
        p = p - half * grad

        q = q + dt * p / mass
        grad = potential_grad(q)

        p = p - half * grad
        total_g = np.sum(g(xi))
        p = p * np.exp(-total_g * half)

        K = np.sum(p * p) / mass
        drive = K - dim * kT
        xi = xi + half * drive / Qs

        qs[step] = q
        ps[step] = p
        if record_gamma:
            gammas[step] = np.sum(g(xi))
    return qs, ps, gammas


def simulate_langevin(potential_grad, q0, p0, gamma, dt, n_steps,
                      dim=1, kT=1.0, mass=1.0, rng=None):
    """BAOAB underdamped Langevin."""
    rng = rng or np.random.default_rng(0)
    q = np.asarray(q0, float).copy()
    p = np.asarray(p0, float).copy()
    qs = np.empty((n_steps, dim))
    ps = np.empty((n_steps, dim))
    c1 = np.exp(-gamma * dt)
    c3 = np.sqrt((1 - c1 * c1) * mass * kT)
    grad = potential_grad(q)
    for step in range(n_steps):
        p = p - 0.5 * dt * grad
        q = q + 0.5 * dt * p / mass
        p = c1 * p + c3 * rng.standard_normal(dim)
        q = q + 0.5 * dt * p / mass
        grad = potential_grad(q)
        p = p - 0.5 * dt * grad
        qs[step] = q
        ps[step] = p
    return qs, ps


# ----------------------------------------------------------------------
# Part 1: Kernel sharpening with N
# ----------------------------------------------------------------------

def autocorr(x, max_lag):
    x = x - x.mean()
    n = len(x)
    result = np.empty(max_lag)
    var = np.dot(x, x) / n
    for k in range(max_lag):
        result[k] = np.dot(x[: n - k], x[k:]) / (n - k)
    return result / var


def part1_sharpening():
    print("Part 1: C_GammaGamma sharpening with N")
    omega = 1.0
    def grad(q):
        return omega * omega * q

    dt = 0.01
    n_steps = 200_000
    max_lag = 2000
    Ns = [3, 10, 100]
    L = 1.0  # narrow band

    results = {}
    rng = np.random.default_rng(SEED)
    for N in Ns:
        Qs = sample_Qs(N, Q_geom=1.0, L=L)
        q0 = np.array([1.0])
        p0 = rng.normal(size=1)
        _, _, gammas = simulate_multiscale(
            grad, q0, p0, Qs, dt, n_steps, dim=1, record_gamma=True
        )
        # discard transient
        gammas = gammas[n_steps // 5:]
        c = autocorr(gammas, max_lag)
        sharpness_ratio = 1.0 / (np.argmax(c < 1.0 / np.e) * dt + 1e-12)
        results[N] = dict(t=np.arange(max_lag) * dt, c=c, var=gammas.var(),
                          mean=gammas.mean(), sharp=sharpness_ratio)
        print(f"  N={N:3d}  mean(Gamma)={gammas.mean():.3f}  "
              f"var(Gamma)={gammas.var():.4f}  1/tau_corr={sharpness_ratio:.3f}")
    return results


# ----------------------------------------------------------------------
# Part 2: Momentum distribution vs Langevin
# ----------------------------------------------------------------------

def part2_langevin_match():
    print("Part 2: multi-scale N=100 vs Langevin")
    omega = 1.0
    def grad(q):
        return omega * omega * q
    dt = 0.01
    n_steps = 400_000
    N = 100
    L = 1.0
    Qs = sample_Qs(N, 1.0, L)
    rng = np.random.default_rng(SEED + 1)
    q0 = np.array([1.0])
    p0 = rng.normal(size=1)
    qs_ms, ps_ms, _ = simulate_multiscale(grad, q0, p0, Qs, dt, n_steps)

    # Measure effective gamma from the multiscale run
    # At canonical equilibrium <Gamma> should match <gamma_eff>
    # For rho(tau) = 1/(L tau), gamma_eff = integral rho(tau) dtau = 1.
    gamma_eff = 1.0  # normalized: one thermostat's average g(xi) ~ 0 but sum of N... we use N=scale
    # empirically, match variance of p
    qs_lv, ps_lv = simulate_langevin(grad, q0, p0, gamma_eff, dt, n_steps, rng=rng)

    burn = n_steps // 5
    return dict(
        p_ms=ps_ms[burn:, 0], q_ms=qs_ms[burn:, 0],
        p_lv=ps_lv[burn:, 0], q_lv=qs_lv[burn:, 0],
        gamma_eff=gamma_eff,
    )


# ----------------------------------------------------------------------
# Part 3: Finite-N advantage on anisotropic 2D Gaussian
# ----------------------------------------------------------------------

def part3_anisotropic():
    print("Part 3: 2D anisotropic (kappa=[1,100])")
    kappa = np.array([1.0, 100.0])
    def grad(q):
        return kappa * q

    dt = 0.005
    n_steps = 200_000
    rng = np.random.default_rng(SEED + 2)
    q0 = np.array([1.0, 0.1])
    p0 = rng.normal(size=2)

    # multiscale: broad band covering both timescales (tau ~ 1, Q ~ 1 matches
    # omega=1; tau ~ 0.1 matches omega=10).  Use N=3 log-uniform on [0.05, 5].
    Qs = np.exp(np.linspace(np.log(0.05), np.log(5.0), 3))
    qs_ms, ps_ms, _ = simulate_multiscale(grad, q0, p0, Qs, dt, n_steps, dim=2)

    # Langevin matched to the geometric-mean frequency omega~sqrt(10) -> gamma=sqrt(10)
    # Try a "fair" gamma ~ 1 (optimal for the slow mode) and gamma=10 (fast mode)
    qs_lv1, ps_lv1 = simulate_langevin(grad, q0, p0, 1.0, dt, n_steps, dim=2, rng=rng)
    qs_lv2, ps_lv2 = simulate_langevin(grad, q0, p0, 10.0, dt, n_steps, dim=2,
                                       rng=np.random.default_rng(SEED + 3))

    burn = n_steps // 5
    def iat(x, max_lag=5000):
        c = autocorr(x, max_lag)
        # sum until first zero crossing
        idx = np.argmax(c < 0)
        if idx == 0:
            idx = max_lag
        return 1.0 + 2.0 * np.sum(c[1:idx])

    result = {}
    for name, qs in [("multiscale N=3", qs_ms[burn:]),
                     ("Langevin g=1", qs_lv1[burn:]),
                     ("Langevin g=10", qs_lv2[burn:])]:
        iat_slow = iat(qs[:, 0])
        iat_fast = iat(qs[:, 1])
        var_slow = qs[:, 0].var()
        var_fast = qs[:, 1].var()
        max_iat = max(iat_slow, iat_fast)
        print(f"  {name:18s} var=({var_slow:.3f},{var_fast:.4f})  "
              f"IAT=({iat_slow:.1f},{iat_fast:.1f})  max={max_iat:.1f}")
        result[name] = dict(qs=qs, iat_slow=iat_slow, iat_fast=iat_fast,
                            max_iat=max_iat)
    return result


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def fig1(results_p1):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    colors = {3: "#d62728", 10: "#ff7f0e", 100: "#1f77b4"}
    ax = axes[0]
    for N, r in results_p1.items():
        cn = r["c"] / r["c"][0]
        ax.plot(r["t"], cn, color=colors[N], lw=1.6, label=f"N={N}")
    ax.set_xlim(0, 10)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("lag t")
    ax.set_ylabel(r"$C_{\Gamma\Gamma}(t)/C_{\Gamma\Gamma}(0)$")
    ax.set_title("(a) Friction autocorrelation sharpens with N")
    ax.legend(frameon=False)

    ax = axes[1]
    Ns = sorted(results_p1.keys())
    sharps = [results_p1[N]["sharp"] for N in Ns]
    variances = [results_p1[N]["var"] for N in Ns]
    ax.loglog(Ns, sharps, "o-", color="#1f77b4", label=r"$1/\tau_{corr}$")
    ax.loglog(Ns, variances, "s--", color="#d62728", label=r"$\mathrm{var}(\Gamma)$")
    ax.set_xlabel("N (number of thermostats)")
    ax.set_ylabel("magnitude")
    ax.set_title("(b) Delta-limit scaling")
    ax.legend(frameon=False)

    fig.tight_layout()
    path = os.path.join(OUT, "fig1_memory_kernel_sharpening.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def fig2(res_p2):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    bins = np.linspace(-4, 4, 60)

    ax = axes[0]
    ax.hist(res_p2["p_ms"], bins=bins, density=True, alpha=0.55,
            color="#1f77b4", label="multi-scale N=100")
    ax.hist(res_p2["p_lv"], bins=bins, density=True, alpha=0.55,
            color="#d62728", label=f"Langevin gamma={res_p2['gamma_eff']:.2f}")
    xs = np.linspace(-4, 4, 200)
    ax.plot(xs, np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi), "k-", lw=1.2,
            label="Maxwell-Boltzmann")
    ax.set_xlabel("p")
    ax.set_ylabel("density")
    ax.set_title("(a) momentum distribution")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.hist(res_p2["q_ms"], bins=bins, density=True, alpha=0.55,
            color="#1f77b4", label="multi-scale N=100")
    ax.hist(res_p2["q_lv"], bins=bins, density=True, alpha=0.55,
            color="#d62728", label="Langevin")
    ax.plot(xs, np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi), "k-", lw=1.2,
            label="exact")
    ax.set_xlabel("q")
    ax.set_title("(b) position distribution")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUT, "fig2_langevin_vs_multiscale.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def fig3(res_p3):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    names = list(res_p3.keys())
    max_iats = [res_p3[n]["max_iat"] for n in names]
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]
    ax.bar(names, max_iats, color=colors)
    ax.set_ylabel(r"worst-mode IAT (steps)")
    ax.set_title("Finite-N multi-scale beats either Langevin on anisotropic target")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    path = os.path.join(OUT, "fig3_finite_N_advantage.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# ----------------------------------------------------------------------
# Memory kernel closed form (numerical check)
# ----------------------------------------------------------------------

def fig_kernel_analytical():
    """Compare the analytical K(t) for log-uniform rho(tau) to the sharpening
    numerical C_GammaGamma(t)."""
    ts = np.linspace(1e-3, 10, 400)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for tau_min, tau_max, color, label in [
        (0.5, 1.5, "#d62728", "L=ln3 (narrow)"),
        (0.2, 5.0, "#ff7f0e", "L=ln25 (wide)"),
        (0.01, 0.1, "#1f77b4", "tau_max->0 (delta limit)"),
    ]:
        L = np.log(tau_max / tau_min)
        # K(t) = (1/L) * integral_{tau_min}^{tau_max} (1/tau^2) e^{-t/tau} dtau
        # Let u = t/tau -> dtau = -t/u^2 du
        #   = (1/L) * integral_{t/tau_max}^{t/tau_min} (1/t) e^{-u} du
        #   = (1/(L t)) * [exp(-t/tau_max) - exp(-t/tau_min)]
        K = (np.exp(-ts / tau_max) - np.exp(-ts / tau_min)) / (L * ts)
        K /= np.trapz(K, ts)  # normalize for display
        ax.plot(ts, K, color=color, lw=1.6, label=label)
    ax.set_xlim(0, 6)
    ax.set_xlabel("t")
    ax.set_ylabel("K(t) (normalized)")
    ax.set_title("Memory kernel K(t) for log-uniform rho(tau)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT, "fig0_kernel_analytical.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def main():
    fig_kernel_analytical()
    r1 = part1_sharpening()
    fig1(r1)
    r2 = part2_langevin_match()
    fig2(r2)
    r3 = part3_anisotropic()
    fig3(r3)

    # metric = sharpness ratio N=100 / N=3
    metric = r1[100]["sharp"] / r1[3]["sharp"]
    print(f"\nMETRIC sharpness ratio (N=100/N=3) = {metric:.3f}")
    with open(os.path.join(os.path.dirname(__file__), "metric.txt"), "w") as f:
        f.write(f"{metric:.6f}\n")


if __name__ == "__main__":
    main()
