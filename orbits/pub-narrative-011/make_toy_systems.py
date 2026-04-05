#!/usr/bin/env python3
"""Generate toy system illustration figures for the paper.

These show actual dynamics (simulated with simple integrators) on
toy systems, illustrating the key phenomena.

Generates:
  - fig_toy_doublewell.png   (1D double-well with trajectory)
  - fig_toy_harmonic.png     (HO phase portrait: NH vs Log-Osc)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- Style constants (from research/style.md) ---
COLOR_NH = "#1f77b4"
COLOR_NHC = "#ff7f0e"
COLOR_LOGOSC = "#2ca02c"
COLOR_GRAY = "#888888"

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)

DPI = 300
FONT_LABEL = 14
FONT_TICK = 12
FONT_TITLE = 16
FONT_ANNOT = 11

SEED = 42


# =============================================================================
# Helpers: simple thermostat integrators for toy illustrations
# =============================================================================

def g_logosc(xi):
    """Bounded friction: g(xi) = 2*xi/(1+xi^2)."""
    return 2.0 * xi / (1.0 + xi**2)


def simulate_nh_1d_ho(Q=1.0, dt=0.005, n_steps=200000, kT=1.0, omega=1.0, seed=SEED):
    """Simulate 1D harmonic oscillator with Nose-Hoover thermostat."""
    rng = np.random.default_rng(seed)
    q = rng.normal(0, np.sqrt(kT / omega**2))
    p = rng.normal(0, np.sqrt(kT))
    xi = 0.0

    qs, ps = [q], [p]
    for _ in range(n_steps):
        # Velocity Verlet with NH
        xi += 0.5 * dt * (p**2 - kT) / Q
        p *= np.exp(-xi * 0.5 * dt)
        p -= 0.5 * dt * omega**2 * q
        q += dt * p
        grad_U = omega**2 * q
        p -= 0.5 * dt * grad_U
        p *= np.exp(-xi * 0.5 * dt)
        xi += 0.5 * dt * (p**2 - kT) / Q

        qs.append(q)
        ps.append(p)

    return np.array(qs), np.array(ps)


def simulate_logosc_1d_ho(Q=0.5, dt=0.005, n_steps=200000, kT=1.0, omega=1.0, seed=SEED):
    """Simulate 1D harmonic oscillator with Log-Osc thermostat."""
    rng = np.random.default_rng(seed)
    q = rng.normal(0, np.sqrt(kT / omega**2))
    p = rng.normal(0, np.sqrt(kT))
    xi = 0.0

    qs, ps = [q], [p]
    for _ in range(n_steps):
        xi += 0.5 * dt * (p**2 - kT) / Q
        gxi = g_logosc(xi)
        p *= np.exp(-gxi * 0.5 * dt)
        p -= 0.5 * dt * omega**2 * q
        q += dt * p
        grad_U = omega**2 * q
        p -= 0.5 * dt * grad_U
        gxi = g_logosc(xi)
        p *= np.exp(-gxi * 0.5 * dt)
        xi += 0.5 * dt * (p**2 - kT) / Q

        qs.append(q)
        ps.append(p)

    return np.array(qs), np.array(ps)


def simulate_dw_trajectory(Q=1.0, dt=0.01, n_steps=500000, kT=1.0, seed=SEED):
    """Simulate 1D double-well U(x) = (x^2-1)^2 with Log-Osc thermostat."""
    rng = np.random.default_rng(seed)
    q = rng.choice([-1.0, 1.0])
    p = rng.normal(0, np.sqrt(kT))
    xi = 0.0

    qs = [q]
    for _ in range(n_steps):
        grad_U = 4 * q * (q**2 - 1)

        xi += 0.5 * dt * (p**2 - kT) / Q
        gxi = g_logosc(xi)
        p *= np.exp(-gxi * 0.5 * dt)
        p -= 0.5 * dt * grad_U
        q += dt * p
        grad_U = 4 * q * (q**2 - 1)
        p -= 0.5 * dt * grad_U
        gxi = g_logosc(xi)
        p *= np.exp(-gxi * 0.5 * dt)
        xi += 0.5 * dt * (p**2 - kT) / Q

        qs.append(q)

    return np.array(qs)


# =============================================================================
# Toy 1: 1D Double-Well with Trajectory
# =============================================================================
def make_doublewell_figure():
    print("  Simulating double-well trajectory...")
    qs = simulate_dw_trajectory(Q=0.5, dt=0.01, n_steps=300000)

    fig, (ax_pot, ax_ts) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[1, 1],
                                         gridspec_kw={"hspace": 0.35})
    fig.suptitle("1D Double-Well: Barrier Crossing Dynamics",
                 fontsize=FONT_TITLE, fontweight="bold", y=0.98)

    # --- Top: Potential with trajectory overlay ---
    x = np.linspace(-2, 2, 500)
    U = (x**2 - 1)**2
    ax_pot.plot(x, U, color="black", lw=3, zorder=5)
    ax_pot.fill_between(x, 0, U, alpha=0.08, color="black")

    # Mark wells and barrier
    ax_pot.plot([-1, 1], [0, 0], "o", color=COLOR_LOGOSC, markersize=10, zorder=6)
    ax_pot.plot([0], [1], "^", color="red", markersize=12, zorder=6)

    # Annotations
    ax_pot.annotate("well\n$q=-1$", xy=(-1, 0), xytext=(-1.6, 0.7),
                    fontsize=FONT_ANNOT, ha="center", fontweight="bold", color=COLOR_LOGOSC,
                    arrowprops=dict(arrowstyle="->", color=COLOR_LOGOSC, lw=1.5))
    ax_pot.annotate("well\n$q=+1$", xy=(1, 0), xytext=(1.6, 0.7),
                    fontsize=FONT_ANNOT, ha="center", fontweight="bold", color=COLOR_LOGOSC,
                    arrowprops=dict(arrowstyle="->", color=COLOR_LOGOSC, lw=1.5))
    ax_pot.annotate("barrier\n$\\Delta U = 1$", xy=(0, 1), xytext=(0.6, 1.6),
                    fontsize=FONT_ANNOT, ha="center", fontweight="bold", color="red",
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

    # Show a schematic trajectory hopping over the barrier
    # Use a subset of the simulated trajectory that shows crossings
    # Find crossing events
    subsample = qs[::100]  # subsample for plotting on potential
    # Plot trajectory as scatter on the potential surface
    U_traj = (subsample**2 - 1)**2
    # Only plot a segment that shows a crossing
    # Find first crossing
    crossings = np.where(np.diff(np.sign(subsample)))[0]
    if len(crossings) > 2:
        start = max(0, crossings[0] - 50)
        end = min(len(subsample), crossings[2] + 50)
        seg = subsample[start:end]
        U_seg = (seg**2 - 1)**2
        ax_pot.plot(seg, U_seg + 0.05, color=COLOR_LOGOSC, lw=1.5, alpha=0.7, zorder=4)
        # Arrow at crossing point
        cx_idx = crossings[0] - start
        if 0 < cx_idx < len(seg) - 1:
            ax_pot.annotate("", xy=(seg[cx_idx+1], U_seg[cx_idx+1] + 0.05),
                           xytext=(seg[cx_idx-1], U_seg[cx_idx-1] + 0.05),
                           arrowprops=dict(arrowstyle="-|>", color=COLOR_LOGOSC, lw=2))

    ax_pot.set_xlabel("Position $q$", fontsize=FONT_LABEL)
    ax_pot.set_ylabel("$U(q) = (q^2-1)^2$", fontsize=FONT_LABEL)
    ax_pot.set_ylim(-0.1, 2.0)
    ax_pot.tick_params(labelsize=FONT_TICK)

    # --- Bottom: Time series ---
    # Show a window with clear crossings
    t_steps = np.arange(len(qs)) * 0.01
    # Find a good window with multiple crossings
    window_size = 50000
    best_start = 0
    best_crossings = 0
    for s in range(0, len(qs) - window_size, window_size // 4):
        seg = qs[s:s+window_size]
        nc = len(np.where(np.diff(np.sign(seg)))[0])
        if nc > best_crossings:
            best_crossings = nc
            best_start = s

    s, e = best_start, best_start + window_size
    t_win = t_steps[s:e]
    q_win = qs[s:e]

    ax_ts.plot(t_win, q_win, color=COLOR_LOGOSC, lw=0.5, alpha=0.8)
    ax_ts.axhline(0, color=COLOR_GRAY, ls="--", lw=1, alpha=0.4)
    ax_ts.axhline(1, color=COLOR_GRAY, ls=":", lw=0.8, alpha=0.3)
    ax_ts.axhline(-1, color=COLOR_GRAY, ls=":", lw=0.8, alpha=0.3)

    # Mark crossings
    cross_idx = np.where(np.diff(np.sign(q_win)))[0]
    n_shown = min(len(cross_idx), 15)
    for ci in cross_idx[:n_shown]:
        ax_ts.axvline(t_win[ci], color="red", lw=0.5, alpha=0.3)

    # Annotate residence times
    ax_ts.fill_between(t_win, -2, 0, alpha=0.04, color="blue")
    ax_ts.fill_between(t_win, 0, 2, alpha=0.04, color="orange")
    ax_ts.text(t_win[len(t_win)//4], 1.5, "right well ($q > 0$)",
               fontsize=9, ha="center", color="darkorange", fontstyle="italic")
    ax_ts.text(t_win[len(t_win)//4], -1.5, "left well ($q < 0$)",
               fontsize=9, ha="center", color="blue", fontstyle="italic")

    n_cross = len(cross_idx)
    ax_ts.text(t_win[-1], 1.8, f"{n_cross} crossings\nin window",
               fontsize=10, ha="right", va="top", fontweight="bold", color="red",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

    ax_ts.set_xlabel("Time", fontsize=FONT_LABEL)
    ax_ts.set_ylabel("Position $q(t)$", fontsize=FONT_LABEL)
    ax_ts.set_ylim(-2.2, 2.2)
    ax_ts.tick_params(labelsize=FONT_TICK)

    plt.savefig(os.path.join(FIGDIR, "fig_toy_doublewell.png"), dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  -> fig_toy_doublewell.png")


# =============================================================================
# Toy 2: Harmonic Oscillator Phase Portrait
# =============================================================================
def make_harmonic_figure():
    print("  Simulating NH trajectory...")
    qs_nh, ps_nh = simulate_nh_1d_ho(Q=1.0, dt=0.005, n_steps=200000)
    print("  Simulating Log-Osc trajectory...")
    qs_lo, ps_lo = simulate_logosc_1d_ho(Q=0.5, dt=0.005, n_steps=200000)

    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.5], hspace=0.35, wspace=0.35,
                          left=0.07, right=0.97, top=0.88, bottom=0.08)
    fig.suptitle("1D Harmonic Oscillator: Phase Space and Marginals",
                 fontsize=FONT_TITLE, fontweight="bold", y=0.97)

    # --- NH phase portrait ---
    ax_nh = fig.add_subplot(gs[:, 0])
    ax_nh.set_title("Nose-Hoover ($Q=1$)", fontsize=FONT_LABEL, fontweight="bold", color=COLOR_NH)

    # Plot trajectory (subsample for speed)
    skip = 5
    ax_nh.plot(qs_nh[::skip], ps_nh[::skip], ",", color=COLOR_NH, alpha=0.15, markersize=0.5)

    # Target Gaussian contours
    theta = np.linspace(0, 2*np.pi, 200)
    for sigma in [1, 2, 3]:
        ax_nh.plot(sigma * np.cos(theta), sigma * np.sin(theta),
                   color=COLOR_GRAY, ls="--", lw=1, alpha=0.5)

    ax_nh.set_xlabel("$q$", fontsize=FONT_LABEL)
    ax_nh.set_ylabel("$p$", fontsize=FONT_LABEL)
    ax_nh.set_xlim(-4, 4)
    ax_nh.set_ylim(-4, 4)
    ax_nh.set_aspect("equal")
    ax_nh.tick_params(labelsize=FONT_TICK)
    ax_nh.text(0, -3.5, "KAM torus: trajectory\nconfined to thin ring",
               ha="center", fontsize=9, color="red", fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.9))

    # --- Log-Osc phase portrait ---
    ax_lo = fig.add_subplot(gs[:, 1])
    ax_lo.set_title("Log-Osc ($Q=0.5$)", fontsize=FONT_LABEL, fontweight="bold", color=COLOR_LOGOSC)

    ax_lo.plot(qs_lo[::skip], ps_lo[::skip], ",", color=COLOR_LOGOSC, alpha=0.15, markersize=0.5)

    for sigma in [1, 2, 3]:
        ax_lo.plot(sigma * np.cos(theta), sigma * np.sin(theta),
                   color=COLOR_GRAY, ls="--", lw=1, alpha=0.5)

    ax_lo.set_xlabel("$q$", fontsize=FONT_LABEL)
    ax_lo.set_ylabel("$p$", fontsize=FONT_LABEL)
    ax_lo.set_xlim(-4, 4)
    ax_lo.set_ylim(-4, 4)
    ax_lo.set_aspect("equal")
    ax_lo.tick_params(labelsize=FONT_TICK)
    ax_lo.text(0, -3.5, "Space-filling: trajectory\ncovers full phase space",
               ha="center", fontsize=9, color=COLOR_LOGOSC, fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.9))

    # --- Marginal comparison (right panels) ---
    ax_marg_q = fig.add_subplot(gs[0, 2])
    ax_marg_p = fig.add_subplot(gs[1, 2])

    q_range = np.linspace(-4, 4, 200)
    target_q = norm.pdf(q_range, 0, 1)  # sigma_q = sqrt(kT/omega^2) = 1

    # q marginal
    ax_marg_q.hist(qs_nh[10000:], bins=80, density=True, alpha=0.4, color=COLOR_NH, label="NH")
    ax_marg_q.hist(qs_lo[10000:], bins=80, density=True, alpha=0.4, color=COLOR_LOGOSC, label="Log-Osc")
    ax_marg_q.plot(q_range, target_q, "k--", lw=2, label="Target")
    ax_marg_q.set_xlabel("$q$", fontsize=FONT_ANNOT)
    ax_marg_q.set_ylabel("$P(q)$", fontsize=FONT_ANNOT)
    ax_marg_q.set_title("Position marginal", fontsize=FONT_ANNOT, fontweight="bold")
    ax_marg_q.legend(fontsize=8)
    ax_marg_q.tick_params(labelsize=9)

    # p marginal
    ax_marg_p.hist(ps_nh[10000:], bins=80, density=True, alpha=0.4, color=COLOR_NH, label="NH")
    ax_marg_p.hist(ps_lo[10000:], bins=80, density=True, alpha=0.4, color=COLOR_LOGOSC, label="Log-Osc")
    ax_marg_p.plot(q_range, target_q, "k--", lw=2, label="Target")
    ax_marg_p.set_xlabel("$p$", fontsize=FONT_ANNOT)
    ax_marg_p.set_ylabel("$P(p)$", fontsize=FONT_ANNOT)
    ax_marg_p.set_title("Momentum marginal", fontsize=FONT_ANNOT, fontweight="bold")
    ax_marg_p.legend(fontsize=8)
    ax_marg_p.tick_params(labelsize=9)

    plt.savefig(os.path.join(FIGDIR, "fig_toy_harmonic.png"), dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  -> fig_toy_harmonic.png")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("Generating toy system illustrations...")
    make_doublewell_figure()
    make_harmonic_figure()
    print("Done. All toy figures saved to:", FIGDIR)
