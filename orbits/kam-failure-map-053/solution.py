"""KAM failure map: N=1 log-osc vs N=1 tanh on 1D harmonic oscillator V=1/2 omega^2 q^2.

Vectorized numpy sweep over (Q, omega, seed) grid — all runs propagate in lockstep.
"""
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

ORBIT = Path('/Users/wujiewang/code/det-sampler/.worktrees/kam-failure-map-053/orbits/kam-failure-map-053')
(ORBIT / 'figures').mkdir(exist_ok=True)

# --- friction functions ---
def g_logosc(xi):
    return 2.0 * xi / (1.0 + xi * xi)

def g_tanh(xi):
    return np.tanh(xi)

def run_grid(g_fn, Q_grid, W_grid, seeds, n_steps=1_000_000, dt=0.005, kT=1.0, burnin_frac=0.1):
    """Vectorized N=1 thermostat on 1D HO for a batch of (Q, omega, seed) cells.

    Arrays are broadcast: shape (nQ, nW, nS).
    """
    nQ, nW, nS = len(Q_grid), len(W_grid), len(seeds)
    Q = Q_grid[:, None, None] * np.ones((nQ, nW, nS))
    W = W_grid[None, :, None] * np.ones((nQ, nW, nS))
    W2 = W * W

    # per-cell RNG state: build with explicit seeds for reproducibility
    q = np.zeros((nQ, nW, nS))
    p = np.zeros((nQ, nW, nS))
    xi = np.zeros((nQ, nW, nS))
    for s_idx, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        q[:, :, s_idx] = rng.standard_normal((nQ, nW)) * np.sqrt(kT) / W_grid[None, :]
        p[:, :, s_idx] = rng.standard_normal((nQ, nW)) * np.sqrt(kT)
        xi[:, :, s_idx] = rng.standard_normal((nQ, nW)) * 0.1

    burnin = int(n_steps * burnin_frac)
    hdt = 0.5 * dt

    q_sq_sum = np.zeros_like(q)
    n_acc = 0

    for step in range(n_steps):
        # Half xi
        xi = xi + hdt * (p * p - kT) / Q
        # Friction half
        p = p * np.exp(-g_fn(xi) * hdt)
        # Force half
        p = p - hdt * W2 * q
        # Drift
        q = q + dt * p
        # Force half
        p = p - hdt * W2 * q
        # Friction half
        p = p * np.exp(-g_fn(xi) * hdt)
        # Half xi
        xi = xi + hdt * (p * p - kT) / Q

        if step >= burnin:
            q_sq_sum += q * q
            n_acc += 1

    var_q = q_sq_sum / n_acc
    # target variance = kT / omega^2
    var_ratio = var_q * W2 / kT
    return var_ratio  # (nQ, nW, nS)


def main():
    Q_grid = np.logspace(np.log10(0.1), np.log10(20.0), 20)
    W_grid = np.logspace(np.log10(0.1), np.log10(5.0), 20)
    seeds = [42, 123, 999]

    # Start coarser to validate, then full run
    n_steps = 1_000_000
    dt = 0.005

    print(f"Grid: {len(Q_grid)} Q x {len(W_grid)} omega x {len(seeds)} seeds = {len(Q_grid)*len(W_grid)*len(seeds)} runs")
    print(f"n_steps={n_steps}, dt={dt}")

    print("\n[1/2] N=1 log-osc sweep ...", flush=True)
    vr_losc = run_grid(g_logosc, Q_grid, W_grid, seeds, n_steps=n_steps, dt=dt)
    print(f"  done. mean var_ratio={vr_losc.mean():.3f}, min={vr_losc.min():.3f}, max={vr_losc.max():.3f}")

    print("\n[2/2] N=1 tanh sweep ...", flush=True)
    vr_tanh = run_grid(g_tanh, Q_grid, W_grid, seeds, n_steps=n_steps, dt=dt)
    print(f"  done. mean var_ratio={vr_tanh.mean():.3f}, min={vr_tanh.min():.3f}, max={vr_tanh.max():.3f}")

    # Mean over seeds
    mean_losc = vr_losc.mean(axis=2)
    mean_tanh = vr_tanh.mean(axis=2)
    std_losc = vr_losc.std(axis=2)
    std_tanh = vr_tanh.std(axis=2)

    # ergodic fraction (mean over seeds within 0.05 of 1)
    erg_losc_mask = np.abs(mean_losc - 1.0) < 0.05
    erg_tanh_mask = np.abs(mean_tanh - 1.0) < 0.05
    f_losc = erg_losc_mask.mean()
    f_tanh = erg_tanh_mask.mean()

    print(f"\nErgodic fraction (|var_ratio-1|<0.05):")
    print(f"  log-osc: {f_losc:.3f} ({erg_losc_mask.sum()}/{erg_losc_mask.size})")
    print(f"  tanh:    {f_tanh:.3f} ({erg_tanh_mask.sum()}/{erg_tanh_mask.size})")

    np.savez(ORBIT / 'results.npz',
             Q_grid=Q_grid, W_grid=W_grid, seeds=np.array(seeds),
             vr_losc=vr_losc, vr_tanh=vr_tanh,
             mean_losc=mean_losc, mean_tanh=mean_tanh,
             std_losc=std_losc, std_tanh=std_tanh,
             n_steps=n_steps, dt=dt)

    summary = {
        'Q_grid': Q_grid.tolist(),
        'W_grid': W_grid.tolist(),
        'seeds': list(seeds),
        'n_steps': n_steps,
        'dt': dt,
        'ergodic_fraction_logosc': float(f_losc),
        'ergodic_fraction_tanh': float(f_tanh),
        'ratio_tanh_over_logosc': float(f_tanh / max(f_losc, 1e-12)),
        'mean_var_ratio_logosc_overall': float(mean_losc.mean()),
        'mean_var_ratio_tanh_overall': float(mean_tanh.mean()),
    }
    with open(ORBIT / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    make_figure(Q_grid, W_grid, mean_losc, mean_tanh, f_losc, f_tanh)
    return f_losc, f_tanh


def make_figure(Q_grid, W_grid, mean_losc, mean_tanh, f_losc, f_tanh):
    mpl.rcParams.update({
        'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11,
        'figure.dpi': 150, 'savefig.dpi': 300,
    })

    # plot log10 |var_ratio - 1|
    Zl = np.log10(np.abs(mean_losc - 1.0) + 1e-6).T  # transpose so y=omega, x=Q
    Zt = np.log10(np.abs(mean_tanh - 1.0) + 1e-6).T

    vmin, vmax = -2.5, 1.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True, sharey=True)
    extent = [np.log10(Q_grid[0]), np.log10(Q_grid[-1]),
              np.log10(W_grid[0]), np.log10(W_grid[-1])]

    for ax, Z, title, wmax in [
        (axes[0], Zl, f'(a) N=1 log-osc  [erg frac = {f_losc:.2f}]', 0.732),
        (axes[1], Zt, f'(b) N=1 tanh  [erg frac = {f_tanh:.2f}]', 1.0),
    ]:
        im = ax.imshow(Z, origin='lower', extent=extent, aspect='auto',
                       cmap='RdBu_r', vmin=vmin, vmax=vmax)
        # Resonance curve: omega*Q = 1 -> log omega = -log Q
        qq = np.logspace(np.log10(Q_grid[0]), np.log10(Q_grid[-1]), 200)
        ax.plot(np.log10(qq), np.log10(1.0 / qq), 'k--', lw=2, label=r'$\omega Q=1$')
        # freq ceiling
        ax.axhline(np.log10(wmax), color='black', ls=':', lw=2,
                   label=rf'$\omega_{{\max}}={wmax}$')
        # Q > kT/2 = 0.5
        ax.axvline(np.log10(0.5), color='black', ls='-', lw=1.5, alpha=0.7,
                   label=r'$Q=k_B T/2$')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(r'$\log_{10} Q$')
        ax.legend(loc='upper right', frameon=True, fontsize=10)

    axes[0].set_ylabel(r'$\log_{10} \omega$')

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label(r'$\log_{10}\,|\mathrm{var\_ratio}-1|$')

    out = ORBIT / 'figures' / 'kam_surface.png'
    fig.savefig(out, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"Figure -> {out}")


if __name__ == '__main__':
    main()
