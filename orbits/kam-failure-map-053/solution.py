"""KAM failure map: N=1 log-osc vs N=1 tanh on 1D harmonic oscillator V=1/2 omega^2 q^2.

Vectorized numpy sweep over (Q, omega, seed) grid -- all runs propagate in lockstep.
Integrator: symmetric Trotter splitting (G-B-O-A-O-B-G) for Nose-Hoover (N=1).
"""
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

ORBIT = Path(__file__).resolve().parent
(ORBIT / 'figures').mkdir(exist_ok=True)

# --- friction functions ---
def g_logosc(xi):
    return 2.0 * xi / (1.0 + xi * xi)

def g_tanh(xi):
    return np.tanh(xi)

def run_grid(g_fn, Q_grid, W_grid, seeds, n_steps=1_000_000, dt=0.005, kT=1.0, burnin_frac=0.1):
    """Vectorized N=1 thermostat on 1D HO for a batch of (Q, omega, seed) cells."""
    nQ, nW, nS = len(Q_grid), len(W_grid), len(seeds)
    Q = Q_grid[:, None, None] * np.ones((nQ, nW, nS))
    W = W_grid[None, :, None] * np.ones((nQ, nW, nS))
    W2 = W * W

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
        xi = xi + hdt * (p * p - kT) / Q
        p = p * np.exp(-g_fn(xi) * hdt)
        p = p - hdt * W2 * q
        q = q + dt * p
        p = p - hdt * W2 * q
        p = p * np.exp(-g_fn(xi) * hdt)
        xi = xi + hdt * (p * p - kT) / Q

        if step >= burnin:
            q_sq_sum += q * q
            n_acc += 1

    var_q = q_sq_sum / n_acc
    var_ratio = var_q * W2 / kT
    return var_ratio  # (nQ, nW, nS)


def compute_metrics(vr, W_grid, thresh=0.05, majority=6):
    """Return dict of metrics for a (nQ, nW, nS) var_ratio array."""
    nS = vr.shape[2]
    # mean-first
    mean = vr.mean(axis=2)
    std = vr.std(axis=2)
    mask_mean = np.abs(mean - 1.0) < thresh
    f_mean = float(mask_mean.mean())
    # majority-vote: per-seed ergodic, then majority
    per_seed_erg = np.abs(vr - 1.0) < thresh   # (nQ, nW, nS)
    maj_mask = per_seed_erg.sum(axis=2) >= majority
    f_maj = float(maj_mask.mean())
    # per-seed erg fraction (fraction of cells ergodic for each seed)
    per_seed_frac = per_seed_erg.reshape(-1, nS).mean(axis=0).tolist()
    # restricted windows
    W = W_grid
    win1 = (W > 0.732) & (W <= 1.0)  # critical window
    win2 = W <= 2.0                   # discretization-safe
    def _restrict(mask, wsel):
        return float(mask[:, wsel].mean()) if wsel.any() else float('nan')
    return {
        'mean_first_frac': f_mean,
        'majority_vote_frac': f_maj,
        'per_seed_frac': per_seed_frac,
        'restricted_critical_mean': _restrict(mask_mean, win1),
        'restricted_critical_majority': _restrict(maj_mask, win1),
        'restricted_wle2_mean': _restrict(mask_mean, win2),
        'restricted_wle2_majority': _restrict(maj_mask, win2),
        'mean_map': mean,
        'std_map': std,
        'maj_map': maj_mask,
    }


def main():
    Q_grid = np.logspace(np.log10(0.1), np.log10(20.0), 20)
    W_grid = np.logspace(np.log10(0.1), np.log10(5.0), 20)
    seeds = [42, 123, 999, 777, 2024, 31415, 271828, 61803, 1414, 1618]

    n_steps = 1_000_000
    dt = 0.005

    print(f"Grid: {len(Q_grid)} Q x {len(W_grid)} omega x {len(seeds)} seeds = "
          f"{len(Q_grid)*len(W_grid)*len(seeds)} runs")
    print(f"n_steps={n_steps}, dt={dt}")

    print("\n[1/2] N=1 log-osc sweep ...", flush=True)
    vr_losc = run_grid(g_logosc, Q_grid, W_grid, seeds, n_steps=n_steps, dt=dt)
    print(f"  done. mean={vr_losc.mean():.3f}")

    print("\n[2/2] N=1 tanh sweep ...", flush=True)
    vr_tanh = run_grid(g_tanh, Q_grid, W_grid, seeds, n_steps=n_steps, dt=dt)
    print(f"  done. mean={vr_tanh.mean():.3f}")

    m_losc = compute_metrics(vr_losc, W_grid)
    m_tanh = compute_metrics(vr_tanh, W_grid)

    print("\nlog-osc  mean-first={:.4f}  majority={:.4f}".format(
        m_losc['mean_first_frac'], m_losc['majority_vote_frac']))
    print("tanh     mean-first={:.4f}  majority={:.4f}".format(
        m_tanh['mean_first_frac'], m_tanh['majority_vote_frac']))

    np.savez(ORBIT / 'results.npz',
             Q_grid=Q_grid, W_grid=W_grid, seeds=np.array(seeds),
             vr_losc=vr_losc, vr_tanh=vr_tanh,
             mean_losc=m_losc['mean_map'], mean_tanh=m_tanh['mean_map'],
             std_losc=m_losc['std_map'], std_tanh=m_tanh['std_map'],
             n_steps=n_steps, dt=dt)

    summary = {
        'Q_grid': Q_grid.tolist(),
        'W_grid': W_grid.tolist(),
        'seeds': list(seeds),
        'n_steps': n_steps,
        'dt': dt,
        'integrator': 'symmetric Trotter splitting (G-B-O-A-O-B-G) for Nose-Hoover N=1',
        'threshold': 0.05,
        'logosc': {
            'mean_first_frac': m_losc['mean_first_frac'],
            'majority_vote_frac': m_losc['majority_vote_frac'],
            'per_seed_frac': m_losc['per_seed_frac'],
            'restricted_critical_mean_first': m_losc['restricted_critical_mean'],
            'restricted_critical_majority': m_losc['restricted_critical_majority'],
            'restricted_wle2_mean_first': m_losc['restricted_wle2_mean'],
            'restricted_wle2_majority': m_losc['restricted_wle2_majority'],
        },
        'tanh': {
            'mean_first_frac': m_tanh['mean_first_frac'],
            'majority_vote_frac': m_tanh['majority_vote_frac'],
            'per_seed_frac': m_tanh['per_seed_frac'],
            'restricted_critical_mean_first': m_tanh['restricted_critical_mean'],
            'restricted_critical_majority': m_tanh['restricted_critical_majority'],
            'restricted_wle2_mean_first': m_tanh['restricted_wle2_mean'],
            'restricted_wle2_majority': m_tanh['restricted_wle2_majority'],
        },
        'metric_tanh_majority_vote': m_tanh['majority_vote_frac'],
    }
    with open(ORBIT / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    make_figure(Q_grid, W_grid, m_losc, m_tanh)
    return m_losc, m_tanh


def make_figure(Q_grid, W_grid, m_losc, m_tanh):
    mpl.rcParams.update({
        'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 10,
        'figure.dpi': 150, 'savefig.dpi': 300,
    })

    Zl = np.log10(np.abs(m_losc['mean_map'] - 1.0) + 1e-6).T
    Zt = np.log10(np.abs(m_tanh['mean_map'] - 1.0) + 1e-6).T
    Zs = np.log10(m_losc['std_map'] + 1e-6).T

    vmin, vmax = -2.5, 1.0
    extent = [np.log10(Q_grid[0]), np.log10(Q_grid[-1]),
              np.log10(W_grid[0]), np.log10(W_grid[-1])]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True, sharey=True)

    f_losc = m_losc['majority_vote_frac']
    f_tanh = m_tanh['majority_vote_frac']

    panels = [
        (axes[0], Zl, f'(a) N=1 log-osc  [maj={f_losc:.2f}]', 0.732, vmin, vmax, 'RdBu_r'),
        (axes[1], Zt, f'(b) N=1 tanh  [maj={f_tanh:.2f}]', 1.0, vmin, vmax, 'RdBu_r'),
    ]
    ims = []
    for ax, Z, title, wmax, vlo, vhi, cmap in panels:
        im = ax.imshow(Z, origin='lower', extent=extent, aspect='auto',
                       cmap=cmap, vmin=vlo, vmax=vhi)
        ims.append(im)
        qq = np.logspace(np.log10(Q_grid[0]), np.log10(Q_grid[-1]), 200)
        ax.plot(np.log10(qq), np.log10(1.0 / qq), 'k--', lw=2, label=r'$\omega Q=1$')
        ax.axhline(np.log10(wmax), color='black', ls=':', lw=2,
                   label=rf'$\omega_{{\max}}={wmax}$')
        ax.axvline(np.log10(0.5), color='black', ls='-', lw=1.2, alpha=0.6,
                   label=r'$Q=k_B T/2$')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(r'$\log_{10} Q$')
        ax.legend(loc='upper right', frameon=True, fontsize=9)

    # Panel (c): log10 std across seeds for log-osc
    ax = axes[2]
    im_c = ax.imshow(Zs, origin='lower', extent=extent, aspect='auto',
                     cmap='viridis', vmin=-2.5, vmax=0.5)
    qq = np.logspace(np.log10(Q_grid[0]), np.log10(Q_grid[-1]), 200)
    ax.plot(np.log10(qq), np.log10(1.0 / qq), 'w--', lw=2, label=r'$\omega Q=1$')
    ax.set_title('(c) log-osc seed std', fontweight='bold')
    ax.set_xlabel(r'$\log_{10} Q$')
    ax.legend(loc='upper right', frameon=True, fontsize=9)

    axes[0].set_ylabel(r'$\log_{10} \omega$')

    cbar_ab = fig.colorbar(ims[0], ax=axes[:2], shrink=0.85, pad=0.02, location='bottom')
    cbar_ab.set_label(r'$\log_{10}\,|\mathrm{var\_ratio}-1|$')
    cbar_c = fig.colorbar(im_c, ax=axes[2], shrink=0.85, pad=0.02, location='bottom')
    cbar_c.set_label(r'$\log_{10}\,\mathrm{std}_{\mathrm{seeds}}$')

    out = ORBIT / 'figures' / 'kam_surface.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"Figure -> {out}")


if __name__ == '__main__':
    main()
