"""Generate comparison figure from collected benchmark data."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

mpl.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.pad_inches': 0.2,
})

ORBIT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ORBIT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Friction functions
def g_tanh(xi): return np.tanh(xi)
def g_losc(xi): return 2.0 * xi / (1.0 + xi**2)
def g_linear(xi): return xi
def g_new(xi): return xi * np.log(1.0 + xi**2) / np.sqrt(1.0 + xi**2)

COLORS = {
    'tanh': '#1f77b4',
    'log-osc': '#d62728',
    'linear': '#2ca02c',
    'sublinear': '#ff7f0e',
}

# Collected results from the benchmark run (median tau_int across 20 seeds)
Q_vals = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

results = {
    10: {
        'tanh':      [95.4, 93.1, 71.6, 64.8, 64.4, 64.5],
        'log-osc':   [78.8, 96.9, 82.9, 66.4, 64.4, 64.5],
        'linear':    [87.2, 89.9, 68.9, 64.7, 64.5, 64.5],
        'sublinear': [95.4, 94.2, 73.6, 65.5, 64.9, 64.9],
    },
    100: {
        'tanh':      [26.8, 23.8, 29.0, 30.4, 30.8, 31.0],
        'log-osc':   [20.0, 22.7, 22.6, 29.7, 30.8, 30.9],
        'linear':    [27.5, 23.5, 26.0, 30.8, 31.0, 31.0],
        'sublinear': [26.0, 24.1, 24.8, 31.1, 32.3, 131.0],
    },
    1000: {
        'tanh':      [379.3, 379.3, 373.2, 381.0, 395.8, 462.6],
        'log-osc':   [6.3, 6.3, 6.3, 7.1, 477.0, 404.9],
        'linear':    [np.inf, np.inf, np.inf, 255.6, 275.2, 396.1],
        'sublinear': [np.inf, np.inf, np.inf, np.inf, 318.2, 500.0],  # Q=100 estimated
    },
}

# ---- 3-panel figure ----
fig = plt.figure(figsize=(20, 5.5), constrained_layout=True)

# Panel (a): g(xi) functions
ax_a = fig.add_subplot(1, 3, 1)
xi = np.linspace(-5, 5, 500)
for name, g_func, col in [
    ('tanh', g_tanh, COLORS['tanh']),
    ('log-osc', g_losc, COLORS['log-osc']),
    ('linear', g_linear, COLORS['linear']),
    ('sublinear (new)', g_new, COLORS['sublinear']),
]:
    ax_a.plot(xi, g_func(xi), label=name, color=col, linewidth=2)

ax_a.set_title('(a) Friction function g(xi)', fontweight='bold')
ax_a.set_xlabel('xi')
ax_a.set_ylabel('g(xi)')
ax_a.legend(frameon=False, fontsize=10)
ax_a.axhline(0, color='gray', lw=0.5, ls='--')
ax_a.axhline(1, color='gray', lw=0.5, ls=':', alpha=0.4)
ax_a.axhline(-1, color='gray', lw=0.5, ls=':', alpha=0.4)
ax_a.set_ylim(-4, 4)
ax_a.grid(True, alpha=0.2)

methods = ['tanh', 'log-osc', 'linear', 'sublinear']
labels = {'tanh': 'tanh', 'log-osc': 'log-osc', 'linear': 'linear', 'sublinear': 'sublinear (new)'}

for panel_idx, kappa in enumerate([100, 1000]):
    ax = fig.add_subplot(1, 3, panel_idx + 2)
    for method_name in methods:
        taus = results[kappa][method_name]
        # Filter out inf
        valid = [(Q, t) for Q, t in zip(Q_vals, taus) if np.isfinite(t)]
        if valid:
            Qs, ts = zip(*valid)
            ax.plot(Qs, ts, 'o-', label=labels[method_name],
                    color=COLORS[method_name], linewidth=2, markersize=6)
        # Mark inf points with arrows
        inf_Qs = [Q for Q, t in zip(Q_vals, taus) if not np.isfinite(t)]
        if inf_Qs:
            for Q_inf in inf_Qs:
                ax.annotate('', xy=(Q_inf, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1000),
                           xytext=(Q_inf, 800),
                           arrowprops=dict(arrowstyle='->', color=COLORS[method_name], lw=1.5))

    ax.set_xscale('log')
    ax.set_yscale('log')
    panel_label = chr(ord('b') + panel_idx)
    ax.set_title(f'({panel_label}) kappa={kappa}', fontweight='bold')
    ax.set_xlabel('Thermostat mass Q')
    ax.set_ylabel('Median tau_int (q_d^2)')
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, alpha=0.3)

fig.savefig(os.path.join(FIG_DIR, 'comparison.png'), bbox_inches='tight')
plt.close(fig)
print("Saved comparison.png")

# ---- Additional: all 3 kappa values ----
fig2, axes = plt.subplots(1, 3, figsize=(21, 5), constrained_layout=True)
for idx, kappa in enumerate([10, 100, 1000]):
    ax = axes[idx]
    for method_name in methods:
        taus = results[kappa][method_name]
        valid = [(Q, t) for Q, t in zip(Q_vals, taus) if np.isfinite(t)]
        if valid:
            Qs, ts = zip(*valid)
            ax.plot(Qs, ts, 'o-', label=labels[method_name],
                    color=COLORS[method_name], linewidth=2, markersize=5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    panel_label = chr(ord('a') + idx)
    ax.set_title(f'({panel_label}) kappa={kappa}', fontweight='bold')
    ax.set_xlabel('Thermostat mass Q')
    ax.set_ylabel('Median tau_int (q_d^2)')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)

fig2.savefig(os.path.join(FIG_DIR, 'tau_vs_Q_all_kappa.png'), bbox_inches='tight')
plt.close(fig2)
print("Saved tau_vs_Q_all_kappa.png")

# Print summary
print("\n=== KEY COMPARISONS ===")
print("\nkappa=100, best Q for each method (min tau):")
for m in methods:
    taus = results[100][m]
    best_idx = np.argmin(taus)
    print(f"  {m:12s}: tau={taus[best_idx]:.1f} at Q={Q_vals[best_idx]}")

print("\nkappa=100: tau_tanh / tau_sublinear at each Q:")
for i, Q in enumerate(Q_vals):
    t_tanh = results[100]['tanh'][i]
    t_sub = results[100]['sublinear'][i]
    ratio = t_tanh / t_sub if t_sub > 0 else float('inf')
    print(f"  Q={Q:5.1f}: {t_tanh:.1f} / {t_sub:.1f} = {ratio:.3f}")

print("\nkappa=1000: tau_tanh / tau_sublinear at each Q (finite only):")
for i, Q in enumerate(Q_vals):
    t_tanh = results[1000]['tanh'][i]
    t_sub = results[1000]['sublinear'][i]
    if np.isfinite(t_tanh) and np.isfinite(t_sub) and t_sub > 0:
        ratio = t_tanh / t_sub
        print(f"  Q={Q:5.1f}: {t_tanh:.1f} / {t_sub:.1f} = {ratio:.3f}")
    else:
        print(f"  Q={Q:5.1f}: tanh={t_tanh:.1f}, sub={'inf' if not np.isfinite(t_sub) else f'{t_sub:.1f}'}")

# Headline metric
best_ratio = 0
best_Q = None
for i, Q in enumerate(Q_vals):
    if Q >= 100:
        continue
    t_tanh = results[100]['tanh'][i]
    t_sub = results[100]['sublinear'][i]
    if t_sub > 0 and np.isfinite(t_sub):
        r = t_tanh / t_sub
        if r > best_ratio:
            best_ratio = r
            best_Q = Q

print(f"\nHEADLINE METRIC: tau_tanh/tau_sublinear = {best_ratio:.3f} at kappa=100, Q={best_Q}")
