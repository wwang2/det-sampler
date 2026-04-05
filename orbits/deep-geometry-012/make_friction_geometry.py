"""Friction Function Geometry from the Master Theorem catalog.

3-row figure:
  Row 1: V(xi) thermostat potentials (quadratic, log, cosh, arctan-derived)
  Row 2: g(xi) = V'(xi)/Q friction functions
  Row 3: p(xi) marginal distributions ~ exp(-V(xi)/kT)

Shows how different confining potentials yield different friction behaviors.
"""

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- Style ----
LABEL_SIZE = 14; TICK_SIZE = 12; TITLE_SIZE = 15
# Colors: NH=blue, NHC=orange, novel=tab10[2:]
colors = {
    'NH (quadratic)': '#1f77b4',
    'Log-Osc': '#2ca02c',
    'Cosh': '#d62728',
    'Double-well': '#9467bd',
}

kT = 1.0
Q = 1.0  # thermostat mass

# ---- Thermostat catalog from Master Theorem ----
# V(xi), g(xi) = V'(xi)/Q, marginal ~ exp(-V(xi)/kT)

xi = np.linspace(-6, 6, 1000)

# 1. Quadratic (standard Nose-Hoover): V = Q*xi^2/2
V_quad = Q * xi**2 / 2
g_quad = xi  # V'/Q = Q*xi/Q = xi
p_quad = np.exp(-V_quad / kT)

# 2. Logarithmic (Log-Osc): V = Q*log(1+xi^2)
V_log = Q * np.log(1 + xi**2)
g_log = 2*xi / (1 + xi**2)  # V'/Q = Q*2*xi/(1+xi^2)/Q
p_log = np.exp(-V_log / kT)  # = (1+xi^2)^{-Q/kT} = (1+xi^2)^{-1} for Q=kT=1

# 3. Cosh: V = Q*log(cosh(xi))  =>  g = tanh(xi)
V_cosh = Q * np.log(np.cosh(xi))
g_cosh = np.tanh(xi)  # V'/Q = tanh(xi)
p_cosh = np.exp(-V_cosh / kT)  # = 1/cosh(xi) = sech(xi) for Q=kT=1

# 4. Double-well thermostat: V = Q*(xi^4/4 - xi^2/2 + 0.5)
# This creates a bimodal xi distribution -- interesting for multi-scale sampling
V_dw = Q * (xi**4/4 - xi**2/2 + 0.5)
g_dw = xi**3 - xi  # V'/Q
p_dw = np.exp(-V_dw / kT)

catalog = [
    ('NH (quadratic)', V_quad, g_quad, p_quad),
    ('Log-Osc', V_log, g_log, p_log),
    ('Cosh', V_cosh, g_cosh, p_cosh),
    ('Double-well', V_dw, g_dw, p_dw),
]

# ---- Plotting ----
fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

# Row 1: V(xi) potentials
ax = axes[0]
for name, V, g, p in catalog:
    ax.plot(xi, V, color=colors[name], lw=2.5, label=name)
ax.set_ylabel(r'$V(\xi)$', fontsize=LABEL_SIZE)
ax.set_title('(a) Thermostat potentials from the Master Theorem', fontsize=TITLE_SIZE)
ax.set_ylim(-0.5, 10)
ax.legend(fontsize=12, loc='upper center', ncol=2, framealpha=0.9)
ax.tick_params(labelsize=TICK_SIZE)
ax.axhline(0, color='gray', lw=0.5, ls='-')

# Row 2: g(xi) friction functions
ax = axes[1]
for name, V, g, p in catalog:
    ax.plot(xi, g, color=colors[name], lw=2.5, label=name)
ax.set_ylabel(r'$g(\xi) = V^\prime(\xi)/Q$', fontsize=LABEL_SIZE)
ax.set_title('(b) Friction functions', fontsize=TITLE_SIZE)
ax.set_ylim(-5, 5)
ax.tick_params(labelsize=TICK_SIZE)
ax.axhline(0, color='gray', lw=0.5, ls='-')
ax.axhline(1, color='gray', lw=0.8, ls=':', alpha=0.7)
ax.axhline(-1, color='gray', lw=0.8, ls=':', alpha=0.7)

# Annotate bounded vs unbounded
ax.annotate('bounded: $|g| \\leq 1$', xy=(4, 0.95), fontsize=11,
            color=colors['Log-Osc'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=colors['Log-Osc'], alpha=0.9))
ax.annotate('unbounded: $g \\to \\infty$', xy=(4.5, 4.2), fontsize=11,
            color=colors['NH (quadratic)'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=colors['NH (quadratic)'], alpha=0.9))

# Row 3: Marginal distributions p(xi)
ax = axes[2]
for name, V, g, p in catalog:
    # Normalize for display
    p_norm = p / (np.trapezoid(p, xi) + 1e-30)
    ax.plot(xi, p_norm, color=colors[name], lw=2.5, label=name)
ax.set_ylabel(r'$\rho(\xi) \propto e^{-V(\xi)/kT}$', fontsize=LABEL_SIZE)
ax.set_xlabel(r'$\xi$', fontsize=LABEL_SIZE)
ax.set_title(r'(c) Thermostat variable marginal distributions ($Q = kT = 1$)', fontsize=TITLE_SIZE)
ax.tick_params(labelsize=TICK_SIZE)
ax.axhline(0, color='gray', lw=0.5, ls='-')

# Annotate tail behavior
ax.annotate('Gaussian tails', xy=(-2.5, 0.02), fontsize=10,
            color=colors['NH (quadratic)'], fontstyle='italic')
ax.annotate('heavy (Cauchy) tails', xy=(2.5, 0.05), fontsize=10,
            color=colors['Log-Osc'], fontstyle='italic')

plt.tight_layout(pad=1.5)
outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', 'friction_geometry.png')
plt.savefig(outpath, dpi=300, bbox_inches='tight')
print(f"Saved: {outpath}")
plt.close()
