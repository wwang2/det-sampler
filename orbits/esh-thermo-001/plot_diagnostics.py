"""Diagnostic plots for SinhDrive-NHC thermostat."""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/wujiewang/code/det-sampler/.worktrees/esh-thermo-001')

BASE = '/Users/wujiewang/code/det-sampler/.worktrees/esh-thermo-001/orbits/esh-thermo-001'


def plot_phase_space_comparison():
    """Phase space coverage: NHC vs SinhDrive-NHC on 1D HO."""
    nhc = np.load(f'{BASE}/nhc_samples.npz')
    sd = np.load(f'{BASE}/sdnhc_samples.npz')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Expected Gaussian contours
    sigma_q = 1.0  # sqrt(kT/omega^2) = 1 for kT=1, omega=1
    sigma_p = 1.0  # sqrt(m*kT) = 1

    for ax, data, title, color in [
        (axes[0], nhc, 'NHC (M=3, Q=1.0)', '#ff7f0e'),
        (axes[1], sd, 'SinhDrive-NHC (M=3, Q=0.15, beta=0.05)', '#2ca02c'),
    ]:
        q, p = data['q'], data['p']
        # Subsample for scatter
        idx = np.random.default_rng(42).choice(len(q), min(5000, len(q)), replace=False)
        ax.scatter(q[idx], p[idx], s=1, alpha=0.3, color=color, rasterized=True)

        # Gaussian contours
        theta = np.linspace(0, 2*np.pi, 200)
        for nsig in [1, 2, 3]:
            ax.plot(nsig*sigma_q*np.cos(theta), nsig*sigma_p*np.sin(theta),
                    'k--', alpha=0.4, linewidth=1)

        ax.set_xlim(-4*sigma_q, 4*sigma_q)
        ax.set_ylim(-4*sigma_p, 4*sigma_p)
        ax.set_xlabel('q (position)', fontsize=14)
        ax.set_ylabel('p (momentum)', fontsize=14)
        ax.set_title(title, fontsize=14)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=12)

    fig.suptitle('Phase Space Coverage: 1D Harmonic Oscillator', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{BASE}/figures/phase_space_ho.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved phase_space_ho.png')


def plot_sinh_drive_function():
    """Visualize the sinh drive vs linear drive."""
    fig, ax = plt.subplots(figsize=(8, 6))

    G = np.linspace(-4, 4, 500)  # G_standard = K - dim*kT

    # Linear drive (standard NHC)
    ax.plot(G, G, color='#ff7f0e', linewidth=2, label='Standard NHC: g(G) = G')

    # Sinh drives for various beta
    for beta, ls in [(0.05, '-'), (0.2, '--'), (0.5, ':'), (1.0, '-.')]:
        g = np.sinh(beta * G) / beta
        color = '#2ca02c' if beta == 0.05 else 'gray'
        lw = 2.5 if beta == 0.05 else 1.5
        ax.plot(G, g, color=color, linewidth=lw, linestyle=ls,
                label=f'Sinh: g(G) = sinh({beta}*G)/{beta}')

    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('G = K - dim*kT (kinetic energy deviation)', fontsize=14)
    ax.set_ylabel('g(G) (thermostat drive signal)', fontsize=14)
    ax.set_title('SinhDrive-NHC: Nonlinear Drive Function', fontsize=16)
    ax.legend(fontsize=12, loc='upper left')
    ax.tick_params(labelsize=12)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-8, 8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{BASE}/figures/sinh_drive_function.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved sinh_drive_function.png')


def plot_ergodicity_comparison():
    """Bar chart comparing ergodicity metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Data from evaluations
    methods = ['NH\n(M=1)', 'NHC\n(M=3)', 'SD-NHC\n(M=3)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # HO ergodicity scores (from our runs)
    erg_scores = [0.54, 0.924, 0.954]
    ax = axes[0]
    bars = ax.bar(methods, erg_scores, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(0.85, color='red', linestyle='--', linewidth=1.5, label='Ergodic threshold')
    ax.set_ylabel('Ergodicity Score', fontsize=14)
    ax.set_title('1D Harmonic Oscillator: Ergodicity', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)
    for bar, val in zip(bars, erg_scores):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # DW KL divergence
    kl_dw = [0.037, 0.029, 0.029]
    ax = axes[1]
    bars = ax.bar(methods, kl_dw, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(0.01, color='red', linestyle='--', linewidth=1.5, label='Target KL')
    ax.set_ylabel('KL Divergence', fontsize=14)
    ax.set_title('2D Double Well: Distribution Accuracy', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)
    for bar, val in zip(bars, kl_dw):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.001, f'{val:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{BASE}/figures/ergodicity_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved ergodicity_comparison.png')


def plot_parameter_sensitivity():
    """Show ergodicity vs beta_drive parameter."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Data from our scans (Q=0.15)
    betas = [0.0, 0.05, 0.10, 0.15, 0.20]
    ergs = [0.924, 0.954, 0.925, 0.928, 0.913]
    kls = [0.0016, 0.0010, 0.0017, 0.0011, 0.0010]

    ax2 = ax.twinx()

    l1, = ax.plot(betas, ergs, 'o-', color='#2ca02c', linewidth=2, markersize=8,
                  label='Ergodicity score')
    ax.axhline(0.924, color='#ff7f0e', linestyle='--', linewidth=1.5,
               label='NHC baseline (0.924)')
    ax.axhline(0.85, color='red', linestyle=':', linewidth=1, alpha=0.7,
               label='Ergodic threshold (0.85)')

    l2, = ax2.plot(betas, kls, 's--', color='#1f77b4', linewidth=2, markersize=8,
                   label='KL divergence')

    ax.set_xlabel('beta_drive (sinh nonlinearity)', fontsize=14)
    ax.set_ylabel('Ergodicity Score (1D HO)', fontsize=14, color='#2ca02c')
    ax2.set_ylabel('KL Divergence (1D HO)', fontsize=14, color='#1f77b4')
    ax.set_title('SinhDrive-NHC: Parameter Sensitivity (Q=0.15, M=3)', fontsize=16)
    ax.tick_params(labelsize=12)
    ax2.tick_params(labelsize=12)

    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=12, loc='center left')

    ax.set_ylim(0.85, 0.97)
    ax2.set_ylim(0, 0.005)

    plt.tight_layout()
    plt.savefig(f'{BASE}/figures/parameter_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved parameter_sensitivity.png')


if __name__ == '__main__':
    plot_sinh_drive_function()
    plot_ergodicity_comparison()
    plot_parameter_sensitivity()

    # Only plot phase space if samples exist
    try:
        plot_phase_space_comparison()
    except FileNotFoundError:
        print('Skipping phase space plot (samples not yet collected)')
