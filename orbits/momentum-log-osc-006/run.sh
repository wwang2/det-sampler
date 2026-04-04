#!/bin/bash
# Reproduce the Rippled Log-Osc (MLOSC-B) thermostat experiment
# Seeds: all use numpy default_rng(42) via the evaluator
set -e

cd "$(dirname "$0")/../.."

echo "=== Rippled Log-Osc Thermostat (MLOSC-B): Full Evaluation ==="
echo "Best config: eps=0.3, omega_xi=5.0"
echo ""

# Update WORKTREE path in evaluate.py to current directory
WORKTREE=$(pwd)

uv run python -c "
import sys, importlib.util
sys.path.insert(0, '${WORKTREE}')
spec = importlib.util.spec_from_file_location('solution', '${WORKTREE}/orbits/momentum-log-osc-006/solution.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from research.eval.evaluator import run_sampler
from research.eval.potentials import HarmonicOscillator1D, DoubleWell2D, GaussianMixture2D, Rosenbrock2D

# Stage 1: 1D Harmonic Oscillator
print('=== 1D HO: MLOSC-B eps=0.3, w=5, Q=0.8, dt=0.005 ===')
pot = HarmonicOscillator1D()
dyn = mod.RippledLogOsc(dim=1, kT=1.0, Q=0.8, epsilon=0.3, omega_xi=5.0)
r = run_sampler(dyn, pot, dt=0.005, n_force_evals=1_000_000, kT=1.0, integrator_cls=mod.RippledLogOscVerlet)
print(f'  KL: {r[\"kl_divergence\"]:.4f}')
print(f'  Ergodicity: {r[\"ergodicity\"][\"score\"]:.4f} (ergodic={r[\"ergodicity\"][\"ergodic\"]})')
print(f'  ESS/force_eval: {r[\"ess_metrics\"][\"ess_per_force_eval\"]:.6f}')
print()

# Stage 1: 2D Double-Well
print('=== 2D DW: MLOSC-B eps=0.3, w=5, Q=1.0, dt=0.04 ===')
pot = DoubleWell2D()
dyn = mod.RippledLogOsc(dim=2, kT=1.0, Q=1.0, epsilon=0.3, omega_xi=5.0)
r = run_sampler(dyn, pot, dt=0.04, n_force_evals=1_000_000, kT=1.0, integrator_cls=mod.RippledLogOscVerlet)
print(f'  KL: {r[\"kl_divergence\"]:.4f}')
print(f'  ESS/force_eval: {r[\"ess_metrics\"][\"ess_per_force_eval\"]:.6f}')
print(f'  Time to KL<0.01: {r[\"time_to_threshold_force_evals\"]}')
print()

# Stage 2: GMM
print('=== 2D GMM: MLOSC-B eps=0.3, w=5, Q=1.0, dt=0.03 ===')
pot = GaussianMixture2D()
dyn = mod.RippledLogOsc(dim=2, kT=1.0, Q=1.0, epsilon=0.3, omega_xi=5.0)
r = run_sampler(dyn, pot, dt=0.03, n_force_evals=1_000_000, kT=1.0, integrator_cls=mod.RippledLogOscVerlet)
print(f'  KL: {r[\"kl_divergence\"]:.4f}')
print(f'  ESS/force_eval: {r[\"ess_metrics\"][\"ess_per_force_eval\"]:.6f}')
print()

# Stage 2: Rosenbrock
print('=== 2D Rosenbrock: MLOSC-B eps=0.3, w=5, Q=1.0, dt=0.02 ===')
pot = Rosenbrock2D()
dyn = mod.RippledLogOsc(dim=2, kT=1.0, Q=1.0, epsilon=0.3, omega_xi=5.0)
r = run_sampler(dyn, pot, dt=0.02, n_force_evals=1_000_000, kT=1.0, integrator_cls=mod.RippledLogOscVerlet)
print(f'  KL: {r[\"kl_divergence\"]:.4f}')
print(f'  ESS/force_eval: {r[\"ess_metrics\"][\"ess_per_force_eval\"]:.6f}')
"

echo ""
echo "=== Generating figures ==="
uv run python orbits/momentum-log-osc-006/plot_diagnostics.py 2>/dev/null || echo "(plot generation skipped -- run separately if needed)"

echo ""
echo "Done."
