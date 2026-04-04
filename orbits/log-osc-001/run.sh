#!/bin/bash
# Reproduce the Log-Osc thermostat experiment
# Seeds: all use numpy default_rng(42) via the evaluator
set -e

cd "$(dirname "$0")/../.."

echo "=== Log-Osc Thermostat: Full Evaluation ==="
echo ""

# Run the evaluation
uv run python -c "
import sys, importlib.util
sys.path.insert(0, '.')
spec = importlib.util.spec_from_file_location('solution', 'orbits/log-osc-001/solution.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from research.eval.evaluator import run_sampler
from research.eval.potentials import HarmonicOscillator1D, DoubleWell2D

print('=== 1D Harmonic Oscillator: LogOsc Q=0.8, dt=0.005 ===')
pot = HarmonicOscillator1D()
dyn = mod.LogOscThermostat(dim=1, kT=1.0, Q=0.8)
r = run_sampler(dyn, pot, dt=0.005, n_force_evals=1_000_000, kT=1.0, integrator_cls=mod.LogOscVelocityVerlet)
print(f'  KL divergence: {r[\"kl_divergence\"]:.4f}')
print(f'  Ergodicity score: {r[\"ergodicity\"][\"score\"]:.4f} (ergodic={r[\"ergodicity\"][\"ergodic\"]})')
print(f'  ESS/force_eval: {r[\"ess_metrics\"][\"ess_per_force_eval\"]:.6f}')
print(f'  Wall time: {r[\"wall_seconds\"]:.1f}s')
print()

print('=== 2D Double-Well: LogOsc Q=1.0, dt=0.035 ===')
pot = DoubleWell2D()
dyn = mod.LogOscThermostat(dim=2, kT=1.0, Q=1.0)
r = run_sampler(dyn, pot, dt=0.035, n_force_evals=1_000_000, kT=1.0, integrator_cls=mod.LogOscVelocityVerlet)
print(f'  KL divergence: {r[\"kl_divergence\"]:.4f}')
print(f'  ESS/force_eval: {r[\"ess_metrics\"][\"ess_per_force_eval\"]:.6f}')
print(f'  Time to KL<0.01: {r[\"time_to_threshold_force_evals\"]}')
print(f'  Wall time: {r[\"wall_seconds\"]:.1f}s')
"

echo ""
echo "=== Generating figures ==="
uv run python orbits/log-osc-001/plot_diagnostics.py
echo ""
echo "Done. Figures saved to orbits/log-osc-001/figures/"
