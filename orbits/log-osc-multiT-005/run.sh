#!/bin/bash
# Reproduce Multi-Scale Log-Osc thermostat evaluation
# Best config: Qs=[0.1, 0.7, 10.0], dt=0.03
#
# Usage:
#   cd /path/to/det-sampler
#   bash orbits/log-osc-multiT-005/run.sh

set -e

PYTHON="${PYTHON:-python3}"
WORKDIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$WORKDIR"

echo "=== Multi-Scale Log-Osc Thermostat Evaluation ==="
echo "Working directory: $WORKDIR"
echo "Python: $PYTHON"
echo ""

$PYTHON -u -c "
import sys
sys.path.insert(0, '.')
import importlib, numpy as np
sol = importlib.import_module('orbits.log-osc-multiT-005.solution')
from research.eval.potentials import (
    GaussianMixture2D, HarmonicOscillator1D, DoubleWell2D, Rosenbrock2D
)
from research.eval.evaluator import run_sampler

N = 1_000_000
best_Qs = [0.1, 0.7, 10.0]

print('Config: Qs={}, seed=42'.format(best_Qs))
print()

# Stage 1
ho = HarmonicOscillator1D(omega=1.0)
dyn = sol.MultiScaleLogOsc(dim=1, kT=1.0, Qs=best_Qs)
r = run_sampler(dyn, ho, dt=0.005, n_force_evals=N, integrator_cls=sol.MultiScaleLogOscVerlet)
erg = r['ergodicity']['score'] if r.get('ergodicity') else 0
print(f'1D HO:        KL={r[\"kl_divergence\"]:.4f}, ergo={erg:.4f}')

dw = DoubleWell2D(barrier_height=1.0, y_stiffness=0.5)
dyn = sol.MultiScaleLogOsc(dim=2, kT=1.0, Qs=best_Qs)
r = run_sampler(dyn, dw, dt=0.03, n_force_evals=N, integrator_cls=sol.MultiScaleLogOscVerlet)
print(f'2D DW:        KL={r[\"kl_divergence\"]:.4f}')

# Stage 2
gmm = GaussianMixture2D(n_modes=5, radius=3.0, sigma=0.5)
dyn = sol.MultiScaleLogOsc(dim=2, kT=1.0, Qs=best_Qs)
r = run_sampler(dyn, gmm, dt=0.03, n_force_evals=N, integrator_cls=sol.MultiScaleLogOscVerlet)
print(f'GMM (5-mode): KL={r[\"kl_divergence\"]:.4f}')

rb = Rosenbrock2D(a=0.0, b=5.0)
dyn = sol.MultiScaleLogOsc(dim=2, kT=1.0, Qs=best_Qs)
r = run_sampler(dyn, rb, dt=0.03, n_force_evals=N, integrator_cls=sol.MultiScaleLogOscVerlet)
print(f'Rosenbrock:   KL={r[\"kl_divergence\"]:.4f}')

print()
print('Done.')
"
