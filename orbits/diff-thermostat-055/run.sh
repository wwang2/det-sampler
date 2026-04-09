#!/bin/bash
# Reproduce all experiments for orbit/diff-thermostat-055
# Differentiable simulation to map Q_eff surface, learn g(xi), learn Q distribution
set -e
cd "$(dirname "$0")"

echo "=== E1: Q_eff(kappa, D) surface ==="
/opt/homebrew/bin/python3 e1_qeff_surface.py

echo ""
echo "=== E2: Learn g(xi) as MLP ==="
/opt/homebrew/bin/python3 e2_learn_g.py

echo ""
echo "=== E3: Learn (Q_1,...,Q_N) jointly ==="
/opt/homebrew/bin/python3 e3_learn_Q_distribution.py

echo ""
echo "All experiments complete."
