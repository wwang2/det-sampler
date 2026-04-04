#!/bin/bash
# Reproduce dual-bath thermostat evaluation from seed
# Run from repository root (the worktree):
#   cd /Users/wujiewang/code/det-sampler/.worktrees/dual-bath-001
#   bash orbits/dual-bath-001/run.sh

set -e
cd "$(dirname "$0")/../.."

echo "=== Dual-Bath Thermostat: NHC(2) + Hamiltonian Rotation ==="
echo "Seed: 42"
echo "Parameters: Q_xi=1.0, Q_eta=1.0, alpha=0.5"
echo ""

uv run python orbits/dual-bath-001/run_eval.py
