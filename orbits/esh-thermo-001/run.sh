#!/bin/bash
# Reproduce SinhDrive-NHC evaluation
# Run from the worktree root: /Users/wujiewang/code/det-sampler/.worktrees/esh-thermo-001

set -e

WORKTREE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="/Users/wujiewang/code/det-sampler/.venv/bin/python3"

cd "$WORKTREE_ROOT"

echo "=== SinhDrive-NHC Thermostat Evaluation ==="
echo "Parameters: Q=0.15, beta_drive=0.05, M=3, dt=0.01, seed=42"
echo ""

# Stage 1 evaluation
PYTHONPATH=. $PYTHON orbits/esh-thermo-001/solution.py \
    --stage 1 \
    --dt 0.01 \
    --Q 0.15 \
    --beta-drive 0.05 \
    --chain-length 3 \
    --seed 42

echo ""
echo "=== Generating diagnostic plots ==="
PYTHONPATH=. $PYTHON orbits/esh-thermo-001/plot_diagnostics.py

echo ""
echo "Done. Results in orbits/esh-thermo-001/figures/"
