#!/bin/bash
# Reproduce high-dimensional scaling study
# Seed: 42
# Run from project root: cd /Users/wujiewang/code/det-sampler/.worktrees/high-dim-008
set -e
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-/Users/wujiewang/code/det-sampler/.venv/bin/python}"

echo "=== Quick test (500K evals) ==="
PYTHONPATH=. $PYTHON orbits/high-dim-008/run_all.py --quick --seed 42

# Uncomment for full budget:
# echo "=== Full run (2M evals) ==="
# PYTHONPATH=. $PYTHON orbits/high-dim-008/run_all.py --budget 2000000 --seed 42

# Uncomment for maximum budget:
# echo "=== Max run (5-10M evals) ==="
# PYTHONPATH=. $PYTHON orbits/high-dim-008/run_all.py --seed 42
