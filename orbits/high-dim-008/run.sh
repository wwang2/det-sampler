#!/bin/bash
# Reproduce high-dimensional scaling study
# Seed: 42
# Run from project root: cd /Users/wujiewang/code/det-sampler/.worktrees/high-dim-008

set -e

cd "$(dirname "$0")/../.."

echo "=== Quick test (500K evals) ==="
python -m orbits.high_dim_008.solution --quick --seed 42

echo ""
echo "=== Full run (5-10M evals) ==="
# python -m orbits.high_dim_008.solution --seed 42
