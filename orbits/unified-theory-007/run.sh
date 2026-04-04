#!/bin/bash
# Reproduce all experiments for the unified theory orbit.
# Run from the repository root: bash orbits/unified-theory-007/run.sh

set -e

echo "============================================"
echo "Unified Theory of Generalized Friction Thermostats"
echo "============================================"

# Ensure figures directory exists
mkdir -p orbits/unified-theory-007/figures

echo ""
echo "Step 1: Symbolic verification of Master Theorem..."
uv run python orbits/unified-theory-007/verify_theorem.py

echo ""
echo "Step 2: Computing Lyapunov exponents (short scan)..."
uv run python orbits/unified-theory-007/lyapunov.py

echo ""
echo "Step 3: Extended Lyapunov at key Q values + phase portraits..."
PYTHONUNBUFFERED=1 uv run python orbits/unified-theory-007/lyapunov_long.py

echo ""
echo "Step 4: Spectral analysis of multi-scale thermostats..."
uv run python orbits/unified-theory-007/spectral.py

echo ""
echo "Step 5: Ergodicity score & coverage analysis..."
PYTHONUNBUFFERED=1 uv run python orbits/unified-theory-007/coverage.py

echo ""
echo "Done. Figures saved in orbits/unified-theory-007/figures/"
echo "Theory document: orbits/unified-theory-007/theory.md"
