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
echo "Step 1: Computing Lyapunov exponents..."
uv run python orbits/unified-theory-007/lyapunov.py

echo ""
echo "Step 2: Spectral analysis of multi-scale thermostats..."
uv run python orbits/unified-theory-007/spectral.py

echo ""
echo "Done. Figures saved in orbits/unified-theory-007/figures/"
