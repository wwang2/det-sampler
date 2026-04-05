#!/bin/bash
# Reproduce all publication figures for the deterministic thermostat sampler paper.
# Run from the repository root (or worktree root).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "=== Figure 1: The Problem (NH fails on HO) ==="
uv run python "$SCRIPT_DIR/make_fig1.py"

echo "=== Figure 2: The Solution (Bounded friction breaks KAM tori) ==="
uv run python "$SCRIPT_DIR/make_fig2.py"

echo "=== Figure 3: Trajectory on Landscape ==="
uv run python "$SCRIPT_DIR/make_fig3.py"

echo "=== Figure 4: Quantitative Comparison ==="
uv run python "$SCRIPT_DIR/make_fig4.py"

echo "=== Figure 5: Multi-Scale Mechanism ==="
uv run python "$SCRIPT_DIR/make_fig5.py"

echo "=== Figure 6: Scaling to High Dimensions ==="
uv run python "$SCRIPT_DIR/make_fig6.py"

echo ""
echo "All figures saved to: $SCRIPT_DIR/figures/"
ls -la "$SCRIPT_DIR/figures/"
