#!/usr/bin/env bash
# Reproduce all visual validation figures from scratch.
# Run from the worktree root:
#   cd /Users/wujiewang/code/det-sampler/.worktrees/visual-validation-013
#   bash orbits/visual-validation-013/run.sh

set -euo pipefail
SCRIPT_DIR="orbits/visual-validation-013"

echo "=== Visual Validation: Generating all figures ==="
echo ""

echo "[1/5] Trajectory Overlays..."
uv run python "$SCRIPT_DIR/make_trajectory_overlays.py"
echo ""

echo "[2/5] Density Comparison..."
uv run python "$SCRIPT_DIR/make_density_comparison.py"
echo ""

echo "[3/5] Convergence Studies..."
uv run python "$SCRIPT_DIR/make_convergence_studies.py"
echo ""

echo "[4/5] Efficiency Comparison..."
uv run python "$SCRIPT_DIR/make_efficiency_comparison.py"
echo ""

echo "[5/5] HO Phase Space Coverage..."
uv run python "$SCRIPT_DIR/make_ho_coverage.py"
echo ""

echo "=== All figures saved to $SCRIPT_DIR/figures/ ==="
ls -la "$SCRIPT_DIR/figures/"
