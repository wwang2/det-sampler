#!/usr/bin/env bash
# Reproduce all deep-geometry-012 figures.
# Run from the repository root (det-sampler/).
# Requires: uv, Python 3.14, numpy, matplotlib
#
# Each script is self-contained and produces one multi-panel figure
# in orbits/deep-geometry-012/figures/
#
# Seeds: all scripts use seed=42 for reproducibility.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

echo "=== deep-geometry-012: Generating publication figures ==="
echo "Output: $SCRIPT_DIR/figures/"
mkdir -p "$SCRIPT_DIR/figures"

echo ""
echo "[1/5] Friction function geometry (fast)..."
uv run python "$SCRIPT_DIR/make_friction_geometry.py"

echo ""
echo "[2/5] Mechanism schematic (50k steps, ~10s)..."
uv run python "$SCRIPT_DIR/make_mechanism_schematic.py"

echo ""
echo "[3/5] 3D Phase space flow (500k steps x2, ~60s)..."
uv run python "$SCRIPT_DIR/make_phase_space_3d.py"

echo ""
echo "[4/5] Torus comparison (500k steps x6, ~5min)..."
uv run python "$SCRIPT_DIR/make_torus_comparison.py"

echo ""
echo "[5/5] Poincare sections (1M steps x3, ~5min)..."
uv run python "$SCRIPT_DIR/make_poincare_sections.py"

echo ""
echo "=== All figures generated ==="
ls -la "$SCRIPT_DIR/figures/"
