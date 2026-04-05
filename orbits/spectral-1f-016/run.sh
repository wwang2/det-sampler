#!/usr/bin/env bash
# Reproduce all spectral-1f-016 experiments from scratch.
# Run from the repo root: bash orbits/spectral-1f-016/run.sh
set -euo pipefail

ORBIT_DIR="orbits/spectral-1f-016"

echo "=== Step 1: PSD Analysis (N=1,3,5,7,10 scales on 1D HO) ==="
uv run python "$ORBIT_DIR/make_psd.py"

echo ""
echo "=== Step 2: Lorentzian Decomposition (N=3, Q=[0.1, 1.0, 10.0]) ==="
uv run python "$ORBIT_DIR/make_decomposition.py"

echo ""
echo "=== Step 3: Spectral Matching on GMM (5 seeds, 2M evals) ==="
uv run python "$ORBIT_DIR/make_spectral_match.py"

echo ""
echo "=== Step 4: Consolidated Figure ==="
uv run python "$ORBIT_DIR/make_fig_spectral.py"

echo ""
echo "=== Done. Figures in $ORBIT_DIR/figures/ ==="
ls -la "$ORBIT_DIR/figures/"
