#!/bin/bash
# Reproduce the generalized chain thermostat experiments.
# Seed: 42 (pinned in solution.py)
#
# Usage: cd <project_root> && bash orbits/general-chains-015/run.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "=== Step 1: SymPy Verification ==="
python "$SCRIPT_DIR/verify_chains.py"

echo ""
echo "=== Step 2: Benchmark (1M force evals each) ==="
python "$SCRIPT_DIR/solution.py"

echo ""
echo "=== Done. Check orbits/general-chains-015/figures/ for plots ==="
