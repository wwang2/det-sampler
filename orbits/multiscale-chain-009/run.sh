#!/bin/bash
# Reproduce Multi-Scale LOCR experiments
# Run from the worktree root: .worktrees/multiscale-chain-009/
set -e

PYTHON="${PYTHON:-/Users/wujiewang/code/det-sampler/.venv/bin/python3}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKTREE="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$WORKTREE"

echo "=== Running Multi-Scale LOCR evaluation ==="
echo "Worktree: $WORKTREE"
echo "Python: $PYTHON"

$PYTHON "$SCRIPT_DIR/evaluate.py"

echo ""
echo "=== Generating figures ==="
$PYTHON "$SCRIPT_DIR/plot.py"

echo ""
echo "Done. Results in $SCRIPT_DIR/results.json"
echo "Figures in $SCRIPT_DIR/figures/"
