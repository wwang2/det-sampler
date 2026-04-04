#!/bin/bash
# Reproduce LOCR thermostat evaluation from seed
# Usage: cd /Users/wujiewang/code/det-sampler/.worktrees/log-osc-chain-002 && bash orbits/log-osc-chain-002/run.sh

set -e

BASEDIR="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$BASEDIR/orbits/log-osc-chain-002/.venv"

# Create venv if needed
if [ ! -d "$VENV" ]; then
    echo "Creating Python 3.12 venv..."
    uv venv --python 3.12 "$VENV"
    uv pip install numpy scipy matplotlib --python "$VENV/bin/python"
fi

cd "$BASEDIR"
"$VENV/bin/python" orbits/log-osc-chain-002/run_eval.py
