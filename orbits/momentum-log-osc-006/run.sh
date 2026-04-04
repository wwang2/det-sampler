#!/bin/bash
# Reproduce the Momentum-Dependent Log-Osc thermostat experiment
# Seeds: all use numpy default_rng(42) via the evaluator
set -e

cd "$(dirname "$0")/../.."

echo "=== Momentum-Dependent Log-Osc Thermostat: Full Evaluation ==="
echo ""

# Run the evaluation (alpha scan + rippled scan + Stage 2)
uv run python orbits/momentum-log-osc-006/evaluate.py

echo ""
echo "=== Generating figures ==="
uv run python orbits/momentum-log-osc-006/plot_diagnostics.py

echo ""
echo "Done. Figures saved to orbits/momentum-log-osc-006/figures/"
