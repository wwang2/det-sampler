#!/bin/bash
# Reproduce all experiments for hormander-ergo-017
# Lie Bracket Analysis for Thermostat Ergodicity
#
# Prerequisites: uv installed, project dependencies available
# All random seeds: 42 (deterministic)

set -e
cd "$(dirname "$0")/../.."

echo "=== Lie Bracket Symbolic Computation ==="
uv run python orbits/hormander-ergo-017/lie_brackets.py

echo ""
echo "=== Rank Analysis (numerical grid + Monte Carlo) ==="
uv run python orbits/hormander-ergo-017/rank_analysis.py

echo ""
echo "=== Comparison with Lyapunov Data ==="
uv run python orbits/hormander-ergo-017/compare_lyapunov.py

echo ""
echo "=== Generate Figures ==="
uv run python orbits/hormander-ergo-017/plot_figures.py

echo ""
echo "=== Done ==="
echo "Figures in orbits/hormander-ergo-017/figures/"
echo "Theory write-up: orbits/hormander-ergo-017/theory.md"
