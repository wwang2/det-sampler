#!/bin/bash
# gprime-ablation-052: Isolate role of sign(g') from g'(0) coupling.
set -e
cd "$(dirname "$0")"

echo "=== Main sweep (4 methods x 7 Q_c x 20 seeds) ==="
python3 solution.py --sweep

echo ""
echo "=== Control experiments (floor + double-well) + active summary ==="
python3 solution.py --controls

echo ""
echo "=== Generate figures ==="
python3 make_figure.py
