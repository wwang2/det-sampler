#!/bin/bash
# Reproduce all publication narrative figures
# Seed: 42 (deterministic simulations in toy systems)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "=== Generating schematic figures ==="
python3 orbits/pub-narrative-011/make_schematics.py

echo ""
echo "=== Generating toy system illustrations ==="
python3 orbits/pub-narrative-011/make_toy_systems.py

echo ""
echo "=== All figures generated ==="
ls -la orbits/pub-narrative-011/figures/
