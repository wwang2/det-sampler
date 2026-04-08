#!/bin/bash
# Reproduce thermostat-dynamics-046 experiments
# Expected runtime: ~4 minutes on a modern laptop
set -e
cd "$(dirname "$0")"
python3 run_experiment.py
echo "Done. Figures in figures/"
