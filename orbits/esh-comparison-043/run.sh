#!/bin/bash
# Reproduce ESH comparison experiment
# Expected runtime: ~10-15 min on a laptop
set -e
cd "$(dirname "$0")"
python3 run_experiment.py
echo "Results in results.json, figures in figures/"
