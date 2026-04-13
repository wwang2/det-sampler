#!/bin/bash
# Reproduce sublinear-g-069 experiment
# Usage: bash orbits/sublinear-g-069/run.sh
set -e
cd "$(dirname "$0")/../.."
python -m orbits.sublinear-g-069.solution 2>&1 || python orbits/sublinear-g-069/solution.py 2>&1
