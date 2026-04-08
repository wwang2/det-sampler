#!/bin/bash
# Reproduce paper-final-049 experiments
# Usage: bash orbits/paper-final-049/run.sh
set -e
cd "$(dirname "$0")/../.."
python3 orbits/paper-final-049/run_experiment.py
