#!/bin/bash
# mode-hopping-042: Systematic mode-hopping benchmark
# Reproduces all experiments from seed
set -e
cd "$(dirname "$0")"
python run_experiment.py
