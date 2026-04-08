#!/bin/bash
# Reproduce per-mode-coupling-048 experiments
set -e
cd "$(dirname "$0")"
python run_experiment.py
