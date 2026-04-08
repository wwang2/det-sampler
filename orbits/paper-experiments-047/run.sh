#!/bin/bash
# Reproduce paper-experiments-047: definitive experiments for the thermostat paper
# Expected runtime: ~2-3 hours on 10-core machine
set -e
cd "$(dirname "$0")"
python3 run_experiment.py
