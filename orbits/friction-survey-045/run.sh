#!/bin/bash
# friction-survey-045: Survey bounded friction functions g(xi)
# Reproduces all results from seed
set -e
cd "$(dirname "$0")"
python run_experiment.py
