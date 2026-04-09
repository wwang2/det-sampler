#!/bin/bash
# Reproduce experiment for orbit/bayesian-posterior-056
# NH thermostat as CNF for Bayesian posterior sampling
set -e
cd "$(dirname "$0")"
python3 experiment.py
