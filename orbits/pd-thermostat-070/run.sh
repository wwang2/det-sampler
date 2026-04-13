#!/bin/bash
# Reproduce PD thermostat experiment
# Usage: cd <repo-root> && bash orbits/pd-thermostat-070/run.sh
set -e
cd "$(dirname "$0")/../.."
python -m orbits.pd-thermostat-070.solution 2>&1 || python orbits/pd-thermostat-070/solution.py 2>&1
