#!/usr/bin/env bash
# Reproduce q-omega-mapping-040
set -e
cd "$(dirname "$0")/../.."
python3 orbits/q-omega-mapping-040/run_experiment.py
python3 orbits/q-omega-mapping-040/make_figures.py
