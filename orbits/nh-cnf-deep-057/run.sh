#!/usr/bin/env bash
# Reproduce NH-CNF deep experiments
# Run from repo root: bash orbits/nh-cnf-deep-057/run.sh
set -e
cd "$(dirname "$0")/../.."

# Canonical entry point (refined 3): analytical potentials for E1
uv run python orbits/nh-cnf-deep-057/experiment_refine3.py

# Refined experiments E2, E3, E4, E5, E6, E7 (with Langevin overlay in E6)
uv run python orbits/nh-cnf-deep-057/experiment_refine2.py

# Original runs (historical, for reproducibility)
# uv run python orbits/nh-cnf-deep-057/experiment.py  # uncomment to reproduce initial iteration
