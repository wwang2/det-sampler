#!/bin/bash
# Reproduce NH-CNF deep experiments
# Run from repo root: bash orbits/nh-cnf-deep-057/run.sh
set -e
cd "$(dirname "$0")/../.."
uv run python orbits/nh-cnf-deep-057/experiment.py
