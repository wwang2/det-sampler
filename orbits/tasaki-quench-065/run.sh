#!/bin/bash
# Reproduce tasaki-quench-065 experiment
# Requires: python3, numpy, matplotlib, scipy
set -e
cd "$(dirname "$0")"
python3 experiment.py --all --seeds 42,123,7
