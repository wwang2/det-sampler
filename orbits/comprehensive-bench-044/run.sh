#!/bin/bash
# Reproduce comprehensive benchmark from seed
set -e
cd "$(dirname "$0")"
python run_experiment.py
