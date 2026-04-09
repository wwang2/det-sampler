#!/bin/bash
# Reproduce forensic-qeff-054 experiments
# Usage: bash run.sh
set -e
cd "$(dirname "$0")"
python3 solution.py
