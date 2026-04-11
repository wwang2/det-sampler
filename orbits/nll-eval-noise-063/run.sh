#!/usr/bin/env bash
# Reproduce nll-eval-noise-063.
set -euo pipefail
cd "$(dirname "$0")"
python3 experiment.py
