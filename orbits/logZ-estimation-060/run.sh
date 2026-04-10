#!/bin/bash
# Reproduce logZ estimation experiments
# Usage: cd orbits/logZ-estimation-060 && bash run.sh
set -e
cd "$(dirname "$0")"
python experiment.py
