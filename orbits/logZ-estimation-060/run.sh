#!/bin/bash
# Reproduce logZ estimation experiments
# Usage: cd orbits/logZ-estimation-060 && bash run.sh
set -e
cd "$(dirname "$0")"

# v1: all experiments (E1-E4), fast pass
python experiment.py

# v2: better-equilibrated E1 and E3 (overwrites e1_logZ.png and e3_scaling.png)
python experiment_v2.py
