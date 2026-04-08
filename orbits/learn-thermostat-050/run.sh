#!/bin/bash
# Reproduce learn-thermostat-050 experiment.
# Requires: pytorch 2.8.0+, uni-diffsim at /Users/wujiewang/code/uni-diffsim
set -e
cd "$(dirname "$0")/../.."
python3 -u orbits/learn-thermostat-050/run_experiment.py | tee orbits/learn-thermostat-050/run.log
python3 orbits/learn-thermostat-050/make_figures.py
