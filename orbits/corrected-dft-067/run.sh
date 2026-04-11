#!/bin/bash
# corrected-dft-067: Crooks DFT test on sigma_bath per-trajectory
# Reproduces the experiment from scratch (re-runs quench simulations)
set -e
cd "$(dirname "$0")"
python3 analysis.py
