#!/bin/bash
set -e
cd "$(dirname "$0")"
python3 e3_variance.py
python3 e3_walltime.py
python3 e3_training.py
python3 e2_bnn_uci.py
