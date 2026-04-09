#!/bin/bash
# gprime-ablation-052: Isolate role of sign(g') from g'(0) coupling.
set -e
cd "$(dirname "$0")"
python3 solution.py
python3 make_figure.py
