#!/bin/bash
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
cd "$ROOT"
python3 -u "$HERE/run_experiment.py"
python3 -u "$HERE/make_figures.py"
