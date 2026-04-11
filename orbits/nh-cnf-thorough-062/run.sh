#!/bin/bash
# Reproduces every headline figure in this orbit end-to-end.
#
#   e3_variance.py        -> fig_variance_scaling.png     (E3.1)
#   e3_walltime.py        -> fig_walltime.png             (E3.4)
#   e3_training.py        -> fig_training_dynamics.png    (E3.3 toy)
#   e3_training_highd.py  -> fig_training_stability.png   (E3.3 headline)
#   e2_bnn_uci.py         -> fig_bnn_uci.png              (E2)
#   _replot_from_json.py  -> regenerates the two cleanup-polished panels
#                            (training-stability + variance-scaling) from
#                            the saved JSON with the cleanup annotations.
set -e
cd "$(dirname "$0")"
python3 e3_variance.py
python3 e3_walltime.py
python3 e3_training.py
python3 e3_training_highd.py
python3 e2_bnn_uci.py
python3 _replot_from_json.py
