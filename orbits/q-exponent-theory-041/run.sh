#!/bin/bash
# q-exponent-theory-041: reproduce the analytical derivation and numerical verification
set -e
cd "$(dirname "$0")"
python3 verify.py
