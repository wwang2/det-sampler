#!/usr/bin/env bash
# Reproduce triple-identity-064.
# Single CPU run; ~30 seconds wall clock.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
python3 experiment.py
