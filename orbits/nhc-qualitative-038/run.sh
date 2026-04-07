#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
uv run python orbits/nhc-qualitative-038/make_figures.py
