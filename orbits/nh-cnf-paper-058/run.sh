#!/bin/bash
# Compile the NH-CNF paper
# Requires: pdflatex (from any TeX distribution)
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex  # second pass for references
echo "Output: paper.pdf"
