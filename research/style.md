# Plotting Style Guide

All figures in this project should follow these conventions for consistency.

## General
- Use matplotlib with the default style
- Figure size: (8, 6) for single plots, (12, 5) for side-by-side
- DPI: 150 for screen, 300 for paper
- Font size: 14 for labels, 12 for ticks, 16 for titles

## Colors
- Baselines: gray dashed lines
- NH: `#1f77b4` (blue)
- NHC: `#ff7f0e` (orange)  
- Novel samplers: use tab10 colormap starting from index 2
- Error regions: alpha=0.2 fill_between

## Axes
- Always label axes with units where applicable
- Use log scale for KL divergence traces
- Force evaluation count on x-axis (not time steps)

## Specific Plot Types

### KL Convergence Trace
- x: force evaluations (log scale)
- y: KL divergence (log scale)
- Horizontal dashed line at KL=0.01 threshold
- Legend in upper right

### Phase Space Coverage (1D HO)
- 2D scatter plot of (q, p)
- Overlay expected Gaussian contours (1σ, 2σ, 3σ)
- Equal aspect ratio

### Energy Distribution
- Histogram of total energies
- Overlay expected Boltzmann distribution
- Report mean and std in legend

### Benchmark Comparison Table
- Bar chart grouped by potential
- Metric on y-axis, samplers on x-axis
- Error bars from 3 independent runs with different seeds
