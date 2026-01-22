# Deep kernel learning for geostatistics

Implementation of the "Deep kernel learning for geostatistics" paper, written by Thomas Romary, Solal Raymondjean and Nicolas Dessasis (Mines Paris). Preprint available  at: https://minesparis-psl.hal.science/hal-05165114v2

## Overview

This project addresses the challenge of **non-stationarity** in Gaussian Processes (GP) for geostatistics. While traditional GPs assume second-order stationarity, this implementation uses **Normalizing Flows (RealNVP)** to learn a bijective space deformation.

By transporting the geographic input space into a latent "deformed" space where stationarity and isotropy hold, we can leverage the power of GPs on complex, large-scale datasets.

## Installation

To install it in editable mode, run:

```bash
git clone https://github.com/SolalRay/DKL.git
cd DKL
pip install -e .
```
You might want to use a virtual environment

## Usage

### Interactive Notebooks
Two primary notebooks are provided to demonstrate the workflow and evaluate the performance of Gaussian Processes combined with Real NVP architectures:

* **`DKL_learning.ipynb`**: Benchmarks the DKL approach on synthetic non-stationary data against stationary and "ideal" baselines.
* **`DKL_real_data.ipynb`**: Demonstrates the application of the model to real-world geostatistical datasets.

### Scripting
The **`script/`** folder contains Python scripts to run evaluations and training directly from the command line.

## Project Structure

The codebase is structured as a Python package, with all modules documented and organized by functionality.

- `DKL.models`: RealNVP architecture and GP model definitions.

- `DKL.data`: Dataset generation (synthetic non-stationary fields) and spatial transformations.

- `DKL.training`: Joint optimization loops, Early Stopping logic, and checkpointing.

- `DKL.utils`: Visualization tools and performance evaluation (MSE, CRPS).



