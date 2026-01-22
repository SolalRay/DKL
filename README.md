# Deep kernel learning for geostatistics

Implementation of the "Deep kernel learning for geostatistics" paper, written by Thomas Romary, Solal Raymondjean and Nicolas Dessasis (Mines Paris). Paper available in preprint at: https://minesparis-psl.hal.science/hal-05165114v2

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

## Use

The codebase is written as a package, with all functions clearly defined and commented in appropriate folders. 
In addition, two notebooks show how to use the function and evaluate the performance of Gaussian Processes coupled to Real NVP architectures.

More Specifically :
- `DKL_learning.ipynb`: Comparison of DKL performance on synthetic data against stationary and "ideal" baselines.
- `DKL_real_data.ipynb`: Application of the approach to real-world geostatistical datasets.

Also, one can use the `script` folder to write direct python code to evaluate GP real data without using ipykernels interface.

## Project Structure

The codebase is organized as a modular Python package:

- `DKL.models`: RealNVP architecture and GP model definitions.

- `DKL.data`: Dataset generation (synthetic non-stationary fields) and spatial transformations.

- `DKL.training`: Joint optimization loops, Early Stopping logic, and checkpointing.

- `DKL.utils`: Visualization tools and performance evaluation (MSE, CRPS).



