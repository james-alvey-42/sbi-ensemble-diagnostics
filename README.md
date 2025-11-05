# SBI Ensemble Diagnostics

[![arXiv](https://img.shields.io/badge/arXiv-2507.13495-b31b1b.svg)](https://arxiv.org/abs/2507.13495)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code accompanying the paper

**Simulation-based inference with deep ensembles: Evaluating calibration uncertainty and detecting model misspecification**

James Alvey, Carlo R. Contaldi, Mauro Pieroni (2025)

## Overview

This repository contains code and short runnable examples used in the paper above. The code focuses on training ensembles of neural density estimators (normalizing flows) for simulation-based inference (SBI) and on simple ensemble diagnostics (pairwise KL divergence, training/validation monitoring) used to evaluate calibration uncertainty and to detect possible model misspecification.

## Highlights

- Lightweight utilities to build and train normalizing-flow density estimators with `sbi` and `PyTorch`.
- Small example scripts that reproduce experiment flows used in the manuscript (self-contained, easy to tweak).
- Monte Carlo KL estimators used to quantify ensemble spread and simple plotting helpers.

## Installation

Create a virtual environment (recommended) and install the package in editable mode:

```bash
git clone <repository-url>
cd sbi-ensemble-diagnostics
pip install -e .
```

If you prefer to use `pyproject.toml`, install via your build tool of choice. Otherwise `pip install -e .` will use `setup.py` metadata.

## Dependencies

- Python >= 3.8
- PyTorch
- sbi (simulation-based inference library)
- sbibm (benchmark tasks used by some examples)
- NumPy, matplotlib, tqdm

## Quick examples

Examples live in the `examples/` directory. Two scripts of particular interest:

- `examples/sbi_KL.py` — train an ensemble on an `sbibm` benchmark and compute pairwise KL diagnostics.
- `examples/extrapolation.py` — a synthetic Gaussian-frequency example showing ensemble training and diagnostics.

You can run a short smoke test by editing the example script to reduce dataset sizes / epochs and then just running the script.

## Project structure

```
sbi_ensemble_diagnostics/
  __init__.py
  utils.py            # helpers: datasets, KL estimators, builders, plotting
examples/
  sbi_KL.py
  extrapolation.py
pyproject.toml        # optional PEP 621 metadata (if present)
README.md
LICENSE
```

## Key modules

- `sbi_ensemble_diagnostics.utils` — helper functions used by the examples: `NPEData` dataset wrapper, KL estimators, `build_density_estimator`, `setup_scheduler`, and plotting utilities.

## Design notes

- Examples are intentionally minimal and configuration is exposed as top-level variables inside each script for readability and quick experimentation.
- KL estimation is performed by Monte Carlo sampling and uses a running average over repeated batches to stabilise the estimate; tolerance and sample-count options are available in the scripts.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request for bug reports, feature requests, or improvements.

## Citation

If you use this code in your research, please cite the paper listed at the top of this README.

## License

This project is released under the MIT License — see the `LICENSE` file for details.

## Contact

For questions about the code or experiments, open a GitHub issue in this repository.
