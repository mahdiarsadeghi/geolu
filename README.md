# Geolu
*Where Algorithms Predict Value*

## Live dashboard

[Open the dashboard](https://mahdiarsadeghi.github.io/geolu/)

## Overview

Geolu is an experimental forecasting system for Gold, Oil, Bitcoin, and the S&P 500. It reads historical price data, applies the Bachata multi-scale Fourier model, evaluates forecasts on chronological holdout windows, and exports data for the dashboard.

## What it does

- Tracks and visualizes historical prices across four assets.
- Produces weekly, monthly, and yearly forecasts.
- Evaluates predictions with normalized errors across rolling historical windows.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python predictor.py
```

The workflow entry point is `predictor.py`. It writes prediction history and dashboard data to the tracked CSV, JSON, and `docs/` outputs.

## Limitations

This is an experimental quantitative model. Forecasts carry uncertainty and should be compared with simple baselines before being used for decisions. The repository's evaluation results are historical experiments, not financial advice.

## License

Proprietary algorithms and predictive methodologies. Educational and personal use permitted. Commercial use requires permission from the copyright holder.
