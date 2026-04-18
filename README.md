# WiDS-Global-Datathon-2026

When a wildfire ignites, emergency managers and responders must decide which communities to warn, when to warn them, and where to position scarce resources. Decisions must be made before certainty is available. The response requires both prioritization (which fires are most urgent soon) and calibrated risk estimates (how likely a fire is to threaten evacuation zones within actionable time windows).

This competition turns that operational need into a survival analysis challenge. You will generate calibrated probability forecasts across multiple time horizons to support real-world decisions.

## Quickstart

```

git clone https://github.com/wmeikle33/Click-Through-Rate-Prediction.git
cd Click-Through-Rate-Prediction
python -m venv .venv
source .venv/bin/activate
pip install -e ".[data]"
python scripts/download_data.py

```bash
pip install -e .
ctr-train --csv data/raw/train.csv --model lgbm --model-path models/lgbm.joblib

```

## Summary 

```

wids-global-datathon-2026/
├── src/                   # reusable code (data, features, model)
├── scripts/               # CLI entrypoints: train/predict
├── notebooks/             # original notebook + exported .py
├── data/raw/              # place raw data here (gitignored)
├── models/                # saved models (gitignored)
├── reports/figures/       # plots (gitignored)
├── tests/                 # add unit tests if needed
├── requirements.txt
└── README.md

```

