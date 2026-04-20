from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from datathon.data import load_csv, save_csv
from datathon.model import load_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="submission.csv")
    ap.add_argument("--event_id", default="id")
    return ap.parse_args()


def main():
    args = parse_args()

    df = load_csv(args.input)
    if args.id_col not in df.columns:
        raise ValueError(f"Missing id column: {args.id_col}")

    ids = df[args.id_col].copy()
    X = df.drop(columns=[args.id_col])

    model = load_model(args.model)
    fallback_72 = train[train['y_72'].notna()]['y_72'].mean()
    if pd.isna(fallback_72):
        fallback_72 = 0.5
    
    # Now conditionally override or just log
    if 72 not in models:
        print(f"Horizon 72h fallback probability: {fallback_72:.4f}")
    
    # Build submission — now fallback_72 is always defined
    submission = pd.DataFrame({
        'event_id': test['event_id'],
        'prob_12h': preds.get(12, np.full(len(test), np.nan)),
        'prob_24h': preds.get(24, np.full(len(test), np.nan)),
        'prob_48h': preds.get(48, np.full(len(test), np.nan)),
        'prob_72h': preds.get(72, np.full(len(test), fallback_72)),
    })

    save_csv(submission, args.output)
    print(f"Saved submission to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
