import argparse
import pandas as pd
from joblib import load

from .data import load_csv, save_csv


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/model.joblib")
    ap.add_argument("--input", required=True, help="CSV with feature columns")
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--id-col", default="id", help="ID column for submission output")
    return ap.parse_args()


def main():
    args = parse_args()

    model = load(args.model)
    X = load_csv(args.input)
    preds = {}
    skipped = []
    
    for h in horizons:
        if h not in models:
            print(f"Skipping horizon {h}h — no model trained (insufficient data or single class)")
            skipped.append(h)
            continue
        
        preds[h] = models[h].predict_proba(test[features])[:, 1]
        print(f"Horizon {h}h → predictions generated, shape: {preds[h].shape}")
    
    print(f"\nPredictions generated for: {list(preds.keys())}")
    if skipped:
        print(f"Skipped horizons (no model): {skipped}")

    save_csv(out, args.output)
    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
