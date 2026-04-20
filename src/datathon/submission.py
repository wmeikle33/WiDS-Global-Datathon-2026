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
    ap.add_argument("--id-col", default="id")
    return ap.parse_args()


def main():
    args = parse_args()

    df = load_csv(args.input)
    if args.id_col not in df.columns:
        raise ValueError(f"Missing id column: {args.id_col}")

    ids = df[args.id_col].copy()
    X = df.drop(columns=[args.id_col])

    model = load_model(args.model)
    preds = model.predict_proba(X)[:, 1]

    submission = pd.DataFrame({"id": ids, "click": preds})
    save_csv(submission, args.output)
    print(f"Saved submission to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
