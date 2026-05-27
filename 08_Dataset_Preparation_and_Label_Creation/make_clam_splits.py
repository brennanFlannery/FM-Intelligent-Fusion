#!/usr/bin/env python3
"""
make_clam_splits.py
Create CLAM-compatible k-fold split CSVs from a master slide list.

Required master-CSV columns
  case_id   – patient identifier (all slides from one patient stay together)
  slide_id  – basename of the .h5/.pt feature file (no suffix)
  label     – integer class label

Example:
  python make_clam_splits.py --csv_path dataset_csv/grade_labels.csv \
                             --out_dir splits/kidney_grade --k 5
"""
import argparse, random, os, math, pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

# ────────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate CLAM k-fold split CSVs")
    p.add_argument('--csv_path',  type=str, required=True)
    p.add_argument('--out_dir',   type=str, required=True)
    p.add_argument('--k',         type=int, default=5,   help='number of folds')
    p.add_argument('--val_frac',  type=float, default=0.10,
                   help='fraction of remaining data used for val inside each train fold')
    p.add_argument('--test_frac', type=float, default=0.10,
                   help='fraction of patients held out as test fold each round')
    p.add_argument('--seed',      type=int, default=42)
    return p

# ────────────────────────────────────────────────────────────────────────────────
def make_splits(df: pd.DataFrame, k: int, val_frac: float,
                test_frac: float, rng: random.Random):
    """
    • 10% of patients → fixed TEST set (identical in every fold)
    • Remaining 90% → split k times into
        – VAL  = val_frac of ALL patients (here 0.10)
        – TRAIN = the other ≈80%
    • Patient-level stratification by label guarantees no empty classes.
    """

    # 1) collapse to one row per patient
    patient_tbl = (
        df.groupby('case_id')
          .agg({'slide_id': list, 'label': 'first'})
          .reset_index()
    )
    pats = patient_tbl['case_id'].values
    labs = patient_tbl['label'].astype(int).values

    # 2) single outer split → TRAIN_VAL pool / TEST (10%)
    outer = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_frac,
        random_state=rng.randint(0, 2**31 - 1)
    )
    pool_idx, test_idx = next(outer.split(pats, labs))
    pool_pats = pats[pool_idx]   # 90% of patients
    pool_labs = labs[pool_idx]
    test_pats = pats[test_idx]   # 10% of patients

    # 3) inside the 90% pool, draw k *independent* VAL sets so that
    #    each VAL = val_frac of ALL patients
    val_size = val_frac / (1 - test_frac)  # 0.10 / 0.90 ≈ 0.111…
    folds = []
    for _ in range(k):
        inner = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_size,
            random_state=rng.randint(0, 2**31 - 1)
        )
        train_idx, val_idx = next(inner.split(pool_pats, pool_labs))
        train_pats = pool_pats[train_idx]
        val_pats   = pool_pats[val_idx]

        split = defaultdict(list)
        # expand patient → slide_ids
        for p in train_pats:
            split['train'].extend(df.loc[df.case_id == p, 'slide_id'])
        for p in val_pats:
            split['val'].extend(df.loc[df.case_id == p, 'slide_id'])
        for p in test_pats:
            split['test'].extend(df.loc[df.case_id == p, 'slide_id'])

        folds.append(split)

    return folds

# ────────────────────────────────────────────────────────────────────────────────
def save_split(split: dict, path: Path):
    # pad columns to equal length so pandas writes rectangular CSV
    max_len = max(len(v) for v in split.values())
    data = {k: v + [''] * (max_len - len(v)) for k, v in split.items()}
    pd.DataFrame(data, dtype=str).to_csv(path, index=False)

# ────────────────────────────────────────────────────────────────────────────────
def main():
    args = build_parser().parse_args()
    rng  = random.Random(args.seed)

    df = pd.read_csv(args.csv_path, dtype=str)
    assert {'case_id', 'slide_id', 'label'}.issubset(df.columns), \
        "csv must contain case_id, slide_id, label columns"

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    folds = make_splits(df, args.k, args.val_frac, args.test_frac, rng)

    for i, split in enumerate(folds):
        save_split(split, Path(args.out_dir) / f'splits_{i}.csv')
    print(f"✔ wrote {args.k} folds to {args.out_dir}")

if __name__ == "__main__":
    main()
