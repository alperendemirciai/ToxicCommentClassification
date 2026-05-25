"""Data preparation: subsampling and multi-label stratified K-fold splits."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from src.config import (
    FOLDS_JSON,
    INNER_VAL_RATIO,
    LABELS,
    N_FOLDS,
    ORIGINAL_CSV,
    SEED,
    SUBSAMPLE_CSV,
    SUBSAMPLE_SIZE,
)
from src.utils import read_json, write_json


def load_full() -> pd.DataFrame:
    """Load the full Jigsaw training CSV (159,571 rows)."""
    df = pd.read_csv(ORIGINAL_CSV)
    assert set(LABELS).issubset(df.columns), f"Missing label columns. Got {df.columns}"
    assert "comment_text" in df.columns and "id" in df.columns
    return df


def stratified_subsample(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """Draw an exactly-n-row, label-distribution-preserving subsample.

    Strategy:
      1. Pick the first fold of a k-fold MultilabelStratifiedKFold whose fold
         size is closest to n. This yields a multi-label-stratified bucket.
      2. If that bucket is too small, top up by drawing additional rows from
         the remainder via a second stratified split sized to produce the
         needed delta.
      3. If too large, trim by another stratified split that produces an
         exactly-n bucket from the oversized bucket.
    """
    if len(df) <= n:
        return df.reset_index(drop=True).copy()

    y = df[LABELS].values
    n_splits = max(2, round(len(df) / n))
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _train_idx, test_idx = next(mskf.split(np.zeros(len(df)), y))

    if len(test_idx) == n:
        chosen = test_idx
    elif len(test_idx) < n:
        deficit = n - len(test_idx)
        remainder_idx = np.setdiff1d(np.arange(len(df)), test_idx, assume_unique=False)
        y_rem = y[remainder_idx]
        # Pick a k so that one fold-size ~ deficit.
        rem_splits = max(2, round(len(remainder_idx) / deficit))
        rem_mskf = MultilabelStratifiedKFold(n_splits=rem_splits, shuffle=True, random_state=seed + 1)
        _t, extra_local = next(rem_mskf.split(np.zeros(len(remainder_idx)), y_rem))
        # Trim if extra is too long.
        extra = remainder_idx[extra_local[:deficit]]
        chosen = np.concatenate([test_idx, extra])
    else:                                                       # too large -> stratified trim
        excess = len(test_idx) - n
        y_sub = y[test_idx]
        trim_splits = max(2, round(len(test_idx) / (len(test_idx) - n)))
        trim_mskf = MultilabelStratifiedKFold(n_splits=trim_splits, shuffle=True, random_state=seed + 2)
        # Drop one fold's worth (~excess rows). Trim local indices stratified.
        _keep_local, drop_local = next(trim_mskf.split(np.zeros(len(test_idx)), y_sub))
        drop_local = drop_local[:excess]
        keep_mask = np.ones(len(test_idx), dtype=bool)
        keep_mask[drop_local] = False
        chosen = test_idx[keep_mask]

    sub = df.iloc[chosen].reset_index(drop=True).copy()
    assert len(sub) == n, f"subsample size mismatch: got {len(sub)} expected {n}"
    return sub


def build_folds(df: pd.DataFrame, n_splits: int = N_FOLDS, seed: int = SEED) -> dict:
    """Build n_splits multilabel-stratified folds. Returns {fold_idx: {test_idx, train_idx, val_idx}}.

    Inside each fold's training partition, a further stratified 1-fold split
    pulls out ~INNER_VAL_RATIO rows for validation (best-epoch selection).
    """
    y = df[LABELS].values
    outer = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: dict = {}

    for fold_idx, (train_idx, test_idx) in enumerate(outer.split(np.zeros(len(df)), y)):
        # Inner train/val split via MultilabelStratifiedKFold with k = round(1/ratio).
        inner_k = max(2, round(1.0 / INNER_VAL_RATIO))
        inner = MultilabelStratifiedKFold(n_splits=inner_k, shuffle=True, random_state=seed + fold_idx)
        y_train = y[train_idx]
        inner_train_idx, inner_val_idx = next(inner.split(np.zeros(len(train_idx)), y_train))
        folds[str(fold_idx)] = {
            "train_idx": train_idx[inner_train_idx].tolist(),
            "val_idx":   train_idx[inner_val_idx].tolist(),
            "test_idx":  test_idx.tolist(),
        }
    return folds


def save_folds(folds: dict, path: Path = FOLDS_JSON) -> None:
    write_json(path, folds)


def load_folds(path: Path = FOLDS_JSON) -> dict:
    return read_json(path)


def load_subsample() -> pd.DataFrame:
    return pd.read_csv(SUBSAMPLE_CSV)


def label_frequency_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-label positive count and proportion."""
    counts = df[LABELS].sum().astype(int)
    props = (counts / len(df)).round(6)
    out = pd.DataFrame({"label": LABELS, "count": counts.values, "proportion": props.values})
    return out


def iter_folds(df: pd.DataFrame) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Yield (fold_idx, train_df, val_df, test_df) tuples in order."""
    folds = load_folds()
    for fold_idx_str in sorted(folds.keys(), key=int):
        fold_idx = int(fold_idx_str)
        f = folds[fold_idx_str]
        train_df = df.iloc[f["train_idx"]].reset_index(drop=True)
        val_df = df.iloc[f["val_idx"]].reset_index(drop=True)
        test_df = df.iloc[f["test_idx"]].reset_index(drop=True)
        yield fold_idx, train_df, val_df, test_df
