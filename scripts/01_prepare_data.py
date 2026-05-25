"""Build the 8K stratified subsample, the 5-fold split definitions, and the
label-distribution table. Run once at the start of the project; folds are
then reused identically by BERT and LLM evaluation runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/01_prepare_data.py` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import (
    FOLDS_JSON,
    LABELS,
    LOGS_DIR,
    SEED,
    SPLITS_DIR,
    SUBSAMPLE_CSV,
    SUBSAMPLE_SIZE,
    TABLES_DIR,
)
from src.data import (
    build_folds,
    label_frequency_table,
    load_full,
    save_folds,
    stratified_subsample,
)
from src.utils import set_seed, setup_logger, write_json


def main() -> None:
    set_seed(SEED)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logger("prepare_data", LOGS_DIR / "01_prepare_data.log")

    log.info("Loading full Jigsaw training CSV...")
    full = load_full()
    log.info(f"Full dataset: {len(full):,} rows")

    full_freq = label_frequency_table(full)
    log.info("Full label distribution:\n" + full_freq.to_string(index=False))

    log.info(f"Drawing {SUBSAMPLE_SIZE}-row stratified subsample (seed={SEED})...")
    sub = stratified_subsample(full, n=SUBSAMPLE_SIZE, seed=SEED)
    assert len(sub) == SUBSAMPLE_SIZE, f"got {len(sub)}"
    sub.to_csv(SUBSAMPLE_CSV, index=False)
    log.info(f"Subsample saved -> {SUBSAMPLE_CSV} ({len(sub):,} rows)")

    sub_freq = label_frequency_table(sub)
    log.info("Subsample label distribution:\n" + sub_freq.to_string(index=False))

    # Save label-frequency comparison table (csv + md).
    comp = pd.DataFrame({
        "label": LABELS,
        "full_count":      full_freq["count"].values,
        "full_proportion": full_freq["proportion"].values,
        "sub_count":       sub_freq["count"].values,
        "sub_proportion":  sub_freq["proportion"].values,
    })
    comp_csv = TABLES_DIR / "label_distribution.csv"
    comp_md = TABLES_DIR / "label_distribution.md"
    comp.to_csv(comp_csv, index=False)
    with open(comp_md, "w") as f:
        f.write("# Label Distribution: Full vs 8K Subsample\n\n")
        f.write(comp.to_markdown(index=False))
        f.write("\n")
    log.info(f"Label-distribution table -> {comp_csv}, {comp_md}")

    log.info("Building 5-fold multilabel-stratified splits...")
    folds = build_folds(sub)
    for k, v in folds.items():
        n_tr, n_val, n_te = len(v["train_idx"]), len(v["val_idx"]), len(v["test_idx"])
        log.info(f"  fold {k}: train={n_tr}  val={n_val}  test={n_te}")
    save_folds(folds, FOLDS_JSON)
    log.info(f"Folds saved -> {FOLDS_JSON}")

    # Sanity: per-fold test label distribution to confirm stratification quality.
    sanity_rows = []
    for k, v in folds.items():
        test_sub = sub.iloc[v["test_idx"]]
        row = {"fold": k, "n_test": len(test_sub)}
        for lbl in LABELS:
            row[f"{lbl}_pos"] = int(test_sub[lbl].sum())
        sanity_rows.append(row)
    sanity = pd.DataFrame(sanity_rows)
    sanity_path = TABLES_DIR / "fold_label_distribution.csv"
    sanity.to_csv(sanity_path, index=False)
    log.info(f"Per-fold test label counts -> {sanity_path}")
    log.info("Per-fold test label counts:\n" + sanity.to_string(index=False))


if __name__ == "__main__":
    main()
