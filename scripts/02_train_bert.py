"""Train and evaluate a BERT-family model across all 5 folds.

Usage:
    python scripts/02_train_bert.py --model distilbert
    python scripts/02_train_bert.py --model roberta
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from src.config import (
    BERT_MAX_LEN,
    BERT_MAX_LEN_FALLBACK,
    BERT_USE_WEIGHTED_LOSS,
    LABELS,
    LOGS_DIR,
    MODEL_REGISTRY,
    RESULTS_DIR,
    SEED,
)
from src.bert_model import TrainConfig, train_one_fold
from src.data import iter_folds, load_subsample
from src.metrics import aggregate_folds
from src.utils import set_seed, setup_logger, write_json


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--max-len", type=int, default=BERT_MAX_LEN)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--only-fold", type=int, default=None, help="run only this fold idx")
    w_group = ap.add_mutually_exclusive_group()
    w_group.add_argument("--weighted", dest="weighted", action="store_true",
                         help="enable BCE pos_weight to counter label imbalance")
    w_group.add_argument("--no-weighted", dest="weighted", action="store_false")
    ap.set_defaults(weighted=BERT_USE_WEIGHTED_LOSS)
    ap.add_argument("--tag", default=None,
                    help="optional suffix appended to results/bert_<model> output dir")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)

    model_key = args.model
    model_name = MODEL_REGISTRY[model_key]
    suffix = ""
    if args.tag:
        suffix = f"_{args.tag}"
    elif args.weighted:
        suffix = "_weighted"
    out_root = RESULTS_DIR / f"bert_{model_key}{suffix}"
    out_root.mkdir(parents=True, exist_ok=True)
    log = setup_logger(f"bert_{model_key}{suffix}", LOGS_DIR / f"02_bert_{model_key}{suffix}.log")
    log.info(f"Model: {model_name}  output: {out_root}")
    log.info(f"max_len={args.max_len}  epochs={args.epochs}  batch_size={args.batch_size}  weighted={args.weighted}")

    df = load_subsample()
    fold_metrics: list[dict] = []

    for fold_idx, train_df, val_df, test_df in iter_folds(df):
        if args.only_fold is not None and fold_idx != args.only_fold:
            continue
        fold_dir = out_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"=== fold {fold_idx}  train={len(train_df)}  val={len(val_df)}  test={len(test_df)} ===")

        # Build a TrainConfig with overrides if provided.
        cfg_kwargs = dict(model_name=model_name, output_dir=fold_dir, max_len=args.max_len,
                          seed=SEED, use_weighted_loss=args.weighted)
        if args.epochs is not None:
            cfg_kwargs["epochs"] = args.epochs
        if args.batch_size is not None:
            cfg_kwargs["batch_size"] = args.batch_size

        # Try requested max_len; if OOM, fall back once to a shorter length.
        try:
            cfg = TrainConfig(**cfg_kwargs)
            result = train_one_fold(train_df, val_df, test_df, cfg, log=log.info)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            fallback = BERT_MAX_LEN_FALLBACK
            log.warning(f"  OOM at max_len={cfg_kwargs['max_len']} — retrying at {fallback}")
            cfg_kwargs["max_len"] = fallback
            cfg = TrainConfig(**cfg_kwargs)
            result = train_one_fold(train_df, val_df, test_df, cfg, log=log.info)

        # Persist predictions (with probs and raw text id) and metrics.
        pred_df = pd.DataFrame({"id": test_df["id"].values})
        for i, lbl in enumerate(LABELS):
            pred_df[f"true_{lbl}"] = test_df[lbl].values
            pred_df[f"prob_{lbl}"] = result["probs"][:, i]
            pred_df[f"pred_{lbl}"] = result["preds"][:, i]
        pred_df.to_csv(fold_dir / "predictions.csv", index=False)

        write_json(fold_dir / "metrics.json", {
            "fold": fold_idx,
            "model": model_name,
            "max_len": cfg.max_len,
            "epochs_run": cfg.epochs,
            "best_epoch": result["best_epoch"],
            "best_val_macro_f1": result["best_val_macro_f1"],
            "train_time_seconds": result["train_time_seconds"],
            "history": result["history"],
            **result["metrics"],
        })
        log.info(f"  fold {fold_idx}: macro_f1={result['metrics']['macro_f1']:.4f}  "
                 f"micro_f1={result['metrics']['micro_f1']:.4f}")
        fold_metrics.append(result["metrics"])

    if args.only_fold is None:
        summary = aggregate_folds(fold_metrics)
        summary["model"] = model_name
        write_json(out_root / "summary.json", summary)
        log.info(
            f"DONE  macro_f1 mean±std = {summary['macro_f1_mean']:.4f} ± "
            f"{summary['macro_f1_std']:.4f}  micro_f1 mean±std = "
            f"{summary['micro_f1_mean']:.4f} ± {summary['micro_f1_std']:.4f}"
        )


if __name__ == "__main__":
    main()
