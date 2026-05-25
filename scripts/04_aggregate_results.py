"""Aggregate per-fold metrics across all completed runs into unified tables.

Walks results/ for any directory containing per-fold metrics.json files and
emits:
  - tables/per_fold_results.{csv,md}: every (model, mode, fold, label) row.
  - tables/comparison_table.{csv,md}: model × mode summary (mean±std).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pandas as pd

from src.config import LABELS, RESULTS_DIR, TABLES_DIR
from src.metrics import aggregate_folds
from src.utils import setup_logger, write_json


def _discover_runs() -> list[dict]:
    """Find every completed run as (model_key, mode_label, run_dir).

    run_dir contains fold_*/metrics.json files. mode_label is one of
    'finetune', 'zero_shot', 'few_shot'.
    """
    runs = []
    if not RESULTS_DIR.exists():
        return runs

    for model_dir in sorted(RESULTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        # BERT runs: results/bert_<key>/fold_*/metrics.json directly.
        if model_dir.name.startswith("bert_"):
            if list(model_dir.glob("fold_*/metrics.json")):
                runs.append({
                    "model_key": model_dir.name,
                    "mode": "finetune",
                    "run_dir": model_dir,
                })
        # LLM runs: results/llm_<key>/{zero_shot,few_shot}/fold_*/metrics.json.
        elif model_dir.name.startswith("llm_"):
            for sub in sorted(model_dir.iterdir()):
                if sub.is_dir() and list(sub.glob("fold_*/metrics.json")):
                    runs.append({
                        "model_key": model_dir.name,
                        "mode": sub.name,
                        "run_dir": sub,
                    })
    return runs


def _load_fold_metrics(run_dir: Path) -> list[dict]:
    fold_files = sorted(run_dir.glob("fold_*/metrics.json"))
    out = []
    for f in fold_files:
        with open(f) as h:
            out.append(json.load(h))
    return out


def _per_fold_rows(model_key: str, mode: str, fold_metrics: list[dict]) -> list[dict]:
    rows = []
    for fm in fold_metrics:
        for lbl in LABELS:
            pl = fm["per_label"][lbl]
            rows.append({
                "model": model_key, "mode": mode, "fold": fm["fold"],
                "label": lbl,
                "precision": round(pl["precision"], 4),
                "recall":    round(pl["recall"], 4),
                "f1":        round(pl["f1"], 4),
                "support":   pl["support"],
            })
        rows.append({
            "model": model_key, "mode": mode, "fold": fm["fold"],
            "label": "MACRO_F1",
            "precision": "", "recall": "",
            "f1": round(fm["macro_f1"], 4), "support": "",
        })
        rows.append({
            "model": model_key, "mode": mode, "fold": fm["fold"],
            "label": "MICRO_F1",
            "precision": "", "recall": "",
            "f1": round(fm["micro_f1"], 4), "support": "",
        })
    return rows


def _summary_row(model_key: str, mode: str, fold_metrics: list[dict]) -> dict:
    agg = aggregate_folds(fold_metrics)
    row = {
        "model": model_key, "mode": mode, "n_folds": len(fold_metrics),
        "macro_f1_mean": round(agg["macro_f1_mean"], 4),
        "macro_f1_std":  round(agg["macro_f1_std"], 4),
        "micro_f1_mean": round(agg["micro_f1_mean"], 4),
        "micro_f1_std":  round(agg["micro_f1_std"], 4),
    }
    for lbl in LABELS:
        pl = agg["per_label"][lbl]
        row[f"{lbl}_f1_mean"] = round(pl["f1_mean"], 4)
        row[f"{lbl}_f1_std"]  = round(pl["f1_std"], 4)
    return row


def _mean_std(metric_lists: list[list[float]]) -> tuple[float, float]:
    arr = np.array(metric_lists)
    return float(arr.mean()), float(arr.std(ddof=0))


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logger("aggregate", None)

    runs = _discover_runs()
    if not runs:
        log.warning("No completed runs found under results/.")
        return

    per_fold_rows: list[dict] = []
    summary_rows: list[dict] = []
    for r in runs:
        fm = _load_fold_metrics(r["run_dir"])
        log.info(f"{r['model_key']}/{r['mode']}: {len(fm)} folds")
        per_fold_rows.extend(_per_fold_rows(r["model_key"], r["mode"], fm))
        if len(fm) > 0:
            summary_rows.append(_summary_row(r["model_key"], r["mode"], fm))
            # Also write/refresh summary.json inside the run dir.
            agg = aggregate_folds(fm)
            agg["model_key"] = r["model_key"]
            agg["mode"] = r["mode"]
            write_json(r["run_dir"] / "summary.json", agg)

    pf = pd.DataFrame(per_fold_rows)
    pf_csv = TABLES_DIR / "per_fold_results.csv"
    pf_md = TABLES_DIR / "per_fold_results.md"
    pf.to_csv(pf_csv, index=False)
    with open(pf_md, "w") as h:
        h.write("# Per-Fold Per-Label Results (all runs)\n\n")
        h.write(pf.to_markdown(index=False))
        h.write("\n")
    log.info(f"Wrote {pf_csv}, {pf_md}  ({len(pf)} rows)")

    sm = pd.DataFrame(summary_rows)
    sm_csv = TABLES_DIR / "comparison_table.csv"
    sm_md = TABLES_DIR / "comparison_table.md"
    sm.to_csv(sm_csv, index=False)

    # A more readable comparison MD: pretty mean±std format.
    display = pd.DataFrame()
    display["model"] = sm["model"]
    display["mode"] = sm["mode"]
    display["macro_F1"] = sm.apply(lambda r: f"{r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f}", axis=1)
    display["micro_F1"] = sm.apply(lambda r: f"{r['micro_f1_mean']:.4f} ± {r['micro_f1_std']:.4f}", axis=1)
    for lbl in LABELS:
        display[f"{lbl}_F1"] = sm.apply(
            lambda r: f"{r[f'{lbl}_f1_mean']:.4f} ± {r[f'{lbl}_f1_std']:.4f}", axis=1
        )
    with open(sm_md, "w") as h:
        h.write("# Comparison Table (5-fold mean ± std)\n\n")
        h.write(display.to_markdown(index=False))
        h.write("\n")
    log.info(f"Wrote {sm_csv}, {sm_md}  ({len(sm)} runs)")

    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
