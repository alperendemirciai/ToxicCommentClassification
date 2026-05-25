"""Run a local LLM (via Ollama) on all 5 folds in either zero-shot or few-shot mode.

Usage:
    python scripts/03_run_llm.py --model llama3 --mode zero
    python scripts/03_run_llm.py --model llama3 --mode few
    python scripts/03_run_llm.py --model qwen   --mode zero
    python scripts/03_run_llm.py --model qwen   --mode few
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import (
    LABELS,
    LLM_FEW_SHOT_K,
    LLM_REGISTRY,
    LLM_THINK,
    LOGS_DIR,
    RESULTS_DIR,
    SEED,
)
from src.data import iter_folds, load_subsample
from src.llm_model import LLMPrediction, ensure_model_available, predict_one
from src.metrics import aggregate_folds, compute_metrics
from src.prompts import (
    build_few_shot_messages,
    build_zero_shot_messages,
    render_messages_as_text,
    select_few_shot_examples,
)
from src.utils import set_seed, setup_logger, write_json


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(LLM_REGISTRY.keys()))
    ap.add_argument("--mode", required=True, choices=["zero", "few"])
    ap.add_argument("--only-fold", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="optional cap on examples per fold (smoke testing)")
    think_group = ap.add_mutually_exclusive_group()
    think_group.add_argument("--think", dest="think", action="store_true",
                             help="enable reasoning (chain-of-thought) for reasoning-capable models")
    think_group.add_argument("--no-think", dest="think", action="store_false",
                             help="disable reasoning (default for reasoning models)")
    ap.set_defaults(think=LLM_THINK)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)

    model_key = args.model
    model_name = LLM_REGISTRY[model_key]
    mode_dir = "zero_shot" if args.mode == "zero" else "few_shot"
    # Route think/no-think runs to distinct subdirs so they don't overwrite.
    think_tag = "think" if args.think else "nothink"
    out_root = RESULTS_DIR / f"llm_{model_key}" / mode_dir / think_tag
    out_root.mkdir(parents=True, exist_ok=True)

    log = setup_logger(
        f"llm_{model_key}_{mode_dir}_{think_tag}",
        LOGS_DIR / f"03_llm_{model_key}_{mode_dir}_{think_tag}.log",
    )
    log.info(f"Model: {model_name} ({mode_dir}, think={args.think})  output: {out_root}")
    ensure_model_available(model_name)
    log.info("Ollama server reachable; model is pulled.")

    df = load_subsample()
    fold_metrics: list[dict] = []

    for fold_idx, train_df, _val_df, test_df in iter_folds(df):
        if args.only_fold is not None and fold_idx != args.only_fold:
            continue
        fold_dir = out_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        examples = []
        if args.mode == "few":
            examples = select_few_shot_examples(train_df)
            # Persist the selected exemplars for the report's appendix.
            ex_records = [{
                "comment_text": ex.text,
                **{lbl: ex.labels[lbl] for lbl in LABELS},
            } for ex in examples]
            pd.DataFrame(ex_records).to_csv(fold_dir / "few_shot_examples.csv", index=False)

        test_iter = test_df if args.limit is None else test_df.head(args.limit)
        n = len(test_iter)
        log.info(f"=== fold {fold_idx}  test_size={n}  exemplars={len(examples)} ===")

        preds_matrix = np.zeros((n, len(LABELS)), dtype=int)
        raw_responses: list[str] = []
        parse_failures = 0
        latencies: list[float] = []
        t0 = time.time()

        # Save the exact prompt used (with a placeholder comment) for the report.
        if args.mode == "zero":
            sample_messages = build_zero_shot_messages("<<COMMENT_TEXT>>")
        else:
            sample_messages = build_few_shot_messages("<<COMMENT_TEXT>>", examples)
        (fold_dir / "prompt_template.txt").write_text(render_messages_as_text(sample_messages))

        for i in range(n):
            comment = test_iter.iloc[i]["comment_text"]
            if args.mode == "zero":
                messages = build_zero_shot_messages(comment)
            else:
                messages = build_few_shot_messages(comment, examples)

            try:
                pred: LLMPrediction = predict_one(model_name, messages, think=args.think)
            except Exception as e:                                  # network / timeout etc.
                log.warning(f"  fold {fold_idx} idx {i} call failed: {e!r}")
                pred = LLMPrediction(labels={lbl: 0 for lbl in LABELS},
                                     raw=f"<<error: {e!r}>>",
                                     parse_failure=True, latency_seconds=0.0)

            for j, lbl in enumerate(LABELS):
                preds_matrix[i, j] = pred.labels[lbl]
            raw_responses.append(pred.raw)
            latencies.append(pred.latency_seconds)
            if pred.parse_failure:
                parse_failures += 1

            if (i + 1) % 50 == 0 or (i + 1) == n:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                eta = (n - i - 1) / max(rate, 1e-6)
                log.info(f"  fold {fold_idx}  {i+1}/{n}  "
                         f"{rate:.2f} req/s  parse_fail={parse_failures}  "
                         f"eta={eta/60:.1f}min")

        elapsed = time.time() - t0
        log.info(f"  fold {fold_idx} inference done in {elapsed/60:.1f} min  "
                 f"parse_failures={parse_failures}/{n}")

        # Save predictions.csv: id, true_*, pred_*, raw_response.
        pred_df = pd.DataFrame({"id": test_iter["id"].values, "raw_response": raw_responses})
        for j, lbl in enumerate(LABELS):
            pred_df[f"true_{lbl}"] = test_iter[lbl].values
            pred_df[f"pred_{lbl}"] = preds_matrix[:, j]
        pred_df.to_csv(fold_dir / "predictions.csv", index=False)

        metrics = compute_metrics(test_iter[LABELS].values, preds_matrix)
        write_json(fold_dir / "metrics.json", {
            "fold": fold_idx,
            "model": model_name,
            "mode": args.mode,
            "n_test": n,
            "parse_failures": parse_failures,
            "parse_failure_rate": parse_failures / max(n, 1),
            "mean_latency_seconds": float(np.mean(latencies)) if latencies else 0.0,
            "total_inference_seconds": elapsed,
            **metrics,
        })
        log.info(f"  fold {fold_idx}: macro_f1={metrics['macro_f1']:.4f}  "
                 f"micro_f1={metrics['micro_f1']:.4f}  parse_fail_rate="
                 f"{parse_failures/max(n,1):.3f}")
        fold_metrics.append(metrics)

    if args.only_fold is None and args.limit is None:
        summary = aggregate_folds(fold_metrics)
        summary["model"] = model_name
        summary["mode"] = args.mode
        write_json(out_root / "summary.json", summary)
        log.info(
            f"DONE  macro_f1 mean±std = {summary['macro_f1_mean']:.4f} ± "
            f"{summary['macro_f1_std']:.4f}  micro_f1 mean±std = "
            f"{summary['micro_f1_mean']:.4f} ± {summary['micro_f1_std']:.4f}"
        )


if __name__ == "__main__":
    main()
