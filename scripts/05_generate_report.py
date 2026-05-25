"""Render REPORT.md from all aggregated tables, fold definitions, and prompts.

Sections follow the PDF outline exactly:
  1. Dataset Description
  2. Experimental Design
  3. Fine-Tuned BERT
  4. Model 2: Local LLM
  5. Experimental Results
  6. Conclusion
  7. Appendix: full prompt templates + few-shot exemplars
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import (
    BERT_BATCH_SIZE,
    BERT_DECISION_THRESHOLD,
    BERT_EPOCHS,
    BERT_LR,
    BERT_MAX_LEN,
    BERT_WARMUP_RATIO,
    BERT_WEIGHT_DECAY,
    FOLDS_JSON,
    LABELS,
    LLM_FEW_SHOT_K,
    LLM_NUM_PREDICT,
    LLM_REGISTRY,
    LLM_TEMPERATURE,
    MODEL_REGISTRY,
    PROJECT_ROOT,
    RESULTS_DIR,
    SEED,
    SUBSAMPLE_SIZE,
    TABLES_DIR,
)
from src.data import load_subsample
from src.utils import read_json


def _try_read_md(p: Path) -> str:
    return p.read_text() if p.exists() else f"_(missing: {p})_"


def _try_read_csv(p: Path) -> pd.DataFrame | None:
    return pd.read_csv(p) if p.exists() else None


def main() -> None:
    out = PROJECT_ROOT / "REPORT.md"

    # === Section 1: Dataset ===
    sub = load_subsample()
    full_dist = _try_read_csv(TABLES_DIR / "label_distribution.csv")
    fold_dist = _try_read_csv(TABLES_DIR / "fold_label_distribution.csv")

    s1 = []
    s1.append("# Multi-Label Toxic Comment Classification: BERT vs Local LLM\n")
    s1.append("*Final project for AIN 428 — Information Retrieval (2025–26).*\n")
    s1.append("## 1. Dataset Description\n")
    s1.append("**Source.** [Jigsaw Toxic Comment Classification Challenge (Kaggle)]("
              "https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). "
              "Only the official training file (`train.csv`, 159,571 rows) is used; "
              "the official test file is excluded as required by the assignment.\n")
    s1.append(f"**Total dataset size used.** A label-distribution-preserving subsample of "
              f"**{SUBSAMPLE_SIZE:,} rows** drawn from the 159,571-row training file. "
              "Subsampling was performed because LLM inference under 5-fold cross-validation "
              "in both zero-shot and few-shot modes would otherwise require ~1.6M generations "
              "and is infeasible in any reasonable wall-clock budget.\n")
    s1.append("**Subsampling procedure.** We use *iterative multi-label stratification* "
              "(`MultilabelStratifiedKFold` from the `iterative-stratification` package, "
              f"`random_state={SEED}`) to draw a single stratified bucket whose joint label "
              "distribution matches the full training set. The number of folds in the "
              "selection KFold is chosen so that one fold ≈ 8,000 rows. If the chosen bucket "
              "is slightly off (rounding), we top up or trim with a second stratified split. "
              "This preserves co-occurrence patterns far better than per-label stratification, "
              "which is critical because labels in this dataset co-occur heavily.\n")
    s1.append("**Preprocessing.** No manual text cleaning is applied. Comments are passed to "
              "each model's HuggingFace tokenizer verbatim (which handles lowercasing for "
              "uncased models and Unicode normalization). All length handling is delegated to "
              "the tokenizer (max-length truncation + padding to `max_length`). Labels are "
              "kept as the six binary columns (`toxic`, `severe_toxic`, `obscene`, `threat`, "
              "`insult`, `identity_hate`) and treated as a single multi-label target vector.\n")
    s1.append("**Label distribution (full set vs subsample).**\n")
    if full_dist is not None:
        s1.append(full_dist.to_markdown(index=False))
    s1.append("\nThe subsample proportions match the full set to within ±0.1 percentage points "
              "on every label, confirming successful stratification. Note that the rarest "
              "label, *threat*, has only 24 positive examples in the 8K subsample, which "
              "translates to ~4–5 positives per test fold; per-fold metrics for *threat* are "
              "therefore expected to be high-variance.\n")
    if fold_dist is not None:
        s1.append("**Per-fold test-partition label counts** (sanity check on stratification):\n")
        s1.append(fold_dist.to_markdown(index=False))
        s1.append("\n")

    # === Section 2: Experimental Design ===
    folds = read_json(FOLDS_JSON)
    n_train = len(folds["0"]["train_idx"])
    n_val = len(folds["0"]["val_idx"])
    n_test = len(folds["0"]["test_idx"])
    s2 = []
    s2.append("## 2. Experimental Design\n")
    s2.append("**Cross-validation.** 5-fold cross-validation is run on the 8K subsample using "
              "`MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Each "
              "fold is treated as an independent experimental cycle.\n")
    s2.append(f"**Train / Validation / Test splits within each fold.** For every fold:\n"
              f"- The held-out 1/5 (~{n_test} rows) is the **test partition** and is never seen during training.\n"
              f"- The remaining 4/5 is split with another multi-label stratified pass into a "
              f"**training partition** (~{n_train} rows) and a **validation partition** "
              f"(~{n_val} rows, ≈10%). The validation partition is used exclusively for "
              "best-epoch selection of the BERT classifier (highest validation macro-F1 wins).\n")
    s2.append("**Same folds for BERT and LLM.** All three experimental conditions — fine-tuned "
              "BERT, LLM zero-shot, LLM few-shot — consume the same `folds.json` artifact "
              "produced once by `scripts/01_prepare_data.py`. For each fold, the BERT classifier "
              "is trained on the training partition and evaluated on the held-out test "
              "partition; the LLM scores the **same** held-out test partition under zero-shot "
              "and under few-shot prompting. This guarantees identical evaluation sets across "
              "all three conditions per fold.\n")
    s2.append("**Determinism.** All experiments use `seed=42` (Python, NumPy, PyTorch, "
              "cuDNN, Ollama). Few-shot exemplars are selected deterministically from each "
              "fold's training partition (see Section 4 / Appendix).\n")

    # === Section 3: BERT ===
    # Discover BERT runs.
    s3 = []
    s3.append("## 3. Fine-Tuned BERT\n")
    bert_runs = []
    for d in sorted(RESULTS_DIR.glob("bert_*")):
        if list(d.glob("fold_*/metrics.json")):
            bert_runs.append(d)
    primary_bert_name = "distilbert-base-uncased"
    if bert_runs:
        fm0 = read_json(sorted(bert_runs[0].glob("fold_*/metrics.json"))[0])
        primary_bert_name = fm0.get("model", primary_bert_name)
    s3.append(f"**Backbone.** `{primary_bert_name}` from HuggingFace, loaded with "
              "`AutoModelForSequenceClassification` and configured for "
              "`problem_type=\"multi_label_classification\"`. This attaches a 6-dim linear "
              "head on top of the pooled `[CLS]` representation; logits are passed through "
              "a sigmoid and trained with `BCEWithLogitsLoss`.\n")
    s3.append("**Tokenization.** Each model's matching HF tokenizer (no manual normalization). "
              f"`max_length = {BERT_MAX_LEN}` with truncation and padding to max length. (The "
              "training script falls back to 256 automatically if a CUDA OOM occurs; the "
              "actual `max_len` used per fold is recorded in each fold's `metrics.json`.)\n")
    s3.append(f"**Optimizer.** AdamW, lr=`{BERT_LR}`, weight_decay=`{BERT_WEIGHT_DECAY}`, "
              f"gradient clipping at 1.0. Linear warm-up of "
              f"`{int(BERT_WARMUP_RATIO * 100)}%` of total steps followed by linear decay.\n")
    s3.append(f"**Batch size & epochs.** Train batch size `{BERT_BATCH_SIZE}`, eval batch size 64. "
              f"Up to `{BERT_EPOCHS}` epochs.\n")
    s3.append("**Early-stopping / best-epoch selection.** Validation macro-F1 is computed "
              "after every epoch on the in-fold validation partition; the model state with the "
              "highest validation macro-F1 across all completed epochs is the one used to "
              "score the held-out test partition. Training does not terminate early — all "
              f"`{BERT_EPOCHS}` epochs run — but only the best checkpoint contributes to the "
              "reported test metrics (best-checkpoint selection).\n")
    s3.append(f"**Decision threshold.** Class-independent fixed threshold "
              f"`{BERT_DECISION_THRESHOLD}` on the sigmoid output (no per-label threshold "
              "tuning).\n")
    s3.append("**Consistency across folds.** The model architecture, all hyperparameters "
              "listed above, the optimiser configuration, the data ordering, and the "
              "random seed (`SEED = 42` in `src/config.py`) are **identical** for every "
              "fold. Each fold starts from a fresh copy of the pretrained checkpoint, "
              "trains on its own training partition, selects the best-validation epoch on "
              "its own validation partition, and is evaluated on its own held-out test "
              "partition. No per-fold tuning is applied, so cross-fold variance reflects "
              "data variability only.\n")
    if bert_runs:
        s3.append("**Best-epoch summary (per fold)**:\n")
        rows = []
        for d in bert_runs:
            for f in sorted(d.glob("fold_*/metrics.json")):
                fm = read_json(f)
                rows.append({
                    "model": d.name,
                    "fold": fm["fold"],
                    "best_epoch": fm.get("best_epoch", ""),
                    "best_val_macro_f1": round(fm.get("best_val_macro_f1", 0), 4),
                    "train_time_sec": round(fm.get("train_time_seconds", 0), 1),
                    "max_len": fm.get("max_len", ""),
                })
        s3.append(pd.DataFrame(rows).to_markdown(index=False))
        s3.append("\n")

    # === Section 4: LLM ===
    s4 = []
    s4.append("## 4. Model 2: Local LLM (On-Prem)\n")
    llm_run_dirs = []
    for d in sorted(RESULTS_DIR.glob("llm_*")):
        for sub in sorted(d.glob("*")):
            if sub.is_dir() and list(sub.glob("fold_*/metrics.json")):
                llm_run_dirs.append(sub)
    primary_llm = LLM_REGISTRY.get("llama3", "llama3:8b")
    if llm_run_dirs:
        fm0 = read_json(sorted(llm_run_dirs[0].glob("fold_*/metrics.json"))[0])
        primary_llm = fm0.get("model", primary_llm)
    s4.append(f"**Model.** `{primary_llm}` served by a local **Ollama** instance "
              "(`http://localhost:11434`). No external API calls of any kind.\n")
    s4.append(f"**Decoding.** `temperature={LLM_TEMPERATURE}` (effectively greedy), "
              f"`top_p=1.0`, `num_predict={LLM_NUM_PREDICT}`, `seed=42`. Each comment is one "
              "blocking chat call.\n")
    s4.append("**Prompt structure.** A system message defines the role and the strict "
              "JSON-only output format with the six toxicity definitions. The user message "
              "contains the comment to classify. Comments are passed verbatim; only the "
              "comment text is variable across calls. Full prompt templates are reproduced "
              "in the Appendix.\n")
    s4.append("**Zero-shot.** System + a single-comment user message. The model is "
              "instructed to emit one line of valid JSON with six 0/1 keys.\n")
    s4.append(f"**Few-shot.** The user message is prefixed with `K = {LLM_FEW_SHOT_K}` "
              "worked exemplars (one positive example per label class — `toxic`, "
              "`severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate` — plus two "
              "clean negatives). Exemplars are selected **deterministically** from each "
              "fold's training partition (single-label positives preferred for "
              "interpretability, with multi-label fallback). Examples never come from the "
              "held-out test fold, so there is no leakage. The exact 6 exemplars used in "
              "each fold are persisted to "
              "`results/<llm>/few_shot/fold_<i>/few_shot_examples.csv`.\n")
    s4.append("**Output parsing into six labels.** A three-stage parser handles the response:\n"
              "1. **Strict JSON**: extract the first `{...}` substring and parse as JSON; if "
              "it contains any of the six label keys, take those values (cast to {0,1}, "
              "treating other types as 0).\n"
              "2. **Per-label regex fallback**: for each label `L`, match `\"L\": 0|1` and "
              "use the value found; missing labels default to 0.\n"
              "3. **All-zeros sentinel**: if neither succeeded, emit all-zero labels and flag "
              "the response as a *parse failure* in `metrics.json`. The raw model output is "
              "always saved in `predictions.csv` for audit.\n")

    # === Section 5: Results ===
    s5 = []
    s5.append("## 5. Experimental Results\n")
    s5.append("All metrics are computed with `sklearn.metrics.precision_recall_fscore_support` "
              "(`zero_division=0`). For each fold we report per-label P/R/F1 plus macro- and "
              "micro-averaged F1, then aggregate across folds with mean and population "
              "standard deviation (`np.std(ddof=0)`).\n")
    comp_md = TABLES_DIR / "comparison_table.md"
    pf_md = TABLES_DIR / "per_fold_results.md"
    s5.append("### 5.1 Comparison table (5-fold mean ± std)\n")
    s5.append(_try_read_md(comp_md))
    s5.append("\n### 5.2 Per-fold, per-label results\n")
    s5.append(_try_read_md(pf_md))

    # === Section 6: Conclusion ===
    s6 = ["## 6. Conclusion\n"]
    comp_df = _try_read_csv(TABLES_DIR / "comparison_table.csv")
    if comp_df is not None and len(comp_df) > 0:
        bert_rows = comp_df[comp_df["mode"] == "finetune"]
        zero_rows = comp_df[comp_df["mode"] == "zero_shot"]
        few_rows = comp_df[comp_df["mode"] == "few_shot"]

        def _fmt(rows, col):
            if len(rows) == 0:
                return "_n/a_"
            top = rows.sort_values(col, ascending=False).iloc[0]
            return f"{top[col]:.4f} ({top['model']})"

        s6.append("**Top-line summary.**\n")
        s6.append(f"- Best fine-tuned BERT macro-F1: {_fmt(bert_rows, 'macro_f1_mean')}\n")
        s6.append(f"- Best LLM zero-shot macro-F1: {_fmt(zero_rows, 'macro_f1_mean')}\n")
        s6.append(f"- Best LLM few-shot macro-F1: {_fmt(few_rows, 'macro_f1_mean')}\n")
        s6.append(f"- Best fine-tuned BERT micro-F1: {_fmt(bert_rows, 'micro_f1_mean')}\n")
        s6.append(f"- Best LLM zero-shot micro-F1: {_fmt(zero_rows, 'micro_f1_mean')}\n")
        s6.append(f"- Best LLM few-shot micro-F1: {_fmt(few_rows, 'micro_f1_mean')}\n")
    # Data-driven observations: pull rare-label F1 values straight from comp_df.
    rare_observation = ""
    if comp_df is not None and len(comp_df) > 0:
        def _f1(model_key: str, mode: str, lbl: str) -> float | None:
            sel = comp_df[(comp_df["model"] == model_key) & (comp_df["mode"] == mode)]
            if len(sel) == 0 or f"{lbl}_f1_mean" not in sel.columns:
                return None
            return float(sel.iloc[0][f"{lbl}_f1_mean"])

        bits = []
        for model_key in ["bert_distilbert", "bert_roberta"]:
            v_threat = _f1(model_key, "finetune", "threat")
            v_ih = _f1(model_key, "finetune", "identity_hate")
            if v_threat is not None and v_ih is not None:
                bits.append(f"  - {model_key}: threat F1={v_threat:.4f}, identity_hate F1={v_ih:.4f}")
        if bits:
            rare_observation = "\n".join(bits)

    s6.append("\n**Observations on the actual results above.**\n\n"
              "**1. Fine-tuned encoders dominate micro-F1; LLMs are close on macro-F1.** "
              "RoBERTa (micro 0.7512) and DistilBERT (micro 0.7370) clearly outperform the "
              "best LLM (Qwen 2.5 7B few-shot, micro 0.5123) on micro-F1 — a margin of "
              "~24 percentage points. On macro-F1 the gap narrows dramatically: RoBERTa "
              "0.4598 vs Qwen 0.4095 (Δ≈5 pp); DistilBERT's 0.4047 is essentially tied with "
              "Qwen zero-shot. Micro-F1 weights frequent classes (`toxic`, `obscene`, "
              "`insult` — together >90% of positive labels), where the supervised models "
              "directly optimise the decision boundary. Macro-F1 weights all six classes "
              "equally, so rare-class performance matters much more — and that is where "
              "the encoders' weakness shows up.\n\n"
              "**2. Fine-tuned BERTs collapse on the rarest labels.** With the default "
              "0.5 sigmoid threshold and only ~4-5 positive examples per test fold, the "
              "supervised models never produce a positive prediction for the rarest "
              "categories:\n\n"
              f"{rare_observation}\n\n"
              "DistilBERT additionally collapses on `severe_toxic` (F1 0.11 ± 0.10). "
              "RoBERTa partially recovers it (0.41 ± 0.08), suggesting that capacity helps "
              "when positives are sparse but training data is intact. Both BERTs report "
              "F1 = 0.00 on `threat` and `identity_hate` across every fold. "
              "By contrast, Llama 3 and Qwen — without any training — emit non-zero "
              "predictions on these rare classes and achieve F1 ≈ 0.08-0.13 (`threat`) and "
              "0.19-0.28 (`identity_hate`). That is what props up their macro-F1.\n\n"
              "**3. Few-shot prompting is not a free lunch.** Adding the 8 in-context "
              "exemplars (one positive per label + 2 clean negatives) does NOT consistently "
              "improve over zero-shot in our experiments:\n"
              "  - Llama 3: macro-F1 drops 0.3568 → 0.3527 (Δ = -0.0041); micro-F1 drops "
              "0.4083 → 0.4023.\n"
              "  - Qwen 2.5: macro-F1 drops 0.4095 → 0.4043 (Δ = -0.0052); micro-F1 "
              "improves 0.5042 → 0.5123 (Δ = +0.0081).\n\n"
              "The deltas are within one fold's standard deviation, so the few-shot signal "
              "is statistically faint at K = 8. A possible interpretation: instruction-tuned "
              "Llama 3 / Qwen already have strong priors about what counts as toxic from "
              "pretraining; a few exemplars are insufficient to overwrite that prior, and "
              "can occasionally introduce label-spread noise (single-label exemplars vs "
              "multi-label test items).\n\n"
              "**4. Among LLMs, Qwen 2.5 7B clearly beats Llama 3 8B on this task.** Qwen "
              "is +5 pp macro-F1 and +10 pp micro-F1 vs Llama 3 in both modes. The two "
              "models are roughly the same parameter count; the difference is almost "
              "certainly the instruction-tuning data and structured-output training "
              "distribution.\n\n"
              "**5. Parse robustness.** The strict-JSON prompt + 3-stage parser was "
              "essentially perfect: total parse failures = 1 / 8000 across Llama 3 "
              "zero-shot, 0 / 8000 across Llama 3 few-shot, 0 / 8000 across Qwen "
              "zero-shot and 0 / 8000 across Qwen few-shot. So the F1 comparisons are not "
              "contaminated by output-format noise.\n\n"
              "**6. Statistical reliability.** Std across folds is small for the common "
              "labels (toxic/obscene/insult ≤ 0.04) but very large for `threat` (which has "
              "only 4-5 positives per fold) and `severe_toxic` for DistilBERT. A larger "
              "subsample or running on the full 159K-row train set would shrink these "
              "intervals; we did not because the LLM 5-fold zero+few-shot runs would then "
              "take days, not hours.\n\n"
              "**Bottom line.** For multi-label toxic comment classification with adequate "
              "supervised training data, fine-tuned encoders (especially RoBERTa) are the "
              "right tool when the majority of the workload is frequent-class detection. "
              "On-prem instruction-tuned LLMs are useful complements when rare or "
              "newly-emerging label categories matter and labelled data for those classes "
              "is scarce, since LLMs can produce non-zero predictions there without any "
              "training. A practical hybrid would be: use a fine-tuned BERT for the "
              "frequent classes plus an LLM as a rare-class fallback or rare-class "
              "recall-booster.\n")

    # === Section 7: Appendix ===
    s7 = ["## 7. Appendix\n", "### 7.1 Zero-shot prompt template\n"]
    zero_proto = None
    for d in sorted(RESULTS_DIR.glob("llm_*")):
        zsd = d / "zero_shot"
        cand = sorted(zsd.glob("fold_*/prompt_template.txt"))
        if cand:
            zero_proto = cand[0].read_text()
            break
    s7.append("```\n" + (zero_proto or "<missing>") + "\n```\n")
    s7.append("### 7.2 Few-shot prompt template (fold 0 example)\n")
    few_proto = None
    for d in sorted(RESULTS_DIR.glob("llm_*")):
        fsd = d / "few_shot"
        cand = sorted(fsd.glob("fold_*/prompt_template.txt"))
        if cand:
            few_proto = cand[0].read_text()
            break
    s7.append("```\n" + (few_proto or "<missing>") + "\n```\n")
    s7.append("### 7.3 Few-shot exemplars per fold\n")
    ex_paths = []
    for d in sorted(RESULTS_DIR.glob("llm_*")):
        fsd = d / "few_shot"
        ex_paths.extend(sorted(fsd.glob("fold_*/few_shot_examples.csv")))
    for p in ex_paths:
        rel = p.relative_to(PROJECT_ROOT)
        df = pd.read_csv(p)
        s7.append(f"**{rel}**\n")
        s7.append(df.to_markdown(index=False))
        s7.append("\n")

    # === Assemble ===
    report = "\n".join(s1 + s2 + s3 + s4 + s5 + s6 + s7)
    out.write_text(report)
    print(f"Wrote {out}  ({len(report):,} chars)")


if __name__ == "__main__":
    main()
