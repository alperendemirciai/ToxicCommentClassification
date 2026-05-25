"""Fine-tune a HuggingFace transformer (DistilBERT, BERT, RoBERTa, ...) for
multi-label toxic comment classification. One call = one fold.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.config import (
    BERT_BATCH_SIZE,
    BERT_DECISION_THRESHOLD,
    BERT_EPOCHS,
    BERT_EVAL_BATCH_SIZE,
    BERT_LR,
    BERT_MAX_LEN,
    BERT_WARMUP_RATIO,
    BERT_WEIGHT_DECAY,
    LABELS,
    NUM_LABELS,
    SEED,
)
from src.metrics import compute_metrics
from src.utils import set_seed


class ToxicDataset(Dataset):
    """Tokenize-on-the-fly multi-label dataset."""

    def __init__(self, texts: list[str], labels: np.ndarray | None, tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx] if isinstance(self.texts[idx], str) else ""
        enc = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


@dataclass
class TrainConfig:
    model_name: str
    output_dir: Path
    max_len: int = BERT_MAX_LEN
    batch_size: int = BERT_BATCH_SIZE
    eval_batch_size: int = BERT_EVAL_BATCH_SIZE
    epochs: int = BERT_EPOCHS
    lr: float = BERT_LR
    weight_decay: float = BERT_WEIGHT_DECAY
    warmup_ratio: float = BERT_WARMUP_RATIO
    threshold: float = BERT_DECISION_THRESHOLD
    seed: int = SEED
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _predict_probs(model, loader, device: str) -> np.ndarray:
    model.eval()
    probs_all = []
    sigmoid = nn.Sigmoid()
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        logits = model(**inputs).logits
        probs_all.append(sigmoid(logits).cpu().numpy())
    return np.concatenate(probs_all, axis=0)


def train_one_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: TrainConfig,
    log: Callable[[str], None] = print,
) -> dict:
    """Train a fresh model on train_df, choose best epoch on val_df macro-F1,
    evaluate on test_df. Returns dict with metrics + predictions arrays.
    """
    set_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    ).to(cfg.device)

    train_ds = ToxicDataset(train_df["comment_text"].tolist(),
                            train_df[LABELS].values.astype(np.float32),
                            tokenizer, cfg.max_len)
    val_ds = ToxicDataset(val_df["comment_text"].tolist(),
                          val_df[LABELS].values.astype(np.float32),
                          tokenizer, cfg.max_len)
    test_ds = ToxicDataset(test_df["comment_text"].tolist(),
                           test_df[LABELS].values.astype(np.float32),
                           tokenizer, cfg.max_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.eval_batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.eval_batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    num_training_steps = len(train_loader) * cfg.epochs
    num_warmup_steps = int(cfg.warmup_ratio * num_training_steps)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    best_macro_f1 = -1.0
    best_epoch = -1
    best_state = None
    history = []

    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        n_steps = 0
        for batch in train_loader:
            inputs = {k: v.to(cfg.device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            n_steps += 1
        train_loss = running_loss / max(n_steps, 1)

        val_probs = _predict_probs(model, val_loader, cfg.device)
        val_preds = (val_probs >= cfg.threshold).astype(int)
        val_metrics = compute_metrics(val_df[LABELS].values, val_preds)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_micro_f1": val_metrics["micro_f1"],
        })
        log(f"  epoch {epoch}/{cfg.epochs}  loss={train_loss:.4f}  "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}  "
            f"val_micro_f1={val_metrics['micro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.time() - t0
    log(f"  best epoch={best_epoch}  val_macro_f1={best_macro_f1:.4f}  train_time={elapsed:.1f}s")

    test_probs = _predict_probs(model, test_loader, cfg.device)
    test_preds = (test_probs >= cfg.threshold).astype(int)
    test_metrics = compute_metrics(test_df[LABELS].values, test_preds)

    # Free GPU memory.
    del model, best_state
    torch.cuda.empty_cache()

    return {
        "metrics": test_metrics,
        "probs": test_probs,
        "preds": test_preds,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "train_time_seconds": elapsed,
    }
