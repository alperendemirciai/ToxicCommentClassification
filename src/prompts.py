"""Prompt templates and few-shot example selection for the LLM evaluation.

The same JSON-only output spec is shared between zero-shot and few-shot to keep
the parser identical. Few-shot examples are selected deterministically from the
fold's TRAINING partition so no test-fold information leaks into the prompt.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from src.config import LABELS, LLM_FEW_SHOT_K

SYSTEM_PROMPT = (
    "You are a strict multi-label content moderation classifier. "
    "Given an internet comment, decide for EACH of these six categories whether the comment belongs to it: "
    "toxic, severe_toxic, obscene, threat, insult, identity_hate. "
    "Definitions: "
    "(toxic) rude, disrespectful, or unreasonable language likely to make someone leave a discussion; "
    "(severe_toxic) extremely aggressive, hateful, or violent content; "
    "(obscene) profanity, sexual content, or vulgar language; "
    "(threat) statements expressing intent to harm a person or group; "
    "(insult) targeted personal attacks or demeaning statements; "
    "(identity_hate) hate or prejudice toward a person/group based on race, ethnicity, religion, gender, sexual orientation, disability, or nationality. "
    "Multiple labels may apply at once. Output ONLY one line containing a single JSON object with exactly these six integer keys "
    "(values 0 or 1) and no other text, no explanation, no markdown: "
    '{"toxic":0|1,"severe_toxic":0|1,"obscene":0|1,"threat":0|1,"insult":0|1,"identity_hate":0|1}'
)

USER_PREFIX_ZERO_SHOT = "Classify the following comment.\n\nComment: "
USER_SUFFIX = "\n\nJSON:"

USER_PREFIX_FEW_SHOT_HEADER = "Here are labeled examples. After them, classify the new comment.\n\n"
FEW_SHOT_EXAMPLE_TEMPLATE = "Comment: {text}\nJSON: {json}\n\n"
USER_PREFIX_FEW_SHOT_QUERY = "Now classify this comment.\n\nComment: "


@dataclass
class FewShotExample:
    text: str
    labels: dict


def labels_to_json(label_dict: dict) -> str:
    """Render a label dict as compact JSON in canonical order."""
    return json.dumps({lbl: int(label_dict[lbl]) for lbl in LABELS}, separators=(",", ":"))


def _truncate(text: str, max_chars: int = 800) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= max_chars else text[:max_chars] + " ..."


def select_few_shot_examples(train_df: pd.DataFrame, seed: int = 0) -> list[FewShotExample]:
    """Deterministically choose 6 examples from the fold's training data:
    one positive example per label (single-label positives preferred, falling
    back to any positive) plus 2 'clean' (all-zero) negatives.

    The same examples will be reused for every test query in this fold.
    """
    examples: list[FewShotExample] = []
    used_ids: set = set()
    train_df = train_df.copy()

    # 1 example per label, prefer single-label samples for clarity.
    label_sums = train_df[LABELS].sum(axis=1)
    for lbl in LABELS:
        single = train_df[(train_df[lbl] == 1) & (label_sums == 1)]
        if len(single) > 0:
            cand = single.sample(n=1, random_state=seed).iloc[0]
        else:
            multi = train_df[train_df[lbl] == 1]
            if len(multi) == 0:
                # extremely rare; skip — caller can handle by padding.
                continue
            cand = multi.sample(n=1, random_state=seed).iloc[0]
        used_ids.add(cand["id"])
        examples.append(FewShotExample(
            text=_truncate(cand["comment_text"]),
            labels={l: int(cand[l]) for l in LABELS},
        ))

    # 2 clean negatives.
    clean = train_df[(label_sums == 0) & (~train_df["id"].isin(used_ids))]
    if len(clean) >= 2:
        chosen = clean.sample(n=2, random_state=seed)
    else:
        chosen = clean
    for _, r in chosen.iterrows():
        examples.append(FewShotExample(
            text=_truncate(r["comment_text"]),
            labels={l: 0 for l in LABELS},
        ))

    # Trim/pad to LLM_FEW_SHOT_K.
    return examples[:LLM_FEW_SHOT_K]


def build_zero_shot_messages(comment: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_PREFIX_ZERO_SHOT + _truncate(comment) + USER_SUFFIX},
    ]


def build_few_shot_messages(comment: str, examples: Iterable[FewShotExample]) -> list[dict]:
    parts = [USER_PREFIX_FEW_SHOT_HEADER]
    for ex in examples:
        parts.append(FEW_SHOT_EXAMPLE_TEMPLATE.format(text=ex.text, json=labels_to_json(ex.labels)))
    parts.append(USER_PREFIX_FEW_SHOT_QUERY + _truncate(comment) + USER_SUFFIX)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": "".join(parts)},
    ]


def render_messages_as_text(messages: list[dict]) -> str:
    """Plain-text rendering for the appendix / logs."""
    return "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages)
