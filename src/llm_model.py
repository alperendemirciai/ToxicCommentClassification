"""Ollama client + robust JSON parser for multi-label predictions.

Calls the local Ollama server at OLLAMA_HOST. We pass a pre-built chat
messages list (system + user) and expect one line of JSON back. The parser
tries:
  1. Strict JSON over the first {...} substring,
  2. Per-label regex fallback ("<label>": 0|1),
  3. All-zeros sentinel if both fail (marked parse_failure=True so we can
     audit how often the LLM produced unparseable output).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import requests

from src.config import (
    LABELS,
    LLM_NUM_PREDICT,
    LLM_NUM_PREDICT_THINK,
    LLM_REQUEST_TIMEOUT,
    LLM_TEMPERATURE,
    LLM_THINK,
    OLLAMA_HOST,
)

_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_LABEL_PATTERNS = {
    lbl: re.compile(rf'"\s*{re.escape(lbl)}\s*"\s*:\s*([01])', re.IGNORECASE)
    for lbl in LABELS
}


@dataclass
class LLMPrediction:
    labels: dict[str, int]
    raw: str
    parse_failure: bool
    latency_seconds: float


def parse_response(raw: str) -> tuple[dict[str, int], bool]:
    """Return (labels_dict, parse_failure_flag).

    parse_failure=True means we had to fall back below strict JSON parsing.
    A pure all-zeros sentinel result also flags parse_failure.
    """
    if not raw:
        return {lbl: 0 for lbl in LABELS}, True

    text = raw.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    for match in _JSON_OBJECT_RE.findall(text):
        try:
            obj = json.loads(match)
            if isinstance(obj, dict) and any(lbl in obj for lbl in LABELS):
                out = {lbl: int(bool(obj.get(lbl, 0))) for lbl in LABELS}
                return out, False
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    out = {}
    found_any = False
    for lbl, pat in _LABEL_PATTERNS.items():
        m = pat.search(text)
        if m:
            out[lbl] = int(m.group(1))
            found_any = True
        else:
            out[lbl] = 0
    if found_any:
        return out, True

    return {lbl: 0 for lbl in LABELS}, True


_REASONING_MODELS = ("gpt-oss", "deepseek-r1", "qwq", "o1")


def _is_reasoning_model(model: str) -> bool:
    m = model.lower()
    return any(tag in m for tag in _REASONING_MODELS)


def call_ollama(
    model: str,
    messages: list[dict],
    *,
    host: str = OLLAMA_HOST,
    num_predict: int = LLM_NUM_PREDICT,
    temperature: float = LLM_TEMPERATURE,
    timeout: int = LLM_REQUEST_TIMEOUT,
    think: bool = LLM_THINK,
) -> tuple[str, float]:
    """One blocking chat call. messages is a list of {role, content} dicts.
    Returns (response_text, latency_seconds).

    `think=True` lets reasoning models emit a chain-of-thought before the JSON
    (slower, larger token budget). `think=False` asks them to answer directly.
    Ignored for non-reasoning models.
    """
    url = f"{host}/api/chat"
    reasoning = _is_reasoning_model(model)
    if reasoning and think:
        budget = max(num_predict, LLM_NUM_PREDICT_THINK)
    elif reasoning:
        budget = max(num_predict, 512)
    else:
        budget = num_predict
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        # Grammar-constrain to a valid JSON object. Suppresses preambles,
        # markdown fences, and stray prose from non-reasoning models, and keeps
        # reasoning models' visible content strictly JSON.
        "format": "json",
        "think": bool(think),
        "options": {
            "temperature": temperature,
            "num_predict": budget,
            "top_p": 1.0,
            "seed": 42,
        },
    }
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    elapsed = time.time() - t0
    data = r.json()
    msg = data.get("message", {})
    content = msg.get("content", "") or ""
    # Safety net: if a reasoning model still routed its answer into `thinking`
    # (e.g. Ollama version that ignores `think: false`), parse from there.
    if not content.strip():
        content = msg.get("thinking", "") or ""
    return content, elapsed


def predict_one(model: str, messages: list[dict], *, think: bool = LLM_THINK) -> LLMPrediction:
    raw, elapsed = call_ollama(model, messages, think=think)
    labels, failure = parse_response(raw)
    return LLMPrediction(labels=labels, raw=raw, parse_failure=failure, latency_seconds=elapsed)


def ensure_model_available(model: str, host: str = OLLAMA_HOST) -> None:
    """Verify the model is pulled on the local Ollama server; raise otherwise."""
    r = requests.get(f"{host}/api/tags", timeout=10)
    r.raise_for_status()
    available = [m["name"] for m in r.json().get("models", [])]
    if model not in available:
        raise RuntimeError(f"Model {model!r} not pulled in Ollama. Available: {available}")
