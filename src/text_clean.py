"""Text cleaning for Jigsaw Wikipedia talk-page comments.

Implements two cleaning layers, both designed to be safe for transformer
tokenizers (no stopword removal, no case folding, no aggressive normalization):

  1. Structural noise — HTML entities/tags, wiki markup, URLs, user mentions,
     IPs, and whitespace collapse.
  2. Unicode hygiene — NFKC normalization, zero-width character removal.

Disabled by default (CLEAN_TEXT in config). When enabled, applied once at
dataset-construction time for BERT and at prompt-build time for the LLM, so the
runtime cost is negligible.
"""
from __future__ import annotations

import html
import re
import unicodedata

# --- structural patterns ---------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_WIKI_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")          # {{tl|foo}}
_WIKI_HEADING_RE = re.compile(r"={2,}\s*([^=]+?)\s*={2,}") # == heading ==
_WIKI_USER_LINK_RE = re.compile(
    r"\[\[\s*(?:User|User talk|Special:Contributions)\s*:[^\]]*\]\]",
    re.IGNORECASE,
)
_WIKI_LINK_RE = re.compile(r"\[\[([^\]|]+?)(?:\|[^\]]+)?\]\]")  # [[Page|Text]] -> Text/Page
_SIGNATURE_RE = re.compile(r"~{3,}")
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_MULTI_WS_RE = re.compile(r"\s+")

# --- unicode patterns ------------------------------------------------------
_ZERO_WIDTH_RE = re.compile(r"[​-‍⁠﻿]")


def _strip_structural_noise(text: str) -> str:
    # HTML entities first so &amp; becomes & before tag-strip operates.
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)

    text = _WIKI_USER_LINK_RE.sub(" <USER> ", text)
    text = _WIKI_TEMPLATE_RE.sub(" ", text)
    text = _WIKI_HEADING_RE.sub(r" \1 ", text)
    text = _WIKI_LINK_RE.sub(r"\1", text)
    text = _SIGNATURE_RE.sub(" ", text)

    text = _URL_RE.sub(" <URL> ", text)
    text = _IP_RE.sub(" <IP> ", text)

    text = _MULTI_WS_RE.sub(" ", text).strip()
    return text


def _unicode_hygiene(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    return text


def clean_text(text: str | None, enabled: bool = True) -> str:
    """Apply structural + unicode cleaning. Returns empty string for None/NaN."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not enabled:
        return text
    text = _unicode_hygiene(text)
    text = _strip_structural_noise(text)
    return text
