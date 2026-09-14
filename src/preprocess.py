"""Text preprocessing and cleaning utilities for news articles."""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Dict, Any


def clean_article_text(text: str) -> str:
    """Cleans article text while strictly preserving factual content.

    Preserves dates, names, numbers, locations, organizations, quotes,
    and event details needed for factual verification.
    """
    if not text:
        return ""

    # Decode HTML entities (e.g. &amp;, &quot;, &#39;)
    cleaned = html.unescape(text)

    # Strip inline HTML tags if any remain
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)

    # Normalize Unicode characters (e.g. smart quotes, em-dashes, accented chars)
    cleaned = unicodedata.normalize("NFKC", cleaned)

    # Normalize diverse quotation marks and dashes to standard forms
    cleaned = re.sub(r'[\u201c\u201d\u201e\u201f\u00ab\u00bb]', '"', cleaned)
    cleaned = re.sub(r"[\u2018\u2019\u201a\u201b]", "'", cleaned)
    cleaned = re.sub(r"[\u2013\u2014\u2015]", " - ", cleaned)

    # Remove typical web navigation/boilerplate phrases if isolated on lines
    boilerplate_patterns = [
        r"(?i)^\s*(share this article|follow us on|read more|advertisement|sponsored content|subscribe now|sign up for our newsletter|cookie policy|terms of service)\s*$",
        r"(?i)^\s*(photo by|image credit|getty images|reuters\/|ap photo)\s*.*$",
    ]
    lines = []
    for line in cleaned.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        is_boilerplate = any(re.match(p, line_stripped) for p in boilerplate_patterns)
        if not is_boilerplate:
            lines.append(line_stripped)

    cleaned = " ".join(lines)

    # Collapse excessive spaces, tabs, and duplicate punctuation
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    cleaned = cleaned.strip()

    return cleaned


def extract_text_statistics(text: str) -> Dict[str, Any]:
    """Extracts basic statistical metadata from cleaned article text."""
    if not text:
        return {"word_count": 0, "char_count": 0, "sentence_count": 0}

    words = re.findall(r"\b\w+\b", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    return {
        "word_count": len(words),
        "char_count": len(text),
        "sentence_count": len(sentences),
    }
