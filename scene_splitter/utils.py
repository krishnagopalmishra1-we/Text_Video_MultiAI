"""Utility functions for the scene splitter."""
from __future__ import annotations

import re
import unicodedata


def estimate_duration(
    word_count: int,
    words_per_second: float = 2.5,
    buffer_seconds: float = 1.5,
) -> float:
    """Estimate scene duration from word count."""
    return (word_count / words_per_second) + buffer_seconds


def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable characters."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r\n?", "\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """
    Simple sentence splitter that handles common abbreviations.
    Falls back to regex when nltk is unavailable.
    """
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        return nltk.sent_tokenize(text)
    except ImportError:
        pass

    # Regex fallback
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def word_count(text: str) -> int:
    return len(text.split())


def srt_time_to_seconds(time_str: str) -> float:
    """Convert SRT timestamp (00:01:23,456) to seconds."""
    time_str = time_str.replace(",", ".")
    parts = time_str.split(":")
    h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + s
