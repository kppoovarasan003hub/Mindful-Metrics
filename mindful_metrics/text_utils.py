"""Shared text normalisation for the student wellbeing demo.

The model (TF-IDF) and the recommendation engine must agree on what the
"same" text means. Both call :func:`normalize_text` so that, for example,
"I CAN'T SLEEP!" and "i cant sleep" produce identical features.

Normalisation is intentionally conservative: it lowercases, strips
punctuation and collapses whitespace. It never deletes words, never
rewrites wording and never changes the meaning of what a student wrote.
Stop-word removal is left to the TF-IDF vectorizer, where it belongs.
"""

from __future__ import annotations

import re
import unicodedata

# Contractions are joined, not split: "can't" -> "cant", "don't" -> "dont".
# Deleting the apostrophe (rather than replacing it with a space) keeps the
# token intact, so "can't sleep" matches "cant sleep".
_APOSTROPHES = re.compile(r"['\u2019\u02bc]", flags=re.UNICODE)
# Remaining punctuation (commas, periods, dashes) becomes a word boundary.
_NON_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+", flags=re.UNICODE)


def normalize_text(text: str) -> str:
    """Return a lower-cased, punctuation-free, whitespace-collapsed string.

    Parameters
    ----------
    text:
        Raw user input. Callers are responsible for confirming this is a
        ``str`` first; passing a non-string raises ``TypeError``.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"normalize_text expects str, got {type(text).__name__}"
        )

    # NFKC folds full-width / compatibility characters (e.g. "sleeping　") into
    # their normal ASCII-ish equivalents so the vocabulary is not split.
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _APOSTROPHES.sub("", text)
    text = _NON_WORD.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()