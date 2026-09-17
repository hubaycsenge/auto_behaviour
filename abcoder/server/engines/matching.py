"""Matching an ethogram entry against a fixed vocabulary of keywords.

Both the audio and the pose engine have to answer the same question: which of
my built-in categories does this researcher's behaviour code mean? They match
keywords, and the order they match in turns out to matter a great deal.

A BORIS description is a full sentence with subordinate clauses, and those
clauses routinely mention things the behaviour is *not*. In the reference
ethogram, "Backing" is described as "moving backwards but not toward the owner
while orienting at the agent" -- searching the whole string and taking the
longest hit picks "orienting at the agent" and codes a retreat as an
orientation. So the code is matched first, on its own, and the description is
consulted only when the code says nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def match_keywords(
    code: str,
    description: str,
    vocabulary: Mapping[str, Sequence[str]],
) -> str | None:
    """Return the vocabulary key that best matches, or ``None``.

    Within each stage the longest matching keyword wins, so "aggressive bark"
    beats the bare "bark" and "approaching owner" beats "approach".
    """
    for haystack in ((code or "").lower(), (description or "").lower()):
        if not haystack:
            continue
        best: tuple[int, str] | None = None
        for key, keywords in vocabulary.items():
            for kw in keywords:
                if kw in haystack and (best is None or len(kw) > best[0]):
                    best = (len(kw), key)
        if best is not None:
            return best[1]
    return None
