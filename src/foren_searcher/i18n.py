"""Language switch for harness-authored text that models read (10-05: inputs and outputs in English).

``tr(zh, en)`` returns the English text when RuntimeGuards.english is on (spec harness_params.language="en",
hi@0.7.0 / gia@0.8.0 onward) and the original Chinese text otherwise, so older specs stay reproducible.
Externally obtained content (web pages, captions, transcripts, search titles) is never translated.
"""

from __future__ import annotations


def english() -> bool:
    from .policies import active_guards

    return active_guards().english


def tr(zh: str, en: str) -> str:
    return en if english() else zh


__all__ = ["english", "tr"]
