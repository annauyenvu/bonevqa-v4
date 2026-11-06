"""Definitions of clinically reported VQA endpoints.

Keep endpoint membership rules here and independent of model/runtime imports so
data preparation, training, inference, and evaluation cannot drift apart.
"""

import re


# This endpoint is deliberately narrower than a generic search for the word
# ``fracture``.  FracAtlas contains other yes/no tasks such as "Is the hand
# fractured?" (including deliberately wrong-region negatives) and "Are there
# multiple fractures?".  Mixing those tasks into the image-level fracture
# endpoint changes its prevalence and makes its calibration uninterpretable.
#
# The alternatives below are the three templates used by the real dataset,
# smoke fixtures, and historical evaluation exports.  Matching is anchored so
# a region, count, location, or other qualifier cannot silently enter the
# clinical endpoint.
_CANONICAL_FRACTURE_PRESENCE_RE = re.compile(
    r"^is there a fracture(?: in this (?:image|x[ -]?ray))?$"
)


def is_canonical_fracture_presence_question(question: str) -> bool:
    """Return whether *question* is the global binary fracture endpoint.

    Case, repeated whitespace, a trailing sentence mark, and common X-ray
    spelling variants are normalized.  The semantic template itself is an
    exact allow-list: anatomy-specific, multiplicity, and localization
    questions must be handled by their own VQA endpoints.
    """

    q = str(question or "").strip().lower()
    q = q.translate(str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-"}))
    q = re.sub(r"[?!.]+$", "", q).strip()
    q = re.sub(r"\s+", " ", q)
    return _CANONICAL_FRACTURE_PRESENCE_RE.fullmatch(q) is not None
