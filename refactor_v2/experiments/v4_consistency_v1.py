"""Rewrite every fracture-dependent answer so it agrees with one final fracture decision.  No generation: region comes from the model's own
sentence, location from the model's sentence or the detector's top box.  check() is an independent contradiction count."""
import json, re
from pathlib import Path
from src.data.fracatlas_vqa import _position
from src.utils import normalize_answer as na
CAN = "Is there a fracture in this X-ray?"; DESC = "Describe the finding."; ABN = "What abnormality is seen in this X-ray?"; CNT = "How many fracture sites are visible?"; WHERE = "Where is the fracture located?"
WORDS = {1: "one", 2: "two", 3: "three"}; ARTICLE = {"one": "A fracture is", "two": "Two fractures are", "three": "Three fractures are"}


def region(text, fallback):
    m = re.search(r"(?:in|of) the ([a-z ]+?)(?: at |\.|$)", text.lower()); return m.group(1).strip() if m else fallback


def pos(det):
    if not det or not det["boxes"]: return "center"
    return _position(det["boxes"][0], *det["size"])


def fix(r, yes, det, location):
    """location: 'model' keeps the model's location when its sentence already reports a fracture, 'detector' always uses the top box."""
    q, p = r["question"], r["pred"]; t = p.lower()
    if q == CAN: return "yes" if yes else "no"
    if re.fullmatch(r"(is the .* fractured\?|are there multiple fractures\?)", q.lower()): return p if yes else "no"
    if q == ABN: return "fracture" if yes else "no abnormality"
    if q == CNT:
        if not yes: return "zero"
        return "one" if na(p) == "zero" else p
    if q == DESC:
        reg = region(p, "bone")
        if not yes: return f"No fracture is seen in the {reg}."
        if not t.startswith("no fracture") and location == "model": return p
        head = re.match(r"^(two fractures are|three fractures are|four fractures are)", t)
        return f"{head.group(1).capitalize() if head else 'A fracture is'} seen in the {reg} at the {pos(det)}."
    if q == WHERE:
        if not yes: return "no fracture seen"
        reg = region(p, "bone")
        return p if location == "model" and "no fracture" not in t else f"{pos(det)} of the {reg}"
    return p


def contradictions(rows):
    """Rows of one image with a canonical answer: number of other rows that disagree with it."""
    base = next((r["pred"] for r in rows if r["question"] == CAN), None)
    if base is None: return 0
    yes = na(base) == "yes"; c = 0
    for r in rows:
        q, p = r["question"], r["pred"].lower().strip(" .")
        if q == DESC: c += p.startswith("no fracture") == yes
        elif q == ABN: c += (p == "no abnormality") == yes
        elif q == CNT: c += (p == "zero") == yes
        elif q.lower().startswith("is the") and q.lower().endswith("fractured?") or q.lower() == "are there multiple fractures?": c += (not yes) and p == "yes"
        elif q == WHERE: c += (not yes) and "no fracture" not in p
    return int(c)
