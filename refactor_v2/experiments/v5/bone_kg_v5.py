"""Bone prior-knowledge graph for the LaPA prior-knowledge fusion (PF) module.

LaPA builds PF on the SLAKE organ-disease graph.  The previous BoneVQA port
re-used a 409-node SLAKE graph (chest/abdomen/brain), which carries almost no
musculoskeletal knowledge.  This module replaces it with a bone graph:

* static edges: textbook anatomy (region -> bone -> joint), radiographic
  vocabulary (views, modality, hardware, location words) and finding semantics;
* data edges: co-occurrence counted ONLY on the TRAIN records of one fold
  (region-view, region-hardware, region-fracture, region-count).  Validation,
  calibration and test records are never read here.

The graph is returned as plain JSON-serialisable lists so it can be stored in
the checkpoint config and reproduced exactly at inference.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

REGIONS = ["hand", "leg", "hip", "shoulder"]
ANATOMY: Dict[str, List[str]] = {
    "hand": ["phalanx", "metacarpal", "carpal bone", "wrist", "distal radius", "distal ulna"],
    "leg": ["femur", "knee", "patella", "tibia", "fibula", "ankle", "tarsal bone", "metatarsal", "foot"],
    "hip": ["pelvis", "acetabulum", "femoral head", "femoral neck"],
    "shoulder": ["humerus", "clavicle", "scapula", "glenoid", "acromion"],
}
JOINTS = {"wrist": ["distal radius", "distal ulna", "carpal bone"],
          "knee": ["femur", "tibia", "patella"],
          "ankle": ["tibia", "fibula", "tarsal bone"],
          "femoral head": ["acetabulum"],
          "glenoid": ["humerus"]}
VIEWS = ["frontal view", "lateral view", "oblique view"]
FINDINGS = ["fracture", "no fracture", "no abnormality", "multiple fractures", "orthopedic hardware"]
COUNTS = ["zero", "one", "two", "three", "four"]
LOCATIONS = ["center", "upper", "lower", "left", "right", "middle"]
MODALITY = ["x-ray", "radiograph"]

STATIC_SEMANTIC_EDGES: List[Tuple[str, str]] = [
    ("x-ray", "radiograph"),
    ("fracture", "multiple fractures"),
    ("no fracture", "no abnormality"),
    ("no fracture", "zero"),
    ("fracture", "one"), ("multiple fractures", "two"), ("multiple fractures", "three"),
    ("multiple fractures", "four"),
    ("orthopedic hardware", "fracture"),
    ("radiograph", "frontal view"), ("radiograph", "lateral view"), ("radiograph", "oblique view"),
    ("fracture", "center"), ("fracture", "upper"), ("fracture", "lower"),
    ("fracture", "left"), ("fracture", "right"), ("fracture", "middle"),
]


def node_names() -> List[str]:
    names: List[str] = []
    for group in (REGIONS, *ANATOMY.values(), VIEWS, FINDINGS, COUNTS, LOCATIONS, MODALITY):
        for n in group:
            if n not in names:
                names.append(n)
    return names


def _split_regions(region: str) -> List[str]:
    text = str(region or "").replace(" and ", ",")
    return [p.strip() for p in text.split(",") if p.strip() in REGIONS]


def _count_word(n: int) -> str:
    return COUNTS[min(max(int(n), 0), len(COUNTS) - 1)]


def build_bone_kg(train_records: Iterable[dict], min_cooccurrence: int = 5) -> dict:
    """Build the fold-specific graph from TRAIN records only.

    ``min_cooccurrence`` is counted per *image* (not per QA row) so that the
    number of templated questions per image cannot inflate an edge.
    """
    names = node_names()
    index = {n: i for i, n in enumerate(names)}
    edges = set()

    def add(a: str, b: str):
        i, j = index[a], index[b]
        if i != j:
            edges.add((min(i, j), max(i, j)))

    for region, parts in ANATOMY.items():
        for p in parts:
            add(region, p)
    for joint, parts in JOINTS.items():
        for p in parts:
            add(joint, p)
    for a, b in STATIC_SEMANTIC_EDGES:
        add(a, b)

    per_image: Dict[str, dict] = {}
    for r in train_records:
        per_image.setdefault(str(r["image_id"]), r)
    counter: Counter = Counter()
    for r in per_image.values():
        regions = _split_regions(r.get("region"))
        views = [f"{v} view" for v in (r.get("views") or []) if f"{v} view" in index]
        facts = list(views)
        fractured = int(r.get("fractured") or 0)
        facts.append("fracture" if fractured else "no fracture")
        if int(r.get("hardware") or 0):
            facts.append("orthopedic hardware")
        facts.append(_count_word(r.get("fracture_count") or 0))
        if len(regions) > 1:
            for a in regions:
                for b in regions:
                    if a < b:
                        counter[(a, b)] += 1
        for reg in regions:
            for f in facts:
                counter[(reg, f)] += 1
    data_edges = sorted((a, b, n) for (a, b), n in counter.items() if n >= min_cooccurrence)
    for a, b, _ in data_edges:
        add(a, b)
    graph = {
        "version": "bone_kg_v5",
        "nodes": names,
        "edges": sorted([list(e) for e in edges]),
        "data_edges": [[a, b, int(n)] for a, b, n in data_edges],
        "train_images": len(per_image),
        "min_cooccurrence": min_cooccurrence,
    }
    graph["sha256"] = graph_hash(graph)
    return graph


def graph_hash(graph: dict) -> str:
    payload = json.dumps({"nodes": graph["nodes"], "edges": graph["edges"]}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def degree_preserving_rewire(graph: dict, seed: int = 20261030, swaps_per_edge: int = 10) -> dict:
    """Control graph: same nodes and identical degree sequence, edges shuffled.

    Used as the semantic negative control for PF (LaPA Table 2 analogue):
    if a bone graph helps only through capacity, the rewired graph helps
    equally.  Double-edge swaps keep every node degree unchanged.
    """
    rng = random.Random(seed)
    edges = [tuple(e) for e in graph["edges"]]
    present = set(edges)
    n_swaps = swaps_per_edge * len(edges)
    for _ in range(n_swaps):
        (a, b), (c, d) = rng.sample(edges, 2)
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4:
            continue
        e1, e2 = (min(a, d), max(a, d)), (min(c, b), max(c, b))
        if e1 in present or e2 in present:
            continue
        present.discard((min(a, b), max(a, b)))
        present.discard((min(c, d), max(c, d)))
        present.add(e1)
        present.add(e2)
        edges = sorted(present)
    out = dict(graph)
    out["version"] = graph["version"] + "_rewired"
    out["edges"] = sorted([list(e) for e in present])
    out["rewire_seed"] = seed
    out["sha256"] = graph_hash(out)
    return out


def degrees(graph: dict) -> List[int]:
    deg = [0] * len(graph["nodes"])
    for a, b in graph["edges"]:
        deg[a] += 1
        deg[b] += 1
    return deg


def adjacency(graph: dict, self_loops: bool = True):
    import torch

    n = len(graph["nodes"])
    adj = torch.zeros(n, n, dtype=torch.bool)
    for a, b in graph["edges"]:
        adj[a, b] = True
        adj[b, a] = True
    if self_loops:
        adj |= torch.eye(n, dtype=torch.bool)
    return adj


def assert_train_only(train_records: Sequence[dict], forbidden_records: Sequence[dict]) -> None:
    """Fail closed if any non-TRAIN image id were passed to the builder."""
    train_ids = {str(r["image_id"]) for r in train_records}
    bad = train_ids & {str(r["image_id"]) for r in forbidden_records}
    if bad:
        raise AssertionError(f"KG builder received {len(bad)} non-TRAIN images")
