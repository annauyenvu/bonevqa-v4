"""Deterministic, multiplicity-preserving component-disjoint LF donors."""
from collections import defaultdict
import heapq
import random
import hashlib
import json
from pathlib import Path


def lf_context_mode(cfg, training):
    """An explicit eval rule prevents a training placebo becoming batch-dependent inference."""
    train_mode = cfg.get("latent_context_control", "conditioned")
    mode = train_mode if training else cfg.get("latent_context_control_eval", train_mode)
    if mode not in ("conditioned", "unconditioned", "shuffled"):
        raise ValueError("Unknown LF context mode")
    if not training and "latent_context_control_eval" in cfg and mode == "shuffled":
        raise ValueError("Explicit scientific LF evaluation must not depend on shuffled batch donors")
    return mode


def component_disjoint_donors(group_ids):
    """Return a bijection of rows, rejecting impossible majority-group batches.

    IDs must be duplicate-component IDs, not question IDs. Sorting groups then
    rotating by the largest group size yields a cross-group derangement iff
    no group occupies more than half the rows. No rows are dropped/repeated.
    """
    groups = defaultdict(list)
    for index, group in enumerate(group_ids):
        if group is None:
            raise ValueError("LF donor component IDs must be present")
        groups[group].append(index)
    count = len(group_ids)
    if count < 2:
        raise ValueError("LF donors require at least two rows")
    largest = max(map(len, groups.values()))
    if largest * 2 > count:
        raise ValueError("No bijective cross-component donors: dominant component exceeds half the batch")
    ordered = [index for group in groups.values() for index in group]
    donors = [None] * count
    for position, index in enumerate(ordered):
        donors[index] = ordered[(position + largest) % count]
    if sorted(donors) != list(range(count)) or any(group_ids[i] == group_ids[j] for i, j in enumerate(donors)):
        raise RuntimeError("Invalid component-disjoint donor construction")
    return donors


def component_balanced_batches(group_ids, batch_size, seed):
    """Build a common TRAIN schedule without dropping/repeating QA rows.

    Greedy balancing is deliberately fail-closed, not a proof that every
    feasible dataset can be scheduled. The same returned schedule must be used
    by real AND placebo arms; never filter only the placebo's difficult rows.
    """
    if batch_size < 4 or len(group_ids) < 2:
        raise ValueError("Balanced LF schedules require batch_size>=4 and at least two rows")
    rng = random.Random(seed)
    groups = defaultdict(list)
    for index, group in enumerate(group_ids):
        if group is None:
            raise ValueError("LF donor component IDs must be present")
        groups[group].append(index)
    keys = list(groups)
    rng.shuffle(keys)
    queues = [groups[key][:] for key in keys]
    for queue in queues:
        rng.shuffle(queue)
    heap = [(-len(queue), rank) for rank, queue in enumerate(queues)]
    heapq.heapify(heap)
    sizes = [batch_size] * (len(group_ids) // batch_size)
    remainder = len(group_ids) % batch_size
    if remainder == 1 and sizes:
        sizes[-1] -= 1
        remainder = 2
    if remainder:
        sizes.append(remainder)
    batches = []
    for size in sizes:
        used, held, batch = defaultdict(int), [], []
        for _ in range(size):
            if not heap:
                raise ValueError("Cannot construct common component-balanced schedule without dropping rows")
            negative_count, rank = heapq.heappop(heap)
            batch.append(queues[rank].pop())
            used[rank] += 1
            remaining = negative_count + 1
            if remaining:
                if used[rank] < size // 2:
                    heapq.heappush(heap, (remaining, rank))
                else:
                    held.append((remaining, rank))
        for item in held:
            heapq.heappush(heap, item)
        rng.shuffle(batch)
        component_disjoint_donors([group_ids[index] for index in batch])
        batches.append(batch)
    if sorted(index for batch in batches for index in batch) != list(range(len(group_ids))):
        raise RuntimeError("Balanced schedule lost or repeated rows")
    return batches


class ComponentBatchSampler:
    """Epoch-addressable sampler: schedule replay needs no mutable RNG state."""
    def __init__(self, group_ids, batch_size, seed):
        self.group_ids = list(group_ids)
        self.batch_size, self.seed = int(batch_size), int(seed)
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self.batches = component_balanced_batches(self.group_ids, self.batch_size, self.seed + self.epoch)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

    def binding(self):
        return {"policy": "component_balanced_no_drop_v1", "batch_size": self.batch_size,
                "seed": self.seed, "ordered_groups_sha256": hashlib.sha256(json.dumps(self.group_ids).encode()).hexdigest()}


def load_component_lookup(manifest, seal, root):
    manifest, seal, root = Path(manifest).resolve(), Path(seal).resolve(), Path(root).resolve()
    content = manifest.read_bytes()
    sealed = json.loads(seal.read_text())
    digest = hashlib.sha256(content).hexdigest()
    if (sealed.get("status") != "sealed_development_data" or
            sealed.get("artifacts", {}).get(manifest.relative_to(root).as_posix()) != digest):
        raise ValueError("Component manifest not bound to approved data seal")
    lookup = {}
    for group, images in enumerate(json.loads(content)["components"]):
        for image in images:
            if image in lookup:
                raise ValueError("Image appears in multiple components")
            lookup[image] = group
    return lookup, digest
