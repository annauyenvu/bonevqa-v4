"""TRAIN-only token-length/content-preservation audit using locked tokenizers."""
import hashlib
import json
from pathlib import Path

import open_clip
from transformers import AutoTokenizer

from src.data.sealed_manifest import load_sealed_records
from src.data.prompts import textual_prompt
from src.models.bonevqa import BoneVQAModel
from src.models.encoders import BIOMEDCLIP_ID
from src.utils import load_config


def main():
    config = load_config("refactor_v2/configs/fold0_b_sealed_v3.yaml")
    loader = config["data"]["loader_kwargs"]
    path = Path(loader["sealed_records_by_split"]["train"])
    rows = load_sealed_records(path, loader["seal_path"], loader["project_root"])
    questions = sorted({row["question"] for row in rows})
    clip_tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_ID)
    biomedical = getattr(clip_tokenizer, "tokenizer", clip_tokenizer)
    language = AutoTokenizer.from_pretrained(config["model"]["llm_name"], local_files_only=True)
    budgets = {"biomedical": config["model"]["max_question_tokens"],
               "language": config["model"]["max_question_tokens"] + 8}
    def ids(question, branch):
        tokenizer = biomedical if branch == "biomedical" else language
        text = question if branch == "biomedical" else BoneVQAModel._format_llm_prompt(question)
        return tokenizer(text, add_special_tokens=branch == "biomedical", truncation=False)["input_ids"]
    styles = ["raw", "clinical", "clinical_paraphrase", "irrelevant", "irrelevant_matched"]
    metrics, mismatches = {}, []
    for style in styles:
        lengths = {branch: [] for branch in budgets}
        overflows = {branch: [] for branch in budgets}
        for question in questions:
            text = question if style == "raw" else textual_prompt(question, region=None, style=style)
            for branch, budget in budgets.items():
                size = len(ids(text, branch))
                lengths[branch].append(size)
                if size > budget:
                    overflows[branch].append({"question": question, "tokens": size})
        metrics[style] = {branch: {"min": min(lengths[branch]), "max": max(lengths[branch]),
                                  "over_budget_questions": overflows[branch]} for branch in budgets}
    for question in questions:
        real = textual_prompt(question, region=None, style="clinical")
        control = textual_prompt(question, region=None, style="irrelevant_matched")
        for branch in budgets:
            if len(ids(real, branch)) != len(ids(control, branch)):
                mismatches.append({"question": question, "branch": branch,
                                   "clinical": len(ids(real, branch)), "control": len(ids(control, branch))})
    no_overflow = all(not value["over_budget_questions"] for style in metrics.values() for value in style.values())
    report = {"scope": "fold0 TRAIN questions only; token matching is not semantic efficacy",
              "source_manifest": str(path), "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "train_rows": len(rows), "unique_questions": len(questions), "budgets": budgets,
              "styles": metrics, "matched_control_mismatches": mismatches,
              "all_questions_preserved_without_truncation": no_overflow,
              "passed": no_overflow and not mismatches}
    Path("refactor_v2/reports/textual_token_budget_fold0_v3.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
