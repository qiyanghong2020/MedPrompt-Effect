from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from datasets import Dataset, load_dataset


@dataclass(frozen=True)
class MCQExample:
    dataset: str
    id: str
    question: str
    options: str
    choices: list[str]
    answer: str
    meta: dict[str, Any]


def _fmt_options(letter_to_text: dict[str, str]) -> str:
    return "\n".join([f"{k}. {v}" for k, v in letter_to_text.items()])


_OPTION_LABEL_RE = re.compile(r"(?m)^\s*([A-Z])\s*[\).:]\s+")
_MEDX_INLINE_CHOICE_RE = re.compile(r"\(([A-Z])\)\s*")


def infer_choices_from_options(options_text: str) -> list[str]:
    """
    Infer option labels from an options block.

    We prefer line-anchored parsing (e.g., "A. ...") to avoid false matches such as "E. coli".
    """
    if not options_text:
        return []
    labels: list[str] = []
    seen: set[str] = set()
    for m in _OPTION_LABEL_RE.finditer(options_text):
        lab = m.group(1).upper()
        if lab not in seen:
            labels.append(lab)
            seen.add(lab)
    return labels


def parse_medxpertqa_question(question: str) -> tuple[str, str | None]:
    """
    MedXpertQA-text often embeds the full answer choice list at the end of the question as:
    "Answer Choices: (A) ... (B) ... ...".

    Returns (stem_without_choices, options_block_or_none).
    """
    if not question:
        return "", None
    if "Answer Choices:" not in question:
        return question.strip(), None
    stem, choice_text = question.split("Answer Choices:", 1)
    stem = stem.strip()
    choice_text = choice_text.strip()

    matches = list(_MEDX_INLINE_CHOICE_RE.finditer(choice_text))
    if not matches:
        return stem, None

    parts: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        label = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(choice_text)
        opt = choice_text[start:end].strip()
        opt = re.sub(r"\s+", " ", opt)
        opt = opt.strip(" ;\n\t")
        parts.append((label, opt))
    options_block = "\n".join([f"{k}. {v}" for k, v in parts if v])
    return stem, options_block or None


def load_medxpertqa(split: str = "test", token: str | None = None) -> list[MCQExample]:
    ds: Dataset = load_dataset("simwit/medxpertqa-text", split=split, token=token or os.environ.get("HF_TOKEN"))
    out: list[MCQExample] = []
    for row in ds:
        q = str(row["question"])
        stem, opts_from_q = parse_medxpertqa_question(q)
        opts = opts_from_q if opts_from_q else str(row["options"])
        choices = infer_choices_from_options(opts)
        out.append(
            MCQExample(
                dataset="MedXpertQA",
                id=str(row.get("id", "")),
                question=stem,
                options=opts,
                choices=choices,
                answer=str(row["answer"]).strip().upper(),
                meta={
                    "medical_task": row.get("medical_task"),
                    "body_system": row.get("body_system"),
                    "question_type": row.get("question_type"),
                },
            )
        )
    return out


def load_medmcqa(split: str = "validation") -> list[MCQExample]:
    ds: Dataset = load_dataset("medmcqa", split=split)
    letters = ["A", "B", "C", "D"]
    out: list[MCQExample] = []
    for row in ds:
        choices = {"A": row["opa"], "B": row["opb"], "C": row["opc"], "D": row["opd"]}
        gold = letters[int(row["cop"])]
        out.append(
            MCQExample(
                dataset="MedMCQA",
                id=str(row.get("id", "")),
                question=str(row["question"]),
                options=_fmt_options(choices),
                choices=letters,
                answer=gold,
                meta={
                    "subject": row.get("subject_name"),
                    "topic": row.get("topic_name"),
                    "choice_type": row.get("choice_type"),
                },
            )
        )
    return out


def load_medqa_usmle_4_options(split: str = "test") -> list[MCQExample]:
    ds: Dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
    out: list[MCQExample] = []
    for i, row in enumerate(ds):
        options: dict[str, str] = dict(row["options"])
        # Ensure consistent order
        ordered = {k: options[k] for k in ["A", "B", "C", "D"] if k in options}
        out.append(
            MCQExample(
                dataset="MedQA-USMLE",
                # meta_info is not a unique identifier in this dataset; use split index for stable uniqueness.
                id=f"{split}-{i}",
                question=str(row["question"]),
                options=_fmt_options(ordered),
                choices=list(ordered.keys()),
                answer=str(row["answer_idx"]).strip().upper(),
                meta={},
            )
        )
    return out
