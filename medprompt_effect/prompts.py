from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    template: str

    def render(self, *, question: str, options: str, choices: list[str], magic: str | None = None) -> str:
        choices_str = "/".join([c.strip().upper() for c in choices if c and c.strip()])
        base = self.template.format(question=question, options=options, choices=choices_str)
        if magic:
            return f"{magic}\n\n{base}"
        return base


BASELINE_PROMPTS: list[PromptTemplate] = [
    PromptTemplate(
        name="p0_final_rationale",
        template=(
            "You are answering a multiple-choice clinical question.\n"
            "The FIRST line of your response must be the final answer in this exact format:\n"
            "FINAL: <{choices}>\n"
            "Then provide a brief rationale (1–3 sentences).\n\n"
            "Question:\n{question}\n\n"
            "Options:\n{options}\n"
        ),
    ),
    PromptTemplate(
        name="p1_exam_rationale",
        template=(
            "Medical board-style multiple-choice question.\n"
            "First line must be: FINAL: <{choices}>\n"
            "Then briefly explain (1–3 sentences).\n\n"
            "{question}\n\n"
            "{options}\n"
        ),
    ),
    PromptTemplate(
        name="p2_minimal_final",
        template=(
            "Answer the MCQ.\n"
            "First line: FINAL: <{choices}>\n"
            "Then 1–2 short sentences of rationale.\n\n"
            "Q: {question}\n\n"
            "Choices:\n{options}\n"
        ),
    ),
    PromptTemplate(
        name="p3_structured_final",
        template=(
            "Task: choose the single best answer.\n"
            "Format (exactly):\n"
            "FINAL: <{choices}>\n"
            "Rationale: (1–3 sentences)\n\n"
            "Stem:\n{question}\n\n"
            "Options:\n{options}\n"
        ),
    ),
]


# Candidate short “magic prompt” injections to be searched on a dev set.
MAGIC_CANDIDATES: list[str] = [
    "Let's think step by step.",
    "Take a deep breath; be methodical and check for traps.",
    "Be a cautious clinician: avoid premature closure; verify key clues.",
    "Use differential diagnosis and eliminate options before answering.",
    "Focus on discriminative details; ignore distractors.",
    "Prioritize patient safety and the standard of care.",
    "Double-check whether the question asks diagnosis vs next step.",
    "If uncertain, re-read the stem and reassess before answering.",
    "Be strict: choose one best answer; do not hedge.",
    "Work systematically: extract key clues, then eliminate options.",
    "Check for red flags, contraindications, and the safest next step.",
]
