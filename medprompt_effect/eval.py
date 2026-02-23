from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from medprompt_effect.data import MCQExample
from medprompt_effect.openrouter import ChatRequest, OpenRouterClient


_FINAL_RE = re.compile(r"(?im)^\s*(?:[-*]\s*)?FINAL\s*[:：\-]\s*([A-Z])\b")
_ANSWER_LINE_RE = re.compile(r"(?im)^\s*(?:[-*]\s*)?(?:ANSWER|FINAL\s*ANSWER|答案)\s*[:：\-]\s*([A-Z])\b")
_ANSWER_SENTENCE_RE = re.compile(r"(?i)\b(?:final\s*answer|answer)\s*(?:is|:)\s*([A-Z])\b")
_STANDALONE_LETTER_RE = re.compile(r"(?im)^\s*([A-Z])\s*$")


def extract_answer(text: str, allowed_choices: Sequence[str]) -> str | None:
    if not text:
        return None
    allowed = {c.strip().upper() for c in allowed_choices if c and c.strip()}
    if not allowed:
        return None

    # Prefer explicit "FINAL:" line (take the last match in case the model revises itself).
    finals = _FINAL_RE.findall(text)
    if finals:
        cand = finals[-1].upper()
        return cand if cand in allowed else None

    # Other explicit answer labels.
    ans_lines = _ANSWER_LINE_RE.findall(text)
    if ans_lines:
        cand = ans_lines[-1].upper()
        return cand if cand in allowed else None

    ans_sent = _ANSWER_SENTENCE_RE.findall(text)
    if ans_sent:
        cand = ans_sent[-1].upper()
        return cand if cand in allowed else None

    # Conservative fallback: standalone letter in the last few non-empty lines.
    tail_lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()][-6:]
    for ln in reversed(tail_lines):
        m = _STANDALONE_LETTER_RE.match(ln)
        if m:
            cand = m.group(1).upper()
            return cand if cand in allowed else None

    return None


@dataclass(frozen=True)
class RunRecord:
    ts: float
    dataset: str
    example_id: str
    model: str
    prompt_name: str
    magic: str | None
    gold: str
    pred: str | None
    correct: bool
    output_text: str
    cost: float | None
    prompt_tokens: int | None
    completion_tokens: int | None

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


def run_one(
    *,
    client: OpenRouterClient,
    model: str,
    prompt_name: str,
    prompt: str,
    ex: MCQExample,
    magic: str | None,
    temperature: float = 0.0,
    max_tokens: int = 32,
    stop: list[str] | None = None,
) -> RunRecord:
    req = ChatRequest(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
    )
    data = client.chat(req)
    msg = data["choices"][0]["message"]
    # Some providers may populate different fields; "content" is the standard.
    output = (msg.get("content") or "").strip()
    pred = extract_answer(output, ex.choices)
    gold = ex.answer.strip().upper()
    correct = pred == gold
    usage = data.get("usage") or {}
    return RunRecord(
        ts=time.time(),
        dataset=ex.dataset,
        example_id=ex.id,
        model=model,
        prompt_name=prompt_name,
        magic=magic,
        gold=gold,
        pred=pred,
        correct=bool(correct),
        output_text=output,
        cost=usage.get("cost"),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )


def write_jsonl(path: str | Path, records: Iterable[RunRecord]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(r.to_json() + "\n")
