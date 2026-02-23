from __future__ import annotations

import argparse
import random
import re
from pathlib import Path


def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).strip())
    s = s.replace(" ;", ";").replace(" ,", ",")
    return s


def _clip(s: str, max_len: int) -> str:
    s = _norm(s)
    if len(s) <= max_len:
        return s
    # Prefer clipping at a safe boundary.
    cut = s[: max_len + 1]
    for sep in ["; ", ". ", ", "]:
        if sep in cut:
            cut = cut.rsplit(sep, 1)[0]
    cut = cut.strip().rstrip(";,.，。")
    return cut if cut else s[:max_len].strip()


def generate_candidates(n: int, *, seed: int, max_len: int) -> list[str]:
    rng = random.Random(int(seed))

    # Start with a few common baselines / folk prompts (both EN and ZH).
    fixed = [
        "Let's think step by step.",
        "Follow the required output format strictly; be concise.",
        "Be systematic: extract key clues, eliminate distractors, then answer.",
        "Double-check what the question is asking before answering.",
        "Think step by step internally, but keep the final response concise.",
        "You are a professional doctor.",
        "你是一名专业医生。",
        "不要偷懒。",
        "不要偷懒",
        "请仔细读题再作答。",
        "严格按要求输出格式。",
    ]

    roles_en = [
        "a cautious clinician",
        "a board-certified physician",
        "a medical exam expert",
        "a careful medical consultant",
        "a meticulous diagnostician",
        "a senior resident preparing for boards",
    ]
    roles_zh = [
        "一名严谨的临床医生",
        "一名经验丰富的医生",
        "一名医学考试专家",
        "一名慎重的内科医生",
        "一名细致的诊断医生",
        "一名负责的临床老师",
    ]

    actions_en = [
        "Read the stem carefully.",
        "Identify exactly what is being asked (diagnosis vs next step vs treatment).",
        "Extract the discriminative clues (age, timing, vitals, labs).",
        "Use differential diagnosis; eliminate options.",
        "Check for contraindications, red flags, and the safest next step.",
        "Avoid premature closure; verify key clues.",
        "Re-read the question if uncertain.",
        "Do not guess; choose the single best supported option.",
        "Consider common pitfalls and traps.",
        "Double-check units, negations, and timing.",
        "Be concise and decisive.",
        "Follow the required output format strictly.",
    ]
    actions_zh = [
        "仔细读题。",
        "先确认题目问的是诊断/下一步/治疗。",
        "提取关键鉴别线索（年龄、病程、生命体征、化验）。",
        "做鉴别诊断并排除干扰项。",
        "检查禁忌证、红旗征象与最安全下一步。",
        "避免过早下结论，复核关键线索。",
        "不确定就重读题干再判断。",
        "不要猜，选择证据最充分的唯一最佳选项。",
        "注意题目陷阱与否定词。",
        "复核单位、时间线与关键数值。",
        "保持简洁果断。",
        "严格按要求输出格式。",
    ]

    openers_en = [
        "Don't rush.",
        "Be thorough.",
        "Be methodical.",
        "Take a second to double-check.",
        "Work stepwise.",
        "No shortcuts.",
    ]
    openers_zh = [
        "别着急。",
        "要严谨。",
        "要有条理。",
        "先复核一遍。",
        "按步骤来。",
        "不要走捷径。",
        "不要偷懒。",
    ]

    templates_en = [
        "You are {role}. {a1} {a2}",
        "{o} {a1} {a2}",
        "{a1} {a2}",
        "{a1}; {a2}; {a3}",
        "Before answering: {a1} {a2}",
    ]
    templates_zh = [
        "你是{role}。{a1}{a2}",
        "{o}{a1}{a2}",
        "{a1}{a2}",
        "{a1}{a2}{a3}",
        "作答前：{a1}{a2}",
    ]

    out: list[str] = []
    seen: set[str] = set()

    def _add(s: str) -> None:
        s2 = _clip(str(s), max_len)
        if not s2:
            return
        if s2 in seen:
            return
        out.append(s2)
        seen.add(s2)

    for s in fixed:
        _add(s)

    # Generate a bilingual pool with mild variations; keep most candidates short.
    while len(out) < n:
        use_zh = rng.random() < 0.35  # bias toward English to match baseline prompts
        if use_zh:
            tmpl = rng.choice(templates_zh)
            role = rng.choice(roles_zh)
            a1 = rng.choice(actions_zh)
            a2 = rng.choice(actions_zh)
            a3 = rng.choice(actions_zh)
            o = rng.choice(openers_zh)
            s = tmpl.format(role=role, a1=a1, a2=a2, a3=a3, o=o)
        else:
            tmpl = rng.choice(templates_en)
            role = rng.choice(roles_en)
            a1 = rng.choice(actions_en)
            a2 = rng.choice(actions_en)
            a3 = rng.choice(actions_en)
            o = rng.choice(openers_en)
            s = tmpl.format(role=role, a1=a1, a2=a2, a3=a3, o=o)

        s = _norm(s).rstrip(";,.，。")
        # Drop trivial duplicates like identical actions repeated.
        if re.search(r"(Read the stem carefully\.)\s+\1", s):
            continue
        _add(s)

    return out[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260211)
    ap.add_argument("--max-len", type=int, default=140)
    ap.add_argument("--out", default="outputs/candidates/magic_1000.txt")
    args = ap.parse_args()

    n = int(args.n)
    if n <= 0:
        raise SystemExit("--n must be positive")

    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cands = generate_candidates(n=n, seed=int(args.seed), max_len=int(args.max_len))
    out_path.write_text("\n".join(cands) + "\n", encoding="utf-8")
    print(f"WROTE {len(cands)} candidates -> {out_path}")


if __name__ == "__main__":
    main()

