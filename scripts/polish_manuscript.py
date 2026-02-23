from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from medprompt_effect.openrouter import ChatRequest, OpenRouterClient
from medprompt_effect.utils import ensure_dir, load_dotenv


@dataclass(frozen=True)
class FactsPack:
    run_id: str
    key_results: dict
    tables: dict[str, str]

    def render(self) -> str:
        kr = self.key_results.get("key_results", self.key_results)
        parts = [
            "FACTS (do not change; cite these numbers verbatim):",
            f"- run_id: {self.run_id}",
            f"- n_questions (test): {kr.get('n_questions')}",
            f"- core_models_n: {kr.get('core_models_n')}",
            f"- prompt_range_median_pp: {kr.get('prompt_range_median')} (IQR {kr.get('prompt_range_iqr_low')}–{kr.get('prompt_range_iqr_high')})",
            f"- magic_prompt: {self.key_results.get('magic_prompt')}",
            f"- magic_gain_mean_pp: {kr.get('magic_gain_mean')} (95% CI {kr.get('magic_gain_ci_low')} to {kr.get('magic_gain_ci_high')})",
            f"- invalid_reduction_pp (baseline − magic): {kr.get('invalid_reduction')}",
            "",
            "CORE TABLE (per dataset × model):",
            self.tables.get("core_results", "").strip(),
            "",
            "EXTENDED TABLE (MedXpertQA subset):",
            self.tables.get("extended_results", "").strip(),
        ]
        return "\n".join([p for p in parts if p is not None])


REVIEWER_SYSTEM = """You are Reviewer #2 for a top-tier journal (Nature-level).
Be rigorous, concrete, and constructive.

Goals:
1) Enforce internal logic + story continuity.
2) Enforce strict factual consistency with the provided FACTS and TABLES. Do not invent results.
3) Flag overclaims, missing controls, missing limitations, unclear methods, and unclear figures/tables.
4) Recommend edits that can be implemented immediately.

Output format:
- Major issues (bullet list, max 8)
- Minor issues (bullet list, max 8)
- Required edits (numbered list, max 10)
"""


WRITER_SYSTEM = """You are the Corresponding Author writing a Nature-level manuscript.

Hard constraints:
- Do NOT change any numbers in FACTS/TABLES. Do NOT add new results.
- Keep a coherent narrative arc: motivation → gap → approach → results → implications → limitations.
- Maintain technical precision: avoid causal/safety claims beyond evidence.
- Keep writing crisp, active voice, minimal jargon.

Task:
Revise the full manuscript in Markdown, implementing the Reviewer’s Required edits.
Return ONLY the revised manuscript (no preface, no commentary).
"""


def _chat(client: OpenRouterClient, *, model: str, system: str, user: str, temperature: float, max_tokens: int) -> str:
    req = ChatRequest(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    data = client.chat(req)
    msg = data["choices"][0]["message"]
    return (msg.get("content") or "").strip()


def load_facts_pack(run_id: str) -> FactsPack:
    aggs = Path("outputs") / "aggregates" / run_id
    key = json.loads((aggs / "key_results.json").read_text(encoding="utf-8"))

    core_tbl = pd.read_csv(aggs / "test_accuracy.csv")
    # Create the same markdown table used by scripts/run_all.py (dataset × model).
    d = core_tbl.copy()
    d["condition"] = d["magic"].apply(lambda x: "baseline" if pd.isna(x) else "magic")
    piv = (
        d.pivot_table(index=["dataset", "model"], columns="condition", values="acc", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    piv["gain_pp"] = (piv["magic"] - piv["baseline"]) * 100.0
    piv["baseline_acc_%"] = piv["baseline"] * 100.0
    piv["magic_acc_%"] = piv["magic"] * 100.0
    core_table = piv[["dataset", "model", "baseline_acc_%", "magic_acc_%", "gain_pp"]].to_markdown(
        index=False, floatfmt=".1f"
    )

    ext = pd.read_csv(aggs / "extended_medx_accuracy.csv")
    ext["condition"] = ext["magic"].apply(lambda x: "baseline" if pd.isna(x) else "magic")
    ep = (
        ext.pivot_table(index=["model"], columns="condition", values="acc", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    ep["gain_pp"] = (ep["magic"] - ep["baseline"]) * 100.0
    ep["baseline_acc_%"] = ep["baseline"] * 100.0
    ep["magic_acc_%"] = ep["magic"] * 100.0
    ext_table = ep[["model", "baseline_acc_%", "magic_acc_%", "gain_pp"]].to_markdown(index=False, floatfmt=".1f")

    return FactsPack(run_id=run_id, key_results=key, tables={"core_results": core_table, "extended_results": ext_table})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="Run id under outputs/aggregates/<run_id>/")
    ap.add_argument("--input", default="manuscript/Nature_draft.md")
    ap.add_argument("--output", default="manuscript/Nature_final.md")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--reviewer-model", default="anthropic/claude-sonnet-4")
    ap.add_argument("--writer-model", default="openai/gpt-4.1")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--artifacts-dir", default=None, help="Directory to store intermediate drafts/reviews.")
    args = ap.parse_args()

    load_dotenv()
    client = OpenRouterClient()

    facts = load_facts_pack(args.run_id)
    manuscript = Path(args.input).read_text(encoding="utf-8")

    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else Path("manuscript") / "polish_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_dir(artifacts_dir)

    for r in range(1, args.rounds + 1):
        review_prompt = "\n\n".join(
            [
                facts.render(),
                "",
                "MANUSCRIPT TO REVIEW:",
                manuscript,
            ]
        )
        review = _chat(
            client,
            model=args.reviewer_model,
            system=REVIEWER_SYSTEM,
            user=review_prompt,
            temperature=0.0,
            max_tokens=args.max_tokens,
        )
        (artifacts_dir / f"round_{r:02d}_review.md").write_text(review + "\n", encoding="utf-8")

        writer_prompt = "\n\n".join(
            [
                facts.render(),
                "",
                "REVIEWER FEEDBACK:",
                review,
                "",
                "CURRENT MANUSCRIPT:",
                manuscript,
            ]
        )
        manuscript = _chat(
            client,
            model=args.writer_model,
            system=WRITER_SYSTEM,
            user=writer_prompt,
            temperature=0.2,
            max_tokens=args.max_tokens,
        )
        (artifacts_dir / f"round_{r:02d}_draft.md").write_text(manuscript + "\n", encoding="utf-8")

    Path(args.output).write_text(manuscript + "\n", encoding="utf-8")
    print(f"DONE: {args.output}")
    print(f"Artifacts: {artifacts_dir}")


if __name__ == "__main__":
    main()

