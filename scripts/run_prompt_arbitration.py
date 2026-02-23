from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medprompt_effect import analysis as an
from medprompt_effect import data as data_mod
from medprompt_effect import prompts as prompts_mod
from medprompt_effect.eval import RunRecord, run_one, write_jsonl
from medprompt_effect.openrouter import OpenRouterClient
from medprompt_effect.utils import ensure_dir, load_dotenv


def sample_n(examples: list[data_mod.MCQExample], n: int, rng: random.Random) -> list[data_mod.MCQExample]:
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return []
    if n > len(examples):
        raise ValueError(f"Requested n={n} but dataset only has {len(examples)} items")
    return rng.sample(examples, n)


def run_pairs(
    *,
    client: OpenRouterClient,
    pairs: list[tuple[str, data_mod.MCQExample]],
    prompt_template: prompts_mod.PromptTemplate,
    out_jsonl: Path,
    max_workers: int,
    max_tokens: int,
) -> pd.DataFrame:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if out_jsonl.exists():
        out_jsonl.unlink()

    futures = []
    future_meta: dict[object, tuple[str, data_mod.MCQExample]] = {}
    records: list[RunRecord] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for model, item in pairs:
            prompt = prompt_template.render(question=item.question, options=item.options, choices=item.choices, magic=None)
            fut = ex.submit(
                run_one,
                client=client,
                model=model,
                prompt_name=prompt_template.name,
                prompt=prompt,
                ex=item,
                magic=None,
                max_tokens=max_tokens,
                stop=None,
            )
            futures.append(fut)
            future_meta[fut] = (model, item)

        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"pairs {prompt_template.name}"):
            try:
                records.append(fut.result())
            except Exception as e:
                model, item = future_meta.get(fut, ("<unknown>", pairs[0][1]))
                records.append(
                    RunRecord(
                        ts=float("nan"),
                        dataset=item.dataset,
                        example_id=item.id,
                        model=model,
                        prompt_name=prompt_template.name,
                        magic=None,
                        gold=item.answer,
                        pred=None,
                        correct=False,
                        output_text=f"ERROR: {type(e).__name__}: {e}",
                        cost=None,
                        prompt_tokens=None,
                        completion_tokens=None,
                    )
                )

    write_jsonl(out_jsonl, records)
    df = pd.DataFrame([asdict(r) for r in records])
    df["pair_id"] = df["dataset"].astype(str) + "|" + df["model"].astype(str) + "|" + df["example_id"].astype(str)
    return df


def make_adjudication_prompt(*, question: str, options: str, choices: list[str], cand_a: str, cand_b: str) -> str:
    choice_str = "/".join([c.strip().upper() for c in choices if c and c.strip()])
    return (
        "You are adjudicating two candidate answers to a multiple-choice medical question.\n"
        f"Candidate A (from Prompt 0): {cand_a}\n"
        f"Candidate B (from Prompt 1): {cand_b}\n\n"
        "Pick the single best answer.\n"
        "If you cannot justify choosing B over A, choose A.\n\n"
        "The FIRST line of your response must be the final answer in this exact format:\n"
        f"FINAL: <{choice_str}>\n"
        "Then provide one short sentence of justification.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{options}\n"
    )


def run_adjudication_pairs(
    *,
    client: OpenRouterClient,
    items: list[tuple[str, data_mod.MCQExample, str, str]],
    out_jsonl: Path,
    max_workers: int,
    max_tokens: int,
) -> pd.DataFrame:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if out_jsonl.exists():
        out_jsonl.unlink()

    futures = []
    future_meta: dict[object, tuple[str, data_mod.MCQExample]] = {}
    records: list[RunRecord] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for model, ex_item, cand_a, cand_b in items:
            prompt = make_adjudication_prompt(
                question=ex_item.question,
                options=ex_item.options,
                choices=ex_item.choices,
                cand_a=cand_a,
                cand_b=cand_b,
            )
            fut = ex.submit(
                run_one,
                client=client,
                model=model,
                prompt_name="adjudicate_p0_vs_p1",
                prompt=prompt,
                ex=ex_item,
                magic=None,
                max_tokens=max_tokens,
                stop=None,
            )
            futures.append(fut)
            future_meta[fut] = (model, ex_item)

        for fut in tqdm(as_completed(futures), total=len(futures), desc="pairs adjudicate"):
            try:
                records.append(fut.result())
            except Exception as e:
                model, ex_item = future_meta.get(fut, ("<unknown>", items[0][1]))
                records.append(
                    RunRecord(
                        ts=float("nan"),
                        dataset=ex_item.dataset,
                        example_id=ex_item.id,
                        model=model,
                        prompt_name="adjudicate_p0_vs_p1",
                        magic=None,
                        gold=ex_item.answer,
                        pred=None,
                        correct=False,
                        output_text=f"ERROR: {type(e).__name__}: {e}",
                        cost=None,
                        prompt_tokens=None,
                        completion_tokens=None,
                    )
                )

    write_jsonl(out_jsonl, records)
    df = pd.DataFrame([asdict(r) for r in records])
    df["pair_id"] = df["dataset"].astype(str) + "|" + df["model"].astype(str) + "|" + df["example_id"].astype(str)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="openai/gpt-4.1,anthropic/claude-sonnet-4,google/gemini-2.5-flash")
    ap.add_argument("--seed", type=int, default=20260211)
    ap.add_argument("--n-medx", type=int, default=12)
    ap.add_argument("--n-medmc", type=int, default=12)
    ap.add_argument("--n-medqa", type=int, default=12)
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    load_dotenv()
    client = OpenRouterClient()

    models = [m.strip() for m in str(args.models).split(",") if m.strip()]
    if not models:
        raise SystemExit("No models provided")

    run_id = args.run_id or datetime.now().strftime("arb_%Y%m%d_%H%M%S")
    out_runs = ensure_dir(Path("outputs") / "runs" / run_id)
    out_aggs = ensure_dir(Path("outputs") / "aggregates" / run_id)

    rng = random.Random(int(args.seed))
    test: list[data_mod.MCQExample] = []
    datasets_meta: list[dict[str, object]] = []

    if int(args.n_medx) > 0:
        medx = data_mod.load_medxpertqa(split="test")
        picked = sample_n(medx, int(args.n_medx), rng)
        test += picked
        datasets_meta.append({"name": "MedXpertQA", "n": len(picked)})
    if int(args.n_medmc) > 0:
        medmc = data_mod.load_medmcqa(split="validation")
        picked = sample_n(medmc, int(args.n_medmc), rng)
        test += picked
        datasets_meta.append({"name": "MedMCQA", "n": len(picked)})
    if int(args.n_medqa) > 0:
        medqa = data_mod.load_medqa_usmle_4_options(split="test")
        picked = sample_n(medqa, int(args.n_medqa), rng)
        test += picked
        datasets_meta.append({"name": "MedQA-USMLE", "n": len(picked)})

    if not test:
        raise SystemExit("No test examples selected (all n=0)")

    pairs = [(m, ex) for m in models for ex in test]
    p0 = prompts_mod.BASELINE_PROMPTS[0]
    p1 = prompts_mod.BASELINE_PROMPTS[1]

    df_p0 = run_pairs(
        client=client,
        pairs=pairs,
        prompt_template=p0,
        out_jsonl=out_runs / "arb_p0.jsonl",
        max_workers=int(args.max_workers),
        max_tokens=int(args.max_tokens),
    )
    df_p1 = run_pairs(
        client=client,
        pairs=pairs,
        prompt_template=p1,
        out_jsonl=out_runs / "arb_p1.jsonl",
        max_workers=int(args.max_workers),
        max_tokens=int(args.max_tokens),
    )

    m0 = df_p0.set_index("pair_id")
    m1 = df_p1.set_index("pair_id")
    shared = m0.index.intersection(m1.index)

    # Build adjudication list for disagreements/invalids.
    pid_to_pair: dict[str, tuple[str, data_mod.MCQExample]] = {}
    for model, ex in pairs:
        pid_to_pair[f"{ex.dataset}|{model}|{ex.id}"] = (model, ex)

    adjudicate: list[tuple[str, data_mod.MCQExample, str, str]] = []
    for pid in shared:
        a = m0.loc[pid, "pred"]
        b = m1.loc[pid, "pred"]
        a_s = None if pd.isna(a) else str(a)
        b_s = None if pd.isna(b) else str(b)
        if not a_s or not b_s or a_s != b_s:
            model, ex_item = pid_to_pair[pid]
            adjudicate.append((model, ex_item, a_s or "?", b_s or "?"))

    df_adj = run_adjudication_pairs(
        client=client,
        items=adjudicate,
        out_jsonl=out_runs / "arb_adjudicate.jsonl",
        max_workers=int(args.max_workers),
        max_tokens=int(args.max_tokens),
    )
    adj_map = df_adj.set_index("pair_id") if len(df_adj) else pd.DataFrame().set_index(pd.Index([]))

    # Protocol decision per pair.
    rows = []
    for pid in shared:
        r0 = m0.loc[pid]
        r1 = m1.loc[pid]
        a = None if pd.isna(r0["pred"]) else str(r0["pred"])
        b = None if pd.isna(r1["pred"]) else str(r1["pred"])

        chosen = None
        calls = 2
        cost = 0.0
        prompt_tokens = 0.0
        completion_tokens = 0.0
        for r in (r0, r1):
            if not pd.isna(r.get("cost")):
                cost += float(r["cost"])
            if not pd.isna(r.get("prompt_tokens")):
                prompt_tokens += float(r["prompt_tokens"])
            if not pd.isna(r.get("completion_tokens")):
                completion_tokens += float(r["completion_tokens"])

        if a and b and a == b:
            chosen = a
        else:
            if pid in adj_map.index:
                ra = adj_map.loc[pid]
                c = None if pd.isna(ra["pred"]) else str(ra["pred"])
                calls = 3
                if not pd.isna(ra.get("cost")):
                    cost += float(ra["cost"])
                if not pd.isna(ra.get("prompt_tokens")):
                    prompt_tokens += float(ra["prompt_tokens"])
                if not pd.isna(ra.get("completion_tokens")):
                    completion_tokens += float(ra["completion_tokens"])
                chosen = c

        chosen = chosen or a or b
        gold = str(r0["gold"]).strip().upper()
        correct = bool(chosen is not None and chosen == gold)
        rows.append(
            {
                "pair_id": pid,
                "dataset": r0["dataset"],
                "example_id": r0["example_id"],
                "model": r0["model"],
                "pred": chosen,
                "gold": gold,
                "correct": correct,
                "calls": calls,
                "cost": cost if cost > 0 else np.nan,
                "prompt_tokens": prompt_tokens if prompt_tokens > 0 else np.nan,
                "completion_tokens": completion_tokens if completion_tokens > 0 else np.nan,
            }
        )

    df_proto = pd.DataFrame(rows)
    base = df_p0[df_p0["pair_id"].isin(shared)].copy()

    # Bootstrap CI of mean gain.
    mean_diff, (ci_lo, ci_hi) = an.paired_bootstrap_diff(base, df_proto, id_col="pair_id", n_boot=int(args.n_boot), seed=int(args.seed))

    # Summary table.
    base_agg = an.agg_accuracy(base, by=["dataset", "model"]).rename(columns={"acc": "baseline_acc", "invalid_rate": "baseline_invalid"})
    proto_agg = an.agg_accuracy(df_proto, by=["dataset", "model"]).rename(columns={"acc": "protocol_acc", "invalid_rate": "protocol_invalid"})
    summ = base_agg.merge(proto_agg, on=["dataset", "model"], how="inner")
    summ["gain_pp"] = (summ["protocol_acc"] - summ["baseline_acc"]) * 100.0
    summ.to_csv(out_aggs / "arb_accuracy.csv", index=False)

    disagree_mask = (m0.loc[shared, "pred"].fillna("") != m1.loc[shared, "pred"].fillna("")).to_numpy(dtype=bool)
    disagreement_rate = float(disagree_mask.mean()) if len(disagree_mask) else float("nan")

    key = {
        "run_id": run_id,
        "seed": int(args.seed),
        "models": models,
        "datasets": datasets_meta,
        "n_pairs": int(len(shared)),
        "baseline_acc": float(base["correct"].mean()),
        "protocol_acc": float(df_proto["correct"].mean()),
        "mean_gain_pp": float(mean_diff * 100.0),
        "ci_low_pp": float(ci_lo * 100.0),
        "ci_high_pp": float(ci_hi * 100.0),
        "disagreement_rate": disagreement_rate,
        "adjudication_calls_n": int(len(df_adj)),
    }
    (out_aggs / "arb_key_results.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")

    ds_str = ", ".join([f"{d['name']} (n={d['n']})" for d in datasets_meta])
    report_lines = [
        "# Prompt arbitration protocol (small-sample)\n",
        f"- run_id: `{run_id}`",
        f"- models: {', '.join(models)}",
        f"- datasets: {ds_str}",
        "- protocol: run `p0_final_rationale` and `p1_exam_rationale`; adjudicate only when they disagree (bias toward Candidate A = p0).",
        "",
        "## Headline results",
        f"- baseline acc (p0): {key['baseline_acc']*100:.1f}%",
        f"- protocol acc: {key['protocol_acc']*100:.1f}%",
        f"- mean gain (pp): {key['mean_gain_pp']:+.2f} (95% CI {key['ci_low_pp']:+.2f} to {key['ci_high_pp']:+.2f})",
        f"- disagreements: {key['disagreement_rate']*100:.1f}% (adjudication calls n={key['adjudication_calls_n']})",
        "",
        "## Gains by dataset × model (pp)",
        summ[["dataset", "model", "gain_pp"]].sort_values(["dataset", "model"]).to_markdown(index=False, floatfmt=".2f"),
        "",
    ]
    (out_aggs / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    df_proto.to_csv(out_aggs / "protocol_raw.csv", index=False)
    base.to_csv(out_aggs / "baseline_p0_raw.csv", index=False)

    print("DONE")
    print(f"Run id: {run_id}")
    print(f"Report: {out_aggs / 'report.md'}")


if __name__ == "__main__":
    main()

