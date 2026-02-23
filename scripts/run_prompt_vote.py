from __future__ import annotations

import argparse
import json
import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

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
    magic: str | None,
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
            prompt = prompt_template.render(question=item.question, options=item.options, choices=item.choices, magic=magic)
            fut = ex.submit(
                run_one,
                client=client,
                model=model,
                prompt_name=prompt_template.name,
                prompt=prompt,
                ex=item,
                magic=magic,
                max_tokens=max_tokens,
                stop=None,
            )
            futures.append(fut)
            future_meta[fut] = (model, item)

        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"pairs {prompt_template.name} magic={bool(magic)}"):
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
                        magic=magic,
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
    return pd.DataFrame([asdict(r) for r in records])


def majority_vote(preds: Iterable[str | None]) -> str | None:
    counts: dict[str, int] = {}
    for p in preds:
        if not p:
            continue
        counts[p] = counts.get(p, 0) + 1
    if not counts:
        return None
    best = max(counts.values())
    winners = [k for k, v in counts.items() if v == best]
    if len(winners) == 1:
        return winners[0]
    return None


def build_report(
    *,
    run_id: str,
    models: list[str],
    datasets: list[dict[str, object]],
    prompt_names: list[str],
    base: pd.DataFrame,
    voted: pd.DataFrame,
    mean_gain_pp: float,
    ci_lo_pp: float,
    ci_hi_pp: float,
    extra_calls_mean: float,
) -> str:
    def _pct(x: float) -> str:
        return f"{x*100:.1f}"

    base_acc = float(base["correct"].mean()) if len(base) else float("nan")
    vote_acc = float(voted["correct"].mean()) if len(voted) else float("nan")
    base_inv = float(base["pred"].isna().mean()) if len(base) else float("nan")
    vote_inv = float(voted["pred"].isna().mean()) if len(voted) else float("nan")

    # Per dataset×model gain table.
    base_agg = an.agg_accuracy(base, by=["dataset", "model"]).rename(columns={"acc": "baseline_acc", "invalid_rate": "baseline_invalid"})
    vote_agg = an.agg_accuracy(voted, by=["dataset", "model"]).rename(columns={"acc": "vote_acc", "invalid_rate": "vote_invalid"})
    cond = base_agg.merge(vote_agg, on=["dataset", "model"], how="inner")
    cond["gain_pp"] = (cond["vote_acc"] - cond["baseline_acc"]) * 100.0
    cond_tbl = cond[["dataset", "model", "gain_pp"]].sort_values(["dataset", "model"]).to_markdown(index=False, floatfmt=".2f")

    lines = []
    lines.append("# PromptVote protocol (small-sample)\n")
    lines.append(f"- run_id: `{run_id}`")
    lines.append(f"- models: {', '.join(models)}")
    ds_str = ", ".join([f"{d['name']} (n={d['n']})" for d in datasets])
    lines.append(f"- datasets: {ds_str}")
    lines.append(f"- prompts voted: {', '.join([f'`{p}`' for p in prompt_names])}")
    lines.append("")
    lines.append("## Headline results")
    lines.append(f"- baseline acc: {_pct(base_acc)}% (invalid {_pct(base_inv)}%)")
    lines.append(f"- PromptVote acc: {_pct(vote_acc)}% (invalid {_pct(vote_inv)}%)")
    lines.append(f"- mean gain (pp): {mean_gain_pp:+.2f} (95% bootstrap CI {ci_lo_pp:+.2f} to {ci_hi_pp:+.2f})")
    lines.append(f"- mean extra calls per item: {extra_calls_mean:.2f}")
    lines.append("")
    lines.append("## Gains by dataset × model (pp)")
    lines.append(cond_tbl)
    lines.append("")
    return "\n".join(lines)


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

    run_id = args.run_id or datetime.now().strftime("vote_%Y%m%d_%H%M%S")
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

    # Build model×example pairs (keeps dataset mixed; pair_id will disambiguate).
    pairs = [(m, ex) for m in models for ex in test]

    p0, p1, p2 = prompts_mod.BASELINE_PROMPTS[0], prompts_mod.BASELINE_PROMPTS[1], prompts_mod.BASELINE_PROMPTS[2]
    prompt_names = [p0.name, p1.name, p2.name]

    # Stage 1: run p0 and p1 for all pairs.
    df_p0 = run_pairs(
        client=client,
        pairs=pairs,
        prompt_template=p0,
        magic=None,
        out_jsonl=out_runs / "vote_p0.jsonl",
        max_workers=int(args.max_workers),
        max_tokens=int(args.max_tokens),
    )
    df_p1 = run_pairs(
        client=client,
        pairs=pairs,
        prompt_template=p1,
        magic=None,
        out_jsonl=out_runs / "vote_p1.jsonl",
        max_workers=int(args.max_workers),
        max_tokens=int(args.max_tokens),
    )

    # Identify which pairs need a third prompt.
    for d in (df_p0, df_p1):
        d["pair_id"] = d["dataset"].astype(str) + "|" + d["model"].astype(str) + "|" + d["example_id"].astype(str)
    m0 = df_p0.set_index("pair_id")
    m1 = df_p1.set_index("pair_id")
    shared = m0.index.intersection(m1.index)
    need = []
    for pid in shared:
        a = m0.loc[pid, "pred"]
        b = m1.loc[pid, "pred"]
        if pd.isna(a) or pd.isna(b) or str(a) != str(b):
            # Need tie-break.
            need.append(pid)

    # Map pid -> (model, example)
    pid_to_pair: dict[str, tuple[str, data_mod.MCQExample]] = {}
    for model, ex in pairs:
        pid = f"{ex.dataset}|{model}|{ex.id}"
        pid_to_pair[pid] = (model, ex)

    need_pairs = [pid_to_pair[pid] for pid in need if pid in pid_to_pair]

    df_p2 = run_pairs(
        client=client,
        pairs=need_pairs,
        prompt_template=p2,
        magic=None,
        out_jsonl=out_runs / "vote_p2.jsonl",
        max_workers=int(args.max_workers),
        max_tokens=int(args.max_tokens),
    )
    if len(df_p2):
        df_p2["pair_id"] = df_p2["dataset"].astype(str) + "|" + df_p2["model"].astype(str) + "|" + df_p2["example_id"].astype(str)
    p2_map = df_p2.set_index("pair_id") if len(df_p2) else pd.DataFrame().set_index(pd.Index([]))

    # Aggregate into a single voted decision per pair_id.
    voted_rows = []
    extra_calls = []
    for pid in shared:
        r0 = m0.loc[pid]
        r1 = m1.loc[pid]
        a = None if pd.isna(r0["pred"]) else str(r0["pred"])
        b = None if pd.isna(r1["pred"]) else str(r1["pred"])
        c = None
        cost = 0.0
        prompt_tokens = 0.0
        completion_tokens = 0.0
        calls = 2
        for r in (r0, r1):
            if not pd.isna(r.get("cost")):
                cost += float(r["cost"])
            if not pd.isna(r.get("prompt_tokens")):
                prompt_tokens += float(r["prompt_tokens"])
            if not pd.isna(r.get("completion_tokens")):
                completion_tokens += float(r["completion_tokens"])

        if pid in p2_map.index:
            r2 = p2_map.loc[pid]
            c = None if pd.isna(r2["pred"]) else str(r2["pred"])
            calls = 3
            if not pd.isna(r2.get("cost")):
                cost += float(r2["cost"])
            if not pd.isna(r2.get("prompt_tokens")):
                prompt_tokens += float(r2["prompt_tokens"])
            if not pd.isna(r2.get("completion_tokens")):
                completion_tokens += float(r2["completion_tokens"])

        chosen = majority_vote([a, b, c])
        if chosen is None:
            # Tie-breaker: prefer p0 if valid; else first valid in order.
            chosen = a or b or c

        gold = str(r0["gold"]).strip().upper()
        correct = bool(chosen is not None and chosen == gold)
        voted_rows.append(
            {
                "pair_id": pid,
                "dataset": r0["dataset"],
                "example_id": r0["example_id"],
                "model": r0["model"],
                "pred": chosen,
                "gold": gold,
                "correct": correct,
                "cost": cost if cost > 0 else np.nan,
                "prompt_tokens": prompt_tokens if prompt_tokens > 0 else np.nan,
                "completion_tokens": completion_tokens if completion_tokens > 0 else np.nan,
                "calls": calls,
            }
        )
        extra_calls.append(calls - 1)  # baseline is 1 call (p0)

    df_vote = pd.DataFrame(voted_rows)

    # Baseline per pair (p0).
    base = df_p0.copy()
    base = base[base["pair_id"].isin(shared)].copy()

    # Bootstrap CI on paired diff (vote - baseline).
    vote_for_boot = df_vote.copy()
    vote_for_boot["pair_id"] = vote_for_boot["pair_id"].astype(str)
    base_for_boot = base.copy()
    base_for_boot["pair_id"] = base_for_boot["pair_id"].astype(str)
    mean_diff, (ci_lo, ci_hi) = an.paired_bootstrap_diff(
        base_for_boot.rename(columns={"pair_id": "id"}), vote_for_boot.rename(columns={"pair_id": "id"}), id_col="id", n_boot=int(args.n_boot), seed=int(args.seed)
    )

    mean_gain_pp = float(mean_diff * 100.0)
    ci_lo_pp = float(ci_lo * 100.0)
    ci_hi_pp = float(ci_hi * 100.0)
    extra_calls_mean = float(np.mean(extra_calls)) if extra_calls else float("nan")

    # Save aggregates.
    base.to_csv(out_aggs / "baseline_p0_raw.csv", index=False)
    df_vote.to_csv(out_aggs / "vote_raw.csv", index=False)

    # Summary table.
    base_agg = an.agg_accuracy(base, by=["dataset", "model"]).rename(columns={"acc": "baseline_acc", "invalid_rate": "baseline_invalid"})
    vote_agg = an.agg_accuracy(df_vote, by=["dataset", "model"]).rename(columns={"acc": "vote_acc", "invalid_rate": "vote_invalid"})
    summ = base_agg.merge(vote_agg, on=["dataset", "model"], how="inner")
    summ["gain_pp"] = (summ["vote_acc"] - summ["baseline_acc"]) * 100.0
    summ.to_csv(out_aggs / "vote_accuracy.csv", index=False)

    key = {
        "run_id": run_id,
        "seed": int(args.seed),
        "models": models,
        "datasets": datasets_meta,
        "prompts_voted": prompt_names,
        "n_pairs": int(len(shared)),
        "baseline_acc": float(base["correct"].mean()),
        "vote_acc": float(df_vote["correct"].mean()),
        "mean_gain_pp": mean_gain_pp,
        "ci_low_pp": ci_lo_pp,
        "ci_high_pp": ci_hi_pp,
        "extra_calls_mean": extra_calls_mean,
    }
    (out_aggs / "vote_key_results.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")

    report = build_report(
        run_id=run_id,
        models=models,
        datasets=datasets_meta,
        prompt_names=prompt_names,
        base=base,
        voted=df_vote,
        mean_gain_pp=mean_gain_pp,
        ci_lo_pp=ci_lo_pp,
        ci_hi_pp=ci_hi_pp,
        extra_calls_mean=extra_calls_mean,
    )
    (out_aggs / "report.md").write_text(report, encoding="utf-8")

    print("DONE")
    print(f"Run id: {run_id}")
    print(f"Report: {out_aggs / 'report.md'}")


if __name__ == "__main__":
    main()
