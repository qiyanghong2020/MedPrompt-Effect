from __future__ import annotations

import argparse
import json
import math
import random
import re
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


def _slug(text: str, max_len: int = 48) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    if not s:
        s = "magic"
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s


def sample_disjoint(examples: list[data_mod.MCQExample], n_dev: int, n_test: int, rng: random.Random) -> tuple[list[data_mod.MCQExample], list[data_mod.MCQExample]]:
    if n_dev < 0 or n_test < 0:
        raise ValueError("n_dev and n_test must be non-negative")
    if n_dev + n_test > len(examples):
        raise ValueError(f"Requested n_dev+n_test={n_dev+n_test} but dataset only has {len(examples)} items")
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    dev = [examples[i] for i in idx[:n_dev]]
    test = [examples[i] for i in idx[n_dev : n_dev + n_test]]
    return dev, test


def run_grid(
    *,
    client: OpenRouterClient,
    examples: list[data_mod.MCQExample],
    models: list[str],
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
        for m in models:
            for item in examples:
                prompt = prompt_template.render(question=item.question, options=item.options, choices=item.choices, magic=magic)
                fut = ex.submit(
                    run_one,
                    client=client,
                    model=m,
                    prompt_name=prompt_template.name,
                    prompt=prompt,
                    ex=item,
                    magic=magic,
                    max_tokens=max_tokens,
                    stop=None,
                )
                futures.append(fut)
                future_meta[fut] = (m, item)

        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"grid {prompt_template.name} magic={bool(magic)}"):
            try:
                records.append(fut.result())
            except Exception as e:
                model, item = future_meta.get(fut, ("<unknown>", examples[0]))
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


def _score(gains_pp: np.ndarray, invalid_diff_pp: float) -> float:
    if gains_pp.size == 0:
        return -math.inf
    min_gain = float(np.nanmin(gains_pp))
    mean_gain = float(np.nanmean(gains_pp))
    invalid_pen = max(0.0, float(invalid_diff_pp)) * 0.25
    return min_gain + 0.10 * mean_gain - invalid_pen


def summarize_candidate(
    *,
    df_base: pd.DataFrame,
    df_magic: pd.DataFrame,
    models: list[str],
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    base_acc = an.agg_accuracy(df_base, by=["model"]).set_index("model")["acc"].to_dict()
    magic_acc = an.agg_accuracy(df_magic, by=["model"]).set_index("model")["acc"].to_dict()
    gains_pp = np.array([(float(magic_acc.get(m, float("nan")) - base_acc.get(m, float("nan"))) * 100.0) for m in models], dtype=float)

    base_inv = float(df_base["pred"].isna().mean()) * 100.0 if len(df_base) else float("nan")
    magic_inv = float(df_magic["pred"].isna().mean()) * 100.0 if len(df_magic) else float("nan")
    invalid_diff_pp = magic_inv - base_inv

    # Bootstrap CI of mean gain over model×item pairs.
    a = df_base.copy()
    b = df_magic.copy()
    for d in (a, b):
        d["pair_id"] = d["dataset"].astype(str) + "|" + d["model"].astype(str) + "|" + d["example_id"].astype(str)
    mean_diff, (ci_lo, ci_hi) = an.paired_bootstrap_diff(a, b, id_col="pair_id", n_boot=n_boot, seed=seed)

    return {
        "score": _score(gains_pp=gains_pp, invalid_diff_pp=invalid_diff_pp),
        "mean_gain_pp": float(np.nanmean(gains_pp)),
        "min_gain_pp": float(np.nanmin(gains_pp)),
        "invalid_diff_pp": float(invalid_diff_pp),
        "boot_mean_gain_pp": float(mean_diff * 100.0),
        "boot_ci_low_pp": float(ci_lo * 100.0),
        "boot_ci_high_pp": float(ci_hi * 100.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="openai/gpt-4.1,anthropic/claude-sonnet-4,google/gemini-2.5-flash")
    ap.add_argument("--seed", type=int, default=20260211)
    ap.add_argument("--dev-n-per-ds", type=int, default=6)
    ap.add_argument("--test-n-per-ds", type=int, default=12)
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

    run_id = args.run_id or datetime.now().strftime("policy_%Y%m%d_%H%M%S")
    out_runs = ensure_dir(Path("outputs") / "runs" / run_id)
    out_aggs = ensure_dir(Path("outputs") / "aggregates" / run_id)

    baseline = prompts_mod.BASELINE_PROMPTS[0]

    rng = random.Random(int(args.seed))
    medx = data_mod.load_medxpertqa(split="test")
    medmc = data_mod.load_medmcqa(split="validation")
    medqa = data_mod.load_medqa_usmle_4_options(split="test")

    dev_medx, test_medx = sample_disjoint(medx, int(args.dev_n_per_ds), int(args.test_n_per_ds), rng)
    dev_medmc, test_medmc = sample_disjoint(medmc, int(args.dev_n_per_ds), int(args.test_n_per_ds), rng)
    dev_medqa, test_medqa = sample_disjoint(medqa, int(args.dev_n_per_ds), int(args.test_n_per_ds), rng)

    per_ds = {
        "MedXpertQA": {"dev": dev_medx, "test": test_medx},
        "MedMCQA": {"dev": dev_medmc, "test": test_medmc},
        "MedQA-USMLE": {"dev": dev_medqa, "test": test_medqa},
    }

    candidates = list(prompts_mod.MAGIC_CANDIDATES)
    candidates += [
        "Follow the required output format strictly; be concise.",
        "Be systematic: extract key clues, eliminate distractors, then answer.",
        "Double-check what the question is asking before answering.",
        "Think step by step internally, but keep the final response concise.",
    ]
    # De-duplicate while preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    candidates = uniq

    cand_ids = {f"c{i:02d}_{_slug(c)}": c for i, c in enumerate(candidates, start=1)}
    (out_aggs / "candidates.json").write_text(json.dumps(cand_ids, ensure_ascii=False, indent=2), encoding="utf-8")

    # Dev baselines per dataset.
    dev_base: dict[str, pd.DataFrame] = {}
    for ds_name, pack in per_ds.items():
        dev_base[ds_name] = run_grid(
            client=client,
            examples=pack["dev"],
            models=models,
            prompt_template=baseline,
            magic=None,
            out_jsonl=out_runs / f"dev_{ds_name}_baseline.jsonl",
            max_workers=int(args.max_workers),
            max_tokens=int(args.max_tokens),
        )

    # Select per-dataset injection (or None) on dev.
    selected: dict[str, dict[str, object]] = {}
    for ds_name, pack in per_ds.items():
        df_base = dev_base[ds_name]
        best = {"candidate_id": None, "candidate_text": None, "score": 0.0, "mean_gain_pp": 0.0, "min_gain_pp": 0.0, "invalid_diff_pp": 0.0}
        # Candidate = None baseline.
        best.update({"candidate_id": "none", "candidate_text": None, "score": 0.0})

        rows = []
        for cid, ctext in cand_ids.items():
            df_magic = run_grid(
                client=client,
                examples=pack["dev"],
                models=models,
                prompt_template=baseline,
                magic=ctext,
                out_jsonl=out_runs / f"dev_{ds_name}_magic_{cid}.jsonl",
                max_workers=int(args.max_workers),
                max_tokens=int(args.max_tokens),
            )
            summ = summarize_candidate(df_base=df_base, df_magic=df_magic, models=models, n_boot=int(args.n_boot), seed=int(args.seed))
            rows.append({"dataset": ds_name, "candidate_id": cid, "candidate_text": ctext, **summ})

            if float(summ["score"]) > float(best["score"]):
                best = {"candidate_id": cid, "candidate_text": ctext, **summ}

        pd.DataFrame(rows).sort_values(["score", "mean_gain_pp"], ascending=False).to_csv(out_aggs / f"dev_scores_{ds_name}.csv", index=False)
        selected[ds_name] = best

    (out_aggs / "selected_policy.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")

    # Test baselines per dataset.
    test_base: dict[str, pd.DataFrame] = {}
    test_magic: dict[str, pd.DataFrame] = {}
    for ds_name, pack in per_ds.items():
        test_base[ds_name] = run_grid(
            client=client,
            examples=pack["test"],
            models=models,
            prompt_template=baseline,
            magic=None,
            out_jsonl=out_runs / f"test_{ds_name}_baseline.jsonl",
            max_workers=int(args.max_workers),
            max_tokens=int(args.max_tokens),
        )
        sel_text = selected[ds_name].get("candidate_text")
        if sel_text:
            test_magic[ds_name] = run_grid(
                client=client,
                examples=pack["test"],
                models=models,
                prompt_template=baseline,
                magic=str(sel_text),
                out_jsonl=out_runs / f"test_{ds_name}_policy_magic.jsonl",
                max_workers=int(args.max_workers),
                max_tokens=int(args.max_tokens),
            )

    # Combine into overall baseline vs policy on test (paired by dataset×model×example).
    dfs_base = []
    dfs_policy = []
    for ds_name in per_ds.keys():
        b = test_base[ds_name].copy()
        b["pair_id"] = b["dataset"].astype(str) + "|" + b["model"].astype(str) + "|" + b["example_id"].astype(str)
        dfs_base.append(b)
        if ds_name in test_magic:
            m = test_magic[ds_name].copy()
            m["pair_id"] = m["dataset"].astype(str) + "|" + m["model"].astype(str) + "|" + m["example_id"].astype(str)
            dfs_policy.append(m)
        else:
            dfs_policy.append(b)

    df_base_all = pd.concat(dfs_base, ignore_index=True)
    df_policy_all = pd.concat(dfs_policy, ignore_index=True)

    # Paired bootstrap CI of mean(policy - baseline).
    mean_diff, (ci_lo, ci_hi) = an.paired_bootstrap_diff(df_base_all, df_policy_all, id_col="pair_id", n_boot=int(args.n_boot), seed=int(args.seed))

    base_acc = float(df_base_all["correct"].mean())
    policy_acc = float(df_policy_all["correct"].mean())
    base_inv = float(df_base_all["pred"].isna().mean()) * 100.0
    policy_inv = float(df_policy_all["pred"].isna().mean()) * 100.0

    key = {
        "run_id": run_id,
        "seed": int(args.seed),
        "models": models,
        "dev_n_per_ds": int(args.dev_n_per_ds),
        "test_n_per_ds": int(args.test_n_per_ds),
        "selected_policy": {k: v.get("candidate_text") for k, v in selected.items()},
        "baseline_acc": base_acc,
        "policy_acc": policy_acc,
        "gain_pp": float((policy_acc - base_acc) * 100.0),
        "boot_mean_gain_pp": float(mean_diff * 100.0),
        "boot_ci_low_pp": float(ci_lo * 100.0),
        "boot_ci_high_pp": float(ci_hi * 100.0),
        "baseline_invalid_pp": base_inv,
        "policy_invalid_pp": policy_inv,
    }
    (out_aggs / "policy_key_results.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")

    # Per dataset results.
    ds_rows = []
    for ds_name in per_ds.keys():
        b = test_base[ds_name]
        p = test_magic.get(ds_name, b)
        ds_rows.append(
            {
                "dataset": ds_name,
                "selected_magic": selected[ds_name].get("candidate_text"),
                "baseline_acc": float(b["correct"].mean()),
                "policy_acc": float(p["correct"].mean()),
                "gain_pp": float((p["correct"].mean() - b["correct"].mean()) * 100.0),
            }
        )
    pd.DataFrame(ds_rows).to_csv(out_aggs / "test_dataset_summary.csv", index=False)

    # Markdown report.
    ds_str = ", ".join([f"{k}→{(v.get('candidate_text') or 'None')}" for k, v in selected.items()])
    report = "\n".join(
        [
            "# Magic prompt policy (per-dataset) — small-sample\n",
            f"- run_id: `{run_id}`",
            f"- models: {', '.join(models)}",
            f"- dev/test per dataset: {int(args.dev_n_per_ds)}/{int(args.test_n_per_ds)}",
            f"- selected policy: {ds_str}",
            "",
            "## Test results (policy vs baseline)",
            f"- baseline acc: {base_acc*100:.1f}% (invalid {base_inv:.2f}pp)",
            f"- policy acc: {policy_acc*100:.1f}% (invalid {policy_inv:.2f}pp)",
            f"- gain (pp): {(policy_acc-base_acc)*100:+.2f}",
            f"- bootstrap mean gain (pp): {mean_diff*100:+.2f} [{ci_lo*100:+.2f}, {ci_hi*100:+.2f}]",
            "",
        ]
    )
    (out_aggs / "report.md").write_text(report, encoding="utf-8")

    print("DONE")
    print(f"Run id: {run_id}")
    print(f"Report: {out_aggs / 'report.md'}")


if __name__ == "__main__":
    main()

