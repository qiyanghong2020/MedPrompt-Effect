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
from medprompt_effect.manuscript import ManuscriptContext, render_template, write_manuscript
from medprompt_effect.openrouter import OpenRouterClient
from medprompt_effect.utils import ensure_dir, load_dotenv


TITLE = "The MedPrompt Effect: prompt sensitivity and the limits of universal “magic prompts” in clinical reasoning models"


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
    prompt_name: str,
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
                    prompt_name=prompt_name,
                    prompt=prompt,
                    ex=item,
                    magic=magic,
                    max_tokens=max_tokens,
                    stop=None,
                )
                futures.append(fut)
                future_meta[fut] = (m, item)
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"run {prompt_name} magic={bool(magic)}"):
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
                        prompt_name=prompt_name,
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


def _score_from_gains(gains_pp: np.ndarray, invalid_diff_pp: float) -> float:
    """
    Universal-ish score: prioritize worst-case over mean, penalize invalid increases.
    """
    if gains_pp.size == 0:
        return -math.inf
    min_gain = float(np.nanmin(gains_pp))
    mean_gain = float(np.nanmean(gains_pp))
    invalid_pen = max(0.0, float(invalid_diff_pp)) * 0.25
    return min_gain + 0.10 * mean_gain - invalid_pen


def summarize_candidate(
    *,
    cand_id: str,
    cand_text: str,
    df_base: pd.DataFrame,
    df_magic: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    base_cond = an.agg_accuracy(df_base, by=["dataset", "model"]).rename(
        columns={"acc": "base_acc", "invalid_rate": "base_invalid"}
    )
    magic_cond = an.agg_accuracy(df_magic, by=["dataset", "model"]).rename(
        columns={"acc": "magic_acc", "invalid_rate": "magic_invalid"}
    )
    cond = base_cond.merge(magic_cond, on=["dataset", "model"], how="inner")
    cond["gain_pp"] = (cond["magic_acc"] - cond["base_acc"]) * 100.0
    gains_pp = cond["gain_pp"].to_numpy(dtype=float)

    base_acc = float(df_base["correct"].mean()) if len(df_base) else float("nan")
    magic_acc = float(df_magic["correct"].mean()) if len(df_magic) else float("nan")
    mean_gain_pp = (magic_acc - base_acc) * 100.0

    base_inv = float(df_base["pred"].isna().mean()) * 100.0 if len(df_base) else float("nan")
    magic_inv = float(df_magic["pred"].isna().mean()) * 100.0 if len(df_magic) else float("nan")
    invalid_diff_pp = magic_inv - base_inv

    # Paired bootstrap CI on per-item paired ids.
    df_a = df_base.copy()
    df_b = df_magic.copy()
    for d in (df_a, df_b):
        d["pair_id"] = d["dataset"].astype(str) + "|" + d["model"].astype(str) + "|" + d["example_id"].astype(str)
    mean_diff, (ci_lo, ci_hi) = an.paired_bootstrap_diff(df_a, df_b, id_col="pair_id", n_boot=n_boot, seed=seed)

    score = _score_from_gains(gains_pp=gains_pp, invalid_diff_pp=invalid_diff_pp)
    return {
        "candidate_id": cand_id,
        "candidate_text": cand_text,
        "score": score,
        "mean_gain_pp": float(mean_gain_pp),
        "min_gain_pp": float(np.nanmin(gains_pp)) if gains_pp.size else float("nan"),
        "p25_gain_pp": float(np.nanquantile(gains_pp, 0.25)) if gains_pp.size else float("nan"),
        "p50_gain_pp": float(np.nanquantile(gains_pp, 0.50)) if gains_pp.size else float("nan"),
        "p75_gain_pp": float(np.nanquantile(gains_pp, 0.75)) if gains_pp.size else float("nan"),
        "frac_nonneg": float(np.mean(gains_pp >= 0.0)) if gains_pp.size else float("nan"),
        "invalid_diff_pp": float(invalid_diff_pp),
        "bootstrap_mean_gain_pp": float(mean_diff * 100.0),
        "bootstrap_ci_low_pp": float(ci_lo * 100.0),
        "bootstrap_ci_high_pp": float(ci_hi * 100.0),
        "per_condition": cond[["dataset", "model", "base_acc", "magic_acc", "gain_pp", "base_invalid", "magic_invalid"]]
        .sort_values(["dataset", "model"])
        .to_dict(orient="records"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="openai/gpt-4.1,anthropic/claude-sonnet-4,google/gemini-2.5-flash")
    ap.add_argument(
        "--datasets",
        default="MedXpertQA,MedMCQA,MedQA-USMLE",
        help="Comma-separated subset of {MedXpertQA, MedMCQA, MedQA-USMLE}",
    )
    ap.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Magic prompt injection candidate text. May be provided multiple times. If any are provided, they replace the built-in candidate list.",
    )
    ap.add_argument(
        "--candidates-file",
        default=None,
        help="Path to a UTF-8 text file containing one candidate per line (blank lines and lines starting with # are ignored). If provided, candidates come from this file (plus any --candidate overrides).",
    )
    ap.add_argument("--seed", type=int, default=20260211)
    ap.add_argument("--dev-n-per-ds", type=int, default=6)
    ap.add_argument("--test-n-per-ds", type=int, default=12)
    ap.add_argument("--top-k", type=int, default=3)
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

    run_id = args.run_id or datetime.now().strftime("universal_%Y%m%d_%H%M%S")
    out_runs = ensure_dir(Path("outputs") / "runs" / run_id)
    out_aggs = ensure_dir(Path("outputs") / "aggregates" / run_id)
    ensure_dir("manuscript")

    rng = random.Random(int(args.seed))
    ds_sel = [d.strip() for d in str(args.datasets).split(",") if d.strip()]
    allowed = {"MedXpertQA", "MedMCQA", "MedQA-USMLE"}
    unknown = [d for d in ds_sel if d not in allowed]
    if unknown:
        raise SystemExit(f"Unknown datasets: {unknown}. Allowed: {sorted(allowed)}")
    if not ds_sel:
        raise SystemExit("No datasets selected")

    dev: list[data_mod.MCQExample] = []
    test: list[data_mod.MCQExample] = []
    ds_meta: list[dict[str, object]] = []

    if "MedXpertQA" in ds_sel:
        medx = data_mod.load_medxpertqa(split="test")
        dev_medx, test_medx = sample_disjoint(medx, int(args.dev_n_per_ds), int(args.test_n_per_ds), rng)
        dev += dev_medx
        test += test_medx
        ds_meta.append({"name": "MedXpertQA", "dev_n": len(dev_medx), "test_n": len(test_medx)})
    if "MedMCQA" in ds_sel:
        medmc = data_mod.load_medmcqa(split="validation")
        dev_medmc, test_medmc = sample_disjoint(medmc, int(args.dev_n_per_ds), int(args.test_n_per_ds), rng)
        dev += dev_medmc
        test += test_medmc
        ds_meta.append({"name": "MedMCQA", "dev_n": len(dev_medmc), "test_n": len(test_medmc)})
    if "MedQA-USMLE" in ds_sel:
        medqa = data_mod.load_medqa_usmle_4_options(split="test")
        dev_medqa, test_medqa = sample_disjoint(medqa, int(args.dev_n_per_ds), int(args.test_n_per_ds), rng)
        dev += dev_medqa
        test += test_medqa
        ds_meta.append({"name": "MedQA-USMLE", "dev_n": len(dev_medqa), "test_n": len(test_medqa)})

    baseline = prompts_mod.BASELINE_PROMPTS[0]

    meta = {
        "run_id": run_id,
        "seed": int(args.seed),
        "models": models,
        "datasets_selected": ds_sel,
        "dev_n_per_ds": int(args.dev_n_per_ds),
        "test_n_per_ds": int(args.test_n_per_ds),
        "datasets": ds_meta,
        "baseline_prompt": baseline.name,
    }
    (out_aggs / "search_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Baselines
    df_dev_base = run_grid(
        client=client,
        examples=dev,
        models=models,
        prompt_template=baseline,
        prompt_name=baseline.name,
        magic=None,
        out_jsonl=out_runs / "dev_baseline.jsonl",
        max_workers=int(args.max_workers),
        max_tokens=int(args.max_tokens),
    )
    df_test_base = run_grid(
        client=client,
        examples=test,
        models=models,
        prompt_template=baseline,
        prompt_name=baseline.name,
        magic=None,
        out_jsonl=out_runs / "test_baseline.jsonl",
        max_workers=int(args.max_workers),
        max_tokens=int(args.max_tokens),
    )

    candidates: list[str] = []
    if args.candidates_file:
        p = Path(str(args.candidates_file))
        if not p.exists():
            raise SystemExit(f"--candidates-file not found: {p}")
        for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            candidates.append(line)
    if args.candidate:
        # If explicit candidates are provided, prefer them (append) so users can override quickly.
        candidates += [str(c).strip() for c in args.candidate if str(c).strip()]
    if not candidates:
        candidates = list(prompts_mod.MAGIC_CANDIDATES)
        # Small, compliance-oriented additions (keep short).
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

    cand_map: dict[str, str] = {}
    for i, c in enumerate(candidates, start=1):
        cand_id = f"c{i:02d}_{_slug(c)}"
        cand_map[cand_id] = c
    (out_aggs / "candidates.json").write_text(json.dumps(cand_map, ensure_ascii=False, indent=2), encoding="utf-8")

    # Dev eval over candidates.
    dev_summaries: list[dict[str, object]] = []
    for cand_id, cand_text in cand_map.items():
        df_dev_magic = run_grid(
            client=client,
            examples=dev,
            models=models,
            prompt_template=baseline,
            prompt_name=baseline.name,
            magic=cand_text,
            out_jsonl=out_runs / f"dev_magic_{cand_id}.jsonl",
            max_workers=int(args.max_workers),
            max_tokens=int(args.max_tokens),
        )
        dev_summaries.append(
            summarize_candidate(
                cand_id=cand_id,
                cand_text=cand_text,
                df_base=df_dev_base,
                df_magic=df_dev_magic,
                n_boot=int(args.n_boot),
                seed=int(args.seed),
            )
        )

    df_dev = pd.DataFrame(
        [
            {k: v for k, v in s.items() if k != "per_condition"}
            for s in dev_summaries
        ]
    ).sort_values(["score", "mean_gain_pp"], ascending=False)
    df_dev.to_csv(out_aggs / "dev_candidate_scores.csv", index=False)

    # Pick top-K for test confirmation.
    top_k = int(args.top_k)
    picked = df_dev.head(top_k)["candidate_id"].tolist()

    test_summaries: list[dict[str, object]] = []
    for cand_id in picked:
        cand_text = cand_map[cand_id]
        df_test_magic = run_grid(
            client=client,
            examples=test,
            models=models,
            prompt_template=baseline,
            prompt_name=baseline.name,
            magic=cand_text,
            out_jsonl=out_runs / f"test_magic_{cand_id}.jsonl",
            max_workers=int(args.max_workers),
            max_tokens=int(args.max_tokens),
        )
        test_summaries.append(
            summarize_candidate(
                cand_id=cand_id,
                cand_text=cand_text,
                df_base=df_test_base,
                df_magic=df_test_magic,
                n_boot=int(args.n_boot),
                seed=int(args.seed),
            )
        )

    df_test = pd.DataFrame([{k: v for k, v in s.items() if k != "per_condition"} for s in test_summaries]).sort_values(
        ["score", "mean_gain_pp"], ascending=False
    )
    df_test.to_csv(out_aggs / "test_topk_scores.csv", index=False)

    best = test_summaries[0]
    if not df_test.empty:
        best_id = str(df_test.iloc[0]["candidate_id"])
        best = next(s for s in test_summaries if s["candidate_id"] == best_id)

    (out_aggs / "selected_magic.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")

    # Render a concise markdown report for quick reading.
    def _fmt_pp(x: float) -> str:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "nan"
        return f"{x:+.2f}"

    lines = []
    lines.append(f"# Universal magic prompt search (small-sample)\n")
    lines.append(f"- run_id: `{run_id}`")
    lines.append(f"- models: {', '.join(models)}")
    lines.append(f"- dev: {len(dev)} questions total; test: {len(test)} questions total")
    lines.append(f"- baseline prompt: `{baseline.name}`")
    lines.append("")
    lines.append("## Selected prompt (by test score)")
    lines.append(f"- prompt: `{best['candidate_text']}`")
    lines.append(f"- test mean gain (pp): {_fmt_pp(float(best['mean_gain_pp']))}")
    lines.append(
        f"- test bootstrap mean gain (pp): {_fmt_pp(float(best['bootstrap_mean_gain_pp']))} "
        f"[{_fmt_pp(float(best['bootstrap_ci_low_pp']))}, {_fmt_pp(float(best['bootstrap_ci_high_pp']))}]"
    )
    lines.append(f"- test min gain across dataset×model (pp): {_fmt_pp(float(best['min_gain_pp']))}")
    lines.append(f"- test invalid diff (pp): {_fmt_pp(float(best['invalid_diff_pp']))}")
    lines.append("")
    lines.append("## Per-condition gains (test)")
    per = pd.DataFrame(best["per_condition"])
    per["gain_pp"] = per["gain_pp"].astype(float)
    lines.append(per[["dataset", "model", "gain_pp"]].to_markdown(index=False, floatfmt=".2f"))
    lines.append("")
    lines.append("## Top candidates (dev)")
    top_dev = df_dev.head(min(8, len(df_dev)))[["candidate_id", "mean_gain_pp", "min_gain_pp", "frac_nonneg", "invalid_diff_pp", "score"]]
    lines.append(top_dev.to_markdown(index=False, floatfmt=".2f"))
    lines.append("")
    lines.append("## Top candidates (test)")
    top_test = df_test[["candidate_id", "mean_gain_pp", "min_gain_pp", "frac_nonneg", "invalid_diff_pp", "score"]]
    lines.append(top_test.to_markdown(index=False, floatfmt=".2f"))
    lines.append("")

    report_md = "\n".join(lines) + "\n"
    (out_aggs / "report.md").write_text(report_md, encoding="utf-8")

    # Optional: write a short manuscript stub referencing this run (kept separate from Nature_draft.md).
    ctx = ManuscriptContext(
        title=TITLE,
        magic_prompt=str(best["candidate_text"]),
        model_panel=models,
        datasets=meta["datasets"],
        key_results={
            "n_questions": len(test),
            "core_models_n": len(models),
            "magic_gain_mean": f"{float(best['bootstrap_mean_gain_pp']):.2f}",
            "magic_gain_ci_low": f"{float(best['bootstrap_ci_low_pp']):.2f}",
            "magic_gain_ci_high": f"{float(best['bootstrap_ci_high_pp']):.2f}",
        },
        tables=None,
    )
    ms = render_template("manuscript/template.md.j2", ctx)
    write_manuscript(out_aggs / "manuscript_stub.md", ms)

    print("DONE")
    print(f"Run id: {run_id}")
    print(f"Selected prompt: {best['candidate_text']}")
    print(f"Report: {out_aggs / 'report.md'}")


if __name__ == "__main__":
    main()
