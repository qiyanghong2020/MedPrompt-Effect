from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
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


TITLE = "The MedPrompt Effect: A universal prompt-level amplification phenomenon in clinical reasoning models"

CORE_MODELS: list[str] = [
    "openai/gpt-4.1",
    "anthropic/claude-sonnet-4",
    "google/gemini-2.5-flash",
    "deepseek/deepseek-chat-v3.1",
    "qwen/qwen3-max",
]

EXTENDED_MODELS: list[str] = [
    "openai/gpt-5.2-pro",
    "anthropic/claude-opus-4.6",
    "meta-llama/llama-4-scout",
    "z-ai/glm-4.7-flash",
    "moonshotai/kimi-k2.5",
    "mistralai/mistral-small-creative",
]

SEED = 20260211


def sample_disjoint(
    examples: list[data_mod.MCQExample], n: int, *, used_ids: set[str], rng: random.Random
) -> list[data_mod.MCQExample]:
    pool = [e for e in examples if e.id not in used_ids]
    rng.shuffle(pool)
    picked = pool[:n]
    used_ids.update([e.id for e in picked])
    return picked


def run_grid(
    *,
    client: OpenRouterClient,
    examples: list[data_mod.MCQExample],
    models: list[str],
    prompt_template: prompts_mod.PromptTemplate,
    prompt_name: str,
    magic: str | None,
    out_jsonl: Path,
    max_workers: int = 8,
    max_tokens: int = 192,
) -> pd.DataFrame:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    stop = None

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
                    stop=stop,
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


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="Existing run id under outputs/runs/ and outputs/aggregates/")
    args = ap.parse_args()

    load_dotenv()

    run_id = str(args.run_id)
    out_runs = Path("outputs") / "runs" / run_id
    out_aggs = Path("outputs") / "aggregates" / run_id
    out_runs.mkdir(parents=True, exist_ok=True)
    out_aggs.mkdir(parents=True, exist_ok=True)

    magic_path = out_runs / "magic_search" / "selected_magic.json"
    if not magic_path.exists():
        raise SystemExit(f"Missing selected magic prompt: {magic_path}")
    magic = json.loads(magic_path.read_text(encoding="utf-8"))["magic"]

    client = OpenRouterClient()

    # Recreate the exact same sampling sequence as scripts/run_all.py for SEED=20260211.
    medx = data_mod.load_medxpertqa(split="test")
    medmc = data_mod.load_medmcqa(split="validation")
    medqa = data_mod.load_medqa_usmle_4_options(split="test")

    rng = random.Random(SEED)
    used: set[str] = set()
    _dev = sample_disjoint(medx, 30, used_ids=used, rng=rng)
    _sensitivity = sample_disjoint(medx, 30, used_ids=used, rng=rng)
    _test_medx = sample_disjoint(medx, 80, used_ids=used, rng=rng)
    _test_medmc = rng.sample(medmc, 60)
    test_medqa = rng.sample(medqa, 60)

    baseline = prompts_mod.BASELINE_PROMPTS[0]

    # Remove old MedQA logs (example_id was not unique previously) then rerun using cache.
    p_base = out_runs / "test_MedQA-USMLE_baseline.jsonl"
    p_magic = out_runs / "test_MedQA-USMLE_magic.jsonl"
    for p in (p_base, p_magic):
        if p.exists():
            p.unlink()

    df_base = run_grid(
        client=client,
        examples=test_medqa,
        models=CORE_MODELS,
        prompt_template=baseline,
        prompt_name=baseline.name,
        magic=None,
        out_jsonl=p_base,
        max_workers=8,
        max_tokens=192,
    )
    df_magic = run_grid(
        client=client,
        examples=test_medqa,
        models=CORE_MODELS,
        prompt_template=baseline,
        prompt_name=baseline.name,
        magic=magic,
        out_jsonl=p_magic,
        max_workers=8,
        max_tokens=192,
    )

    # Recompute combined test aggregates and manuscript.
    df_test = pd.concat(
        [
            read_jsonl(out_runs / "test_MedXpertQA_baseline.jsonl"),
            read_jsonl(out_runs / "test_MedXpertQA_magic.jsonl"),
            read_jsonl(out_runs / "test_MedMCQA_baseline.jsonl"),
            read_jsonl(out_runs / "test_MedMCQA_magic.jsonl"),
            df_base,
            df_magic,
        ],
        ignore_index=True,
    )
    df_test.to_csv(out_aggs / "test_all_raw.csv", index=False)

    test_agg = an.agg_accuracy(df_test, by=["dataset", "model", "magic"])
    test_agg.to_csv(out_aggs / "test_accuracy.csv", index=False)

    # Bootstrap CI of mean gain across all core models/items.
    df_base_all = df_test[df_test["magic"].isna()].copy()
    df_magic_all = df_test[df_test["magic"].notna()].copy()
    for d in (df_base_all, df_magic_all):
        d["pair_id"] = d["dataset"].astype(str) + "|" + d["model"].astype(str) + "|" + d["example_id"].astype(str)
    mean_diff, (ci_lo, ci_hi) = an.paired_bootstrap_diff(df_base_all, df_magic_all, id_col="pair_id", n_boot=2000, seed=SEED)

    inv_base = float(df_base_all["pred"].isna().mean()) * 100.0
    inv_magic = float(df_magic_all["pred"].isna().mean()) * 100.0

    # Tables for manuscript.
    def _mk_condition(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["condition"] = np.where(d["magic"].isna(), "baseline", "magic")
        return d

    core = _mk_condition(test_agg)
    core_pivot = (
        core.pivot_table(index=["dataset", "model"], columns="condition", values="acc", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if "baseline" in core_pivot.columns and "magic" in core_pivot.columns:
        core_pivot["gain_pp"] = (core_pivot["magic"] - core_pivot["baseline"]) * 100.0
    core_pivot["baseline_acc_%"] = core_pivot.get("baseline", np.nan) * 100.0
    core_pivot["magic_acc_%"] = core_pivot.get("magic", np.nan) * 100.0
    core_table = core_pivot[["dataset", "model", "baseline_acc_%", "magic_acc_%", "gain_pp"]].copy()

    ext_acc_path = out_aggs / "extended_medx_accuracy.csv"
    if not ext_acc_path.exists():
        raise SystemExit(f"Missing extended model results: {ext_acc_path}")
    ext_agg = pd.read_csv(ext_acc_path)
    ext = _mk_condition(ext_agg)
    ext_pivot = (
        ext.pivot_table(index=["model"], columns="condition", values="acc", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if "baseline" in ext_pivot.columns and "magic" in ext_pivot.columns:
        ext_pivot["gain_pp"] = (ext_pivot["magic"] - ext_pivot["baseline"]) * 100.0
    ext_pivot["baseline_acc_%"] = ext_pivot.get("baseline", np.nan) * 100.0
    ext_pivot["magic_acc_%"] = ext_pivot.get("magic", np.nan) * 100.0
    ext_table = ext_pivot[["model", "baseline_acc_%", "magic_acc_%", "gain_pp"]].copy()

    tables = {
        "core_results": core_table.to_markdown(index=False, floatfmt=".1f"),
        "extended_results": ext_table.to_markdown(index=False, floatfmt=".1f"),
    }

    # Prompt sensitivity range stats (already computed by scripts/run_all.py).
    sens_range_path = out_aggs / "prompt_sensitivity_range.csv"
    if not sens_range_path.exists():
        raise SystemExit(f"Missing prompt sensitivity stats: {sens_range_path}")
    sens_range = pd.read_csv(sens_range_path)
    prompt_ranges = sens_range["range_pp"].to_numpy()

    n_questions = 200
    key_results = {
        "n_questions": int(n_questions),
        "n_model_evals": int(len(df_test)),
        "core_models_n": int(len(CORE_MODELS)),
        "extended_models_n": int(len(EXTENDED_MODELS)),
        "total_models_n": int(len(CORE_MODELS) + len(EXTENDED_MODELS)),
        "prompt_range_median": f"{np.median(prompt_ranges):.1f}",
        "prompt_range_iqr_low": f"{np.quantile(prompt_ranges, 0.25):.1f}",
        "prompt_range_iqr_high": f"{np.quantile(prompt_ranges, 0.75):.1f}",
        "magic_gain_mean": f"{mean_diff*100.0:.1f}",
        "magic_gain_ci_low": f"{ci_lo*100.0:.1f}",
        "magic_gain_ci_high": f"{ci_hi*100.0:.1f}",
        "invalid_reduction": f"{(inv_base - inv_magic):.2f}",
    }

    ctx = ManuscriptContext(
        title=TITLE,
        magic_prompt=magic,
        model_panel=CORE_MODELS + EXTENDED_MODELS,
        datasets=[
            {"name": "MedXpertQA", "n": 80},
            {"name": "MedMCQA", "n": 60},
            {"name": "MedQA-USMLE", "n": 60},
        ],
        key_results=key_results,
        tables=tables,
    )
    manuscript = render_template("manuscript/template.md.j2", ctx)
    write_manuscript("manuscript/Nature_draft.md", manuscript)

    (out_aggs / "key_results.json").write_text(
        json.dumps({"run_id": run_id, "magic_prompt": magic, "key_results": key_results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("REFRESHED")
    print("Manuscript: manuscript/Nature_draft.md")
    print(f"Run id: {run_id}")


if __name__ == "__main__":
    main()

