from __future__ import annotations

import json
import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
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


CORE_MODELS: list[str] = [
    "openai/gpt-4.1",
    "anthropic/claude-sonnet-4",
    "google/gemini-2.5-flash",
    "deepseek/deepseek-chat-v3.1",
    "qwen/qwen3-max",
]

# Additional models for breadth (smaller eval subset).
EXTENDED_MODELS: list[str] = [
    "openai/gpt-5.2-pro",
    "anthropic/claude-opus-4.6",
    "meta-llama/llama-4-scout",
    "z-ai/glm-4.7-flash",
    "moonshotai/kimi-k2.5",
    "mistralai/mistral-small-creative",
]


SEED = 20260211
random.seed(SEED)
np.random.seed(SEED)


def sample_disjoint(examples: list[data_mod.MCQExample], n: int, *, used_ids: set[str], rng: random.Random) -> list[data_mod.MCQExample]:
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
    ensure_dir(out_jsonl.parent)
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


def pick_magic_prompt(
    *,
    client: OpenRouterClient,
    dev: list[data_mod.MCQExample],
    baseline_prompt: prompts_mod.PromptTemplate,
    baseline_name: str,
    models: list[str],
    out_dir: Path,
) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Baseline (no magic) for paired comparison.
    df_base = run_grid(
        client=client,
        examples=dev,
        models=models,
        prompt_template=baseline_prompt,
        prompt_name=baseline_name,
        magic=None,
        out_jsonl=out_dir / "dev_baseline.jsonl",
        max_workers=10,
    )
    base_acc = an.agg_accuracy(df_base, by=["model"]).set_index("model")["acc"].to_dict()

    best = None
    best_score = -math.inf
    for magic in prompts_mod.MAGIC_CANDIDATES:
        df_magic = run_grid(
            client=client,
            examples=dev,
            models=models,
            prompt_template=baseline_prompt,
            prompt_name=baseline_name,
            magic=magic,
            out_jsonl=out_dir / "dev_magic_candidates.jsonl",
            max_workers=10,
        )
        magic_acc = an.agg_accuracy(df_magic, by=["model"]).set_index("model")["acc"].to_dict()
        improvements = []
        for m in models:
            improvements.append(float(magic_acc.get(m, float("nan")) - base_acc.get(m, float("nan"))))
        score = float(np.nanmean(improvements))
        if score > best_score:
            best_score = score
            best = magic

    assert best is not None
    (out_dir / "selected_magic.json").write_text(json.dumps({"magic": best, "score": best_score}, ensure_ascii=False, indent=2), encoding="utf-8")
    return best


def main() -> None:
    load_dotenv()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_runs = ensure_dir(Path("outputs") / "runs" / run_id)
    out_aggs = ensure_dir(Path("outputs") / "aggregates" / run_id)
    figs = ensure_dir(Path("figures") / run_id)
    ensure_dir("manuscript")

    client = OpenRouterClient()

    # Load datasets
    medx = data_mod.load_medxpertqa(split="test")
    medmc = data_mod.load_medmcqa(split="validation")
    medqa = data_mod.load_medqa_usmle_4_options(split="test")

    rng = random.Random(SEED)
    used: set[str] = set()
    dev = sample_disjoint(medx, 30, used_ids=used, rng=rng)
    sensitivity = sample_disjoint(medx, 30, used_ids=used, rng=rng)
    test_medx = sample_disjoint(medx, 80, used_ids=used, rng=rng)
    test_medmc = rng.sample(medmc, 60)
    test_medqa = rng.sample(medqa, 60)

    # 1) Prompt sensitivity (meaning-equivalent paraphrases)
    dfs = []
    for tpl in prompts_mod.BASELINE_PROMPTS:
        df = run_grid(
            client=client,
            examples=sensitivity,
            models=CORE_MODELS,
            prompt_template=tpl,
            prompt_name=tpl.name,
            magic=None,
            out_jsonl=out_runs / "prompt_sensitivity.jsonl",
            max_workers=8,
            max_tokens=192,
        )
        dfs.append(df)
    df_sens = pd.concat(dfs, ignore_index=True)
    sens_agg = an.agg_accuracy(df_sens, by=["model", "prompt_name"])
    sens_agg.to_csv(out_aggs / "prompt_sensitivity.csv", index=False)

    # Within-model prompt range (pp)
    sens_range = (
        sens_agg.groupby("model")
        .agg(acc_min=("acc", "min"), acc_max=("acc", "max"))
        .assign(range_pp=lambda d: (d["acc_max"] - d["acc_min"]) * 100.0)
        .reset_index()
    )
    sens_range.to_csv(out_aggs / "prompt_sensitivity_range.csv", index=False)

    plt.figure(figsize=(10, 4))
    sns.boxplot(data=sens_range, x="range_pp")
    plt.title("Prompt sensitivity (within-model accuracy range, pp)")
    plt.xlabel("Accuracy range across paraphrases (percentage points)")
    plt.tight_layout()
    plt.savefig(figs / "prompt_sensitivity_range.png", dpi=200)
    plt.close()

    # 2) Magic prompt search on disjoint dev set
    baseline = prompts_mod.BASELINE_PROMPTS[0]
    magic = pick_magic_prompt(
        client=client,
        dev=dev,
        baseline_prompt=baseline,
        baseline_name=baseline.name,
        models=["openai/gpt-4.1", "anthropic/claude-sonnet-4"],
        out_dir=out_runs / "magic_search",
    )

    # 3) Final evaluation (baseline vs baseline+magic)
    all_test = [
        ("MedXpertQA", test_medx),
        ("MedMCQA", test_medmc),
        ("MedQA-USMLE", test_medqa),
    ]
    df_tests = []
    for ds_name, exs in all_test:
        df_base = run_grid(
            client=client,
            examples=exs,
            models=CORE_MODELS,
            prompt_template=baseline,
            prompt_name=baseline.name,
            magic=None,
            out_jsonl=out_runs / f"test_{ds_name}_baseline.jsonl",
            max_workers=8,
            max_tokens=192,
        )
        df_magic = run_grid(
            client=client,
            examples=exs,
            models=CORE_MODELS,
            prompt_template=baseline,
            prompt_name=baseline.name,
            magic=magic,
            out_jsonl=out_runs / f"test_{ds_name}_magic.jsonl",
            max_workers=8,
            max_tokens=192,
        )
        df_tests.append(df_base)
        df_tests.append(df_magic)
    df_test = pd.concat(df_tests, ignore_index=True)
    df_test.to_csv(out_aggs / "test_all_raw.csv", index=False)

    test_agg = an.agg_accuracy(df_test, by=["dataset", "model", "magic"])
    test_agg.to_csv(out_aggs / "test_accuracy.csv", index=False)

    # Compute bootstrap CI of mean gain across all core models/items
    df_base_all = df_test[df_test["magic"].isna()]
    df_magic_all = df_test[df_test["magic"].notna()]
    # Pair by (dataset, model, example_id) to avoid collisions across models/datasets.
    df_base_all = df_base_all.copy()
    df_magic_all = df_magic_all.copy()
    for d in (df_base_all, df_magic_all):
        d["pair_id"] = d["dataset"].astype(str) + "|" + d["model"].astype(str) + "|" + d["example_id"].astype(str)
    mean_diff, (ci_lo, ci_hi) = an.paired_bootstrap_diff(
        df_base_all, df_magic_all, id_col="pair_id", n_boot=2000, seed=SEED
    )

    # Invalid reduction (pp)
    inv_base = float(df_base_all["pred"].isna().mean()) * 100.0
    inv_magic = float(df_magic_all["pred"].isna().mean()) * 100.0

    # Figure: per-model improvement (pp) on MedXpertQA
    medx_base = an.agg_accuracy(df_base_all[df_base_all["dataset"] == "MedXpertQA"], by=["model"]).set_index("model")["acc"]
    medx_magic = an.agg_accuracy(df_magic_all[df_magic_all["dataset"] == "MedXpertQA"], by=["model"]).set_index("model")["acc"]
    imp = ((medx_magic - medx_base) * 100.0).reset_index()
    imp.columns = ["model", "improvement_pp"]
    imp.to_csv(out_aggs / "medx_model_improvement.csv", index=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=imp, y="model", x="improvement_pp", orient="h")
    plt.axvline(0, color="black", linewidth=1)
    plt.title("MedXpertQA: Magic prompt improvement by model (pp)")
    plt.xlabel("Accuracy gain (percentage points)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(figs / "medx_improvement_by_model.png", dpi=200)
    plt.close()

    # 4) Extended model breadth (MedXpertQA small subset)
    ext_subset = rng.sample(test_medx, 40)
    df_ext_base = run_grid(
        client=client,
        examples=ext_subset,
        models=EXTENDED_MODELS,
        prompt_template=baseline,
        prompt_name=baseline.name,
        magic=None,
        out_jsonl=out_runs / "extended_medx_baseline.jsonl",
        max_workers=8,
        max_tokens=192,
    )
    df_ext_magic = run_grid(
        client=client,
        examples=ext_subset,
        models=EXTENDED_MODELS,
        prompt_template=baseline,
        prompt_name=baseline.name,
        magic=magic,
        out_jsonl=out_runs / "extended_medx_magic.jsonl",
        max_workers=8,
        max_tokens=192,
    )
    df_ext = pd.concat([df_ext_base, df_ext_magic], ignore_index=True)
    ext_agg = an.agg_accuracy(df_ext, by=["model", "magic"])
    ext_agg.to_csv(out_aggs / "extended_medx_accuracy.csv", index=False)

    # Markdown tables for manuscript
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

    # Manuscript context
    n_questions = int(len(test_medx) + len(test_medmc) + len(test_medqa))
    prompt_ranges = sens_range["range_pp"].to_numpy()
    key_results = {
        "n_questions": n_questions,
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
            {"name": "MedXpertQA", "n": len(test_medx)},
            {"name": "MedMCQA", "n": len(test_medmc)},
            {"name": "MedQA-USMLE", "n": len(test_medqa)},
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

    print("DONE")
    print("Manuscript: manuscript/Nature_draft.md")
    print(f"Run id: {run_id}")


if __name__ == "__main__":
    main()
