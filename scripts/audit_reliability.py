from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medprompt_effect import prompts as prompts_mod
from medprompt_effect.utils import ensure_dir


def kendall_tau_from_ranks(rank_a: np.ndarray, rank_b: np.ndarray) -> float:
    if len(rank_a) != len(rank_b):
        raise ValueError("rank arrays must be the same length")
    n = int(len(rank_a))
    if n < 2:
        return float("nan")

    conc = 0
    disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (rank_a[i] - rank_a[j]) * (rank_b[i] - rank_b[j])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    denom = n * (n - 1) / 2
    return float((conc - disc) / denom) if denom else float("nan")


def compute_prompt_ranks(sens: pd.DataFrame, prompt_order: list[str]) -> pd.DataFrame:
    need = {"model", "prompt_name", "acc"}
    missing = need - set(sens.columns)
    if missing:
        raise ValueError(f"prompt_sensitivity.csv missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    models = sorted(sens["model"].unique().tolist())
    for p in prompt_order:
        g = sens[sens["prompt_name"] == p][["model", "acc"]].copy()
        if g.empty:
            continue
        # Deterministic ranking: accuracy desc, then model name.
        g = g.sort_values(["acc", "model"], ascending=[False, True]).reset_index(drop=True)
        g["rank"] = np.arange(1, len(g) + 1)
        for _, r in g.iterrows():
            rows.append({"prompt_name": p, "model": r["model"], "acc": float(r["acc"]), "rank": int(r["rank"])})

    out = pd.DataFrame(rows)
    out["prompt_name"] = pd.Categorical(out["prompt_name"], categories=prompt_order, ordered=True)
    out["model"] = pd.Categorical(out["model"], categories=models, ordered=True)
    return out.sort_values(["prompt_name", "rank", "model"]).reset_index(drop=True)


def compute_rank_stability_metrics(prompt_ranks: pd.DataFrame) -> tuple[dict[str, float | int], pd.DataFrame]:
    prompts = [str(p) for p in prompt_ranks["prompt_name"].cat.categories if p in set(prompt_ranks["prompt_name"])]
    models = [str(m) for m in prompt_ranks["model"].cat.categories if m in set(prompt_ranks["model"])]

    rank_pivot = prompt_ranks.pivot(index="model", columns="prompt_name", values="rank").loc[models, prompts]
    acc_pivot = prompt_ranks.pivot(index="model", columns="prompt_name", values="acc").loc[models, prompts]

    # Kendall tau across prompt pairs.
    tau_rows = []
    taus = []
    for p1, p2 in combinations(prompts, 2):
        ra = rank_pivot[p1].to_numpy(dtype=float)
        rb = rank_pivot[p2].to_numpy(dtype=float)
        tau = kendall_tau_from_ranks(ra, rb)
        taus.append(tau)
        tau_rows.append({"prompt_a": p1, "prompt_b": p2, "kendall_tau": tau})
    tau_df = pd.DataFrame(tau_rows).sort_values(["kendall_tau", "prompt_a", "prompt_b"], ascending=[True, True, True])

    # Pairwise order flips: for each (model_a, model_b), check if relative ordering changes across prompts.
    flip_any = 0
    for a, b in combinations(models, 2):
        rel = []
        for p in prompts:
            rel.append(int(rank_pivot.loc[a, p] < rank_pivot.loc[b, p]))
        if any(rel) and not all(rel):
            flip_any += 1

    # Global flip probability across prompt pairs × model pairs, excluding ties in accuracy.
    flip_events = 0
    flip_total = 0
    for p1, p2 in combinations(prompts, 2):
        for a, b in combinations(models, 2):
            d1 = float(acc_pivot.loc[a, p1] - acc_pivot.loc[b, p1])
            d2 = float(acc_pivot.loc[a, p2] - acc_pivot.loc[b, p2])
            if d1 == 0.0 or d2 == 0.0:
                continue
            flip_total += 1
            if (d1 > 0) != (d2 > 0):
                flip_events += 1
    flip_rate = float(flip_events / flip_total) if flip_total else float("nan")

    # Rank movement (max - min) per model.
    rank_movement = (rank_pivot.max(axis=1) - rank_pivot.min(axis=1)).astype(float)

    # Distinct top-1 across prompts.
    top_models = []
    for p in prompts:
        top = prompt_ranks[prompt_ranks["prompt_name"] == p].sort_values(["rank", "model"]).iloc[0]["model"]
        top_models.append(str(top))
    unique_top = len(set(top_models))

    # Within-model range vs between-model gap (pp).
    within_ranges_pp = (acc_pivot.max(axis=1) - acc_pivot.min(axis=1)) * 100.0

    between_gaps_pp = []
    for p in prompts:
        vals = acc_pivot[p].to_numpy(dtype=float)
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                between_gaps_pp.append(abs(vals[i] - vals[j]) * 100.0)

    metrics: dict[str, float | int] = {
        "n_models": int(len(models)),
        "n_prompts": int(len(prompts)),
        "kendall_tau_median": float(np.median(taus)) if taus else float("nan"),
        "kendall_tau_min": float(np.min(taus)) if taus else float("nan"),
        "kendall_tau_max": float(np.max(taus)) if taus else float("nan"),
        "model_pairs_flip_any": int(flip_any),
        "model_pairs_total": int(len(list(combinations(models, 2)))),
        "pairwise_flip_rate": float(flip_rate),
        "unique_top_models": int(unique_top),
        "max_rank_change": int(rank_movement.max()) if len(rank_movement) else 0,
        "median_within_model_range_pp": float(np.median(within_ranges_pp)) if len(within_ranges_pp) else float("nan"),
        "median_between_model_gap_pp": float(np.median(between_gaps_pp)) if between_gaps_pp else float("nan"),
    }
    return metrics, tau_df


def plot_leaderboard_instability(prompt_ranks: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_order = list(prompt_ranks["prompt_name"].cat.categories)
    prompt_order = [p for p in prompt_order if p in set(prompt_ranks["prompt_name"])]
    models = [str(m) for m in prompt_ranks["model"].cat.categories if m in set(prompt_ranks["model"])]

    # Build pivots in display order.
    acc = prompt_ranks.pivot(index="model", columns="prompt_name", values="acc").loc[models, prompt_order]
    ranks = prompt_ranks.pivot(index="model", columns="prompt_name", values="rank").loc[models, prompt_order]

    palette = sns.color_palette("tab10", n_colors=len(models))
    model_to_color = {m: palette[i] for i, m in enumerate(models)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"wspace": 0.25})

    xs = np.arange(len(prompt_order))

    # Left: accuracy slopegraph.
    ax = axes[0]
    for m in models:
        ax.plot(xs, acc.loc[m].to_numpy(dtype=float) * 100.0, marker="o", linewidth=2, color=model_to_color[m], label=m)
    ax.set_title("Accuracy varies across prompt paraphrases")
    ax.set_xticks(xs, prompt_order, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, axis="y", alpha=0.3)

    # Right: rank slopegraph (1=best).
    ax = axes[1]
    for m in models:
        ax.plot(xs, ranks.loc[m].to_numpy(dtype=float), marker="o", linewidth=2, color=model_to_color[m], label=m)
    ax.set_title("Leaderboard ranks can shift")
    ax.set_xticks(xs, prompt_order, rotation=20, ha="right")
    ax.set_ylabel("Rank (1=best)")
    ax.set_yticks(np.arange(1, len(models) + 1))
    ax.invert_yaxis()
    ax.grid(True, axis="y", alpha=0.3)

    # Shared legend.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(models), 3), frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_paired_test(df_test_all_raw: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    need = {"dataset", "model", "example_id", "magic", "correct"}
    missing = need - set(df_test_all_raw.columns)
    if missing:
        raise ValueError(f"test_all_raw.csv missing columns: {sorted(missing)}")

    df = df_test_all_raw.copy()
    df["magic"] = df["magic"].fillna("").astype(str)
    df["correct"] = df["correct"].astype(bool).astype(int)

    base = df[df["magic"].str.strip() == ""][["dataset", "model", "example_id", "correct"]].rename(columns={"correct": "base_correct"})
    mag = df[df["magic"].str.strip() != ""][["dataset", "model", "example_id", "correct", "magic"]].rename(columns={"correct": "magic_correct"})

    magic_prompt = str(mag["magic"].dropna().unique().tolist())
    if len(mag["magic"].dropna().unique()) == 1:
        magic_prompt = str(mag["magic"].dropna().unique()[0])

    paired = base.merge(mag.drop(columns=["magic"]), on=["dataset", "model", "example_id"], how="inner")
    if paired.empty:
        raise ValueError("No paired rows after merge (baseline vs magic)")
    return paired, magic_prompt


@dataclass(frozen=True)
class SelectionSummary:
    k: int
    mean_gain_pp: float
    median_gain_pp: float
    p05_gain_pp: float
    p95_gain_pp: float
    prob_negative: float


def simulate_dataset_selection(
    *,
    paired: pd.DataFrame,
    k: int,
    threshold_pp: float,
    n_splits: int,
    seed: int,
) -> SelectionSummary:
    rng = np.random.default_rng(int(seed))

    datasets = sorted(paired["dataset"].unique().tolist())
    ds_ids: dict[str, np.ndarray] = {}
    for ds in datasets:
        ids = paired.loc[paired["dataset"] == ds, "example_id"].astype(str).unique()
        if k >= len(ids):
            raise ValueError(f"k={k} must be < #examples for dataset {ds} (n={len(ids)})")
        ds_ids[ds] = np.array(sorted(ids.tolist()), dtype=object)

    gains = []
    for _ in range(int(n_splits)):
        holdout_parts = []
        for ds in datasets:
            calib_ids = rng.choice(ds_ids[ds], size=int(k), replace=False)
            calib = paired[(paired["dataset"] == ds) & (paired["example_id"].isin(calib_ids))]
            calib_gain_pp = float((calib["magic_correct"].mean() - calib["base_correct"].mean()) * 100.0)
            use_magic = calib_gain_pp > float(threshold_pp)

            holdout_ids = np.setdiff1d(ds_ids[ds], calib_ids, assume_unique=False)
            hold = paired[(paired["dataset"] == ds) & (paired["example_id"].isin(holdout_ids))].copy()
            hold["policy_correct"] = hold["magic_correct"] if use_magic else hold["base_correct"]
            holdout_parts.append(hold)

        holdout = pd.concat(holdout_parts, ignore_index=True)
        baseline_acc = float(holdout["base_correct"].mean())
        policy_acc = float(holdout["policy_correct"].mean())
        gains.append(float((policy_acc - baseline_acc) * 100.0))

    arr = np.asarray(gains, dtype=float)
    return SelectionSummary(
        k=int(k),
        mean_gain_pp=float(np.mean(arr)),
        median_gain_pp=float(np.median(arr)),
        p05_gain_pp=float(np.quantile(arr, 0.05)),
        p95_gain_pp=float(np.quantile(arr, 0.95)),
        prob_negative=float(np.mean(arr < 0.0)),
    )


def plot_calibration_curve(curve: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(7.5, 4.0))
    x = curve["k"].to_numpy(dtype=float)
    mean = curve["mean_gain_pp"].to_numpy(dtype=float)
    p05 = curve["p05_gain_pp"].to_numpy(dtype=float)
    p95 = curve["p95_gain_pp"].to_numpy(dtype=float)
    prob_neg = curve["prob_negative"].to_numpy(dtype=float) * 100.0

    ax1.plot(x, mean, marker="o", linewidth=2, color="#1f77b4", label="E[gain] (pp)")
    ax1.fill_between(x, p05, p95, alpha=0.2, color="#1f77b4", label="p05–p95 (pp)")
    ax1.axhline(0, color="black", linewidth=1)
    ax1.set_xlabel("Calibration size per dataset (k)")
    ax1.set_ylabel("Gain vs baseline (pp)")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, prob_neg, marker="s", linewidth=2, color="#d62728", label="P(drawdown)")
    ax2.set_ylabel("P(gain < 0) (%)")
    ax2.set_ylim(0, 100)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=False)

    ax1.set_title("Calibration-based selective injection: expected gain vs drawdown risk")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="Run id under outputs/aggregates/<run_id>/")
    ap.add_argument("--seed", type=int, default=20260211)
    ap.add_argument("--n-splits", type=int, default=5000)
    ap.add_argument("--threshold-pp", type=float, default=0.0)
    ap.add_argument("--k-min", type=int, default=5)
    ap.add_argument("--k-max", type=int, default=None)
    ap.add_argument("--k-step", type=int, default=5)
    args = ap.parse_args()

    run_id = str(args.run_id).strip()
    out_aggs = ensure_dir(Path("outputs") / "aggregates" / run_id)
    out_figs = ensure_dir(Path("figures") / run_id)

    prompt_order = [tpl.name for tpl in prompts_mod.BASELINE_PROMPTS]

    sens_csv = out_aggs / "prompt_sensitivity.csv"
    if not sens_csv.exists():
        raise SystemExit(f"Missing: {sens_csv}")
    sens = pd.read_csv(sens_csv)

    prompt_ranks = compute_prompt_ranks(sens, prompt_order=prompt_order)
    prompt_ranks.to_csv(out_aggs / "prompt_rankings.csv", index=False)

    metrics, tau_df = compute_rank_stability_metrics(prompt_ranks)
    (out_aggs / "rank_stability.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    tau_df.to_csv(out_aggs / "rank_kendall_tau.csv", index=False)

    plot_leaderboard_instability(prompt_ranks, out_figs / "leaderboard_instability.png")

    report_lines = [
        "# Benchmark reliability audit (prompt as measurement instrument)\n",
        f"- run_id: `{run_id}`",
        "",
        "## Leaderboard stability across paraphrased prompts",
        f"- models: {metrics['n_models']}",
        f"- prompts: {metrics['n_prompts']}",
        f"- median within-model range (pp): {metrics['median_within_model_range_pp']:.2f}",
        f"- median between-model gap (pp): {metrics['median_between_model_gap_pp']:.2f}",
        f"- model pairs with any rank reversal: {metrics['model_pairs_flip_any']}/{metrics['model_pairs_total']}",
        f"- pairwise flip rate across prompt pairs: {metrics['pairwise_flip_rate']:.3f}",
        f"- Kendall τ (median/min/max): {metrics['kendall_tau_median']:.2f}/{metrics['kendall_tau_min']:.2f}/{metrics['kendall_tau_max']:.2f}",
        "",
        f"- figure: `{out_figs / 'leaderboard_instability.png'}`",
        "",
    ]

    test_csv = out_aggs / "test_all_raw.csv"
    if test_csv.exists():
        df_test = pd.read_csv(test_csv)
        paired, magic_prompt = build_paired_test(df_test)

        # Determine max k from dataset sizes unless user overrides.
        ds_sizes = paired.groupby("dataset")["example_id"].nunique().to_dict()
        max_allowed = int(min(ds_sizes.values()) - 1)
        k_max = int(args.k_max) if args.k_max is not None else max_allowed
        k_max = min(k_max, max_allowed)
        if int(args.k_min) > k_max:
            raise SystemExit(f"k-min={int(args.k_min)} must be <= k-max={k_max} (max allowed {max_allowed})")

        k_vals = list(range(int(args.k_min), k_max + 1, int(args.k_step)))
        if not k_vals:
            k_vals = [int(args.k_min)]

        curve_rows = []
        for k in k_vals:
            summ = simulate_dataset_selection(
                paired=paired,
                k=int(k),
                threshold_pp=float(args.threshold_pp),
                n_splits=int(args.n_splits),
                seed=int(args.seed),
            )
            curve_rows.append(
                {
                    "k": summ.k,
                    "mean_gain_pp": summ.mean_gain_pp,
                    "median_gain_pp": summ.median_gain_pp,
                    "p05_gain_pp": summ.p05_gain_pp,
                    "p95_gain_pp": summ.p95_gain_pp,
                    "prob_negative": summ.prob_negative,
                }
            )

        curve = pd.DataFrame(curve_rows).sort_values("k")
        curve.to_csv(out_aggs / "calibration_curve.csv", index=False)
        plot_calibration_curve(curve, out_figs / "calibration_curve.png")

        report_lines += [
            "## Calibration-based selective injection (offline simulation)",
            f"- magic prompt (from test runs): {magic_prompt}",
            f"- datasets: {', '.join([f'{k} (n={v})' for k, v in ds_sizes.items()])}",
            f"- n_splits: {int(args.n_splits)} (seed {int(args.seed)})",
            f"- threshold: {float(args.threshold_pp):.2f} pp",
            f"- curve: `{out_aggs / 'calibration_curve.csv'}`",
            f"- figure: `{out_figs / 'calibration_curve.png'}`",
            "",
        ]

    (out_aggs / "audit_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("DONE")
    print(f"Report: {out_aggs / 'audit_report.md'}")


if __name__ == "__main__":
    main()
