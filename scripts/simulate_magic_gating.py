from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Literal

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medprompt_effect.utils import ensure_dir


GatingLevel = Literal["dataset", "model", "pair"]


def _to_bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f"}:
        return False
    return False


def load_paired_results(path: str | Path) -> tuple[pd.DataFrame, str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")

    df = pd.read_csv(p)
    for col in ("dataset", "model", "example_id", "magic", "correct"):
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in {p}")

    df = df.copy()
    df["magic"] = df["magic"].fillna("").astype(str)
    df["correct"] = df["correct"].map(_to_bool).astype(int)

    df_base = df[df["magic"].str.strip() == ""][["dataset", "model", "example_id", "correct"]].rename(columns={"correct": "base_correct"})
    df_magic = df[df["magic"].str.strip() != ""][["dataset", "model", "example_id", "correct", "magic"]].rename(
        columns={"correct": "magic_correct"}
    )

    magic_prompt = str(df_magic["magic"].dropna().unique().tolist())
    if len(df_magic["magic"].dropna().unique()) == 1:
        magic_prompt = str(df_magic["magic"].dropna().unique()[0])

    paired = df_base.merge(df_magic.drop(columns=["magic"]), on=["dataset", "model", "example_id"], how="inner")
    if paired.empty:
        raise ValueError(f"No paired rows after merge. Check that {p} contains baseline (magic='') and magic rows.")

    return paired, magic_prompt


def _setdiff(all_ids: np.ndarray, sub_ids: np.ndarray) -> np.ndarray:
    keep = np.ones(len(all_ids), dtype=bool)
    idx = {v: i for i, v in enumerate(all_ids)}
    for v in sub_ids:
        j = idx.get(v)
        if j is not None:
            keep[j] = False
    return all_ids[keep]


@dataclass(frozen=True)
class SplitRow:
    split: int
    baseline_acc: float
    policy_acc: float
    gain_pp: float


def simulate_gating(
    *,
    paired: pd.DataFrame,
    level: GatingLevel,
    calib_n: int,
    threshold_pp: float,
    n_splits: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))

    datasets = sorted(paired["dataset"].unique().tolist())
    models = sorted(paired["model"].unique().tolist())

    ds_ids: dict[str, np.ndarray] = {}
    for ds in datasets:
        ids = paired.loc[paired["dataset"] == ds, "example_id"].astype(str).unique()
        if calib_n >= len(ids):
            raise ValueError(f"calib_n={calib_n} must be < #examples for dataset {ds} (n={len(ids)})")
        ds_ids[ds] = np.array(sorted(ids.tolist()), dtype=object)

    split_rows: list[SplitRow] = []
    decision_rows: list[dict[str, object]] = []

    for s in range(int(n_splits)):
        # Decide where to use magic based on calibration set.
        if level == "dataset":
            decisions: dict[tuple[str, str] | str, bool] = {}
            calib_ids_by_ds = {ds: rng.choice(ds_ids[ds], size=int(calib_n), replace=False) for ds in datasets}
            for ds in datasets:
                calib = paired[(paired["dataset"] == ds) & (paired["example_id"].isin(calib_ids_by_ds[ds]))]
                gain_pp = float((calib["magic_correct"].mean() - calib["base_correct"].mean()) * 100.0)
                decisions[ds] = gain_pp > float(threshold_pp)

            holdout_parts = []
            for ds in datasets:
                holdout_ids = _setdiff(ds_ids[ds], calib_ids_by_ds[ds])
                hold = paired[(paired["dataset"] == ds) & (paired["example_id"].isin(holdout_ids))].copy()
                use_magic = bool(decisions[ds])
                hold["policy_correct"] = hold["magic_correct"] if use_magic else hold["base_correct"]
                holdout_parts.append(hold)

            holdout = pd.concat(holdout_parts, ignore_index=True)
            for ds in datasets:
                decision_rows.append({"split": s, "level": "dataset", "dataset": ds, "model": None, "use_magic": bool(decisions[ds])})

        elif level == "model":
            decisions = {}
            calib_ids_by_model_ds: dict[tuple[str, str], np.ndarray] = {}
            for m in models:
                for ds in datasets:
                    calib_ids_by_model_ds[(m, ds)] = rng.choice(ds_ids[ds], size=int(calib_n), replace=False)

                calib_parts = []
                for ds in datasets:
                    calib_parts.append(
                        paired[
                            (paired["model"] == m)
                            & (paired["dataset"] == ds)
                            & (paired["example_id"].isin(calib_ids_by_model_ds[(m, ds)]))
                        ]
                    )
                calib = pd.concat(calib_parts, ignore_index=True)
                gain_pp = float((calib["magic_correct"].mean() - calib["base_correct"].mean()) * 100.0)
                decisions[m] = gain_pp > float(threshold_pp)

            holdout_parts = []
            for m in models:
                for ds in datasets:
                    holdout_ids = _setdiff(ds_ids[ds], calib_ids_by_model_ds[(m, ds)])
                    hold = paired[
                        (paired["model"] == m) & (paired["dataset"] == ds) & (paired["example_id"].isin(holdout_ids))
                    ].copy()
                    use_magic = bool(decisions[m])
                    hold["policy_correct"] = hold["magic_correct"] if use_magic else hold["base_correct"]
                    holdout_parts.append(hold)

                decision_rows.append({"split": s, "level": "model", "dataset": None, "model": m, "use_magic": bool(decisions[m])})

            holdout = pd.concat(holdout_parts, ignore_index=True)

        elif level == "pair":
            decisions = {}
            calib_ids_by_pair: dict[tuple[str, str], np.ndarray] = {}
            holdout_parts = []
            for ds in datasets:
                for m in models:
                    calib_ids_by_pair[(ds, m)] = rng.choice(ds_ids[ds], size=int(calib_n), replace=False)
                    calib = paired[
                        (paired["dataset"] == ds) & (paired["model"] == m) & (paired["example_id"].isin(calib_ids_by_pair[(ds, m)]))
                    ]
                    gain_pp = float((calib["magic_correct"].mean() - calib["base_correct"].mean()) * 100.0)
                    decisions[(ds, m)] = gain_pp > float(threshold_pp)

                    holdout_ids = _setdiff(ds_ids[ds], calib_ids_by_pair[(ds, m)])
                    hold = paired[
                        (paired["dataset"] == ds) & (paired["model"] == m) & (paired["example_id"].isin(holdout_ids))
                    ].copy()
                    use_magic = bool(decisions[(ds, m)])
                    hold["policy_correct"] = hold["magic_correct"] if use_magic else hold["base_correct"]
                    holdout_parts.append(hold)
                    decision_rows.append({"split": s, "level": "pair", "dataset": ds, "model": m, "use_magic": use_magic})

            holdout = pd.concat(holdout_parts, ignore_index=True)

        else:
            raise ValueError(f"Unknown level: {level}")

        baseline_acc = float(holdout["base_correct"].mean())
        policy_acc = float(holdout["policy_correct"].mean())
        gain_pp = float((policy_acc - baseline_acc) * 100.0)
        split_rows.append(SplitRow(split=s, baseline_acc=baseline_acc, policy_acc=policy_acc, gain_pp=gain_pp))

    splits_df = pd.DataFrame([asdict(r) for r in split_rows])
    decisions_df = pd.DataFrame(decision_rows)
    return splits_df, decisions_df


def _fmt_pp(x: float) -> str:
    if np.isnan(x):
        return "nan"
    return f"{x:+.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run-id", default="20260212_080555", help="Run id under outputs/aggregates/<run_id>/test_all_raw.csv")
    ap.add_argument("--level", choices=["dataset", "model", "pair"], default="dataset")
    ap.add_argument("--calib-n", type=int, default=10)
    ap.add_argument("--threshold-pp", type=float, default=0.0, help="Use magic iff calibration gain_pp > threshold_pp")
    ap.add_argument("--n-splits", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260211)
    ap.add_argument("--out-id", default=None)
    args = ap.parse_args()

    source_run_id = str(args.source_run_id).strip()
    src_csv = Path("outputs") / "aggregates" / source_run_id / "test_all_raw.csv"
    paired, magic_prompt = load_paired_results(src_csv)

    out_id = args.out_id or datetime.now().strftime("gating_%Y%m%d_%H%M%S")
    out_dir = ensure_dir(Path("outputs") / "aggregates" / out_id)

    baseline_all = float(paired["base_correct"].mean())
    magic_all = float(paired["magic_correct"].mean())
    always_gain_pp = float((magic_all - baseline_all) * 100.0)

    splits_df, decisions_df = simulate_gating(
        paired=paired,
        level=str(args.level),  # type: ignore[arg-type]
        calib_n=int(args.calib_n),
        threshold_pp=float(args.threshold_pp),
        n_splits=int(args.n_splits),
        seed=int(args.seed),
    )

    gain = splits_df["gain_pp"].to_numpy(dtype=float)
    summary = {
        "out_id": out_id,
        "source_run_id": source_run_id,
        "source_csv": str(src_csv),
        "magic_prompt": magic_prompt,
        "level": str(args.level),
        "calib_n": int(args.calib_n),
        "threshold_pp": float(args.threshold_pp),
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "baseline_acc_all": baseline_all,
        "always_magic_acc_all": magic_all,
        "always_magic_gain_pp": always_gain_pp,
        "gating_gain_mean_pp": float(np.mean(gain)),
        "gating_gain_median_pp": float(np.median(gain)),
        "gating_gain_p05_pp": float(np.quantile(gain, 0.05)),
        "gating_gain_p95_pp": float(np.quantile(gain, 0.95)),
        "prob_negative": float(np.mean(gain < 0.0)),
        "prob_nonneg": float(np.mean(gain >= 0.0)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    splits_df.to_csv(out_dir / "splits.csv", index=False)

    if not decisions_df.empty:
        decisions_df.to_csv(out_dir / "decisions.csv", index=False)

        # Selection rates.
        rates = (
            decisions_df.groupby(["level", "dataset", "model"], dropna=False)
            .agg(use_magic_rate=("use_magic", "mean"))
            .reset_index()
            .sort_values(["level", "dataset", "model"], na_position="first")
        )
        rates.to_csv(out_dir / "selection_rates.csv", index=False)
    else:
        rates = pd.DataFrame()

    report_lines = [
        "# Magic prompt calibration-based selection simulation (offline)\n",
        f"- out_id: `{out_id}`",
        f"- source_run_id: `{source_run_id}`",
        f"- source_csv: `{src_csv}`",
        f"- magic_prompt: {magic_prompt}",
        f"- level: {args.level}",
        f"- calib_n: {int(args.calib_n)}",
        f"- threshold_pp: {float(args.threshold_pp):.2f}",
        f"- n_splits: {int(args.n_splits)}",
        "",
        "## Context (always-magic vs baseline on full test set)",
        f"- baseline acc: {baseline_all*100:.2f}%",
        f"- always-magic acc: {magic_all*100:.2f}%",
        f"- always-magic gain (pp): {_fmt_pp(always_gain_pp)}",
        "",
        "## Calibration-based selection policy (calibrate → decide → deploy)",
        f"- mean gain (pp): {_fmt_pp(summary['gating_gain_mean_pp'])}",
        f"- median gain (pp): {_fmt_pp(summary['gating_gain_median_pp'])}",
        f"- p05/p95 gain (pp): {_fmt_pp(summary['gating_gain_p05_pp'])} / {_fmt_pp(summary['gating_gain_p95_pp'])}",
        f"- P(gain < 0): {summary['prob_negative']*100:.1f}%",
        "",
    ]
    if not rates.empty:
        report_lines += [
            "## Selection rates",
            rates.to_markdown(index=False, floatfmt=".3f"),
            "",
        ]
    (out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("DONE")
    print(f"Out: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
