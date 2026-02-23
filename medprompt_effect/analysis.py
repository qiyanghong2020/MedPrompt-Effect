from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Summary:
    n: int
    acc: float
    invalid_rate: float


def summarize(df: pd.DataFrame) -> Summary:
    n = int(len(df))
    acc = float(df["correct"].mean()) if n else float("nan")
    invalid_rate = float(df["pred"].isna().mean()) if n else float("nan")
    return Summary(n=n, acc=acc, invalid_rate=invalid_rate)


def agg_accuracy(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    out = (
        df.groupby(by, dropna=False)
        .agg(
            n=("correct", "size"),
            acc=("correct", "mean"),
            invalid_rate=("pred", lambda s: float(pd.isna(s).mean())),
            cost=("cost", "sum"),
            prompt_tokens=("prompt_tokens", "sum"),
            completion_tokens=("completion_tokens", "sum"),
        )
        .reset_index()
    )
    return out


def paired_bootstrap_diff(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    id_col: str = "example_id",
    metric_col: str = "correct",
    n_boot: int = 2000,
    seed: int = 20260211,
) -> tuple[float, tuple[float, float]]:
    """
    Paired bootstrap of mean(metric_b - metric_a) over shared ids.
    Returns (mean_diff, (ci_low, ci_high)).
    """
    a = df_a.set_index(id_col)[metric_col].astype(float)
    b = df_b.set_index(id_col)[metric_col].astype(float)
    shared = a.index.intersection(b.index)
    a = a.loc[shared].to_numpy()
    b = b.loc[shared].to_numpy()
    if len(shared) == 0:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(shared)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs.append(float((b[idx].mean() - a[idx].mean())))
    diffs = np.array(diffs)
    return float(diffs.mean()), (float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975)))


def save_markdown_table(df: pd.DataFrame, path: str | Path, floatfmt: str = ".3f") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(df.to_markdown(index=False, floatfmt=floatfmt), encoding="utf-8")

