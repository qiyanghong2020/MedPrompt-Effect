# The MedPrompt Effect

This repo studies **prompt sensitivity** in clinical multiple-choice question (MCQ) benchmarks and tests the limits of short, generic **“magic prompt”** injections that are often claimed to universally improve reasoning.

The working design doc is in Chinese: `项目设计.md`.

## What’s included

- **Prompt sensitivity**: meaning-aligned prompt paraphrases can induce within-model accuracy swings (“MedPrompt Effect”).
- **Universal magic prompt**: select a short injection on a disjoint dev set, then evaluate on held-out test sets.
- **Benchmark reliability audit**: leaderboard stability metrics (rank reversals, Kendall’s τ, within-model range vs between-model gap).
- **Risk-controlled deployment idea (offline)**: calibration-based *selective injection* simulation to trade off expected gain vs drawdown risk.
- **Manuscript generation**: render a Nature-style draft from a Jinja2 template; optionally polish with an LLM reviewer/writer loop.

## Quickstart

### 1) Install

Requires **Python 3.10+**.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure API keys

Create a `.env` file (it is git-ignored) or export env vars:

```bash
OPENROUTER_API_KEY=...
# Optional:
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
HF_TOKEN=...  # for Hugging Face datasets if needed
```

### 3) Run the end-to-end pipeline

```bash
python scripts/run_all.py
```

This will:
1) run prompt-paraphrase sensitivity (MedXpertQA)
2) pick a “magic prompt” on a disjoint dev split
3) evaluate baseline vs baseline+magic on MedXpertQA / MedMCQA / MedQA-USMLE
4) generate figures and a manuscript draft

## Useful scripts

### Reliability audit (offline; no extra model calls)

Run after you have an `outputs/aggregates/<RUN_ID>/prompt_sensitivity.csv` (e.g., from `run_all.py`):

```bash
python scripts/audit_reliability.py --run-id <RUN_ID>
```

Outputs an audit report and plots under `outputs/aggregates/<RUN_ID>/` and `figures/<RUN_ID>/`.

### Calibration-based selective injection (offline; no extra model calls)

Uses paired baseline vs magic results from `outputs/aggregates/<RUN_ID>/test_all_raw.csv`:

```bash
python scripts/simulate_magic_gating.py --source-run-id <RUN_ID> --level dataset --calib-n 10
```

Levels:
- `dataset`: decide per dataset
- `model`: decide per model
- `pair`: decide per (dataset, model)

### Per-dataset policy search (small samples; requires model calls)

```bash
python scripts/search_magic_policy.py
```

### Universal magic prompt search (small samples; requires model calls)

```bash
python scripts/search_universal_magic.py
```

### Generate additional candidate injections

```bash
python scripts/generate_magic_candidates.py --n 1000 --out outputs/candidates/magic_1000.txt
```

### Polish the manuscript with an LLM (requires model calls)

```bash
python scripts/polish_manuscript.py --run-id <RUN_ID>
```

## Outputs

Generated artifacts are intentionally git-ignored:

- Raw runs (JSONL): `outputs/runs/<RUN_ID>/`
- Aggregates (CSV/JSON/MD): `outputs/aggregates/<RUN_ID>/`
- Figures: `figures/<RUN_ID>/`
- OpenRouter cache (responses): `cache/openrouter/`

The main manuscript entrypoints:

- Draft: `manuscript/Nature_draft.md`
- Polished: `manuscript/Nature_final.md`

## Notes

- Default evaluation uses OpenRouter chat completions with `temperature=0` and a strict `FINAL: <A/B/C/D/...>` first-line format (invalid outputs are tracked).
- Model availability / names can change on OpenRouter; adjust lists in `scripts/run_all.py` if needed.
- Caching is enabled by default (`cache/openrouter/`), which can make reruns cheaper and faster.

## Safety

This repository is for research evaluation on published datasets only. It is **not** a clinical decision support system and provides **no medical advice**.
