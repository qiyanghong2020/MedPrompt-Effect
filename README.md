# The MedPrompt Effect

Implements the project described in `项目设计.md`: **prompt sensitivity (MedPrompt Effect)** in clinical reasoning models, and a controlled test of a short **“magic prompt”** injection (which may help or hurt depending on model/dataset).

## Run

`.env` is expected to contain:

- `OPENROUTER_API_KEY`
- `OPENROUTER_BASE_URL` (optional; default `https://openrouter.ai/api/v1`)
- `HF_TOKEN` (optional; for Hugging Face datasets)

Then run:

```bash
python scripts/run_all.py
```

Optional:

- Per-dataset policy search (small sample): `python scripts/search_magic_policy.py`
- Offline calibration-based selection simulation (no extra model calls): `python scripts/simulate_calibration_selection.py --source-run-id <RUN_ID>`
- Offline reliability audit pack (leaderboard stability + k-curve): `python scripts/audit_reliability.py --run-id <RUN_ID>`

## Key outputs

- Manuscript: `manuscript/Nature_draft.md`
- Polished manuscript: `manuscript/Nature_final.md`
- Figures: `figures/`
- Aggregates: `outputs/aggregates/`
- Raw per-item runs: `outputs/runs/`

## Safety

This repository evaluates LLMs on published datasets. It is **not** a clinical decision support tool and provides **no medical advice**.
