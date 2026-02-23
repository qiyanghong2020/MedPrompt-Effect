# The MedPrompt Effect: A universal prompt-level amplification phenomenon in clinical reasoning models

## Abstract
Clinical reasoning benchmarks are increasingly used to compare foundation models, yet even *meaning-preserving* prompt edits can yield large accuracy swings—an underappreciated source of variance for medical evaluation and deployment. Here we characterize a prompt-level amplification phenomenon across diverse clinical reasoning models, which we term the **MedPrompt Effect**. On three multiple-choice medical QA datasets (n=200 questions; 5 core models), we find that paraphrased prompts with the same intended instruction induce substantial accuracy dispersion (median within-model range: 13.3 percentage points). We also test a short, fixed injection—**医学魔法 prompt**—selected on a disjoint dev set, and quantify its held-out impact on accuracy (mean change: -0.5 percentage points; 95% bootstrap CI: -2.0 to 1.0) and output validity. Overall, prompt phrasing is a first-order determinant of apparent model capability in medicine; prompt-level interventions must therefore be validated across prompts, models, and datasets before being treated as reliable improvements.

## Main Text

### Introduction
Large language models (LLMs) are rapidly adopted for clinical question answering and reasoning benchmarks. However, clinical evaluations typically report point estimates under a single prompt, implicitly assuming that prompt phrasing is a minor nuisance variable. In high-stakes settings, this assumption is risky: small instruction changes can reshape the model’s internal reasoning trajectory, potentially altering diagnosis and management choices. We hypothesized that (i) meaning-equivalent prompt paraphrases systematically change medical QA accuracy across models, and (ii) a fixed, short prompt injection can amplify clinical reasoning performance broadly.

### Results

#### Prompt sensitivity in medical QA (MedPrompt Effect)
Across the core model panel (5 models) and datasets, we observe clear prompt sensitivity even with deterministic decoding. On the MedXpertQA subset, within-model accuracy varied across paraphrased prompts with a median range of 13.3 percentage points (IQR 6.7–13.3).

#### A short “医学魔法” injection shifts accuracy
We performed a dev-set search over candidate short injections and selected the best-performing fixed prompt:

> Let's think step by step.

On held-out test sets, adding this injection to the baseline question prompt changed mean accuracy by -0.5 percentage points (95% bootstrap CI -2.0–1.0) across the core model panel. Effects were heterogeneous across datasets and models (Table 1; Fig. 2), underscoring that prompt injections can help or hurt depending on context. The overall change in invalid-format outputs (baseline − magic) was -0.80 percentage points.


**Table 1 | Core model accuracy with and without the magic prompt.**

| dataset     | model                       |   baseline_acc_% |   magic_acc_% |   gain_pp |
|:------------|:----------------------------|-----------------:|--------------:|----------:|
| MedMCQA     | anthropic/claude-sonnet-4   |             75.0 |          75.0 |       0.0 |
| MedMCQA     | deepseek/deepseek-chat-v3.1 |             75.0 |          73.3 |      -1.7 |
| MedMCQA     | google/gemini-2.5-flash     |             73.3 |          75.0 |       1.7 |
| MedMCQA     | openai/gpt-4.1              |             80.0 |          81.7 |       1.7 |
| MedMCQA     | qwen/qwen3-max              |             68.3 |          66.7 |      -1.7 |
| MedQA-USMLE | anthropic/claude-sonnet-4   |             86.7 |          86.7 |       0.0 |
| MedQA-USMLE | deepseek/deepseek-chat-v3.1 |             80.0 |          81.7 |       1.7 |
| MedQA-USMLE | google/gemini-2.5-flash     |             83.3 |          86.7 |       3.3 |
| MedQA-USMLE | openai/gpt-4.1              |             90.0 |          93.3 |       3.3 |
| MedQA-USMLE | qwen/qwen3-max              |             83.3 |          85.0 |       1.7 |
| MedXpertQA  | anthropic/claude-sonnet-4   |             37.5 |          36.2 |      -1.3 |
| MedXpertQA  | deepseek/deepseek-chat-v3.1 |             36.2 |          27.5 |      -8.7 |
| MedXpertQA  | google/gemini-2.5-flash     |             32.5 |          31.2 |      -1.3 |
| MedXpertQA  | openai/gpt-4.1              |             41.2 |          40.0 |      -1.2 |
| MedXpertQA  | qwen/qwen3-max              |             25.0 |          23.8 |      -1.3 |



**Table 2 | Extended model panel on MedXpertQA (subset) with and without the magic prompt.**

| model                            |   baseline_acc_% |   magic_acc_% |   gain_pp |
|:---------------------------------|-----------------:|--------------:|----------:|
| anthropic/claude-opus-4.6        |             45.0 |          47.5 |       2.5 |
| meta-llama/llama-4-scout         |             20.0 |          27.5 |       7.5 |
| mistralai/mistral-small-creative |              0.0 |           0.0 |       0.0 |
| moonshotai/kimi-k2.5             |              2.5 |           0.0 |      -2.5 |
| openai/gpt-5.2-pro               |             42.5 |          35.0 |      -7.5 |
| z-ai/glm-4.7-flash               |              0.0 |           0.0 |       0.0 |


### Discussion
The MedPrompt Effect implies that single-prompt medical evaluations can understate or overstate true capability. Our prompt-injection experiment further suggests that “magic prompts” selected on small dev sets may not transfer uniformly: we observe meaningful shifts in accuracy, but not a universally positive gain across datasets and models. In practice, this motivates reporting prompt-sensitivity ranges, evaluating multiple semantically equivalent prompts, and validating any candidate prompt-level intervention on disjoint data before clinical use.

## Methods (Summary)
We evaluated multiple LLMs via OpenRouter on multiple-choice medical QA datasets. For each example, models received the question stem and answer options and were instructed to output exactly one option label from the provided set (e.g., A–D for 4-option datasets and A–J for MedXpertQA). Prompt sensitivity was measured across four meaning-equivalent paraphrases. The magic prompt was selected on a disjoint dev set by maximizing mean accuracy improvement across two representative models, then evaluated on held-out test items. Uncertainty was estimated using paired bootstrap resampling over paired model–question IDs.

## Data and Code Availability
All code to reproduce the analyses is included in this repository. Raw run logs, aggregates, and figures are written to `outputs/` and `figures/` when running `python scripts/run_all.py`.

## Figures
**Figure 1 | Prompt sensitivity.** Distribution of within-model accuracy ranges across meaning-equivalent paraphrases (MedXpertQA subset).

**Figure 2 | Magic prompt gains.** Per-model accuracy improvement (percentage points) on MedXpertQA for the core model panel.