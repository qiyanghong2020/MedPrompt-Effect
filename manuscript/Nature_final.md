# The MedPrompt Effect: prompt sensitivity and the limits of universal “magic prompts” in clinical reasoning models

## Abstract
Clinical reasoning benchmarks are increasingly used to compare foundation models and to motivate clinical deployments. Yet evaluations are often reported under a single prompt, implicitly treating wording as a nuisance variable. Here we show that prompt phrasing is a first-order determinant of apparent clinical reasoning performance. Across five frontier models and semantically aligned prompt paraphrases, the same multiple-choice medical questions yield large within-model accuracy dispersion (median within-model range: 13.3 percentage points; MedXpertQA subset). We then test a fixed, short injection prompt—popularized as a general reasoning cue and selected on a disjoint dev set—on held-out test items from three medical QA datasets (n=200 questions; 5 models). Contrary to “universal magic prompt” claims, the injection produces heterogeneous effects across datasets and models, with an overall mean accuracy change of −0.5 percentage points (95% paired bootstrap CI −2.0 to 1.0) and a small increase in invalid-format outputs. Finally, we show that a simple calibration-based *selective injection* policy can reduce expected regressions from injection prompting, but does not guarantee improvement. These results imply that single-prompt reporting can materially misstate medical QA capability, and that prompt-level interventions should be treated as model changes that require cross-prompt, cross-dataset validation before being relied upon in high-stakes settings.

## Main Text

### Introduction
Large language models (LLMs) now match or exceed strong baselines on many medical multiple-choice question answering (MCQ) benchmarks and are increasingly evaluated as candidate clinical reasoning systems. In practice, however, model performance is not a single number: it is the outcome of an experimental protocol that includes the dataset, the model, the decoding configuration, and the prompt. While recent work has emphasized dataset and model choice, prompt wording is often treated as inconsequential—despite a growing literature showing that instruction framing, paraphrasing, and even formatting can strongly shape model behavior (refs. 7–10).

This matters for medicine. If meaning-preserving prompt edits shift measured accuracy by amounts comparable to reported model-to-model gaps, then (i) leaderboard rankings can be unstable, (ii) prompt “improvements” can reflect evaluation artifacts rather than capability, and (iii) small prompt edits in a clinical workflow may yield unanticipated regressions. We refer to this phenomenon—large performance dispersion under semantically aligned prompts—as the **MedPrompt Effect**.

We test two hypotheses. First, that clinical MCQ accuracy exhibits substantial dispersion across paraphrased prompts even under deterministic decoding. Second, that a *fixed*, short injection prompt—commonly described as a “magic prompt”—can robustly amplify clinical reasoning accuracy across models. We find strong evidence for the first hypothesis, while the second does not hold as a universal improvement under our controlled, held-out evaluation.

### Related work and positioning
Prompt sensitivity has been observed across tasks and model families, motivating both benchmarking efforts and proposed summary indices of prompt robustness (refs. 8–9). In clinical NLP specifically, prior work has documented that clinically oriented LLMs can be sensitive to instruction phrasing (ref. 7). In parallel, evidence suggests that seemingly superficial prompt properties—such as formatting—can materially change LLM outputs (ref. 10). Our results align with these observations, and we do not claim prompt sensitivity as a new phenomenon.

Instead, we focus on two under-emphasized points for medical MCQ evaluation. First, prompt sensitivity is not merely about semantics: under the strict output constraints used for automated scoring, prompt paraphrases can change *format validity* and thereby change measured accuracy. Second, “magic prompt” claims are often presented as universal upgrades; we test this claim under a controlled dev/test protocol across multiple datasets and multiple frontier models, and characterize the resulting heterogeneity. Finally, we provide a simple, deployment-relevant mitigation analysis: when a fixed injection is not universally beneficial, a small calibration stage can be used to decide whether to apply it in a given context.

### Results

#### Prompt sensitivity induces large swings in measured clinical accuracy
We evaluated a core panel of five frontier models (OpenRouter API) on a MedXpertQA subset using four semantically aligned prompt paraphrases that preserved the same task instruction (choose one best option) and the same constrained output format (“FINAL: <letter>”). Even with deterministic decoding (temperature 0), accuracy varied substantially across prompts within the same model. The median within-model accuracy range across the four paraphrases was 13.3 percentage points (IQR 6.7–13.3) (Fig. 1).

On this sensitivity set, the magnitude of within-model prompt variation was comparable to typical between-model differences: the median within-model range (13.3 pp) exceeded the median between-model accuracy gap observed under a single paraphrase (10.0 pp; all model pairs across prompts). Moreover, 4 of 10 model pairwise orderings reversed across paraphrases, indicating that prompt choice can change comparative conclusions even when the task meaning is preserved. Consistent with this instability, the agreement between prompt-induced leaderboards was only moderate (median Kendall’s τ=0.50 across prompt pairs; min 0.20).

Importantly, prompt sensitivity reflected both *content* and *format compliance*. For some model–prompt pairs, minor formatting changes increased invalid outputs (i.e., responses not containing a parseable final option) and thereby reduced measured accuracy. This indicates that prompt choice can change not only what a model “knows”, but also whether it reliably produces an evaluation-compliant answer—an effect that is operationally relevant for clinical interfaces.

#### A fixed “magic prompt” does not generalize as a universal accuracy amplifier
Next, we asked whether a short, fixed injection—popularized by chain-of-thought prompting and its variants (refs. 4–6)—could serve as a robust, model-agnostic improvement. On a disjoint MedXpertQA dev set, we searched a small set of candidate injections and selected:

> Let's think step by step.

We then evaluated this fixed injection on held-out test items from three datasets (MedXpertQA n=80; MedMCQA n=60; MedQA‑USMLE n=60) across the same five-model panel (n=200 questions; 2,000 model–question evaluations under baseline vs. injected prompts).

Across all paired model–question instances, the injection changed accuracy by −0.5 percentage points on average (95% paired bootstrap CI −2.0 to 1.0). Effects were strongly heterogeneous by dataset: MedQA‑USMLE increased by +2.0 points, MedMCQA showed no net change, and MedXpertQA decreased by −2.8 points (Table 1). Effects were also heterogeneous by model: two models improved by +1.0 point overall, while one model decreased by −3.5 points, driven primarily by a large MedXpertQA regression (Table 2).

The injection did not improve output validity. The overall rate of invalid-format outputs increased by 0.8 percentage points under injection (baseline − injection: −0.80 pp), consistent with the idea that prompting changes can introduce new failure modes even when they occasionally improve reasoning accuracy.

**Table 1 | Dataset-level accuracy with and without the injection prompt (core model panel).**

| dataset     | baseline_acc_% | magic_acc_% | gain_pp |
|:------------|---------------:|------------:|--------:|
| MedMCQA     |           74.3 |        74.3 |     0.0 |
| MedQA-USMLE |           84.7 |        86.7 |     2.0 |
| MedXpertQA  |           34.5 |        31.8 |    -2.8 |
| **Overall** |           61.5 |        61.0 |    -0.5 |

**Table 2 | Core model accuracy with and without the injection prompt (per dataset).**

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

#### Breadth check: heterogeneity persists in an extended model panel
To test whether heterogeneity was idiosyncratic to the core panel, we evaluated six additional models on a MedXpertQA subset (n=40) under baseline and injected prompts. The injection again showed mixed effects: some models improved (e.g., +7.5 points for one model), while others regressed (e.g., −7.5 points), and some remained unchanged (Table 3). This supports the conclusion that short injections are not reliably transferable improvements, even when they are popular and seemingly benign.

**Table 3 | Extended model panel on MedXpertQA (subset) with and without the injection prompt.**

| model                            |   baseline_acc_% |   magic_acc_% |   gain_pp |
|:---------------------------------|-----------------:|--------------:|----------:|
| anthropic/claude-opus-4.6        |             45.0 |          47.5 |       2.5 |
| meta-llama/llama-4-scout         |             20.0 |          27.5 |       7.5 |
| mistralai/mistral-small-creative |              0.0 |           0.0 |       0.0 |
| moonshotai/kimi-k2.5             |              2.5 |           0.0 |      -2.5 |
| openai/gpt-5.2-pro               |             42.5 |          35.0 |      -7.5 |
| z-ai/glm-4.7-flash               |              0.0 |           0.0 |       0.0 |

#### A calibration-based selective injection policy partially mitigates injection risk
Because injection effects were heterogeneous, a practical alternative to a universal injection is a *selective* policy: reserve a small calibration set in the deployment context, estimate whether the injection helps, and apply the injection only if the observed gain is positive. Using the paired baseline and injected outputs from our held-out evaluation, we simulated a two-stage protocol in which each dataset contributes *k* calibration questions and the remaining questions are scored under either baseline or baseline+injection according to the calibration decision.

With dataset-level selection, expected gain was small but became slightly positive for moderate calibration sizes, while regression risk remained non-trivial. For example, with *k*=20 calibration items per dataset, the expected overall gain became +0.17 pp on average across random splits, and the probability of an overall regression decreased to 17.9%. With *k*=10, the expected gain was +0.07 pp (P[gain<0]=24.6%). Sweeping *k* revealed a trade-off: larger *k* can reduce decision noise but also reduces the remaining evaluation set; in our simulation the best expected gains occurred at intermediate *k* (Fig. 3). By contrast, more granular selection (per-model or per dataset×model) tended to overfit the small calibration set and did not improve expected performance. These results suggest that calibration can reduce risk when using a fixed injection, but that this selectivity rule is itself a statistical decision procedure that must be evaluated and sized appropriately.

### Discussion
Our results support two practical conclusions for medical LLM evaluation and deployment.

First, the MedPrompt Effect implies that *single-prompt* evaluations can meaningfully overstate or understate a model’s capability. In our controlled setting, semantically aligned prompt paraphrases shifted within-model accuracy by a median of 13.3 percentage points. Because medical benchmark leaderboards often report differences of similar magnitude, prompt choice can invert apparent rankings or exaggerate gains attributed to model changes. For high-stakes clinical use, this also suggests that “prompt engineering” is not a cosmetic layer: it is an intervention that can change both reasoning behavior and the reliability of producing a usable answer format.

Second, a fixed “magic prompt” injection is not a universal amplifier. The widely used cue “Let's think step by step.” improved one dataset but degraded another, yielding a net effect consistent with zero and a small increase in invalid outputs. This heterogeneity is compatible with several mechanisms: step-by-step cues can increase verbosity and raise the risk of truncation or formatting errors; they may encourage brittle reasoning heuristics that help some item types while harming others; and they can interact with model-specific instruction-following behavior. Regardless of mechanism, our findings argue against treating short injections as drop-in clinical upgrades without robust validation.

These observations motivate a simple reporting standard: benchmark claims about clinical reasoning should include (i) the exact prompt(s), (ii) a prompt-sensitivity envelope across meaning-aligned paraphrases, and (iii) disjoint validation of any prompt-level intervention across datasets and models. For deployment, we recommend treating prompts as versioned artifacts with regression testing, analogous to model checkpointing.

### Limitations
This study is intentionally cost-conscious and focuses on controlled MCQ evaluation. Our test set size is modest (n=200 questions) and uses three public datasets; effects may differ on free-form clinical notes, longitudinal tasks, or under different decoding settings (e.g., sampling, self-consistency). We evaluated a limited candidate set of injections and selected one on a small dev set; broader searches could yield different injections but would also increase the risk of overfitting the prompt to the dev distribution. Finally, we do not claim clinical safety: the study evaluates benchmark answers and output-format compliance, not patient outcomes.

## Methods (Detailed)

### Datasets and sampling
We used three publicly available multiple-choice medical QA datasets: MedXpertQA (Hugging Face: `simwit/medxpertqa-text`), MedMCQA (`medmcqa`), and MedQA‑USMLE 4-options (`GBaker/MedQA-USMLE-4-options`). From MedXpertQA (test split), we sampled three disjoint subsets with a fixed random seed (20260211): a dev set (n=30) for injection selection, a sensitivity set (n=30) for prompt paraphrase evaluation, and a held-out test set (n=80) for injection evaluation. From MedMCQA (validation split) and MedQA‑USMLE (test split), we sampled n=60 questions each for held-out evaluation.

### Models and inference protocol
All models were queried via the OpenRouter API with deterministic decoding (temperature 0). Prompts instructed models to answer each MCQ by outputting a single option label. For evaluation runs, we limited completions to 192 tokens to control cost and to standardize truncation risk across conditions. Exact model identifiers are recorded in run logs under `outputs/runs/`.

### Prompt sets
Prompt sensitivity was measured using four semantically aligned paraphrases (defined in `medprompt_effect/prompts.py`) that preserved the same task and output constraint but varied wording and formatting. The injection prompt was selected from 11 short candidates by maximizing mean dev-set accuracy improvement across two representative models (OpenAI GPT‑4.1 and Anthropic Claude Sonnet). The selected injection was then fixed and evaluated on held-out test sets without further tuning.

### Outcome definitions and parsing
The primary endpoint was accuracy (fraction correct). Responses were parsed using a conservative extractor that first searches for an explicit `FINAL:` line and validates that the extracted label is among the allowed choices; unparsable responses were marked invalid and counted as incorrect. Invalid-rate is the fraction of responses without a parseable final label.

### Statistical analysis
To estimate uncertainty in the overall effect of injection prompting, we computed a paired bootstrap confidence interval over paired model–question instances (pair ID = dataset × model × question ID) for the mean difference in correctness (injection − baseline), using 2,000 bootstrap resamples.

### Calibration-based selective injection simulation
To evaluate a practical mitigation for heterogeneous injection effects, we simulated a two-stage decision policy using the held-out baseline and injected test outputs. For each dataset, we repeatedly sampled a calibration subset of size *k* without replacement and computed the observed calibration gain (injection − baseline). If the gain was positive, we applied injection to the remaining items in that dataset; otherwise we used the baseline prompt. We report the distribution of overall gains across 5,000 random splits. This analysis uses no additional model calls and is intended as a deployment-motivated simulation rather than a claim of guaranteed improvement. Code to reproduce this simulation is provided in `scripts/simulate_calibration_selection.py`.

## Data and Code Availability
Code to reproduce all analyses is included in this repository. Running `python scripts/run_all.py` produces raw logs (`outputs/runs/`), aggregates (`outputs/aggregates/`), figures (`figures/`), and an auto-generated manuscript draft (`manuscript/Nature_draft.md`). Additional reliability-audit outputs (leaderboard stability metrics and the calibration *k* curve) can be generated offline for an existing run via `python scripts/audit_reliability.py --run-id <RUN_ID>`.

## Safety statement
This repository evaluates LLMs on published datasets. It is not a clinical decision support tool and provides no medical advice.

## Figures
- **Figure 1 | Prompt sensitivity.** Distribution of within-model accuracy ranges across prompt paraphrases (MedXpertQA sensitivity subset). See `figures/20260212_080555/prompt_sensitivity_range.png`.
- **Figure 2 | Injection prompt gains.** Per-model accuracy change on MedXpertQA under injection vs baseline (core panel). See `figures/20260212_080555/medx_improvement_by_model.png`.
- **Figure 3 | Leaderboard instability and calibration-based selective injection.** Prompt-paraphrase accuracy/rank slopegraphs and the calibration *k* trade-off curve. See `figures/20260212_080555/leaderboard_instability.png` and `figures/20260212_080555/calibration_curve.png`.

## References
1. Pal, A. et al. MedMCQA: A large-scale multi-subject multi-choice dataset for medical domain question answering. *CHIL 2022* (2022).
2. Jin, D. et al. What disease does this patient have? A large-scale open domain question answering dataset from medical exams. *Appl. Sci.* 11, 6421 (2021). (Also available as arXiv:2009.13081.)
3. Zuo, Y. et al. MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding. arXiv:2501.18362 (2025).
4. Wei, J. et al. Chain-of-thought prompting elicits reasoning in large language models. arXiv:2201.11903 (2022).
5. Wang, X. et al. Self-consistency improves chain of thought reasoning in language models. arXiv:2203.11171 (2022).
6. Kojima, T. et al. Large language models are zero-shot reasoners. arXiv:2205.11916 (2022).
7. Ceballos Arroyo, A. M. et al. Open (Clinical) LLMs are Sensitive to Instruction Phrasings. arXiv:2407.09429 (2024).
8. Zhu, K. et al. PromptRobust: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts. arXiv:2306.04528 (2023).
9. Chatterjee, A. et al. POSIX: A Prompt Sensitivity Index For Large Language Models. arXiv:2410.02185 (2024).
10. He, J. et al. Does Prompt Formatting Have Any Impact on LLM Performance? arXiv:2411.10541 (2024).
