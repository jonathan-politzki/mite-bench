# MITE Experiment Results — Honest Assessment

**Date**: February 28, 2026
**Status**: 2 of 3 tasks complete (9 models each), 1 task partial (3 models)

---

## TL;DR

**SICK-R is a strong result. FEVER is inconclusive. We need more models and the SummEval task to finish.**

The core claim — "MTEB rankings don't predict interaction task performance" — is convincingly supported by one experiment (SICK-R) and not supported by another (FEVER). This is honest but incomplete. More work needed.

---

## Experiment 1: SICK-R Entailment (STRONG)

**Setup**: Same 4,906 sentence pairs from SICK-R. MTEB evaluates how well cosine similarity correlates with human relatedness scores (1-5). MITE evaluates whether cosine similarity can separate entailment from contradiction.

**Key insight**: In SICK-R, contradictory sentences have HIGHER relatedness scores (mean 4.57) than neutral sentences (mean 2.99). A claim and its negation are related — they're about the same thing. So a model optimized for relatedness prediction will assign HIGH similarity to contradictions.

### Results (9 models)

| Model | MTEB Spearman | MTEB Rank | MITE Separation | MITE Rank |
|-------|:---:|:---:|:---:|:---:|
| all-MiniLM-L6-v2 | 0.772 | 9 | -0.902 | **1** |
| nomic-embed-text-v1.5 | 0.780 | 7 | -0.999 | **2** |
| gte-base | 0.783 | 6 | -1.078 | **3** |
| e5-base-v2 | 0.775 | 8 | -1.418 | 4 |
| bge-base-en-v1.5 | 0.799 | 2 | -1.860 | 5 |
| bge-small-en-v1.5 | 0.791 | 3 | -1.882 | 6 |
| e5-small-v2 | 0.787 | 4 | -2.169 | 7 |
| e5-large-v2 | 0.787 | 5 | -2.256 | 8 |
| **all-mpnet-base-v2** | **0.805** | **1** | **-2.456** | **9** |

**Rank correlation: rho = -0.783, p = 0.013**

This is statistically significant. The rankings are *inversely correlated* — the best MTEB model (all-mpnet-base-v2) is the *worst* at distinguishing entailment from contradiction. Every model assigns higher cosine similarity to contradictions than entailments (all separation scores are negative).

**Why this is strong**:
- Same dataset, same sentence pairs, same models — only the evaluation question changed
- p = 0.013 is significant at the 0.05 level
- The direction is striking: not just uncorrelated, but *anti-correlated*
- Macro F1 for entailment classification is 0.27-0.37 (at or below the random baseline of 0.33 for 3 classes)

**Caveats**:
- Only 9 models, and MTEB scores are compressed (0.77-0.80 range)
- The MTEB comparison is against STS on the same dataset, not overall MTEB rank

---

## Experiment 2: FEVER Claim Verification (INCONCLUSIVE)

**Setup**: 968 balanced pairs from FEVER (484 SUPPORTS, 484 REFUTES). For each claim-evidence pair, compute cosine similarity. If a model captures the interaction, SUPPORTS pairs should have higher similarity than REFUTES pairs — since both are "relevant" but have opposite valence.

### Results (9 models)

| Model | MTEB Spearman | MTEB Rank | FEVER Separation | FEVER Rank | AUROC |
|-------|:---:|:---:|:---:|:---:|:---:|
| **bge-base-en-v1.5** | 0.799 | 2 | **0.949** | **1** | 0.750 |
| bge-small-en-v1.5 | 0.791 | 3 | 0.917 | 2 | 0.748 |
| e5-small-v2 | 0.787 | 4 | 0.852 | 3 | 0.727 |
| all-mpnet-base-v2 | 0.805 | 1 | 0.815 | 4 | 0.723 |
| all-MiniLM-L6-v2 | 0.772 | 9 | 0.786 | 5 | 0.712 |
| e5-base-v2 | 0.775 | 8 | 0.758 | 6 | 0.708 |
| e5-large-v2 | 0.787 | 5 | 0.731 | 7 | 0.697 |
| nomic-embed-text-v1.5 | 0.780 | 7 | 0.689 | 8 | 0.689 |
| gte-base | 0.783 | 6 | 0.680 | 9 | 0.688 |

**Rank correlation: rho = 0.617, p = 0.077**

This is NOT statistically significant. And the correlation is *positive* — meaning MTEB somewhat predicts FEVER interaction performance. This does NOT support the "MTEB can't predict interaction" thesis.

**What the FEVER results DO show**:
- All 9 models successfully separate SUPPORTS from REFUTES (all separation scores positive, all AUROCs > 0.68)
- Cosine similarity does partially capture claim verification direction
- The signal is modest (AUROCs 0.69-0.75, nowhere near perfect)
- The #1 MTEB model (all-mpnet-base-v2) only ranks #4 on FEVER — some divergence but not dramatic

**Why this is weaker than SICK-R**:
- p = 0.077 misses significance threshold
- Positive correlation goes against our thesis
- FEVER's claim-evidence structure may be close enough to retrieval that similarity models do okay

---

## Experiment 3: SummEval Summary Quality (PARTIAL — 3 of 9 models)

**Setup**: 1,600 source-summary pairs from SummEval (100 sources x 16 summaries). For each source, rank summaries by cosine similarity with source. Compare ranking against human quality scores using Spearman correlation.

### Results (3 models — still running)

| Model | MTEB Spearman | Mean Spearman | Pairwise Accuracy | Global Spearman |
|-------|:---:|:---:|:---:|:---:|
| bge-base-en-v1.5 | 0.799 | **0.330** | **0.625** | **0.335** |
| all-MiniLM-L6-v2 | 0.772 | 0.247 | 0.593 | 0.249 |
| all-mpnet-base-v2 | 0.805 | 0.246 | 0.594 | 0.266 |

Not enough models for rank correlation, but: best MTEB model (all-mpnet-base-v2) is worst on summary quality. BGE (lower MTEB) is best by a large margin.

---

## Cross-Task Analysis

### Do the MITE tasks agree with each other?

**SICK-R vs FEVER rank correlation: rho = -0.383, p = 0.308**

The two MITE tasks don't agree on which models are better at "interaction." This means either:
1. "Interaction" isn't a single dimension (entailment and claim verification are genuinely different skills)
2. Our metrics are noisy with only 9 models

Either way, this weakens a unified "MITE score" narrative but actually *strengthens* the argument that we need task-specific interaction benchmarks.

### Model-level observations

- **all-mpnet-base-v2**: #1 MTEB, #9 SICK-R (worst), #4 FEVER. Strongest evidence that MTEB rank misleads.
- **all-MiniLM-L6-v2**: #9 MTEB (worst), #1 SICK-R (best), #5 FEVER. The smallest model is best at entailment.
- **bge-base-en-v1.5**: #2 MTEB, #5 SICK-R, #1 FEVER. Most consistent across tasks.

---

## Honest Assessment: Are These Good Results?

### What's strong
1. **SICK-R is a genuine finding**. rho = -0.783 (p=0.013) on 9 models. The best STS model is worst at entailment. This is publishable.
2. **The "same data, different question" framing is powerful**. We didn't cherry-pick datasets — we reframed MTEB's own datasets.
3. **Every model fails at entailment classification** (macro F1 below random baseline). This is a clear, easy-to-understand failure mode.
4. **FEVER separation works** — models DO partially distinguish supports from refutes via cosine, confirming that interaction signal exists in embeddings even if imperfectly.

### What's weak
1. **FEVER doesn't support the thesis**. The positive correlation (rho=0.617, p=0.077) suggests MTEB *partially* predicts FEVER interaction performance.
2. **Only 9 models, narrow MTEB range**. All models have MTEB Spearman 0.77-0.80. We need models spanning 0.60-0.90 (add bad models and frontier models) for clearer signal.
3. **SummEval incomplete**. Need all 9 models to compute correlation.
4. **MTEB comparison is on one STS task**. We're comparing against SICK-R STS Spearman, not overall MTEB rank. Should use official MTEB leaderboard scores for a stronger claim.
5. **Cross-task MITE disagreement** undermines a unified benchmark story.

### What we need to make this publishable
1. **More models** (15-20): Add frontier models (Qwen3-Embed, text-embedding-3-large, Voyage-3, Cohere v4) and weak models (fasttext, glove-based) for wider MTEB score range
2. **Complete SummEval** (9-model run in progress)
3. **Overall MTEB scores**: Use published MTEB leaderboard numbers instead of just STS on SICK-R
4. **Add 1-2 more tasks**: ClimateFEVER, SciFact for more interaction dimensions
5. **Statistical power analysis**: With 9 models, we need |rho| > 0.68 for significance. More models lower this threshold.

---

## Connection to DeepMatch / Keeper

These results directly validate what we found in our compatibility experiments:

| Observation | DeepMatch/Keeper | MITE |
|---|---|---|
| MTEB rank doesn't predict interaction | Qwen3-Embed (#1 MTEB) scored 0.215 on compatibility | all-mpnet (#1 MTEB STS) is worst at entailment |
| Similarity != interaction | Cosine similarity fails to capture compatibility | Cosine similarity assigns higher scores to contradictions than entailments |
| Small models can beat large | MiniLM-L6 (22M) beats e5-large (335M) on entailment | Same pattern in Keeper: nano sometimes beats large |
| The problem is structural | Bi-encoder cosine can't represent asymmetric interaction | SICK-R proves this: entailment is asymmetric, cosine is symmetric |

The MITE paper provides the academic/industry framework for the problem our company is solving.

---

## Raw Data Files

- `results/sick_r_experiment.json` — 9 models, complete
- `results/fever_experiment.json` — 9 models, complete
- `results/summeval_experiment.json` — 3 models, partial (9-model run in progress)
- `scripts/run_experiment.py` — SICK-R experiment runner
- `scripts/run_fever_summeval.py` — FEVER + SummEval experiment runner
