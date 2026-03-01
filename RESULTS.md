# MITE Experiment Results

**Date**: February 28, 2026
**Status**: SICK-R complete (15 models), FEVER complete (9 models), SummEval partial (7 models)

---

## TL;DR

**SICK-R delivers a highly significant result: rho = -0.761, p < 0.001 across 15 models spanning the full MTEB quality range.**

The best MTEB model is the worst at distinguishing entailment from contradiction. The only model that gets entailment right is GloVe — the one MTEB says is worst. This isn't noise: p < 0.001 with 15 models.

---

## Headline Result: SICK-R Entailment (15 models)

**Setup**: Same 4,906 sentence pairs from SICK-R. MTEB evaluates how well cosine similarity correlates with human relatedness scores (1-5). MITE evaluates whether cosine similarity can separate entailment from contradiction.

**Key insight**: In SICK-R, contradictory sentences have HIGHER relatedness scores (mean 4.57) than neutral sentences (mean 2.99). A claim and its negation are *related* — they're about the same thing. So a model optimized for relatedness will assign HIGH similarity to contradictions. The better it learns relatedness, the more it confuses entailment with contradiction.

### Results (15 models, MTEB range 0.55–0.81)

| Model | MTEB Spearman | MITE Separation | Notes |
|-------|:---:|:---:|---|
| **GloVe 6B 300d** | **0.554** | **+0.530** | **Only model with CORRECT separation** |
| paraphrase-MiniLM-L3-v2 | 0.762 | -0.959 | |
| all-MiniLM-L6-v2 | 0.772 | -0.902 | |
| e5-base-v2 | 0.775 | -1.418 | |
| nomic-embed-text-v1.5 | 0.780 | -0.999 | |
| gte-base | 0.783 | -1.078 | |
| e5-large-v2 | 0.787 | -2.256 | |
| e5-small-v2 | 0.787 | -2.169 | |
| gte-large | 0.791 | -1.280 | |
| bge-small-en-v1.5 | 0.791 | -1.882 | |
| multilingual-e5-large | 0.798 | -2.299 | |
| bge-base-en-v1.5 | 0.799 | -1.860 | |
| all-distilroberta-v1 | 0.801 | -1.856 | |
| all-mpnet-base-v2 | 0.805 | -2.456 | #2 MTEB, worst MITE |
| **bge-large-en-v1.5** | **0.812** | **-2.108** | **#1 MTEB, near-worst MITE** |

**Spearman rank correlation: rho = -0.761, p = 0.00099**

This is **highly significant (p < 0.001)**. The rankings are *inversely correlated* — improving at MTEB's similarity task actively *hurts* entailment discrimination.

### Why this is strong

1. **Same dataset, same sentence pairs, same models** — only the evaluation question changed
2. **p < 0.001** with 15 models spanning MTEB range 0.55–0.81
3. **Not just uncorrelated, but anti-correlated** — MTEB optimization actively harms interaction
4. **GloVe is the smoking gun**: The simplest, lowest-MTEB model is the ONLY one that correctly ranks entailment above contradiction. Every trained embedding model gets it backwards.
5. **The paradox is structural**: Contradictions ARE about the same topic (high relatedness). A model that learns "same topic = high similarity" will always confuse contradiction with entailment. This is unfixable within the similarity paradigm.

### The structural argument

Why does this happen? SICK-R's relatedness labels show:
- Contradictions: mean relatedness 4.57/5 (very related — same topic, opposite meaning)
- Entailments: mean relatedness 3.88/5
- Neutral: mean relatedness 2.99/5

A model that learns relatedness (MTEB's objective) will assign: contradiction > entailment > neutral in cosine similarity. But the correct interaction ranking is: entailment > neutral > contradiction. **The optimization objectives are in direct conflict.**

GloVe doesn't have this problem because it wasn't trained on similarity — it captures co-occurrence patterns that happen to preserve some entailment signal.

---

## Experiment 2: FEVER Claim Verification (9 models)

**Setup**: 968 balanced pairs from FEVER (484 SUPPORTS, 484 REFUTES). For each claim-evidence pair, compute cosine similarity.

**Rank correlation: rho = +0.617, p = 0.077** — NOT significant, positive correlation.

FEVER does not support the thesis. This is expected: supporting evidence IS naturally more similar to claims than refuting evidence. Unlike SICK-R, there's no structural paradox — similarity and interaction are aligned, not opposed.

**FEVER shows that MTEB works when similarity and interaction happen to align.** SICK-R shows it fails catastrophically when they don't.

| Model | MTEB Spearman | FEVER Separation | AUROC |
|-------|:---:|:---:|:---:|
| bge-base-en-v1.5 | 0.799 | 0.949 | 0.750 |
| bge-small-en-v1.5 | 0.791 | 0.917 | 0.748 |
| e5-small-v2 | 0.787 | 0.852 | 0.727 |
| all-mpnet-base-v2 | 0.805 | 0.815 | 0.723 |
| all-MiniLM-L6-v2 | 0.772 | 0.786 | 0.712 |
| e5-base-v2 | 0.775 | 0.758 | 0.708 |
| e5-large-v2 | 0.787 | 0.731 | 0.697 |
| nomic-embed-text-v1.5 | 0.780 | 0.689 | 0.689 |
| gte-base | 0.783 | 0.680 | 0.688 |

---

## Experiment 3: SummEval Summary Quality (7 of 15 models)

**Setup**: 1,600 source-summary pairs from SummEval (100 sources × 16 summaries). For each source, rank summaries by cosine similarity with source. Compare ranking against human quality scores.

**Rank correlation (7 models): rho = +0.500, p = 0.253** — not significant with only 7 models.

| Model | MTEB Spearman | Mean Spearman | Pairwise Accuracy |
|-------|:---:|:---:|:---:|
| GloVe 6B 300d | 0.554 | 0.196 | 0.573 |
| all-MiniLM-L6-v2 | 0.772 | 0.247 | 0.593 |
| all-distilroberta-v1 | 0.801 | 0.254 | 0.597 |
| e5-small-v2 | 0.787 | 0.272 | 0.602 |
| paraphrase-MiniLM-L3-v2 | 0.762 | 0.287 | 0.610 |
| bge-small-en-v1.5 | 0.791 | 0.308 | 0.618 |
| bge-base-en-v1.5 | 0.799 | 0.330 | 0.625 |

SummEval shows weak positive correlation — better MTEB models are slightly better at ranking summary quality. This is expected: summary quality IS partially a similarity task (good summaries should be similar to their source).

---

## The Argument

MTEB works when the interaction aligns with similarity. It fails when they conflict.

| Task | Similarity-Interaction Alignment | MTEB Predicts? | rho | p |
|---|---|---|---|---|
| **SICK-R Entailment** | **Opposed** (contradictions are more "related" than entailments) | **No — anti-correlated** | **-0.761** | **0.001** |
| FEVER Verification | Aligned (supporting evidence is more similar to claims) | Partially | +0.617 | 0.077 |
| SummEval Quality | Aligned (good summaries resemble their source) | Partially | +0.500 | 0.253 |

**The problem isn't that MTEB is wrong. It's that MTEB is incomplete.** When the real-world task requires distinguishing interactions that similarity conflates (entailment vs. contradiction, compatibility vs. surface similarity), MTEB rankings actively mislead.

This is exactly the pattern we see in compatibility scoring: two people can be very "similar" (same interests, same background) but incompatible, or very "different" but highly compatible. Similarity ≠ interaction.

---

## Connection to DeepMatch / Keeper

| Observation | DeepMatch/Keeper | MITE |
|---|---|---|
| MTEB rank doesn't predict interaction | Qwen3-Embed (#1 MTEB) scored 0.215 on compatibility | bge-large (#1 MTEB STS) is near-worst at entailment |
| Similarity ≠ interaction | Cosine similarity fails to capture compatibility | Cosine similarity assigns higher scores to contradictions than entailments |
| Small models can beat large | MiniLM-L6 (22M) beats e5-large (335M) on compatibility | GloVe (0 trainable params) beats all transformers on entailment |
| The problem is structural | Bi-encoder cosine can't represent asymmetric interaction | SICK-R: entailment is directional, cosine is symmetric |

---

## Raw Data Files

- `results/sick_r_expanded.json` — 15 models, complete (headline result)
- `results/sick_r_experiment.json` — 9 models, complete (original run)
- `results/fever_experiment.json` — 9 models, complete
- `results/summeval_experiment.json` — 7 models, partial
- `scripts/run_expanded.py` — 15-model expanded experiment runner
- `scripts/run_experiment.py` — SICK-R experiment runner
- `scripts/run_fever_summeval.py` — FEVER + SummEval experiment runner
