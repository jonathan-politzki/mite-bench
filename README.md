# MITE: Massive Interaction Task Evaluation

**The Interaction Gap in Text Embedding Benchmarks**

MTEB (Massive Text Embedding Benchmark) is the de facto standard for evaluating text embeddings, but all 8 of its task types measure **similarity** — none measure **interaction** (complementarity, asymmetric matching, conditional compatibility).

MITE reframes existing MTEB datasets to evaluate **interaction tasks**: given two texts, does their combination produce a meaningful outcome beyond topical similarity?

## The Core Finding

MTEB rankings fail to predict interaction task performance. We take the **same datasets** used by MTEB, evaluate the **same models**, but ask an **interaction question** instead of a **similarity question**. The rankings diverge.

| Dataset | MTEB Question (Similarity) | MITE Question (Interaction) |
|---|---|---|
| **SICK-R** | How similar are these sentences? (1-5) | Does sentence A entail sentence B? (asymmetric) |
| **FEVER** | Retrieve relevant docs for this claim | Does evidence support or refute the claim? |
| **FiQA** | Find relevant financial answers | How well does this answer resolve the question? |
| **SummEval** | How similar is summary to source? | How well does this summary capture key information? |

## Installation

```bash
pip install -e .

# For API model support (OpenAI, Voyage, Cohere):
pip install -e ".[api]"
```

## Quick Start

```python
from mite.tasks import SICKREntailmentTask
from mite.models import SentenceTransformerModel

# Load a task
task = SICKREntailmentTask()
task.load_data()

# Evaluate a model
model = SentenceTransformerModel("all-MiniLM-L6-v2")
results = task.evaluate(model)
print(results)
```

## Run Full Benchmark

```bash
# Run MITE evaluation on all tasks
python scripts/run_mite.py --models all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5

# Compare with MTEB rankings
python scripts/compare_rankings.py --results-dir results/
```

## Tasks

| Task | Dataset | Interaction Type | Metric |
|---|---|---|---|
| `SICKREntailmentTask` | SICK-R | Textual entailment (asymmetric) | Macro F1, Directional accuracy |
| `FEVERInteractionTask` | FEVER | Claim verification (support vs refute) | Separation score, AUROC |
| `ClimateFEVERInteractionTask` | ClimateFEVER | Claim verification | Separation score, AUROC |
| `SciFActInteractionTask` | SciFact | Scientific claim verification | Separation score, AUROC |
| `FiQAInteractionTask` | FiQA-2018 | Answer quality ranking | Spearman correlation |
| `CQADupstackInteractionTask` | CQADupstack | Answer acceptance prediction | AUROC, Accuracy |
| `SummEvalInteractionTask` | SummEval | Summary quality assessment | Spearman correlation |

## Paper

See `paper/mite_paper.tex` for the full paper: *"The Interaction Gap: Why Text Embedding Benchmarks Miss What Matters"*.

## License

Apache 2.0
