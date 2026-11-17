<div align="center">

# Cost-Constrained Generative Social Choice (CC-GSC)

**Implementation and evaluation of generative social choice on the abortion-policy opinion survey using a locally served open-weight LLM**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Model](https://img.shields.io/badge/Model-qwen3:8b-orange.svg)](https://arxiv.org/abs/2505.09388)
[![Status](https://img.shields.io/badge/Status-Research%20prototype-yellow.svg)](#)

</div>

---

## 📌 Overview

Classical social choice assumes a **fixed, small set of alternatives**, which cannot represent open-ended questions such as collectively writing a statement about abortion policy. **Generative social choice** combines social choice theory with large language models to summarise free-form opinions into a short, proportionally representative slate of statements.

This repository implements a **cost-constrained** variant of that idea on the original 100-participant abortion survey:

- Each statement has a **cost equal to its length in words**.
- A statement is admitted only if its supporters meet the proportional requirement `⌈c(α)·n / B⌉`, where `B = 500` is the total word budget.
- Both the **discriminative** (agreement scoring) and **generative** (statement writing) queries are answered by `qwen3:8b` served locally through Ollama, with no proprietary API.

> The reported results are **descriptive measurements** of the implemented pipeline. They do not establish theoretical representation guarantees.

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Reproducing the Evaluation](#-reproducing-the-evaluation)
- [Results](#-results)
- [Evaluation Artifacts](#-evaluation-artifacts)
- [Citation](#-citation)
- [License](#-license)

---

## ✨ Key Features

- **Cost-aware statement selection** — word budget governs both the number and the length of selected statements.
- **Balanced assignment** — each selected statement represents exactly its required number of highest-scoring supporters.
- **Local open-weight models** — query layer runs on `qwen3:8b`; participant embeddings use `qwen3-embedding:0.6b`.
- **Validated query contracts** — generation output is parsed from an `<opinion>` block and checked against the word limit; approval scores are validated integers in `1–6`.
- **Incremental checkpointing** — full run state (cursor, candidates, selections, scores, RNG state) is saved so the experiment can be resumed and inspected.
- **Read-only evaluation** — all metrics computed from the saved checkpoint with no new model calls.

---

## 📂 Repository Structure

```text
CC-GSC/
├── PROSE/
│   ├── algorithm1_abortion.py              # budgeted selection controller
│   ├── datasets/abortion_adapter.py        # survey loading and validation
│   ├── queries/
│   │   ├── abortion_query_contracts.py     # Disc/Gen query contracts
│   │   ├── approval_scorers.py             # Ollama agreement scorer
│   │   ├── fast_queries.py                 # FastAgent / FastGenerator
│   │   └── prompts/                        # prompt templates
│   ├── utils/ollama_provider.py            # Ollama backend
│   └── test/                               # unit tests + diagnostic test
├── calibration/
│   ├── run_cost_constrained_experiment.py  # run Algorithm 1 under a word budget
│   ├── evaluate_500_partial.py             # read-only evaluation
│   ├── participant_summaries.py            # build participant profiles
│   ├── generate_participant_embeddings.py  # embedding generation
│   ├── participant_group_selection.py      # nearest-neighbour groups
│   ├── cost_constrained_experiment_500_final/   # checkpoint, manifest
│   └── evaluation_500_partial/             # CSVs, JSONs, charts, report
├── data/
│   ├── abortion_generation_prose_cleaned.csv
│   └── cleaned_preprocessing_report.txt
├── report/                                 # add LaTeX source and figures here
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

Python **3.12** is required.

```bash
git clone <your-repo-url>
cd CC-GSC
pip install -r requirements.txt
```

The experiment expects an Ollama server with `qwen3:8b` available. The default endpoint used by the scripts is `http://172.21.112.1:11434`; pass a different endpoint/model to the runner if your setup differs.

---

## 🔁 Reproducing the Evaluation

The evaluation needs **no model calls** and never modifies the original checkpoint.

```bash
PYTHONPATH=. python calibration/evaluate_500_partial.py
```

Outputs are written to `calibration/evaluation_500_partial/`. A complete copy of the generated results is already included in this repository.

To run the full experiment (requires Ollama):

```bash
PYTHONPATH=. python calibration/run_cost_constrained_experiment.py --budgets 500
```

---

## 📊 Results

Summary of the recorded 500-word-budget run:

| Measure | Value |
|---|---:|
| Candidate records | 251 (156 successful, 95 failed) |
| Selected statements | 18 (16 unique) |
| Participants represented | 79 / 100 |
| Words consumed | 359 / 500 (71.8%) |
| Approval scores | 4,603 (mean 3.54, median 4) |
| Mean pairwise token Jaccard | 0.177 |
| Recorded runtime | 2,071.93 s |
| Reconciliation checks passed | 16 / 16 |

The run was stopped before the budget was exhausted, so these figures describe the saved partial run. Full tables and figures are in `calibration/evaluation_500_partial/`.

---

## 📈 Evaluation Artifacts

- `evaluation_summary.json` — all computed metrics in one file.
- `extracted_*.csv` — selected statements, candidate records, approval scores, generation failures, score cache.
- `metric_*.csv` / `metrics_*.json` — per-measure results.
- `chart_1_query_accuracy.png` … `chart_5_runtime.png` — evaluation charts.
- `evaluation_report.md` — methodology, results, interpretation and limitations.
- `validation_checks.csv` — all 16 reconciliation checks.

---

## 📚 Citation

This work builds on the following papers, cited throughout the accompanying report:

- Fish et al. *Generative Social Choice*. **Journal of the ACM**, 73(2), Article 11, 2026.
- Boehmer, Fish and Procaccia. *Generative Social Choice: The Next Generation*. arXiv:2505.22939, 2025.
- Aziz et al. *Justified Representation in Approval-Based Committee Voting*. Social Choice and Welfare, 48(2), 2017.
- Yang et al. *Qwen3 Technical Report*. arXiv:2505.09388, 2025.

The implementation adapts the public PROSE codebase (`github.com/sara-fish/gen-soc-choice-next-gen`).

---

## 📄 License

Released under the MIT License — see [LICENSE](LICENSE). The original PROSE code is copyright Niclas Boehmer, Sara Fish and Ariel Procaccia.

<div align="center">

Made for the cost-constrained generative social choice research report.

</div>
