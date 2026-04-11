# Document Chunking Benchmark for LLM Content Generation

Companion code and data for the paper:

> **"Position-Aware Versus Semantic Chunking for Content Generation: Small Gains, Big Trade-Offs"**
> Gourav Singla — IEEE Access, 2025/2026

## Overview

This repository provides the experiment pipeline, evaluation scripts, and aggregated results for a multi-model, multi-task study comparing four document chunking strategies for LLM-based content generation.

**Chunking Methods:**
- Truncation (baseline)
- Fixed-Size First-Last
- Semantic Breakpoint Chunking
- Position-Aware Chunking (PAC) with W-shaped scoring curve

**Key Findings:**
- Simple truncation achieves the highest overall slide score (3.98/5)
- PAC outperforms all methods for medium-length documents (5K–10K words)
- On Q&A, truncation dominates with 61.2% accuracy vs PAC's 43.3%
- Three independent LLM judges agree on rankings (Pearson r = 0.52–0.61)
- Human evaluation on 50 blind pairs confirms AI judge rankings

## Repository Structure

```
src/
├── config.py                       # Centralized experiment configuration
├── methods/                        # Chunking implementations
│   ├── truncation.py
│   ├── fixed_size_first_last.py
│   ├── semantic_breakpoint.py
│   └── pac_position_aware.py
├── experiments/                    # Experiment runners
│   ├── run_pipeline.py             # Master orchestrator
│   ├── run_experiment.py           # Chunking runner
│   ├── run_generation.py           # Multi-model slide/summary generation
│   ├── run_generation_parallel.py  # Parallel Gemini generation
│   ├── run_qa.py                   # Q&A pipeline (questions + answers + eval)
│   ├── run_evaluation_multi.py     # Multi-judge evaluation
│   ├── run_eval_parallel.py        # Parallel evaluation
│   ├── run_ablation.py             # Chunk budget ablation
│   ├── run_sensitivity.py          # W-curve parameter sensitivity
│   ├── run_human_eval.py           # Human evaluation export/import
│   └── shared.py                   # Shared utilities (API clients, batch, retry)
├── analysis/                       # Statistical analysis
│   ├── compute_statistics_v2.py    # Multi-model multi-judge stats
│   └── full_analysis.py            # Comprehensive analysis
└── utils/
    └── text_io.py                  # PDF/text loading

data/
├── dataset_listing.csv             # 221 document filenames and titles
├── dataset_papers.csv              # Original 101-paper metadata
└── raw/                            # Source PDFs (not in repo, see below)

results/
├── full_analysis.json              # All aggregated statistics
├── statistical_analysis_v2.json    # Multi-model paired t-tests
├── sensitivity_analysis.json       # W-curve parameter sensitivity
├── ablation_config.json            # Ablation study metadata
├── human_evaluation_results.json   # 50-pair human evaluation
└── qa/
    └── qa_results_summary.json     # Q&A accuracy by method
```

## Dataset

The 218 source documents are **not included** in this repository (copyright). The file `data/dataset_listing.csv` lists each document by filename. Documents were collected from publicly available sources (arXiv, PubMed, SSRN, institutional repositories, government websites, corporate publications).

To reconstruct the corpus, search for each document title and download the PDF to `data/raw/`.

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install sentence-transformers openai anthropic google-genai scipy numpy

# 2. Set API keys (or create .env file)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=AIza...

# 3. Place PDFs in data/raw/

# 4. Run the full pipeline
cd src
python experiments/run_pipeline.py --batch --dry-run  # Cost estimate
python experiments/run_pipeline.py --batch              # Full run

# Or run individual stages:
python experiments/run_pipeline.py --stages chunk
python experiments/run_pipeline.py --stages generate --batch
python experiments/run_pipeline.py --stages evaluate --batch
python experiments/run_qa.py --stage all --workers 15
```

## Models Used

**Generation:** GPT-4o, Claude Sonnet 4, Gemini 2.0 Flash
**Evaluation (Judges):** Gemini 2.0 Flash, GPT-4o-mini, Claude Sonnet 4
**Embeddings:** all-MiniLM-L6-v2 (for semantic chunking)

## Citation

```bibtex
@article{singla2025chunking,
  title={Position-Aware Versus Semantic Chunking for Content Generation: Small Gains, Big Trade-Offs},
  author={Singla, Gourav},
  journal={IEEE Access},
  year={2025}
}
```

## License

Code is released under the MIT License. The source documents retain their original copyrights.
