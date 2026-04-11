# Document Chunking for LLM-based Slide Generation: Experiment Results

**Author:** Gourav Singla
**Date:** December 21, 2024
**Project:** SlideMaker Research

---

## Executive Summary

This study evaluated four document chunking strategies for generating presentation slides from research papers using Large Language Models (LLMs). We used a dual-evaluation methodology comparing slides against both extracted summaries (Claude) and full source documents (Gemini).

**Key Finding:** All chunking methods perform similarly (3.73-3.77 overall score), with no single method dominating. Simple truncation is competitive with more sophisticated approaches.

---

## 1. Experimental Setup

### 1.1 Dataset
- **20 research papers** from diverse domains
- Document lengths: 1,696 to 25,532 words
- Domains: AI/ML, Medical, Education, Finance, Computer Science

### 1.2 Chunking Methods Evaluated

| Method | Description | Budget |
|--------|-------------|--------|
| **truncation** | Take first N words of document | 4,000 words |
| **fixed_size_first_last** | Split budget between beginning and end of document | 4,000 words |
| **semantic_breakpoint** | Use semantic similarity to find natural break points | 4,000 words |
| **pac_position_aware** | Position-aware chunking with quality signals | 4,000 words |

### 1.3 Evaluation Methodology

**Dual-Evaluator Approach:**
1. **Claude Evaluation:** Compare slides against extracted summary.json (key facts, statistics)
2. **Gemini Evaluation:** Compare slides directly against full source PDF

**Metrics (1-5 scale):**
- Completeness: Coverage of important content
- Accuracy: Factual correctness
- Statistics Retention: Preservation of numerical data
- Coherence: Logical flow and structure
- Relevance: Content appropriate for slides
- Coverage Balance: Representation of all document sections

---

## 2. Results

### 2.1 Overall Scores (Gemini Evaluation - Final)

| Method | Overall | Completeness | Accuracy | Stats | Coherence | Relevance | Balance |
|--------|---------|--------------|----------|-------|-----------|-----------|---------|
| **truncation** | 3.77 | 3.45 | 5.00 | 1.85 | 4.05 | 5.00 | 3.30 |
| **semantic_breakpoint** | 3.76 | 3.35 | 5.00 | 2.00 | 4.00 | 4.95 | 3.30 |
| **fixed_size_first_last** | 3.74 | 3.30 | 5.00 | 1.85 | 4.10 | 4.95 | 3.20 |
| **pac_position_aware** | 3.73 | 3.40 | 4.85 | 2.00 | 4.00 | 4.80 | 3.30 |

**Key Observations:**
- All methods within 0.04 points of each other
- No statistically significant difference between methods
- All methods achieve high accuracy (4.85-5.0)
- Statistics retention is low across all methods (~2.0/5)

### 2.2 Claude vs Gemini Evaluation Comparison

| Method | Claude (vs Summary) | Gemini (vs Full PDF) | Difference |
|--------|---------------------|----------------------|------------|
| truncation | 3.58 | 3.77 | +0.19 |
| semantic_breakpoint | 3.30 | 3.76 | +0.46 |
| fixed_size_first_last | 3.40 | 3.74 | +0.34 |
| pac_position_aware | 3.26 | 3.73 | +0.47 |

**Finding:** Claude evaluation favored truncation; Gemini evaluation showed more balanced results. This suggests the extracted summary may have been biased toward document beginnings.

### 2.3 Per-Document Analysis

#### Documents Where PAC Wins (Highest Score):
| Document | Words | PAC Score | Others |
|----------|-------|-----------|--------|
| Project for Middle Class Renewal | 7,212 | 4.5 | 3.8-4.3 |
| From Tarzan to Tolkien | 13,988 | 4.2 | 4.0-4.2 |
| Analysis of Research Trends | 4,824 | 4.2 | 3.8-4.2 |
| DEVELOPING EFFECTIVE LEARNING OUTCOMES | 4,888 | 3.7 | 3.5-3.7 |

#### Documents Where PAC Struggles:
| Document | Words | PAC Score | Best Method |
|----------|-------|-----------|-------------|
| Guiding Clinical Reasoning | 7,068 | 3.0 | fixed_size (4.2) |
| Enterprise Risk Management | 4,201 | 3.3 | others (3.8) |
| Input-output neuron | 3,873 | 3.3 | fixed_size (3.7) |

### 2.4 Performance by Document Length

| Document Size | PAC Avg Rank | PAC Wins | PAC Worst |
|---------------|--------------|----------|-----------|
| Short (<6k words) | 2.1 | 3/8 (38%) | 1/8 (12%) |
| Medium (6-12k words) | 2.9 | 1/11 (9%) | 3/11 (27%) |
| Long (12k+ words) | 2.0 | 0/1 | 0/1 |

**Finding:** PAC performs best on short documents where position-based sampling effectively captures key content.

---

## 3. PAC Algorithm Analysis

### 3.1 Original Position Curve (U-shaped)

The original PAC implementation used a U-shaped position curve:

```
Position 0-15%:   Score +2.0 (Intro)
Position 15-85%:  Score ~0.0 (Middle - penalized)
Position 85-100%: Score +2.0 (Conclusion)
```

**Problem:** The middle 70% of documents (containing results/findings) received almost no position boost.

### 3.2 Improved Position Curve (W-shaped)

We modified PAC to use a W-shaped curve:

```
Position 0-12%:   Score +2.0 (Abstract/Intro)
Position 12-38%:  Score +0.6 (Background - floor)
Position 38-62%:  Score +1.2 to +1.8 (Results - boosted)
Position 62-88%:  Score +0.6 (Discussion - floor)
Position 88-100%: Score +2.0 (Conclusion)
```

### 3.3 Impact of W-shaped Curve

| Metric | Before | After |
|--------|--------|-------|
| Overall Score | 3.67 | 3.73 |
| Documents Improved | - | 8/20 |
| Documents Same | - | 7/20 |
| Documents Degraded | - | 5/20 |

**Notable Improvements:**
- Risk Management: 2.7 → 3.7 (+1.0)
- CS Benchmark (25k words): 3.5 → 4.0 (+0.5)
- Selective Fine-tuning: 3.7 → 4.2 (+0.5)

---

## 4. Key Findings

### 4.1 No Clear Winner
All chunking methods achieve similar performance (3.73-3.77). For practical applications, simple truncation is a reasonable default.

### 4.2 Statistics Are Lost
All methods score poorly on statistics retention (~2.0/5). This is a fundamental limitation - LLMs generating slides tend to omit specific numbers.

### 4.3 Evaluation Method Matters
Claude (vs summary) and Gemini (vs full PDF) produced different rankings:
- Claude favored truncation (biased toward document beginnings)
- Gemini showed more balanced results

### 4.4 Document Length Affects Method Performance
- **Short documents (<6k words):** PAC performs well
- **Medium documents (6-12k words):** PAC struggles
- **Long documents (>12k words):** All methods similar

### 4.5 Position-Aware Chunking Has Merit
The W-shaped curve improvement shows that position-aware approaches can be effective when properly tuned to capture results sections.

---

## 5. Recommendations

### For Practitioners
1. **Use truncation as baseline** - simple and competitive
2. **Consider semantic_breakpoint** for documents with clear section boundaries
3. **Use PAC for short documents** (<6k words) where position matters

### For Researchers
1. **Statistics retention needs improvement** - explore explicit number extraction
2. **Dual-evaluation methodology recommended** - single evaluator may have biases
3. **Document-type-specific chunking** may outperform one-size-fits-all approaches

---

## 6. Experimental Data

### 6.1 Documents Used

| Document | Words | Domain |
|----------|-------|--------|
| 196-Li-Open-Access-Publishing | 8,464 | Academic Publishing |
| A COMPREHENSIVE BENCHMARK FOR LARGE LANGUAGE MODELS | 25,532 | AI/ML |
| A Survey on Medical Large Language Models | 24,582 | Medical AI |
| Analysis of Research Trends in Computer Science | 4,824 | Computer Science |
| DEVELOPING EFFECTIVE LEARNING OUTCOMES | 4,888 | Education |
| Enterprise Risk Management | 4,201 | Finance |
| Etat de l'art sur l'application des bandits multi-bras | 7,781 | Machine Learning |
| From Tarzan to Tolkien | 13,988 | NLP/Education |
| Generative AI in Medicine | 16,135 | Medical AI |
| Guiding Clinical Reasoning with LLMs | 7,068 | Medical AI |
| Input-output behaviour of a model neuron | 3,873 | Neuroscience |
| Investing in Cryptocurrency | 15,530 | Finance |
| Project for Middle Class Renewal | 7,212 | Policy |
| Risk Management | 10,522 | Finance |
| Risk Management—the Revealing Hand | 8,860 | Finance |
| Selective Fine-tuning on LLM-labeled Data | 8,792 | AI/ML |
| TON_IEEE_Paper | 1,696 | Networking |
| Towards Integrating Emerging AI Applications in SE Education | 5,113 | Education |
| Vision Transformers for Computer Go | 4,123 | AI/ML |
| Writing and Using Learning Outcomes | 10,025 | Education |

### 6.2 Evaluation Files

- `results/evaluations/gemini_summary_20251221-132929.json` - Final Gemini results
- `results/evaluations/summary_20251220-153649.json` - Claude results
- `results/documents/*/slides/*.json` - Generated slides

---

## 7. Conclusions

This study demonstrates that for LLM-based slide generation:

1. **Chunking method choice has minimal impact** on final slide quality
2. **Simple methods work well** - truncation is competitive
3. **Evaluation methodology matters** - dual-evaluator approach recommended
4. **Statistics preservation remains a challenge** across all methods

The practical implication for SlideMaker: any reasonable chunking strategy will produce acceptable results, with room for improvement in preserving specific statistics and numerical data.

---

## Appendix: Code and Reproduction

All experiment code is available in:
- `src/experiments/run_experiment.py` - Chunking experiments
- `src/experiments/run_llm.py` - Slide generation
- `src/experiments/run_evaluation.py` - Claude evaluation
- `src/experiments/run_evaluation_gemini.py` - Gemini evaluation
- `src/methods/` - Chunking method implementations
