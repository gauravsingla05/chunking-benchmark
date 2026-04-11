# Dataset: 101 Research Papers for Chunking Benchmark

This directory contains metadata for the 101 research papers used in the study "Position-Aware Versus Semantic Chunking for Content Generation."

## Dataset Overview

| Property | Value |
|----------|-------|
| Total documents | 101 |
| Minimum word count | 2,985 |
| Maximum word count | 29,337 |
| Median word count | 8,800 |
| Mean word count | 10,247 |

### Domain Distribution

| Domain | Count |
|--------|-------|
| Artificial Intelligence / Machine Learning | 32 |
| Medical Research | 24 |
| Education | 18 |
| Finance / Economics | 15 |
| General Computer Science | 12 |

## Files

- `dataset_papers.csv` - Complete list of all 101 papers with metadata
- `raw/` - Directory where PDF files should be placed (not included in repository)

## How to Obtain the Papers

Due to copyright restrictions, the original PDF files are not included in this repository. To reproduce the experiments:

1. **Download papers individually** using the information in `dataset_papers.csv`
2. **Sources include:**
   - arXiv (arxiv.org) - Open access preprints
   - PubMed Central (ncbi.nlm.nih.gov/pmc) - Open access medical papers
   - SSRN (ssrn.com) - Working papers in economics/finance
   - Institutional repositories - University-hosted papers
   - PLOS ONE (journals.plos.org) - Open access research

3. **Place downloaded PDFs** in the `raw/` directory with filenames matching those in `dataset_papers.csv`

## Selection Criteria

Papers were selected based on:
1. English language
2. PDF format with extractable text
3. Standard academic structure (abstract, introduction, methodology, results, conclusion)
4. Word count between 2,500 and 30,000 words
5. Publication date between 2019 and 2024

Papers were excluded if they:
- Contained primarily figures or tables without substantial prose
- Were non-research documents (reviews, editorials, commentaries)
- Had corrupted or incomplete text extraction

## Citation

If you use this dataset, please cite:

```
@article{singla2026chunking,
  title={Position-Aware Versus Semantic Chunking for Content Generation: Small Gains, Big Trade-Offs},
  author={Singla, Gourav},
  journal={IEEE Access},
  year={2026}
}
```

## Contact

For questions about the dataset, contact: gouravsingla05@gmail.com
