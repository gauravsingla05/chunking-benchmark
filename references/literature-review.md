# Literature Review Tracker

## Papers to Cite (Organized by Category)

---

## A. Chunking for RAG (Core Related Work)

### 1. ✅ Qu et al. (2025) - NAACL
**Title:** "Is Semantic Chunking Worth the Computational Cost?"
**URL:** https://aclanthology.org/2025.findings-naacl.114/
**Key Finding:** Semantic chunking NOT justified by performance gains
**Cite For:** Main comparison, shows gap in generation research
**Status:** Read

### 2. ✅ Bhat et al. (2024) - arXiv
**Title:** "Rethinking Chunk Size For Long-Document Retrieval"
**URL:** https://arxiv.org/abs/2505.21700
**Key Finding:** Optimal size depends on dataset (64-1024 tokens)
**Cite For:** Chunk size considerations
**Status:** Read

### 3. ✅ Mayo Clinic (2024) - PMC
**Title:** "Comparative Evaluation of Advanced Chunking for RAG in Clinical Decision Support"
**URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC12649634/
**Key Finding:** Adaptive chunking wins (87% vs 50%)
**Cite For:** Multiple chunking methods comparison
**Status:** Read

---

## B. Semantic Chunking Methods (To Find)

### 4. ⬜ LangChain Recursive Splitter
**Search:** "recursive text splitter langchain"
**Need:** Original paper or documentation
**Status:** Not started

### 5. ⬜ Sentence-BERT / Embeddings
**Search:** "sentence-transformers reimers gurevych"
**Need:** For semantic chunking baseline
**Status:** Not started

---

## C. Document Summarization (Position Importance)

### 6. ⬜ Lead Bias in Summarization
**Search:** "lead bias news summarization"
**Need:** Supports position-based importance
**Status:** Not started

### 7. ⬜ TextRank
**Search:** "textrank mihalcea tarau 2004"
**Need:** Classic extractive summarization
**Status:** Not started

### 8. ⬜ BertSum / PreSumm
**Search:** "bertsum extractive summarization liu lapata"
**Need:** Neural extractive methods
**Status:** Not started

---

## D. Long Document Processing

### 9. ⬜ Lost in the Middle
**Search:** "lost in the middle liu et al 2023"
**Need:** Why position matters for LLMs
**Status:** Not started

### 10. ⬜ Longformer / BigBird
**Search:** "longformer beltagy"
**Need:** Context length approaches
**Status:** Not started

---

## E. Document-to-Presentation / Content Generation

### 11. ⬜ DOC2PPT
**Search:** "doc2ppt document to presentation"
**Need:** Prior work on slide generation
**Status:** Not started

### 12. ⬜ Automatic Slide Generation
**Search:** "automatic slide generation from documents"
**Need:** Task definition
**Status:** Not started

---

## F. Evaluation Methods

### 13. ⬜ ROUGE Score
**Search:** "rouge lin 2004 summarization"
**Need:** Evaluation metric reference
**Status:** Not started

### 14. ⬜ BERTScore
**Search:** "bertscore zhang et al"
**Need:** Semantic similarity metric
**Status:** Not started

### 15. ⬜ LLM-as-Judge
**Search:** "llm as judge evaluation zheng 2023"
**Need:** Quality evaluation approach
**Status:** Not started

---

## Search Queries to Run

### Google Scholar:
```
1. "document chunking" "language model"
2. "text segmentation" RAG retrieval
3. "extractive summarization" position
4. "document to slides" OR "document to presentation"
5. "long document" LLM context
6. "chunk size" embedding retrieval
```

### arXiv:
```
1. cs.CL chunking retrieval
2. cs.CL document summarization extractive
3. cs.CL long context LLM
```

### Semantic Scholar:
```
1. semantic chunking RAG
2. document compression language model
3. slide generation from documents
```

---

## Citation Format (IEEE)

```
[1] R. Qu, R. Tu, and F. S. Bao, "Is Semantic Chunking Worth the Computational Cost?,"
    in Findings of NAACL, 2025.

[2] S. R. Bhat et al., "Rethinking Chunk Size For Long-Document Retrieval:
    A Multi-Dataset Analysis," arXiv:2505.21700, 2024.

[3] C. A. Gomez-Cabello et al., "Comparative Evaluation of Advanced Chunking for
    Retrieval-Augmented Generation," PMC, 2024.
```

---

## Notes

### Key Themes to Highlight:
1. **Gap:** All existing work = retrieval, none = generation
2. **Position importance:** Known in summarization, not applied to chunking
3. **Cost-benefit:** Semantic chunking expensive, may not be worth it
4. **Practical:** Real-world systems need efficiency

### Potential Reviewers Might Ask:
- "How is this different from extractive summarization?" → Different goal, chunking preserves more
- "Why not just use longer context models?" → Cost, lost-in-middle problem
- "Is position-based generalizable?" → Test on multiple doc types
