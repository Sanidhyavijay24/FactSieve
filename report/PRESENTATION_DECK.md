# Presentation Deck: FactSieve (RefChecker-Mini)
## Fine-Grained Knowledge-Triplet Fact Verification & Ablation Study (EMNLP 2024)

---

### Slide 1: Title & Motivation
* **Title:** FactSieve: Fine-Grained Hallucination Detection in LLM Responses
* **Paper Basis:** *Xiangkun Hu et al., EMNLP 2024* (RefChecker)
* **The Core Problem:** Sentence-level checking often misses hallucinations when a sentence contains both true and false facts (e.g., *"Sydney is Australia's capital and largest city"*).
* **Our Solution:** Extract fine-grained `(subject, relation, object)` knowledge triplets, check each against evidence, and aggregate deterministically.

---

### Slide 2: Research Questions
* **RQ1 (Granularity):** Does decomposing responses into atomic claims or triplets improve hallucination detection over full sentences?
* **RQ2 (Checker Ablation):** How does NLI (DeBERTa-v3) compare against an LLM Judge (Gemini) and Semantic Similarity (MiniLM) on triplets?
* **RQ3 (Robustness):** How well do triplet methods handle noisy evidence containing distractor passages?

---

### Slide 3: System Architecture
```
[Question + Context] ───► [LLM Response Generation]
                                 │
                                 ▼
   ┌─────────────────────────────────────────────────────────────┐
   │             STAGE 1: Granularity Extraction                 │
   │  • Sentence Segmentation (spaCy)                            │
   │  • Atomic Claim Extraction (Gemini Structured Prompt)        │
   │  • Triplet Extraction [s, r, o] + Verbalizer                │
   └─────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
   ┌─────────────────────────────────────────────────────────────┐
   │             STAGE 2: Fact Verification Checkers             │
   │  • Local GPU NLI (DeBERTa-v3 MNLI)                          │
   │  • Few-Shot LLM Judge (Gemini 3.5 Flash-Lite)               │
   │  • Cosine Similarity (Sentence-Transformers MiniLM)         │
   └─────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
   ┌─────────────────────────────────────────────────────────────┐
   │       STAGE 3: Aggregation & Hallucination Scoring          │
   │  Strict Rule: Any Unit Contradiction ──► Response Hallucinated│
   └─────────────────────────────────────────────────────────────┘
```

---

### Slide 4: Key Results Table
| Representation / Verifier | Accuracy | Macro-F1 | Hallucination Precision | Hallucination Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Sentence $\to$ NLI (Baseline)** | 80.00% | 0.5714 | 0.9286 | 0.9286 |
| **Claims $\to$ NLI (Atomic)** | 83.33% | 0.5827 | 0.8750 | **1.0000** |
| **Triplets $\to$ NLI (RefChecker)** | **83.33%** | **0.5854** | **0.9286** | 0.9286 |
| **Triplets $\to$ LLM Judge** | **90.00%** | **0.6221** | **1.0000** | 0.9286 |
| **Triplets $\to$ Semantic Similarity** | 40.00% | 0.2500 | 0.0000 | 0.0000 |

---

### Slide 5: Core Takeaways & Insights
1. **Finer Granularity Works:** Decomposing sentences into atomic claims or triplets increased accuracy by +3.33% and reached **100% Hallucination Recall** for atomic claims.
2. **Triplets Deliver Higher Precision:** Triplets avoid false alarms by strictly binding entities to relationships `(subject, relation, object)` rather than free-floating clauses.
3. **Semantic Similarity is Blind to Truth:** Embeddings measure topical overlap, not factual correctness, catching 0/14 factual errors.
4. **Strong Noise Robustness:** Triplet NLI achieved **100% Accuracy on Noisy Context with Distractors**, proving triplets cleanly isolate target facts inside cluttered documents.

---

### Slide 6: Deliverables & Artifacts
* Complete Reproducible Codebase in Python 3.10 / Conda / PyTorch CUDA
* Persistent Disk Cache with SHA-256 keying for zero API spend
* Publication Plots in `results/plots/` (`rq1_granularity_comparison.png`, `rq2_checker_comparison.png`, `confusion_matrices_grid.png`)
* Full Markdown Research Report in `report/FINAL_PROJECT_REPORT.md`
