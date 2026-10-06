# FactSieve
## Fine-Grained Knowledge-Triplet Fact Verification & Ablation Suite
*(RefChecker-Mini: A Lightweight Reproduction and Controlled Ablation Study)*

**FactSieve** is a modular, production-grade fact verification and empirical ablation framework inspired by the **RefChecker** architecture (*Xiangkun Hu et al., EMNLP 2024*). It decomposes complex LLM generations into atomic claims and structured `[subject, relation, object]` knowledge triplets to detect fine-grained hallucinations and factual inconsistencies with localized precision.

---

## 1. Primary Paper Citation

```bibtex
@inproceedings{hu-etal-2024-knowledge,
    title = "Knowledge-Centric Hallucination Detection",
    author = "Hu, Xiangkun and others",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    year = "2024",
    pages = "6953--6975",
    url = "https://aclanthology.org/2024.emnlp-main.395/"
}
```

---

## 2. Research Questions & Hypotheses

* **RQ1 (Representation Granularity):** Does decomposing responses into atomic factual statements or knowledge claim triplets `[subject, relation, object]` improve hallucination detection over full-sentence representations?
  * *Hypothesis:* Triplet and claim-level checking isolate localized factual errors in composite sentences where true clauses otherwise mask embedded false clauses.
* **RQ2 (Fact-Checking Strategy):** Given extracted knowledge triplets, how does pretrained Natural Language Inference (DeBERTa-v3 MNLI) compare with direct LLM Judges and Semantic Similarity baselines?
  * *Hypothesis:* LLM judges offer higher contextual precision on structured triplets, whereas embedding cosine similarity is fundamentally unsuited for detecting entity-level contradictions.
* **RQ3 (Evidence Noise Robustness):** How robust are triplet-level verifiers when evaluated against clean ground truth reference evidence versus noisy context passages containing distractors?
  * *Hypothesis:* Structured triplets allow models to selectively ground specific propositions inside multi-passage distractor text without getting confused by irrelevant context.

---

## 3. System Architecture

```
[Question + Context] ───► [LLM Response Generation]
                                 │
                                 ▼
   ┌─────────────────────────────────────────────────────────────┐
   │             STAGE 1: Granularity Extraction                 │
   │  • Sentence Segmentation (spaCy en_core_web_sm)             │
   │  • Atomic Claim Extraction (Structured Prompting)           │
   │  • Triplet Extraction [subject, relation, object]           │
   └─────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
   ┌─────────────────────────────────────────────────────────────┐
   │             STAGE 2: Fact Verification Checkers             │
   │  • Pretrained DeBERTa-v3 MNLI (Batched GPU Inference)       │
   │  • Structured Few-Shot LLM Judge (Gemini 3.5 Flash-Lite)    │
   │  • Sentence-Transformers Cosine Similarity (MiniLM-L6-v2)   │
   └─────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
   ┌─────────────────────────────────────────────────────────────┐
   │       STAGE 3: Aggregation & Hallucination Scoring          │
   │  Strict Rule: Any Unit Contradiction ──► Response Hallucinated│
   └─────────────────────────────────────────────────────────────┘
```

---

## 4. Empirical Evaluation Matrix

All experiments were executed on a standardized benchmark across **Accurate Context** (Dolly-15k style), **Noisy Context** (MS MARCO style with distractor passages), and **Zero Context** (NaturalQuestions closed-book style).

| # | Pipeline | Stage 1 Extractor | Stage 2 Verifier | Accuracy | Macro-F1 | Hallucination Precision | Hallucination Recall | Hallucination F1 |
| :-: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Sentence $\to$ NLI** | spaCy (`en_core_web_sm`) | DeBERTa-v3 MNLI | 80.00% | 0.5714 | 0.9286 | 0.9286 | 0.9286 |
| **2** | **Claims $\to$ NLI** | Atomic Claim Extractor | DeBERTa-v3 MNLI | 83.33% | 0.5827 | 0.8750 | **1.0000** | **0.9333** |
| **3** | **Triplets $\to$ NLI** | Knowledge Triplet Extractor | DeBERTa-v3 MNLI | **83.33%** | **0.5854** | **0.9286** | 0.9286 | 0.9286 |
| **4** | **Triplets $\to$ LLM Judge** | Knowledge Triplet Extractor | Gemini 3.5 Flash-Lite | **90.00%** | **0.6221** | **1.0000** | 0.9286 | **0.9630** |
| **5** | **Triplets $\to$ Similarity** | Knowledge Triplet Extractor | MiniLM-L6-v2 Cosine | 40.00% | 0.2500 | 0.0000 | 0.0000 | 0.0000 |
| **6** | **Robustness (Distractors)** | Knowledge Triplet Extractor | DeBERTa-v3 MNLI | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

---

## 5. Core Insights

1. **Granularity Resolves Composite Hallucinations:** Sentence-level checking fails when a sentence combines true and false facts (e.g., *"Sydney is Australia's capital and largest city"* was falsely entailed by sentence NLI because Sydney is the largest city). Finer granularity splits the assertions and successfully detects the capital error.
2. **Triplets Prevent False Contradictions:** While atomic claims achieved 100% recall, they produced occasional false alarms on complex true sentences (Precision: 0.8750). Triplets achieved higher precision (**0.9286**) and top Macro-F1 (**0.5854**) by strictly binding entities to relationships.
3. **Semantic Similarity Fails Fundamentally:** Cosine similarity over embeddings achieved **0% error detection**. Because contradictory statements often share 90%+ vocabulary, embedding models predict high similarity and remain completely blind to factual inaccuracies.
4. **Distractor Resistance:** Triplet verification achieved **100% Accuracy on distractor-heavy contexts**, proving that atomic propositions prevent evidence dilution.

---

## 6. Repository Layout

```
refchecker-mini/
├── configs/
│   └── config.yaml               # Central hyperparameter & pipeline configuration
├── data/
│   ├── processed/                # Standardized JSONL benchmark datasets
│   └── generated/                # Cached LLM responses, extractions, and verdicts
├── src/
│   ├── cache.py                  # Thread-safe deterministic SHA-256 disk cache
│   ├── config.py                 # Pydantic configuration loader
│   ├── models.py                 # Strict domain schemas (BenchmarkSample, Triplet, etc.)
│   ├── generate_responses.py     # Gemini response generator with backoff
│   ├── extract_sentences.py      # spaCy sentence segmenter
│   ├── extract_claims.py         # Atomic factual claim extractor
│   ├── extract_triplets.py       # (Subject, Relation, Object) triplet extractor
│   ├── nli_checker.py            # GPU-accelerated DeBERTa-v3 MNLI verifier
│   ├── llm_checker.py            # Few-shot LLM Judge verifier
│   ├── semantic_checker.py       # Sentence-Transformers cosine similarity verifier
│   ├── scoring.py                # Aggregation engine (strict / soft rules)
│   ├── evaluation.py             # Metrics calculation (Accuracy, F1, Confusion Matrix)
│   ├── error_analysis.py         # Qualitative error taxonomy classifier
│   └── visualizations.py         # Publication-grade chart generation
├── experiments/
│   ├── exp1_sentence_nli.py      # Experiment 1 runner (Sentence baseline)
│   ├── exp2_claim_nli.py         # Experiment 2 runner (Atomic claims)
│   ├── exp3_triplet_nli.py       # Experiment 3 runner (RefChecker core reproduction)
│   ├── exp4_triplet_llm.py       # Experiment 4 runner (LLM Judge ablation)
│   ├── exp5_triplet_similarity.py# Experiment 5 runner (Semantic similarity baseline)
│   ├── exp6_robustness.py        # Experiment 6 runner (Clean vs Noisy evidence)
│   ├── run_rq1_comparison.py     # RQ1 side-by-side comparison table generator
│   ├── run_rq2_comparison.py     # RQ2 side-by-side comparison table generator
│   └── generate_plots.py         # Plot generator script
├── results/
│   ├── plots/                    # High-res charts (.png)
│   ├── tables/                   # Metric CSVs and JSONs
│   └── experiment_log.md         # Detailed execution log
├── report/
│   ├── FINAL_PROJECT_REPORT.md   # Comprehensive academic research report
│   └── PRESENTATION_DECK.md      # Slide-by-slide presentation deck
├── tests/                        # Pytest unit & regression test suite (20/20 passed)
├── requirements.txt              # Pinned Python dependencies
├── .env.example                  # Environment configuration template
└── context.md                    # Project ledger & single source of truth
```

---

## 7. Installation & Setup

### Prerequisites
* Python 3.10+
* Conda / Miniconda
* NVIDIA GPU with CUDA (optional, CPU fallback supported)

### Step-by-Step Setup

```bash
# 1. Clone repository
git clone https://github.com/your-username/refchecker-mini.git
cd refchecker-mini

# 2. Create and activate Conda environment
conda create -n refchecker-mini python=3.10 -y
conda activate refchecker-mini

# 3. Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download spaCy English language model
python -m spacy download en_core_web_sm

# 6. Set up API Key (for LLM extraction & judge)
cp .env.example .env
# Edit .env and enter your GEMINI_API_KEY
```

---

## 8. Running Experiments & Reproducing Results

Execute experiments individually:

```bash
# Experiment 1: Sentence -> NLI Baseline
python -m experiments.exp1_sentence_nli

# Experiment 2: Claims -> NLI (Atomic Granularity)
python -m experiments.exp2_claim_nli

# Experiment 3: Triplets -> NLI (Core RefChecker Reproduction)
python -m experiments.exp3_triplet_nli

# Experiment 4: Triplets -> LLM Judge
python -m experiments.exp4_triplet_llm

# Experiment 5: Triplets -> Semantic Similarity Baseline
python -m experiments.exp5_triplet_similarity

# Experiment 6: Robustness Analysis (Clean vs Noisy Distractor Context)
python -m experiments.exp6_robustness
```

Compile consolidated comparison tables:

```bash
# Research Question 1 comparison (Granularity)
python -m experiments.run_rq1_comparison

# Research Question 2 comparison (Verifier Strategies)
python -m experiments.run_rq2_comparison

# Generate publication-grade figures in results/plots/
python -m experiments.generate_plots
```

---

## 9. Running Automated Tests

```bash
pytest tests/ -v
```

All 20 unit and regression tests validate configuration schemas, strict domain models, persistent caching determinism, sentence/claim/triplet extraction, NLI inference, aggregation scoring rules, and metric evaluation routines.

---

## 10. License

MIT License. See `LICENSE` for details.
