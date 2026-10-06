# Project Context: FactSieve (RefChecker-Mini)

## 1. Project Overview
- **Name:** FactSieve (RefChecker-Mini: Knowledge-Triplet Fact Verification & Ablation Study)
- **Primary Reference:** *Xiangkun Hu et al.*, "Knowledge-Centric Hallucination Detection", EMNLP 2024.
- **Mission:** Reproduce the fine-grained claim-triplet hallucination detection mechanism, run controlled ablations (Sentence vs Claims vs Triplets; NLI vs LLM Judge vs Semantic Similarity), and evaluate robustness under accurate vs noisy context.
- **Repository Strategy:** Lightweight inference pipelines with zero-waste disk caching of all API/LLM calls.

## 2. Tech Stack
- **Language / Runtime:** Python 3.10 (Conda Environment: `refchecker-mini`)
- **Deep Learning / NLP:** PyTorch 2.6+ (CUDA 12.4), Hugging Face Transformers, Sentence-Transformers, spaCy (`en_core_web_sm`)
- **LLM API:** Google Gemini API (`gemini-2.5-flash` / `gemini-1.5-flash`)
- **Data & Validation:** Pydantic v2 (Strict Schema Enforcement), PyYAML, Pandas, NumPy
- **Testing & QA:** Pytest 8+, Black formatting, PEP 257 docstrings

## 3. Architecture & Directory Layout
```
refchecker-mini/
├── configs/
│   └── config.yaml               # Central hyperparameter & pipeline configuration
├── data/
│   ├── processed/                # Unified benchmark JSONL datasets
│   └── generated/                # Cached LLM responses, extractions, and judge verdicts
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuration loader & Pydantic settings
│   ├── models.py                 # Core domain schemas (BenchmarkSample, Triplet, etc.)
│   ├── generate_responses.py     # Gemini response generator with disk caching
│   ├── extract_claims.py         # Atomic claim extractor
│   ├── extract_triplets.py       # Triplet (s, r, o) extractor & verbalizer
│   ├── nli_checker.py            # HF DeBERTa-v3 MNLI verifier
│   ├── llm_checker.py            # LLM Judge verifier with cached judgments
│   ├── semantic_checker.py       # SentenceTransformers cosine similarity verifier
│   ├── scoring.py                # Aggregation functions (strict / soft)
│   └── evaluation.py             # Metrics calculation (Accuracy, F1, Confusion Matrix)
├── experiments/
│   ├── exp1_sentence_nli.py      # Experiment 1: Sentence -> NLI baseline
│   ├── exp2_claim_nli.py         # Experiment 2: Claim -> NLI
│   ├── exp3_triplet_nli.py       # Experiment 3: Triplet -> NLI (Core reproduction)
│   ├── exp4_triplet_llm.py       # Experiment 4: Triplet -> LLM Judge ablation
│   ├── exp5_triplet_similarity.py# Experiment 5: Triplet -> Semantic Similarity
│   └── exp6_robustness.py        # Experiment 6: Accurate vs Noisy Context Robustness
├── results/
│   ├── plots/                    # Confusion matrices, ROC curves, bar charts
│   └── tables/                   # Summary CSV/JSON metric tables
├── tests/
│   ├── test_config.py            # Configuration & model tests
│   ├── test_extraction.py        # Sentence, claim, and triplet extraction tests
│   ├── test_checkers.py          # Verifiers & aggregation unit tests
│   └── test_evaluation.py        # Metric calculation unit tests
├── report/                       # Final report markdown & presentation notes
├── context.md                    # Project single source of truth ledger
├── requirements.txt              # Pinned Python package dependencies
├── .env.example                  # Environment variables template
└── .gitignore
```

### Data Flow
1. `Benchmark Data Ingestion` -> `data/processed/benchmark_subset.jsonl`
2. `Response Generation / Caching` -> `data/generated/llm_cache.jsonl`
3. `Granularity Extraction` (Sentence / Atomic Claim / Triplet) -> `data/generated/extraction_cache.jsonl`
4. `Verifier Execution` (NLI / LLM Judge / Semantic Similarity) -> `VerificationUnitResult`
5. `Response-Level Scoring & Aggregation` -> `ResponseEvaluationResult`
6. `Evaluation & Matrix Compilation` -> `results/tables/` & `results/plots/`

## 4. Feature Status Checklist
- [x] **Phase 1: Project Foundation & Environment Setup**
  - [x] Repository structure creation
  - [x] `requirements.txt` & `.env.example`
  - [x] `configs/config.yaml`
  - [x] `src/config.py` with Pydantic validation
  - [x] `src/models.py` with strict domain contracts
  - [x] `tests/test_config.py` passed
- [x] **Phase 2: Benchmark Data Acquisition & Response Generation Pipeline**
  - [x] Raw benchmark ingestion & dataset creation across Zero, Noisy, and Accurate Context settings (`data/processed/benchmark_subset.jsonl`)
  - [x] Thread-safe deterministic disk cache (`src/cache.py`) with SHA-256 key hashing
  - [x] LLM Response Generator module (`src/generate_responses.py`) with rate-limit pacing and automatic backoff
  - [x] `tests/test_data_and_generation.py` passed (7 total test cases passing)
- [x] **Phase 3: Extraction Engines (Sentence, Claim, Triplet)**
  - [x] spaCy sentence segmentation module (`src/extract_sentences.py`) with bullet cleaning & fallbacks
  - [x] Atomic claim extraction module (`src/extract_claims.py`) with Gemini structured prompt & caching
  - [x] Knowledge triplet extractor & verbalizer (`src/extract_triplets.py`) with structured JSON parsing & fallback heuristics
  - [x] `tests/test_extraction.py` passed (14 total test cases passing)
- [x] **Phase 4: Checkers & Scoring Engines**
  - [x] NLI verifier (`src/nli_checker.py`) with GPU-accelerated DeBERTa-v3 MNLI inference
  - [x] LLM Judge verifier (`src/llm_checker.py`) with cached 3-way verdicts
  - [x] Semantic Similarity verifier (`src/semantic_checker.py`) with Sentence-Transformers embeddings
  - [x] Scoring aggregation module (`src/scoring.py`) with strict & soft decision rules
  - [x] Evaluation engine (`src/evaluation.py`) computing Accuracy, Macro-F1, Hallucination Precision/Recall, and per-setting breakdown
  - [x] `tests/test_checkers.py` & `tests/test_evaluation.py` passed (20 total test cases passing)
- [x] **Phase 5: Core Representation Experiments (RQ1: Exp 1–3)**
  - [x] `exp1_sentence_nli.py` built with rich logging and metrics export
  - [x] `exp2_claim_nli.py` built with atomic claim decomposition & NLI verification
  - [x] `exp3_triplet_nli.py` built with structured knowledge triplet extraction & NLI checking
  - [x] `run_rq1_comparison.py` built for consolidated cross-granularity analysis
- [x] **Phase 6: Verifier Ablations & Robustness (RQ2 & RQ3: Exp 4–6)**
  - [x] `exp4_triplet_llm.py` built for LLM Judge verification ablation (RQ2)
  - [x] `exp5_triplet_similarity.py` built for non-NLI semantic similarity baseline (RQ2)
  - [x] `exp6_robustness.py` built for clean vs noisy context robustness study (RQ3)
  - [x] `run_rq2_comparison.py` built for verifier comparison analysis
- [x] **Phase 7: Error Analysis, Visualizations & Final Deliverables**
  - [x] Qualitative error taxonomy engine (`src/error_analysis.py`)
  - [x] Publication-ready visualizations in `results/plots/` (`rq1_granularity_comparison.png`, `rq2_checker_comparison.png`, `confusion_matrices_grid.png`)
  - [x] Comprehensive final research report in [report/FINAL_PROJECT_REPORT.md](file:///c:/Codes/refchecker-mini/report/FINAL_PROJECT_REPORT.md)
  - [x] Slide-by-slide executive briefing deck in [report/PRESENTATION_DECK.md](file:///c:/Codes/refchecker-mini/report/PRESENTATION_DECK.md)
  - [x] Full reproduction experiment logs in [results/experiment_log.md](file:///c:/Codes/refchecker-mini/results/experiment_log.md)

## 5. Data Models
- **`BenchmarkSample`**: `id`, `question`, `context`, `reference`, `setting`, `generator`, `response`, `ground_truth_label`
- **`ExtractedSentence`**: `sample_id`, `sentence_index`, `text`
- **`ExtractedClaim`**: `sample_id`, `claim_index`, `claim_text`, `source_sentence`
- **`ClaimTriplet`**: `sample_id`, `triplet_index`, `subject`, `relation`, `object`, `verbalized_claim`
- **`VerificationUnitResult`**: `unit_id`, `sample_id`, `unit_type`, `unit_text`, `verifier`, `label`, `probabilities`, `metadata`
- **`ResponseEvaluationResult`**: `sample_id`, `setting`, `verifier`, `granularity`, `unit_results`, `response_label`, `is_hallucinated`, `contradiction_count`, `neutral_count`, `entailment_count`, `score`

## 6. API Contracts & External Interfaces
- **Gemini API:**
  - Input: System prompt + response text / query
  - Output: JSON structured output with schema validation
  - Caching: SHA-256 keyed prompt hash in `data/generated/llm_cache.jsonl`
- **HuggingFace NLI Pipeline:**
  - Input: `(Premise=Reference/Context, Hypothesis=Verbalized Unit)`
  - Output: Softmax logits mapped to `{"Entailment": float, "Neutral": float, "Contradiction": float}`

## 7. Technical Debt & Risks
- **Free API Quota:** Gemini API free tier rate limits (15 RPM / 1500 RPD) managed via aggressive local disk caching.
- **Extraction Failures:** Malformed JSON from LLM extraction handled gracefully with fallback regex and explicit logging.
