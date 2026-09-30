# RegGuard: Project Status & Workflow Tracker

This document tracks the migration and completion status of the RegGuard modular architecture (M0-M10).

## 📊 Overview
| Module | Name | Status | Key Deliverables |
|--------|------|--------|------------------|
| **M0** | `common/` | ✅ Complete | Pydantic Schemas, JSONL I/O, Config YAML, Dates, LLM Client (w/ Caching), Fixtures |
| **M1** | `ingest/` | ✅ Complete | PyMuPDF parsing, Regex Legal Chunker (strict boundaries), Date Metadata extraction, Version Graph, Scrapers |
| **M2** | `retrieval/` | ✅ Complete | Hybrid RRF (BM25 + Dense Chroma), Cross-Encoder Rerank, Strict Temporal Graph Filter, Query Rewrite |
| **M3** | `generation/`| ✅ Complete | vLLM Generation, Prompt v1, Citation extraction, Logprob capturing, No-context baseline extraction |
| **M4** | `atoms/` | ✅ Complete | Two-pass extraction (Regex fast-pass + LLM), 6 atom types, span alignment, dedup, "Not found" short-circuit |
| **M5** | `verify/` | ✅ Complete | V1 Deterministic Normalizers + Literal Verifier, V2 NLI (DeBERTa-v3-large lazy loader), Cascade, NLI Data Perturbers, NLI Fine-tune Train Script |
| **M6** | `signals/` | ✅ Complete | `s_div`, `s_ret`, `s_ver`, `s_nli`, `s_ent`, `s_mech` signals, `compute_all_signals()` dispatcher, LightGBM Aggregator (`dataset.py`, `train.py`, `calibrate_probs.py`, `model.py`), `configs/signals.yaml` |
| **M7** | `guarantee/` | ❌ Not Started | Learn-then-Test statistical bounds, Mondrian Stratification, Drift-aware Calibration |
| **M8** | `judge/` | ❌ Not Started | Strong LLM Judge (V3) for Uncertain Atoms, Decision Router |
| **M9** | `app/` | ❌ Not Started | FastAPI + Streamlit interface |
| **M10**| `bench/` | ❌ Not Started | Evaluation datasets, metrics computation (AURC, Recall, F1) |

---

## 🛠️ Module Details & Architectural Decisions

### M0: Common Core
- Completely typed using `pydantic.BaseModel` to guarantee nested JSON serialization without data loss.
- Implemented Indian date format parsers (e.g., `FY 2023-24`, `12/03/2024`).
- Built `LLMClient` backed by Tenacity for automatic retries, which hashes prompts (model + temp + user) to a local `.llm_cache` to drastically reduce API costs during reruns.

### M1: Ingestion & Version Graph
- Moved away from generic token-overlap chunking. Chunks are strictly boundary-matched using Regex (`Chapter I`, `Regulation 52`) to ensure isolated context.
- Text cleaning strips header/footers, repairs hyphenation, and forces all currency values to `₹`.
- Implemented `VersionGraph`. The graph dynamically links amendments (`amends`, `supersedes`) to construct historical snapshots of the law.

### M2: Retrieval
- Custom `BM25` tokenization expressly preserves parenthesis references (e.g. `52(4)`).
- Fused dense vectors (BGE-M3 on ChromaDB) with sparse lexical tokens via a purely mathematical `Reciprocal Rank Fusion (RRF)` pipeline.
- Applied `Temporal Filter`: Drops chunks that are historically superseded or not-yet-effective compared to the user's `query_date`.

### M3: Generation
- Instructs the LLM via strict prompts to yield bracketed citations `[chunk_id]`.
- Enforces returning logprobs for downstream token entropy signals (M6).
- Automatically executes a secondary "No-context" call to extract a baseline hallucination answer (needed for M6 divergence testing).

### M4: Atoms
- Two-pass extraction: fast regex pass (RATE, THRESHOLD, SECTION, DATE) followed by LLM pass (ENTITY, APPLICABILITY + richer claims).
- Merges and deduplicates by (type, text); regex fills gaps LLM misses; span-aligns every atom back to character offsets in `answer_text`.
- Short-circuits on "not found in evidence" answers (returns empty list).

### M5: Verification Cascade
- **`normalizers.py`**: Canonical normalization for monetary amounts (lakh/crore/thousand), percentages (% / per cent / p.a.), Indian date formats (dd/mm/yyyy, "12 March 2024", FY 2023-24), and section references.
- **`v1_deterministic.py`**: Scans retrieved chunks for MATCH/MISMATCH/NOT_FOUND/NA. Cited chunk is checked first; falls back across all chunks if not found. ENTITY/APPLICABILITY atoms return NA (V2 only).
- **`v2_nli.py`**: DeBERTa-v3-large zero-shot NLI wrapped in a lazy singleton. Falls back gracefully to neutral (0.5) if the model checkpoint is unavailable (pre-training phase).
- **`cascade.py`**: Routes each atom: V1 MATCH/MISMATCH → skip V2; V1 NOT_FOUND or atom is ENTITY/APPLICABILITY → run V2. Every atom guaranteed a `VerifiedAtom` output (no silent drops).
- **`nli_data/perturb.py`**: Type-specific perturbation functions (rate ±%, amount ±%, section ±N, date ±1–3 years) for generating CONTRADICTION training pairs.
- **`nli_data/make_pairs.py`**: Builds entailment / contradiction / neutral JSONL pairs for NLI fine-tuning.
- **`nli_train/train.py`**: HuggingFace Trainer script for fine-tuning DeBERTa-v3-large (bf16, early stopping on macro-F1).
- **`configs/verify.yaml`**: Model path, training hyperparameters, data paths.
- **Tests**: 61 tests covering 30 normalization cases, V1 MATCH/MISMATCH/NOT_FOUND/NA for all atom types, multi-chunk fallback, cascade metadata flags. All pass ✅.

### M6: Signals + Aggregator
- **`signals/s_ver.py`**: Encodes V1 status as scalar (MATCH=0, NOT_FOUND/NA=0.5, MISMATCH=1) plus one-hot.
- **`signals/s_nli.py`**: `1 - p(entail)` from V2; defaults to 0.5 + `nli_skipped=1` flag when NLI was skipped.
- **`signals/s_ent.py`**: Mean/max token entropy from logprobs over atom's character span (normalized to [0,1] with `max_clip=5.0`).
- **`signals/s_ret.py`**: Retrieval reranker score for the cited chunk (min-max normalized); rank-based fallback when no reranker scores.
- **`signals/s_div.py`**: Context divergence = `1 - cosine_sim(claim, no_context_answer)` via BGE-small; Strategy B uses KL over token logprob spans.
- **`signals/s_mech.py`**: Lookback ratio from Qwen attention weights (`1 - attn_to_prompt/attn_total`); disk-cached; falls back to 0.5 when no GPU.
- **`signals/__init__.py`**: `compute_all_signals()` — single entry point returning flat feature dict for all 6 signals.
- **`aggregate/dataset.py`**: Builds feature table (Parquet) from labeled JSONL atoms with all signals + atom-type/regulator one-hots.
- **`aggregate/train.py`**: LogReg baseline + LightGBM (300 trees, lr=0.05, max_depth=4); cross-validates by question_family; saves `aggregator_v1.pkl` + ablation table.
- **`aggregate/calibrate_probs.py`**: Isotonic regression (≥1000 samples) or Platt scaling calibration on val split.
- **`aggregate/model.py`**: `score()` inference function — loads model, builds feature vector from `compute_all_signals()`, applies calibration, returns `List[ScoredAtom]`; heuristic weighted-average fallback when no model trained.
- **`configs/signals.yaml`**: All signal hyperparameters, model paths, LightGBM config.
- **Tests**: 74 tests (41 signals + 33 aggregator). All pass ✅.

---

## 🚀 Next Priority
- Start building **M7 (Guarantee)**: Learn-then-Test (LTT) p-value testing for certified thresholds, Mondrian stratification by regulator×atom_type, drift-aware calibration (sliding window, recency-weighted, triggered).

