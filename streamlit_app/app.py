"""
RBI & SEBI Regulatory — Streamlit frontend.

Single page with tabs for the full pipeline.
Atomic Claims live on a separate page under pages/.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATASET_DIR, OLLAMA_HOST, TOP_K_DENSE, TOP_K_RERANKED, TOP_K_SPARSE  # noqa: E402
from main import inspect_retrieval, run_evaluation, run_ingest, run_query  # noqa: E402
from streamlit_app.pipeline_cache import (  # noqa: E402
    build_metadata_filter,
    clear_pipeline_cache,
    get_cached_pipeline,
    get_system_status,
    scan_documents,
)
from streamlit_app.styles import (  # noqa: E402
    answer_card,
    inject_css,
    render_hero,
    render_sidebar_brand,
    status_chip,
)

st.set_page_config(
    page_title="RBI & SEBI Regulatory",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
render_sidebar_brand("Temporal · Retrieval · Generation")
st.sidebar.markdown("---")
if st.sidebar.button("Reload indexes", help="Clear cached models after ingest"):
    clear_pipeline_cache()
    st.sidebar.success("Pipeline cache cleared. Reloads on next use.")


def _escape(text: str) -> str:
    return html.escape(str(text) if text is not None else "")


def _chunk_rows(hits: list) -> list:
    rows = []
    for h in hits:
        meta = h.get("metadata") or {}
        score = h.get("rerank_score", h.get("score", 0.0))
        rows.append(
            {
                "Doc": meta.get("doc_title", ""),
                "Section": meta.get("section", ""),
                "Year": meta.get("year", ""),
                "Issuer": meta.get("issuer", ""),
                "Score": round(float(score), 4) if score is not None else None,
                "Preview": (h.get("text") or "")[:180],
            }
        )
    return rows


def tab_overview() -> None:
    st.subheader("System architecture")
    st.caption(
        "Indian financial regulations (RBI Master Directions / Circulars and SEBI guidelines) "
        "processed through a temporal layer and a multi-stage retrieval layer."
    )

    st.markdown(
        """
<div class="layer-row">
  <div class="layer-card featured">
    <div class="layer-tag">Core layer</div>
    <h4>Temporal Layer</h4>
    <p>Publication year, issuer era, and document vintage preserved in breadcrumbs and metadata
    so answers respect when a regulation applied.</p>
  </div>
  <div class="layer-card featured">
    <div class="layer-tag">Core layer</div>
    <h4>Retrieval Layer</h4>
    <p>Dual-channel search — dense semantic vectors (Chroma) + sparse BM25 — fused with RRF
    and refined by a cross-encoder reranker.</p>
  </div>
  <div class="layer-card">
    <div class="layer-tag">Output</div>
    <h4>Grounded Generation</h4>
    <p>Qwen 2.5 synthesises answers with section citations and discrete atomic claims
    for SLM verification.</p>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### Pipeline stages")
    st.markdown(
        """
<div class="module-grid">
  <div class="module-card"><strong>1. Ingestion</strong><span>PDF scan &amp; PyMuPDF parse</span></div>
  <div class="module-card"><strong>2. Chunking</strong><span>Structure-aware breadcrumbs</span></div>
  <div class="module-card"><strong>3. Temporal index</strong><span>Year · issuer · category metadata</span></div>
  <div class="module-card"><strong>4. Retrieval layer</strong><span>Hybrid RRF + cross-encoder</span></div>
  <div class="module-card"><strong>5. Generation</strong><span>Qwen 2.5 via Ollama</span></div>
  <div class="module-card"><strong>6. Evaluation</strong><span>Hit Rate · MRR · Precision</span></div>
</div>
        """,
        unsafe_allow_html=True,
    )

    status = get_system_status()
    st.markdown("##### Desk status")
    chips = [
        status_chip(f"PDFs: {status['pdf_count']}", ok=status["pdf_count"] > 0),
        status_chip(
            f"Chroma chunks: {status['chroma_count']}",
            ok=status["chroma_count"] > 0,
        ),
        status_chip("BM25 index", ok=status["bm25_ok"]),
        status_chip(f"Ollama: {status['ollama_host']}", ok=None),
    ]
    st.markdown(" ".join(chips), unsafe_allow_html=True)

    with st.expander("Paths & models", expanded=False):
        st.code(
            f"Dataset: {status['dataset_dir']}\n"
            f"Chroma:  {status['persist_directory']}\n"
            f"BM25:    {status['sparse_index_path']}\n"
            f"Model:   {status['ollama_model']}",
            language="text",
        )

    if status["pdf_count"] == 0:
        st.warning(
            f"No PDFs found under `{DATASET_DIR}`. "
            "Place RBI/SEBI guideline PDFs there, then use the **Ingestion** tab."
        )


def tab_ingestion() -> None:
    st.subheader("Ingestion & indexing")
    st.caption(
        "Scan → parse → structure-aware chunk → temporal metadata → Chroma + BM25 dual index"
    )

    docs = scan_documents()
    if not docs:
        st.error(
            f"No PDF files in `{DATASET_DIR}`. Add regulatory PDFs and refresh."
        )
        return

    st.markdown(f"**{len(docs)}** PDF document(s) discovered.")
    preview = [
        {
            "Filename": d.get("filename"),
            "Issuer": d.get("issuer"),
            "Category": d.get("category"),
            "Year": d.get("year"),
            "Title": d.get("doc_title"),
        }
        for d in docs[:50]
    ]
    st.dataframe(preview, use_container_width=True, hide_index=True)

    max_docs = st.slider(
        "Max documents to ingest (0 = all)",
        min_value=0,
        max_value=max(len(docs), 1),
        value=min(10, len(docs)),
    )
    limit = None if max_docs == 0 else max_docs

    if st.button("Run ingestion", type="primary", key="btn_ingest"):
        progress = st.empty()
        log_lines: list[str] = []

        def _cb(msg: str) -> None:
            log_lines.append(msg)
            progress.code("\n".join(log_lines[-12:]), language="text")

        with st.spinner("Ingesting regulatory PDFs…"):
            result = run_ingest(max_docs=limit, progress_callback=_cb)

        if result.get("success"):
            clear_pipeline_cache()
            st.success(
                f"Indexed **{result['num_docs']}** docs → "
                f"**{result['num_chunks']}** chunks. Pipeline cache cleared."
            )
        else:
            st.error(result.get("message", "Ingestion failed."))


def tab_retrieval() -> None:
    st.subheader("Retrieval layer")
    st.caption(
        "Dense (Chroma) · Sparse (BM25) · RRF fusion · Cross-encoder rerank — inspection only, no LLM"
    )

    q = st.text_input(
        "Inspection query",
        placeholder="e.g. Cash Reserve Ratio maintenance requirements",
        key="retrieval_q",
    )
    col_a, col_b = st.columns([1, 1])
    with col_a:
        issuer = st.selectbox("Issuer filter", ["All", "RBI", "SEBI"], key="retrieval_issuer")
    with col_b:
        year_hint = st.text_input(
            "Temporal filter (year, optional)",
            placeholder="e.g. 2024",
            key="retrieval_year",
            help="Filters dense search by publication year metadata when set.",
        )

    if st.button("Inspect retrieval layer", type="primary", key="btn_retrieve") and q.strip():
        with st.spinner("Running retrieval layer…"):
            try:
                vector_store, bm25, hybrid, _gen = get_cached_pipeline()
                filt = build_metadata_filter(issuer=issuer, year=year_hint)
                hits = inspect_retrieval(
                    q.strip(),
                    vector_store,
                    bm25,
                    hybrid,
                    filter_dict=filt,
                )
            except Exception as e:
                st.error(f"Retrieval failed: {e}")
                return

        st.markdown("##### Retrieval layer channels")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Dense channel**")
            st.dataframe(
                _chunk_rows(hits["dense"][:TOP_K_DENSE]),
                use_container_width=True,
                hide_index=True,
            )
        with c2:
            st.markdown("**Sparse BM25 channel**")
            st.dataframe(
                _chunk_rows(hits["sparse"][:TOP_K_SPARSE]),
                use_container_width=True,
                hide_index=True,
            )
        with c3:
            st.markdown("**Reranked output**")
            st.dataframe(
                _chunk_rows(hits["reranked"][:TOP_K_RERANKED]),
                use_container_width=True,
                hide_index=True,
            )


def tab_query_generation() -> None:
    st.subheader("Query & grounded generation")
    st.caption(
        "Temporal metadata filters → retrieval layer → Qwen 2.5 answer · citations · discrete claims"
    )

    q = st.text_area(
        "Regulatory question",
        height=100,
        placeholder="What are the KYC risk categorization requirements for banks?",
        key="gen_q",
    )
    col_a, col_b = st.columns(2)
    with col_a:
        issuer = st.selectbox("Issuer filter", ["All", "RBI", "SEBI"], key="gen_issuer")
    with col_b:
        year_hint = st.text_input(
            "Temporal filter (year, optional)",
            placeholder="e.g. 2023",
            key="gen_year",
        )

    if st.button("Ask", type="primary", key="btn_ask") and q.strip():
        with st.spinner("Retrieval layer + generation…"):
            try:
                _vs, _bm25, hybrid, generator = get_cached_pipeline()
                filt = build_metadata_filter(issuer=issuer, year=year_hint)
                result = run_query(q.strip(), hybrid, generator, filter_dict=filt)
            except Exception as e:
                st.error(f"Query failed: {e}")
                return

        st.session_state["last_rag_result"] = result

        st.markdown(
            answer_card(
                "Grounded regulatory answer",
                _escape(result["answer"]).replace("\n", "<br>"),
                muted=f"Engine: {_escape(result.get('model_used', ''))}",
            ),
            unsafe_allow_html=True,
        )

        st.markdown("#### Source citations")
        if result.get("citations"):
            for c in result["citations"]:
                st.markdown(f"- `{c}`")
        else:
            st.caption("No citations extracted.")

        st.markdown("#### Discrete claims (preview)")
        st.caption("Full claim review → **Atomic Claims** page in the sidebar.")
        for i, claim in enumerate(result.get("claims") or [], start=1):
            st.markdown(f"**[{i}]** {claim}")

        with st.expander("Retrieved context (retrieval layer)", expanded=False):
            st.dataframe(
                _chunk_rows(result.get("chunks") or []),
                use_container_width=True,
                hide_index=True,
            )

    elif "last_rag_result" in st.session_state:
        prev = st.session_state["last_rag_result"]
        st.info(f"Last query: {prev.get('query', '')}")
        st.markdown(
            answer_card(
                "Grounded regulatory answer",
                _escape(prev.get("answer", "")).replace("\n", "<br>"),
                muted=f"Engine: {_escape(prev.get('model_used', ''))}",
            ),
            unsafe_allow_html=True,
        )


def tab_evaluation() -> None:
    st.subheader("Benchmark evaluation")
    st.caption("Hit Rate · MRR · Precision@K · citation coverage · avg claims")

    if st.button("Run evaluation", type="primary", key="btn_eval"):
        with st.spinner("Running evaluation suite (retrieval + generation per query)…"):
            try:
                _vs, _bm25, hybrid, generator = get_cached_pipeline()
                summary = run_evaluation(hybrid, generator, save_report=True)
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                return

        if not summary:
            st.warning("No evaluation results returned. Check data/eval_dataset.json.")
            return

        st.session_state["last_eval_summary"] = summary

    summary = st.session_state.get("last_eval_summary")
    if not summary:
        st.caption("Click **Run evaluation** to score against the benchmark dataset.")
        return

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Hit Rate", f"{summary.get('hit_rate', 0) * 100:.1f}%")
    m2.metric("MRR", f"{summary.get('mrr', 0):.4f}")
    m3.metric("Precision@K", f"{summary.get('mean_precision_at_k', 0):.4f}")
    m4.metric("Citation coverage", f"{summary.get('citation_coverage_rate', 0) * 100:.1f}%")
    m5.metric("Avg claims/query", f"{summary.get('avg_claims_per_query', 0)}")

    details = summary.get("detailed_results") or []
    if details:
        st.dataframe(details, use_container_width=True, hide_index=True)
    st.caption("Full report also written to `evaluation_report.json`.")


# —— Page shell ——
render_hero()

tabs = st.tabs(
    [
        "Overview",
        "Ingestion",
        "Retrieval Layer",
        "Query & Generation",
        "Evaluation",
    ]
)
with tabs[0]:
    tab_overview()
with tabs[1]:
    tab_ingestion()
with tabs[2]:
    tab_retrieval()
with tabs[3]:
    tab_query_generation()
with tabs[4]:
    tab_evaluation()
