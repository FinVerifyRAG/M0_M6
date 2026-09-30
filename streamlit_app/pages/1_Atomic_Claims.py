"""
RegGuard — Atomic Extraction workspace.

Consumes live RAG session results (`st.session_state["last_rag_result"]`).
Does not invent financial facts or verification outcomes.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import run_query  # noqa: E402
from streamlit_app.atom_extraction import (  # noqa: E402
    ATOM_TYPES,
    build_atoms_payload,
    summarize_atoms,
)
from streamlit_app.pipeline_cache import (  # noqa: E402
    build_metadata_filter,
    get_cached_pipeline,
)
from streamlit_app.styles import inject_css, render_sidebar_brand  # noqa: E402

st.set_page_config(
    page_title="RegGuard · Atomic Extraction",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
render_sidebar_brand("RegGuard · Atomic Extraction")

# —— RegGuard Atomic Extraction CSS (page-local) ——
st.markdown(
    """
<style>
  .rg-header {
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 0.75rem;
    background: #0B1F33; color: #F6F3EC;
    border-radius: 10px; padding: 1rem 1.25rem;
    border: 1px solid rgba(196,163,90,0.35);
    margin-bottom: 1.1rem;
  }
  .rg-brand { display: flex; align-items: baseline; gap: 0.75rem; flex-wrap: wrap; }
  .rg-logo {
    font-family: "Source Serif 4", Georgia, serif;
    font-weight: 700; font-size: 1.35rem; color: #F6F3EC; margin: 0;
  }
  .rg-logo span { color: #C4A35A; }
  .rg-module {
    font-size: 0.85rem; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; color: #C4A35A;
  }
  .rg-status {
    display: inline-flex; align-items: center; gap: 0.4rem;
    font-size: 0.78rem; font-weight: 600;
    background: rgba(31,107,74,0.25); color: #B8E0C8;
    border: 1px solid rgba(31,107,74,0.45);
    padding: 0.3rem 0.65rem; border-radius: 999px;
  }
  .rg-status .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #3CB371; box-shadow: 0 0 0 2px rgba(60,179,113,0.25);
  }
  .rg-stages {
    display: flex; flex-wrap: wrap; gap: 0.35rem; align-items: center;
    margin-top: 0.55rem; width: 100%;
  }
  .rg-stage {
    font-size: 0.7rem; font-weight: 650; letter-spacing: 0.04em;
    text-transform: uppercase; padding: 0.28rem 0.55rem; border-radius: 4px;
    background: rgba(255,255,255,0.08); color: rgba(246,243,236,0.65);
    border: 1px solid rgba(255,255,255,0.1);
  }
  .rg-stage.done { color: #B8E0C8; border-color: rgba(31,107,74,0.4); }
  .rg-stage.active {
    background: #C4A35A; color: #0B1F33; border-color: #C4A35A;
  }
  .rg-stage.todo { opacity: 0.7; }
  .rg-arrow { color: rgba(246,243,236,0.4); font-size: 0.75rem; }

  .rg-card {
    background: #fff; border: 1px solid #E2DCCE; border-radius: 10px;
    padding: 1rem 1.15rem; margin-bottom: 0.9rem;
    box-shadow: 0 1px 4px rgba(11,31,51,0.04);
  }
  .rg-card-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #5A6A7A; margin: 0 0 0.45rem 0;
  }
  .rg-query {
    display: flex; gap: 0.75rem; align-items: flex-start;
  }
  .rg-q-icon {
    flex-shrink: 0; width: 36px; height: 36px; border-radius: 8px;
    background: #EEF3F8; color: #16324F; display: flex;
    align-items: center; justify-content: center; font-weight: 700;
    border: 1px solid #D5DEE8; font-size: 1rem;
  }
  .rg-query p {
    margin: 0; color: #0E1A28 !important; font-size: 1.02rem;
    line-height: 1.5; font-weight: 500;
  }
  .rg-badge {
    display: inline-block; font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.04em; text-transform: uppercase;
    padding: 0.2rem 0.5rem; border-radius: 4px; margin-right: 0.35rem;
  }
  .rg-badge-rag { background: #EEF3F8; color: #16324F; border: 1px solid #C5D2E0; }
  .rg-badge-ok { background: #eef6f1; color: #1F6B4A; border: 1px solid #b7d4c4; }
  .rg-badge-warn { background: #f7f1e6; color: #9A6B2F; border: 1px solid #e0cda8; }
  .rg-answer-body {
    color: #0E1A28 !important; line-height: 1.6; margin: 0.55rem 0 0 0;
    white-space: pre-wrap;
  }
  .rg-flow {
    display: flex; align-items: center; justify-content: center;
    flex-wrap: wrap; gap: 0.5rem; padding: 0.85rem 0.5rem 1.1rem;
  }
  .rg-flow-node {
    background: #fff; border: 1px solid #E2DCCE; border-radius: 8px;
    padding: 0.55rem 0.9rem; text-align: center; min-width: 120px;
  }
  .rg-flow-node strong { display: block; color: #0B1F33; font-size: 0.85rem; }
  .rg-flow-node span { color: #5A6A7A; font-size: 0.75rem; }
  .rg-flow-node.hl { border-color: #C4A35A; background: #FBF7EE; }
  .rg-flow-arrow { color: #C4A35A; font-weight: 700; }

  .atom-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
    gap: 0.75rem;
  }
  .atom-card {
    background: #fff; border: 1px solid #E2DCCE; border-radius: 10px;
    padding: 0.9rem 1rem; cursor: default;
    border-top: 3px solid #5A6A7A;
    transition: box-shadow 0.15s, border-color 0.15s;
  }
  .atom-card.selected {
    box-shadow: 0 0 0 2px #C4A35A;
    border-color: #C4A35A;
  }
  .atom-card.type-RATE { border-top-color: #2F5D8A; }
  .atom-card.type-THRESHOLD { border-top-color: #5B6B3A; }
  .atom-card.type-SECTION { border-top-color: #4A3F6B; }
  .atom-card.type-DATE { border-top-color: #6B4A2F; }
  .atom-card.type-ENTITY { border-top-color: #2F6B5D; }
  .atom-card.type-APPLICABILITY { border-top-color: #5A6A7A; }
  .atom-type {
    font-size: 0.68rem; font-weight: 750; letter-spacing: 0.1em;
    color: #5A6A7A; margin-bottom: 0.4rem;
  }
  .atom-value {
    font-family: "Source Serif 4", Georgia, serif;
    font-size: 1.2rem; font-weight: 700; color: #0B1F33;
    margin: 0 0 0.45rem 0; line-height: 1.25; word-break: break-word;
  }
  .atom-text {
    font-size: 0.82rem; color: #4A5A6A; line-height: 1.4;
    margin: 0 0 0.65rem 0; font-style: italic;
  }
  .atom-foot {
    display: flex; justify-content: space-between; align-items: center;
    gap: 0.4rem; flex-wrap: wrap;
  }
  .atom-pending {
    font-size: 0.72rem; font-weight: 650; color: #9A6B2F;
  }
  .atom-pending::before {
    content: "●"; margin-right: 0.3rem; color: #C4A35A;
  }
  .atom-id { font-size: 0.72rem; font-weight: 600; color: #5A6A7A; }

  .hl-answer {
    line-height: 1.75; color: #0E1A28; font-size: 0.98rem;
  }
  .atom-mark {
    background: #EEF3F8; border-bottom: 2px solid #2F5D8A;
    border-radius: 3px; padding: 0.05rem 0.2rem; cursor: pointer;
  }
  .atom-mark.RATE { background: #E8F0F7; border-bottom-color: #2F5D8A; }
  .atom-mark.THRESHOLD { background: #F0F3E8; border-bottom-color: #5B6B3A; }
  .atom-mark.SECTION { background: #F0ECF6; border-bottom-color: #4A3F6B; }
  .atom-mark.DATE { background: #F6EEE8; border-bottom-color: #6B4A2F; }
  .atom-mark.ENTITY { background: #E8F4F1; border-bottom-color: #2F6B5D; }
  .atom-mark.APPLICABILITY { background: #EEF0F2; border-bottom-color: #5A6A7A; }
  .atom-mark.active {
    background: #C4A35A !important; color: #0B1F33;
    border-bottom-color: #0B1F33;
  }

  .detail-panel {
    background: #fff; border: 1px solid #E2DCCE; border-radius: 10px;
    padding: 1.1rem 1.15rem; position: sticky; top: 1rem;
    box-shadow: 0 2px 10px rgba(11,31,51,0.06);
  }
  .detail-panel h3 {
    margin: 0 0 0.85rem 0; color: #0B1F33 !important;
    font-family: "Source Serif 4", Georgia, serif; font-size: 1.25rem;
  }
  .detail-row { margin-bottom: 0.7rem; }
  .detail-row .k {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #5A6A7A; margin-bottom: 0.15rem;
  }
  .detail-row .v { color: #0E1A28; font-size: 0.92rem; line-height: 1.4; }

  .summary-box {
    background: #fff; border: 1px solid #E2DCCE; border-radius: 10px;
    padding: 1.15rem 1.25rem; margin-top: 0.5rem;
  }
  .summary-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.4rem 1rem;
    margin: 0.75rem 0 1rem;
  }
  .summary-grid div { font-size: 0.85rem; color: #0E1A28; }
  .summary-grid span { color: #5A6A7A; font-weight: 600; min-width: 7rem; display: inline-block; }
  .extract-done { color: #1F6B4A; font-weight: 650; font-size: 0.9rem; }
</style>
    """,
    unsafe_allow_html=True,
)


def _esc(text) -> str:
    return html.escape(str(text) if text is not None else "")


def _highlighted_answer(answer: str, atoms: list, selected_id: str | None) -> str:
    """Wrap atom spans in the answer. Non-overlapping, left-to-right."""
    if not answer:
        return "<p class='hl-answer'><em>No generated answer available.</em></p>"

    # Only atoms with valid spans
    spannable = [
        a
        for a in atoms
        if a.get("start") is not None
        and a.get("end") is not None
        and 0 <= a["start"] < a["end"] <= len(answer)
    ]
    spannable.sort(key=lambda a: (a["start"], -(a["end"] - a["start"])))

    # Greedy non-overlapping
    chosen = []
    cursor = 0
    for a in spannable:
        if a["start"] >= cursor:
            chosen.append(a)
            cursor = a["end"]

    parts = []
    pos = 0
    for a in chosen:
        s, e = a["start"], a["end"]
        if s > pos:
            parts.append(_esc(answer[pos:s]))
        active = " active" if a["atom_id"] == selected_id else ""
        parts.append(
            f'<mark class="atom-mark { _esc(a["type"]) }{active}" '
            f'title="{_esc(a["atom_id"])} · {_esc(a["type"])}">{_esc(answer[s:e])}</mark>'
        )
        pos = e
    if pos < len(answer):
        parts.append(_esc(answer[pos:]))

    return f'<div class="hl-answer">{"".join(parts)}</div>'


def _atom_card_html(atom: dict, selected: bool) -> str:
    sel = " selected" if selected else ""
    return f"""
<div class="atom-card type-{_esc(atom.get('type',''))}{sel}">
  <div class="atom-type">{_esc(atom.get('type',''))}</div>
  <p class="atom-value">{_esc(atom.get('value',''))}</p>
  <p class="atom-text">"{_esc(atom.get('text',''))}"</p>
  <div class="atom-foot">
    <span class="atom-pending">Pending Verification</span>
    <span class="atom-id">Atom {_esc(atom.get('atom_id',''))}</span>
  </div>
</div>
"""


def _detail_html(atom: dict | None) -> str:
    if not atom:
        return """
<div class="detail-panel">
  <h3>Atom detail</h3>
  <p style="color:#5A6A7A;margin:0;font-size:0.9rem;">
    Select an atom card to inspect its extraction metadata.
  </p>
</div>
"""
    conf = atom.get("confidence")
    conf_display = "Not provided" if conf is None else str(conf)
    status = atom.get("status") or "PENDING"
    status_label = "PENDING VERIFICATION" if status == "PENDING" else status
    return f"""
<div class="detail-panel">
  <h3>ATOM {_esc(atom.get('atom_id',''))}</h3>
  <div class="detail-row"><div class="k">Atom Type</div><div class="v">{_esc(atom.get('type',''))}</div></div>
  <div class="detail-row"><div class="k">Extracted Value</div><div class="v"><strong>{_esc(atom.get('value',''))}</strong></div></div>
  <div class="detail-row"><div class="k">Original Text</div><div class="v">{_esc(atom.get('text',''))}</div></div>
  <div class="detail-row"><div class="k">Source Sentence</div><div class="v">{_esc(atom.get('source_sentence') or atom.get('text') or '—')}</div></div>
  <div class="detail-row"><div class="k">Extraction Confidence</div><div class="v">{_esc(conf_display)}</div></div>
  <div class="detail-row"><div class="k">Verification Status</div><div class="v"><span class="rg-badge rg-badge-warn">{_esc(status_label)}</span></div></div>
  <div class="detail-row"><div class="k">Next Module</div><div class="v">M4 Verification Cascade</div></div>
</div>
"""


# —— Optional: fetch from pipeline if session empty ——
with st.sidebar.expander("Load from pipeline", expanded=False):
    st.caption("Runs the live regulatory pipeline if no session result exists.")
    q_side = st.text_area("Question", key="rg_atom_q", height=70)
    issuer = st.selectbox("Issuer", ["All", "RBI", "SEBI"], key="rg_atom_issuer")
    if st.button("Run & extract", key="rg_atom_run") and q_side.strip():
        with st.spinner("Connecting to pipeline…"):
            try:
                _vs, _bm25, hybrid, generator = get_cached_pipeline()
                live = run_query(
                    q_side.strip(),
                    hybrid,
                    generator,
                    filter_dict=build_metadata_filter(issuer=issuer),
                )
                st.session_state["last_rag_result"] = live
                st.session_state.pop("selected_atom_id", None)
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline error: {e}")

rag = st.session_state.get("last_rag_result")
loading = False

if not rag:
    st.markdown(
        """
<div class="rg-header">
  <div>
    <div class="rg-brand">
      <p class="rg-logo">Reg<span>Guard</span></p>
      <span class="rg-module">Atomic Extraction</span>
    </div>
    <div class="rg-stages">
      <span class="rg-stage todo">RAG</span><span class="rg-arrow">→</span>
      <span class="rg-stage todo">Generation</span><span class="rg-arrow">→</span>
      <span class="rg-stage active">Atom Extraction</span><span class="rg-arrow">→</span>
      <span class="rg-stage todo">Verification</span>
    </div>
  </div>
  <span class="rg-status"><span class="dot"></span> Waiting for pipeline data</span>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "No generated answer in session yet. Run a query on the main desk "
        "(**Query & Generation**), or use **Load from pipeline** in the sidebar."
    )
    st.stop()

# Build atom payload from live RAG data only
try:
    payload = build_atoms_payload(rag)
except Exception as e:
    st.error(f"Atomic extraction failed: {e}")
    st.stop()

query = payload.get("query") or ""
answer = payload.get("answer") or ""
atoms = payload.get("atoms") or []
has_context = bool(payload.get("has_context"))
model_used = payload.get("model_used") or ""

# Header
st.markdown(
    f"""
<div class="rg-header">
  <div>
    <div class="rg-brand">
      <p class="rg-logo">Reg<span>Guard</span></p>
      <span class="rg-module">Atomic Extraction</span>
    </div>
    <div class="rg-stages">
      <span class="rg-stage done">RAG</span><span class="rg-arrow">→</span>
      <span class="rg-stage done">Generation</span><span class="rg-arrow">→</span>
      <span class="rg-stage active">Atom Extraction</span><span class="rg-arrow">→</span>
      <span class="rg-stage todo">Verification</span>
    </div>
  </div>
  <span class="rg-status"><span class="dot"></span> Pipeline Connected</span>
</div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "RegGuard does not blindly verify an entire LLM response. "
    "It first decomposes the response into small factual atoms so every important fact "
    "can be independently checked."
)

# Selection state
atom_ids = [a["atom_id"] for a in atoms]
selected = st.session_state.get("selected_atom_id")
if selected not in atom_ids:
    selected = atom_ids[0] if atom_ids else None
    st.session_state["selected_atom_id"] = selected

selected_atom = next((a for a in atoms if a["atom_id"] == selected), None)

main_col, detail_col = st.columns([1.65, 1], gap="large")

with main_col:
    # Section 1 — User Query
    st.markdown(
        f"""
<div class="rg-card">
  <p class="rg-card-label">User Query</p>
  <div class="rg-query">
    <div class="rg-q-icon">?</div>
    <p>{_esc(query) if query else "<em>No query provided by backend.</em>"}</p>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # Section 2 — Generated Answer
    ctx_badge = (
        '<span class="rg-badge rg-badge-ok">Context available</span>'
        if has_context
        else '<span class="rg-badge rg-badge-warn">No source context</span>'
    )
    st.markdown(
        f"""
<div class="rg-card">
  <p class="rg-card-label">Generated Answer</p>
  <div>
    <span class="rg-badge rg-badge-rag">Generated by RAG</span>
    {ctx_badge}
    <span class="rg-badge rg-badge-rag">{_esc(model_used) if model_used else "Engine unknown"}</span>
  </div>
  <p class="rg-answer-body">{_esc(answer) if answer else "<em>No answer returned by backend.</em>"}</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    # Section 3 — Atomic Extraction flow + cards
    st.markdown(
        f"""
<div class="rg-card">
  <p class="rg-card-label">Atomic Claims</p>
  <h3 style="margin:0 0 0.25rem 0;color:#0B1F33;font-family:'Source Serif 4',Georgia,serif;">
    Atomic Claims
  </h3>
  <p style="margin:0 0 0.5rem 0;color:#5A6A7A;font-size:0.9rem;">
    Breaking the generated answer into independently verifiable facts
  </p>
  <div class="rg-flow">
    <div class="rg-flow-node"><strong>Generated Answer</strong><span>LLM output</span></div>
    <span class="rg-flow-arrow">↓</span>
    <div class="rg-flow-node hl"><strong>Claim Extraction</strong><span>Decomposition</span></div>
    <span class="rg-flow-arrow">↓</span>
    <div class="rg-flow-node"><strong>{len(atoms)} Atomic Claims</strong><span>Ready to verify</span></div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    if not atoms:
        st.warning(
            "No atoms could be extracted from the backend response. "
            "The answer or discrete claims list may be empty."
        )
    else:
        # Selection via radio (accessible) + visual cards
        labels = {
            a["atom_id"]: f"{a['atom_id']} · {a['type']} · {a.get('value','')[:40]}"
            for a in atoms
        }
        pick = st.radio(
            "Select atom",
            options=atom_ids,
            format_func=lambda i: labels.get(i, i),
            index=atom_ids.index(selected) if selected in atom_ids else 0,
            horizontal=True,
            key="atom_picker",
            label_visibility="collapsed",
        )
        if pick != selected:
            st.session_state["selected_atom_id"] = pick
            selected = pick
            selected_atom = next((a for a in atoms if a["atom_id"] == selected), None)
            st.rerun()

        cards_html = '<div class="atom-grid">' + "".join(
            _atom_card_html(a, a["atom_id"] == selected) for a in atoms
        ) + "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)

    # Section 4 — Answer highlighting
    st.markdown(
        """
<div class="rg-card">
  <p class="rg-card-label">Answer Highlighting</p>
  <p style="margin:0 0 0.55rem 0;color:#5A6A7A;font-size:0.85rem;">
    Highlighted spans map to extracted atoms. Select an atom above to emphasize its span.
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="rg-card" style="margin-top:-0.55rem;">{_highlighted_answer(answer, atoms, selected)}</div>',
        unsafe_allow_html=True,
    )

with detail_col:
    # Section 5 — Atom detail panel
    st.markdown(_detail_html(selected_atom), unsafe_allow_html=True)

# Section 6 — Extraction summary
counts = summarize_atoms(atoms)
breakdown_rows = "".join(
    f"<div><span>{t}</span> {counts.get(t, 0)}</div>" for t in ATOM_TYPES
)

st.markdown(
    f"""
<div class="summary-box">
  <p class="rg-card-label">Extraction Summary</p>
  <p style="margin:0;font-size:1.05rem;color:#0B1F33;">
    <strong>Atoms Extracted:</strong> {len(atoms)}
  </p>
  <p style="margin:0.65rem 0 0.25rem;font-size:0.8rem;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:#5A6A7A;">
    Breakdown
  </p>
  <div class="summary-grid">{breakdown_rows}</div>
  <p class="extract-done">{"✓ Extraction Complete" if atoms else "○ No atoms extracted"}</p>
</div>
    """,
    unsafe_allow_html=True,
)

st.write("")
send = st.button("Send Atoms to Verification →", type="primary", disabled=not atoms)

if send:
    # Ready for M4 — no fake verification results
    verification_endpoint = st.session_state.get("verification_api_url")
    outbound = {
        "query": query,
        "answer": answer,
        "atoms": atoms,
        "source": "RegGuard Atomic Extraction",
    }
    st.session_state["atoms_for_verification"] = outbound

    if verification_endpoint:
        try:
            import requests

            resp = requests.post(verification_endpoint, json=outbound, timeout=60)
            if resp.ok:
                st.success("Atoms submitted to verification API.")
                st.json(resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"status": resp.status_code})
            else:
                st.error(f"Verification API returned HTTP {resp.status_code}. Atoms were not accepted.")
        except Exception as e:
            st.error(f"Could not reach verification API: {e}")
            st.info("Atoms are stored in session (`atoms_for_verification`) for M4 when the endpoint is available.")
    else:
        st.info(
            "Atoms packaged for **M4 Verification Cascade**. "
            "No verification API URL is configured yet — nothing was fabricated. "
            "Set `st.session_state['verification_api_url']` when M4 is ready."
        )
        with st.expander("Outbound atom payload (for M4)"):
            st.code(json.dumps(outbound, indent=2), language="json")
