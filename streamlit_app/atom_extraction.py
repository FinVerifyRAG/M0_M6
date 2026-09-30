"""
Convert RAG pipeline output into RegGuard atomic claims.

Does not invent financial facts — only structures text already produced
by the generator (answer + discrete claims).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

ATOM_TYPES = (
    "RATE",
    "THRESHOLD",
    "SECTION",
    "DATE",
    "ENTITY",
    "APPLICABILITY",
)

# Classification cues (applied to existing claim/answer text only)
_TYPE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (
        "RATE",
        re.compile(
            r"\b\d+(\.\d+)?\s*%|\brate\b|\bratio\b|\bCRR\b|\bSLR\b|\bTDS\b|\binterest\b",
            re.I,
        ),
    ),
    (
        "THRESHOLD",
        re.compile(
            r"\bminimum\b|\bmaximum\b|\bthreshold\b|\bnot less than\b|\bnot more than\b|"
            r"\bat least\b|\bceiling\b|\bfloor\b|\blimit\b|\babove\b|\bbelow\b",
            re.I,
        ),
    ),
    (
        "SECTION",
        re.compile(
            r"\bsection\s+\d+[A-Za-z]?\b|\bclause\b|\bchapter\b|\bparagraph\b|"
            r"\bregulation\s+\d+|\bsub[- ]?section\b",
            re.I,
        ),
    ),
    (
        "DATE",
        re.compile(
            r"\b\d{1,2}\s+(January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{4}\b|"
            r"\b(19|20)\d{2}\b|\bfrom\s+\d|\bw\.e\.f\b|\beffective\b|\bdated\b",
            re.I,
        ),
    ),
    (
        "ENTITY",
        re.compile(
            r"\bRBI\b|\bSEBI\b|\bNBFC\b|\bbank\b|\bbanks\b|\bReserve Bank\b|"
            r"\bcompany\b|\bcompanies\b|\bfund\b|\bissuer\b|\bsponsor\b",
            re.I,
        ),
    ),
    (
        "APPLICABILITY",
        re.compile(
            r"\bapplicable\b|\bshall apply\b|\bapplies to\b|\beligible\b|"
            r"\bmust\b|\brequired\b|\bmandatory\b|\bshall\b|\bcover(ed|age)?\b",
            re.I,
        ),
    ),
]

_VALUE_EXTRACTORS: Dict[str, re.Pattern] = {
    "RATE": re.compile(r"\b\d+(\.\d+)?\s*%", re.I),
    "SECTION": re.compile(
        r"(Section\s+\d+[A-Za-z]?|Chapter\s+[IVXLCDM\d]+|Clause\s+\d+[A-Za-z]?)",
        re.I,
    ),
    "DATE": re.compile(
        r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{4}\b|\b(?:19|20)\d{2}\b",
        re.I,
    ),
    "THRESHOLD": re.compile(
        r"(?:minimum|maximum|not less than|not more than|at least)\s+[^\.;,]{3,40}",
        re.I,
    ),
    "ENTITY": re.compile(
        r"\b(?:Reserve Bank of India|RBI|SEBI|NBFC|scheduled commercial banks?|"
        r"banks?|merchant bankers?)\b",
        re.I,
    ),
    "APPLICABILITY": re.compile(
        r"(?:applicable to|shall apply to|applies to|eligible)\s+[^\.;,]{3,50}",
        re.I,
    ),
}


def classify_atom_type(text: str) -> str:
    """Pick the best-matching atom type for a claim string."""
    scores: Dict[str, int] = {t: 0 for t in ATOM_TYPES}
    for atom_type, pattern in _TYPE_PATTERNS:
        if pattern.search(text or ""):
            scores[atom_type] += 1
            # Prefer more specific types when they match a concrete value
            if atom_type in ("RATE", "SECTION", "DATE") and _VALUE_EXTRACTORS[atom_type].search(text or ""):
                scores[atom_type] += 2
    best = max(ATOM_TYPES, key=lambda t: scores[t])
    if scores[best] == 0:
        return "APPLICABILITY"
    return best


def extract_value(atom_type: str, text: str) -> str:
    """Pull a short value span from existing text; fall back to trimmed claim."""
    text = (text or "").strip()
    if not text:
        return ""
    pattern = _VALUE_EXTRACTORS.get(atom_type)
    if pattern:
        m = pattern.search(text)
        if m:
            return m.group(0).strip()
    # First clause / short preview — still from original text only
    short = re.split(r"[.;]", text)[0].strip()
    return short[:80] if short else text[:80]


def find_span(answer: str, needle: str) -> Tuple[Optional[int], Optional[int]]:
    """Locate needle inside answer (case-insensitive). Returns (start, end) or (None, None)."""
    if not answer or not needle:
        return None, None
    idx = answer.find(needle)
    if idx >= 0:
        return idx, idx + len(needle)
    lower_a = answer.lower()
    lower_n = needle.lower()
    idx = lower_a.find(lower_n)
    if idx >= 0:
        return idx, idx + len(needle)
    # Try value-only match for short needles
    if len(needle) > 12:
        fragment = needle[:40]
        idx = lower_a.find(fragment.lower())
        if idx >= 0:
            return idx, idx + len(fragment)
    return None, None


def _claim_units(rag_result: Dict[str, Any]) -> List[str]:
    """Prefer backend discrete claims; otherwise sentence-split the answer."""
    claims = [c.strip() for c in (rag_result.get("claims") or []) if str(c).strip()]
    # Drop placeholder / empty-context claims
    claims = [
        c
        for c in claims
        if c.lower() not in {
            "no relevant regulatory guidelines available.",
            "generated claim supported by retrieved regulatory text.",
        }
    ]
    if claims:
        return claims

    answer = (rag_result.get("answer") or "").strip()
    if not answer:
        return []
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 20]
    return parts[:12]


def build_atoms_payload(rag_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build RegGuard atomic-extraction API-shaped payload from a live RAG result.

    Shape:
      { query, answer, atoms: [{ atom_id, type, value, text, start, end, confidence, status }], ... }
    """
    if not rag_result:
        return {
            "query": "",
            "answer": "",
            "atoms": [],
            "error": "No RAG result available",
            "model_used": "",
            "citations": [],
            "has_context": False,
        }

    query = rag_result.get("query") or ""
    answer = rag_result.get("answer") or ""
    citations = rag_result.get("citations") or []
    chunks = rag_result.get("chunks") or []
    units = _claim_units(rag_result)

    atoms: List[Dict[str, Any]] = []
    for i, unit in enumerate(units, start=1):
        atom_type = classify_atom_type(unit)
        value = extract_value(atom_type, unit)
        # Prefer locating the value in the full answer for highlighting
        start, end = find_span(answer, value)
        if start is None:
            start, end = find_span(answer, unit)
        text_span = unit
        if start is not None and end is not None and answer:
            text_span = answer[start:end]

        atoms.append(
            {
                "atom_id": f"A{i:02d}",
                "type": atom_type,
                "value": value,
                "text": text_span,
                "source_sentence": unit,
                "start": start,
                "end": end,
                "confidence": None,  # never invent
                "status": "PENDING",
            }
        )

    return {
        "query": query,
        "answer": answer,
        "atoms": atoms,
        "model_used": rag_result.get("model_used") or "",
        "citations": citations,
        "has_context": bool(chunks) or bool(citations),
        "error": None,
    }


def summarize_atoms(atoms: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {t: 0 for t in ATOM_TYPES}
    for a in atoms:
        t = a.get("type")
        if t in counts:
            counts[t] += 1
    return counts
