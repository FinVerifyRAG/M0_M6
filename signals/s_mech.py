"""
signals/s_mech.py  —  Mechanistic (Attention) Signal
------------------------------------------------------
s_mech implements a "lookback ratio" style mechanistic signal inspired by
the ReDeEP / lookback-ratio line of work.

Intuition
---------
When generating the atom's tokens, does the model attend MORE to the
evidence (prompt tokens) or to previously generated tokens (context)?

    lookback_ratio = sum(attn to prompt tokens) / sum(attn to all tokens)

    s_mech = 1 - lookback_ratio
           = fraction of attention going to generated tokens, not evidence.

High s_mech → model relies on its OWN prior (not evidence) → higher risk.
Low  s_mech → model heavily attends to evidence          → lower risk.

Implementation
--------------
Requires running Qwen with `output_attentions=True`. This is expensive,
so results are CACHED to disk in `.signal_cache/s_mech/<answer_id>.json`.

If attentions are not available (no GPU, no cache hit), s_mech falls back
to 0.5 (neutral). The `s_mech_skipped` flag lets the aggregator model this.

The attention is averaged across all heads and all decoder layers at the
atom's token span positions.

Cache
-----
Cache key: hash(answer.answer_text + query)[:16]
Cache path: .signal_cache/s_mech/<key>.json
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

from common.schemas import VerifiedAtom, GeneratedAnswer, RetrievalResult

logger = logging.getLogger("signals.s_mech")

_CACHE_DIR = Path(".signal_cache/s_mech")
_CHARS_PER_TOKEN = 4
_FALLBACK = 0.5


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

def _cache_key(answer_text: str, query: str) -> str:
    raw = (answer_text + query).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _cache_load(key: str) -> Optional[dict]:
    path = _CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _cache_save(key: str, data: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{key}.json"
    with open(path, "w") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# Token span helper (shared with s_ent)
# ---------------------------------------------------------------------------

def _token_span(char_span: Optional[Tuple[int, int]], answer_text: str) -> Tuple[int, int]:
    if char_span is None:
        return (0, max(1, len(answer_text) // _CHARS_PER_TOKEN))
    start = max(0, char_span[0] // _CHARS_PER_TOKEN)
    end   = max(start + 1, char_span[1] // _CHARS_PER_TOKEN)
    return (start, end)


# ---------------------------------------------------------------------------
# Attention extraction (heavy — requires transformers + GPU)
# ---------------------------------------------------------------------------

def _compute_lookback_from_model(
    answer: GeneratedAnswer,
    rr: RetrievalResult,
    atom_tok_start: int,
    atom_tok_end: int,
) -> Optional[float]:
    """
    Run Qwen with output_attentions=True on the full prompt+answer, then
    compute the lookback ratio for the atom's token span.

    Returns None if model is unavailable.
    """
    try:
        import torch  # type: ignore
        from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
    except ImportError:
        logger.warning("s_mech: torch/transformers not available. Skipping.")
        return None

    model_id = answer.model_id or "Qwen/Qwen2.5-7B-Instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, output_attentions=True, torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()
    except Exception as exc:
        logger.warning("s_mech: Could not load model %s: %s", model_id, exc)
        return None

    # Build the prompt (simplified — just context + answer)
    ctx = "\n\n".join(c.text[:500] for c in rr.chunks[:3])
    full_text = f"Evidence:\n{ctx}\n\nAnswer:\n{answer.answer_text}"
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # attentions: tuple of (num_layers,) each shape [batch, heads, seq, seq]
    # Average across layers and heads → [seq, seq]
    attn_stack = torch.stack(outputs.attentions, dim=0)   # [L, 1, H, S, S]
    attn_avg = attn_stack.mean(dim=(0, 1, 2))              # [S, S]

    # For the atom's generated token positions, compute lookback ratio
    gen_start = prompt_len   # first generated token index
    a_start = gen_start + atom_tok_start
    a_end   = gen_start + atom_tok_end
    a_end   = min(a_end, attn_avg.shape[0])

    if a_start >= attn_avg.shape[0]:
        return None

    attn_slice = attn_avg[a_start:a_end, :]   # [atom_toks, seq]
    if attn_slice.numel() == 0:
        return None

    # prompt attention = attention paid to prompt tokens [0:prompt_len]
    attn_to_prompt = attn_slice[:, :prompt_len].sum(dim=-1).mean().item()
    attn_total     = attn_slice.sum(dim=-1).mean().item()

    if attn_total == 0:
        return None

    lookback_ratio = attn_to_prompt / attn_total
    return round(lookback_ratio, 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_s_mech(
    verified_atom: VerifiedAtom,
    answer: GeneratedAnswer,
    rr: RetrievalResult,
    use_cache: bool = True,
) -> float:
    """
    Compute mechanistic (attention lookback) signal.
    s_mech = 1 - lookback_ratio ∈ [0, 1].
    Falls back to 0.5 when attentions are unavailable.
    """
    feats = s_mech_features(verified_atom, answer, rr, use_cache=use_cache)
    return feats["s_mech"]


def s_mech_features(
    verified_atom: VerifiedAtom,
    answer: GeneratedAnswer,
    rr: RetrievalResult,
    use_cache: bool = True,
) -> dict:
    """
    Return:
        {
          "s_mech":         float — 1 - lookback_ratio,
          "s_mech_skipped": 0/1  — flag when attentions unavailable,
        }
    """
    atom = verified_atom.atom
    tok_start, tok_end = _token_span(atom.span, answer.answer_text)

    # Check disk cache first
    cache_key = _cache_key(answer.answer_text, rr.query)
    if use_cache:
        cached = _cache_load(cache_key)
        if cached and atom.atom_id in cached:
            val = cached[atom.atom_id]
            return {"s_mech": val, "s_mech_skipped": 0}

    # Try computing from model
    lookback = _compute_lookback_from_model(answer, rr, tok_start, tok_end)

    if lookback is None:
        logger.debug("s_mech: skipped for atom %s (model unavailable).", atom.atom_id)
        return {"s_mech": _FALLBACK, "s_mech_skipped": 1}

    s_mech_val = round(1.0 - lookback, 4)

    # Save to cache
    if use_cache:
        existing = _cache_load(cache_key) or {}
        existing[atom.atom_id] = s_mech_val
        _cache_save(cache_key, existing)

    return {"s_mech": s_mech_val, "s_mech_skipped": 0}
