"""
M6: signals/ — Multi-signal atom risk computation.

Public API
----------
    from signals import compute_all_signals

    feats = compute_all_signals(
        verified_atom,
        answer,
        rr,
        no_context_logprobs=None,   # optional
        use_mech_cache=True,        # s_mech disk cache
    )
    # feats is a flat dict with all signal columns ready for the aggregator:
    # {s_div, s_ret, s_ver, s_nli, nli_skipped, s_ent, s_ent_max,
    #  s_ent_skipped, s_mech, s_mech_skipped, v1_match, v1_mismatch,
    #  v1_not_found, v1_na}
"""
from __future__ import annotations

from typing import List, Optional

from common.schemas import GeneratedAnswer, RetrievalResult, VerifiedAtom

from signals.s_div import compute_s_div
from signals.s_ent import s_ent_features
from signals.s_mech import s_mech_features
from signals.s_nli import s_nli_features
from signals.s_ret import compute_s_ret
from signals.s_ver import s_ver_features


def compute_all_signals(
    verified_atom: VerifiedAtom,
    answer: GeneratedAnswer,
    rr: RetrievalResult,
    no_context_logprobs: Optional[List[float]] = None,
    use_mech_cache: bool = True,
) -> dict:
    """
    Compute all M6 signals for one VerifiedAtom and return a flat feature dict.

    Parameters
    ----------
    verified_atom       : VerifiedAtom from M5 cascade.
    answer              : GeneratedAnswer (with token_logprobs if available).
    rr                  : RetrievalResult (with rerank_scores in metadata if available).
    no_context_logprobs : Optional per-token logprobs of the no-context answer
                          (used for s_div Strategy B; Strategy A uses
                          answer.metadata["no_context_answer"]).
    use_mech_cache      : Whether to use disk cache for s_mech (default True).

    Returns
    -------
    dict with keys:
        s_div, s_ret, s_ver, s_nli, nli_skipped,
        s_ent, s_ent_max, s_ent_skipped,
        s_mech, s_mech_skipped,
        v1_match, v1_mismatch, v1_not_found, v1_na
    """
    feats: dict = {}

    # --- s_ver: V1 deterministic encoding ---
    feats.update(s_ver_features(verified_atom))

    # --- s_nli: 1 - p(entail) ---
    feats.update(s_nli_features(verified_atom))

    # --- s_ent: model token entropy ---
    feats.update(s_ent_features(verified_atom, answer))

    # --- s_ret: retrieval reranker strength ---
    feats["s_ret"] = compute_s_ret(verified_atom, rr)

    # --- s_div: context divergence ---
    feats["s_div"] = compute_s_div(verified_atom, answer, no_context_logprobs)

    # --- s_mech: mechanistic lookback ratio ---
    feats.update(s_mech_features(verified_atom, answer, rr, use_cache=use_mech_cache))

    return feats


__all__ = ["compute_all_signals"]
