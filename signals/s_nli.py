"""
signals/s_nli.py  —  NLI Entailment Signal
--------------------------------------------
s_nli converts V2's raw entailment probability into a risk score.

    s_nli = 1 - p(entailment)

Interpretation:
    p_entail = 1.0  →  s_nli = 0.0  (strongly supported → low risk)
    p_entail = 0.0  →  s_nli = 1.0  (strongly contradicted → high risk)

When V2 was skipped (v2_entail_prob is None), s_nli defaults to 0.5
and a boolean flag `nli_skipped = True` is set so the aggregator can
treat "missing NLI" as its own informative pattern.

Public functions
----------------
    compute_s_nli(verified_atom) → float
    s_nli_features(verified_atom) → dict
"""
from __future__ import annotations

from common.schemas import VerifiedAtom

_DEFAULT_SKIPPED = 0.5   # neutral fallback when NLI was not run


def compute_s_nli(verified_atom: VerifiedAtom) -> float:
    """
    Return s_nli = 1 - p(entail), or 0.5 when NLI was skipped.
    """
    p = verified_atom.v2_entail_prob
    if p is None:
        return _DEFAULT_SKIPPED
    return round(1.0 - float(p), 4)


def s_nli_features(verified_atom: VerifiedAtom) -> dict:
    """
    Return:
        {
          "s_nli":        float in [0, 1],
          "nli_skipped":  0 or 1  (flag for missing NLI — important feature!),
        }
    """
    p = verified_atom.v2_entail_prob
    skipped = int(p is None)
    s_nli = _DEFAULT_SKIPPED if p is None else round(1.0 - float(p), 4)
    return {
        "s_nli":       s_nli,
        "nli_skipped": skipped,
    }
