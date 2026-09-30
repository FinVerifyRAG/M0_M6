"""
aggregate/model.py
-------------------
Inference-time aggregator: turns VerifiedAtoms into ScoredAtoms.

This module loads the saved LightGBM / LogReg model + probability calibrator
and applies them to a list of VerifiedAtoms to produce per-atom risk scores
in [0, 1].

Public API
----------
    from aggregate.model import score

    scored_atoms = score(verified_atoms, answer, rr)

    # Each ScoredAtom has:
    #   .verified  — the original VerifiedAtom
    #   .risk      — float in [0,1], 0=safe, 1=high risk of being wrong
    #   .metadata  — includes the raw signal feature dict

Usage without a trained model
------------------------------
If `aggregator_v1.pkl` is not found, `score()` falls back to a simple
weighted average of the available signals so the pipeline can still run
end-to-end before training data is collected.

Configuration
-------------
Model paths are loaded from `configs/signals.yaml` (key: aggregator_model_path
and calibrator_path). Defaults point to `models/aggregator/`.
"""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional

from common.schemas import GeneratedAnswer, RetrievalResult, ScoredAtom, VerifiedAtom
from signals import compute_all_signals
from aggregate.dataset import FEATURE_COLS

logger = logging.getLogger("aggregate.model")

# Default paths — overridden by signals.yaml / environment variables
_DEFAULT_MODEL_PATH = Path("models/aggregator/aggregator_v1.pkl")
_DEFAULT_CAL_PATH   = Path("models/aggregator/calibrator_v1.pkl")

# -----------------------------------------------------------------------
# Lazy singleton model loader
# -----------------------------------------------------------------------

_model = None
_calibrator = None


def _load_model(model_path: Optional[str] = None) -> Optional[object]:
    global _model
    if _model is not None:
        return _model

    p = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
    if not p.exists():
        logger.warning(
            "Aggregator model not found at %s. "
            "Running in fallback (weighted-average) mode. "
            "Train with: python -m aggregate.train",
            p,
        )
        return None

    with open(p, "rb") as f:
        _model = pickle.load(f)
    logger.info("Loaded aggregator model from %s", p)
    return _model


def _load_calibrator(cal_path: Optional[str] = None) -> Optional[object]:
    global _calibrator
    if _calibrator is not None:
        return _calibrator

    p = Path(cal_path) if cal_path else _DEFAULT_CAL_PATH
    if not p.exists():
        return None

    with open(p, "rb") as f:
        _calibrator = pickle.load(f)
    logger.info("Loaded calibrator from %s", p)
    return _calibrator


# -----------------------------------------------------------------------
# Fallback: simple weighted average when no model is trained yet
# -----------------------------------------------------------------------

_FALLBACK_WEIGHTS = {
    "s_ver":   0.30,
    "s_nli":   0.25,
    "s_ent":   0.15,
    "s_ret":   0.10,   # inverted: high s_ret → low risk
    "s_div":   0.10,   # inverted: high s_div → low risk
    "s_mech":  0.10,
}


def _fallback_risk(feats: dict) -> float:
    """
    Heuristic weighted-average risk when no aggregator model is available.
    s_ret and s_div are inverted (high value → evidence support → lower risk).
    """
    total_w = 0.0
    score   = 0.0
    for sig, w in _FALLBACK_WEIGHTS.items():
        if sig not in feats:
            continue
        val = feats[sig]
        # Invert retrieval and divergence signals (high = good = low risk)
        if sig in {"s_ret", "s_div"}:
            val = 1.0 - val
        score   += w * val
        total_w += w
    return round(score / total_w, 4) if total_w > 0 else 0.5


# -----------------------------------------------------------------------
# Feature vector builder
# -----------------------------------------------------------------------

def _build_feature_vector(feats: dict) -> list:
    """
    Build a feature vector in FEATURE_COLS order, filling missing columns
    with 0.5 (neutral). This vector is passed to the sklearn/lightgbm model.
    """
    return [feats.get(col, 0.5) for col in FEATURE_COLS]


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def score(
    verified_atoms: List[VerifiedAtom],
    answer: GeneratedAnswer,
    rr: RetrievalResult,
    model_path: Optional[str] = None,
    calibrator_path: Optional[str] = None,
    use_mech_cache: bool = True,
) -> List[ScoredAtom]:
    """
    Compute a risk score in [0, 1] for every VerifiedAtom.

    Parameters
    ----------
    verified_atoms   : List of VerifiedAtom objects from M5.
    answer           : GeneratedAnswer with token_logprobs and metadata.
    rr               : RetrievalResult with rerank_scores in metadata.
    model_path       : Override path to aggregator_v1.pkl.
    calibrator_path  : Override path to calibrator_v1.pkl.
    use_mech_cache   : Whether to use disk cache for expensive s_mech signal.

    Returns
    -------
    List[ScoredAtom] in the same order as input, each with .risk in [0, 1].
    """
    model = _load_model(model_path)
    calibrator = _load_calibrator(calibrator_path)

    scored: List[ScoredAtom] = []

    # Build feature matrix for batch prediction (if model available)
    all_feats = []
    for va in verified_atoms:
        feats = compute_all_signals(va, answer, rr, use_mech_cache=use_mech_cache)

        # Add atom-type one-hot
        atom_type = va.atom.type
        from aggregate.dataset import ATOM_TYPES, REGULATORS
        for t in ATOM_TYPES:
            feats[f"type_{t}"] = int(atom_type == t)

        # Add regulator one-hot
        reg = (va.atom.metadata.get("regulator", "") if hasattr(va.atom, "metadata")
               else "").upper()
        if reg not in REGULATORS:
            reg = "OTHER"
        for r in REGULATORS:
            feats[f"reg_{r}"] = int(reg == r)

        all_feats.append(feats)

    if model is not None:
        import numpy as np
        X = np.array([_build_feature_vector(f) for f in all_feats], dtype=float)
        X = np.nan_to_num(X, nan=0.5)
        raw_probs = model.predict_proba(X)[:, 1]

        # Apply calibration if available
        if calibrator is not None:
            cal_obj = calibrator
            method  = cal_obj.get("method", "isotonic") if isinstance(cal_obj, dict) else "isotonic"
            cal     = cal_obj.get("calibrator", cal_obj) if isinstance(cal_obj, dict) else cal_obj
            if method == "isotonic":
                calibrated = cal.predict(raw_probs)
            elif method == "platt":
                calibrated = cal.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
            else:
                calibrated = raw_probs
        else:
            calibrated = raw_probs

        for va, feats, risk_val in zip(verified_atoms, all_feats, calibrated):
            scored.append(
                ScoredAtom(
                    verified=va,
                    risk=round(float(risk_val), 4),
                    metadata={
                        "signals": {k: v for k, v in feats.items()
                                    if k.startswith("s_") or k.startswith("v1_")
                                    or k.endswith("_skipped")},
                        "model_mode": "lightgbm" if hasattr(model, "predict_proba") else "logreg",
                        "calibrated": calibrator is not None,
                    },
                )
            )
    else:
        # Fallback mode — no model trained yet
        for va, feats in zip(verified_atoms, all_feats):
            risk_val = _fallback_risk(feats)
            scored.append(
                ScoredAtom(
                    verified=va,
                    risk=risk_val,
                    metadata={
                        "signals": {k: v for k, v in feats.items()
                                    if k.startswith("s_") or k.startswith("v1_")
                                    or k.endswith("_skipped")},
                        "model_mode": "fallback_weighted_average",
                        "calibrated": False,
                    },
                )
            )

    logger.debug(
        "Scored %d atoms. Risks: %s",
        len(scored),
        [s.risk for s in scored],
    )
    return scored
