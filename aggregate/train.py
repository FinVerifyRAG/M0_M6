"""
aggregate/train.py
-------------------
Train the multi-signal risk aggregator.

Steps
-----
1. Load feature table (Parquet built by dataset.py).
2. Logistic Regression baseline.
3. LightGBM classifier (main model).
4. Cross-validate by question-family column (if present), else k-fold.
5. Signal ablation: drop one signal at a time, re-train, compare AUROC.
6. Save the best model as `aggregator_v1.pkl` and the feature list as
   `features.json`.

Usage
-----
    python -m aggregate.train \\
        --features data/processed/feature_table.parquet \\
        --out      models/aggregator/ \\
        --ablation

Pitfall: this script must ONLY see agg_train data. Calibration and test
splits must be held out for M7.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd  # type: ignore

from aggregate.dataset import FEATURE_COLS, SIGNAL_COLS

logger = logging.getLogger("aggregate.train")
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_table(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    logger.info("Loaded feature table: %d rows", len(df))
    return df


def _check_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing feature columns as 0.5 (neutral) so training never errors."""
    for col in FEATURE_COLS:
        if col not in df.columns:
            logger.warning("Feature column '%s' missing — filling with 0.5", col)
            df[col] = 0.5
    return df


def _get_X_y(df: pd.DataFrame):
    X = df[FEATURE_COLS].fillna(0.5).values
    y = df["label"].values
    return X, y


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_logreg(X_train, y_train, X_val, y_val):
    """Logistic Regression baseline."""
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.metrics import roc_auc_score             # type: ignore

    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_val)[:, 1]
    auroc = roc_auc_score(y_val, proba)
    logger.info("LogReg AUROC (val): %.4f", auroc)
    return clf, auroc


def train_lgbm(X_train, y_train, X_val, y_val):
    """LightGBM main model."""
    from lightgbm import LGBMClassifier   # type: ignore
    from sklearn.metrics import roc_auc_score  # type: ignore

    clf = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[],
    )
    proba = clf.predict_proba(X_val)[:, 1]
    auroc = roc_auc_score(y_val, proba)
    logger.info("LightGBM AUROC (val): %.4f", auroc)
    return clf, auroc


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

def run_ablation(X_train, y_train, X_val, y_val, feature_names: List[str]) -> dict:
    """
    Drop one SIGNAL column at a time, retrain LightGBM, record delta AUROC.
    Returns dict {signal_name: auroc_without_it}.
    """
    from lightgbm import LGBMClassifier  # type: ignore
    from sklearn.metrics import roc_auc_score  # type: ignore

    results = {}
    signal_indices = [feature_names.index(s) for s in SIGNAL_COLS if s in feature_names]

    for idx in signal_indices:
        sig_name = feature_names[idx]
        mask = [i for i in range(X_train.shape[1]) if i != idx]
        Xt = X_train[:, mask]
        Xv = X_val[:, mask]

        clf = LGBMClassifier(n_estimators=200, learning_rate=0.05,
                             max_depth=4, random_state=42, verbose=-1)
        clf.fit(Xt, y_train)
        auc = roc_auc_score(y_val, clf.predict_proba(Xv)[:, 1])
        results[sig_name] = round(auc, 4)
        logger.info("Ablation drop %s → AUROC=%.4f", sig_name, auc)

    return results


# ---------------------------------------------------------------------------
# Cross-validation split
# ---------------------------------------------------------------------------

def _train_val_split(df: pd.DataFrame, val_frac: float = 0.15, seed: int = 42):
    """
    Split by question_family if present, else random.
    """
    if "question_family" in df.columns:
        families = df["question_family"].unique()
        rng = np.random.default_rng(seed)
        n_val = max(1, int(len(families) * val_frac))
        val_fams = set(rng.choice(families, n_val, replace=False))
        val_mask = df["question_family"].isin(val_fams)
        return df[~val_mask], df[val_mask]
    else:
        val_mask = np.random.default_rng(seed).random(len(df)) < val_frac
        return df[~val_mask], df[val_mask]


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    features_path: str,
    out_dir: str,
    run_ablation_flag: bool = False,
    val_frac: float = 0.15,
    seed: int = 42,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = _load_table(features_path)
    df = _check_cols(df)

    train_df, val_df = _train_val_split(df, val_frac=val_frac, seed=seed)
    logger.info("Split: train=%d, val=%d", len(train_df), len(val_df))

    X_train, y_train = _get_X_y(train_df)
    X_val,   y_val   = _get_X_y(val_df)

    # --- Baseline: LogReg ---
    lr_clf, lr_auroc = train_logreg(X_train, y_train, X_val, y_val)

    # --- Main: LightGBM ---
    lgbm_clf, lgbm_auroc = train_lgbm(X_train, y_train, X_val, y_val)

    # --- Save best model ---
    best_clf = lgbm_clf if lgbm_auroc >= lr_auroc else lr_clf
    model_path = out / "aggregator_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_clf, f)
    logger.info("Saved best model to %s (AUROC=%.4f)", model_path,
                max(lgbm_auroc, lr_auroc))

    # --- Save feature list ---
    feat_path = out / "features.json"
    with open(feat_path, "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)
    logger.info("Saved feature list to %s", feat_path)

    # --- Save metrics ---
    metrics = {
        "logreg_auroc": lr_auroc,
        "lgbm_auroc":   lgbm_auroc,
        "n_train":      len(train_df),
        "n_val":        len(val_df),
    }

    # --- Ablation (optional) ---
    if run_ablation_flag:
        ablation = run_ablation(X_train, y_train, X_val, y_val, FEATURE_COLS)
        metrics["ablation"] = ablation
        abl_path = out / "ablation.json"
        with open(abl_path, "w") as f:
            json.dump(ablation, f, indent=2)
        logger.info("Ablation saved to %s", abl_path)

    metrics_path = out / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics: %s", metrics)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train M6 risk aggregator.")
    parser.add_argument("--features", required=True, help="Path to feature_table.parquet")
    parser.add_argument("--out",      required=True, help="Output directory for model")
    parser.add_argument("--ablation", action="store_true", help="Run signal ablation")
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed",     type=int,   default=42)
    args = parser.parse_args()

    train(
        features_path=args.features,
        out_dir=args.out,
        run_ablation_flag=args.ablation,
        val_frac=args.val_frac,
        seed=args.seed,
    )
