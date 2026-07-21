"""Train the match-outcome model.

Default is a calibrated Random Forest. Pass model="xgb" to use XGBoost instead
(requires `uv sync --extra xgb`). Both expose predict_proba returning
[P(away win), P(draw), P(home win)].
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score

from .data import FEATURES


def _make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=600,
        max_depth=12,
        min_samples_leaf=20,
        max_features="sqrt",   # feature randomness at each split
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )


def _make_xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        max_depth=5,
        n_estimators=400,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
    )


def train(df: pd.DataFrame, model: str = "rf", split_date: str = "2024-01-01",
          calibrate: bool = True):
    """Time-based split (never random for time series), fit, and report metrics."""
    train_df = df[df.date < split_date]
    test_df = df[df.date >= split_date]

    X_tr, y_tr = train_df[FEATURES], train_df.y
    X_te, y_te = test_df[FEATURES], test_df.y

    base = _make_xgb() if model == "xgb" else _make_rf()

    if calibrate and model == "rf":
        # honest probabilities to sample from in the Monte Carlo
        clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
        clf.fit(X_tr, y_tr)
    else:
        clf = base
        clf.fit(X_tr, y_tr)

    if len(test_df):
        proba = clf.predict_proba(X_te)
        ll = log_loss(y_te, proba, labels=[0, 1, 2])
        acc = accuracy_score(y_te, clf.predict(X_te))
        print(f"[{model}] holdout log loss = {ll:.4f}   accuracy = {acc:.3f}   "
              f"(n_test={len(test_df)})")

    return clf
