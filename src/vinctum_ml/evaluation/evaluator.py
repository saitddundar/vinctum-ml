"""
Model evaluation and benchmarking utilities.
Generates detailed metrics, plots feature importance, and compares model runs.
"""

import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from vinctum_ml.data.generator import generate_route_scoring_data, generate_anomaly_data
from vinctum_ml.models.route_scorer import FEATURES as ROUTE_FEATURES, TARGET as ROUTE_TARGET
from vinctum_ml.models.anomaly_detector import FEATURES as ANOMALY_FEATURES, TARGET as ANOMALY_TARGET

EVAL_DIR = Path(__file__).resolve().parents[3] / "models" / "evaluation"


def evaluate_route_scorer(n_samples: int = 10000, seed: int = 42, n_folds: int = 5) -> dict:
    """Full evaluation of the route scoring model with cross-validation."""
    df = generate_route_scoring_data(n_samples=n_samples, seed=seed)
    X = df[ROUTE_FEATURES]
    y = df[ROUTE_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=seed, objective="reg:squarederror",
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = np.clip(model.predict(X_test), 0.0, 1.0)

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=n_folds, scoring="r2")

    # Feature importance
    importance = dict(zip(ROUTE_FEATURES, model.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    # Error distribution by score buckets
    buckets = {}
    for low, high, label in [(0, 0.3, "low"), (0.3, 0.7, "mid"), (0.7, 1.0, "high")]:
        mask = (y_test >= low) & (y_test < high)
        if mask.sum() > 0:
            buckets[label] = {
                "count": int(mask.sum()),
                "mae": float(mean_absolute_error(y_test[mask], y_pred[mask])),
                "rmse": float(np.sqrt(mean_squared_error(y_test[mask], y_pred[mask]))),
            }

    result = {
        "model": "route_scorer",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": n_samples,
        "test_size": len(y_test),
        "metrics": {
            "mse": float(mean_squared_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        },
        "cross_validation": {
            "n_folds": n_folds,
            "r2_mean": float(cv_scores.mean()),
            "r2_std": float(cv_scores.std()),
            "r2_scores": cv_scores.tolist(),
        },
        "feature_importance": importance,
        "error_by_bucket": buckets,
    }

    _save_report(result, "route_scorer")
    return result


def evaluate_anomaly_detector(
    n_normal: int = 8000, n_anomaly: int = 2000, seed: int = 42
) -> dict:
    """Full evaluation of the anomaly detection model."""
    df = generate_anomaly_data(n_normal=n_normal, n_anomaly=n_anomaly, seed=seed)
    X = df[ANOMALY_FEATURES]
    y = df[ANOMALY_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    contamination = y_train.mean()
    model = IsolationForest(
        n_estimators=200, contamination=contamination,
        max_samples="auto", random_state=seed, n_jobs=-1,
    )
    model.fit(X_train)

    y_pred_raw = model.predict(X_test)
    y_pred = (y_pred_raw == -1).astype(int)

    # Decision scores for threshold analysis
    decision_scores = model.decision_function(X_test)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Per-anomaly-type analysis (re-generate to get labels)
    anomaly_mask = y_test == 1
    normal_mask = y_test == 0

    result = {
        "model": "anomaly_detector",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(df),
        "test_size": len(y_test),
        "contamination": float(contamination),
        "metrics": {
            "f1": float(f1_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "decision_scores": {
            "normal_mean": float(decision_scores[normal_mask].mean()),
            "normal_std": float(decision_scores[normal_mask].std()),
            "anomaly_mean": float(decision_scores[anomaly_mask].mean()),
            "anomaly_std": float(decision_scores[anomaly_mask].std()),
        },
    }

    _save_report(result, "anomaly_detector")
    return result


def evaluate_all(**kwargs) -> dict:
    """Run evaluation for all models."""
    return {
        "route_scorer": evaluate_route_scorer(**kwargs),
        "anomaly_detector": evaluate_anomaly_detector(**kwargs),
    }


def _save_report(result: dict, model_name: str):
    """Save evaluation report as JSON."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = EVAL_DIR / f"{model_name}_{ts}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    # Also save as latest
    latest = EVAL_DIR / f"{model_name}_latest.json"
    with open(latest, "w") as f:
        json.dump(result, f, indent=2)
