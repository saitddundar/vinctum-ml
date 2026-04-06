"""
XGBoost-based route scoring model.
Predicts node quality score (0.0-1.0) from network metrics.
"""

import pickle
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from vinctum_ml.data.generator import generate_route_scoring_data

FEATURES = [
    "total_events",
    "successes",
    "failures",
    "timeouts",
    "reroutes",
    "circuit_opens",
    "avg_latency_ms",
    "min_latency_ms",
    "max_latency_ms",
    "p95_latency_ms",
    "total_bytes",
    "avg_bytes_per_op",
    "failure_rate",
    "uptime",
]

TARGET = "quality_score"

MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "exported"


def train(n_samples: int = 10000, seed: int = 42) -> dict:
    """Train XGBoost route scoring model and export to ONNX."""
    df = generate_route_scoring_data(n_samples=n_samples, seed=seed)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        objective="reg:squarederror",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0.0, 1.0)

    metrics = {
        "mse": float(mean_squared_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Native XGBoost format
    model_path = MODEL_DIR / "route_scorer.json"
    model.save_model(str(model_path))

    # ONNX export
    onnx_path = MODEL_DIR / "route_scorer.onnx"
    _export_onnx(model, X_train, onnx_path)

    # Feature names for serving
    meta_path = MODEL_DIR / "route_scorer_meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump({"features": FEATURES, "metrics": metrics}, f)

    return {"model_path": str(model_path), "onnx_path": str(onnx_path), "metrics": metrics}


def _export_onnx(model: xgb.XGBRegressor, X_sample, onnx_path: Path):
    """Export trained model to ONNX format using onnxmltools."""
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    import onnx

    initial_type = [("input", FloatTensorType([None, X_sample.shape[1]]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type)
    onnx.save_model(onnx_model, str(onnx_path))


if __name__ == "__main__":
    result = train()
    print("Route Scorer Training Complete")
    print(f"  Model: {result['model_path']}")
    print(f"  ONNX:  {result['onnx_path']}")
    print(f"  Metrics:")
    for k, v in result["metrics"].items():
        print(f"    {k}: {v:.4f}")
