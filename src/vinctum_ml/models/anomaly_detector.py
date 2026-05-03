"""
Isolation Forest-based anomaly detection model.
Detects anomalous network nodes from traffic metrics.
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from vinctum_ml.data.generator import generate_anomaly_data

FEATURES = [
    "total_events",
    "successes",
    "failures",
    "timeouts",
    "reroutes",
    "circuit_opens",
    "avg_latency_ms",
    "p95_latency_ms",
    "avg_bytes_per_op",
    "failure_rate",
    "events_per_minute",
]

TARGET = "is_anomaly"

MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "exported"


def train(n_normal: int = 8000, n_anomaly: int = 2000, seed: int = 42) -> dict:
    """Train Isolation Forest anomaly detection model and export to ONNX."""
    df = generate_anomaly_data(n_normal=n_normal, n_anomaly=n_anomaly, seed=seed)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # Isolation Forest: contamination = expected anomaly ratio
    contamination = y_train.mean()

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=seed,
        n_jobs=-1,
    )

    # Isolation Forest is unsupervised — fit on all training data
    model.fit(X_train)

    # Predict: IsolationForest returns 1 (normal) and -1 (anomaly)
    y_pred_raw = model.predict(X_test)
    y_pred = (y_pred_raw == -1).astype(int)  # Convert to 0/1

    metrics = {
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "report": classification_report(y_test, y_pred, target_names=["normal", "anomaly"]),
    }

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    pkl_path = MODEL_DIR / "anomaly_detector.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)

    # ONNX export
    onnx_path = MODEL_DIR / "anomaly_detector.onnx"
    _export_onnx(model, X_train, onnx_path)

    # Feature names for serving
    meta_path = MODEL_DIR / "anomaly_detector_meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump({"features": FEATURES, "metrics": metrics}, f)

    return {"pkl_path": str(pkl_path), "onnx_path": str(onnx_path), "metrics": metrics}


def _export_onnx(model: IsolationForest, X_sample, onnx_path: Path):
    """Export trained model to ONNX format."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [("input", FloatTensorType([None, X_sample.shape[1]]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset={"": 17, "ai.onnx.ml": 3},
    )

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())


if __name__ == "__main__":
    result = train()
    print("Anomaly Detector Training Complete")
    print(f"  Model: {result['pkl_path']}")
    print(f"  ONNX:  {result['onnx_path']}")
    print(f"  Metrics:")
    print(f"    F1:        {result['metrics']['f1']:.4f}")
    print(f"    Precision: {result['metrics']['precision']:.4f}")
    print(f"    Recall:    {result['metrics']['recall']:.4f}")
    print(f"\n{result['metrics']['report']}")
