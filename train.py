"""Train all Vinctum ML models."""

import sys
sys.path.insert(0, "src")

from vinctum_ml.models.route_scorer import train as train_route
from vinctum_ml.models.anomaly_detector import train as train_anomaly
from vinctum_ml.versioning import log_run


def main():
    print("=" * 60)
    print("VINCTUM ML — Model Training Pipeline")
    print("=" * 60)

    # 1. Route Scoring
    print("\n[1/2] Training Route Scoring Model (XGBoost)...")
    route_result = train_route()
    print(f"  Model saved: {route_result['model_path']}")
    print(f"  ONNX saved:  {route_result['onnx_path']}")
    print(f"  R2 Score:    {route_result['metrics']['r2']:.4f}")
    print(f"  RMSE:        {route_result['metrics']['rmse']:.4f}")
    print(f"  MAE:         {route_result['metrics']['mae']:.4f}")

    run_id = log_run(
        model_name="route_scorer",
        params={"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
        metrics=route_result["metrics"],
        artifacts={"model": route_result["model_path"], "onnx": route_result["onnx_path"]},
    )
    print(f"  Run logged:  {run_id}")

    # 2. Anomaly Detection
    print("\n[2/2] Training Anomaly Detection Model (Isolation Forest)...")
    anomaly_result = train_anomaly()
    print(f"  Model saved: {anomaly_result['pkl_path']}")
    print(f"  ONNX saved:  {anomaly_result['onnx_path']}")
    print(f"  F1 Score:    {anomaly_result['metrics']['f1']:.4f}")
    print(f"  Precision:   {anomaly_result['metrics']['precision']:.4f}")
    print(f"  Recall:      {anomaly_result['metrics']['recall']:.4f}")
    print(f"\n{anomaly_result['metrics']['report']}")

    run_id = log_run(
        model_name="anomaly_detector",
        params={"n_estimators": 200, "contamination": "auto"},
        metrics={k: v for k, v in anomaly_result["metrics"].items() if k != "report"},
        artifacts={"model": anomaly_result["pkl_path"], "onnx": anomaly_result["onnx_path"]},
    )
    print(f"  Run logged:  {run_id}")

    print("=" * 60)
    print("All models trained and exported. Run the API:")
    print("  py -m uvicorn vinctum_ml.serving.app:app --reload")
    print("=" * 60)


if __name__ == "__main__":
    main()
