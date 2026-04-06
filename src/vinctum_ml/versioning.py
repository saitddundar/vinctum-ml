"""
Model versioning and experiment tracking.
Tracks training runs with metrics, parameters, and artifact paths.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone


REGISTRY_DIR = Path(__file__).resolve().parents[2] / "models" / "registry"
REGISTRY_FILE = REGISTRY_DIR / "registry.json"


def _load_registry() -> dict:
    """Load the model registry."""
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE) as f:
            return json.load(f)
    return {"models": {}, "runs": []}


def _save_registry(registry: dict):
    """Save the model registry."""
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def _file_hash(path: str) -> str | None:
    """Compute SHA256 hash of a file."""
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()[:16]


def log_run(
    model_name: str,
    params: dict,
    metrics: dict,
    artifacts: dict[str, str],
    tags: dict[str, str] | None = None,
) -> str:
    """Log a training run to the registry.

    Args:
        model_name: e.g. "route_scorer" or "anomaly_detector"
        params: hyperparameters used
        metrics: evaluation metrics
        artifacts: {"model": path, "onnx": path, ...}
        tags: optional key-value tags

    Returns:
        run_id: unique identifier for this run
    """
    registry = _load_registry()

    timestamp = datetime.now(timezone.utc).isoformat()
    run_id = f"{model_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    artifact_hashes = {k: _file_hash(v) for k, v in artifacts.items()}

    run = {
        "run_id": run_id,
        "model_name": model_name,
        "timestamp": timestamp,
        "params": params,
        "metrics": metrics,
        "artifacts": artifacts,
        "artifact_hashes": artifact_hashes,
        "tags": tags or {},
    }

    registry["runs"].append(run)

    # Update latest pointer for this model
    registry["models"][model_name] = {
        "latest_run_id": run_id,
        "latest_metrics": metrics,
        "updated_at": timestamp,
    }

    _save_registry(registry)
    return run_id


def get_latest_run(model_name: str) -> dict | None:
    """Get the latest run for a model."""
    registry = _load_registry()
    model_runs = [r for r in registry["runs"] if r["model_name"] == model_name]
    return model_runs[-1] if model_runs else None


def get_run_history(model_name: str) -> list[dict]:
    """Get all runs for a model, newest first."""
    registry = _load_registry()
    runs = [r for r in registry["runs"] if r["model_name"] == model_name]
    return list(reversed(runs))


def compare_runs(run_id_a: str, run_id_b: str) -> dict:
    """Compare metrics between two runs."""
    registry = _load_registry()
    runs_by_id = {r["run_id"]: r for r in registry["runs"]}

    a = runs_by_id.get(run_id_a)
    b = runs_by_id.get(run_id_b)
    if not a or not b:
        raise ValueError(f"Run not found: {run_id_a if not a else run_id_b}")

    diff = {}
    all_keys = set(a["metrics"]) | set(b["metrics"])
    for key in sorted(all_keys):
        va = a["metrics"].get(key)
        vb = b["metrics"].get(key)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            diff[key] = {"a": va, "b": vb, "delta": vb - va}
        else:
            diff[key] = {"a": va, "b": vb}

    return {
        "run_a": run_id_a,
        "run_b": run_id_b,
        "metric_diff": diff,
    }


def list_models() -> dict:
    """List all registered models with their latest info."""
    registry = _load_registry()
    return registry["models"]
