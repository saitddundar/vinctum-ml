"""
FastAPI serving layer for Vinctum ML models.
Provides /score and /anomaly endpoints using ONNX Runtime inference.
"""

from pathlib import Path
from contextlib import asynccontextmanager
import time

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from vinctum_ml.logging import get_logger

logger = get_logger("vinctum_ml.serving")

MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "exported"

# Global ONNX sessions
route_session: ort.InferenceSession | None = None
anomaly_session: ort.InferenceSession | None = None


# -- Request/Response schemas --

class NodeMetrics(BaseModel):
    """Input matching vinctum-core NodeMetrics struct."""
    total_events: int
    successes: int
    failures: int
    timeouts: int
    reroutes: int
    circuit_opens: int
    avg_latency_ms: float
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p95_latency_ms: float
    total_bytes: int = 0
    avg_bytes_per_op: float
    failure_rate: float
    uptime: float = 0.0


class ScoreRequest(BaseModel):
    node_id: str
    metrics: NodeMetrics


class ScoreResponse(BaseModel):
    node_id: str
    score: float
    confidence: float


class AnomalyRequest(BaseModel):
    node_id: str
    metrics: NodeMetrics
    events_per_minute: float = 10.0


class AnomalyResponse(BaseModel):
    node_id: str
    is_anomaly: bool
    anomaly_score: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]


class RouteRequest(BaseModel):
    nodes: list[ScoreRequest]


class RouteResponse(BaseModel):
    scores: list[ScoreResponse]
    best_node: str
    route_score: float


# -- App --

@asynccontextmanager
async def lifespan(app: FastAPI):
    global route_session, anomaly_session

    route_onnx = MODEL_DIR / "route_scorer.onnx"
    anomaly_onnx = MODEL_DIR / "anomaly_detector.onnx"

    if route_onnx.exists():
        route_session = ort.InferenceSession(str(route_onnx))
        logger.info("Route scoring model loaded")

    if anomaly_onnx.exists():
        anomaly_session = ort.InferenceSession(str(anomaly_onnx))
        logger.info("Anomaly detection model loaded")

    if not route_session and not anomaly_session:
        logger.warning("No models found, run training first")

    yield

    route_session = None
    anomaly_session = None


app = FastAPI(
    title="Vinctum ML",
    description="AI/ML layer for Vinctum decentralized data courier platform",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(f"{request.method} {request.url.path} {response.status_code} {duration_ms:.1f}ms")
    return response


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        models_loaded={
            "route_scorer": route_session is not None,
            "anomaly_detector": anomaly_session is not None,
        },
    )


@app.post("/score", response_model=ScoreResponse)
async def score_node(req: ScoreRequest):
    """Score a single node for route quality."""
    if route_session is None:
        raise HTTPException(503, "Route scoring model not loaded. Run training first.")

    features = _extract_route_features(req.metrics)
    input_array = np.array([features], dtype=np.float32)

    input_name = route_session.get_inputs()[0].name
    result = route_session.run(None, {input_name: input_array})

    score = float(np.clip(result[0][0], 0.0, 1.0))
    confidence = min(req.metrics.total_events / 50.0, 1.0)

    return ScoreResponse(node_id=req.node_id, score=round(score, 4), confidence=round(confidence, 4))


@app.post("/anomaly", response_model=AnomalyResponse)
async def detect_anomaly(req: AnomalyRequest):
    """Check if a node is anomalous."""
    if anomaly_session is None:
        raise HTTPException(503, "Anomaly detection model not loaded. Run training first.")

    features = _extract_anomaly_features(req.metrics, req.events_per_minute)
    input_array = np.array([features], dtype=np.float32)

    input_name = anomaly_session.get_inputs()[0].name
    outputs = anomaly_session.run(None, {input_name: input_array})

    # IsolationForest ONNX: output[0] = label (1=normal, -1=anomaly), output[1] = scores
    label = int(outputs[0][0])
    is_anomaly = label == -1

    # Anomaly score from decision function (lower = more anomalous)
    anomaly_score = 0.5
    if len(outputs) > 1:
        raw_scores = outputs[1]
        if hasattr(raw_scores, '__len__') and len(raw_scores) > 0:
            score_val = float(raw_scores[0][1]) if raw_scores.ndim > 1 else float(raw_scores[0])
            anomaly_score = round(1.0 - max(0, min(1, (score_val + 0.5))), 4)

    return AnomalyResponse(
        node_id=req.node_id,
        is_anomaly=is_anomaly,
        anomaly_score=anomaly_score,
    )


@app.post("/route", response_model=RouteResponse)
async def score_route(req: RouteRequest):
    """Score multiple nodes and pick the best route."""
    if route_session is None:
        raise HTTPException(503, "Route scoring model not loaded. Run training first.")

    scores = []
    for node_req in req.nodes:
        features = _extract_route_features(node_req.metrics)
        input_array = np.array([features], dtype=np.float32)
        input_name = route_session.get_inputs()[0].name
        result = route_session.run(None, {input_name: input_array})

        score = float(np.clip(result[0][0], 0.0, 1.0))
        confidence = min(node_req.metrics.total_events / 50.0, 1.0)
        scores.append(ScoreResponse(
            node_id=node_req.node_id,
            score=round(score, 4),
            confidence=round(confidence, 4),
        ))

    best = max(scores, key=lambda s: s.score)
    route_score = round(np.prod([s.score for s in scores]) ** (1 / len(scores)), 4)

    return RouteResponse(scores=scores, best_node=best.node_id, route_score=route_score)


def _extract_route_features(m: NodeMetrics) -> list[float]:
    """Extract feature vector for route scoring model."""
    return [
        m.total_events, m.successes, m.failures, m.timeouts,
        m.reroutes, m.circuit_opens,
        m.avg_latency_ms, m.min_latency_ms, m.max_latency_ms, m.p95_latency_ms,
        m.total_bytes, m.avg_bytes_per_op,
        m.failure_rate, m.uptime,
    ]


def _extract_anomaly_features(m: NodeMetrics, events_per_minute: float) -> list[float]:
    """Extract feature vector for anomaly detection model."""
    return [
        m.total_events, m.successes, m.failures, m.timeouts,
        m.reroutes, m.circuit_opens,
        m.avg_latency_ms, m.p95_latency_ms,
        m.avg_bytes_per_op, m.failure_rate,
        events_per_minute,
    ]
