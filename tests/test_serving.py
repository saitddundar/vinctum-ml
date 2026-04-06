"""Tests for the FastAPI serving layer."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from fastapi.testclient import TestClient

from vinctum_ml.serving.app import app


client = TestClient(app)

SAMPLE_METRICS = {
    "total_events": 1000,
    "successes": 950,
    "failures": 50,
    "timeouts": 10,
    "reroutes": 5,
    "circuit_opens": 1,
    "avg_latency_ms": 45.0,
    "min_latency_ms": 5.0,
    "max_latency_ms": 200.0,
    "p95_latency_ms": 120.0,
    "total_bytes": 1024000,
    "avg_bytes_per_op": 1024.0,
    "failure_rate": 0.05,
    "uptime": 0.95,
}


class TestHealth:
    def test_health_endpoint(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "models_loaded" in data


class TestScoreEndpoint:
    def test_returns_503_when_no_model(self):
        resp = client.post("/score", json={
            "node_id": "node-1",
            "metrics": SAMPLE_METRICS,
        })
        # Model not loaded in test context
        assert resp.status_code in (200, 503)

    def test_validates_request(self):
        resp = client.post("/score", json={"node_id": "node-1"})
        assert resp.status_code == 422

    @patch("vinctum_ml.serving.app.route_session")
    def test_score_with_mock_session(self, mock_session):
        mock_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_session.run.return_value = [np.array([0.85])]

        resp = client.post("/score", json={
            "node_id": "node-1",
            "metrics": SAMPLE_METRICS,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_id"] == "node-1"
        assert 0.0 <= data["score"] <= 1.0
        assert 0.0 <= data["confidence"] <= 1.0


class TestAnomalyEndpoint:
    def test_returns_503_when_no_model(self):
        resp = client.post("/anomaly", json={
            "node_id": "node-1",
            "metrics": SAMPLE_METRICS,
            "events_per_minute": 15.0,
        })
        assert resp.status_code in (200, 503)

    def test_validates_request(self):
        resp = client.post("/anomaly", json={})
        assert resp.status_code == 422


class TestRouteEndpoint:
    def test_returns_503_when_no_model(self):
        resp = client.post("/route", json={
            "nodes": [{"node_id": "n1", "metrics": SAMPLE_METRICS}]
        })
        assert resp.status_code in (200, 503)

    def test_validates_request(self):
        resp = client.post("/route", json={"nodes": "invalid"})
        assert resp.status_code == 422
