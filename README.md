# Vinctum ML

AI/ML layer for the [Vinctum](https://github.com/saitddundar/vinctum-core) decentralized data courier platform.

## Overview

Vinctum ML provides intelligent network decision-making for the Vinctum P2P file transfer system. It runs on the central server and enhances routing and security without ever touching file contents.

### Models

| Model | Task | Algorithm | Description |
|-------|------|-----------|-------------|
| **Route Scorer** | Route Selection | XGBoost | Predicts node quality score (0.0–1.0) from network metrics like latency, uptime, throughput, and stability |
| **Anomaly Detector** | Threat Detection | Isolation Forest | Detects anomalous nodes — latency spikes, high failure rates, traffic anomalies, unresponsive peers |

### Architecture

```
vinctum-core (Go)                    vinctum-ml (Python)
┌──────────────────────┐             ┌──────────────────────────┐
│ Routing Service      │  ── HTTP ─► │ FastAPI                  │
│                      │             │  ├─ POST /score          │
│ NodeIntelligence     │  ◄── JSON ─ │  ├─ POST /anomaly       │
│  ├─ ScoreNode()      │             │  ├─ POST /route          │
│  └─ IsAnomalous()    │             │  └─ GET  /health         │
└──────────────────────┘             │                          │
                                     │ ONNX Runtime Inference   │
                                     │  ├─ route_scorer.onnx    │
                                     │  └─ anomaly_detector.onnx│
                                     └──────────────────────────┘
```

## Setup

```bash
# Create virtual environment
py -m venv venv
source venv/Scripts/activate  # Windows
# source venv/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Train Models

```bash
py train.py
```

This generates synthetic data, trains both models, and exports them as ONNX files to `models/exported/`.

### Run API Server

```bash
cd src
py -m uvicorn vinctum_ml.serving.app:app --reload
```

API docs available at `http://localhost:8000/docs`

### API Endpoints

**Score a node:**
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "node_id": "peer_42",
    "metrics": {
      "total_events": 500,
      "successes": 480,
      "failures": 20,
      "timeouts": 5,
      "reroutes": 2,
      "circuit_opens": 0,
      "avg_latency_ms": 35.0,
      "p95_latency_ms": 78.0,
      "avg_bytes_per_op": 102400.0,
      "failure_rate": 0.04,
      "uptime": 0.96
    }
  }'
```

**Detect anomaly:**
```bash
curl -X POST http://localhost:8000/anomaly \
  -H "Content-Type: application/json" \
  -d '{
    "node_id": "peer_17",
    "metrics": {
      "total_events": 100,
      "successes": 10,
      "failures": 90,
      "timeouts": 60,
      "reroutes": 15,
      "circuit_opens": 8,
      "avg_latency_ms": 1500.0,
      "p95_latency_ms": 4200.0,
      "avg_bytes_per_op": 512.0,
      "failure_rate": 0.90,
      "uptime": 0.10
    },
    "events_per_minute": 0.5
  }'
```

**Score a route (multiple nodes):**
```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{
    "nodes": [
      {"node_id": "peer_A", "metrics": {"total_events": 300, "successes": 290, "failures": 10, "timeouts": 2, "reroutes": 1, "circuit_opens": 0, "avg_latency_ms": 25.0, "p95_latency_ms": 55.0, "avg_bytes_per_op": 81920.0, "failure_rate": 0.03, "uptime": 0.97}},
      {"node_id": "peer_B", "metrics": {"total_events": 150, "successes": 120, "failures": 30, "timeouts": 10, "reroutes": 5, "circuit_opens": 2, "avg_latency_ms": 120.0, "p95_latency_ms": 280.0, "avg_bytes_per_op": 40960.0, "failure_rate": 0.20, "uptime": 0.80}}
    ]
  }'
```

## Project Structure

```
vinctum-ml/
├── train.py                      # Unified training pipeline
├── pyproject.toml
├── requirements.txt
├── models/exported/              # Trained model artifacts (gitignored)
├── src/vinctum_ml/
│   ├── data/
│   │   └── generator.py          # Synthetic data generation
│   ├── models/
│   │   ├── route_scorer.py       # XGBoost training + ONNX export
│   │   └── anomaly_detector.py   # Isolation Forest training + ONNX export
│   ├── serving/
│   │   └── app.py                # FastAPI endpoints
│   └── evaluation/               # Model evaluation (planned)
└── notebooks/                    # Jupyter experiments (planned)
```

## Tech Stack

- **XGBoost** — Route scoring (regression)
- **scikit-learn** — Isolation Forest anomaly detection
- **ONNX Runtime** — Model inference
- **FastAPI** — REST API serving
- **NumPy / Pandas** — Data processing

## License

MIT
