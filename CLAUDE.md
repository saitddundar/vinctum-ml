# vinctum-ml

Vinctum P2P platformunun AI/ML katmani. Go backend (vinctum-core) ile HTTP uzerinden entegre olur.

## Tech Stack

- Python 3.11+, XGBoost, scikit-learn, ONNX Runtime, FastAPI
- Synthetic data generation (gercek veri yok, generator.py uretir)
- ONNX export ile inference

## Project Structure

```
src/vinctum_ml/
  data/generator.py       # Sentetik veri uretici
  models/route_scorer.py  # XGBoost route scoring (node kalite skoru)
  models/anomaly_detector.py  # Isolation Forest anomali tespiti
  serving/app.py          # FastAPI serving layer (/score /anomaly /route)
train.py                  # Unified training pipeline
models/exported/          # Trained model artifacts (.json, .onnx, .pkl)
```

## Commands

```bash
# Setup
python -m venv venv && source venv/Scripts/activate
pip install -e ".[dev]"

# Train models
python train.py

# Run API server
uvicorn vinctum_ml.serving.app:app --reload

# Tests
pytest
```

## Conventions

- Model export: native format + ONNX + metadata pickle
- XGBoost ONNX export uses `onnxmltools` (not skl2onnx)
- scikit-learn ONNX export uses `skl2onnx`
- API responses use Pydantic models
- Sentetik veri profilleri: good, average, bad, unstable
- .gitignore: .onnx, .pkl, .csv, .env dosyalari tracked degil
