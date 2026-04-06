FROM python:3.12-slim AS base

WORKDIR /app

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt onnxmltools

COPY src/ src/
COPY train.py .

RUN pip install --no-cache-dir -e .

# Train models at build time so they're baked into the image
RUN python train.py

EXPOSE 8000

CMD ["uvicorn", "vinctum_ml.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
