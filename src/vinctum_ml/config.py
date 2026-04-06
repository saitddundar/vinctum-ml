"""Environment-based configuration for Vinctum ML."""

import os
from pathlib import Path


class Config:
    """Configuration loaded from environment variables with sensible defaults."""

    # Paths
    MODEL_DIR: Path = Path(os.getenv(
        "VINCTUM_ML_MODEL_DIR",
        str(Path(__file__).resolve().parents[2] / "models" / "exported"),
    ))

    # Serving
    HOST: str = os.getenv("VINCTUM_ML_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("VINCTUM_ML_PORT", "8000"))
    WORKERS: int = int(os.getenv("VINCTUM_ML_WORKERS", "1"))

    # Logging
    LOG_LEVEL: str = os.getenv("VINCTUM_ML_LOG_LEVEL", "INFO")

    # Training
    ROUTE_N_SAMPLES: int = int(os.getenv("VINCTUM_ML_ROUTE_SAMPLES", "10000"))
    ROUTE_N_ESTIMATORS: int = int(os.getenv("VINCTUM_ML_ROUTE_ESTIMATORS", "200"))
    ROUTE_MAX_DEPTH: int = int(os.getenv("VINCTUM_ML_ROUTE_MAX_DEPTH", "6"))
    ROUTE_LEARNING_RATE: float = float(os.getenv("VINCTUM_ML_ROUTE_LR", "0.1"))

    ANOMALY_N_NORMAL: int = int(os.getenv("VINCTUM_ML_ANOMALY_NORMAL", "8000"))
    ANOMALY_N_ANOMALY: int = int(os.getenv("VINCTUM_ML_ANOMALY_ANOMALY", "2000"))
    ANOMALY_N_ESTIMATORS: int = int(os.getenv("VINCTUM_ML_ANOMALY_ESTIMATORS", "200"))

    # API Security
    API_KEY: str | None = os.getenv("VINCTUM_ML_API_KEY")

    SEED: int = int(os.getenv("VINCTUM_ML_SEED", "42"))


config = Config()
