"""Tests for model training and inference."""

import numpy as np
import pytest
import xgboost as xgb
from sklearn.ensemble import IsolationForest

from vinctum_ml.data.generator import generate_route_scoring_data, generate_anomaly_data
from vinctum_ml.models.route_scorer import FEATURES as ROUTE_FEATURES, TARGET as ROUTE_TARGET
from vinctum_ml.models.anomaly_detector import FEATURES as ANOMALY_FEATURES, TARGET as ANOMALY_TARGET


class TestRouteScorer:
    @pytest.fixture(autouse=True)
    def setup(self):
        df = generate_route_scoring_data(n_samples=500, seed=42)
        self.X = df[ROUTE_FEATURES]
        self.y = df[ROUTE_TARGET]

    def test_model_trains(self):
        model = xgb.XGBRegressor(n_estimators=10, max_depth=3, random_state=42)
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        assert len(preds) == len(self.y)

    def test_predictions_in_range(self):
        model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
        model.fit(self.X, self.y)
        preds = np.clip(model.predict(self.X), 0.0, 1.0)
        assert preds.min() >= 0.0
        assert preds.max() <= 1.0

    def test_reasonable_r2(self):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score

        X_tr, X_te, y_tr, y_te = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        model = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_tr, y_tr, verbose=False)
        r2 = r2_score(y_te, model.predict(X_te))
        assert r2 > 0.5, f"R2 too low: {r2}"

    def test_feature_count_matches(self):
        assert len(ROUTE_FEATURES) == self.X.shape[1]


class TestAnomalyDetector:
    @pytest.fixture(autouse=True)
    def setup(self):
        df = generate_anomaly_data(n_normal=400, n_anomaly=100, seed=42)
        self.X = df[ANOMALY_FEATURES]
        self.y = df[ANOMALY_TARGET]

    def test_model_trains(self):
        model = IsolationForest(n_estimators=50, contamination=0.2, random_state=42)
        model.fit(self.X)
        preds = model.predict(self.X)
        assert len(preds) == len(self.y)

    def test_detects_anomalies(self):
        model = IsolationForest(n_estimators=100, contamination=0.2, random_state=42)
        model.fit(self.X)
        preds = (model.predict(self.X) == -1).astype(int)
        # Should detect at least some anomalies
        assert preds.sum() > 0

    def test_decision_function(self):
        model = IsolationForest(n_estimators=100, contamination=0.2, random_state=42)
        model.fit(self.X)
        scores = model.decision_function(self.X)
        assert len(scores) == len(self.y)

    def test_feature_count_matches(self):
        assert len(ANOMALY_FEATURES) == self.X.shape[1]
