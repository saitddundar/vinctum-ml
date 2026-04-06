"""Tests for synthetic data generation."""

import pandas as pd
import pytest

from vinctum_ml.data.generator import generate_route_scoring_data, generate_anomaly_data


class TestRouteScoring:
    def test_shape(self):
        df = generate_route_scoring_data(n_samples=100, seed=0)
        assert len(df) == 100
        assert "quality_score" in df.columns

    def test_features_present(self):
        df = generate_route_scoring_data(n_samples=50, seed=0)
        expected = [
            "total_events", "successes", "failures", "timeouts",
            "reroutes", "circuit_opens", "avg_latency_ms", "min_latency_ms",
            "max_latency_ms", "p95_latency_ms", "total_bytes", "avg_bytes_per_op",
            "failure_rate", "uptime", "quality_score",
        ]
        for col in expected:
            assert col in df.columns

    def test_score_range(self):
        df = generate_route_scoring_data(n_samples=500, seed=0)
        assert df["quality_score"].min() >= 0.0
        assert df["quality_score"].max() <= 1.0

    def test_failure_rate_range(self):
        df = generate_route_scoring_data(n_samples=500, seed=0)
        assert df["failure_rate"].min() >= 0.0
        assert df["failure_rate"].max() <= 1.0

    def test_reproducible(self):
        df1 = generate_route_scoring_data(n_samples=50, seed=123)
        df2 = generate_route_scoring_data(n_samples=50, seed=123)
        pd.testing.assert_frame_equal(df1, df2)


class TestAnomalyData:
    def test_shape(self):
        df = generate_anomaly_data(n_normal=100, n_anomaly=50, seed=0)
        assert len(df) == 150

    def test_anomaly_ratio(self):
        df = generate_anomaly_data(n_normal=800, n_anomaly=200, seed=0)
        assert abs(df["is_anomaly"].mean() - 0.2) < 0.01

    def test_features_present(self):
        df = generate_anomaly_data(n_normal=50, n_anomaly=50, seed=0)
        expected = [
            "total_events", "successes", "failures", "timeouts",
            "reroutes", "circuit_opens", "avg_latency_ms", "p95_latency_ms",
            "avg_bytes_per_op", "failure_rate", "events_per_minute", "is_anomaly",
        ]
        for col in expected:
            assert col in df.columns

    def test_binary_target(self):
        df = generate_anomaly_data(n_normal=100, n_anomaly=50, seed=0)
        assert set(df["is_anomaly"].unique()) == {0, 1}

    def test_reproducible(self):
        df1 = generate_anomaly_data(n_normal=50, n_anomaly=20, seed=99)
        df2 = generate_anomaly_data(n_normal=50, n_anomaly=20, seed=99)
        pd.testing.assert_frame_equal(df1, df2)
