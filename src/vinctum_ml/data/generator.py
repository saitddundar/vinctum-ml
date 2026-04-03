

import numpy as np
import pandas as pd


def generate_route_scoring_data(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic node metrics for route scoring model.

    Features mirror vinctum-core NodeMetrics:
    - total_events, successes, failures, timeouts, reroutes, circuit_opens
    - avg_latency_ms, min_latency_ms, max_latency_ms, p95_latency_ms
    - total_bytes, avg_bytes_per_op
    - failure_rate, uptime

    Target: quality_score (0.0 - 1.0)
    """
    rng = np.random.default_rng(seed)

    # -- Node profiles: good, average, bad, unstable --
    profiles = {
        "good":     {"weight": 0.35, "fail_rate": (0.00, 0.05), "latency": (5, 50),   "throughput": (50, 200), "stability": (0.95, 1.0)},
        "average":  {"weight": 0.30, "fail_rate": (0.05, 0.15), "latency": (30, 150),  "throughput": (20, 80),  "stability": (0.80, 0.95)},
        "bad":      {"weight": 0.20, "fail_rate": (0.20, 0.60), "latency": (100, 500), "throughput": (1, 20),   "stability": (0.40, 0.75)},
        "unstable": {"weight": 0.15, "fail_rate": (0.10, 0.40), "latency": (20, 300),  "throughput": (10, 100), "stability": (0.20, 0.50)},
    }

    rows = []
    for _ in range(n_samples):
        # Pick profile
        p_name = rng.choice(
            list(profiles.keys()),
            p=[p["weight"] for p in profiles.values()],
        )
        p = profiles[p_name]

        # Core counters
        total_events = int(rng.integers(10, 5000))
        failure_rate = rng.uniform(*p["fail_rate"])
        failures = int(total_events * failure_rate)
        successes = total_events - failures
        timeouts = int(failures * rng.uniform(0.1, 0.5))
        reroutes = int(rng.integers(0, max(1, int(total_events * (1 - p["stability"][0]) * 0.3))))
        circuit_opens = int(rng.integers(0, max(1, int(total_events * (1 - p["stability"][0]) * 0.1))))

        # Latency
        avg_latency = rng.uniform(*p["latency"])
        min_latency = max(1, avg_latency * rng.uniform(0.2, 0.6))
        max_latency = avg_latency * rng.uniform(1.5, 5.0)
        p95_latency = avg_latency * rng.uniform(1.2, 2.5)

        # Throughput
        avg_bytes_per_op = rng.uniform(*p["throughput"]) * 1024  # KB to bytes
        total_bytes = int(avg_bytes_per_op * successes)

        # Derived
        uptime = 1.0 - failure_rate
        stability = rng.uniform(*p["stability"])

        # Target: quality score
        score = (
            0.40 * uptime
            + 0.30 * max(0, 1.0 - avg_latency / 500)
            + 0.15 * min(1.0, np.log2(avg_bytes_per_op / 1024 + 1) / 10)
            + 0.15 * stability
        )
        # Add noise
        score = np.clip(score + rng.normal(0, 0.03), 0.0, 1.0)

        rows.append({
            "total_events": total_events,
            "successes": successes,
            "failures": failures,
            "timeouts": timeouts,
            "reroutes": reroutes,
            "circuit_opens": circuit_opens,
            "avg_latency_ms": round(avg_latency, 2),
            "min_latency_ms": round(min_latency, 2),
            "max_latency_ms": round(max_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "total_bytes": total_bytes,
            "avg_bytes_per_op": round(avg_bytes_per_op, 2),
            "failure_rate": round(failure_rate, 4),
            "uptime": round(uptime, 4),
            "quality_score": round(score, 4),
        })

    return pd.DataFrame(rows)


def generate_anomaly_data(n_normal: int = 8000, n_anomaly: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic data for anomaly detection model.

    Normal nodes: typical network behavior.
    Anomalous nodes: latency spikes, high failure, traffic spikes, unresponsive.
    """
    rng = np.random.default_rng(seed)
    rows = []

    # -- Normal nodes --
    for _ in range(n_normal):
        total_events = int(rng.integers(50, 3000))
        failure_rate = rng.uniform(0.0, 0.10)
        failures = int(total_events * failure_rate)
        successes = total_events - failures
        avg_latency = rng.uniform(10, 120)
        avg_bytes = rng.uniform(10_000, 200_000)

        rows.append({
            "total_events": total_events,
            "successes": successes,
            "failures": failures,
            "timeouts": int(failures * rng.uniform(0.0, 0.3)),
            "reroutes": int(rng.integers(0, max(1, total_events // 100))),
            "circuit_opens": int(rng.integers(0, 3)),
            "avg_latency_ms": round(avg_latency, 2),
            "p95_latency_ms": round(avg_latency * rng.uniform(1.1, 1.8), 2),
            "avg_bytes_per_op": round(avg_bytes, 2),
            "failure_rate": round(failure_rate, 4),
            "events_per_minute": round(rng.uniform(1, 30), 2),
            "is_anomaly": 0,
        })

    # -- Anomalous nodes --
    anomaly_types = ["latency_spike", "high_failure", "traffic_spike", "unresponsive"]
    for _ in range(n_anomaly):
        a_type = rng.choice(anomaly_types)
        total_events = int(rng.integers(20, 3000))

        if a_type == "latency_spike":
            avg_latency = rng.uniform(400, 2000)
            failure_rate = rng.uniform(0.05, 0.20)
            avg_bytes = rng.uniform(5_000, 100_000)
            events_per_min = rng.uniform(1, 20)
        elif a_type == "high_failure":
            avg_latency = rng.uniform(50, 300)
            failure_rate = rng.uniform(0.35, 0.90)
            avg_bytes = rng.uniform(1_000, 50_000)
            events_per_min = rng.uniform(1, 25)
        elif a_type == "traffic_spike":
            avg_latency = rng.uniform(20, 100)
            failure_rate = rng.uniform(0.0, 0.10)
            avg_bytes = rng.uniform(500_000, 5_000_000)
            events_per_min = rng.uniform(50, 500)
        else:  # unresponsive
            avg_latency = rng.uniform(3000, 10000)
            failure_rate = rng.uniform(0.80, 1.0)
            avg_bytes = rng.uniform(0, 1_000)
            events_per_min = rng.uniform(0.1, 2)

        failures = int(total_events * failure_rate)
        successes = total_events - failures

        rows.append({
            "total_events": total_events,
            "successes": successes,
            "failures": failures,
            "timeouts": int(failures * rng.uniform(0.2, 0.7)),
            "reroutes": int(rng.integers(0, max(1, total_events // 20))),
            "circuit_opens": int(rng.integers(1, 15)),
            "avg_latency_ms": round(avg_latency, 2),
            "p95_latency_ms": round(avg_latency * rng.uniform(1.3, 3.0), 2),
            "avg_bytes_per_op": round(avg_bytes, 2),
            "failure_rate": round(failure_rate, 4),
            "events_per_minute": round(events_per_min, 2),
            "is_anomaly": 1,
        })

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    route_df = generate_route_scoring_data()
    anomaly_df = generate_anomaly_data()

    route_df.to_csv("data/route_scoring.csv", index=False)
    anomaly_df.to_csv("data/anomaly_detection.csv", index=False)

    print(f"Route scoring: {route_df.shape}")
    print(f"  Score distribution:\n{route_df['quality_score'].describe()}\n")
    print(f"Anomaly detection: {anomaly_df.shape}")
    print(f"  Anomaly ratio: {anomaly_df['is_anomaly'].mean():.2%}")
