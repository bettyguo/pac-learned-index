"""Query execution engine for benchmark evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from pac_index.storage.structures import PWLIndex


@dataclass
class QueryResult:
    """Result of a single query execution."""
    predicted_position: float
    true_position: float | None = None
    error: float = 0.0
    latency_ns: float = 0.0


def execute_point_queries(
    index: PWLIndex,
    query_keys: np.ndarray,
    true_positions: np.ndarray | None = None,
) -> list[QueryResult]:
    """Execute a batch of point queries against a PWL index."""
    results = []
    for i, key in enumerate(query_keys):
        start = time.perf_counter_ns()
        pred = index.predict(float(key))
        elapsed = time.perf_counter_ns() - start

        true_pos = float(true_positions[i]) if true_positions is not None else None
        error = abs(pred - true_pos) if true_pos is not None else 0.0

        results.append(QueryResult(
            predicted_position=pred,
            true_position=true_pos,
            error=error,
            latency_ns=float(elapsed),
        ))
    return results


def compute_latency_percentiles(
    results: list[QueryResult],
) -> dict[str, float]:
    """Compute latency percentiles from query results."""
    latencies = np.array([r.latency_ns for r in results])
    if len(latencies) == 0:
        return {}
    return {
        "p50": float(np.percentile(latencies, 50)),
        "p90": float(np.percentile(latencies, 90)),
        "p95": float(np.percentile(latencies, 95)),
        "p99": float(np.percentile(latencies, 99)),
        "mean": float(np.mean(latencies)),
        "std": float(np.std(latencies)),
    }
