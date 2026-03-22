"""Performance metrics for learned index evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PredictionMetrics:
    """Prediction error metrics for a learned index."""

    max_error: float = 0.0
    avg_error: float = 0.0
    median_error: float = 0.0
    p95_error: float = 0.0
    p99_error: float = 0.0

    @classmethod
    def from_errors(cls, errors: np.ndarray) -> "PredictionMetrics":
        if len(errors) == 0:
            return cls()
        return cls(
            max_error=float(np.max(errors)),
            avg_error=float(np.mean(errors)),
            median_error=float(np.median(errors)),
            p95_error=float(np.percentile(errors, 95)),
            p99_error=float(np.percentile(errors, 99)),
        )


@dataclass
class LatencyMetrics:
    """Query latency metrics in nanoseconds."""

    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    p999: float = 0.0
    mean: float = 0.0
    std: float = 0.0

    @classmethod
    def from_latencies(cls, latencies: np.ndarray) -> "LatencyMetrics":
        if len(latencies) == 0:
            return cls()
        return cls(
            p50=float(np.percentile(latencies, 50)),
            p90=float(np.percentile(latencies, 90)),
            p95=float(np.percentile(latencies, 95)),
            p99=float(np.percentile(latencies, 99)),
            p999=float(np.percentile(latencies, 99.9)),
            mean=float(np.mean(latencies)),
            std=float(np.std(latencies)),
        )


@dataclass
class MemoryMetrics:
    """Memory footprint metrics in megabytes."""

    index_size_mb: float = 0.0
    total_size_mb: float = 0.0
    segments_count: int = 0


@dataclass
class BuildMetrics:
    """Index build performance metrics."""

    build_time_s: float = 0.0
    segments_built: int = 0


@dataclass
class IndexBenchmarkResult:
    """Complete benchmark result for one (index, dataset, sample_size) combination."""

    index_name: str
    dataset: str
    sample_size: int
    seed: int
    prediction: PredictionMetrics = field(default_factory=PredictionMetrics)
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    memory: MemoryMetrics = field(default_factory=MemoryMetrics)
    build: BuildMetrics = field(default_factory=BuildMetrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_name": self.index_name,
            "dataset": self.dataset,
            "sample_size": self.sample_size,
            "seed": self.seed,
            "eps_max": self.prediction.max_error,
            "eps_avg": self.prediction.avg_error,
            "latency_p50": self.latency.p50,
            "latency_p95": self.latency.p95,
            "latency_p99": self.latency.p99,
            "memory_mb": self.memory.index_size_mb,
            "build_time_s": self.build.build_time_s,
            "segments": self.memory.segments_count,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Paper reference data (exact values from experimental tables)
# ─────────────────────────────────────────────────────────────────────────────

LATENCY_TABLE_DATA: dict[str, dict[int, dict[str, float]]] = {
    "amzn": {
        10_000:     {"eps_max": 1247, "p50": 892, "p95": 1156, "p99": 1423, "search": 10.3, "mem": 0.4},
        100_000:    {"eps_max": 412,  "p50": 356, "p95": 487,  "p99": 621,  "search": 8.7,  "mem": 1.2},
        1_000_000:  {"eps_max": 134,  "p50": 198, "p95": 312,  "p99": 398,  "search": 7.1,  "mem": 4.8},
        10_000_000: {"eps_max": 47,   "p50": 142, "p95": 245,  "p99": 312,  "search": 5.6,  "mem": 18.4},
        100_000_000:{"eps_max": 18,   "p50": 121, "p95": 198,  "p99": 267,  "search": 4.3,  "mem": 72.3},
    },
    "face": {
        10_000:     {"eps_max": 1589, "p50": 967, "p95": 1234, "p99": 1512, "search": 10.6, "mem": 0.5},
        100_000:    {"eps_max": 523,  "p50": 412, "p95": 534,  "p99": 687,  "search": 9.0,  "mem": 1.4},
        1_000_000:  {"eps_max": 178,  "p50": 234, "p95": 345,  "p99": 423,  "search": 7.5,  "mem": 5.2},
        10_000_000: {"eps_max": 62,   "p50": 167, "p95": 278,  "p99": 356,  "search": 5.9,  "mem": 19.8},
        100_000_000:{"eps_max": 24,   "p50": 134, "p95": 212,  "p99": 289,  "search": 4.6,  "mem": 78.1},
    },
    "osm": {
        10_000:     {"eps_max": 3421, "p50": 1234, "p95": 1567, "p99": 1923, "search": 11.7, "mem": 0.8},
        100_000:    {"eps_max": 1123, "p50": 623,  "p95": 812,  "p99": 1034, "search": 10.1, "mem": 2.1},
        1_000_000:  {"eps_max": 378,  "p50": 345,  "p95": 512,  "p99": 678,  "search": 8.5,  "mem": 7.8},
        10_000_000: {"eps_max": 134,  "p50": 223,  "p95": 389,  "p99": 512,  "search": 7.1,  "mem": 28.4},
        100_000_000:{"eps_max": 52,   "p50": 178,  "p95": 312,  "p99": 423,  "search": 5.7,  "mem": 112.5},
    },
    "wiki": {
        10_000:     {"eps_max": 1834, "p50": 1012, "p95": 1289, "p99": 1578, "search": 10.8, "mem": 0.5},
        100_000:    {"eps_max": 612,  "p50": 445,  "p95": 567,  "p99": 723,  "search": 9.3,  "mem": 1.3},
        1_000_000:  {"eps_max": 203,  "p50": 256,  "p95": 378,  "p99": 467,  "search": 7.7,  "mem": 5.0},
        10_000_000: {"eps_max": 71,   "p50": 178,  "p95": 289,  "p99": 378,  "search": 6.2,  "mem": 18.9},
        100_000_000:{"eps_max": 27,   "p50": 145,  "p95": 234,  "p99": 312,  "search": 4.8,  "mem": 74.5},
    },
}

BASELINE_COMPARISON_DATA: dict[str, dict[str, float]] = {
    "PGM-index":   {"eps_max": 47,  "p50": 142, "p99": 312, "mem": 18.4,   "build": 2.3},
    "ALEX":        {"eps_max": 52,  "p50": 156, "p99": 334, "mem": 24.1,   "build": 4.7},
    "LIPP":        {"eps_max": 0,   "p50": 89,  "p99": 156, "mem": 892.4,  "build": 8.2},
    "RMI":         {"eps_max": 78,  "p50": 178, "p99": 389, "mem": 12.3,   "build": 1.8},
    "RadixSpline": {"eps_max": 63,  "p50": 167, "p99": 356, "mem": 21.7,   "build": 0.9},
    "B-tree":      {"eps_max": -1,  "p50": 312, "p99": 589, "mem": 3200.0, "build": 45.2},
    "ART":         {"eps_max": -1,  "p50": 198, "p99": 423, "mem": 1456.0, "build": 28.4},
    "HOT":         {"eps_max": -1,  "p50": 167, "p99": 389, "mem": 892.0,  "build": 34.1},
    "FAST":        {"eps_max": -1,  "p50": 223, "p99": 478, "mem": 2100.0, "build": 12.3},
}

DISTRIBUTION_VALIDATION_DATA: list[dict[str, Any]] = [
    {"dataset": "amzn", "cv": 0.31, "gap_rho": 0.08, "predicted_coeff": 0.096, "observed_coeff": 0.089, "ratio": 1.08},
    {"dataset": "face", "cv": 0.72, "gap_rho": 0.12, "predicted_coeff": 0.518, "observed_coeff": 0.551, "ratio": 0.94},
    {"dataset": "osm",  "cv": 1.85, "gap_rho": 0.31, "predicted_coeff": 3.423, "observed_coeff": 3.891, "ratio": 0.88},
    {"dataset": "wiki", "cv": 0.44, "gap_rho": 0.15, "predicted_coeff": 0.194, "observed_coeff": 0.212, "ratio": 0.92},
]

HPO_COMPARISON_DATA: dict[str, dict[str, dict[str, Any]]] = {
    "amzn": {
        "theory":   {"k": 1923,  "eps": 49.2, "time_s": 12,   "evals": 1},
        "bayesian": {"k": 1847,  "eps": 47.8, "time_s": 1680, "evals": 50},
        "random":   {"k": 2341,  "eps": 52.1, "time_s": 900,  "evals": 100},
        "grid":     {"k": 1856,  "eps": 48.1, "time_s": 7560, "evals": 9},
    },
    "osm": {
        "theory":   {"k": 68450, "eps": 54.3, "time_s": 18,    "evals": 1},
        "bayesian": {"k": 71023, "eps": 51.9, "time_s": 2520,  "evals": 50},
        "random":   {"k": 58234, "eps": 61.4, "time_s": 1380,  "evals": 100},
        "grid":     {"k": 71234, "eps": 52.1, "time_s": 17280, "evals": 9},
    },
}

PRACTICAL_WORKFLOW_DATA: list[dict[str, Any]] = [
    {"dataset": "amzn", "target_eps": 100, "k_theory": 1923,  "k_empirical": 1856,  "deviation_pct": 3.6, "theory_time_s": 12, "grid_time_h": 2.1},
    {"dataset": "face", "target_eps": 100, "k_theory": 10368, "k_empirical": 10789, "deviation_pct": 3.9, "theory_time_s": 14, "grid_time_h": 3.4},
    {"dataset": "osm",  "target_eps": 100, "k_theory": 68450, "k_empirical": 71234, "deviation_pct": 3.9, "theory_time_s": 18, "grid_time_h": 4.8},
    {"dataset": "wiki", "target_eps": 50,  "k_theory": 15488, "k_empirical": 15200, "deviation_pct": 1.9, "theory_time_s": 13, "grid_time_h": 2.8},
]
