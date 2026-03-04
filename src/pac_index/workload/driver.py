"""Workload driver for executing benchmark experiments."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from pac_index.core.config import PACIndexConfig
from pac_index.evaluation.metrics import (
    BuildMetrics,
    IndexBenchmarkResult,
    LatencyMetrics,
    MemoryMetrics,
    PredictionMetrics,
)
from pac_index.storage.structures import PWLIndex
from pac_index.utils.reproducibility import set_seed, warmup_system

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRun:
    """Holds configuration and results for a single benchmark run."""

    dataset_id: str
    sample_size: int
    seed: int
    error_bound: int = 64
    result: IndexBenchmarkResult | None = None


class BenchmarkDriver:
    """Drive benchmark experiments across datasets, sample sizes, and seeds."""

    def __init__(self, config: PACIndexConfig) -> None:
        self.config = config

    def run_single(
        self,
        keys: np.ndarray,
        dataset_id: str,
        sample_size: int,
        seed: int,
        error_bound: int = 64,
        warmup_iterations: int = 1000,
    ) -> IndexBenchmarkResult:
        """Run a single benchmark: build index, measure metrics."""
        set_seed(seed)
        warmup_system(warmup_iterations)

        positions = np.arange(len(keys), dtype=np.float64)

        build_start = time.perf_counter()
        index = PWLIndex.build_optimal(keys, error_bound)
        build_time = time.perf_counter() - build_start

        max_err = index.max_error(keys, positions)
        avg_err = index.avg_error(keys, positions)

        num_queries = min(10000, len(keys))
        rng = np.random.default_rng(seed)
        query_indices = rng.choice(len(keys), size=num_queries, replace=False)
        query_keys = keys[query_indices]
        query_positions = positions[query_indices]

        latencies = []
        for qk in query_keys:
            start_ns = time.perf_counter_ns()
            index.predict(float(qk))
            latencies.append(time.perf_counter_ns() - start_ns)

        latencies_arr = np.array(latencies, dtype=np.float64)
        mem_mb = index.num_segments * 32 / (1024 * 1024)

        return IndexBenchmarkResult(
            index_name="PGM-index",
            dataset=dataset_id,
            sample_size=sample_size,
            seed=seed,
            prediction=PredictionMetrics(max_error=max_err, avg_error=avg_err),
            latency=LatencyMetrics.from_latencies(latencies_arr),
            memory=MemoryMetrics(index_size_mb=mem_mb, segments_count=index.num_segments),
            build=BuildMetrics(build_time_s=build_time, segments_built=index.num_segments),
        )

    def run_multi_seed(
        self,
        keys: np.ndarray,
        dataset_id: str,
        sample_size: int,
        seeds: list[int] | None = None,
        error_bound: int = 64,
    ) -> list[IndexBenchmarkResult]:
        """Run benchmark with multiple seeds for statistical rigor."""
        seeds = seeds or self.config.experiment.seeds
        results = []
        for seed in seeds:
            result = self.run_single(keys, dataset_id, sample_size, seed, error_bound)
            results.append(result)
            logger.info(
                "Run seed=%d: eps_max=%.1f, latency_p50=%.1f ns",
                seed, result.prediction.max_error, result.latency.p50,
            )
        return results
